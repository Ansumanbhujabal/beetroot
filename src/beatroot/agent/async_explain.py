"""Explanation generation, off the request path. Task 23, Spec §17.

By the time `explain` would run, the recommendation is **already fully
determined**: `feasibility`/`retrieve`/`score` chose the recipe, `verify`'s
sibling checks (`t0_invariants.constraints.check_recipe`) re-confirm it
against the `ConstraintSet`, and its nutrition came straight from the
trusted catalog (`NutritionFacts.provenance == "computed"`, never model
output). The only thing left for the model to add is prose. That is a
design property with an operational dividend: prose generation can leave
the critical path entirely, and p95 latency for `/recommend` stops
depending on the slowest component in the system — the model call.

`ExplanationQueue` is that off-path mechanism. **The `ThreadPoolExecutor`
here is a deliberately small implementation of a bigger idea, not the
idea itself.** In production this interface — `.submit()` / `.get()` /
`.status()` — sits in front of a real queue (Celery, RQ, Cloud Tasks,
...): a durable broker, retries with backoff, multi-process workers. Swap
the body of this class for a Celery task and a Redis-backed status store
and nothing upstream (`agent.nodes.explain_node`, `api.main`) has to
change, because nothing upstream talks to a `ThreadPoolExecutor` — it
talks to this interface. A bounded pool of 4 threads is exactly the right
size for what this demo actually does (one IO-bound HTTP call per
submission, no CPU work), and is not meant to model production
concurrency.

**A failed explanation must never invalidate the recommendation.** The
meal was already correct without it — nutrition, trust and constraint
satisfaction do not depend on the model having spoken. `submit()` catches
any exception the completion raises, marks the entry `"failed"`, and
`get()` returns `None` for it. Nothing here can turn a COMMIT into an
ESCALATE; that door closed at `verify` before this queue ever runs.

**Correlation ids do not cross a `ThreadPoolExecutor` boundary for
free.** `obs.logging` binds `request_id`/`profile_id` into
`contextvars.ContextVar`s so every log line inside a request carries
them — but a `ContextVar` is only ever inherited by a *new* thread if
something explicitly copies it there; `ThreadPoolExecutor.submit` does
not do this on its own (unlike `asyncio`, which propagates context by
default). `submit()` below captures `contextvars.copy_context()` on the
calling thread — inside the request, with `request_id` already bound —
and runs the worker through `ctx.run(...)`, so every log line the
background completion emits still carries the request that triggered it,
even though it executes after that request has already returned.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from beatroot.confirm.trust_score import load_thresholds
from beatroot.contracts.trust import CostRecord
from beatroot.eval.verifiers.nutrition_drift import detect_drift
from beatroot.reasoning.prompts import load_prompt

if TYPE_CHECKING:
    from beatroot.contracts.nutrition import NutritionFacts
    from beatroot.reasoning.llm import LLMClient
    from beatroot.t0_invariants.constraints import CheckResult
    from beatroot.trusted.catalog import Recipe

log = logging.getLogger("beatroot.async_explain")

Status = Literal["pending", "ready", "failed"]

# IO-bound on one HTTP call per submission — this is not a CPU-bound pool
# and does not need to scale with core count. See the module docstring.
_DEFAULT_MAX_WORKERS = 4


@dataclass
class _Entry:
    status: Status = "pending"
    text: str | None = None
    cost: CostRecord = field(default_factory=CostRecord)
    error: str | None = None


def _render_prompt(recipe: Recipe, nutrition: NutritionFacts, check: CheckResult) -> str:
    """The exact prompt `agent.nodes.explain_node` renders synchronously
    today — kept here, not duplicated, so the async path and the sync path
    never drift in what they ask the model."""
    facts = ", ".join(
        f"{f}={getattr(nutrition, f)}"
        for f in ("kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g")
    )
    return load_prompt("explain").render(
        name=recipe.name, facts=facts, satisfied=", ".join(check.satisfied)
    )


class ExplanationQueue:
    """Submit a prose-generation job, poll its status, collect the result.

    `.submit()` returns immediately — it never blocks on the model.
    `.status()` never raises on an unknown id (a `rec_id` nobody submitted
    is indistinguishable from one still queued: `"pending"`). `.get()`
    blocks up to `timeout` seconds for a submitted-but-unfinished job and
    returns `None` on timeout, on failure, or on an unknown id — the one
    thing it never does is raise, because a caller (a route handler)
    reaching for a still-cooking explanation is an ordinary outcome, not
    an error.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        on_complete: Callable[[CostRecord], None] | None = None,
    ) -> None:
        """`on_complete` is called with the job's `CostRecord` once it
        finishes, successfully or not.

        It exists because this queue spends real money AFTER the response
        has been sent, at which point no request handler is left to account
        for it. Without it, `/metrics` under-reported every COMMIT by the
        entire explanation call — the money was spent, the tokens were
        counted on the entry, and no metric ever saw either. The container
        wires this to the process `CostLedger`.

        A failed job still reports its cost: a provider call that produced
        an ungrounded explanation was still paid for, and hiding that would
        make the drift check look free.
        """
        self._llm = llm
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="explain")
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._futures: dict[str, Future[None]] = {}
        self._on_complete = on_complete

    def _report_cost(self, cost: CostRecord) -> None:
        """Hand a finished job's spend to whoever is accounting for it.
        Never raises: a metrics failure must not strand the entry."""
        if self._on_complete is None:
            return
        try:
            self._on_complete(cost)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not report explanation cost: %s", exc)

    def submit(
        self,
        rec_id: str,
        recipe: Recipe | None,
        nutrition: NutritionFacts | None,
        check: CheckResult | None,
        prompt: str | None = None,
    ) -> None:
        """Queue one explanation job under `rec_id`.

        `prompt`, when given, is used verbatim (a test double, or a caller
        that already rendered its own text) — otherwise it is built from
        `recipe`/`nutrition`/`check` via the same `prompts/explain.md`
        template `explain_node` renders synchronously.
        """
        rendered = prompt if prompt is not None else _render_prompt(recipe, nutrition, check)  # type: ignore[arg-type]
        with self._lock:
            self._entries[rec_id] = _Entry(status="pending")
        # Copy the CURRENT context (request_id/profile_id already bound by
        # the calling thread) and run the worker inside it, so log lines
        # emitted after the request has returned still carry the request
        # that caused them. See the module docstring.
        ctx = contextvars.copy_context()
        try:
            future = self._executor.submit(ctx.run, self._work, rec_id, rendered, nutrition)
        except RuntimeError as exc:
            # The executor is already shut down. This happens on a SIGTERM /
            # rolling restart: `shutdown()` runs while a request is still in
            # flight, and that request then reaches EXPLAIN. Unguarded,
            # `ThreadPoolExecutor.submit` raises "cannot schedule new futures
            # after shutdown" straight into the request path, turning a
            # graceful drain into a 500 for a diner whose recommendation was
            # already fully computed and verified.
            #
            # Same rule `_work` already follows for provider failures: the
            # explanation is the optional part, so it lands as `failed` and
            # the recommendation stands. Never escape into the caller.
            log.warning("explanation %s not queued, executor shut down: %s", rec_id, exc)
            with self._lock:
                self._entries[rec_id] = _Entry(
                    status="failed", error="server is shutting down; explanation not generated"
                )
            return
        with self._lock:
            self._futures[rec_id] = future

    def _work(self, rec_id: str, prompt: str, nutrition: NutritionFacts | None = None) -> None:
        """Generate the prose AND ground it. The grounding is not optional.

        `verify_node` diffs every number an explanation states against catalog
        truth — but with `async_explanation` on, `state["explanation"]` is
        still `""` when VERIFY runs, so that check finds nothing and the prose
        is generated afterwards, here. Until this, it was then served to the
        diner having never been drift-checked at all: enabling async
        explanation for latency silently disabled the one guarantee that the
        explanation's numbers match the trusted catalog.

        A drift finding lands the entry as `failed`, not `ready`. Showing a
        diner prose containing a fabricated nutrition number is worse than
        showing them no prose, and the recommendation itself is unaffected —
        it was fully determined and verified before this job was ever queued.
        """
        try:
            # `prompt` arrives already rendered from `explain_node`, so the
            # prompt OBJECT is re-resolved here purely for trace provenance
            # — `load_prompt` is process-cached, so this costs a dict lookup,
            # not a fetch, and it keeps the async generation attributable to
            # the same prompt version the sync path reports.
            completion = self._llm.complete(
                prompt, stage="explain", prompt_ref=load_prompt("explain")
            )
        except Exception as exc:
            # Deliberately broad: ANY provider failure (network, timeout,
            # a malformed response LiteLLM itself raised on) must land the
            # job as "failed", never escape this worker thread — nothing
            # is listening for an exception here, and letting one through
            # would just silently strand the entry at "pending" forever.
            log.warning("explanation %s failed: %s: %s", rec_id, type(exc).__name__, exc)
            with self._lock:
                self._entries[rec_id] = _Entry(status="failed", error=str(exc))
            return
        if nutrition is not None and completion.text:
            tolerance = load_thresholds().verifiers.nutrition_drift_pct
            drift = detect_drift(completion.text, nutrition, tolerance=tolerance)
            if drift:
                detail = "; ".join(
                    f"{f.field} stated {f.stated} vs computed {f.computed}" for f in drift
                )
                log.warning("explanation %s failed grounding: %s", rec_id, detail)
                with self._lock:
                    self._entries[rec_id] = _Entry(
                        status="failed",
                        error=f"explanation contradicted catalog nutrition ({detail})",
                        cost=completion.cost,
                    )
                self._report_cost(completion.cost)
                return
        with self._lock:
            self._entries[rec_id] = _Entry(
                status="ready", text=completion.text, cost=completion.cost
            )
        self._report_cost(completion.cost)

    def status(self, rec_id: str) -> Status:
        with self._lock:
            entry = self._entries.get(rec_id)
        return entry.status if entry is not None else "pending"

    def get(self, rec_id: str, timeout: float = 5.0) -> str | None:
        with self._lock:
            future = self._futures.get(rec_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except Exception as exc:
                # A timeout (still cooking) or the completion's own
                # exception (already recorded on the entry by `_work`,
                # which never re-raises out of the executor) — either
                # way `.get()` never raises; the entry lookup below is
                # what actually decides what this call returns.
                log.debug("get(%s) did not observe a finished future: %s", rec_id, exc)
        with self._lock:
            entry = self._entries.get(rec_id)
        if entry is None or entry.status != "ready":
            return None
        return entry.text

    def cost(self, rec_id: str) -> CostRecord:
        """The `CostRecord` a finished job actually spent — `CostRecord()`
        (all zero) for anything not yet `"ready"`. A real queue backend
        would persist this alongside the text; here it lives on the same
        in-memory entry."""
        with self._lock:
            entry = self._entries.get(rec_id)
        return entry.cost if entry is not None else CostRecord()

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)
