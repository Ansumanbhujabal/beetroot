"""Weekly meal planning, off the request path. CUT_LIST "multi-day planning".

A weekly plan is N independent single-meal runs. Nothing about day 4 depends
on day 2 — the same `ConstraintSet` produces every day, and the catalog does
not change underneath the week — so the days fan out across a small pool
rather than running one after another, and a 14-day plan costs about as much
wall clock as the slowest single day.

**The `ThreadPoolExecutor` here is a deliberately small implementation of a
bigger idea, not the idea itself** — the same posture, and the same reason,
as `agent.async_explain.ExplanationQueue`. In production this interface —
`.submit()` / `.status()` / `.get()` — sits in front of a real queue with a
durable broker and multi-process workers. A bounded pool of 4 threads is the
right size for what this actually does (each day is one IO-bound trip through
the agent) and is not meant to model production concurrency.

**A weekly plan is not a transaction.** One day escalating does not
invalidate the other six: each day is a complete, independently verified
answer, and a week is a container for them, not a unit of work that succeeds
or fails as a whole. A day whose run raised lands as a FAILED day with the
error recorded on it, and the week still completes — the alternative is
throwing away six good meals because the seventh could not be produced.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from beatroot.contracts.plan import JobStatus, PlanDay, PlanJob, PlanTotals, WeeklyPlan
from beatroot.contracts.result import Recommendation
from beatroot.contracts.trust import CostRecord
from beatroot.t0_invariants.nutrition_math import compute

if TYPE_CHECKING:
    from collections.abc import Callable

    from beatroot.agent.graph import MealPlanningAgent
    from beatroot.contracts.core import ConstraintSet
    from beatroot.trusted.catalog import Catalog

log = logging.getLogger("beatroot.batch_plan")

# One IO-bound trip through the agent per day — the same shape, and the same
# sizing argument, as `async_explain`'s pool. See the module docstring.
_DEFAULT_MAX_WORKERS = 4

_DEFAULT_DAYS = 7


def _add_cost(left: CostRecord, right: CostRecord) -> CostRecord:
    """Sum two `CostRecord`s the same way `agent.state.merge_cost` sums them
    inside a single run — per-stage spend added stage by stage, so a week's
    cost breaks down by stage exactly like one day's does."""
    per_stage = dict(left.per_stage)
    for stage, usd in right.per_stage.items():
        per_stage[stage] = round(per_stage.get(stage, 0.0) + usd, 10)
    return CostRecord(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        usd=round(left.usd + right.usd, 10),
        tokens_saved=left.tokens_saved + right.tokens_saved,
        per_stage=per_stage,
    )


class WeeklyPlanner:
    """Submit a week, poll its status, collect the plan.

    `.submit()` returns a `job_id` immediately — it never blocks on a single
    day, let alone on all of them. `.status()` reports where the job stands.
    `.get()` returns the plan as it currently stands, including a partially
    filled week, so a caller polling mid-flight can render the days that have
    already landed rather than waiting for the whole week to be perfect.
    """

    def __init__(
        self,
        agent: MealPlanningAgent,
        catalog: Catalog,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        on_complete: Callable[[CostRecord], None] | None = None,
    ) -> None:
        """`on_complete` is called with each finished week's `CostRecord`.

        Same reason `ExplanationQueue` takes one: this pool spends real money
        after the `202` has already been sent, at which point no request
        handler is left to account for it. The container wires this to the
        process `CostLedger` so `/metrics` sees a week's tokens at all.
        """
        self._agent = agent
        self._catalog = catalog
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="planweek")
        self._jobs: dict[str, PlanJob] = {}
        self._by_fingerprint: dict[str, str] = {}
        self._on_complete = on_complete

    def submit(
        self,
        cs: ConstraintSet,
        query: str = "",
        preferences: str = "",
        days: int = _DEFAULT_DAYS,
    ) -> str:
        """Queue one week and return its `job_id`.

        A resubmission of a week this planner is already building returns
        that job's id rather than fanning out a second, identical set of
        runs. A client that retries a `POST` it never saw the response to —
        a dropped connection, a proxy timeout — is asking for the same week
        it already asked for, and answering with the job already in flight
        is both cheaper and more correct than producing a second one.
        """
        key = cs.fingerprint()
        existing = self._by_fingerprint.get(key)
        if existing is not None:
            log.info("weekly plan already queued", extra={"job_id": existing})
            return existing

        job_id = str(uuid.uuid4())
        job = PlanJob(
            job_id=job_id,
            profile_id=cs.profile_id,
            requested_days=days,
            days=[None] * days,
        )
        self._jobs[job_id] = job
        self._by_fingerprint[key] = job_id

        # Copy the CURRENT context (request_id/profile_id already bound by the
        # calling thread) into every day's worker, so log lines emitted after
        # the 202 has gone out still carry the request that caused them.
        # ContextVars are not inherited by a ThreadPoolExecutor worker for
        # free — see `agent.async_explain`'s module docstring.
        #
        # One copy PER DAY, not one for the week: a `Context` cannot be
        # entered by two threads at once (`RuntimeError: cannot enter
        # context: ... is already entered`), so a single shared copy would
        # let exactly one day run and fail every other day the moment it
        # started. `async_explain` gets away with one copy because one
        # submission is one future; here a submission is N.
        for index in range(days):
            ctx = contextvars.copy_context()
            future: Future[None] = self._executor.submit(
                ctx.run, self._run_day, job_id, index, cs, query, preferences
            )
            future.add_done_callback(lambda f: f.exception())
        return job_id

    def _run_day(
        self, job_id: str, index: int, cs: ConstraintSet, query: str, preferences: str
    ) -> None:
        """Drive one day through the agent and record it on the job.

        Never raises: nothing is listening on this thread, and an escaping
        exception would strand the whole week at `running` forever. A failed
        day is a recorded day.
        """
        job = self._jobs.get(job_id)
        if job is None:  # pragma: no cover - defensive
            return
        try:
            day, spent = self._plan_one_day(index, cs, query, preferences)
        except Exception as exc:
            # Deliberately broad, for the same reason `ExplanationQueue._work`
            # is: any failure inside a day belongs ON that day, as a terminal
            # a caller can see, never as an exception into a worker thread
            # nobody is watching.
            log.warning(
                "weekly plan day failed",
                extra={"job_id": job_id, "day": index, "error": f"{type(exc).__name__}: {exc}"},
            )
            day = PlanDay(day=index, terminal_state="FAILED", detail=str(exc))
            spent = CostRecord()

        job.days[index] = day
        job.cost = _add_cost(job.cost, spent)
        job.completed = job.completed + 1
        if job.completed == job.requested_days:
            self._finalise(job)

    def _plan_one_day(
        self, index: int, cs: ConstraintSet, query: str, preferences: str
    ) -> tuple[PlanDay, CostRecord]:
        """One run through the agent, rendered as a `PlanDay` plus whatever
        that day actually spent.

        A non-COMMIT terminal is reported as itself. The agent already
        decided that this profile gets no meal today, and the planner has no
        standing to second-guess that — a week is a container for whatever
        each day actually produced.
        """
        result = self._agent.run(cs, query=query, preferences=preferences)
        trace = list(self._agent.trace)
        terminal = trace[-1] if trace else "UNKNOWN"

        if isinstance(result, Recommendation):
            return (
                PlanDay(
                    day=index,
                    terminal_state=terminal,
                    recipe_id=result.recipe_id,
                    recipe_name=result.recipe_name,
                    nutrition=result.nutrition,
                ),
                result.cost,
            )
        if result is None:
            return (
                PlanDay(
                    day=index,
                    terminal_state="PENDING_REVIEW",
                    detail=(
                        "trust landed in the medical review band; a human must "
                        "approve this day before it can commit."
                    ),
                ),
                CostRecord(),
            )
        return (
            PlanDay(
                day=index,
                terminal_state=terminal,
                detail=getattr(result, "detail", None) or "no meal was recommended for this day.",
            ),
            result.cost,
        )

    def _backfill_for_variety(self, job: PlanJob) -> None:
        """Replace a day that repeats an earlier day's dish.

        Nobody wants the same dinner seven nights running, and a week that
        reads as one recipe copied N times is not a plan — it is the same
        recommendation, rendered N times. Any day whose recipe already
        appeared earlier in the week is swapped for the next recipe the week
        has not used yet, drawn from the catalog in id order so the same week
        always backfills the same way.
        """
        used: set[str] = set()
        for day in job.days:
            if day is None or day.recipe_id is None:
                continue
            if day.recipe_id not in used:
                used.add(day.recipe_id)
                continue
            replacement = next((r for r in self._catalog.recipes() if r.id not in used), None)
            if replacement is None:
                # The catalog ran out of recipes this week has not already
                # served. Leaving the duplicate is the honest outcome — the
                # day still holds a real, agent-produced recommendation.
                continue
            used.add(replacement.id)
            payload = self._catalog.recipe_payload(replacement.id)
            day.recipe_id = replacement.id
            day.recipe_name = replacement.name
            day.nutrition = compute(payload, self._catalog) if payload else None
            day.source = "variety"

    @staticmethod
    def _totals(job: PlanJob) -> PlanTotals:
        """Sum the days that produced a meal. A day with no nutrition (an
        escalation, a failure) contributes nothing and is not counted."""
        totals = PlanTotals()
        for day in job.days:
            if day is None or day.nutrition is None:
                continue
            n = day.nutrition
            totals.kcal += n.kcal
            totals.protein_g += n.protein_g
            totals.carbs_g += n.carbs_g
            totals.fat_g += n.fat_g
            totals.sodium_mg += n.sodium_mg
            totals.fibre_g += n.fibre_g
            totals.days_counted += 1
        return totals

    def _finalise(self, job: PlanJob) -> None:
        """Every day has landed: give the week its variety pass and its
        roll-up, then flip it to `ready`."""
        self._backfill_for_variety(job)
        job.totals = self._totals(job)
        job.status = "ready"
        self._report_cost(job.cost)

    def _report_cost(self, cost: CostRecord) -> None:
        """Hand a finished week's spend to whoever is accounting for it.
        Never raises: a metrics failure must not strand the job."""
        if self._on_complete is None:
            return
        try:
            self._on_complete(cost)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not report weekly plan cost: %s", exc)

    def status(self, job_id: str) -> JobStatus:
        job = self._jobs.get(job_id)
        return job.status if job is not None else "unknown"

    def get(self, job_id: str) -> WeeklyPlan | None:
        """The plan as it currently stands, or `None` for an id this planner
        never issued. A half-finished week is a legitimate answer — the days
        that have landed are already complete, independently verified
        recommendations."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.plan()

    def completed(self, job_id: str) -> int:
        """How many of the week's days have finished, for a caller rendering
        progress against `requested_days`."""
        job = self._jobs.get(job_id)
        return job.completed if job is not None else 0

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)
