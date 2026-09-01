"""Tests for `agent.async_explain.ExplanationQueue`. Task 23, Spec §17.

The load-bearing claim under test throughout this file: nothing the user
needs — the recipe, its nutrition, its trust score, which constraints it
satisfies — ever depends on the model having spoken. Prose is an add-on,
and every failure mode of generating it (slow, broken, still cooking)
must leave the recommendation itself untouched.
"""

import time
from dataclasses import replace

import pytest

from beatroot.agent.async_explain import ExplanationQueue
from beatroot.agent.nodes import make_nodes
from beatroot.contracts.core import ConstraintSet
from beatroot.contracts.trust import Completion
from beatroot.reasoning.llm import LLMClient
from beatroot.settings import get_settings
from beatroot.t0_invariants.constraints import CheckResult


def test_recommendation_is_complete_without_the_explanation(tmp_path, monkeypatch):
    """The load-bearing claim: nothing the user needs depends on the model.

    Runs the real graph, synchronously (the default), and checks the
    parts of the card that must be present and correct regardless of
    whether — or when — prose ever arrives.
    """
    from beatroot.container import build_container

    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    container = build_container(db_path=tmp_path / "load_bearing.db")
    try:
        rec = container.agent.run(ConstraintSet(profile_id="p", constraints=[]), query="rice")
        from beatroot.contracts.result import Recommendation

        assert isinstance(rec, Recommendation)
        assert rec.recipe_id
        assert rec.nutrition.kcal > 0
        assert rec.trust.composite > 0
    finally:
        container.close()


def test_queue_returns_pending_then_ready():
    q = ExplanationQueue(LLMClient.offline())
    q.submit("r1", recipe=None, nutrition=None, check=None, prompt="say hi")
    assert q.status("r1") in {"pending", "ready"}
    assert q.get("r1", timeout=5.0) is not None
    assert q.status("r1") == "ready"


def test_a_failed_explanation_does_not_lose_the_recommendation():
    """`submit()`/`get()` never propagate a provider exception — they
    record `"failed"` and hand back `None`. Nothing about the recipe,
    nutrition or trust this queue was never asked to touch is affected."""
    llm = LLMClient.offline()

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provider down")

    llm.complete = _boom  # type: ignore[method-assign]

    q = ExplanationQueue(llm)
    q.submit("r2", recipe=None, nutrition=None, check=None, prompt="x")
    assert q.get("r2", timeout=5.0) is None
    assert q.status("r2") == "failed"


def test_unknown_id_is_not_an_error():
    assert ExplanationQueue(LLMClient.offline()).status("nope") == "pending"
    assert ExplanationQueue(LLMClient.offline()).get("nope", timeout=0.1) is None


def test_ready_entry_carries_the_real_cost():
    q = ExplanationQueue(LLMClient.offline())
    assert q.cost("r3").usd == 0.0  # nothing submitted yet
    q.submit("r3", recipe=None, nutrition=None, check=None, prompt="hi")
    q.get("r3", timeout=5.0)
    # The offline stub always reports zero cost, but the entry must carry
    # a REAL CostRecord object (not a stub/zero placeholder invented here)
    # — same type a synchronous `explain_node` completion would produce.
    from beatroot.contracts.trust import CostRecord

    assert isinstance(q.cost("r3"), CostRecord)


def test_correlation_id_survives_into_the_worker_thread():
    """`ThreadPoolExecutor` workers do not inherit `contextvars` for free
    — `submit()` must capture the calling thread's context (with
    `request_id` already bound) and run the job inside it, or a log line
    the background completion emits loses the request that caused it."""
    from beatroot.obs import logging as obs_logging

    captured: list[str | None] = []

    class _Capturing(LLMClient):
        def complete(self, prompt: str, *, schema=None, stage="", prompt_ref=None):  # type: ignore[no-untyped-def]
            captured.append(obs_logging._request_id.get())
            return super().complete(prompt, schema=schema, stage=stage)

    q = ExplanationQueue(_Capturing(offline=True))
    with obs_logging.bind_request("req-xyz-123"):
        q.submit("r4", recipe=None, nutrition=None, check=None, prompt="hi")
        # Submission itself returns before the model is ever called —
        # the calling thread's own contextvar is irrelevant to the
        # assertion below by the time we leave this `with` block.

    assert q.get("r4", timeout=5.0) is not None
    assert captured == ["req-xyz-123"]
    # And the CALLING thread's own view of the contextvar is unaffected
    # by whatever ran in the worker thread — contextvars are per-thread.
    assert obs_logging._request_id.get() is None


# ---------------------------------------------------------------------------
# explain_node / verify_node wiring — off by default, opt-in via settings.
# ---------------------------------------------------------------------------


def test_explain_node_stays_synchronous_when_async_is_off(agent_deps, monkeypatch):
    """Synchronous explanation still works when the flag is off — every
    caller that wants it (CLI, eval runners, golden cases) gets exactly the
    prior behaviour.

    The mode is set EXPLICITLY here rather than inherited from whatever
    `config/beatroot.yaml` happens to ship. This test previously asserted
    `async_explanation is False` as an ambient fact and broke the moment the
    shipped default flipped to true for latency — a test of a MODE should
    pin that mode itself, not depend on a config file it does not name.
    """
    monkeypatch.setenv("BEATROOT_ASYNC_EXPLANATION", "0")
    get_settings.cache_clear()
    assert get_settings().async_explanation is False
    assert agent_deps.explanation_queue is None
    nodes = make_nodes(agent_deps)
    recipe = agent_deps.catalog.hydrate(agent_deps.catalog.recipes()[0])
    state = {
        "chosen_id": recipe.id,
        "nutrition": recipe.nutrition.model_dump(mode="json"),
        "check": CheckResult(ok=True, satisfied=["c1"]).model_dump(mode="json"),
        "thread_id": "sync-thread",
    }
    out = nodes["explain"](state)
    assert out["explanation"]  # non-empty prose, generated inline
    assert out["trace"] == ["EXPLAIN"]


def test_explain_node_defers_to_the_queue_when_async_is_enabled(agent_deps, monkeypatch):
    """With the flag on and a queue wired into `Deps`, `explain_node`
    returns immediately with `explanation=""` and the SAME job is
    retrievable from the queue under `state["thread_id"]`."""
    monkeypatch.setenv("BEATROOT_ASYNC_EXPLANATION", "1")
    get_settings.cache_clear()
    try:
        queue = ExplanationQueue(LLMClient.offline())
        deps = replace(agent_deps, explanation_queue=queue)
        nodes = make_nodes(deps)
        recipe = deps.catalog.hydrate(deps.catalog.recipes()[0])
        state = {
            "chosen_id": recipe.id,
            "nutrition": recipe.nutrition.model_dump(mode="json"),
            "check": CheckResult(ok=True, satisfied=["c1"]).model_dump(mode="json"),
            "thread_id": "async-thread-1",
        }
        out = nodes["explain"](state)
        assert out["explanation"] == ""
        assert out["trace"] == ["EXPLAIN"]
        assert queue.status("async-thread-1") in {"pending", "ready"}
        text = queue.get("async-thread-1", timeout=5.0)
        assert text is not None
        assert queue.status("async-thread-1") == "ready"
    finally:
        monkeypatch.delenv("BEATROOT_ASYNC_EXPLANATION", raising=False)
        get_settings.cache_clear()


def test_verify_skips_the_drift_check_on_a_pending_explanation(agent_deps):
    """An empty (still-queued) explanation states no numbers to drift-check
    against — VERIFY must not escalate a recommendation just because prose
    has not arrived yet. The constraint recheck still runs unconditionally."""
    nodes = make_nodes(agent_deps)
    cs = ConstraintSet(profile_id="p", constraints=[])
    recipe = agent_deps.catalog.hydrate(agent_deps.catalog.recipes()[0])
    state = {
        "constraint_set": cs.model_dump(mode="json"),
        "chosen_id": recipe.id,
        "nutrition": recipe.nutrition.model_dump(mode="json"),
        "check": CheckResult(ok=True).model_dump(mode="json"),
        "explanation": "",
    }
    out = nodes["verify"](state)
    assert out.get("terminal") is None
    assert out["trace"] == ["VERIFY"]


@pytest.fixture(autouse=True)
def _clean_settings_cache():
    """Belt-and-suspenders: even if a test above fails before its own
    `finally` runs, no OTHER test in the suite should ever observe a
    `get_settings()` cached from a monkeypatched env var here."""
    yield
    get_settings.cache_clear()


def test_a_finished_job_reports_its_cost_to_whoever_is_accounting(tmp_path):
    """The explanation is paid for AFTER the response has been sent, so no
    request handler is left to account for it.

    Without this hand-off `/metrics` under-reported every COMMIT by the
    entire explanation call: the money was spent, the tokens were recorded
    on the queue entry, and `per_plan_usd` never saw either. `plans` must
    NOT move — this is additional spend on a plan that was already counted,
    not a new plan.
    """
    from beatroot.contracts.trust import CostRecord
    from beatroot.obs.cost import CostLedger

    class _Priced:
        def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
            return Completion(
                text="This meal provides 500.0 kcal.",
                cost=CostRecord(
                    usd=0.0008,
                    prompt_tokens=100,
                    completion_tokens=40,
                    per_stage={"explain": 0.0008},
                ),
            )

    ledger = CostLedger()
    ledger.record_plan()
    queue = ExplanationQueue(_Priced(), on_complete=ledger.fold)
    try:
        queue.submit("rec-cost", recipe=None, nutrition=None, check=None, prompt="prompt")
        queue.get("rec-cost", timeout=5.0)
        assert ledger.per_stage["explain"] == pytest.approx(0.0008)
        assert ledger.tokens == 140
        assert ledger.plans == 1, "an async explanation is not a new plan"
    finally:
        queue.shutdown(wait=True)


def test_a_failed_job_still_reports_the_money_it_spent():
    """A provider call that produced ungrounded prose was still paid for.
    Reporting zero there would make the drift check look free."""
    from beatroot.contracts.nutrition import NutritionFacts
    from beatroot.contracts.trust import CostRecord
    from beatroot.obs.cost import CostLedger

    class _Lying:
        def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
            return Completion(
                text="This meal provides 9000 kcal.",
                cost=CostRecord(usd=0.0009, per_stage={"explain": 0.0009}),
            )

    truth = NutritionFacts(
        kcal=500.0,
        protein_g=20.0,
        carbs_g=60.0,
        fat_g=10.0,
        sodium_mg=300.0,
        fibre_g=5.0,
        coverage=1.0,
    )
    ledger = CostLedger()
    queue = ExplanationQueue(_Lying(), on_complete=ledger.fold)
    try:
        queue.submit("rec-fail", recipe=None, nutrition=truth, check=None, prompt="prompt")
        for _ in range(100):
            if queue.status("rec-fail") == "failed":
                break
            time.sleep(0.05)
        assert queue.status("rec-fail") == "failed", "drift must fail the job"
        assert ledger.per_stage["explain"] == pytest.approx(0.0009)
    finally:
        queue.shutdown(wait=True)
