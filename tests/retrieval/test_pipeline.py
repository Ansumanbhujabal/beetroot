from pathlib import Path

from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.rerank import llm_rerank, retrieve
from beatroot.store.db import connect, seed
from beatroot.trusted.catalog import Catalog

DATA = Path(__file__).parents[2] / "data"


def _catalog(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    return Catalog(conn)


def test_retrieval_never_returns_an_illegal_candidate(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        ],
    )
    results = retrieve("something with rice", cs, cat, LLMClient.offline(), top_k=10)
    assert results, "should find legal candidates"
    assert all("peanut" not in r.tags for r in results), "constraint leakage"


def test_affinity_cannot_promote_an_illegal_candidate(tmp_path):
    """The load-bearing safety property of Task 17: affinity is a THIRD
    ranking fed into RRF, and RRF only ever reorders candidates that
    already passed the legal-set filter (`is_legal`) below — it cannot add
    a candidate back in. A strongly-positive affinity on a tag a MEDICAL
    constraint excludes must still yield zero results carrying that tag.
    """
    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        ],
    )
    # Prove the test isn't vacuous: peanut-tagged recipes really exist in
    # the catalog, so there is something for a rogue affinity to try to
    # promote back in.
    assert any("peanut" in r.tags for r in cat.recipes())

    # The strongest possible affinity, on exactly the excluded tag, plus a
    # few other real tags for good measure.
    affinity = {"peanut": 1.0, "spicy": 1.0, "vegan": 1.0}

    results = retrieve(
        "something with rice",
        cs,
        cat,
        LLMClient.offline(),
        top_k=len(cat.recipes()),  # ask for everything legal, not just top_k
        affinity=affinity,
    )
    assert results, "should still find legal candidates"
    assert all("peanut" not in r.tags for r in results), (
        "affinity promoted an excluded tag past the legality filter"
    )


def test_retrieval_returns_recipe_objects_with_top_k_respected(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    results = retrieve("rice", cs, cat, LLMClient.offline(), top_k=3)
    assert 0 < len(results) <= 3
    assert all(hasattr(r, "id") and hasattr(r, "tags") for r in results)


def test_retrieval_handles_empty_query_without_raising(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    results = retrieve("", cs, cat, LLMClient.offline(), top_k=5)
    assert isinstance(results, list)


def test_retrieval_handles_query_matching_nothing(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    results = retrieve("zzzznonexistentzzzz", cs, cat, LLMClient.offline(), top_k=5)
    assert isinstance(results, list)


def test_rerank_out_of_range_index_falls_back_safely(tmp_path):
    """A model returning choice_index=99 must not crash or escape the legal set."""
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    candidates = retrieve("rice", cs, cat, LLMClient.offline(), top_k=3)

    class _Rogue(LLMClient):
        def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
            c = super().complete(prompt, schema=schema, stage=stage)
            c.parsed = {"choice_index": 99}
            return c

    chosen, _, _, _ = llm_rerank("rice", candidates, _Rogue.offline())
    assert chosen in candidates


def test_rerank_negative_index_falls_back_safely(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    candidates = retrieve("rice", cs, cat, LLMClient.offline(), top_k=3)

    class _Rogue(LLMClient):
        def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
            c = super().complete(prompt, schema=schema, stage=stage)
            c.parsed = {"choice_index": -1}
            return c

    chosen, _, _, _ = llm_rerank("rice", candidates, _Rogue.offline())
    assert chosen in candidates


def test_rerank_non_integer_index_falls_back_safely(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    candidates = retrieve("rice", cs, cat, LLMClient.offline(), top_k=3)

    class _Rogue(LLMClient):
        def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
            c = super().complete(prompt, schema=schema, stage=stage)
            c.parsed = {"choice_index": "not-a-number"}
            return c

    chosen, _, _, _ = llm_rerank("rice", candidates, _Rogue.offline())
    assert chosen in candidates


def test_rerank_missing_key_falls_back_safely(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    candidates = retrieve("rice", cs, cat, LLMClient.offline(), top_k=3)

    class _Rogue(LLMClient):
        def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
            c = super().complete(prompt, schema=schema, stage=stage)
            c.parsed = {}
            return c

    chosen, _, _, _ = llm_rerank("rice", candidates, _Rogue.offline())
    assert chosen in candidates


def test_rerank_no_candidates_returns_none():
    chosen, rationale, score, cost = llm_rerank("rice", [], LLMClient.offline())
    assert chosen is None
    assert rationale == ""
    assert score == 0.0
    assert cost.usd == 0.0


def test_rerank_single_candidate_short_circuits(tmp_path):
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    candidates = retrieve("rice", cs, cat, LLMClient.offline(), top_k=1)
    chosen, rationale, score, cost = llm_rerank("rice", candidates, LLMClient.offline())
    assert chosen == candidates[0]
    assert rationale == "only legal candidate"
    assert score == 1.0
    assert cost.usd == 0.0


def test_rerank_returns_the_completion_cost_not_a_discarded_one(tmp_path):
    """`llm_rerank` must not discard `Completion.cost`: a COMMIT spends
    real money on two model calls (rerank here, explain later), and a
    caller that can't see this one silently understates `per_plan_usd` by
    roughly half against a real provider. Offline cost is always $0, so
    this checks the thing that WAS silently dropped — the `Completion`'s
    own `per_stage` entry for this call — actually reaches the caller."""
    cat = _catalog(tmp_path)
    cs = ConstraintSet(profile_id="p", constraints=[])
    candidates = retrieve("rice", cs, cat, LLMClient.offline(), top_k=3)
    assert len(candidates) > 1, "need multiple candidates to force a real rerank call"
    _, _, _, cost = llm_rerank("rice", candidates, LLMClient.offline())
    assert "rerank" in cost.per_stage
