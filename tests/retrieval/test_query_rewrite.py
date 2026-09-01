"""QUERY REWRITE task: `retrieval.query_rewrite.rewrite_query`.

Three properties matter more than any one example: an empty query never
spends a call, a rewrite failure never costs a recommendation (it degrades
to the original query), and both the original and rewritten text are
always visible on the result.
"""

from beatroot.contracts.trust import Completion, CostRecord
from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.query_rewrite import rewrite_query


class _FakeLLM:
    """A minimal stand-in with the one method `rewrite_query` calls —
    lets tests control exactly what `.complete()` returns/raises without
    depending on the real offline stub's own expansion table."""

    def __init__(self, completion=None, exc=None):
        self._completion = completion
        self._exc = exc
        self.calls = 0

    def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        assert stage == "rewrite_query"
        return self._completion


def test_empty_query_never_calls_the_model():
    llm = _FakeLLM()
    result = rewrite_query("", llm)
    assert result.original == "" and result.rewritten == ""
    assert result.applied is False
    assert result.cost == CostRecord()
    assert llm.calls == 0


def test_whitespace_only_query_is_treated_as_empty():
    llm = _FakeLLM()
    result = rewrite_query("   ", llm)
    assert result.applied is False
    assert llm.calls == 0


def test_a_working_rewrite_shows_both_original_and_rewritten():
    llm = _FakeLLM(
        Completion(
            text="...",
            parsed={
                "rewritten_query": "warm hearty soup stew",
                "terms": ["hearty", "soup", "stew"],
            },
            cost=CostRecord(usd=0.001, per_stage={"rewrite_query": 0.001}),
        )
    )
    result = rewrite_query("warm", llm)
    assert result.original == "warm"
    assert result.rewritten == "warm hearty soup stew"
    assert result.terms == ["hearty", "soup", "stew"]
    assert result.applied is True
    assert result.cost.usd == 0.001


def test_a_raised_exception_degrades_to_the_original_query():
    llm = _FakeLLM(exc=RuntimeError("provider is down"))
    result = rewrite_query("something warm and comforting", llm)
    assert result.original == result.rewritten == "something warm and comforting"
    assert result.applied is False
    assert result.cost == CostRecord()


def test_a_missing_rewritten_query_key_degrades_to_the_original():
    llm = _FakeLLM(Completion(text="not json-shaped", parsed=None, cost=CostRecord(usd=0.0005)))
    result = rewrite_query("dinner", llm)
    assert result.original == result.rewritten == "dinner"
    assert result.applied is False
    # The call genuinely happened — its (nonzero) cost still rides along,
    # a rewrite failure is not free, only non-blocking.
    assert result.cost.usd == 0.0005


def test_a_blank_rewritten_query_degrades_to_the_original():
    llm = _FakeLLM(Completion(text="{}", parsed={"rewritten_query": "   "}, cost=CostRecord()))
    result = rewrite_query("lunch", llm)
    assert result.rewritten == "lunch"
    assert result.applied is False


def test_non_string_terms_are_dropped_not_crashed_on():
    llm = _FakeLLM(
        Completion(
            text="...", parsed={"rewritten_query": "dinner hearty", "terms": ["hearty", 42, None]}
        )
    )
    result = rewrite_query("dinner", llm)
    assert result.terms == ["hearty"]


# ---------------------------------------------------------------------------
# Deterministic offline behaviour (LLMClient.offline()) — the whole point of
# `LLMClient._offline_rewrite_query` is that a credential-free run still
# shows a real, reproducible effect.
# ---------------------------------------------------------------------------


def test_offline_expands_a_known_word_deterministically():
    llm = LLMClient.offline()
    result = rewrite_query("something warm and comforting", llm)
    assert result.applied is True
    assert "hearty" in result.terms
    assert result.original == "something warm and comforting"
    assert result.rewritten != result.original

    # Same input, same output — no randomness, no network.
    again = rewrite_query("something warm and comforting", llm)
    assert again.rewritten == result.rewritten
    assert again.terms == result.terms


def test_offline_leaves_an_unmatched_query_unchanged():
    llm = LLMClient.offline()
    result = rewrite_query("xyzzy plugh", llm)
    assert result.applied is False
    assert result.rewritten == result.original
    assert result.terms == []
