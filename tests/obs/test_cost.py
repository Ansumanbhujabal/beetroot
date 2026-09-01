import pytest

from beatroot.obs.cost import CHARS_PER_TOKEN, CostLedger, estimate_tokens


def test_estimate_tokens_is_zero_for_empty_text():
    """Nothing to estimate for a prompt that was never even built."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_never_truncates_a_short_prompt_to_zero():
    """A short-but-real prompt must round up to at least 1 token, never
    silently disappear to 0 and read as if nothing was skipped."""
    assert estimate_tokens("hi") == 1


def test_estimate_tokens_scales_with_length_at_the_stated_rate():
    text = "x" * 400
    assert estimate_tokens(text) == round(len(text) / CHARS_PER_TOKEN) == 100


def test_add_accumulates_per_stage_cost():
    ledger = CostLedger()
    ledger.add("retrieve", 0.001, tokens=100)
    ledger.add("retrieve", 0.002, tokens=50)
    ledger.add("explain", 0.005, tokens=200)
    assert ledger.per_stage["retrieve"] == pytest.approx(0.003)
    assert ledger.per_stage["explain"] == pytest.approx(0.005)
    assert ledger.tokens == 350


def test_total_usd_sums_every_stage():
    ledger = CostLedger()
    ledger.add("retrieve", 0.001)
    ledger.add("explain", 0.004)
    assert ledger.total_usd == pytest.approx(0.005)


def test_per_plan_usd_is_derivable_and_zero_before_any_plan():
    ledger = CostLedger()
    assert ledger.per_plan_usd == 0.0
    ledger.add("explain", 0.010)
    ledger.record_plan()
    ledger.add("explain", 0.020)
    ledger.record_plan()
    # $0.030 total spend over 2 completed plans -> $0.015/plan, the
    # cost-per-plan headline metric.
    assert ledger.total_usd == pytest.approx(0.030)
    assert ledger.plans == 2
    assert ledger.per_plan_usd == pytest.approx(0.015)


def test_record_short_circuit_tracks_tokens_never_spent():
    """The argument for filter-before-generate: an infeasible profile or a
    feasibility-cache hit never reaches a model call at all."""
    ledger = CostLedger()
    ledger.record_short_circuit(estimated_tokens=800)
    ledger.record_short_circuit(estimated_tokens=200)
    assert ledger.tokens_saved == 1000
    # Tokens saved never inflate actual spend or actual token usage.
    assert ledger.tokens == 0
    assert ledger.total_usd == 0.0


def test_ledger_round_trips_through_json():
    ledger = CostLedger()
    ledger.add("retrieve", 0.0000001234)
    ledger.record_short_circuit(50)
    ledger.record_plan()
    restored = CostLedger.model_validate_json(ledger.model_dump_json())
    assert restored == ledger
