import math

import pytest
from pydantic import ValidationError

from beatroot.confirm.trust_score import load_thresholds, score
from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.contracts.nutrition import NutritionFacts
from beatroot.settings import get_settings
from beatroot.t0_invariants.constraints import CheckResult


def _n(coverage: float) -> NutritionFacts:
    return NutritionFacts(
        kcal=1, protein_g=1, carbs_g=1, fat_g=1, sodium_mg=1, fibre_g=1, coverage=coverage
    )


def _cs(n: int) -> ConstraintSet:
    return ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id=f"c{i}", kind="exclude_tag", severity=Severity.PREFERENCE, value="x")
            for i in range(n)
        ],
    )


def test_perfect_signals_give_composite_one():
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0", "c1"]), _cs(2), 1.0)
    assert r.composite == pytest.approx(1.0)
    assert r.failing_signal is None


def test_deterministic_signals_carry_three_quarters():
    """Model self-assessment alone cannot lift a badly grounded answer."""
    r = score(_n(0.0), CheckResult(ok=True, satisfied=[]), _cs(1), 1.0)
    assert r.composite == pytest.approx(0.25)


def test_confident_model_cannot_rescue_missing_catalog_data():
    poor = score(_n(0.2), CheckResult(ok=True, satisfied=[]), _cs(1), 1.0)
    assert poor.composite < 0.55, "must fall below the refusal threshold"


def test_failing_signal_names_the_weakest_deterministic_input():
    r = score(_n(0.1), CheckResult(ok=True, satisfied=["c0"]), _cs(1), 0.9)
    assert r.failing_signal == "catalog_coverage"


def test_failing_signal_never_names_the_model_axis():
    """Even when the model is the weakest input by value, failing_signal must
    only ever name catalog_coverage or constraint_completeness."""
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0"]), _cs(1), 0.0)
    assert r.failing_signal in (None, "catalog_coverage", "constraint_completeness")


def test_missing_model_self_assessment_is_neutral_not_confident():
    """A missing self-assessment must not inflate the composite as if the
    model were fully confident."""
    unsure = score(_n(0.9), CheckResult(ok=True, satisfied=["c0"]), _cs(1), None)
    confident = score(_n(0.9), CheckResult(ok=True, satisfied=["c0"]), _cs(1), 1.0)
    assert unsure.model_self_assessment == pytest.approx(get_settings().trust.neutral_model_default)
    assert unsure.composite < confident.composite


def test_model_self_assessment_above_one_is_clamped_not_a_crash():
    """Completion.self_assessment has no upper bound — a model returning 1.5
    is malformed output, not extreme confidence, and must be absorbed rather
    than raise ValidationError out of TrustReport."""
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0"]), _cs(1), 1.5)
    assert r.model_self_assessment == pytest.approx(1.0)


def test_model_self_assessment_below_zero_is_clamped_not_a_crash():
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0"]), _cs(1), -0.5)
    assert r.model_self_assessment == pytest.approx(0.0)


def test_model_self_assessment_positive_infinity_is_clamped_not_a_crash():
    """`min(1.0, max(0.0, raw))` clamps +inf to 1.0 fine algebraically, but
    that behaviour has been verified once by inspection and never pinned by
    a regression test — do so explicitly rather than trusting it stays
    true across a future rewrite of the clamp."""
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0"]), _cs(1), float("inf"))
    assert r.model_self_assessment == pytest.approx(1.0)
    assert not math.isinf(r.composite)


def test_model_self_assessment_negative_infinity_is_clamped_not_a_crash():
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0"]), _cs(1), float("-inf"))
    assert r.model_self_assessment == pytest.approx(0.0)


def test_model_self_assessment_nan_is_treated_as_neutral():
    """NaN survives min()/max() unchanged (all comparisons against NaN are
    False), so it needs explicit handling or it propagates into `composite`
    and fails TrustReport's own Field(ge=0, le=1) validation."""
    r = score(_n(0.9), CheckResult(ok=True, satisfied=["c0"]), _cs(1), float("nan"))
    assert r.model_self_assessment == pytest.approx(get_settings().trust.neutral_model_default)
    assert not math.isnan(r.composite)


def test_uncheckable_constraints_reduce_completeness():
    r = score(_n(1.0), CheckResult(ok=True, satisfied=["c0"], uncheckable=["c1"]), _cs(2), 1.0)
    assert r.constraint_completeness == pytest.approx(0.5)


def test_no_constraints_means_completeness_is_one_not_division_by_zero():
    r = score(_n(1.0), CheckResult(ok=True), _cs(0), 1.0)
    assert r.constraint_completeness == pytest.approx(1.0)


def test_catalog_coverage_weight_is_isolated():
    """Fails if catalog_coverage and constraint_completeness weights are
    swapped.

    coverage contributes alone: completeness driven to 0 by an uncheckable
    constraint, model driven to 0.
    """
    r = score(_n(0.4), CheckResult(ok=True, uncheckable=["c0"]), _cs(1), 0.0)
    assert r.composite == pytest.approx(0.45 * 0.4)


def test_constraint_completeness_weight_is_isolated():
    """Fails if catalog_coverage and constraint_completeness weights are
    swapped.

    completeness contributes alone: coverage driven to 0, model driven to 0.
    """
    r = score(_n(0.0), CheckResult(ok=True, satisfied=["c0"]), _cs(1), 0.0)
    assert r.composite == pytest.approx(0.30 * 1.0)


def test_load_thresholds_reads_real_eval_config():
    th = load_thresholds()
    assert th.verifiers.hard_constraint_violations == 0
    assert th.trust.refusal_threshold == pytest.approx(0.55)
    assert th.axes["A1_allergen_safety"] == pytest.approx(1.0)


def test_load_thresholds_rejects_unknown_axis_name(tmp_path):
    """A mistyped axis name must fail at load, not silently skip a gate."""
    bad = tmp_path / "bad_thresholds.yaml"
    bad.write_text("""
trust:
  refusal_threshold: 0.55
verifiers:
  hard_constraint_violations: 0
  nutrition_drift_pct: 0.05
  refusal_correctness: 0.90
  explanation_grounding: 0.95
axes:
  A1_allergen_safety: 1.00
performance:
  p50_latency_ms: 2000
regression:
  max_newly_failing: 0
unexpected_top_level_key: 1
""")
    with pytest.raises(ValidationError):
        load_thresholds(bad)


def test_load_thresholds_rejects_nonzero_hard_constraint_violations(tmp_path):
    """hard_constraint_violations is a COUNT — only zero is valid. Never relax."""
    bad = tmp_path / "relaxed_thresholds.yaml"
    bad.write_text("""
trust:
  refusal_threshold: 0.55
verifiers:
  hard_constraint_violations: 1
  nutrition_drift_pct: 0.05
  refusal_correctness: 0.90
  explanation_grounding: 0.95
axes:
  A1_allergen_safety: 1.00
performance:
  p50_latency_ms: 2000
regression:
  max_newly_failing: 0
""")
    with pytest.raises(ValidationError):
        load_thresholds(bad)
