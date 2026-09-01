from itertools import pairwise
from pathlib import Path

import pytest

from beatroot.eval.calibration import (
    collect_commit_pairs,
    expected_calibration_error,
    reliability_curve,
)


def test_perfectly_calibrated_has_zero_error() -> None:
    pairs = [(1.0, True)] * 50 + [(0.0, False)] * 50
    assert expected_calibration_error(pairs) == pytest.approx(0.0, abs=1e-9)


def test_overconfident_model_has_high_error() -> None:
    """Always says 0.95, is right half the time."""
    pairs = [(0.95, i % 2 == 0) for i in range(100)]
    assert expected_calibration_error(pairs) == pytest.approx(0.45, abs=0.01)


def test_reliability_curve_bins_are_contiguous_and_cover_zero_to_one() -> None:
    curve = reliability_curve([(0.1, True), (0.9, False)], bins=10)
    assert curve[0].lower == pytest.approx(0.0)
    assert curve[-1].upper == pytest.approx(1.0)
    for a, b in pairwise(curve):
        assert a.upper == pytest.approx(b.lower)


def test_top_bin_includes_a_confidence_of_exactly_one() -> None:
    curve = reliability_curve([(1.0, True)], bins=10)
    assert curve[-1].count == 1
    assert curve[-1].accuracy == pytest.approx(1.0)


def test_empty_input_is_zero_not_a_crash() -> None:
    assert expected_calibration_error([]) == 0.0
    assert reliability_curve([]) == []


def test_bins_parameter_changes_granularity() -> None:
    pairs = [(0.05, True), (0.95, False)]
    assert len(reliability_curve(pairs, bins=4)) == 4
    assert len(reliability_curve(pairs, bins=20)) == 20


def test_ece_never_exceeds_one() -> None:
    pairs = [(0.0, True)] * 10 + [(1.0, False)] * 10
    assert 0.0 <= expected_calibration_error(pairs) <= 1.0


# ---- real-system calibration -------------------------------------------


def test_collect_commit_pairs_against_the_real_agent(tmp_path: Path) -> None:
    """Runs the real agent over synthetic cases and reports the real ECE —
    the number that actually matters, per Spec §12. Offline, so this must
    stay deterministic and network-free (see tests/eval/conftest.py)."""
    from beatroot.container import build_container
    from beatroot.eval.synth.profiles import generate_profiles

    container = build_container(tmp_path / "cal.db")
    cases = generate_profiles(container.catalog, n=25, seed=13)
    pairs = collect_commit_pairs(container.agent, cases)

    # Every trust.composite is a valid probability, and every pair's second
    # element is a bool — the shape collect_commit_pairs promises.
    for confidence, correct in pairs:
        assert 0.0 <= confidence <= 1.0
        assert isinstance(correct, bool)

    ece = expected_calibration_error(pairs)
    assert 0.0 <= ece <= 1.0
    curve = reliability_curve(pairs)
    if pairs:
        assert len(curve) == 10
    else:
        assert curve == []
