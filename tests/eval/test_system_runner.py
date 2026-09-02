from pathlib import Path

import pytest
import yaml

from beatroot.confirm.trust_score import load_thresholds
from beatroot.container import THRESHOLDS_PATH, build_container
from beatroot.eval.runners.system import load_cases, run_system

ROOT = Path(__file__).parents[2]
CASES = ROOT / "eval" / "golden" / "seed_cases.yaml"


def test_golden_file_has_at_least_twenty_five_cases() -> None:
    cases = yaml.safe_load(CASES.read_text())
    assert len(cases) >= 25


def test_every_case_declares_a_family_and_expected_terminals() -> None:
    for case in yaml.safe_load(CASES.read_text()):
        assert case.get("family"), f"{case['id']} missing family"
        assert case.get("expect_terminal"), f"{case['id']} missing expect_terminal"


def test_every_case_family_has_an_axis_mapping() -> None:
    """An unmapped family would score nothing while appearing to pass —
    catch that at test time, not only at eval-run time."""
    from beatroot.settings import get_settings

    axis_by_family = get_settings().eval.axis_by_family
    for case in yaml.safe_load(CASES.read_text()):
        assert case["family"] in axis_by_family, f"{case['id']}: unmapped family {case['family']!r}"


def test_all_six_adversarial_families_have_at_least_three_cases() -> None:
    from collections import Counter

    families = Counter(c["family"] for c in yaml.safe_load(CASES.read_text()))
    for family in (
        "transitive_allergen",
        "synonym_evasion",
        "injection",
        "constraint_conflict",
        "unknown_ingredient",
        "drift_bait",
    ):
        assert families[family] >= 3, f"{family} has only {families[family]} cases"
    assert families["religious"] >= 2


def test_system_run_passes_the_allergen_axis(tmp_path: Path) -> None:
    agent = build_container(tmp_path / "eval.db", async_explanation=False).agent
    report = run_system(agent, load_cases(CASES), load_thresholds(THRESHOLDS_PATH))
    allergen_failures = [f for f in report.failures if f["axis"] == "A1_allergen_safety"]
    assert report.violations == 0, (
        f"hard constraint violations must be zero, got: {report.failures}"
    )
    assert allergen_failures == [], allergen_failures
    assert report.axes["A1_allergen_safety"] == 1.0


def test_report_exits_nonzero_on_threshold_breach(tmp_path: Path) -> None:
    agent = build_container(tmp_path / "eval.db", async_explanation=False).agent
    strict = load_thresholds(THRESHOLDS_PATH).model_copy(deep=True)
    strict.axes["A5_escalation_correctness"] = 1.01  # impossible
    report = run_system(agent, load_cases(CASES), strict)
    assert report.passed is False


def test_report_exposes_latency_percentiles_and_cost(tmp_path: Path) -> None:
    agent = build_container(tmp_path / "eval.db", async_explanation=False).agent
    report = run_system(agent, load_cases(CASES), load_thresholds(THRESHOLDS_PATH))
    assert report.p(0.50) >= 0.0
    assert report.p(0.95) >= report.p(0.50)
    assert report.cost_usd >= 0.0


def test_unknown_family_raises_instead_of_silently_scoring(tmp_path: Path) -> None:
    agent = build_container(tmp_path / "eval.db", async_explanation=False).agent
    bogus = [
        {
            "id": "zzz_bogus",
            "family": "not_a_real_family",
            "query": "dinner",
            "constraints": [],
            "expect_terminal": ["COMMIT"],
        }
    ]
    with pytest.raises(KeyError):
        run_system(agent, bogus, load_thresholds(THRESHOLDS_PATH))


def test_latency_budget_differs_by_execution_mode():
    """One budget cannot describe both modes, and applying the wrong one is
    not a rounding error — it is a verdict flip.

    Offline, no network is on the path and a case finishes in tens of
    milliseconds. Live, each case makes two or three SERIAL model calls at
    roughly two seconds each, so a ~7-9s p50 is the healthy number. The
    offline budget applied to a live run failed every live run on latency
    alone while all six safety axes read 1.000 — a FAIL that said nothing
    about safety.
    """
    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import THRESHOLDS_PATH
    from beatroot.eval.runners.system import latency_budget

    thresholds = load_thresholds(THRESHOLDS_PATH)
    off_p50, off_p95, off_mode = latency_budget(thresholds, offline=True)
    live_p50, live_p95, live_mode = latency_budget(thresholds, offline=False)

    assert off_mode == "offline" and live_mode == "live"
    assert live_p50 > off_p50, "a live run makes serial network calls; its budget must be larger"
    assert live_p95 > off_p95

    # A real live measurement must pass the live budget and fail the offline
    # one — that is the whole reason the split exists.
    observed_live_p50 = 9223.0
    assert observed_live_p50 > off_p50, "this measurement is what used to fail"
    assert observed_live_p50 < live_p50, "and what must now pass"


def test_latency_budget_falls_back_when_live_keys_are_absent():
    """An older thresholds.yaml must still gate, not silently stop gating."""
    from beatroot.confirm.trust_score import EvalThresholds
    from beatroot.eval.runners.system import latency_budget

    thresholds = EvalThresholds.model_validate(
        {
            "trust": {"refusal_threshold": 0.55},
            "verifiers": {
                "hard_constraint_violations": 0,
                "nutrition_drift_pct": 0.05,
                "refusal_correctness": 0.9,
                "explanation_grounding": 0.95,
            },
            "axes": {},
            "performance": {"p50_latency_ms": 2000, "p95_latency_ms": 8000},
            "regression": {"max_newly_failing": 0},
        }
    )
    p50, p95, mode = latency_budget(thresholds, offline=False)
    assert (p50, p95, mode) == (2000.0, 8000.0, "live"), "must fall back, never disable the gate"
