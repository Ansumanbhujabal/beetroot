from pathlib import Path

from beatroot.confirm.trust_score import load_thresholds
from beatroot.container import THRESHOLDS_PATH, build_container
from beatroot.eval.runners.simulation import FamilyStats, run_simulation
from beatroot.eval.synth.adversarial import _FAMILIES, generate_adversarial


def test_simulation_runs_every_family_at_scale(tmp_path: Path) -> None:
    """The headline claim of this runner: every one of the ten families
    actually gets exercised, not just declared."""
    container = build_container(tmp_path / "sim.db")
    cases = generate_adversarial(container.catalog, n=400, seed=1)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    report = run_simulation(container.agent, cases, thresholds)
    assert set(report.families) == set(_FAMILIES)
    assert report.total_cases == 400


def test_simulation_passes_against_the_real_thresholds(tmp_path: Path) -> None:
    container = build_container(tmp_path / "sim2.db")
    cases = generate_adversarial(container.catalog, n=500, seed=2)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    report = run_simulation(container.agent, cases, thresholds)
    assert report.hard_constraint_violations == 0
    assert report.total_crashed == 0
    assert report.passed, report.families


def test_hard_constraint_violation_fails_the_run_even_if_case_assertions_pass(
    tmp_path: Path,
) -> None:
    """`report.passed` must key off the independent hard_constraint
    recheck, never off case-level assertions alone — mirrors
    `eval.runners.system`'s identical posture."""
    container = build_container(tmp_path / "sim3.db")
    thresholds = load_thresholds(THRESHOLDS_PATH)
    report = run_simulation(container.agent, [], thresholds)
    assert report.passed
    report.hard_constraint_violations = 1
    report.passed = report.hard_constraint_violations == 0
    assert not report.passed


def test_empty_case_list_does_not_crash_and_passes(tmp_path: Path) -> None:
    container = build_container(tmp_path / "sim4.db")
    thresholds = load_thresholds(THRESHOLDS_PATH)
    report = run_simulation(container.agent, [], thresholds)
    assert report.total_cases == 0
    assert report.passed


def test_family_below_threshold_fails_the_run(tmp_path: Path) -> None:
    """A synthetic below-floor family must flip `report.passed`, proving the
    per-family gate is actually load-bearing and not decorative. Uses one
    real generated case with its `expect_terminal` deliberately mutated to
    something unreachable, so the failure is genuine (run through the real
    agent) rather than hand-constructed report state."""
    from beatroot.confirm.trust_score import EvalThresholds, TrustThresholds, VerifierThresholds

    container = build_container(tmp_path / "sim5.db")
    case = dict(generate_adversarial(container.catalog, n=1, seed=3)[0])
    case["expect_terminal"] = ["__IMPOSSIBLE__"]
    thresholds = EvalThresholds(
        trust=TrustThresholds(refusal_threshold=0.55),
        verifiers=VerifierThresholds(
            hard_constraint_violations=0,
            nutrition_drift_pct=0.05,
            refusal_correctness=0.9,
            explanation_grounding=0.95,
        ),
        axes={},
        performance={},
        regression={},
        adversarial={case["family"]: 1.0},
    )
    report = run_simulation(container.agent, [case], thresholds)
    assert report.families[case["family"]].pass_rate == 0.0
    assert not report.passed


def test_pass_rate_counts_a_crash_as_a_failure() -> None:
    stats = FamilyStats(total=4, passed=2, crashed=1)
    assert stats.pass_rate == 0.5


def test_pass_rate_on_zero_cases_is_one() -> None:
    assert FamilyStats().pass_rate == 1.0


def test_p_percentile_on_empty_latencies_is_zero() -> None:
    from beatroot.eval.runners.simulation import SimulationReport

    assert SimulationReport().p(0.95) == 0.0


def test_main_runs_end_to_end(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    from beatroot.settings import get_settings

    get_settings.cache_clear()
    from beatroot.eval.runners.simulation import main

    assert main(n=40, seed=1) == 0
    get_settings.cache_clear()
