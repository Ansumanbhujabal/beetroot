"""The eval ITERATION runner: one command that runs the whole suite —
system, components, adversarial simulation, calibration — against the real
agent and writes an attributed snapshot. Spec §12 extension (this task).

Where `eval.runners.system`/`components`/`simulation` and `eval.calibration`
each answer "what does this one layer measure right now", this module
answers "did the change I just made actually help, measured against
everything at once, with a name attached to it". Every call MUST pass a
`label` and `note` — a run with no attribution is exactly the "just show a
number moving" failure mode this tool exists to prevent.

Runnable directly: `uv run python -m beatroot.eval.runners.iterate --label
X --note "..."`. Wired as `beatroot eval iterate`.
"""

from __future__ import annotations

from typing import Any

from beatroot.confirm.trust_score import EvalThresholds


def run_iteration(
    *,
    label: str,
    note: str,
    verdict: str = "",
    reason: str = "",
    profiles_seed: int = 0,
    n_adversarial: int | None = None,
    adversarial_seed: int = 0,
    calibration_n: int | None = None,
) -> dict[str, Any]:
    """Run system + components + simulation + calibration once, against a
    freshly built container, and return the assembled snapshot (already
    written to `eval/history/` and folded into a regenerated
    `EVAL_HISTORY.md`).

    Whether this run is offline or live is read from `settings.offline` —
    the SAME toggle every other entry point in this codebase respects
    (`BEATROOT_OFFLINE=1`, or credentials simply absent). This function
    never overrides it: a caller who wants a live measurement sets the
    real environment before invoking this, exactly like every other
    command in `beatroot`.

    `calibration_n`/`n_adversarial` default to `settings.synth`'s own
    defaults when omitted (200 profiles, 100 adversarial cases) — the same
    defaults every other CLI command already uses. A live run should pass
    smaller explicit values: each profile/case here drives the FULL agent
    end to end, and a live call costs real seconds and real cents that an
    offline stub does not.
    """
    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import ROOT, THRESHOLDS_PATH, build_container
    from beatroot.eval.calibration import (
        collect_commit_pairs,
        expected_calibration_error,
        reliability_curve,
    )
    from beatroot.eval.history import build_snapshot, regenerate_history_md, write_snapshot
    from beatroot.eval.runners.components import run_components
    from beatroot.eval.runners.simulation import run_simulation
    from beatroot.eval.runners.system import load_cases, run_system
    from beatroot.eval.synth.adversarial import generate_adversarial
    from beatroot.eval.synth.profiles import generate_profiles
    from beatroot.settings import get_settings

    settings = get_settings()

    container = build_container(async_explanation=False)
    try:
        thresholds = load_thresholds(THRESHOLDS_PATH)

        system_cases = load_cases(ROOT / "eval" / "golden" / "seed_cases.yaml")
        system_report = run_system(container.agent, system_cases, thresholds)

        component_cases = generate_profiles(container.catalog, seed=profiles_seed)
        component_report = run_components(container, component_cases)

        adversarial_cases = generate_adversarial(
            container.catalog, n=n_adversarial, seed=adversarial_seed
        )
        simulation_report = run_simulation(container.agent, adversarial_cases, thresholds)

        calibration_cases = generate_profiles(
            container.catalog, n=calibration_n, seed=profiles_seed
        )
        calibration_pairs = collect_commit_pairs(container.agent, calibration_cases)
        calibration_ece = expected_calibration_error(calibration_pairs)

        metrics: dict[str, Any] = {
            "axes": system_report.axes,
            "system_passed": system_report.passed,
            "hard_constraint_violations": system_report.violations,
            "components": {
                "retrieval_recall_at_k": component_report.retrieval_recall_at_k,
                "retrieval_recall_at_k_hard_only": (
                    component_report.retrieval_recall_at_k_hard_only
                ),
                "retrieval_leakage": component_report.retrieval_leakage,
                "feasibility_accuracy": component_report.feasibility_accuracy,
                "nutrition_exact_match": component_report.nutrition_exact_match,
                "drift_detection_recall": component_report.drift_detection_recall,
            },
            "adversarial": {f: s.pass_rate for f, s in simulation_report.families.items()},
            "adversarial_passed": simulation_report.passed,
            "adversarial_hard_constraint_violations": simulation_report.hard_constraint_violations,
            "calibration": {
                "ece": calibration_ece,
                "pairs": len(calibration_pairs),
                # The reliability BINS, not just the scalar ECE. Without
                # these a low ECE is unreadable: 37 COMMIT pairs that all
                # land in one high-confidence bin and are all correct score
                # ~0.0 identically to genuinely well-spread calibration, and
                # the scalar alone cannot tell those apart. Persisting the
                # per-bin spread is what makes the number falsifiable later.
                "bins": [
                    {
                        "lower": b.lower,
                        "upper": b.upper,
                        "mean_confidence": b.mean_confidence,
                        "accuracy": b.accuracy,
                        "count": b.count,
                    }
                    for b in reliability_curve(calibration_pairs)
                    if b.count
                ],
                "distinct_confidences": len({c for c, _ in calibration_pairs}),
            },
        }

        config = _config_snapshot(settings, thresholds)

        entry = build_snapshot(
            label=label,
            note=note,
            offline=settings.offline,
            config=config,
            metrics=metrics,
            verdict=verdict,
            reason=reason,
        )
        write_snapshot(entry)
        regenerate_history_md()
        return entry
    finally:
        container.close()


def _config_snapshot(settings: Any, thresholds: EvalThresholds) -> dict[str, Any]:
    rc = settings.retrieval
    return {
        "retrieval": {
            "lexical_weight": rc.lexical_weight,
            "dense_weight": rc.dense_weight,
            "affinity_weight": rc.affinity_weight,
            "rrf_k": rc.rrf_k,
            "candidate_limit": rc.candidate_limit,
            "top_k": rc.top_k,
        },
        "llm": {
            "model": settings.llm.model,
            "embedding_model": settings.llm.embedding_model,
        },
        "thresholds": {
            "axes": dict(thresholds.axes),
            "adversarial": dict(thresholds.adversarial),
        },
    }


def _print_summary(entry: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(
        f"[bold]{entry['label']}[/bold] "
        f"({'offline' if entry['offline'] else 'LIVE'}, git {entry['git_sha'][:12]})"
    )
    console.print(f"note: {entry['note']}")
    m = entry["metrics"]
    table = Table("metric", "value")
    for axis, score in m["axes"].items():
        table.add_row(axis, f"{score:.3f}")
    for k, v in m["components"].items():
        table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    table.add_row("calibration ECE", f"{m['calibration']['ece']:.4f}")
    table.add_row("calibration pairs", str(m["calibration"]["pairs"]))
    console.print(table)
    console.print(
        f"system: {'PASS' if m['system_passed'] else 'FAIL'}  "
        f"adversarial: {'PASS' if m['adversarial_passed'] else 'FAIL'}"
    )


def main(
    label: str,
    note: str,
    verdict: str = "",
    reason: str = "",
    profiles_seed: int = 0,
    n_adversarial: int | None = None,
    adversarial_seed: int = 0,
    calibration_n: int | None = None,
) -> int:
    entry = run_iteration(
        label=label,
        note=note,
        verdict=verdict,
        reason=reason,
        profiles_seed=profiles_seed,
        n_adversarial=n_adversarial,
        adversarial_seed=adversarial_seed,
        calibration_n=calibration_n,
    )
    _print_summary(entry)
    return 0 if entry["metrics"]["system_passed"] and entry["metrics"]["adversarial_passed"] else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run one attributed eval iteration snapshot.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--verdict", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--profiles-seed", type=int, default=0)
    parser.add_argument("--n-adversarial", type=int, default=None)
    parser.add_argument("--adversarial-seed", type=int, default=0)
    parser.add_argument("--calibration-n", type=int, default=None)
    args = parser.parse_args()
    sys.exit(
        main(
            label=args.label,
            note=args.note,
            verdict=args.verdict,
            reason=args.reason,
            profiles_seed=args.profiles_seed,
            n_adversarial=args.n_adversarial,
            adversarial_seed=args.adversarial_seed,
            calibration_n=args.calibration_n,
        )
    )
