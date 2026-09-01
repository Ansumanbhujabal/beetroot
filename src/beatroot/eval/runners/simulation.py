"""The adversarial simulation runner. Spec §12.

Where `eval.runners.system` proves the composed pipeline against 33
hand-written golden cases, this module attacks it at scale: it generates N
cases across all ten `eval.synth.adversarial` families and runs every one
through the real `MealPlanningAgent`, then reports **per-family** pass
rates rather than one blended number — a 95% overall figure can hide one
family sitting at 0%, and that is exactly the kind of result this runner
exists to surface rather than average away.

Every COMMIT is re-checked by the same independent `eval.verifiers.
hard_constraint.verify` `eval.runners.system` uses — never
`t0_invariants.constraints.is_legal()` — for the identical reason: a
verifier sharing an implementation with the thing it verifies proves
nothing. `report.hard_constraint_violations` is a COUNT gate, not a rate,
matching `eval/thresholds.yaml`'s existing `verifiers.hard_constraint_
violations` posture.

A CRASH (an unhandled exception escaping `agent.run()`/`agent.resume()`) is
tracked separately from a case that reached a real terminal and then failed
its assertion. Those are different defect classes — one means the system
degraded ungracefully and stopped serving requests entirely, the other
means it answered and the answer was wrong — and folding them into one
"failed" bucket would hide which one actually happened.

Runnable directly: `uv run python -m beatroot.eval.runners.simulation
[--n N] [--seed SEED]`. Wired as `beatroot eval simulation`.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from beatroot.agent.graph import MealPlanningAgent
from beatroot.confirm.trust_score import EvalThresholds
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Escalation, Negotiation, Recommendation
from beatroot.eval.synth.adversarial import generate_adversarial
from beatroot.eval.verifiers import hard_constraint

_Terminal = Negotiation | Escalation | Recommendation | None


@dataclass
class FamilyStats:
    total: int = 0
    passed: int = 0
    crashed: int = 0
    terminal_counts: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    crashes: list[dict[str, str]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """A crash counts as a failed case for this rate — it is never
        silently excluded from the denominator, only reported through a
        SEPARATE channel (`crashed`/`crashes`) as well."""
        return self.passed / self.total if self.total else 1.0


@dataclass
class SimulationReport:
    families: dict[str, FamilyStats] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    cost_usd: float = 0.0
    hard_constraint_violations: int = 0
    passed: bool = True

    def p(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(int(len(ordered) * pct), len(ordered) - 1)]

    @property
    def total_cases(self) -> int:
        return sum(s.total for s in self.families.values())

    @property
    def total_crashed(self) -> int:
        return sum(s.crashed for s in self.families.values())


def _check_case(
    case: dict[str, Any], terminal: str, result: _Terminal, agent: MealPlanningAgent
) -> tuple[bool, str]:
    """Every assertion field a generated case may carry — see
    `eval.synth.adversarial`'s module docstring for which families use
    which. Unlike `eval.runners.system._passes`, this never invokes
    `refusal_correctness`'s independent oracle: these cases are generated,
    not hand-picked to be tightly-specified, so `expect_terminal` is the
    contract, exactly as it is for a golden case with more than one
    accepted terminal.
    """
    expected: list[str] = case["expect_terminal"]
    if terminal not in expected:
        return False, f"terminal {terminal} not in {expected}"

    if isinstance(result, Recommendation):
        recipe = agent.deps.catalog.recipe(result.recipe_id)
        if recipe is None:
            return False, f"committed recipe_id {result.recipe_id!r} not in catalog"
        for tag in case.get("assert_absent_tags", []):
            if tag in recipe.tags:
                return False, f"HARD VIOLATION: {tag} present in {recipe.id}"
        target_ingredient = case.get("assert_absent_ingredient")
        if target_ingredient and target_ingredient in recipe.ingredient_ids:
            return False, f"HARD VIOLATION: ingredient {target_ingredient} present in {recipe.id}"

    if isinstance(result, Negotiation):
        for cid in case.get("assert_locked_contains", []):
            if cid not in result.locked:
                return False, f"{cid} not locked"
        if case.get("assert_relaxations_offered") and not result.relaxations:
            return False, "no relaxations offered"
        if case.get("assert_no_relaxations") and result.relaxations:
            return False, f"expected no relaxations, got {len(result.relaxations)}"

    if isinstance(result, Escalation):
        want = case.get("assert_escalate_reason")
        if want is not None and result.reason != want:
            return False, f"expected escalate reason {want!r}, got {result.reason!r}"
        not_want = case.get("assert_escalate_reason_not")
        if not_want is not None and result.reason == not_want:
            return False, f"escalate reason must not be {not_want!r}"

    return True, ""


def run_simulation(
    agent: MealPlanningAgent, cases: list[dict[str, Any]], thresholds: EvalThresholds
) -> SimulationReport:
    report = SimulationReport()

    for case in cases:
        family = case["family"]
        stats = report.families.setdefault(family, FamilyStats())
        stats.total += 1

        cs = ConstraintSet(
            profile_id=case["id"],
            constraints=[Constraint(**c) for c in case.get("constraints", [])],
        )

        started = perf_counter()
        try:
            result: _Terminal = agent.run(
                cs, query=case.get("query", ""), preferences=case.get("preferences", "")
            )
            if result is None:
                # Paused at interrupt_before=["commit"] awaiting human
                # approval (grey-band trust on a MEDICAL profile) — same
                # auto-approve `eval.runners.system.run_system` uses so
                # every case reaches a real terminal.
                thread_id = agent.last_thread_id
                if thread_id is not None:
                    result = agent.resume(thread_id, approved=True)
            terminal = agent.trace[-1] if agent.trace else "NONE"
        except Exception as exc:  # broad on purpose: a crash is what this bucket exists to catch
            report.latencies_ms.append((perf_counter() - started) * 1000)
            stats.crashed += 1
            stats.crashes.append(
                {
                    "case": case["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        report.latencies_ms.append((perf_counter() - started) * 1000)
        stats.terminal_counts[terminal] = stats.terminal_counts.get(terminal, 0) + 1

        ok, why = _check_case(case, terminal, result, agent)

        if isinstance(result, Recommendation):
            recipe = agent.deps.catalog.recipe(result.recipe_id)
            if recipe is not None:
                hard = hard_constraint.verify(recipe, cs)
                if hard:
                    report.hard_constraint_violations += 1
                    ok = False
                    why = f"independent verifier caught violated constraint(s) {hard}"

        if isinstance(result, Recommendation | Negotiation | Escalation):
            report.cost_usd += result.cost.usd

        if ok:
            stats.passed += 1
        else:
            stats.failures.append({"case": case["id"], "why": why})

    report.passed = report.hard_constraint_violations == 0
    for family, stats in report.families.items():
        floor = thresholds.adversarial.get(family)
        if floor is not None and stats.pass_rate < floor:
            report.passed = False

    return report


def _print_report(report: SimulationReport, thresholds: EvalThresholds) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        "family",
        "n",
        "pass rate",
        "threshold",
        "status",
        "crashed",
        "COMMIT",
        "NEGOTIATE",
        "ESCALATE",
    )
    for family in sorted(report.families):
        stats = report.families[family]
        floor = thresholds.adversarial.get(family)
        mark = "-" if floor is None else ("PASS" if stats.pass_rate >= floor else "FAIL")
        table.add_row(
            family,
            str(stats.total),
            f"{stats.pass_rate:.3f}",
            "n/a" if floor is None else f"{floor}",
            mark,
            str(stats.crashed),
            str(stats.terminal_counts.get("COMMIT", 0)),
            str(stats.terminal_counts.get("NEGOTIATE", 0)),
            str(stats.terminal_counts.get("ESCALATE", 0)),
        )
    console.print(table)
    console.print(
        f"total cases: {report.total_cases}  crashed: {report.total_crashed}  "
        f"hard constraint violations: [bold]{report.hard_constraint_violations}"
        "[/bold] (threshold 0)"
    )
    console.print(
        f"p50 {report.p(0.50):.0f}ms  p95 {report.p(0.95):.0f}ms  cost ${report.cost_usd:.4f}"
    )
    for family in sorted(report.families):
        stats = report.families[family]
        for c in stats.crashes[:5]:
            console.print(f"[bold red]CRASH[/bold red] {family}/{c['case']}: {c['error']}")
        for f in stats.failures[:5]:
            console.print(f"[red]{family}/{f['case']}[/red] {f['why']}")
    verdict = "[bold green]PASS[/bold green]" if report.passed else "[bold red]FAIL[/bold red]"
    console.print(f"\noverall: {verdict}")


def main(n: int | None = None, seed: int = 0) -> int:
    """`uv run python -m beatroot.eval.runners.simulation [--n N] [--seed SEED]`.

    `n` defaults to `settings.synth.default_adversarial` (via
    `generate_adversarial`) when not given.
    """
    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import THRESHOLDS_PATH, build_container

    container = build_container()
    thresholds = load_thresholds(THRESHOLDS_PATH)
    cases = generate_adversarial(container.catalog, n=n, seed=seed)
    report = run_simulation(container.agent, cases, thresholds)
    _print_report(report, thresholds)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the adversarial simulation suite.")
    parser.add_argument(
        "--n", type=int, default=None, help="how many cases (default: settings.synth)"
    )
    parser.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    args = parser.parse_args()
    sys.exit(main(n=args.n, seed=args.seed))
