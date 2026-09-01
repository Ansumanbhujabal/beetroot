"""The system-level eval runner. Spec §12.

Drives every golden case (`eval/golden/seed_cases.yaml`) through the real
`MealPlanningAgent` — the full graph, not a mock — and scores the outcome
against the six adversarial-family axes declared in `config/beatroot.yaml`
(`eval.axis_by_family`). Adding a new adversarial family means adding rows to
that config map, never editing an `if`/`match` chain here.

Every COMMIT is re-checked by `eval.verifiers.hard_constraint`, which
deliberately does NOT call `t0_invariants.constraints.is_legal()` — the exact
function the agent's own retrieval path uses to decide legality. A verifier
sharing an implementation with the thing it verifies proves nothing; this is
the last line of defense against a bug that is present in both the graph's
filtering AND a same-shaped check written to test it.

The A5 (`escalation_correctness`) oracle, `_oracle_has_valid_meal` below,
gets the same treatment for the same reason: it calls
`eval.verifiers.vocabulary.unknown_vocabulary`, a from-scratch
reimplementation, never `t0_invariants.vocabulary.unknown_vocabulary` — see
that module's docstring for the tautology this replaced.

Runnable directly: `uv run python -m beatroot.eval.runners.system`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from beatroot.agent.graph import MealPlanningAgent
from beatroot.confirm.trust_score import EvalThresholds
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Escalation, Negotiation, Recommendation
from beatroot.eval.verifiers import hard_constraint, refusal_correctness
from beatroot.eval.verifiers import vocabulary as vocabulary_oracle

_Terminal = Negotiation | Escalation | Recommendation | None


@dataclass
class SystemReport:
    axes: dict[str, float] = field(default_factory=dict)
    violations: int = 0
    passed: bool = True
    failures: list[dict[str, Any]] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    cost_usd: float = 0.0

    def p(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(int(len(ordered) * pct), len(ordered) - 1)]


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text())
    return list(data) if data else []


def _oracle_has_valid_meal(agent: MealPlanningAgent, cs: ConstraintSet) -> bool:
    """A refusal-correctness oracle independent of the terminal it is used to
    judge: it never asks the agent, `is_legal()`, or `t0_invariants.
    feasibility.assess()` — it re-scans the whole catalog with the SAME
    independent hard-constraint verifier used below to police COMMIT
    results. Soft (GOAL/PREFERENCE) constraints are deliberately ignored: the
    question this answers is "does a SAFE meal exist at all", which is what
    a refusal is or isn't correct about; whether a preference is ALSO
    satisfiable is what NEGOTIATE (and `assert_relaxations_offered`) is
    for, not this oracle.

    A constraint naming a tag/ingredient absent from the catalog vocabulary
    (`eval.verifiers.vocabulary.unknown_vocabulary` — a from-scratch
    reimplementation for exactly this oracle, importing nothing from
    `t0_invariants`; see that module's docstring for why calling the
    production predicate here would make A5 a tautology) makes the answer
    FALSE outright, regardless of what `hard_constraint.verify` reports: an
    unverifiable exclusion is, by construction, never violated by ANY
    recipe (nothing is `in` a vocabulary that doesn't exist), so scanning
    the catalog for a violation-free recipe would always "find" one and
    call an unverifiable profile safe — the exact silent-pass this oracle
    exists to refuse to reproduce. No meal can be PROVEN safe against a
    constraint that cannot be checked at all, so the honest oracle answer
    is "no", which is what makes ESCALATE the correct (not over-cautious)
    terminal for g20-g22.
    """
    if vocabulary_oracle.unknown_vocabulary(cs, agent.deps.catalog):
        return False
    if vocabulary_oracle.uncheckable_constraints(cs):
        # Same principle one step out: a constraint that is vocabulary-valid
        # but still unevaluable (a nutrient the catalog does not track)
        # cannot be proven satisfied by any recipe either.
        return False
    return any(not hard_constraint.verify(r, cs) for r in agent.deps.catalog.recipes())


def _structural_checks(case: dict[str, Any], result: _Terminal) -> tuple[bool, str]:
    if (
        case.get("assert_relaxations_offered")
        and isinstance(result, Negotiation)
        and not result.relaxations
    ):
        return False, "no relaxations offered"

    for cid in case.get("assert_locked_contains", []):
        if isinstance(result, Negotiation) and cid not in result.locked:
            return False, f"{cid} not locked"

    return True, ""


def _passes(
    case: dict[str, Any],
    terminal: str,
    result: _Terminal,
    agent: MealPlanningAgent,
    cs: ConstraintSet,
) -> tuple[bool, str]:
    expected: list[str] = case["expect_terminal"]

    if len(expected) == 1:
        # A tightly-specified case names exactly one correct terminal, so it
        # gets the independent two-direction refusal check (both under- and
        # over-refusal). A loosely-specified case ("COMMIT or ESCALATE are
        # both a SAFE outcome here", spec §12) only gets the plain
        # membership test — running the strict oracle against a case that
        # deliberately accepts either terminal would flag a legitimately
        # cautious escalation as a false "over-refusal".
        oracle = _oracle_has_valid_meal(agent, cs)
        ok, why = refusal_correctness.verify(terminal, expected, oracle)
    else:
        ok = terminal in expected
        why = "" if ok else f"terminal {terminal} not in {expected}"
    if not ok:
        return ok, why

    ok, why = _structural_checks(case, result)
    if not ok:
        return ok, why

    if isinstance(result, Recommendation):
        recipe = agent.deps.catalog.recipe(result.recipe_id)
        if recipe is None:
            return False, f"committed recipe_id {result.recipe_id!r} not in catalog"
        for tag in case.get("assert_absent_tags", []):
            if tag in recipe.tags:
                return False, f"HARD VIOLATION: {tag} present in {recipe.id}"
    return True, ""


def run_system(
    agent: MealPlanningAgent, cases: list[dict[str, Any]], thresholds: EvalThresholds
) -> SystemReport:

    from beatroot.settings import get_settings

    settings = get_settings()
    if agent.deps.explanation_queue is not None:
        # A6 (explanation_grounding) is UNMEASURABLE when this agent runs the
        # ASYNC explanation path, and would silently report a meaningless
        # 1.000.
        #
        # A6 is scored from the `drift_bait` family, which relies on
        # `verify_node` diffing the explanation's numbers against catalog
        # truth. With a queue wired in, `state["explanation"]` is "" at
        # VERIFY — the prose is generated afterwards — so verify sees no
        # numbers, finds no drift, and every drift_bait case passes no matter
        # what the model said. Proven by mutation: a stub stating 9000 kcal
        # for every meal leaves A6 at 1.000 on the async path and drops it to
        # 0.000 on the sync one.
        #
        # This inspects THE AGENT rather than global config, because the two
        # legitimately differ: the eval runners build their container with
        # `async_explanation=False` while `config/beatroot.yaml` ships it on
        # for request latency. Checking the global setting here rejected a
        # correctly-wired synchronous agent.
        #
        # Refusing to run is the only honest option — silently scoring 1.000
        # is how this axis spent the whole build looking green while checking
        # nothing. The async path's own grounding guarantee is covered by
        # tests/agent/test_async_explanation_is_grounded.py.
        raise RuntimeError(
            "A6 (explanation_grounding) cannot be measured against an agent "
            "using the async explanation path: VERIFY sees an empty "
            "explanation, so drift_bait cases pass unconditionally. Build the "
            "container with build_container(async_explanation=False)."
        )

    axis_by_family = settings.eval.axis_by_family
    report = SystemReport()
    tally: dict[str, list[bool]] = {}

    for case in cases:
        cs = ConstraintSet(
            profile_id=case["id"],
            constraints=[Constraint(**c) for c in case.get("constraints", [])],
        )
        started = perf_counter()
        result: _Terminal = agent.run(
            cs, query=case.get("query", ""), preferences=case.get("preferences", "")
        )
        if result is None:
            # Paused at interrupt_before=["commit"] awaiting human approval
            # (grey-band trust on a MEDICAL profile). A golden-dataset run
            # has no human in the loop; auto-approve so every case reaches a
            # real terminal instead of silently scoring against a trace that
            # never finished.
            thread_id = agent.last_thread_id
            if thread_id is not None:
                result = agent.resume(thread_id, approved=True)
        report.latencies_ms.append((perf_counter() - started) * 1000)
        terminal = agent.trace[-1] if agent.trace else "NONE"

        ok, why = _passes(case, terminal, result, agent, cs)

        family = case["family"]
        axis = axis_by_family.get(family)
        if axis is None:
            raise KeyError(
                f"case {case['id']} declares family {family!r} with no axis in "
                "config eval.axis_by_family — an unmapped family would score "
                "nothing while appearing to pass"
            )
        tally.setdefault(axis, []).append(ok)

        if isinstance(result, Recommendation):
            recipe = agent.deps.catalog.recipe(result.recipe_id)
            if recipe is not None:
                hard = hard_constraint.verify(recipe, cs)
                if hard:
                    report.violations += 1
                    ok = False
                    why = f"independent verifier caught violated constraint(s) {hard}"
                    tally[axis][-1] = False

        if isinstance(result, Recommendation | Negotiation | Escalation):
            report.cost_usd += result.cost.usd

        if not ok:
            report.failures.append({"case": case["id"], "axis": axis, "why": why})

    report.axes = {a: (sum(v) / len(v) if v else 1.0) for a, v in tally.items()}

    report.passed = report.violations <= thresholds.verifiers.hard_constraint_violations
    for axis_name, floor in thresholds.axes.items():
        if report.axes.get(axis_name, 1.0) < float(floor):
            report.passed = False
    return report


def _print_report(report: SystemReport, thresholds: EvalThresholds) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table("axis", "score", "threshold", "status")
    for axis, floor in thresholds.axes.items():
        got = report.axes.get(axis)
        mark = "-" if got is None else ("PASS" if got >= float(floor) else "FAIL")
        table.add_row(axis, "n/a" if got is None else f"{got:.3f}", f"{floor}", mark)
    console.print(table)
    console.print(f"hard constraint violations: [bold]{report.violations}[/bold] (threshold 0)")
    console.print(
        f"p50 {report.p(0.50):.0f}ms  p95 {report.p(0.95):.0f}ms  cost ${report.cost_usd:.4f}"
    )
    for f in report.failures:
        console.print(f"[red]{f['case']}[/red] ({f['axis']}) {f['why']}")
    verdict = "[bold green]PASS[/bold green]" if report.passed else "[bold red]FAIL[/bold red]"
    console.print(f"\noverall: {verdict}")


def main() -> int:
    """`uv run python -m beatroot.eval.runners.system`.

    The `beatroot eval` CLI command (out of scope for this module — see the
    Task 13 report) is meant to be a thin wrapper over this exact
    `run_system`/`load_cases` pair, never a second implementation.

    Also persists the result to `eval/last_run.json`
    (`eval.artifact.write_system_result`) alongside the console table —
    the ONLY way that artifact is ever produced. `GET /evals/summary`
    reads it; it never runs this suite itself (see that route's
    docstring for why running an eval suite inside a request handler is
    the bug this exists to fix).
    """
    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import ROOT, THRESHOLDS_PATH, build_container
    from beatroot.eval.artifact import write_system_result


    container = build_container(async_explanation=False)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    cases = load_cases(ROOT / "eval" / "golden" / "seed_cases.yaml")
    report = run_system(container.agent, cases, thresholds)
    _print_report(report, thresholds)
    write_system_result(report, thresholds)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
