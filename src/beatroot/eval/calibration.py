"""Confidence calibration: does a trust score of 0.8 mean "right about 80%
of the time"? Spec §12.

`expected_calibration_error` / `reliability_curve` are pure statistics over
`(confidence, was_this_actually_correct)` pairs — no LLM, no catalog,
nothing domain-specific; the unit tests below only prove that arithmetic is
right. `collect_commit_pairs` is the one domain-specific function: it drives
`eval.synth.profiles`'s free oracle through the real `MealPlanningAgent` and
pairs each COMMIT's own `trust.composite` self-report against whether the
recipe it recommended was genuinely in the oracle's exact valid set. If the
refusal threshold sits at 0.55 but a composite of 0.55 is actually right 30%
of the time, the threshold is arbitrary, not calibrated — this module is
what makes that checkable instead of assumed.

Runnable directly: `uv run python -m beatroot.eval.calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beatroot.agent.graph import MealPlanningAgent
    from beatroot.eval.synth.profiles import SyntheticCase


@dataclass
class Bin:
    lower: float
    upper: float
    mean_confidence: float
    accuracy: float
    count: int


def reliability_curve(pairs: list[tuple[float, bool]], bins: int = 10) -> list[Bin]:
    """Bucket `(confidence, correct)` pairs into `bins` equal-width bins.

    Bins are contiguous and cover [0, 1] exactly: bin i is [i/bins,
    (i+1)/bins), half-open, EXCEPT the last bin, which also swallows a
    confidence of exactly 1.0 — otherwise a perfect-confidence pair would
    fall just past every bin's exclusive upper edge and vanish from the
    curve entirely. Empty input returns `[]`, not a crash.
    """
    if not pairs:
        return []
    width = 1.0 / bins
    out: list[Bin] = []
    for i in range(bins):
        lo, hi = i * width, (i + 1) * width
        inside = [(c, ok) for c, ok in pairs if (lo <= c < hi) or (i == bins - 1 and c == 1.0)]
        if inside:
            confs = [c for c, _ in inside]
            out.append(
                Bin(
                    lower=lo,
                    upper=hi,
                    mean_confidence=sum(confs) / len(confs),
                    accuracy=sum(ok for _, ok in inside) / len(inside),
                    count=len(inside),
                )
            )
        else:
            out.append(Bin(lower=lo, upper=hi, mean_confidence=0.0, accuracy=0.0, count=0))
    return out


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """Weighted average, over non-empty bins, of |accuracy - mean confidence|.

    Empty input is 0.0 — perfectly calibrated over zero evidence is the
    honest answer, not a crash and not a NaN propagating into a report.
    """
    if not pairs:
        return 0.0
    total = len(pairs)
    return sum(
        (b.count / total) * abs(b.accuracy - b.mean_confidence)
        for b in reliability_curve(pairs, bins)
        if b.count
    )


def collect_commit_pairs(
    agent: MealPlanningAgent, cases: list[SyntheticCase]
) -> list[tuple[float, bool]]:
    """Drive every synthetic `case` through the real agent; pair the trust
    score of every COMMIT against whether the oracle actually calls that
    recommendation valid.

    Only COMMIT terminals contribute a pair: NEGOTIATE and ESCALATE are
    refusals, not a confidence claim about one specific recipe, so there is
    nothing here for `trust.composite` to be right or wrong ABOUT. A run
    paused at `interrupt_before=["commit"]` (grey-band trust on a MEDICAL
    profile) is auto-approved, same as `eval.runners.system.run_system` does
    for golden cases — this is an offline calibration sweep with no human in
    the loop.

    `MealPlanningAgent`/`SyntheticCase` are imported only under
    `TYPE_CHECKING` above, so this module (pure statistics otherwise) does
    not force every caller of `expected_calibration_error`/
    `reliability_curve` to pull in langgraph and the synth package just to
    import this file.
    """
    from beatroot.contracts.result import Recommendation

    pairs: list[tuple[float, bool]] = []
    for case in cases:
        result = agent.run(case.constraint_set, query="a balanced meal")
        if result is None:
            thread_id = agent.last_thread_id
            if thread_id is not None:
                result = agent.resume(thread_id, approved=True)
        if isinstance(result, Recommendation):
            pairs.append((result.trust.composite, result.recipe_id in case.oracle_valid_ids))
    return pairs


def _print_report(pairs: list[tuple[float, bool]], offline: bool) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    ece = expected_calibration_error(pairs)
    curve = reliability_curve(pairs)

    console.print(f"COMMIT pairs collected: {len(pairs)}")
    console.print(
        f"Expected Calibration Error (ECE over {len(pairs)} COMMIT-only pairs): {ece:.4f}"
    )

    table = Table("bin", "mean confidence", "accuracy", "count")
    for b in curve:
        if b.count:
            table.add_row(
                f"[{b.lower:.2f}, {b.upper:.2f})",
                f"{b.mean_confidence:.3f}",
                f"{b.accuracy:.3f}",
                str(b.count),
            )
    console.print(table)

    if offline:
        console.print(
            "note: offline, model_self_assessment is a constant 0.5 stub "
            "(25% of the composite weight) — the only real variation running "
            "through trust.composite here is catalog_coverage + "
            "constraint_completeness. Measuring whether the MODEL's own "
            "confidence is calibrated (as opposed to the deterministic "
            "signals) requires a live model."
        )


def main() -> int:
    """`uv run python -m beatroot.eval.calibration`.

    Not yet wired into the `beatroot` CLI — out of scope for this batch; see
    the task report.
    """
    from beatroot.container import build_container
    from beatroot.eval.synth.profiles import generate_profiles
    from beatroot.settings import get_settings

    container = build_container()
    cases = generate_profiles(container.catalog, seed=0)
    pairs = collect_commit_pairs(container.agent, cases)
    _print_report(pairs, offline=get_settings().offline)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
