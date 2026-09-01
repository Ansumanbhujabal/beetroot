"""Trust-gated escalation. Spec §9.

Below the refusal threshold, beatroot refuses — and the refusal names WHICH
signal failed. A generic apology teaches the user nothing and teaches the
healing loop (Task 16) nothing.
"""

from beatroot.contracts.result import Escalation
from beatroot.contracts.trust import TrustReport
from beatroot.settings import Settings, get_settings


def gate(trust: TrustReport, settings: Settings | None = None) -> Escalation | None:
    """Below threshold, refuse and say WHICH signal failed.

    The gate is inclusive at the boundary: a composite exactly equal to the
    threshold passes; a hair below escalates.

    A weak-signal veto sits alongside the numeric threshold, not instead of
    it: `constraint_completeness` (0.30) and `model_self_assessment` (0.25)
    alone sum to exactly the default refusal threshold (0.55), so a
    fully-checked constraint set plus a maximally confident model can carry
    a composite to the threshold *no matter how thin catalog coverage is* —
    a confident model rescuing a catalog the data does not support. `score()`
    already flags this: `trust.failing_signal` is set whenever a
    deterministic axis (catalog coverage or constraint completeness) falls
    below `weak_signal_floor`, regardless of what the weighted composite
    lands on. The gate honours that flag as an escalation trigger in its own
    right, not merely as a label attached after the numeric check.
    """
    limit = (settings or get_settings()).trust.refusal_threshold
    if trust.composite >= limit and trust.failing_signal is None:
        return None
    return Escalation(
        reason="low_trust",
        failing_signal=trust.failing_signal or "composite",
        trust=trust,
        detail=(
            f"Composite trust {trust.composite:.2f} is below the "
            f"{limit:.2f} threshold; weakest signal is "
            f"{trust.failing_signal or 'composite'}."
        ),
    )
