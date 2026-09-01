"""Composite trust scoring. Spec §9.

Composite confidence is 75% deterministic (catalog coverage + constraint
completeness) and 25% model self-assessment. A confident model must never be
able to rescue an answer the trusted catalog does not support — that is the
entire point of the weighting, and every weight below is read from
`beatroot.settings`, never hardcoded here.
"""

import math
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from beatroot.contracts.core import ConstraintSet
from beatroot.contracts.nutrition import NutritionFacts
from beatroot.contracts.trust import TrustReport
from beatroot.settings import get_settings
from beatroot.t0_invariants.constraints import CheckResult


def score(
    nutrition: NutritionFacts,
    check: CheckResult,
    cs: ConstraintSet,
    model_self_assessment: float | None,
) -> TrustReport:
    """Composite confidence, 75% deterministic. A confident model cannot
    rescue an answer the catalog does not support. Spec §9."""
    cfg = get_settings().trust
    w = cfg.weights

    total = len(cs.constraints)
    # Completeness measures how many constraints were CONCLUSIVELY evaluated
    # (satisfied or violated), not how many merely aren't marked uncheckable —
    # a constraint that never made it into any bucket is exactly as
    # unevaluated as one explicitly marked uncheckable. Zero constraints
    # means there was nothing to leave unevaluated — completeness is 1.0,
    # never a division by zero.
    evaluated = len(check.satisfied) + len(check.violated)
    completeness = 1.0 if total == 0 else evaluated / total

    # A missing self-assessment is NEUTRAL, not confident. Defaulting high
    # here would let a silent model inflate its own trust score. This is a
    # distinct concept from weak_signal_floor (the veto threshold) even
    # though both are 0.5 today — tuning one must never silently retune
    # the other.
    if model_self_assessment is None:
        raw = cfg.neutral_model_default
    else:
        raw = float(model_self_assessment)
        if math.isnan(raw):
            # NaN is malformed model output, not "no opinion" and not
            # extreme confidence in either direction. min()/max() do not
            # tame it (comparisons against NaN are always False, so
            # min(1.0, max(0.0, nan)) is still nan) and it would otherwise
            # propagate into `composite`, which would then fail its own
            # Field(ge=0, le=1) validation. Treat it as neutral, same as a
            # missing assessment.
            raw = cfg.neutral_model_default
    # A model reporting 1.5 or -0.5 is malformed output, not high or negative
    # confidence. Absorb it rather than crashing the gate that exists to
    # contain exactly this kind of output.
    model = min(1.0, max(0.0, raw))

    composite = (
        w.catalog_coverage * nutrition.coverage
        + w.constraint_completeness * completeness
        + w.model_self_assessment * model
    )

    # failing_signal names the weakest DETERMINISTIC input, never the model
    # one — a confident model must never be allowed to look like the cause
    # of a refusal it didn't cause.
    deterministic = {
        "catalog_coverage": nutrition.coverage,
        "constraint_completeness": completeness,
    }
    weakest = min(deterministic, key=lambda k: deterministic[k])
    failing = weakest if deterministic[weakest] < cfg.weak_signal_floor else None

    return TrustReport(
        composite=round(min(1.0, max(0.0, composite)), 4),
        catalog_coverage=nutrition.coverage,
        constraint_completeness=completeness,
        model_self_assessment=model,
        failing_signal=failing,
    )


class TrustThresholds(BaseModel):
    refusal_threshold: float = Field(ge=0.0, le=1.0)


class VerifierThresholds(BaseModel):
    hard_constraint_violations: int = Field(ge=0, le=0)  # a COUNT, and only 0 is valid
    nutrition_drift_pct: float = Field(gt=0, lt=1)
    refusal_correctness: float = Field(ge=0, le=1)
    explanation_grounding: float = Field(ge=0, le=1)


class EvalThresholds(BaseModel):
    """Gates as a validated model. An unknown axis name is a load-time error,
    not a silently skipped gate. `extra="forbid"` is the point."""

    model_config = ConfigDict(extra="forbid")

    trust: TrustThresholds
    verifiers: VerifierThresholds
    axes: dict[str, float]
    performance: dict[str, float]
    regression: dict[str, int]
    # Per-family pass-rate floors for the adversarial simulation runner
    # (eval.runners.simulation). Optional — components/system callers that
    # never touch this field are unaffected — but declared here, not left
    # to `extra="ignore"`, for the identical reason `axes` is: an unmapped
    # family threshold should be a load-time error, not a silently skipped
    # gate.
    adversarial: dict[str, float] = {}

    @field_validator("axes", "adversarial")
    @classmethod
    def _axes_are_rates(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: x for k, x in v.items() if not 0 <= x <= 1}
        if bad:
            raise ValueError(f"axis thresholds must be in [0,1]: {bad}")
        return v


@lru_cache(maxsize=1)
def load_thresholds(path: Path | None = None) -> EvalThresholds:
    path = path or Path(__file__).parents[3] / "eval" / "thresholds.yaml"
    return EvalThresholds.model_validate(yaml.safe_load(path.read_text()))
