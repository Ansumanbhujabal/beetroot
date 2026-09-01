"""Trust score contracts: the composite signal weights and the trust tier they resolve to."""

from typing import Any

from pydantic import BaseModel, Field

# The trust weights are NOT defined here. They live in
# `settings.TrustWeights` (values in `config/beatroot.yaml`), which
# validates that they sum to 1 at load time.
#
# Three module-level constants — W_CATALOG_COVERAGE = 0.45,
# W_CONSTRAINT_COMPLETENESS = 0.30, W_MODEL_SELF_ASSESSMENT = 0.25 — used to
# sit here and were imported by nothing. That is worse than redundant: a
# reader editing the composite weighting would naturally edit the constants
# named after it, change no behaviour at all, and have no failing test to
# tell them so. Spec §9's "deterministic signals carry 0.75 of the
# composite" is enforced by the config value, not by a literal here.


class CostRecord(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    tokens_saved: int = 0
    per_stage: dict[str, float] = Field(default_factory=dict)


class Completion(BaseModel):
    text: str
    parsed: dict[str, Any] | None = None
    self_assessment: float | None = None
    cost: CostRecord = Field(default_factory=CostRecord)


class TrustReport(BaseModel):
    composite: float = Field(ge=0.0, le=1.0)
    catalog_coverage: float = Field(ge=0.0, le=1.0)
    constraint_completeness: float = Field(ge=0.0, le=1.0)
    model_self_assessment: float = Field(ge=0.0, le=1.0)
    failing_signal: str | None = None
