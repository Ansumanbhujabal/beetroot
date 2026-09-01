"""Core contract types shared across the pipeline: identifiers, enums, and constraint kinds."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

ConstraintValue = str | float | tuple[float, float] | list[str]

ConstraintKind = Literal[
    "exclude_tag",
    "require_tag",
    "require_any_tag",
    "exclude_ingredient",
    "nutrient_range",
    "budget_max",
    "cuisine_affinity",
    "exclude_cuisine",
    "max_prep_minutes",
]


class Severity(StrEnum):
    MEDICAL = "medical"  # allergy or condition — never relaxable
    RELIGIOUS = "religious"  # never relaxable
    DIETARY = "dietary"  # vegan/vegetarian identity — never relaxable
    GOAL = "goal"  # protein floor, calorie target — relaxable
    PREFERENCE = "preference"  # dislikes, budget — relaxable


# DIETARY exists because its absence caused a real incident: a user on the
# Vegan preset was served chicken. The taxonomy had only MEDICAL and RELIGIOUS
# as hard tiers, and an ethical vegan is neither — so veganism was encoded as
# PREFERENCE, which `is_legal()` does not enforce (soft constraints are
# ranking's job, not filtering's, by design). The engine behaved exactly as
# specified; the vocabulary had nowhere honest to put the requirement.
#
# The inconsistency this removes: `jain` was RELIGIOUS and therefore enforced,
# so a Jain user never saw root vegetables, while a vegan user saw chicken.
# Same class of categorical dietary rule, opposite outcomes, for no principled
# reason beyond which tier happened to fit.
HARD_SEVERITIES = frozenset({Severity.MEDICAL, Severity.RELIGIOUS, Severity.DIETARY})


class Constraint(BaseModel):
    id: str
    kind: ConstraintKind
    severity: Severity
    value: ConstraintValue
    source: Literal["structured", "parsed_free_text"] = "structured"
    nutrient: str | None = None

    @property
    def is_hard(self) -> bool:
        return self.severity in HARD_SEVERITIES


class ConstraintSet(BaseModel):
    profile_id: str
    constraints: list[Constraint]

    def hard(self) -> list[Constraint]:
        return [c for c in self.constraints if c.is_hard]

    def soft(self) -> list[Constraint]:
        return [c for c in self.constraints if not c.is_hard]

    def fingerprint(self) -> str:
        """Stable hash for feasibility caching. Spec §17."""
        import hashlib

        payload = "|".join(sorted(f"{c.kind}:{c.severity}:{c.value}" for c in self.constraints))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
