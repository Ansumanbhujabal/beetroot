"""Contracts for recommendation results: recipes, check results, and terminal outcomes."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from beatroot.contracts.nutrition import NutritionFacts
from beatroot.contracts.trust import CostRecord, TrustReport


class Relaxation(BaseModel):
    constraint_ids: list[str]
    description: str
    unlocks: int
    severity: str


class Negotiation(BaseModel):
    """Terminal state NEGOTIATE. Spec §8."""

    total_candidates: int
    surviving: int = 0
    relaxations: list[Relaxation]
    locked: list[str]
    cost: CostRecord = Field(default_factory=CostRecord)


class Escalation(BaseModel):
    """Terminal state ESCALATE. Spec §9."""

    reason: Literal[
        "low_trust",
        "unknown_ingredient",
        "verification_failed",
        "constraint_uncheckable",
        # FREE-TEXT/NLP task: the request itself has no meal-planning
        # content (`agent.nodes.compile_node`'s scope check) — distinct
        # from every other reason above, which all describe a REAL meal
        # request the system could not safely satisfy. This one describes
        # a request that was never a meal request at all.
        "out_of_scope",
    ]
    failing_signal: str
    trust: TrustReport | None = None
    detail: str
    cost: CostRecord = Field(default_factory=CostRecord)


class RecipeIngredient(BaseModel):
    """One line of a recipe's ingredient list, as a diner would read it.

    Carried on the recommendation because a meal recommendation that will not
    tell you what is IN the meal is not usable — and because the whole safety
    argument here rests on ingredients, so showing them lets a reader check
    the system's claim rather than take it on trust.
    """

    ingredient_id: str
    name: str
    grams: float | None = None


class Recommendation(BaseModel):
    """Terminal state COMMIT. Spec §5."""

    recipe_id: str
    recipe_name: str
    nutrition: NutritionFacts
    trust: TrustReport
    explanation: str
    constraints_satisfied: list[str]
    # The same ids as `constraints_satisfied`, rendered as language. The ids
    # are kept because the audit trail and the eval suite both key on them;
    # this is what a human should ever be shown.
    constraints_satisfied_display: list[str] = Field(default_factory=list)
    ingredients: list[RecipeIngredient] = Field(default_factory=list)
    skill_versions: dict[str, str] = Field(default_factory=dict)
    cost: CostRecord = Field(default_factory=CostRecord)


class Incident(BaseModel):
    """Emitted by every escalation, refusal, drift detection, infeasibility."""

    id: str
    kind: Literal[
        "escalation",
        "drift",
        "infeasible",
        "verification_failure",
        "unknown_ingredient",
    ]
    profile_id: str
    detail: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
