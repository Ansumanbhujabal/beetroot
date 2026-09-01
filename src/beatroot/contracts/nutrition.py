"""Nutrition contracts. NutritionFacts.provenance is always "computed" — never model output."""

from typing import Literal

from pydantic import BaseModel, Field


class NutritionFacts(BaseModel):
    """Always computed from the catalog. The provenance literal makes a
    model-generated instance unrepresentable. Spec §5."""

    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    sodium_mg: float
    fibre_g: float
    coverage: float = Field(ge=0.0, le=1.0)
    provenance: Literal["computed"] = "computed"
