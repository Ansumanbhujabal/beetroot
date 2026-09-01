"""Deterministic nutrition arithmetic used to verify catalog-derived totals."""

from typing import Any

from beatroot.contracts.nutrition import NutritionFacts
from beatroot.trusted.catalog import Catalog

FIELDS = ("kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g")


class MalformedRecipeError(ValueError):
    """A recipe row that cannot be trusted to produce correct nutrition."""


def _usable(value: object) -> bool:
    """A field must be a real, finite, non-negative number to count.

    `f in per100` only tests key presence: `protein_g: null` passes it, earns
    full coverage credit, then crashes at float(None). Nutrition values are
    also never negative.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
        and value >= 0
    )


def _checked_grams(recipe: dict[str, Any], ref: dict[str, Any]) -> float:
    """Extract and validate grams from a recipe ingredient reference.

    Raises MalformedRecipeError if grams is negative or NaN.
    """
    grams = float(ref["grams"])
    if grams < 0 or grams != grams:  # negative or NaN
        raise MalformedRecipeError(
            f"{recipe.get('id', '<unknown>')}: ingredient {ref['ingredient_id']} has grams={grams}"
        )
    return grams


def compute(recipe: dict[str, Any], catalog: Catalog) -> NutritionFacts:
    """Arithmetic over the trusted catalog. No model involvement.

    This exists because model-generated nutrition values were observed running
    ~1.5x high in production. Coverage is MASS-weighted: a missing 1g spice
    should not cost the same confidence as a missing 200g protein. Spec §11.

    An ingredient counts as fully covered only if its per_100g carries every
    usable field in FIELDS. Usable means: a real, finite, non-negative number.
    Partial data contributes what it has but earns only proportional coverage
    (usable_fields / total_fields).
    """
    totals = dict.fromkeys(FIELDS, 0.0)
    total_mass = 0.0
    covered_mass = 0.0

    for ref in recipe["ingredients"]:
        grams = _checked_grams(recipe, ref)
        total_mass += grams
        payload = catalog.ingredient_payload(ref["ingredient_id"])
        if payload is None:
            continue
        per100 = payload.get("per_100g") or {}
        present = [f for f in FIELDS if _usable(per100.get(f))]
        if not present:
            continue  # no data at all: zero contribution, zero coverage
        # Partial data contributes what it has, but earns only proportional coverage.
        covered_mass += grams * (len(present) / len(FIELDS))
        factor = grams / 100.0
        for f in present:
            totals[f] += float(per100[f]) * factor

    coverage = (covered_mass / total_mass) if total_mass else 0.0
    return NutritionFacts(**{f: round(totals[f], 2) for f in FIELDS}, coverage=coverage)


def recipe_cost_inr(recipe: dict[str, Any], catalog: Catalog) -> float | None:
    """Sum cost in INR from per_100g_inr values, or None if any cost is missing.

    Returns None if the ingredient list is empty (unknown cost, not zero cost).
    """
    ingredients = recipe["ingredients"]
    if not ingredients:
        return None

    total = 0.0
    for ref in ingredients:
        grams = _checked_grams(recipe, ref)
        payload = catalog.ingredient_payload(ref["ingredient_id"])
        if payload is None or "cost_per_100g_inr" not in payload:
            return None
        cost = payload["cost_per_100g_inr"]
        if not _usable(cost):
            return None
        total += float(cost) * grams / 100.0
    return round(total, 2)
