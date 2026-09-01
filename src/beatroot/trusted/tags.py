"""Tag vocabulary and transitive-tag derivation for the trusted catalog."""

from typing import Any


class UnknownIngredientError(KeyError):
    """Raised when a recipe references an ingredient absent from the catalog.
    Never swallowed — it must reach ESCALATE. Spec §12 adversarial family 5."""


def derive_recipe_tags(recipe: dict[str, Any], ingredients: dict[str, dict[str, Any]]) -> set[str]:
    """Recipe tags are derived, never hand-set.

    Allergen and religious tags are a UNION over ingredients: one peanut-bearing
    ingredient makes the whole dish peanut-bearing. Dietary tags are an
    INTERSECTION: a dish is vegan only if every ingredient is. Spec §11.
    """
    refs = recipe["ingredients"]
    if not refs:
        return set()

    union: set[str] = set()
    dietary_sets: list[set[str]] = []

    for ref in refs:
        iid = ref["ingredient_id"]
        try:
            ing = ingredients[iid]
        except KeyError as exc:
            raise UnknownIngredientError(iid) from exc
        union |= set(ing.get("allergen_tags", ()))
        union |= set(ing.get("religious_tags", ()))
        dietary_sets.append(set(ing.get("dietary_tags", ())))

    return union | set.intersection(*dietary_sets)
