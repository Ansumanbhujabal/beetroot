"""Constraint vocabulary validation. Spec §12 adversarial family 5.

`exclude_tag` and `exclude_ingredient` are evaluated by pure membership
tests (`t0_invariants.constraints._exclude_tag` / `_exclude_ingredient`):
`"violated" if c.value in recipe.tags/ingredient_ids else "satisfied"`. A
value that names nothing in the ENTIRE catalog — not one recipe, not one
ingredient — is therefore never `in` anything, for any recipe, and comes
back "satisfied" every time. That is not the same thing as actually
checking it: a constraint we cannot verify is not a constraint we
satisfied, and silently passing one is worse than refusing outright — the
user believes their allergy was honoured.

This module closes that gap by validating every `exclude_tag`/
`exclude_ingredient` constraint's value against the catalog's OWN
vocabulary before feasibility ever runs, so an unverifiable profile costs
zero tokens — the same posture as the already-infeasible path. Severity is
irrelevant to the decision here (a PREFERENCE-only unknown must escalate
exactly like a MEDICAL one — see `unknown_vocabulary`'s docstring), only
to how the caller reports it.

Must never import `beatroot.reasoning` — the t0_invariants module rule,
enforced transitively by tests/test_boundaries.py.
"""

from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.trusted.canonical import build_synonym_index, canonicalise
from beatroot.trusted.catalog import Catalog


def _known_tags(catalog: Catalog) -> set[str]:
    """Every tag any recipe in the catalog actually carries — the union
    `trusted.tags.derive_recipe_tags` already computed at seed time, read
    back off `Recipe.tags` rather than recomputed here."""
    tags: set[str] = set()
    for recipe in catalog.recipes():
        tags |= recipe.tags
    return tags


def _resolves_to_known_ingredient(
    value: str, known_ids: set[str], synonym_index: dict[str, str]
) -> bool:
    """A constraint value is a known ingredient if it IS a canonical id, or
    canonicalises to one via the SAME synonym index ingredient resolution
    uses elsewhere (`trusted.canonical`). "groundnut oil" must resolve
    here exactly like it would anywhere else in the pipeline — treating it
    as unknown vocabulary would report a legitimate synonym as unverifiable
    and wrongly escalate the synonym_evasion family this project exists to
    get right.
    """
    if value in known_ids:
        return True
    return canonicalise(value, synonym_index) is not None


def unknown_vocabulary(cs: ConstraintSet, catalog: Catalog) -> list[Constraint]:
    """Every constraint whose `exclude_tag`/`exclude_ingredient` value names
    something the catalog has never heard of.

    Checks EVERY constraint regardless of severity — an unverifiable
    PREFERENCE constraint must not silently proceed any more than an
    unverifiable MEDICAL one would; that would just reintroduce the same
    silent-pass hole one severity down. Only `exclude_tag`/
    `exclude_ingredient` have a "vocabulary" to check against in the first
    place; every other constraint kind (`nutrient_range`, `budget_max`,
    `max_prep_minutes`, `cuisine_affinity`) is left untouched here — their
    own evaluators already return `"uncheckable"` rather than a silent pass
    for a value they cannot make sense of.

    A non-`str` value for either kind is filtered out, never flagged and
    never crashed on — the same "never guess, never crash" posture
    `t0_invariants.feasibility._survivors` takes for the identical case.
    """
    known_tags = _known_tags(catalog)
    ingredients = catalog.ingredients()
    known_ids = set(ingredients.keys())
    synonym_index = build_synonym_index(list(ingredients.values()))

    unknown: list[Constraint] = []
    for c in cs.constraints:
        if not isinstance(c.value, str):
            continue
        unknown_tag = c.kind == "exclude_tag" and c.value not in known_tags
        unknown_ingredient = c.kind == "exclude_ingredient" and not _resolves_to_known_ingredient(
            c.value, known_ids, synonym_index
        )
        if unknown_tag or unknown_ingredient:
            unknown.append(c)
    return unknown
