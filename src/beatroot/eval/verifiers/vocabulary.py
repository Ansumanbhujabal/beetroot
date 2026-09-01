"""Independent re-derivation of "unverifiable constraint vocabulary" for the
system eval oracle. Spec §12.

`eval.runners.system._oracle_has_valid_meal` decides the A5
(`escalation_correctness`) verdict, and it used to do so by calling
`t0_invariants.vocabulary.unknown_vocabulary` — production code, the exact
function `agent.nodes.feasibility` calls to decide whether a profile's
vocabulary is checkable at all — directly. A bug in that predicate would
move the agent's actual answer AND the oracle's answer together, so A5
could keep reading 1.000 with nothing left in the eval to disagree with
it. That is the same tautology `test_oracle_cross_checks_against_
check_recipe` (`tests/eval/test_oracle.py`) already found and fixed once
for `check_recipe`, reintroduced one axis over.

This module is a from-scratch reimplementation: it reads catalog tags,
ingredient ids, and synonyms directly off `Catalog`, and imports nothing
from `t0_invariants`. Independence here is at the CALL-GRAPH level, same
posture as `eval.verifiers.hard_constraint` and `eval.synth.profiles`'s
oracle — see their module docstrings for what that does and does not
buy: a bug in `t0_invariants.vocabulary.unknown_vocabulary`'s own
comparison logic is caught here; a bug shared upstream (`Recipe.tags` as
derived by `trusted.tags.derive_recipe_tags`, or `trusted.canonical`'s
synonym-index construction) is invisible to this module exactly as it is
invisible to the thing it checks, because both read the same catalog-level
primitives. `tests/eval/test_vocabulary_oracle.py` cross-checks this
against production and proves, by monkeypatching production broken, that
the cross-check can actually fail.
"""

from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.trusted.canonical import build_synonym_index, canonicalise
from beatroot.trusted.catalog import Catalog


def _known_tags(catalog: Catalog) -> set[str]:
    """Every tag any recipe in the catalog actually carries."""
    tags: set[str] = set()
    for recipe in catalog.recipes():
        tags |= recipe.tags
    return tags


def _resolves_to_known_ingredient(
    value: str, known_ids: set[str], synonym_index: dict[str, str]
) -> bool:
    """A constraint value is a known ingredient if it IS a canonical id, or
    canonicalises to one via the catalog's own synonym index."""
    if value in known_ids:
        return True
    return canonicalise(value, synonym_index) is not None


def unknown_vocabulary(cs: ConstraintSet, catalog: Catalog) -> list[Constraint]:
    """Every constraint whose `exclude_tag`/`exclude_ingredient` value names
    something the catalog has never heard of — independently derived, for
    the eval oracle, of `t0_invariants.vocabulary.unknown_vocabulary`.

    Semantics deliberately mirror the production predicate (checks every
    constraint regardless of severity; only `exclude_tag`/
    `exclude_ingredient` have a vocabulary to check; a non-`str` value is
    filtered out, never flagged) — see that module's docstring for the
    reasoning. What is NOT shared is the implementation: this reads catalog
    data straight from `Catalog`, never calling into `t0_invariants`.
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

# Fields a recipe's computed nutrition can actually supply. Written out here
# rather than imported from `contracts.nutrition` so this oracle stays
# independent of the code it judges — the same discipline the rest of this
# module follows.
_KNOWN_NUTRIENTS = frozenset(
    {"kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g"}
)


def uncheckable_constraints(cs: ConstraintSet) -> list[Constraint]:
    """Constraints that are vocabulary-valid but still cannot be evaluated.

    `unknown_vocabulary` covers tags and ingredient ids. A `nutrient_range`
    naming a nutrient the catalog does not track slips past it: the value is
    a perfectly ordinary string, so nothing reports it unknown, yet no recipe
    can be checked against it.

    The consequence is the same one this module exists to refuse. An
    unevaluable constraint is never *violated* by any recipe, so a
    violation-scan will happily "find" a safe meal and certify a profile it
    never actually checked. No meal can be proven safe against a constraint
    that cannot be checked, so these make the oracle answer "no" exactly as
    an unknown tag does.
    """
    out: list[Constraint] = []
    for c in cs.constraints:
        if c.kind == "nutrient_range" and (
            c.nutrient is None or c.nutrient not in _KNOWN_NUTRIENTS
        ):
            out.append(c)
    return out
