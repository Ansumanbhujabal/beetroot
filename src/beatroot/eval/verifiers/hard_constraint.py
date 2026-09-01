"""Independent re-derivation of hard-constraint safety. Spec §12.

Deliberately does NOT call `t0_invariants.constraints.is_legal()` — a
verifier that shares an implementation with the thing it verifies proves
nothing. This module re-reads the recipe's own tags and ingredient ids
directly, so a bug in `is_legal()`'s DISPATCH or COMPARISON logic (an
inverted `_exclude_tag`, a mis-registered evaluator, a wrong bucket) is
caught here even though it slipped past `is_legal()` itself.

That independence is at the CALL-GRAPH level, not the assumption level, and
this module should not be read as claiming more than that: `verify()` below
reads `recipe.tags`/`recipe.ingredient_ids`, the exact same catalog-derived
fields `is_legal()`'s `_exclude_tag`/`_exclude_ingredient` read. A bug
upstream of both — in `trusted.tags.derive_recipe_tags` failing to derive a
transitive allergen tag, say — produces the same wrong `recipe.tags` both
functions see, and is invisible to this verifier exactly as it is invisible
to the thing it verifies. `eval.synth.profiles._oracle_valid_ids` carries
the identical caveat for the same reason.

UNHASHABLE `exclude_tag` VALUES: `Constraint.value`'s schema legally
admits `list[str]`, and `recipe.tags` is a `set[str]` — so `c.value in
recipe.tags` raised `TypeError: unhashable type: 'list'` and took an
entire eval run down rather than failing one case. Reachable, not
hypothetical. The fix direction matters more than the crash: the
production path returns False for that input, treating an exclusion it
cannot parse as violated, so this verifier fails closed to match. Merely
SKIPPING the constraint would have stopped the crash while making the
oracle silently disagree with production — an oracle calling an unsafe
recipe safe, which is far the worse of the two bugs.

CAUTION, on record rather than left for a reader to rediscover: this
module's `exclude_ingredient` branch and `t0_invariants.constraints.
_exclude_ingredient` BOTH shipped comparing `c.value` against
`recipe.ingredient_ids` LITERALLY — `c.value` is user-facing text (often a
synonym: "groundnut oil" for `ing_peanut_oil`), `recipe.ingredient_ids` is
catalog canonical ids, and the two never matched, so a MEDICAL exclusion
phrased as a real, catalog-known synonym silently never fired in either
implementation. Two call-graph-independent modules, written deliberately
not to share an implementation, both had the identical hole — because the
omission was conceptual (nobody canonicalised the constraint value at
enforcement time), not a coding slip either implementation could have
caught on its own. Independence at the call-graph level defends against a
DISPATCH or COMPARISON bug local to one implementation; it does nothing
for a gap in the shared MENTAL MODEL both implementations were written
from. The fix below canonicalises `c.value` via `trusted.canonical.
resolve_ingredient_id` — the same primitive `t0_invariants.constraints.
_exclude_ingredient` now uses — before comparing; sharing that low-level
resolution utility is not a reintroduction of shared logic, since neither
module's actual VERIFY/EVALUATE decision calls into the other.
"""

from beatroot.contracts.core import ConstraintSet
from beatroot.trusted.canonical import resolve_ingredient_id
from beatroot.trusted.catalog import Recipe


def verify(recipe: Recipe, cs: ConstraintSet) -> list[str]:
    """Return the ids of every HARD constraint `recipe` violates.

    An empty list means the recipe is safe under this independent check.
    Only `exclude_tag` and `exclude_ingredient` are checked — the two kinds
    that express a hard safety exclusion (medical/religious). Any other hard
    constraint kind (e.g. a malformed `nutrient_range` marked medical) is
    outside what a tag/ingredient re-derivation can prove one way or the
    other, so it is not flagged here.

    `exclude_ingredient` canonicalises `c.value` to an ingredient id before
    comparing — see the module docstring's CAUTION. A value that resolves
    to nothing is never counted as violated here (this function has no
    "uncheckable" bucket to put it in; callers that need that distinction
    use `t0_invariants.vocabulary.unknown_vocabulary` upstream, which is
    exactly what `eval.runners.system._oracle_has_valid_meal` already
    consults before ever trusting this function's silence about such a
    value).
    """
    violated: list[str] = []
    for c in cs.hard():
        # Unhashable `c.value` fails CLOSED here, not skipped — see the
        # module docstring's UNHASHABLE note for why that direction is the
        # only safe one.
        if c.kind == "exclude_tag":
            tag_violation = not isinstance(c.value, str) or c.value in recipe.tags
        else:
            tag_violation = False
        ingredient_violation = False
        if c.kind == "exclude_ingredient" and isinstance(c.value, str):
            resolved = resolve_ingredient_id(c.value)
            ingredient_violation = resolved is not None and resolved in recipe.ingredient_ids
        if tag_violation or ingredient_violation:
            violated.append(c.id)
    return violated
