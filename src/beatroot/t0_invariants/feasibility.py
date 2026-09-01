"""Feature 1: the Infeasibility Negotiator. Spec §8, §17.

Constraint satisfaction is a set-intersection problem, not a scan problem —
but only for the part of it a tag bitmap can express. Be precise about what
that buys:

- Tag constraints (`exclude_tag`) cost NOTHING per catalog item. `TagIndex`
  is an inverted index of `tag -> bitmap over recipe positions`, built once.
  Excluding a tag is one bitwise AND-NOT; walking the survivors afterward is
  O(popcount) via `TagIndex.iter_ids` (`mask & -mask` isolates each set bit
  in turn), not O(catalog). This part really is independent of catalog size.
- Non-tag constraints (`nutrient_range`, `budget_max`, `max_prep_minutes`,
  `exclude_ingredient`) cannot be expressed as a tag bitmap, so they cost one
  `check_recipe` call per SURVIVOR of the tag mask — O(survivors x non-tag
  constraints), not O(catalog x constraints). That is still a large win
  whenever the tag mask has already cut the candidate set down, but it is not
  free, and this module does not claim it is.

The relaxation lattice walk reuses the same bitmap trick: dropping a
constraint is un-ANDing one bitmap, never rescanning the catalog for the tag
portion of the check.

This module is under t0_invariants/ and must NEVER import `beatroot.reasoning`
— tests/test_boundaries.py polices this transitively as well as directly.
"""

from dataclasses import dataclass
from itertools import combinations

from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Negotiation, Relaxation
from beatroot.t0_invariants.constraints import check_recipe
from beatroot.trusted.catalog import Recipe
from beatroot.trusted.index import TagIndex

# The DEFAULT subset size, not the configured one. Callers that hold
# settings pass `settings.feasibility.max_relaxation_subset_size` in
# explicitly; this constant is the fallback for direct callers (tests, the
# eval oracle) that have no settings object.
#
# Deliberately NOT read from `beatroot.settings` here. Nothing under
# `t0_invariants` imports settings, and that purity is the point: this is
# the deterministic layer whose answers must be reproducible from its
# arguments alone. Reaching for global config inside it would make an
# invariant's result depend on ambient process state — exactly what the
# layer exists to rule out. The config value therefore travels DOWN as a
# parameter rather than being pulled UP from a global.
DEFAULT_MAX_RELAXATION_SUBSET_SIZE = 2


@dataclass
class Feasibility:
    feasible: bool
    surviving: list[Recipe]
    negotiation: Negotiation | None = None


def _survivors(
    recipes: list[Recipe], cs: ConstraintSet, index: TagIndex | None = None
) -> list[Recipe]:
    """Filter `recipes` down to the ones that satisfy every constraint in `cs`.

    With an `index`: the tag mask does an O(1)-per-tag AND-NOT over the whole
    catalog, then `iter_ids` walks only the surviving bits (O(popcount)) to
    resolve them back to recipe objects via `index.recipe` (O(1) each). Only
    those survivors — never the full catalog — go through the residual
    `check_recipe` pass for the constraint kinds a tag bitmap cannot express
    (nutrient ranges, budget, prep time). At scale those become additional
    precomputed bucket indices; the residual set is already small because the
    tag mask ran first. Spec §17.

    With `index=None` this degrades to a plain O(catalog) scan — the
    reference implementation the bitmask path is checked against (see
    test_bitmask_path_agrees_with_scan_path / test_assess_agrees_with_and_without_index).
    """
    if index is None:
        return [r for r in recipes if check_recipe(r, cs).ok]

    # `exclude_tag` constraints always carry a str tag; filtering to that
    # here (rather than trusting the union type `Constraint.value` allows)
    # keeps this the same "never guess" posture `_exclude_tag` itself takes
    # — a malformed non-str value simply excludes nothing, never crashes.
    excluded = [
        c.value for c in cs.constraints if c.kind == "exclude_tag" and isinstance(c.value, str)
    ]
    mask = index.exclude_tags(index.all_mask(), excluded)
    if mask == 0:
        return []
    candidates = (index.recipe(rid) for rid in index.iter_ids(mask))
    return [r for r in candidates if r is not None and check_recipe(r, cs).ok]


def _without(cs: ConstraintSet, drop: set[str]) -> ConstraintSet:
    return ConstraintSet(
        profile_id=cs.profile_id,
        constraints=[c for c in cs.constraints if c.id not in drop],
    )


def _plural(n: float, word: str) -> str:
    return word if n == 1 else f"{word}s"


def _describe(c: Constraint) -> str:
    """Describes the RELAXATION, not the constraint.

    Every numeric branch must read as "go beyond this limit". A relaxation
    DROPS the constraint, so the offer is always "go beyond this boundary",
    never "set it to this boundary" — the number is the limit being lifted,
    not a new target. Phrasing it as a new target ("raise budget TO 160")
    falsely implies the limit becomes that exact number."""
    match c.kind:
        case "exclude_tag" | "exclude_ingredient":
            return f"allow {c.value}"
        case "max_prep_minutes" if isinstance(c.value, int | float) and not isinstance(
            c.value, bool
        ):
            value = float(c.value)
            return f"allow more than {value:g} {_plural(value, 'minute')} of prep"
        case "budget_max" if isinstance(c.value, int | float) and not isinstance(c.value, bool):
            return f"raise your budget above ₹{float(c.value):g}"
        case "nutrient_range" if isinstance(c.value, tuple) and len(c.value) == 2:
            lo, hi = c.value
            nutrient = c.nutrient or "nutrient"
            return f"widen the {nutrient} range beyond {float(lo):g}-{float(hi):g}"
    # Falls through here for an unrecognised kind, or a malformed value that
    # doesn't match the expected shape for its kind (never a crash).
    return f"relax {c.id}"


def rank_relaxations(
    cs: ConstraintSet,
    recipes: list[Recipe],
    index: TagIndex | None = None,
    max_subset_size: int = DEFAULT_MAX_RELAXATION_SUBSET_SIZE,
) -> list[Relaxation]:
    """Walk the constraint lattice: singles, then pairs. Hard constraints
    (MEDICAL, RELIGIOUS) are never candidates for relaxation — that is the
    liability boundary. Spec §8.

    O(n + n^2) subset evaluations over ~100 recipes. Triples are deliberately
    not explored by default; if nothing up to `max_subset_size` constraints
    helps, the caller reports that and recommends a profile review.
    """
    relaxable = cs.soft()
    out: list[Relaxation] = []

    for c in relaxable:
        n = len(_survivors(recipes, _without(cs, {c.id}), index))
        if n > 0:
            out.append(
                Relaxation(
                    constraint_ids=[c.id],
                    description=_describe(c),
                    unlocks=n,
                    severity=str(c.severity),
                )
            )

    if not out and max_subset_size >= 2:
        for a, b in combinations(relaxable, 2):
            n = len(_survivors(recipes, _without(cs, {a.id, b.id}), index))
            if n > 0:
                out.append(
                    Relaxation(
                        constraint_ids=[a.id, b.id],
                        description=f"{_describe(a)} and {_describe(b)}",
                        unlocks=n,
                        severity=f"{a.severity}+{b.severity}",
                    )
                )

    return sorted(out, key=lambda r: (-r.unlocks, r.constraint_ids))


def assess(
    cs: ConstraintSet,
    recipes: list[Recipe],
    index: TagIndex | None = None,
    max_subset_size: int = DEFAULT_MAX_RELAXATION_SUBSET_SIZE,
) -> Feasibility:
    surviving = _survivors(recipes, cs, index)
    if surviving:
        return Feasibility(feasible=True, surviving=surviving)

    return Feasibility(
        feasible=False,
        surviving=[],
        negotiation=Negotiation(
            total_candidates=len(recipes),
            surviving=0,
            relaxations=rank_relaxations(cs, recipes, index, max_subset_size),
            locked=[c.id for c in cs.hard()],
        ),
    )
