"""Synthetic profile generation with a FREE, exact oracle. Spec §12.

Because constraints are typed (`Constraint`/`ConstraintSet`) and the catalog
is finite, the exact "which recipes are valid under this profile" answer is
computable by brute-force enumeration over the catalog. That is the entire
oracle — no human labelling, no LLM-as-judge adjudicating facts it cannot
verify.

The enumeration below is deliberately a SEPARATE, from-scratch implementation
of constraint semantics (`_oracle_violates`), not a second call into
`t0_invariants.constraints.check_recipe`. An oracle that calls the function
it exists to validate proves that function is deterministic, not that it is
correct — an inverted comparison in `_exclude_tag` would satisfy both
computations identically, and a test built on "recompute the same call and
compare" would stay green forever. `tests/eval/test_oracle.py` cross-checks
this oracle against `check_recipe` (two independent implementations agreeing
is real evidence; one implementation agreeing with itself is not) and
includes a test that deliberately breaks an evaluator to prove the
cross-check can actually fail.

`_oracle_violates`'s `exclude_ingredient` branch DOES canonicalise
`c.value` via `trusted.canonical.resolve_ingredient_id` — the same
low-level primitive `t0_invariants.constraints._exclude_ingredient` and
`eval.verifiers.hard_constraint.verify` use, not a shared EVALUATOR. This
is deliberate and worth stating plainly: this exact conceptual gap — a
`Constraint.value` that is a real catalog synonym ("groundnut oil" for
`ing_peanut_oil`) compared LITERALLY against `Recipe.ingredient_ids`
instead of being resolved to a canonical id first — shipped independently
in FOUR places at once: `_exclude_ingredient`, `hard_constraint.verify`,
this oracle's `_oracle_violates`, and the golden `synonym_evasion` family
itself (named for exactly the case it never actually tested — every case
in it put the synonym in the free-text query, never in the constraint
value). All four were written deliberately not to share code. Independence
of implementation defends against a DISPATCH or COMPARISON bug local to
one of them; it does nothing for a wrong assumption everyone who wrote
these modules held at once. Only running the system against the real
catalog surfaced it. This oracle is the ground truth every component
metric is measured against — a latent disagreement here is worse than the
same bug anywhere else, because nothing would ever catch it failing; it
would just quietly become the wrong answer everything else is graded
against, including a future `exclude_ingredient` addition to
`_random_constraints`/`_force_infeasible_constraints`, which today never
generates this kind at all.

Two things this oracle is NOT independent of, stated plainly rather than
left for a reader to assume otherwise:

- It reads `Recipe.tags`/`Recipe.ingredient_ids`, the same catalog-derived
  fields `t0_invariants.constraints._exclude_tag`/`_exclude_ingredient` read.
  That makes it independent of check_recipe's DISPATCH and EVALUATOR logic
  (wrong registry entry, an inverted comparison, a misfiled outcome bucket)
  but not of the upstream tag-derivation step itself
  (`trusted.tags.derive_recipe_tags`) — a catalog-level bug there (a missing
  transitively-derived allergen tag, say) would be invisible to this oracle
  exactly as it would be invisible to `check_recipe`. See
  `eval.verifiers.hard_constraint`'s docstring for the identical caveat on
  that independent verifier.
- For `nutrient_range`/`budget_max` it reads `Recipe.nutrition`/`cost_inr`
  as populated by `Catalog.hydrate()`, i.e. `t0_invariants.nutrition_math`.
  Re-deriving gram-weighted nutrient arithmetic a second time from scratch
  here would not buy independence — it would just be a second
  implementation of the same summation, with its own chance of a different
  bug — so this oracle deliberately trusts that module for arithmetic and
  only re-implements the CONSTRAINT semantics built on top of it (the
  comparison against the requested range/limit).

That property — ground truth computed, not labelled or judged — is what
lets this eval suite be BOTH large (hundreds of profiles, generated in
milliseconds, reproducibly from a seed) and rigorous.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.settings import SynthConfig, get_settings
from beatroot.trusted.canonical import resolve_ingredient_id
from beatroot.trusted.catalog import Catalog, Recipe

# A generated constraint draws from every severity — a synthetic profile
# should exercise the hard (never-relaxable) and soft (negotiable) paths the
# same way a real profile does. `max_prep_minutes` is restricted to the two
# SOFT severities: a prep-time ceiling is a preference/goal in this domain,
# never the kind of medical/religious exclusion `exclude_tag` models.
_SEVERITY_POOL: tuple[Severity, ...] = (
    Severity.MEDICAL,
    Severity.RELIGIOUS,
    Severity.GOAL,
    Severity.PREFERENCE,
)
_SOFT_SEVERITY_POOL: tuple[Severity, ...] = (Severity.GOAL, Severity.PREFERENCE)

# Upper bound on attempts `_force_infeasible_constraints` will make before
# falling back to its last-tried candidate. Rejection sampling over a handful
# of small, plausible constraint templates against a 100-recipe catalog
# converges quickly in practice (see the fix-round section of
# task-14-report.md for measured attempt counts); this cap just stops an
# unbounded loop on a catalog where no genuine small conflict exists.
_MAX_INFEASIBILITY_ATTEMPTS = 300

# A profile that excludes most of the catalog's tag vocabulary at once is
# infeasible for a trivial, degenerate reason ("nothing is allowed") rather
# than because a FEW plausible constraints happen to conflict — the specific
# shortcut this module exists to avoid. No candidate `_force_infeasible_
# constraints` accepts may exclude more than this fraction of all known
# tags; `test_forced_infeasible_profiles_stay_within_the_tag_exclusion_cap`
# enforces this so the shortcut cannot quietly creep back in.
_MAX_TAG_EXCLUSION_FRACTION = 0.4


@dataclass
class SyntheticCase:
    """One generated profile, its constraints, and the EXACT set of recipe
    ids that satisfy every one of them — computed once, at generation time,
    by the independent oracle below (`_oracle_valid_ids`), not by calling
    the code under test."""

    id: str
    constraint_set: ConstraintSet
    oracle_valid_ids: set[str]


def _numeric(value: object) -> float | None:
    """Independent numeric coercion. Deliberately re-implemented rather than
    imported from `t0_invariants.constraints._as_number`, even though the
    two are trivially equivalent — a shared numeric-coercion helper is still
    a shared implementation, and this oracle's whole point is having none."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _oracle_violates(recipe: Recipe, c: Constraint) -> bool:
    """True only if `c` is DEFINITELY violated by `recipe`.

    Mirrors `check_recipe`'s own `ok = not violated` semantics: a constraint
    this function cannot evaluate (missing data, a malformed value, an
    unknown nutrient name) is never counted as a violation — "uncheckable"
    and "satisfied" both mean "does not block this recipe", same as they do
    in `t0_invariants.constraints.CheckResult.ok`. Every branch below is
    written from scratch against the constraint kind's plain-English
    definition, not derived from or delegating to the registered evaluator
    in `t0_invariants.constraints`.
    """
    if c.kind == "exclude_tag":
        return isinstance(c.value, str) and c.value in recipe.tags
    if c.kind == "exclude_ingredient":
        # `c.value` may be a catalog SYNONYM ("groundnut oil"), not the
        # canonical id `recipe.ingredient_ids` holds — resolve it first,
        # via the same primitive `t0_invariants.constraints.
        # _exclude_ingredient` and `hard_constraint.verify` use. A value
        # that resolves to nothing (or isn't a string at all) is never
        # counted as a violation here, mirroring "uncheckable is not a
        # violation" the same way every other branch below does for
        # missing/malformed data.
        if not isinstance(c.value, str):
            return False
        resolved = resolve_ingredient_id(c.value)
        return resolved is not None and resolved in recipe.ingredient_ids
    if c.kind == "max_prep_minutes":
        limit = _numeric(c.value)
        if recipe.prep_minutes is None or limit is None:
            return False
        return recipe.prep_minutes > limit
    if c.kind == "budget_max":
        limit = _numeric(c.value)
        if recipe.cost_inr is None or limit is None:
            return False
        return recipe.cost_inr > limit
    if c.kind == "nutrient_range":
        if recipe.nutrition is None or c.nutrient is None:
            return False
        if not isinstance(c.value, tuple) or len(c.value) != 2:
            return False
        lo, hi = c.value
        value = getattr(recipe.nutrition, c.nutrient, None)
        if not isinstance(value, int | float) or isinstance(value, bool):
            return False
        return not (lo <= value <= hi)
    # cuisine_affinity is a ranking signal, never an enforcement gate — and
    # any kind this oracle has never heard of is, like check_recipe's own
    # unregistered-kind path, uncheckable rather than a silent violation.
    return False


def _oracle_valid_ids(
    catalog: Catalog, cs: ConstraintSet, recipes: list[Recipe] | None = None
) -> set[str]:
    """Ground truth derived from catalog DATA, independently of the code
    under test. See the module docstring for exactly what "independently"
    does and does not cover.

    Kept intentionally naive and slow — one Python loop per recipe, one
    per constraint, no bitmask, no index — clarity over speed, since its
    only job is to be trustworthy enough to check the fast path against.

    `recipes`, when given, must already be `catalog.hydrate()`d (see
    `_hydrated_recipes`) — `_force_infeasible_constraints` calls this
    function up to `_MAX_INFEASIBILITY_ATTEMPTS` times per forced profile,
    and `Catalog.hydrate()` recomputes nutrition/cost arithmetic from
    scratch on every call rather than caching it, so re-hydrating fresh
    inside a rejection-sampling loop would redo that arithmetic for the
    entire catalog on every rejected attempt for no reason. Passing it in
    once is a performance detail only — it changes nothing about what gets
    checked or how.
    """
    hydrated = recipes if recipes is not None else _hydrated_recipes(catalog)
    return {r.id for r in hydrated if not any(_oracle_violates(r, c) for c in cs.constraints)}


def _hydrated_recipes(catalog: Catalog) -> list[Recipe]:
    return [catalog.hydrate(r) for r in catalog.recipes()]


def _percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile over an already-sorted `values`. `p` in
    [0, 1]. Used to derive "tight but plausible" numeric thresholds (a
    budget, a protein floor, a prep-time ceiling) FROM the catalog's own
    distribution rather than picking an arbitrary literal — a budget near
    the 15th percentile of what dishes actually cost is a realistic
    constraint; a budget of ₹1 is not."""
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(len(values) * p)))
    return values[idx]


def _random_constraints(
    rng: random.Random, tags: list[str], preps: list[int], cfg: SynthConfig
) -> list[Constraint]:
    n = rng.randint(cfg.min_constraints, cfg.max_constraints)
    constraints: list[Constraint] = []
    for j in range(n):
        if tags and rng.random() < cfg.tag_constraint_probability:
            constraints.append(
                Constraint(
                    id=f"c{j}",
                    kind="exclude_tag",
                    severity=rng.choice(_SEVERITY_POOL),
                    value=rng.choice(tags),
                )
            )
        elif preps:
            constraints.append(
                Constraint(
                    id=f"c{j}",
                    kind="max_prep_minutes",
                    severity=rng.choice(_SOFT_SEVERITY_POOL),
                    value=float(rng.choice(preps)),
                )
            )
    return constraints


# ---- plausible-conflict templates for forced infeasibility ---------------
#
# Each template returns a SMALL (2-4), individually-ordinary constraint set
# that MIGHT jointly rule out the whole catalog — never a large stack of
# exclusions manufactured purely to hit a number. Whether a given draw
# actually IS infeasible is decided by consulting `_oracle_valid_ids`
# (rejection sampling in `_force_infeasible_constraints`), not assumed.


def _template_budget_vs_protein(
    rng: random.Random, costs: list[float], proteins: list[float]
) -> list[Constraint]:
    """Cheap AND protein-rich is a genuine, real-world tension: a tight
    budget (drawn from the LOW end of what dishes in this catalog actually
    cost) paired with a real protein goal (drawn from the HIGH end of what
    dishes actually deliver)."""
    budget = round(_percentile(costs, rng.uniform(0.03, 0.20)), 2)
    protein_floor = round(_percentile(proteins, rng.uniform(0.65, 0.95)), 2)
    return [
        Constraint(id="c0", kind="budget_max", severity=Severity.PREFERENCE, value=budget),
        Constraint(
            id="c1",
            kind="nutrient_range",
            severity=Severity.GOAL,
            value=(protein_floor, float("inf")),
            nutrient="protein_g",
        ),
    ]


def _template_prep_vs_tags(
    rng: random.Random, preps: list[int], tags: list[str]
) -> list[Constraint]:
    """A short prep-time ceiling plus one or two ordinary tag exclusions —
    "something quick, and not X" is an everyday profile, not a manufactured
    one."""
    sorted_preps = sorted(preps)
    limit = float(_percentile(sorted_preps, rng.uniform(0.0, 0.15)))
    excluded = rng.sample(tags, min(rng.choice([1, 2]), len(tags)))
    constraints = [
        Constraint(
            id="c0",
            kind="max_prep_minutes",
            severity=rng.choice(_SOFT_SEVERITY_POOL),
            value=limit,
        )
    ]
    constraints += [
        Constraint(id=f"c{i + 1}", kind="exclude_tag", severity=rng.choice(_SEVERITY_POOL), value=t)
        for i, t in enumerate(excluded)
    ]
    return constraints


def _template_nutrient_vs_tag(
    rng: random.Random, proteins: list[float], tags: list[str]
) -> list[Constraint]:
    """A real nutrient goal (a protein floor near the top of the catalog's
    own distribution) combined with ONE ordinary tag exclusion — "hits a
    protein target, and not X" — a band the remaining recipes may simply
    not reach, without needing a dozen exclusions to prove it."""
    protein_floor = round(_percentile(proteins, rng.uniform(0.6, 0.9)), 2)
    tag = rng.choice(tags)
    return [
        Constraint(id="c0", kind="exclude_tag", severity=rng.choice(_SEVERITY_POOL), value=tag),
        Constraint(
            id="c1",
            kind="nutrient_range",
            severity=Severity.GOAL,
            value=(protein_floor, float("inf")),
            nutrient="protein_g",
        ),
    ]


def _template_multi_tag(rng: random.Random, tags: list[str]) -> list[Constraint]:
    """Two or three exclusions that are each individually ordinary (a real
    profile can easily carry 2-3 dislikes/allergies) but happen to be
    jointly empty against this specific catalog — never the whole
    vocabulary at once."""
    k = min(rng.choice([2, 3]), len(tags))
    chosen = rng.sample(tags, k)
    return [
        Constraint(id=f"c{i}", kind="exclude_tag", severity=rng.choice(_SEVERITY_POOL), value=t)
        for i, t in enumerate(chosen)
    ]


def _tags_excluded_by(constraints: list[Constraint]) -> set[str]:
    return {c.value for c in constraints if c.kind == "exclude_tag" and isinstance(c.value, str)}


def _force_infeasible_constraints(
    rng: random.Random,
    catalog: Catalog,
    tags: list[str],
    preps: list[int],
    costs: list[float],
    proteins: list[float],
    recipes: list[Recipe],
    profile_id: str,
) -> list[Constraint]:
    """Rejection-sample a SMALL (2-4 constraint), individually-plausible
    conflict — never a large stack of exclusions — until the independent
    oracle confirms the result is genuinely infeasible.

    A purely random draw over `settings.synth.min_constraints`/
    `max_constraints` (a handful of tag exclusions) almost never rules out
    every recipe in a ~100-recipe catalog — measured at seed 0/n=200, only 3
    of 200 profiles came out infeasible that way. The first fix for that
    (stacking `exclude_tag` constraints one tag at a time until nothing
    survived) reliably hit the target fraction but did so by excluding a
    mean of 13.4 of the catalog's 16 known tags — 63-100% of the entire
    vocabulary at once, including self-contradictory combinations like
    excluding `vegan` AND `vegetarian` AND `red_meat` simultaneously. That
    is a degenerate "exclude everything" case: infeasible for a trivial
    reason, and inspectable evidence that the eval suite is generating
    noise rather than genuine conflicts.

    This version instead draws from four templates (`_template_budget_vs_
    protein`, `_template_prep_vs_tags`, `_template_nutrient_vs_tag`,
    `_template_multi_tag`), each a SMALL set of constraints that are
    individually ordinary — a real budget, a real protein goal, a couple of
    real exclusions — and checks with `_oracle_valid_ids` whether THIS
    particular combination happens to be infeasible against the real
    catalog. Using the oracle here is not circular: it is a
    generation-time filter deciding which candidate to keep, not the
    ground truth being validated — the accepted profile's `oracle_valid_ids`
    is still computed fresh by the caller, honestly, the same way every
    other profile's is.

    Every candidate that would exclude more than `_MAX_TAG_EXCLUSION_
    FRACTION` of the known tag vocabulary is rejected outright, regardless
    of whether it happens to be infeasible — a large exclusion set is the
    exact shortcut this function exists to avoid, so it is never accepted
    even as a fallback.
    """
    tag_cap = max(1, int(len(tags) * _MAX_TAG_EXCLUSION_FRACTION))
    templates: tuple[Callable[[], list[Constraint]], ...] = (
        lambda: _template_budget_vs_protein(rng, costs, proteins),
        lambda: _template_prep_vs_tags(rng, preps, tags),
        lambda: _template_nutrient_vs_tag(rng, proteins, tags),
        lambda: _template_multi_tag(rng, tags),
    )
    last_within_cap: list[Constraint] | None = None
    for _attempt in range(_MAX_INFEASIBILITY_ATTEMPTS):
        template = templates[rng.randrange(len(templates))]
        constraints = template()
        if len(_tags_excluded_by(constraints)) > tag_cap:
            continue
        last_within_cap = constraints
        cs = ConstraintSet(profile_id=profile_id, constraints=constraints)
        if not _oracle_valid_ids(catalog, cs, recipes):
            return constraints
    # No genuine small conflict turned up within the attempt budget — return
    # the last within-cap candidate tried (feasible, most likely) rather
    # than looping forever or falling back to the "exclude everything"
    # shortcut. The caller's own achieved-fraction band test is what catches
    # this firing often enough to move the numbers out of band.
    return last_within_cap if last_within_cap is not None else _template_multi_tag(rng, tags)


def generate_profiles(catalog: Catalog, n: int | None = None, seed: int = 0) -> list[SyntheticCase]:
    """Generate `n` synthetic constraint profiles, each with an exact oracle.

    `n` defaults to `settings.synth.default_profiles` — never a literal here.
    Reproducible for a fixed `seed`: the same catalog and seed always produce
    the same constraint sets, in the same order.

    Roughly `settings.synth.infeasible_fraction` of profiles are
    DELIBERATELY forced infeasible via a SMALL, plausible conflict
    (`_force_infeasible_constraints`) rather than left to a random draw,
    which makes the infeasible branch vanishingly rare (see that function's
    docstring). The rest are drawn randomly (`_random_constraints`) and may
    happen to be infeasible too — the forced fraction is a floor on how
    often the negative branch is exercised, not a ceiling.
    """
    cfg = get_settings().synth
    resolved_n = cfg.default_profiles if n is None else n
    rng = random.Random(seed)  # noqa: S311 — synthetic test data, not crypto
    recipes = _hydrated_recipes(catalog)
    tags = sorted({t for r in recipes for t in r.tags})
    preps = sorted({r.prep_minutes for r in recipes if r.prep_minutes})
    costs = sorted(r.cost_inr for r in recipes if r.cost_inr is not None)
    proteins = sorted(r.nutrition.protein_g for r in recipes if r.nutrition is not None)

    cases: list[SyntheticCase] = []
    for i in range(resolved_n):
        profile_id = f"synth_{i:04d}"
        if tags and costs and proteins and rng.random() < cfg.infeasible_fraction:
            constraints = _force_infeasible_constraints(
                rng, catalog, tags, preps, costs, proteins, recipes, profile_id
            )
        else:
            constraints = _random_constraints(rng, tags, preps, cfg)
        cs = ConstraintSet(profile_id=profile_id, constraints=constraints)
        cases.append(
            SyntheticCase(
                id=cs.profile_id,
                constraint_set=cs,
                oracle_valid_ids=_oracle_valid_ids(catalog, cs, recipes),
            )
        )
    return cases
