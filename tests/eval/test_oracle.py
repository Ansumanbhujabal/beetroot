from pathlib import Path

import pytest

from beatroot.eval.synth import profiles as profiles_mod
from beatroot.eval.synth.profiles import generate_profiles
from beatroot.store.db import connect, seed
from beatroot.t0_invariants import constraints as t0_constraints
from beatroot.t0_invariants.constraints import check_recipe
from beatroot.trusted.catalog import Catalog

ROOT = Path(__file__).parents[2]


def _catalog(tmp_path: Path) -> Catalog:
    conn = connect(tmp_path / "t.db")
    seed(conn, ROOT / "data")
    return Catalog(conn)


def test_oracle_cross_checks_against_check_recipe(tmp_path: Path) -> None:
    """Two INDEPENDENT implementations of constraint semantics agree.

    `generate_profiles` computes `oracle_valid_ids` with
    `eval.synth.profiles._oracle_valid_ids`, a from-scratch re-implementation
    that never calls `check_recipe`. This test recomputes the valid set a
    SECOND way — via `check_recipe`, the actual code under test — and
    asserts the two agree. That is a genuine cross-check, not a
    self-recomputation: `test_oracle_cross_check_detects_a_broken_evaluator`
    below proves it can actually fail when the two implementations
    disagree, which a "recompute the same call" test structurally cannot
    do. Spec §12.
    """
    cat = _catalog(tmp_path)
    for case in generate_profiles(cat, n=50, seed=7):
        recomputed_via_check_recipe = {
            r.id for r in cat.recipes() if check_recipe(r, case.constraint_set).ok
        }
        assert case.oracle_valid_ids == recomputed_via_check_recipe


def test_oracle_cross_check_detects_a_broken_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the cross-check above is a real test, not a decorative one.

    Breaks `t0_invariants.constraints`' registered `exclude_tag` evaluator
    by inverting its outcome (violated <-> satisfied), then shows the
    independent oracle and `check_recipe` now DISAGREE on a profile that
    actually exercises an `exclude_tag` constraint. If this test ever
    starts passing with the assertion inverted (oracle == check_recipe under
    the broken evaluator), the oracle has stopped being independent.
    """
    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, n=50, seed=7)
    tag_cases = [
        c for c in cases if any(x.kind == "exclude_tag" for x in c.constraint_set.constraints)
    ]
    assert tag_cases, "need at least one exclude_tag case to exercise the broken evaluator"

    def _inverted_exclude_tag(recipe, c):  # type: ignore[no-untyped-def]
        return "satisfied" if c.value in recipe.tags else "violated"

    monkeypatch.setitem(t0_constraints._REGISTRY, "exclude_tag", _inverted_exclude_tag)

    disagreements = 0
    for case in tag_cases:
        recomputed_via_broken_check_recipe = {
            r.id for r in cat.recipes() if check_recipe(r, case.constraint_set).ok
        }
        if case.oracle_valid_ids != recomputed_via_broken_check_recipe:
            disagreements += 1
    assert disagreements > 0, (
        "the independent oracle should disagree with check_recipe once "
        "exclude_tag's evaluator is inverted — if it doesn't, the oracle is "
        "not actually independent of the code it is meant to validate"
    )


def test_generation_is_reproducible(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    a = generate_profiles(cat, n=20, seed=42)
    b = generate_profiles(cat, n=20, seed=42)
    assert [c.constraint_set.fingerprint() for c in a] == [
        c.constraint_set.fingerprint() for c in b
    ]


def test_generation_produces_both_feasible_and_infeasible_profiles(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, n=100, seed=1)
    assert any(c.oracle_valid_ids for c in cases), "need feasible cases"
    assert any(not c.oracle_valid_ids for c in cases), "need infeasible cases"


@pytest.mark.parametrize("seed_value", [0, 2, 3])
def test_infeasible_fraction_is_a_meaningful_share_not_a_hair(
    tmp_path: Path, seed_value: int
) -> None:
    """A purely random draw over a ~100-recipe catalog almost never rules
    out every recipe (measured: 3/200 at seed 0 before forcing was added,
    0/200 at seeds 2 and 3). `_force_infeasible_constraints` targets
    `settings.synth.infeasible_fraction` (0.25) by rejection-sampling small,
    plausible conflicts, so the ACHIEVED fraction across independent seeds
    should sit in a sane band around that target — loose enough to tolerate
    randomness in how many of the *unforced* draws also land infeasible,
    tight enough that a future change silently breaking the forcing path
    fails loudly here."""
    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, n=200, seed=seed_value)
    infeasible = sum(1 for c in cases if not c.oracle_valid_ids)
    fraction = infeasible / len(cases)
    assert 0.15 <= fraction <= 0.45, (
        f"seed={seed_value}: infeasible fraction {fraction:.2f} outside [0.15, 0.45]"
    )


def test_forced_infeasible_profiles_are_genuinely_infeasible(tmp_path: Path) -> None:
    """`_force_infeasible_constraints` must produce constraint sets whose
    oracle-valid set really is empty — checked directly, independent of
    `generate_profiles`'s own bookkeeping."""
    cat = _catalog(tmp_path)
    import random

    rng = random.Random(99)  # noqa: S311 — test data, not crypto
    recipes = profiles_mod._hydrated_recipes(cat)
    tags = sorted({t for r in recipes for t in r.tags})
    preps = sorted({r.prep_minutes for r in recipes if r.prep_minutes})
    costs = sorted(r.cost_inr for r in recipes if r.cost_inr is not None)
    proteins = sorted(r.nutrition.protein_g for r in recipes if r.nutrition is not None)
    from beatroot.contracts.core import ConstraintSet

    constraints = profiles_mod._force_infeasible_constraints(
        rng, cat, tags, preps, costs, proteins, recipes, "forced_test"
    )
    cs = ConstraintSet(profile_id="forced_test", constraints=constraints)
    assert profiles_mod._oracle_valid_ids(cat, cs) == set()


def test_forced_infeasible_profiles_are_small_and_plausible(tmp_path: Path) -> None:
    """The specific failure mode this generation strategy exists to avoid:
    hitting the infeasible-fraction target by excluding most of the tag
    vocabulary at once ("exclude everything" is trivially infeasible and
    tests nothing interesting). Forced-infeasible profiles must instead be
    SMALL (2-4 constraints, per the reviewed templates) and must never
    exclude more than a small fraction of the known tags."""
    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, n=200, seed=0)
    infeasible = [c for c in cases if not c.oracle_valid_ids]
    assert infeasible, "need at least one infeasible case to check"

    all_tags = sorted({t for r in cat.recipes() for t in r.tags})
    tag_cap = max(1, int(len(all_tags) * profiles_mod._MAX_TAG_EXCLUSION_FRACTION))

    for case in infeasible:
        n_constraints = len(case.constraint_set.constraints)
        assert 2 <= n_constraints <= 4, (
            f"{case.id}: {n_constraints} constraints, expected 2-4 (a plausible "
            "profile, not a manufactured stack)"
        )
        excluded_tags = {
            c.value for c in case.constraint_set.constraints if c.kind == "exclude_tag"
        }
        # No generated profile may exclude more than a small fraction of the
        # tag vocabulary — a profile excluding most of the vocabulary is
        # degenerate ("nothing is allowed"), not a genuine, inspectable
        # conflict between a few reasonable constraints. This is the
        # regression guard for the exact shortcut flagged in review.
        assert len(excluded_tags) <= tag_cap, (
            f"{case.id}: excludes {len(excluded_tags)}/{len(all_tags)} tags, "
            f"over the {profiles_mod._MAX_TAG_EXCLUSION_FRACTION:.0%} cap ({tag_cap})"
        )


@pytest.mark.parametrize("seed_value", [0, 2, 3])
def test_forced_infeasible_statistics_stay_plausible(tmp_path: Path, seed_value: int) -> None:
    """Direct statistics on forced-infeasible profiles, across three seeds:
    mean/max constraint count stays small, mean/max tags-excluded stays
    small. Printed via -s for the fix-round report; asserted here so a
    regression toward the "exclude everything" shortcut fails the suite,
    not just an eyeball check."""
    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, n=200, seed=seed_value)
    infeasible = [c for c in cases if not c.oracle_valid_ids]
    assert infeasible

    counts = [len(c.constraint_set.constraints) for c in infeasible]
    tag_counts = [
        len({x.value for x in c.constraint_set.constraints if x.kind == "exclude_tag"})
        for c in infeasible
    ]
    mean_count = sum(counts) / len(counts)
    mean_tags = sum(tag_counts) / len(tag_counts)

    assert mean_count <= 3.0, f"seed={seed_value}: mean constraint count {mean_count:.2f} > 3.0"
    assert max(counts) <= 4, f"seed={seed_value}: max constraint count {max(counts)} > 4"
    assert mean_tags <= 2.0, f"seed={seed_value}: mean tags excluded {mean_tags:.2f} > 2.0"


def test_default_n_comes_from_settings_not_a_literal(tmp_path: Path) -> None:
    from beatroot.settings import get_settings

    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, seed=3)
    assert len(cases) == get_settings().synth.default_profiles


def test_generated_ids_are_unique_and_ordered(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = generate_profiles(cat, n=10, seed=5)
    ids = [c.id for c in cases]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


# ---- unit coverage for the oracle's non-tag constraint kinds -------------
# `generate_profiles` itself only ever draws exclude_tag/max_prep_minutes
# constraints, so budget_max/nutrient_range/exclude_ingredient handling in
# `_oracle_violates` needs its own direct coverage rather than relying on
# generation to happen to exercise it.


def test_oracle_handles_exclude_ingredient(tmp_path: Path) -> None:
    from beatroot.contracts.core import Constraint, ConstraintSet, Severity

    cat = _catalog(tmp_path)
    recipe = next(r for r in cat.recipes() if r.ingredient_ids)
    excluded_id = recipe.ingredient_ids[0]
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(
                id="c1", kind="exclude_ingredient", severity=Severity.MEDICAL, value=excluded_id
            )
        ],
    )
    assert recipe.id not in profiles_mod._oracle_valid_ids(cat, cs)


def test_oracle_resolves_exclude_ingredient_synonym(tmp_path: Path) -> None:
    """The gap this follow-up closes: `_oracle_violates` must canonicalise
    `c.value` before comparing against `Recipe.ingredient_ids`, exactly
    like `t0_invariants.constraints._exclude_ingredient` and
    `eval.verifiers.hard_constraint.verify` — or the ground-truth oracle
    itself would be the wrong answer the moment a synonym value reached
    it. "groundnut oil" is the real synonym recorded against
    `ing_peanut_oil` in data/ingredients.yaml; Aloo tikki is the real
    recipe that carries it (see the bug report reproduction)."""
    from beatroot.contracts.core import Constraint, ConstraintSet, Severity

    cat = _catalog(tmp_path)
    recipe = next(r for r in cat.recipes() if "ing_peanut_oil" in r.ingredient_ids)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(
                id="c1",
                kind="exclude_ingredient",
                severity=Severity.MEDICAL,
                value="groundnut oil",
            )
        ],
    )
    assert recipe.id not in profiles_mod._oracle_valid_ids(cat, cs)
    assert profiles_mod._oracle_violates(recipe, cs.constraints[0]) is True


def test_oracle_unresolvable_exclude_ingredient_is_not_a_violation(tmp_path: Path) -> None:
    """Mirrors `check_recipe`'s own semantics: a value the catalog cannot
    resolve at all is "uncheckable", which this oracle folds into "not a
    violation" — same posture as every other branch's missing/malformed
    data case."""
    from beatroot.contracts.core import Constraint, Severity

    cat = _catalog(tmp_path)
    recipe = next(r for r in cat.recipes() if "ing_peanut_oil" in r.ingredient_ids)
    c = Constraint(
        id="c1",
        kind="exclude_ingredient",
        severity=Severity.MEDICAL,
        value="not_a_real_ingredient_anywhere",
    )
    assert profiles_mod._oracle_violates(recipe, c) is False


def test_oracle_non_str_exclude_tag_value_is_not_a_violation(tmp_path: Path) -> None:
    """A malformed non-str `exclude_tag` value must not crash (it is
    unhashable for a `list`) and must not be silently treated as met."""
    from beatroot.contracts.core import Constraint, Severity

    cat = _catalog(tmp_path)
    recipe = cat.recipes()[0]
    c = Constraint.model_construct(
        id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value=["not", "a", "str"]
    )
    assert profiles_mod._oracle_violates(recipe, c) is False


def test_oracle_handles_budget_max(tmp_path: Path) -> None:
    from beatroot.contracts.core import Constraint, ConstraintSet, Severity

    cat = _catalog(tmp_path)
    valid = profiles_mod._oracle_valid_ids(
        cat,
        ConstraintSet(
            profile_id="p",
            constraints=[
                Constraint(id="c1", kind="budget_max", severity=Severity.PREFERENCE, value=0.0)
            ],
        ),
    )
    # A budget of ₹0 should exclude every recipe with a known, positive cost.
    priced = [r for r in cat.hydrated() if r.cost_inr is not None and r.cost_inr > 0]
    assert priced
    assert valid.isdisjoint({r.id for r in priced})


def test_oracle_handles_nutrient_range(tmp_path: Path) -> None:
    from beatroot.contracts.core import Constraint, ConstraintSet, Severity

    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(
                id="c1",
                kind="nutrient_range",
                severity=Severity.GOAL,
                value=(0.0, 0.0),
                nutrient="kcal",
            )
        ],
    )
    valid = profiles_mod._oracle_valid_ids(cat, cs)
    # A [0, 0] kcal range should exclude every recipe with any known calories.
    caloric = [r for r in cat.hydrated() if r.nutrition and r.nutrition.kcal > 0]
    assert caloric
    assert valid.isdisjoint({r.id for r in caloric})


def test_oracle_uncheckable_never_counts_as_violated(tmp_path: Path) -> None:
    """A nutrient_range naming an unknown nutrient is uncheckable, not a
    violation — mirrors check_recipe's own ok = not violated semantics."""
    from beatroot.contracts.core import Constraint, ConstraintSet, Severity

    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(
                id="c1",
                kind="nutrient_range",
                severity=Severity.GOAL,
                value=(0.0, 100.0),
                nutrient="does_not_exist",
            )
        ],
    )
    assert profiles_mod._oracle_valid_ids(cat, cs) == {r.id for r in cat.recipes()}
