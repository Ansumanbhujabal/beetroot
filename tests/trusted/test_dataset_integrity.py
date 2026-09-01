"""Dataset integrity checks for `data/recipes.yaml` and `data/ingredients.yaml`.

These are not unit tests of engine behaviour — they are a safety net over the
CATALOG itself, seeded and read through the real `store.db` + `trusted.catalog`
path so a bug here means the same bug production would hit. Written as part of
the recipe-catalog expansion (jain/vegan/pescetarian/low-sodium/low-carb
relief); see `.sdd/briefs/dataset-expansion-report.md` for the before/after
legal-candidate counts this expansion was measured against.
"""

from pathlib import Path

import yaml

from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.store.db import connect, seed
from beatroot.t0_invariants.constraints import is_legal
from beatroot.trusted.catalog import Catalog

DATA = Path(__file__).parents[2] / "data"

# Same per-serving sanity band the real catalog's own distribution sits in
# (measured: kcal 85-845, carbs 3-117g, protein 2.5-71g, sodium 4-985mg
# across the pre-expansion 100-recipe catalog). Generous headroom either
# side so this catches a genuine data-entry error (a misplaced decimal, a
# gram figure fat-fingered by 10x) without being a tight metric a future
# recipe has to hug.
MIN_KCAL, MAX_KCAL = 30.0, 1500.0
MAX_CARBS_G = 200.0
MAX_PROTEIN_G = 120.0
MAX_SODIUM_MG = 3000.0
MAX_FAT_G = 120.0


def _catalog(tmp_path: Path) -> Catalog:
    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    return Catalog(conn)


def _raw_yaml() -> tuple[list[dict], list[dict]]:
    ingredients = yaml.safe_load((DATA / "ingredients.yaml").read_text())
    recipes = yaml.safe_load((DATA / "recipes.yaml").read_text())
    return ingredients, recipes


def test_ingredient_ids_are_unique() -> None:
    ingredients, _ = _raw_yaml()
    ids = [i["id"] for i in ingredients]
    assert len(ids) == len(set(ids)), "duplicate ingredient id in data/ingredients.yaml"


def test_recipe_ids_are_unique() -> None:
    _, recipes = _raw_yaml()
    ids = [r["id"] for r in recipes]
    assert len(ids) == len(set(ids)), "duplicate recipe id in data/recipes.yaml"


def test_every_recipe_ingredient_reference_exists() -> None:
    ingredients, recipes = _raw_yaml()
    known = {i["id"] for i in ingredients}
    missing = [
        (r["id"], ref["ingredient_id"])
        for r in recipes
        for ref in r["ingredients"]
        if ref["ingredient_id"] not in known
    ]
    assert not missing, f"recipes reference unknown ingredient ids: {missing}"


def test_no_recipe_has_zero_ingredients() -> None:
    _, recipes = _raw_yaml()
    empty = [r["id"] for r in recipes if not r["ingredients"]]
    assert not empty, f"recipes with no ingredients: {empty}"


def test_recipe_ids_follow_naming_convention() -> None:
    _, recipes = _raw_yaml()
    bad = [r["id"] for r in recipes if not r["id"].startswith("rec_")]
    assert not bad, f"recipe ids not prefixed rec_: {bad}"


def test_ingredient_ids_follow_naming_convention() -> None:
    ingredients, _ = _raw_yaml()
    bad = [i["id"] for i in ingredients if not i["id"].startswith("ing_")]
    assert not bad, f"ingredient ids not prefixed ing_: {bad}"


def test_catalog_currently_holds_at_least_170_recipes() -> None:
    """Regression guard for the dataset-expansion brief: catalog grew from
    100 recipes to 174 (60-80 new, additive only). A drop below this means
    something got deleted rather than added to."""
    _, recipes = _raw_yaml()
    assert len(recipes) >= 170


def test_nutrition_totals_are_nonzero_and_in_sane_per_serving_ranges(tmp_path: Path) -> None:
    """Every recipe's computed nutrition (per_100g summed over its real
    ingredient list — see `t0_invariants.nutrition_math.compute`) must be
    non-zero and plausible for a single serving. Feeds medical constraints
    (carb ceilings, sodium ceilings) directly, so an invented or malformed
    figure here is a safety bug, not a cosmetic one."""
    cat = _catalog(tmp_path)
    recipes = cat.hydrated()
    bad: list[str] = []
    for r in recipes:
        n = r.nutrition
        if n is None:
            bad.append(f"{r.id}: no nutrition computed")
            continue
        if n.kcal <= 0:
            bad.append(f"{r.id}: kcal={n.kcal} <= 0")
        if not (MIN_KCAL <= n.kcal <= MAX_KCAL):
            bad.append(f"{r.id}: kcal={n.kcal} outside [{MIN_KCAL}, {MAX_KCAL}]")
        if not (0 <= n.carbs_g <= MAX_CARBS_G):
            bad.append(f"{r.id}: carbs_g={n.carbs_g} outside [0, {MAX_CARBS_G}]")
        if not (0 <= n.protein_g <= MAX_PROTEIN_G):
            bad.append(f"{r.id}: protein_g={n.protein_g} outside [0, {MAX_PROTEIN_G}]")
        if not (0 <= n.sodium_mg <= MAX_SODIUM_MG):
            bad.append(f"{r.id}: sodium_mg={n.sodium_mg} outside [0, {MAX_SODIUM_MG}]")
        if not (0 <= n.fat_g <= MAX_FAT_G):
            bad.append(f"{r.id}: fat_g={n.fat_g} outside [0, {MAX_FAT_G}]")
    assert not bad, "nutrition sanity failures:\n" + "\n".join(bad)


def test_every_recipe_has_a_priced_cost(tmp_path: Path) -> None:
    """`recipe_cost_inr` returns None only when an ingredient is missing a
    `cost_per_100g_inr` — every catalog recipe should price cleanly."""
    cat = _catalog(tmp_path)
    unpriced = [r.id for r in cat.hydrated() if r.cost_inr is None]
    assert not unpriced, f"recipes with no computable cost: {unpriced}"


def _legal_count(cat: Catalog, cs: ConstraintSet) -> int:
    return sum(1 for r in cat.hydrated() if is_legal(r, cs))


def test_constrained_presets_have_a_healthy_candidate_pool(tmp_path: Path) -> None:
    """Regression guard for the scarcity this expansion set out to fix.

    Mirrors the hard constraints of the tightest `data/profiles.yaml`
    presets directly (never imports from profiles.yaml — that file is owned
    by a different workstream and must stay free to change shape without
    breaking this test). Thresholds sit comfortably below what the expanded
    catalog actually measures (see the report), so this fails loudly only if
    the pool collapses back toward the pre-expansion counts, not on routine
    catalog churn.
    """
    cat = _catalog(tmp_path)

    jain = ConstraintSet(
        profile_id="jain",
        constraints=[
            Constraint(id="a", kind="require_tag", severity=Severity.RELIGIOUS, value="vegetarian"),
            Constraint(id="b", kind="exclude_tag", severity=Severity.RELIGIOUS, value="egg"),
            Constraint(
                id="c", kind="exclude_tag", severity=Severity.RELIGIOUS, value="root_vegetable"
            ),
            Constraint(id="d", kind="exclude_tag", severity=Severity.RELIGIOUS, value="allium"),
        ],
    )
    vegan = ConstraintSet(
        profile_id="vegan",
        constraints=[
            Constraint(id="a", kind="require_tag", severity=Severity.DIETARY, value="vegan"),
        ],
    )
    veg_lactose_intolerant = ConstraintSet(
        profile_id="vegetarian_lactose_intolerant",
        constraints=[
            Constraint(id="a", kind="require_tag", severity=Severity.DIETARY, value="vegetarian"),
            Constraint(id="b", kind="exclude_tag", severity=Severity.DIETARY, value="egg"),
            Constraint(id="c", kind="exclude_tag", severity=Severity.MEDICAL, value="dairy"),
        ],
    )
    pescetarian_peanut_allergy = ConstraintSet(
        profile_id="pescetarian_peanut_allergy",
        constraints=[
            Constraint(
                id="a",
                kind="require_any_tag",
                severity=Severity.DIETARY,
                value=["vegetarian", "fish"],
            ),
            Constraint(id="b", kind="exclude_tag", severity=Severity.DIETARY, value="egg"),
            Constraint(id="c", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        ],
    )
    diabetic = ConstraintSet(
        profile_id="diabetic",
        constraints=[
            Constraint(
                id="a",
                kind="nutrient_range",
                severity=Severity.MEDICAL,
                value=(0.0, 60.0),
                nutrient="carbs_g",
            ),
        ],
    )
    hypertension_low_sodium = ConstraintSet(
        profile_id="hypertension_low_sodium",
        constraints=[
            Constraint(
                id="a",
                kind="nutrient_range",
                severity=Severity.MEDICAL,
                value=(0.0, 400.0),
                nutrient="sodium_mg",
            ),
        ],
    )
    severe_nut_allergy = ConstraintSet(
        profile_id="severe_nut_allergy",
        constraints=[
            Constraint(id="a", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
            Constraint(id="b", kind="exclude_tag", severity=Severity.MEDICAL, value="tree_nut"),
        ],
    )

    # Pre-expansion counts (of 100 recipes) were: jain 27, vegan 42,
    # vegetarian_lactose_intolerant 42, pescetarian_peanut_allergy 68,
    # diabetic 67, hypertension_low_sodium 65, severe_nut_allergy 91.
    # Thresholds below are set well under the post-expansion measured
    # values so this is a floor against regression, not a tight target.
    assert _legal_count(cat, jain) >= 50, "jain candidate pool regressed toward pre-expansion size"
    assert _legal_count(cat, vegan) >= 65
    assert _legal_count(cat, veg_lactose_intolerant) >= 65
    assert _legal_count(cat, pescetarian_peanut_allergy) >= 100
    assert _legal_count(cat, diabetic) >= 90
    assert _legal_count(cat, hypertension_low_sodium) >= 90
    assert _legal_count(cat, severe_nut_allergy) >= 120


def test_pescetarian_has_more_than_a_handful_of_fish_dishes(tmp_path: Path) -> None:
    """Pre-expansion the catalog had exactly 6 recipes carrying the `fish`
    tag, one of which was excluded for egg — a razor-thin pool for the
    pescetarian preset. This is a floor, not the measured count."""
    cat = _catalog(tmp_path)
    fish_recipes = [r for r in cat.recipes() if "fish" in r.tags]
    assert len(fish_recipes) >= 10
