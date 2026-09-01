from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.t0_invariants.constraints import check_recipe, is_legal
from beatroot.trusted.catalog import Recipe


def _recipe(tags, kcal=500.0, protein=20.0, cost=100.0, prep=30):
    return Recipe(
        id="r1",
        name="Test",
        cuisine="test",
        prep_minutes=prep,
        tags=set(tags),
        ingredient_ids=["ing_x"],
    )


def _cs(*constraints):
    return ConstraintSet(profile_id="p", constraints=list(constraints))


def test_allergen_violation_is_detected():
    cs = _cs(Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    res = check_recipe(_recipe({"peanut", "vegan"}), cs)
    assert res.ok is False
    assert "c1" in res.violated


def test_clean_recipe_passes():
    cs = _cs(Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    assert check_recipe(_recipe({"vegan"}), cs).ok is True


# ---- exclude_ingredient synonym canonicalisation ------------------------
#
# The allergen-safety bug this fix round closes: `Constraint.value` for
# `exclude_ingredient` is user-facing text (often a real catalog synonym,
# e.g. "groundnut oil" for `ing_peanut_oil` in data/ingredients.yaml) while
# `Recipe.ingredient_ids` holds canonical ids. `unknown_vocabulary`
# canonicalises to validate such a value as known; `_exclude_ingredient`
# must canonicalise it too, or a validated, accepted constraint is never
# actually enforced. These tests resolve against the REAL repo ingredient
# data (`trusted.canonical.resolve_ingredient_id`'s only data source),
# same as `tests/t0_invariants/test_vocabulary.py` already does.


def test_exclude_ingredient_synonym_is_violated():
    """ "groundnut oil" is a real synonym of ing_peanut_oil — this is the
    exact reproduction from the bug report: a recipe carrying
    ing_peanut_oil must be found illegal, not silently passed."""
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="groundnut oil",
        )
    )
    r = Recipe(id="r1", name="Aloo tikki", ingredient_ids=["ing_peanut_oil"])
    res = check_recipe(r, cs)
    assert res.ok is False
    assert "med1" in res.violated
    assert is_legal(r, cs) is False


def test_exclude_ingredient_canonical_id_still_matches():
    """A value that is ALREADY a canonical id must keep working exactly as
    before — canonicalisation must not regress the already-correct path."""
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="ing_peanut_oil",
        )
    )
    r = Recipe(id="r1", name="Aloo tikki", ingredient_ids=["ing_peanut_oil"])
    assert check_recipe(r, cs).ok is False


def test_exclude_ingredient_synonym_does_not_flag_unrelated_recipe():
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="groundnut oil",
        )
    )
    r = Recipe(id="r2", name="Clean dish", ingredient_ids=["ing_rice"])
    assert check_recipe(r, cs).ok is True


def test_exclude_ingredient_unresolvable_value_is_uncheckable_not_satisfied():
    """A value the catalog has never heard of must be `uncheckable`, never
    `satisfied` — silently passing an unverifiable exclusion is exactly
    the failure this project exists to prevent. A hard (MEDICAL) exclusion
    like this must also make the recipe illegal, same as any other
    uncheckable hard constraint."""
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="not_a_real_ingredient_anywhere",
        )
    )
    r = Recipe(id="r1", name="Test", ingredient_ids=["ing_peanut_oil"])
    res = check_recipe(r, cs)
    assert "med1" in res.uncheckable
    assert "med1" not in res.satisfied
    assert "med1" not in res.violated
    assert is_legal(r, cs) is False


def test_exclude_ingredient_non_str_value_is_uncheckable() -> None:
    """A malformed non-str value has nothing to canonicalise or compare —
    never a crash, never a silent pass."""
    cs = _cs(
        Constraint.model_construct(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value=["ing_peanut_oil"],
        )
    )
    r = Recipe(id="r1", name="Test", ingredient_ids=["ing_peanut_oil"])
    res = check_recipe(r, cs)
    assert "med1" in res.uncheckable


def test_exclude_tag_non_str_value_is_uncheckable() -> None:
    """The sibling asymmetry fix: `_exclude_tag` had no `str` guard, so a
    non-str value either silently read as "satisfied" (any hashable
    scalar) or crashed outright (`TypeError: unhashable type` for a
    `list`, since `Constraint.value` allows `list[str]`). Neither is
    acceptable: a constraint that cannot be evaluated must be
    `uncheckable`, never a silent pass, never a crash."""
    cs = _cs(
        Constraint.model_construct(
            id="med1",
            kind="exclude_tag",
            severity=Severity.MEDICAL,
            value=["peanut"],
        )
    )
    r = _recipe({"peanut"})
    res = check_recipe(r, cs)
    assert "med1" in res.uncheckable
    assert "med1" not in res.satisfied
    assert "med1" not in res.violated
    assert is_legal(r, cs) is False


def test_exclude_tag_str_value_still_matches() -> None:
    """Guarding non-str values must not regress the ordinary str path."""
    cs = _cs(Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    assert check_recipe(_recipe({"peanut"}), cs).ok is False
    assert check_recipe(_recipe({"vegan"}), cs).ok is True


def test_is_legal_ignores_soft_constraints():
    """A GOAL violation must not make a recipe illegal — only hard ones do."""
    cs = _cs(
        Constraint(id="hard", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        Constraint(id="soft", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=10),
    )
    r = _recipe({"vegan"}, prep=60)
    assert is_legal(r, cs) is True
    assert check_recipe(r, cs).ok is False
    assert "soft" in check_recipe(r, cs).violated


def test_unregistered_constraint_kind_is_uncheckable_never_satisfied():
    """A kind nobody knows how to check must never silently pass."""
    from beatroot.t0_invariants.constraints import _evaluate

    rogue = Constraint.model_construct(
        id="x", kind="kind_from_the_future", severity=Severity.MEDICAL, value="?"
    )
    assert _evaluate(_recipe({"vegan"}), rogue) == "uncheckable"


def test_every_declared_constraint_kind_has_an_evaluator():
    """The registry and the ConstraintKind literal must not drift apart."""
    from typing import get_args

    from beatroot.contracts.core import ConstraintKind
    from beatroot.t0_invariants.constraints import _REGISTRY

    assert set(get_args(ConstraintKind)) == set(_REGISTRY)


def test_registered_kinds_is_the_same_set_sorted():
    """FREE-TEXT/NLP task: `agent.nodes.compile_node` reads this at runtime
    to tell the model what constraint vocabulary it may propose — it must
    be exactly the registry's own keys (never a hand-copied list that could
    drift), sorted deterministically for reproducible prompt rendering."""
    from beatroot.t0_invariants.constraints import _REGISTRY, registered_kinds

    kinds = registered_kinds()
    assert kinds == sorted(_REGISTRY)
    assert len(kinds) == len(set(kinds))  # no duplicates


def test_every_registered_kind_has_a_shape_and_a_parser():
    """`evaluator(kind, shape=..., parse=...)` requires both keyword
    arguments, so this should be true by construction — a registered kind
    with no shape text or no parser is a registration Python could not have
    accepted. This is the belt to that braces: it fails loudly if a future
    refactor ever makes `shape`/`parse` optional and a kind slips through
    with only an evaluator, exactly how `exclude_cuisine` was once
    advertised to the model and then silently dropped."""
    from beatroot.t0_invariants.constraints import _PARSERS, _REGISTRY, _SHAPES, kind_shapes

    assert set(_REGISTRY) == set(_SHAPES) == set(_PARSERS)
    assert set(kind_shapes()) == set(_REGISTRY)
    for kind, shape in kind_shapes().items():
        assert isinstance(shape, str) and shape.strip(), f"{kind!r} has no real shape text"


def test_missing_data_is_uncheckable_not_pass():
    """Absence of evidence is never evidence of absence. Feeds trust scoring."""
    cs = _cs(
        Constraint(id="c1", kind="nutrient_range", severity=Severity.GOAL, value=(30.0, 100.0))
    )
    r = _recipe({"vegan"})
    r.nutrition = None
    res = check_recipe(r, cs)
    assert "c1" in res.uncheckable
    assert "c1" not in res.violated


def test_nutrient_range_uncheckable_when_no_nutrition_data():
    """No nutrition data at all -> uncheckable, never a pass."""
    cs = _cs(
        Constraint(
            id="c1",
            kind="nutrient_range",
            severity=Severity.GOAL,
            value=(30.0, 100.0),
            nutrient="protein_g",
        )
    )
    r = _recipe({"vegan"})
    r.nutrition = None
    res = check_recipe(r, cs)
    assert "c1" in res.uncheckable
    assert "c1" not in res.violated
    assert "c1" not in res.satisfied


def test_nutrient_range_uncheckable_when_no_nutrient_named():
    """A range constraint that never says WHICH nutrient it constrains must
    never be silently guessed as protein — it is uncheckable."""
    from beatroot.contracts.nutrition import NutritionFacts

    cs = _cs(
        Constraint(id="c1", kind="nutrient_range", severity=Severity.GOAL, value=(30.0, 100.0))
    )
    r = _recipe({"vegan"})
    r.nutrition = NutritionFacts(
        kcal=500.0,
        protein_g=50.0,
        carbs_g=40.0,
        fat_g=10.0,
        sodium_mg=200.0,
        fibre_g=5.0,
        coverage=1.0,
        provenance="computed",
    )
    res = check_recipe(r, cs)
    assert "c1" in res.uncheckable
    assert "c1" not in res.violated
    assert "c1" not in res.satisfied


def test_nutrient_range_uncheckable_when_nutrient_name_unknown():
    """A `nutrient` naming a field NutritionFacts doesn't have is uncheckable,
    not a silent pass."""
    from beatroot.contracts.nutrition import NutritionFacts

    cs = _cs(
        Constraint(
            id="c1",
            kind="nutrient_range",
            severity=Severity.GOAL,
            value=(30.0, 100.0),
            nutrient="unobtanium_g",
        )
    )
    r = _recipe({"vegan"})
    r.nutrition = NutritionFacts(
        kcal=500.0,
        protein_g=50.0,
        carbs_g=40.0,
        fat_g=10.0,
        sodium_mg=200.0,
        fibre_g=5.0,
        coverage=1.0,
        provenance="computed",
    )
    res = check_recipe(r, cs)
    assert "c1" in res.uncheckable
    assert "c1" not in res.violated
    assert "c1" not in res.satisfied


def test_nutrient_range_uncheckable_when_field_is_non_numeric():
    """A `nutrient` naming a real but non-numeric field (e.g. `provenance`)
    must degrade to uncheckable, never crash and never pass."""
    from beatroot.contracts.nutrition import NutritionFacts

    cs = _cs(
        Constraint(
            id="c1",
            kind="nutrient_range",
            severity=Severity.GOAL,
            value=(30.0, 100.0),
            nutrient="provenance",
        )
    )
    r = _recipe({"vegan"})
    r.nutrition = NutritionFacts(
        kcal=500.0,
        protein_g=50.0,
        carbs_g=40.0,
        fat_g=10.0,
        sodium_mg=200.0,
        fibre_g=5.0,
        coverage=1.0,
        provenance="computed",
    )
    res = check_recipe(r, cs)
    assert "c1" in res.uncheckable
    assert "c1" not in res.violated
    assert "c1" not in res.satisfied


def test_uncheckable_hard_constraint_makes_recipe_illegal_missing_nutrition():
    """We do not serve a meal we cannot prove safe."""
    cs = _cs(
        Constraint(
            id="med",
            kind="nutrient_range",
            severity=Severity.MEDICAL,
            value=(0.0, 500.0),
            nutrient="sodium_mg",
        )
    )
    r = _recipe({"vegan"})
    r.nutrition = None
    assert check_recipe(r, cs).uncheckable == ["med"]
    assert is_legal(r, cs) is False


def test_uncheckable_hard_constraint_makes_recipe_illegal_missing_cost():
    cs = _cs(Constraint(id="med", kind="budget_max", severity=Severity.RELIGIOUS, value=100))
    r = _recipe({"vegan"})
    r.cost_inr = None
    assert is_legal(r, cs) is False


def test_unregistered_hard_constraint_kind_makes_recipe_illegal():
    cs = _cs(
        Constraint.model_construct(
            id="med", kind="kind_from_the_future", severity=Severity.MEDICAL, value="?"
        )
    )
    assert is_legal(_recipe({"vegan"}), cs) is False


def test_uncheckable_soft_constraint_does_not_make_recipe_illegal():
    """The counterpart: soft constraints never gate legality, only reporting."""
    cs = _cs(Constraint(id="goal", kind="budget_max", severity=Severity.GOAL, value=100))
    r = _recipe({"vegan"})
    r.cost_inr = None
    assert is_legal(r, cs) is True
    assert check_recipe(r, cs).uncheckable == ["goal"]
