import pytest

from beatroot.t0_invariants.nutrition_math import MalformedRecipeError, compute, recipe_cost_inr


class _Cat:
    def __init__(self, table):
        self._t = table

    def ingredient_payload(self, iid):
        return self._t.get(iid)


def test_scales_per_100g_by_grams():
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": 10,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "sodium_mg": 0,
                    "fibre_g": 0,
                }
            }
        }
    )
    recipe = {"ingredients": [{"ingredient_id": "a", "grams": 250}]}
    n = compute(recipe, cat)
    assert n.kcal == pytest.approx(250.0)
    assert n.protein_g == pytest.approx(25.0)
    assert n.coverage == 1.0
    assert n.provenance == "computed"


def test_coverage_reflects_missing_ingredient_data():
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": 10,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "sodium_mg": 0,
                    "fibre_g": 0,
                }
            },
            "b": None,
        }
    )
    recipe = {
        "ingredients": [{"ingredient_id": "a", "grams": 100}, {"ingredient_id": "b", "grams": 100}]
    }
    n = compute(recipe, cat)
    assert n.coverage == pytest.approx(0.5)


def test_coverage_is_mass_weighted_not_count_weighted():
    """A missing 1g spice matters less than a missing 200g protein."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": 0,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "sodium_mg": 0,
                    "fibre_g": 0,
                }
            },
            "b": None,
        }
    )
    recipe = {
        "ingredients": [{"ingredient_id": "a", "grams": 199}, {"ingredient_id": "b", "grams": 1}]
    }
    assert compute(recipe, cat).coverage == pytest.approx(0.995)


def test_partial_data_earns_proportional_coverage():
    """An ingredient with only 3/6 fields contributes data but reduces coverage."""
    cat = _Cat(
        {
            "a": {"per_100g": {"kcal": 100, "protein_g": 10}},  # only 2 of 6 fields
            "b": {
                "per_100g": {
                    "kcal": 50,
                    "protein_g": 5,
                    "carbs_g": 20,
                    "fat_g": 2,
                    "sodium_mg": 100,
                    "fibre_g": 1,
                }
            },
        }
    )  # all 6 fields
    recipe = {
        "ingredients": [{"ingredient_id": "a", "grams": 100}, {"ingredient_id": "b", "grams": 100}]
    }
    n = compute(recipe, cat)
    # a has 2/6 fields (33.3%), b has 6/6 (100%) -> coverage = (100*2/6 + 100*6/6)/200
    expected_coverage = (100 * 2 / 6 + 100 * 6 / 6) / 200.0
    assert n.coverage == pytest.approx(expected_coverage)
    # kcal should have contributions from both
    assert n.kcal == pytest.approx(100.0 + 50.0)
    # protein_g should have contributions from both
    assert n.protein_g == pytest.approx(10.0 + 5.0)
    # carbs_g should only come from b
    assert n.carbs_g == pytest.approx(20.0)


def test_negative_grams_raises_error():
    """Negative grams are a data bug and must be rejected."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": 10,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "sodium_mg": 0,
                    "fibre_g": 0,
                }
            }
        }
    )
    recipe = {"id": "test_recipe", "ingredients": [{"ingredient_id": "a", "grams": -50}]}
    with pytest.raises(MalformedRecipeError) as exc_info:
        compute(recipe, cat)
    assert "test_recipe" in str(exc_info.value)
    assert "a" in str(exc_info.value)
    assert "-50" in str(exc_info.value)


def test_nan_grams_raises_error():
    """NaN grams are a data bug and must be rejected."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": 10,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "sodium_mg": 0,
                    "fibre_g": 0,
                }
            }
        }
    )
    recipe = {"id": "test_recipe", "ingredients": [{"ingredient_id": "a", "grams": float("nan")}]}
    with pytest.raises(MalformedRecipeError) as exc_info:
        compute(recipe, cat)
    assert "test_recipe" in str(exc_info.value)
    assert "a" in str(exc_info.value)


def test_null_per100_field_excluded_from_coverage():
    """A null per_100g field must not earn coverage credit and must not crash."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": None,
                    "carbs_g": 5,
                    "fat_g": 2,
                    "sodium_mg": 100,
                    "fibre_g": 1,
                }
            }
        }
    )
    recipe = {"ingredients": [{"ingredient_id": "a", "grams": 100}]}
    n = compute(recipe, cat)
    # protein_g is null, so only 5/6 fields are usable: (100*5/6)/100 = 5/6 coverage
    assert n.coverage == pytest.approx(5.0 / 6.0)
    # kcal should be computed from the usable value
    assert n.kcal == pytest.approx(100.0)
    # protein_g should be 0 since the value is null
    assert n.protein_g == 0.0


def test_string_per100_field_excluded_from_coverage():
    """A string per_100g field must not earn coverage credit."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": "10",
                    "carbs_g": 5,
                    "fat_g": 2,
                    "sodium_mg": 100,
                    "fibre_g": 1,
                }
            }
        }
    )
    recipe = {"ingredients": [{"ingredient_id": "a", "grams": 100}]}
    n = compute(recipe, cat)
    # protein_g is string, so only 5/6 fields are usable: (100*5/6)/100 = 5/6 coverage
    assert n.coverage == pytest.approx(5.0 / 6.0)
    assert n.protein_g == 0.0


def test_negative_per100_field_excluded_from_coverage():
    """A negative per_100g field must not earn coverage credit."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": -5,
                    "carbs_g": 5,
                    "fat_g": 2,
                    "sodium_mg": 100,
                    "fibre_g": 1,
                }
            }
        }
    )
    recipe = {"ingredients": [{"ingredient_id": "a", "grams": 100}]}
    n = compute(recipe, cat)
    # protein_g is negative, so only 5/6 fields are usable
    assert n.coverage == pytest.approx(5.0 / 6.0)
    assert n.protein_g == 0.0


def test_nan_per100_field_excluded_from_coverage():
    """A NaN per_100g field must not earn coverage credit."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": float("nan"),
                    "carbs_g": 5,
                    "fat_g": 2,
                    "sodium_mg": 100,
                    "fibre_g": 1,
                }
            }
        }
    )
    recipe = {"ingredients": [{"ingredient_id": "a", "grams": 100}]}
    n = compute(recipe, cat)
    # protein_g is NaN, so only 5/6 fields are usable
    assert n.coverage == pytest.approx(5.0 / 6.0)
    assert n.protein_g == 0.0


def test_infinity_per100_field_excluded_from_coverage():
    """An infinite per_100g field must not earn coverage credit."""
    cat = _Cat(
        {
            "a": {
                "per_100g": {
                    "kcal": 100,
                    "protein_g": float("inf"),
                    "carbs_g": 5,
                    "fat_g": 2,
                    "sodium_mg": 100,
                    "fibre_g": 1,
                }
            }
        }
    )
    recipe = {"ingredients": [{"ingredient_id": "a", "grams": 100}]}
    n = compute(recipe, cat)
    # protein_g is inf, so only 5/6 fields are usable
    assert n.coverage == pytest.approx(5.0 / 6.0)
    assert n.protein_g == 0.0


def test_empty_recipe_cost_returns_none():
    """An empty ingredient list has unknown cost, not zero cost."""
    cat = _Cat({})
    recipe = {"ingredients": []}
    cost = recipe_cost_inr(recipe, cat)
    assert cost is None


def test_recipe_cost_excludes_unusable_values():
    """A null cost_per_100g_inr returns None, not partial sum."""
    cat = _Cat({"a": {"cost_per_100g_inr": 10.0}, "b": {"cost_per_100g_inr": None}})
    recipe = {
        "ingredients": [{"ingredient_id": "a", "grams": 100}, {"ingredient_id": "b", "grams": 100}]
    }
    cost = recipe_cost_inr(recipe, cat)
    assert cost is None
