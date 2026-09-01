from beatroot.trusted.tags import derive_recipe_tags


def test_allergen_propagates_from_ingredient_to_recipe():
    ingredients = {
        "ing_peanut_oil": {"allergen_tags": ["peanut"], "religious_tags": [], "dietary_tags": []},
        "ing_rice": {"allergen_tags": [], "religious_tags": [], "dietary_tags": ["vegan"]},
    }
    recipe = {
        "ingredients": [
            {"ingredient_id": "ing_peanut_oil", "grams": 10},
            {"ingredient_id": "ing_rice", "grams": 100},
        ]
    }
    assert "peanut" in derive_recipe_tags(recipe, ingredients)


def test_religious_tag_propagates():
    ingredients = {
        "ing_onion": {
            "allergen_tags": [],
            "religious_tags": ["root_vegetable"],
            "dietary_tags": [],
        },
    }
    recipe = {"ingredients": [{"ingredient_id": "ing_onion", "grams": 50}]}
    assert "root_vegetable" in derive_recipe_tags(recipe, ingredients)


def test_dietary_tag_is_intersection_not_union():
    """A recipe is vegan only if EVERY ingredient is vegan."""
    ingredients = {
        "ing_rice": {
            "allergen_tags": [],
            "religious_tags": [],
            "dietary_tags": ["vegan", "vegetarian"],
        },
        "ing_paneer": {
            "allergen_tags": ["dairy"],
            "religious_tags": [],
            "dietary_tags": ["vegetarian"],
        },
    }
    recipe = {
        "ingredients": [
            {"ingredient_id": "ing_rice", "grams": 100},
            {"ingredient_id": "ing_paneer", "grams": 100},
        ]
    }
    tags = derive_recipe_tags(recipe, ingredients)
    assert "vegetarian" in tags
    assert "vegan" not in tags


def test_unknown_ingredient_raises():
    import pytest

    from beatroot.trusted.tags import UnknownIngredientError

    with pytest.raises(UnknownIngredientError):
        derive_recipe_tags({"ingredients": [{"ingredient_id": "nope", "grams": 1}]}, {})
