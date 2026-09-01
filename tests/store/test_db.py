"""Tests for seed-time validation of ingredient data."""

import tempfile
from pathlib import Path

import pytest
import yaml

from beatroot.store.db import connect, seed


def test_seed_rejects_ingredient_with_missing_nutrition_fields():
    """Seed must fail loudly if an ingredient's per_100g is incomplete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create incomplete ingredient (missing protein_g)
        incomplete_ing = {
            "id": "incomplete_spice",
            "name": "Missing Protein Spice",
            "per_100g": {
                "kcal": 100,
                # protein_g is missing!
                "carbs_g": 5,
                "fat_g": 1,
                "sodium_mg": 200,
                "fibre_g": 0,
            },
        }

        (data_dir / "ingredients.yaml").write_text(yaml.dump([incomplete_ing]))
        (data_dir / "recipes.yaml").write_text(yaml.dump([]))

        with pytest.raises(ValueError) as exc_info:
            conn = connect(":memory:")
            seed(conn, data_dir)

        assert "incomplete_spice" in str(exc_info.value)
        assert "protein_g" in str(exc_info.value)


def test_seed_accepts_ingredient_with_all_nutrition_fields():
    """Seed must accept ingredients with complete per_100g data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        complete_ing = {
            "id": "complete_spice",
            "name": "Complete Spice",
            "per_100g": {
                "kcal": 100,
                "protein_g": 2,
                "carbs_g": 5,
                "fat_g": 1,
                "sodium_mg": 200,
                "fibre_g": 0,
            },
        }

        (data_dir / "ingredients.yaml").write_text(yaml.dump([complete_ing]))
        (data_dir / "recipes.yaml").write_text(yaml.dump([]))

        conn = connect(":memory:")
        seed(conn, data_dir)  # Should not raise

        # Verify it was inserted
        row = conn.execute("SELECT * FROM ingredients WHERE id = ?", ("complete_spice",)).fetchone()
        assert row is not None


def test_seed_rejects_ingredient_with_null_per100_value():
    """Seed must fail loudly if a field has a null value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        ing_with_null = {
            "id": "null_field_ing",
            "name": "Null Field Ingredient",
            "per_100g": {
                "kcal": 100,
                "protein_g": None,  # null value
                "carbs_g": 5,
                "fat_g": 1,
                "sodium_mg": 200,
                "fibre_g": 0,
            },
        }

        (data_dir / "ingredients.yaml").write_text(yaml.dump([ing_with_null]))
        (data_dir / "recipes.yaml").write_text(yaml.dump([]))

        with pytest.raises(ValueError) as exc_info:
            conn = connect(":memory:")
            seed(conn, data_dir)

        assert "null_field_ing" in str(exc_info.value)
        assert "unusable" in str(exc_info.value)
        assert "protein_g" in str(exc_info.value)


def test_seed_rejects_ingredient_with_string_per100_value():
    """Seed must fail loudly if a field has a string value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        ing_with_string = {
            "id": "string_field_ing",
            "name": "String Field Ingredient",
            "per_100g": {
                "kcal": 100,
                "protein_g": "10",  # string value
                "carbs_g": 5,
                "fat_g": 1,
                "sodium_mg": 200,
                "fibre_g": 0,
            },
        }

        (data_dir / "ingredients.yaml").write_text(yaml.dump([ing_with_string]))
        (data_dir / "recipes.yaml").write_text(yaml.dump([]))

        with pytest.raises(ValueError) as exc_info:
            conn = connect(":memory:")
            seed(conn, data_dir)

        assert "string_field_ing" in str(exc_info.value)
        assert "unusable" in str(exc_info.value)


def test_seed_rejects_ingredient_with_negative_per100_value():
    """Seed must fail loudly if a field has a negative value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        ing_with_neg = {
            "id": "negative_field_ing",
            "name": "Negative Field Ingredient",
            "per_100g": {
                "kcal": 100,
                "protein_g": -5,  # negative value
                "carbs_g": 5,
                "fat_g": 1,
                "sodium_mg": 200,
                "fibre_g": 0,
            },
        }

        (data_dir / "ingredients.yaml").write_text(yaml.dump([ing_with_neg]))
        (data_dir / "recipes.yaml").write_text(yaml.dump([]))

        with pytest.raises(ValueError) as exc_info:
            conn = connect(":memory:")
            seed(conn, data_dir)

        assert "negative_field_ing" in str(exc_info.value)
        assert "unusable" in str(exc_info.value)


def test_seed_rejects_ingredient_with_infinity_per100_value():
    """Seed must fail loudly if a field has an infinity value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        ing_with_inf = {
            "id": "inf_field_ing",
            "name": "Infinity Field Ingredient",
            "per_100g": {
                "kcal": 100,
                "protein_g": float("inf"),  # infinity value
                "carbs_g": 5,
                "fat_g": 1,
                "sodium_mg": 200,
                "fibre_g": 0,
            },
        }

        (data_dir / "ingredients.yaml").write_text(yaml.dump([ing_with_inf]))
        (data_dir / "recipes.yaml").write_text(yaml.dump([]))

        with pytest.raises(ValueError) as exc_info:
            conn = connect(":memory:")
            seed(conn, data_dir)

        assert "inf_field_ing" in str(exc_info.value)
        assert "unusable" in str(exc_info.value)
