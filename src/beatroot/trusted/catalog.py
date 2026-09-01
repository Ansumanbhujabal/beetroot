"""Catalog loading and lookup: ingredients and recipes backed by the trusted SQLite store."""

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from beatroot.contracts.nutrition import NutritionFacts


class Recipe(BaseModel):
    """Read model. `tags` are DERIVED at seed time, never hand-set."""

    id: str
    name: str
    cuisine: str | None = None
    prep_minutes: int | None = None
    tags: set[str] = Field(default_factory=set)
    ingredient_ids: list[str] = Field(default_factory=list)

    # Populated lazily by Catalog.hydrate(); None means "not yet computed",
    # which the constraint checker reports as `uncheckable` rather than a pass.
    nutrition: NutritionFacts | None = None
    cost_inr: float | None = None


class Catalog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._recipes: list[Recipe] | None = None
        self._payloads: dict[str, dict[str, Any]] = {}
        self._ingredients: dict[str, dict[str, Any]] = {}

    def _load(self) -> None:
        if self._recipes is not None:
            return
        self._ingredients = {
            r["id"]: json.loads(r["payload"])
            for r in self.conn.execute("SELECT id, payload FROM ingredients")
        }
        out: list[Recipe] = []
        for row in self.conn.execute("SELECT * FROM recipes ORDER BY id"):
            payload = json.loads(row["payload"])
            self._payloads[row["id"]] = payload
            out.append(
                Recipe(
                    id=row["id"],
                    name=row["name"],
                    cuisine=row["cuisine"],
                    prep_minutes=row["prep_minutes"],
                    tags=set(json.loads(row["tags"])),
                    ingredient_ids=[i["ingredient_id"] for i in payload["ingredients"]],
                )
            )
        self._recipes = out

    def recipes(self) -> list[Recipe]:
        self._load()
        if self._recipes is None:
            # _load() always populates _recipes before returning; this
            # documents that postcondition (and narrows the type for
            # mypy, which cannot see across the method call on its own)
            # without relying on `assert`, which `python -O` strips.
            raise RuntimeError("Catalog._load() did not populate _recipes")
        return self._recipes

    def recipe(self, recipe_id: str) -> Recipe | None:
        recipes = self.recipes()
        return next((r for r in recipes if r.id == recipe_id), None)

    def recipe_payload(self, recipe_id: str) -> dict[str, Any] | None:
        self._load()
        return self._payloads.get(recipe_id)

    def ingredient_payload(self, ingredient_id: str) -> dict[str, Any] | None:
        self._load()
        return self._ingredients.get(ingredient_id)

    def ingredients(self) -> dict[str, dict[str, Any]]:
        """Every ingredient payload keyed by id, as loaded from the catalog —
        the full known-ingredient vocabulary (ids, names, synonyms). Powers
        `t0_invariants.vocabulary.unknown_vocabulary`; returns a copy so a
        caller mutating the result never corrupts the Catalog's own cache."""
        self._load()
        return dict(self._ingredients)

    def hydrate(self, recipe: Recipe) -> Recipe:
        """Attach computed nutrition and cost so nutrient_range and budget_max
        constraints become checkable. Called by the agent before scoring."""
        from beatroot.t0_invariants.nutrition_math import compute, recipe_cost_inr

        payload = self.recipe_payload(recipe.id)
        if payload is None:
            return recipe
        recipe.nutrition = compute(payload, self)
        recipe.cost_inr = recipe_cost_inr(payload, self)
        return recipe

    def hydrated(self) -> list[Recipe]:
        return [self.hydrate(r) for r in self.recipes()]
