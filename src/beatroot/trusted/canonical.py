"""Synonym-to-canonical-id resolution for ingredient names in the trusted catalog."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Same repo-relative path `container.DATA_DIR` and `store.db.seed` resolve
# to, computed independently here rather than imported from `container` —
# `trusted/` sits BELOW the composition root in the layering this codebase
# enforces (see container.py's module docstring); reaching upward for a
# path constant would invert that, even though nothing here is actually
# circular at import time.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def build_synonym_index(ingredients: list[dict[str, Any]]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for ing in ingredients:
        idx[_norm(ing["name"])] = ing["id"]
        for syn in ing.get("synonyms", ()):
            idx[_norm(syn)] = ing["id"]
    return idx


def _norm(term: str) -> str:
    return " ".join(term.strip().lower().split())


def canonicalise(term: str, index: dict[str, str]) -> str | None:
    return index.get(_norm(term))


@lru_cache(maxsize=1)
def _default_ingredients() -> tuple[dict[str, Any], ...]:
    """Every ingredient row from the repo's own `data/ingredients.yaml` —
    the exact file `store.db.seed` loads into the catalog DB. Cached
    process-wide: this is static repo content, not per-request or
    per-Container state, so there is nothing for a test to swap out from
    under it (see `resolve_ingredient_id`'s docstring for who calls this
    and why it does not take a `Catalog`).
    """
    raw = yaml.safe_load((_DATA_DIR / "ingredients.yaml").read_text())
    return tuple(raw or ())


def default_known_ingredient_ids() -> frozenset[str]:
    """Every canonical ingredient id the trusted catalog defines."""
    return frozenset(ing["id"] for ing in _default_ingredients())


def default_synonym_index() -> dict[str, str]:
    """The synonym index over the repo's own ingredient data — built fresh
    from the cached rows (`build_synonym_index` itself is cheap; the yaml
    read/parse is what `_default_ingredients` caches)."""
    return build_synonym_index(list(_default_ingredients()))


def resolve_ingredient_id(value: str) -> str | None:
    """Resolve a user- or constraint-supplied ingredient term to its
    catalog canonical id.

    Three cases: `value` already IS a canonical id (e.g. `"ing_peanut_oil"`)
    and is returned unchanged; `value` is a known display name or synonym
    (e.g. `"groundnut oil"`) and is canonicalised via `canonicalise`; or
    `value` names nothing the catalog has ever heard of, in which case this
    returns `None` — the caller's job to treat that as "uncheckable", never
    as "no match, so nothing is excluded".

    Exists specifically so `t0_invariants.constraints._exclude_ingredient`
    and `eval.verifiers.hard_constraint.verify` — two independent
    implementations that both compare a `Constraint.value` against
    `Recipe.ingredient_ids` — canonicalise through the SAME resolution
    `t0_invariants.vocabulary.unknown_vocabulary` already uses to validate
    that same value, rather than each inventing (or, as happened once
    already, both omitting) their own. Both of those functions take a bare
    `(recipe, constraint)`/`(recipe, constraint_set)` — no `Catalog` — so
    this reads the repo's own `data/ingredients.yaml` directly (cached,
    process-wide) rather than requiring a `Catalog` neither call site's
    fixed evaluator signature has room to thread through.
    """
    if value in default_known_ingredient_ids():
        return value
    return canonicalise(value, default_synonym_index())
