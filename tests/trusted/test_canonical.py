from beatroot.trusted.canonical import (
    build_synonym_index,
    canonicalise,
    default_known_ingredient_ids,
    resolve_ingredient_id,
)


def test_synonym_resolves_to_canonical_id():
    idx = build_synonym_index(
        [
            {"id": "ing_peanut_oil", "name": "Peanut oil", "synonyms": ["groundnut oil"]},
        ]
    )
    assert canonicalise("groundnut oil", idx) == "ing_peanut_oil"
    assert canonicalise("Groundnut Oil", idx) == "ing_peanut_oil"
    assert canonicalise("  peanut oil ", idx) == "ing_peanut_oil"


def test_unknown_term_returns_none():
    idx = build_synonym_index([])
    assert canonicalise("dragonfruit", idx) is None


# ---- resolve_ingredient_id: reads the real repo data/ingredients.yaml ---


def test_resolve_ingredient_id_synonym():
    assert resolve_ingredient_id("groundnut oil") == "ing_peanut_oil"
    assert resolve_ingredient_id("curd") == "ing_yoghurt"
    assert resolve_ingredient_id("maida") == "ing_refined_flour"


def test_resolve_ingredient_id_already_canonical():
    """A value that is already a canonical id must still work — it need
    not appear as a `name`/`synonyms` entry to resolve to itself."""
    assert resolve_ingredient_id("ing_peanut_oil") == "ing_peanut_oil"


def test_resolve_ingredient_id_unknown_returns_none():
    assert resolve_ingredient_id("not_a_real_ingredient_anywhere") is None


def test_default_known_ingredient_ids_is_nonempty_and_matches_catalog():
    ids = default_known_ingredient_ids()
    assert "ing_peanut_oil" in ids
    assert "ing_yoghurt" in ids
    assert "ing_refined_flour" in ids
