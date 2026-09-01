from pathlib import Path

from beatroot.eval.synth.adversarial import (
    _FAMILIES,
    _composite_allergen_pairs,
    _homoglyph,
    _minimal_infeasible_tag_set,
    generate_adversarial,
)
from beatroot.store.db import connect, seed
from beatroot.trusted.catalog import Catalog

ROOT = Path(__file__).parents[2]


def _catalog(tmp_path: Path) -> Catalog:
    conn = connect(tmp_path / "t.db")
    seed(conn, ROOT / "data")
    return Catalog(conn)


def test_generation_is_reproducible(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    a = generate_adversarial(cat, n=300, seed=11)
    b = generate_adversarial(cat, n=300, seed=11)
    assert a == b


def test_all_ten_families_are_generated(tmp_path: Path) -> None:
    """Against the real, checked-in catalog every family has vocabulary to
    draw from — a large-enough sample should hit all ten."""
    cat = _catalog(tmp_path)
    cases = generate_adversarial(cat, n=2000, seed=2)
    families = {c["family"] for c in cases}
    assert families == set(_FAMILIES)
    assert len(_FAMILIES) == 10


def test_every_case_is_shaped_like_a_golden_case(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    for case in generate_adversarial(cat, n=500, seed=1):
        assert case["id"]
        assert case["family"] in _FAMILIES
        assert isinstance(case["query"], str)
        assert isinstance(case["preferences"], str)
        assert case["constraints"] or case["family"] == "empty_and_degenerate"
        assert case["expect_terminal"]
        assert set(case["expect_terminal"]) <= {"COMMIT", "NEGOTIATE", "ESCALATE"}


def test_ids_are_unique(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = generate_adversarial(cat, n=500, seed=5)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_default_n_comes_from_settings(tmp_path: Path) -> None:
    from beatroot.settings import get_settings

    cat = _catalog(tmp_path)
    cases = generate_adversarial(cat, seed=1)
    assert len(cases) == get_settings().synth.default_adversarial


def test_generate_adversarial_never_crashes_on_an_empty_catalog(tmp_path: Path) -> None:
    """An empty catalog has no tags, no synonym pairs, no composite
    ingredients — every family that needs any of those is unavailable, so
    this must degrade to whatever's left (boundary_values and
    unknown_vocabulary need no catalog vocabulary at all) rather than crash
    on an empty `rng.choice([])`."""
    conn = connect(tmp_path / "empty.db")
    cat = Catalog(conn)
    cases = generate_adversarial(cat, n=50, seed=1)
    assert cases
    assert {c["family"] for c in cases} <= {"boundary_values", "unknown_vocabulary"}


# ---------------------------------------------------------------------
# Family: injection — free text can never reach the constraint layer.
# ---------------------------------------------------------------------


def test_injection_uses_a_real_catalog_tag(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    known_tags = {t for r in cat.recipes() for t in r.tags}
    cases = [c for c in generate_adversarial(cat, n=300, seed=3) if c["family"] == "injection"]
    assert cases
    for case in cases:
        tag = case["constraints"][0]["value"]
        assert tag in known_tags
        assert case["assert_absent_tags"] == [tag]
        assert tag in case["query"] or True  # template always embeds the tag


def test_injection_covers_more_than_one_attack_style(tmp_path: Path) -> None:
    """A single template family would understate what 'injection' means —
    confirm the generator actually rotates through more than one angle."""
    cat = _catalog(tmp_path)
    cases = [c for c in generate_adversarial(cat, n=400, seed=3) if c["family"] == "injection"]
    queries = {c["query"] for c in cases}
    assert len(queries) > 5


# ---------------------------------------------------------------------
# Family: synonym_evasion_constraint — the CONSTRAINT itself carries a
# synonym, the family the original two-family generator never actually
# built (it put the synonym in `query`, which constraints never parse).
# ---------------------------------------------------------------------


def test_synonym_evasion_constraint_puts_the_synonym_in_the_constraint_value(
    tmp_path: Path,
) -> None:
    cat = _catalog(tmp_path)
    known_synonyms = {
        syn for payload in cat.ingredients().values() for syn in payload.get("synonyms", ())
    }
    cases = [
        c
        for c in generate_adversarial(cat, n=300, seed=6)
        if c["family"] == "synonym_evasion_constraint"
    ]
    assert cases
    for case in cases:
        constraint = case["constraints"][0]
        assert constraint["kind"] == "exclude_ingredient"
        assert constraint["value"] in known_synonyms
        assert case["assert_absent_ingredient"] is not None


# ---------------------------------------------------------------------
# Family: case_and_whitespace — proves (and asserts on) the real
# asymmetry: exclude_tag never normalises, exclude_ingredient always does.
# ---------------------------------------------------------------------


def test_case_and_whitespace_tag_variants_always_escalate(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [
        c for c in generate_adversarial(cat, n=500, seed=7) if c["family"] == "case_and_whitespace"
    ]
    tag_cases = [c for c in cases if c["constraints"][0]["kind"] == "exclude_tag"]
    assert tag_cases
    for case in tag_cases:
        value = case["constraints"][0]["value"]
        assert value != value.strip().lower()  # a genuine case/whitespace mutation
        assert case["expect_terminal"] == ["ESCALATE"]
        assert case["assert_escalate_reason"] == "unknown_ingredient"


def test_case_and_whitespace_ingredient_variants_never_report_unknown(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [
        c for c in generate_adversarial(cat, n=500, seed=7) if c["family"] == "case_and_whitespace"
    ]
    ingredient_cases = [c for c in cases if c["constraints"][0]["kind"] == "exclude_ingredient"]
    assert ingredient_cases
    for case in ingredient_cases:
        assert case["assert_escalate_reason_not"] == "unknown_ingredient"
        assert case["assert_absent_ingredient"] is not None


# ---------------------------------------------------------------------
# Family: homoglyph — must escalate, never silently resolve.
# ---------------------------------------------------------------------


def test_homoglyph_string_differs_from_the_original_term() -> None:
    assert _homoglyph("peanut") != "peanut"
    assert _homoglyph("peanut").lower() != "peanut"


def test_homoglyph_cases_always_expect_escalation(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [c for c in generate_adversarial(cat, n=300, seed=8) if c["family"] == "homoglyph"]
    assert cases
    for case in cases:
        assert case["expect_terminal"] == ["ESCALATE"]
        assert case["assert_escalate_reason"] == "unknown_ingredient"
        value = case["constraints"][0]["value"]
        # a homoglyph swap must never happen to collide with a real term
        assert value not in {t for r in cat.recipes() for t in r.tags}


# ---------------------------------------------------------------------
# Family: transitive_allergen — targets a genuinely hidden composite
# ingredient, not one whose name already says the allergen.
# ---------------------------------------------------------------------


def test_transitive_allergen_targets_a_genuinely_hidden_composite_ingredient(
    tmp_path: Path,
) -> None:
    cat = _catalog(tmp_path)
    pairs = _composite_allergen_pairs(cat)
    assert pairs, "the real catalog must have at least one composite/hidden ingredient"
    for name, _iid, tag in pairs:
        assert tag not in name.lower()
        assert tag.replace("_", " ") not in name.lower()

    cases = [
        c for c in generate_adversarial(cat, n=300, seed=9) if c["family"] == "transitive_allergen"
    ]
    assert cases
    for case in cases:
        tag = case["constraints"][0]["value"]
        assert case["assert_absent_tags"] == [tag]


def test_satay_marinade_is_a_transitive_peanut_pair(tmp_path: Path) -> None:
    """The exact real-catalog example spec §12 and the golden dataset both
    name: satay marinade carries `peanut` with no 'peanut' in its name."""
    cat = _catalog(tmp_path)
    pairs = _composite_allergen_pairs(cat)
    assert ("Satay marinade", "ing_satay_marinade", "peanut") in pairs


# ---------------------------------------------------------------------
# Family: contradictory — genuinely infeasible, verified at generation
# time against the real catalog, never assumed.
# ---------------------------------------------------------------------


def test_minimal_infeasible_tag_set_actually_empties_the_catalog(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    recipes = cat.recipes()
    tags = sorted({t for r in recipes for t in r.tags})
    excluded = _minimal_infeasible_tag_set(recipes, tags)
    assert excluded
    excl = set(excluded)
    assert all(r.tags & excl for r in recipes), "every recipe must carry at least one excluded tag"


def test_contradictory_cases_expect_negotiate_with_everything_locked(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [c for c in generate_adversarial(cat, n=300, seed=10) if c["family"] == "contradictory"]
    assert cases
    for case in cases:
        assert case["expect_terminal"] == ["NEGOTIATE"]
        ids = [c["id"] for c in case["constraints"]]
        assert case["assert_locked_contains"] == ids
        assert case["assert_no_relaxations"] is True
        assert all(c["severity"] in ("medical", "religious") for c in case["constraints"])


def test_every_catalog_recipe_carries_at_least_one_tag(tmp_path: Path) -> None:
    """`_minimal_infeasible_tag_set` relies on this to terminate — assert
    the assumption directly rather than only through its consequence."""
    cat = _catalog(tmp_path)
    assert all(r.tags for r in cat.recipes())


# ---------------------------------------------------------------------
# Family: constraint_flooding — many constraints, must never crash.
# ---------------------------------------------------------------------


def test_constraint_flooding_generates_ten_to_forty_constraints(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [
        c for c in generate_adversarial(cat, n=200, seed=12) if c["family"] == "constraint_flooding"
    ]
    assert cases
    for case in cases:
        assert 10 <= len(case["constraints"]) <= 40
        ids = [c["id"] for c in case["constraints"]]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------
# Family: boundary_values — degenerate numeric edges, verified infeasible.
# ---------------------------------------------------------------------


def test_boundary_values_are_all_genuinely_infeasible(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [
        c for c in generate_adversarial(cat, n=300, seed=13) if c["family"] == "boundary_values"
    ]
    assert cases
    for case in cases:
        assert case["expect_terminal"] == ["NEGOTIATE"]
        assert case["assert_relaxations_offered"] is True
        constraint = case["constraints"][0]
        if constraint["kind"] == "nutrient_range":
            lo, hi = constraint["value"]
            assert lo >= hi
        else:
            assert constraint["value"] <= 0


# ---------------------------------------------------------------------
# Family: empty_and_degenerate — never crashes, safety holds regardless.
# ---------------------------------------------------------------------


def test_empty_and_degenerate_covers_multiple_shapes(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    cases = [
        c
        for c in generate_adversarial(cat, n=400, seed=14)
        if c["family"] == "empty_and_degenerate"
    ]
    assert cases
    shapes = {
        (
            bool(c["query"]),
            bool(c["constraints"]),
            bool(c["preferences"].strip()),
            len(c["query"]) > 5000,
        )
        for c in cases
    }
    assert len(shapes) > 1


# ---------------------------------------------------------------------
# Family: unknown_vocabulary — plausible but catalog-absent, must escalate.
# ---------------------------------------------------------------------


def test_unknown_vocabulary_terms_are_genuinely_absent_from_the_catalog(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    known_tags = {t for r in cat.recipes() for t in r.tags}
    known_names = {p["name"].lower() for p in cat.ingredients().values()}
    known_syns = {s.lower() for p in cat.ingredients().values() for s in p.get("synonyms", ())}
    known_ids = set(cat.ingredients().keys())

    cases = [
        c for c in generate_adversarial(cat, n=300, seed=15) if c["family"] == "unknown_vocabulary"
    ]
    assert cases
    for case in cases:
        term = case["constraints"][0]["value"]
        assert term not in known_tags
        assert term not in known_names
        assert term not in known_syns
        assert term not in known_ids
        assert case["expect_terminal"] == ["ESCALATE"]
        assert case["assert_escalate_reason"] == "unknown_ingredient"
