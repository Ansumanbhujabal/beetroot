from pathlib import Path

import pytest

from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.store.db import connect, seed
from beatroot.t0_invariants.vocabulary import unknown_vocabulary
from beatroot.trusted.catalog import Catalog

DATA = Path(__file__).parents[2] / "data"


@pytest.fixture
def catalog(tmp_path) -> Catalog:
    conn = connect(tmp_path / "vocab.db")
    seed(conn, DATA)
    return Catalog(conn)


def _cs(*constraints: Constraint) -> ConstraintSet:
    return ConstraintSet(profile_id="p1", constraints=list(constraints))


def test_known_tag_is_not_flagged(catalog: Catalog) -> None:
    cs = _cs(Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    assert unknown_vocabulary(cs, catalog) == []


def test_unknown_tag_is_flagged(catalog: Catalog) -> None:
    cs = _cs(Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="sulfite"))
    unknown = unknown_vocabulary(cs, catalog)
    assert [c.id for c in unknown] == ["med1"]


def test_unknown_ingredient_id_is_flagged(catalog: Catalog) -> None:
    cs = _cs(
        Constraint(
            id="med1", kind="exclude_ingredient", severity=Severity.MEDICAL, value="ing_durian"
        )
    )
    unknown = unknown_vocabulary(cs, catalog)
    assert [c.id for c in unknown] == ["med1"]


def test_known_ingredient_id_is_not_flagged(catalog: Catalog) -> None:
    cs = _cs(
        Constraint(
            id="med1", kind="exclude_ingredient", severity=Severity.MEDICAL, value="ing_peanuts"
        )
    )
    assert unknown_vocabulary(cs, catalog) == []


def test_synonym_resolves_and_is_not_flagged(catalog: Catalog) -> None:
    """ "groundnut oil" is a real synonym of ing_peanut_oil — canonicalisation
    must run before the unknown check, or a legitimate synonym-evasion
    constraint value would be wrongly reported as unverifiable."""
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="groundnut oil",
        )
    )
    assert unknown_vocabulary(cs, catalog) == []


def test_curd_synonym_resolves(catalog: Catalog) -> None:
    cs = _cs(
        Constraint(id="med1", kind="exclude_ingredient", severity=Severity.MEDICAL, value="curd")
    )
    assert unknown_vocabulary(cs, catalog) == []


def test_mix_of_known_and_unknown_flags_only_the_unknown(catalog: Catalog) -> None:
    cs = _cs(
        Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        Constraint(id="med2", kind="exclude_tag", severity=Severity.MEDICAL, value="kiwi"),
        Constraint(id="rel1", kind="exclude_tag", severity=Severity.RELIGIOUS, value="allium"),
    )
    unknown = unknown_vocabulary(cs, catalog)
    assert [c.id for c in unknown] == ["med2"]


def test_unknown_preference_constraint_is_still_flagged(catalog: Catalog) -> None:
    """Severity affects the MESSAGE, never the decision — an unverifiable
    PREFERENCE must escalate exactly like a MEDICAL one, or the fix just
    reintroduces the same silent-pass hole one severity down."""
    cs = _cs(
        Constraint(
            id="p1", kind="exclude_tag", severity=Severity.PREFERENCE, value="not_a_real_tag"
        )
    )
    unknown = unknown_vocabulary(cs, catalog)
    assert [c.id for c in unknown] == ["p1"]


def test_non_vocabulary_constraint_kinds_are_never_flagged(catalog: Catalog) -> None:
    cs = _cs(
        Constraint(id="p1", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=15),
        Constraint(id="p2", kind="budget_max", severity=Severity.PREFERENCE, value=100),
        Constraint(
            id="goal1",
            kind="nutrient_range",
            severity=Severity.GOAL,
            value=(0.0, 999.0),
            nutrient="kcal",
        ),
    )
    assert unknown_vocabulary(cs, catalog) == []


def test_empty_constraint_set_has_nothing_unknown(catalog: Catalog) -> None:
    assert unknown_vocabulary(_cs(), catalog) == []
