"""Cross-check: `eval.verifiers.vocabulary` (from-scratch) agrees with
`t0_invariants.vocabulary.unknown_vocabulary` (production) — and can be
shown to DISAGREE once production is broken. Spec §12.

`eval.runners.system._oracle_has_valid_meal` decides the A5
(`escalation_correctness`) verdict. It used to call production's
`unknown_vocabulary` directly, which meant a bug there would move the
agent's actual answer and the oracle's answer together, so A5 could keep
reading 1.000 with nothing left in the eval to disagree with it — the same
tautology `tests/eval/test_oracle.py` already found and fixed once for
`check_recipe`, reintroduced one axis over. This file proves the fix: the
independent implementation agrees with production on real data, and it can
be made to disagree by monkeypatching production broken — never by editing
source, per the review note this test exists to satisfy.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.eval.runners import system as system_mod
from beatroot.eval.synth.profiles import generate_profiles
from beatroot.eval.verifiers import vocabulary as vocabulary_oracle
from beatroot.store.db import connect, seed
from beatroot.t0_invariants import vocabulary as production_vocabulary
from beatroot.trusted.catalog import Catalog

ROOT = Path(__file__).parents[2]


def _catalog(tmp_path: Path) -> Catalog:
    conn = connect(tmp_path / "t.db")
    seed(conn, ROOT / "data")
    return Catalog(conn)


def test_vocabulary_oracle_agrees_with_production_on_generated_profiles(tmp_path: Path) -> None:
    """`generate_profiles` only ever draws constraint values from the
    catalog's own vocabulary, so this is agreement on the "everything is
    known" branch — see the deliberately-unknown test below for the branch
    that actually matters."""
    cat = _catalog(tmp_path)
    for case in generate_profiles(cat, n=100, seed=11):
        oracle_unknown = {
            c.id for c in vocabulary_oracle.unknown_vocabulary(case.constraint_set, cat)
        }
        production_unknown = {
            c.id for c in production_vocabulary.unknown_vocabulary(case.constraint_set, cat)
        }
        assert oracle_unknown == production_unknown == set()


def test_vocabulary_oracle_agrees_on_deliberately_unknown_constraints(tmp_path: Path) -> None:
    """`generate_profiles` never draws a genuinely unknown tag/ingredient
    (it only ever samples the catalog's own vocabulary), so the test above
    cannot exercise the actual unknown-vocabulary branch at all. This
    constructs one directly, for both an unknown tag and an unknown
    ingredient."""
    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="sulfite"),
            Constraint(
                id="c2",
                kind="exclude_ingredient",
                severity=Severity.MEDICAL,
                value="not_a_real_ingredient_anywhere",
            ),
            # A real, known tag must NOT be flagged.
            Constraint(id="c3", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        ],
    )
    oracle_unknown = {c.id for c in vocabulary_oracle.unknown_vocabulary(cs, cat)}
    production_unknown = {c.id for c in production_vocabulary.unknown_vocabulary(cs, cat)}
    assert oracle_unknown == production_unknown == {"c1", "c2"}


def test_vocabulary_oracle_resolves_known_synonyms_like_production(tmp_path: Path) -> None:
    """ "groundnut oil" is a real synonym for `ing_peanut_oil` in this
    project's own `data/ingredients.yaml` — both implementations must
    resolve it, not flag it as unverifiable."""
    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(
                id="c1",
                kind="exclude_ingredient",
                severity=Severity.MEDICAL,
                value="groundnut oil",
            )
        ],
    )
    assert vocabulary_oracle.unknown_vocabulary(cs, cat) == []
    assert production_vocabulary.unknown_vocabulary(cs, cat) == []


def test_vocabulary_oracle_cross_check_detects_a_broken_production_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the cross-check above is real, not decorative: breaks
    production's `unknown_vocabulary` by monkeypatching it to always claim
    nothing is unknown — the exact silent-pass A5 was originally built to
    catch (see README.md / CUT_LIST.md's account of that finding) — then
    shows the independent oracle disagrees."""
    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="sulfite"),
        ],
    )

    monkeypatch.setattr(production_vocabulary, "unknown_vocabulary", lambda cs, catalog: [])

    oracle_unknown = {c.id for c in vocabulary_oracle.unknown_vocabulary(cs, cat)}
    production_unknown = {c.id for c in production_vocabulary.unknown_vocabulary(cs, cat)}
    assert oracle_unknown != production_unknown, (
        "the independent oracle should disagree once production's "
        "unknown_vocabulary is broken — if it doesn't, the oracle is not "
        "actually independent of the predicate it exists to check"
    )


def test_system_oracle_uses_the_independent_vocabulary_check_not_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual regression this task closes: `eval.runners.system.
    _oracle_has_valid_meal` decides A5's verdict and must not call
    production's `unknown_vocabulary` to do it. Break production (make it
    claim nothing is ever unknown) and confirm the SYSTEM ORACLE still
    reports no valid meal for a profile with genuinely unknown
    vocabulary — if it secretly called production, this would flip to
    True and A5 would silently stop meaning anything, exactly the
    tautology this module exists to close."""
    cat = _catalog(tmp_path)
    cs = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="sulfite"),
        ],
    )
    fake_agent = SimpleNamespace(deps=SimpleNamespace(catalog=cat))

    monkeypatch.setattr(production_vocabulary, "unknown_vocabulary", lambda cs, catalog: [])

    assert system_mod._oracle_has_valid_meal(fake_agent, cs) is False  # type: ignore[arg-type]
