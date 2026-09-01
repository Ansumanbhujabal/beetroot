"""Degenerate-input edge cases that must never crash the pipeline. Spec §12.

Every test here either runs the real `MealPlanningAgent` end-to-end
(offline, via `build_container`) or exercises a pure `t0_invariants`
function directly — and every assertion is "a sensible terminal or a clean
typed error, never a traceback escaping." A `ValidationError` raised by
`Constraint`/`ConstraintSet` at CONSTRUCTION time (a genuinely malformed
shape Pydantic itself refuses) counts as a clean typed error; anything an
unhandled exception raises further downstream, inside evaluation or the
graph itself, does not.

This is deliberately a separate module from `eval.synth.adversarial`'s
generated families: those are large-scale and probabilistic (the pass
RATE is the signal); these are small, hand-picked, and exhaustive over one
specific degenerate shape each (duplicate ids, a wrong-type value, etc.) —
the kind of case worth pinning down by hand rather than leaving to chance
in a random generator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from beatroot.container import build_container
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Escalation, Negotiation, Recommendation
from beatroot.trusted.catalog import Recipe


def _run(agent, cs: ConstraintSet, query: str = "dinner", preferences: str = ""):
    """Drive one profile through the real agent, auto-approving a
    grey-band pause exactly as every eval runner in this repo does, and
    return `(terminal, result)`. Never catches an exception — a crash here
    must fail the test loudly, not be swallowed."""
    result = agent.run(cs, query=query, preferences=preferences)
    if result is None:
        thread_id = agent.last_thread_id
        if thread_id is not None:
            result = agent.resume(thread_id, approved=True)
    terminal = agent.trace[-1] if agent.trace else "NONE"
    return terminal, result


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory):
    container = build_container(tmp_path_factory.mktemp("edge") / "edge.db")
    try:
        yield container.agent
    finally:
        container.close()


# ---------------------------------------------------------------------
# Empty catalog
# ---------------------------------------------------------------------


def test_empty_catalog_agent_never_crashes(tmp_path: Path) -> None:
    """`build_container(seed_data=False)` gives a 0-recipe catalog — the
    vector store, tag index, and every graph node must all tolerate that,
    not just the ones a unit test happens to isolate."""
    container = build_container(tmp_path / "empty.db", seed_data=False)
    try:
        assert container.catalog.recipes() == []
        cs = ConstraintSet(
            profile_id="empty",
            constraints=[
                Constraint(id="med1", kind="exclude_tag", severity="medical", value="peanut")
            ],
        )
        terminal, result = _run(container.agent, cs)
        assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
        assert not isinstance(result, Recommendation)  # nothing to commit from an empty catalog
    finally:
        container.close()


def test_empty_catalog_with_no_constraints_reaches_negotiate(tmp_path: Path) -> None:
    container = build_container(tmp_path / "empty2.db", seed_data=False)
    try:
        cs = ConstraintSet(profile_id="empty-noconstraints", constraints=[])
        terminal, result = _run(container.agent, cs)
        assert terminal in ("NEGOTIATE", "ESCALATE")
        assert not isinstance(result, Recommendation)
    finally:
        container.close()


def test_check_recipe_and_assess_tolerate_an_empty_recipe_list() -> None:
    from beatroot.t0_invariants.constraints import check_recipe
    from beatroot.t0_invariants.feasibility import assess

    cs = ConstraintSet(
        profile_id="p",
        constraints=[Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut")],
    )
    feasibility = assess(cs, recipes=[], index=None)
    assert feasibility.feasible is False
    assert feasibility.surviving == []
    # check_recipe on a recipe with no tags/ingredients at all
    empty_recipe = Recipe(id="r0", name="Nothing", tags=set(), ingredient_ids=[])
    result = check_recipe(empty_recipe, cs)
    assert result.ok  # nothing to violate


# ---------------------------------------------------------------------
# Duplicate constraint ids / the same tag excluded twice
# ---------------------------------------------------------------------


def test_duplicate_constraint_ids_never_crash_and_stay_safe(agent) -> None:
    cs = ConstraintSet(
        profile_id="dup",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut"),
            Constraint(id="c1", kind="exclude_tag", severity="medical", value="dairy"),
        ],
    )
    terminal, result = _run(agent, cs)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    if isinstance(result, Recommendation):
        recipe = agent.deps.catalog.recipe(result.recipe_id)
        assert recipe is not None
        assert "peanut" not in recipe.tags
        assert "dairy" not in recipe.tags


def test_same_tag_excluded_twice_at_different_severities_never_crashes(agent) -> None:
    cs = ConstraintSet(
        profile_id="same-tag",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut"),
            Constraint(id="c2", kind="exclude_tag", severity="preference", value="peanut"),
        ],
    )
    terminal, result = _run(agent, cs)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    if isinstance(result, Recommendation):
        recipe = agent.deps.catalog.recipe(result.recipe_id)
        assert recipe is not None
        assert "peanut" not in recipe.tags


def test_duplicate_ids_do_not_crash_check_recipe_directly() -> None:
    from beatroot.t0_invariants.constraints import check_recipe

    cs = ConstraintSet(
        profile_id="dup2",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut"),
            Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut"),
        ],
    )
    recipe = Recipe(id="r1", name="Peanut soup", tags={"peanut"}, ingredient_ids=["ing_peanuts"])
    result = check_recipe(recipe, cs)
    assert result.violated == ["c1", "c1"]  # duplicated, never deduplicated silently, never crashes
    assert not result.ok


# ---------------------------------------------------------------------
# A constraint value of the wrong type for its kind
# ---------------------------------------------------------------------


def test_exclude_tag_with_a_non_string_value_never_crashes(agent) -> None:
    cs = ConstraintSet(
        profile_id="wrong-tag-type",
        constraints=[Constraint(id="c1", kind="exclude_tag", severity="medical", value=3.14)],
    )
    terminal, result = _run(agent, cs)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    assert not isinstance(
        result, Recommendation
    )  # an unevaluable MEDICAL exclusion is never served


def test_max_prep_minutes_with_a_string_value_never_crashes(agent) -> None:
    cs = ConstraintSet(
        profile_id="wrong-prep-type",
        constraints=[
            Constraint(id="c1", kind="max_prep_minutes", severity="preference", value="ten minutes")
        ],
    )
    terminal, _result = _run(agent, cs)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")


def test_nutrient_range_with_the_wrong_shape_never_crashes() -> None:
    """A plain string or a bare number, instead of the required (lo, hi)
    pair — `_nutrient_range` must degrade to `uncheckable`, never raise
    trying to unpack it. Both values below ARE valid `ConstraintValue`
    members (str, float) — they construct fine; they are simply the wrong
    SHAPE for this particular kind, which is the actual case this test
    targets. A shape `ConstraintValue`'s union cannot represent at all is
    covered separately below, and rejected earlier, at construction."""
    from beatroot.t0_invariants.constraints import check_recipe

    recipe = Recipe(id="r2", name="Rice", tags=set(), ingredient_ids=[], nutrition=None)
    for bad_value in ("not a range", 42.0):
        cs = ConstraintSet(
            profile_id="bad-range",
            constraints=[
                Constraint(
                    id="c1",
                    kind="nutrient_range",
                    severity="goal",
                    value=bad_value,
                    nutrient="kcal",
                )
            ],
        )
        result = check_recipe(recipe, cs)
        assert result.uncheckable == ["c1"]
        assert result.ok


def test_nutrient_range_with_too_many_elements_is_rejected_at_construction() -> None:
    """A three-element value doesn't match ANY `ConstraintValue` union
    member (not `tuple[float, float]`, not `list[str]` — the elements are
    floats) — Pydantic refuses it at construction with a clean
    `ValidationError`, never lets it reach `check_recipe` malformed."""
    with pytest.raises(ValidationError):
        Constraint(
            id="c1", kind="nutrient_range", severity="goal", value=[1.0, 2.0, 3.0], nutrient="kcal"
        )


def test_constraint_construction_rejects_a_genuinely_malformed_value() -> None:
    """A shape Pydantic's own `ConstraintValue` union cannot represent at
    all (a dict) must fail at CONSTRUCTION with a clean `ValidationError`
    — never construct successfully and crash later, deeper in the graph."""
    with pytest.raises(ValidationError):
        Constraint(id="c1", kind="exclude_tag", severity="medical", value={"nested": "dict"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Extremely long strings
# ---------------------------------------------------------------------


def test_extremely_long_constraint_value_never_crashes(agent) -> None:
    cs = ConstraintSet(
        profile_id="p" * 5000,
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity="medical", value="x" * 50_000)
        ],
    )
    terminal, result = _run(agent, cs, query="d" * 20_000)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    assert not isinstance(result, Recommendation)  # a 50,000-char tag matches nothing real


def test_extremely_long_preferences_never_crashes(agent) -> None:
    cs = ConstraintSet(
        profile_id="long-prefs",
        constraints=[Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut")],
    )
    terminal, result = _run(agent, cs, preferences="ignore the rule " * 5000)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    if isinstance(result, Recommendation):
        recipe = agent.deps.catalog.recipe(result.recipe_id)
        assert recipe is not None
        assert "peanut" not in recipe.tags


# ---------------------------------------------------------------------
# Unicode in every field
# ---------------------------------------------------------------------


def test_unicode_in_every_field_never_crashes_and_stays_safe(agent) -> None:
    cs = ConstraintSet(
        profile_id="用户🙂🥜",
        constraints=[
            Constraint(id="c🌟1", kind="exclude_tag", severity="medical", value="peanut"),
        ],
    )
    terminal, result = _run(
        agent, cs, query="晚餐 🍜 café naïve", preferences="please 请不要花生 🥜"
    )
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    if isinstance(result, Recommendation):
        recipe = agent.deps.catalog.recipe(result.recipe_id)
        assert recipe is not None
        assert "peanut" not in recipe.tags
    if isinstance(result, Negotiation):
        assert "c🌟1" in result.locked
    if isinstance(result, Escalation):
        assert "c🌟1" in result.failing_signal


def test_unicode_zero_width_and_rtl_never_crashes(agent) -> None:
    """Zero-width joiners, combining marks, and right-to-left override
    characters are the classic string-processing edge case — never real
    catalog vocabulary, so the correct behaviour is a clean refusal, never
    a crash."""
    weird = "peanut​́‮"  # zero-width space + combining acute + RTL override
    cs = ConstraintSet(
        profile_id="rtl-test",
        constraints=[Constraint(id="c1", kind="exclude_tag", severity="medical", value=weird)],
    )
    terminal, result = _run(agent, cs)
    assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
    assert not isinstance(result, Recommendation) or "peanut" not in (
        agent.deps.catalog.recipe(result.recipe_id).tags
        if isinstance(result, Recommendation)
        else set()
    )


# ---------------------------------------------------------------------
# A combined fuzz sweep: every one of the above shapes, back to back,
# through the same agent instance — proves no crash leaves the agent (or
# its shared connection) in a state where the NEXT request then crashes.
# ---------------------------------------------------------------------


def test_a_batch_of_every_degenerate_shape_back_to_back_never_crashes(agent) -> None:
    degenerate_constraint_sets = [
        ConstraintSet(profile_id="fuzz-empty", constraints=[]),
        ConstraintSet(
            profile_id="fuzz-dup",
            constraints=[
                Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut"),
                Constraint(id="c1", kind="exclude_tag", severity="medical", value="peanut"),
            ],
        ),
        ConstraintSet(
            profile_id="fuzz-wrong-type",
            constraints=[
                Constraint(id="c1", kind="budget_max", severity="preference", value=[1, 2])
            ],  # type: ignore[list-item]
        ),
        ConstraintSet(
            profile_id="fuzz-long" * 500,
            constraints=[
                Constraint(id="c1", kind="exclude_tag", severity="medical", value="y" * 10_000)
            ],
        ),
        ConstraintSet(
            profile_id="🥕🧄🍛",
            constraints=[
                Constraint(id="c1", kind="exclude_tag", severity="religious", value="allium")
            ],
        ),
    ]
    for cs in degenerate_constraint_sets:
        terminal, _result = _run(agent, cs, query="anything 🎉" * 100)
        assert terminal in ("COMMIT", "NEGOTIATE", "ESCALATE")
