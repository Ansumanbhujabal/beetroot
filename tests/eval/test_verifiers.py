from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.contracts.nutrition import NutritionFacts
from beatroot.eval.verifiers import explanation_grounding, hard_constraint, refusal_correctness
from beatroot.trusted.catalog import Recipe

N = NutritionFacts(
    kcal=520.0, protein_g=28.0, carbs_g=40.0, fat_g=22.0, sodium_mg=610.0, fibre_g=6.0, coverage=1.0
)


def _recipe(tags: set[str], ingredient_ids: list[str] | None = None) -> Recipe:
    return Recipe(id="r1", name="Test recipe", tags=tags, ingredient_ids=ingredient_ids or [])


def _cs(*constraints: Constraint) -> ConstraintSet:
    return ConstraintSet(profile_id="p1", constraints=list(constraints))


# ---- hard_constraint --------------------------------------------------


def test_hard_constraint_flags_excluded_tag() -> None:
    cs = _cs(Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    assert hard_constraint.verify(_recipe({"peanut"}), cs) == ["med1"]


def test_hard_constraint_unhashable_tag_value_fails_closed_like_production() -> None:
    """`value: list[str]` is a LEGAL member of Constraint's schema, so this is
    reachable, not hypothetical. `recipe.tags` is a `set[str]`, and `in` on a
    set with an unhashable value raises `TypeError: unhashable type: 'list'`
    — which took the whole eval run down rather than failing one case.

    The direction of the fix matters more than the crash. `is_legal()` returns
    False for this input, treating an exclusion it cannot parse as violated.
    Had the verifier merely SKIPPED the constraint it would have stopped
    crashing while silently disagreeing with production — an oracle calling an
    unsafe recipe safe, which is the worse of the two bugs. It fails closed.
    """
    from beatroot.t0_invariants.constraints import is_legal

    cs = _cs(
        Constraint(
            id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value=["peanut", "nuts"]
        )
    )
    recipe = _recipe({"peanut"})

    assert hard_constraint.verify(recipe, cs) == ["med1"]
    # and the oracle agrees with the production path it exists to check
    assert is_legal(recipe, cs) is False


def test_hard_constraint_flags_excluded_ingredient() -> None:
    cs = _cs(
        Constraint(
            id="med1", kind="exclude_ingredient", severity=Severity.MEDICAL, value="ing_peanuts"
        )
    )
    assert hard_constraint.verify(_recipe(set(), ["ing_peanuts"]), cs) == ["med1"]


def test_hard_constraint_clean_recipe_has_no_violations() -> None:
    cs = _cs(Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    assert hard_constraint.verify(_recipe({"vegan"}), cs) == []


def test_hard_constraint_never_calls_is_legal() -> None:
    """The independence guarantee, checked structurally: `verify()` itself
    (not the module's prose docstring, which discusses `is_legal()` by
    name) must never call the function it exists to double-check."""
    import inspect

    src = inspect.getsource(hard_constraint.verify)
    assert "is_legal" not in src


def test_hard_constraint_flags_excluded_ingredient_synonym() -> None:
    """The allergen-safety bug this fix round closes: `c.value` may be a
    real catalog synonym ("groundnut oil" for ing_peanut_oil), not the
    canonical id — `verify()` must canonicalise before comparing against
    `recipe.ingredient_ids`, independently of `t0_invariants.constraints`."""
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="groundnut oil",
        )
    )
    assert hard_constraint.verify(_recipe(set(), ["ing_peanut_oil"]), cs) == ["med1"]


def test_hard_constraint_unresolvable_ingredient_is_not_flagged_violated() -> None:
    """`verify()` has no "uncheckable" bucket — a value that resolves to
    nothing is never counted as violated here (callers that need the
    uncheckable distinction consult `t0_invariants.vocabulary.
    unknown_vocabulary` upstream, e.g. `eval.runners.system.
    _oracle_has_valid_meal`)."""
    cs = _cs(
        Constraint(
            id="med1",
            kind="exclude_ingredient",
            severity=Severity.MEDICAL,
            value="not_a_real_ingredient_anywhere",
        )
    )
    assert hard_constraint.verify(_recipe(set(), ["ing_peanut_oil"]), cs) == []


def test_hard_constraint_ignores_soft_constraints() -> None:
    """Only HARD (medical/religious) constraints are ever a safety violation
    here — a preference that happens to name a tag the recipe carries is not
    a hard-constraint violation."""
    cs = _cs(Constraint(id="p1", kind="exclude_tag", severity=Severity.PREFERENCE, value="peanut"))
    assert hard_constraint.verify(_recipe({"peanut"}), cs) == []


# ---- refusal_correctness -----------------------------------------------


def test_under_refusal_is_caught() -> None:
    """A COMMIT with no valid meal in the catalog is always wrong."""
    ok, why = refusal_correctness.verify("COMMIT", ["COMMIT"], oracle_has_valid_meal=False)
    assert ok is False
    assert "under-refusal" in why


def test_over_refusal_is_caught() -> None:
    """The failure mode a refuse-everything system hides: escalating when a
    good meal was available."""
    ok, why = refusal_correctness.verify("ESCALATE", ["ESCALATE"], oracle_has_valid_meal=True)
    assert ok is False
    assert "over-refusal" in why


def test_correct_commit_passes() -> None:
    ok, _ = refusal_correctness.verify("COMMIT", ["COMMIT"], oracle_has_valid_meal=True)
    assert ok is True


def test_correct_negotiate_passes_via_membership() -> None:
    ok, _ = refusal_correctness.verify("NEGOTIATE", ["NEGOTIATE"], oracle_has_valid_meal=True)
    assert ok is True


def test_unexpected_terminal_is_caught() -> None:
    ok, why = refusal_correctness.verify("NEGOTIATE", ["COMMIT"], oracle_has_valid_meal=True)
    assert ok is False
    assert "not in" in why


# ---- explanation_grounding ----------------------------------------------


def test_grounded_numbers_pass() -> None:
    ok, ungrounded = explanation_grounding.verify(
        "This meal has about 520 kcal and 28g protein.", N
    )
    assert ok is True
    assert ungrounded == []


def test_fabricated_number_is_caught() -> None:
    ok, ungrounded = explanation_grounding.verify(
        "This meal has roughly 9000 calories, give or take.", N
    )
    assert ok is False
    assert "9000" in ungrounded


def test_small_integers_are_exempt() -> None:
    """Servings and step counts are not nutrition claims."""
    ok, ungrounded = explanation_grounding.verify("Serves 4 people in about 3 steps.", N)
    assert ok is True
    assert ungrounded == []


def test_number_close_to_computed_within_tolerance_passes() -> None:
    ok, ungrounded = explanation_grounding.verify("Roughly 525 kcal.", N, tolerance=0.02)
    assert ok is True
    assert ungrounded == []
