import pytest
from pydantic import ValidationError

from beatroot.contracts.core import Constraint, ConstraintSet, Severity


def _c(cid: str, severity: Severity) -> Constraint:
    return Constraint(
        id=cid,
        kind="exclude_tag",
        severity=severity,
        value="peanut",
        source="structured",
    )


def test_hard_returns_only_medical_and_religious():
    cs = ConstraintSet(
        profile_id="p1",
        constraints=[
            _c("a", Severity.MEDICAL),
            _c("b", Severity.RELIGIOUS),
            _c("c", Severity.GOAL),
            _c("d", Severity.PREFERENCE),
        ],
    )
    assert {c.id for c in cs.hard()} == {"a", "b"}
    assert {c.id for c in cs.soft()} == {"c", "d"}


def test_hard_and_soft_partition_exactly():
    cs = ConstraintSet(
        profile_id="p1",
        constraints=[
            _c("a", Severity.MEDICAL),
            _c("c", Severity.GOAL),
        ],
    )
    assert len(cs.hard()) + len(cs.soft()) == len(cs.constraints)


def test_nutrition_provenance_cannot_be_forged():
    from beatroot.contracts.nutrition import NutritionFacts

    with pytest.raises(ValidationError):
        NutritionFacts(
            kcal=1.0,
            protein_g=1.0,
            carbs_g=1.0,
            fat_g=1.0,
            sodium_mg=1.0,
            fibre_g=1.0,
            coverage=1.0,
            provenance="model_generated",
        )


def test_fingerprint_separates_nutrient_ranges_on_different_nutrients():
    """Two ranges with identical bounds on DIFFERENT nutrients must not
    share a feasibility-cache entry.

    `nutrient_range` splits its meaning across two fields — the bounds in
    `value`, which nutrient they constrain in `nutrient`. Hashing only the
    first made "20-60 g of protein" and "20-60 mg of sodium" the same key,
    so the second profile silently received the first's surviving-recipe
    list. A wrong answer delivered by a cache hit, with nothing failing
    anywhere to reveal it.
    """
    from beatroot.contracts.core import Constraint, ConstraintSet

    def profile(nutrient: str) -> ConstraintSet:
        return ConstraintSet(
            profile_id="p",
            constraints=[
                Constraint(
                    id="c0",
                    kind="nutrient_range",
                    severity=Severity.GOAL,
                    value=(20.0, 60.0),
                    nutrient=nutrient,
                )
            ],
        )

    assert profile("protein_g").fingerprint() != profile("sodium_mg").fingerprint()


def test_fingerprint_still_shares_between_equivalent_constraint_sets():
    """The flip side, and the reason the cache is keyed on shape at all:
    two profiles whose constraints differ only in `id` and `profile_id`
    admit exactly the same meals, so they SHOULD share an entry."""
    from beatroot.contracts.core import Constraint, ConstraintSet

    a = ConstraintSet(
        profile_id="alice",
        constraints=[
            Constraint(id="a0", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")
        ],
    )
    b = ConstraintSet(
        profile_id="bob",
        constraints=[
            Constraint(id="b0", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")
        ],
    )
    assert a.fingerprint() == b.fingerprint()

