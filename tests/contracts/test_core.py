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
