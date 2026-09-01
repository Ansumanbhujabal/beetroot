from beatroot.contracts.nutrition import NutritionFacts
from beatroot.eval.verifiers.nutrition_drift import detect_drift

N = NutritionFacts(
    kcal=520.0, protein_g=28.0, carbs_g=40.0, fat_g=22.0, sodium_mg=610.0, fibre_g=6.0, coverage=1.0
)


def test_matching_number_in_prose_is_clean():
    assert detect_drift("This meal has about 520 kcal and 28g protein.", N) == []


def test_inflated_number_is_caught():
    """The production failure this exists for: model nutrition ran ~1.5x high."""
    findings = detect_drift("This meal has about 780 kcal.", N, tolerance=0.05)
    assert findings and findings[0].field == "kcal"
    assert findings[0].stated == 780.0 and findings[0].computed == 520.0


def test_within_tolerance_is_not_flagged():
    assert detect_drift("Roughly 530 kcal.", N, tolerance=0.05) == []


def test_number_not_near_a_nutrient_word_is_ignored():
    assert detect_drift("Ready in 35 minutes, serves 2.", N) == []


def test_closest_number_is_compared_not_first_mention():
    """A correct mention elsewhere in the sentence must not mask a wrong one
    right next to the cue word."""
    findings = detect_drift(
        "Serves 520 people at a wedding! This meal has about 780 kcal.", N, tolerance=0.05
    )
    assert findings and findings[0].field == "kcal" and findings[0].stated == 780.0


def test_zero_computed_value_is_never_flagged():
    """A field the catalog computed as exactly zero has nothing to drift
    against — division by zero must never happen here."""
    zero_sodium = NutritionFacts(
        kcal=200.0, protein_g=5.0, carbs_g=20.0, fat_g=5.0, sodium_mg=0.0, fibre_g=2.0, coverage=1.0
    )
    assert detect_drift("This meal has 900mg sodium, a huge claim.", zero_sodium) == []
