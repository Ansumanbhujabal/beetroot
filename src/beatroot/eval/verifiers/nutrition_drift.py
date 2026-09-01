"""The drift ledger: diff every nutrition number a model states in prose
against catalog truth. Spec §9.

This exists because model-generated nutrition claims were observed running
roughly 1.5x high in production — a fluent, confident, and wrong number. The
ledger is the last line of defense: even though `NutritionFacts` can only
ever be constructed with `provenance="computed"` (Task 1), nothing stops the
EXPLANATION text (free prose from the model) from stating a different number
than the one it was handed. VERIFY (Task 11's `agent/nodes.py`) runs this
after every explanation and escalates on any finding.

Only numbers within a fixed window of a nutrient CUE WORD count as a claim —
"ready in 35 minutes, serves 2" mentions two numbers that are not nutrition
claims at all, and must never be flagged.
"""

import re
from dataclasses import dataclass

from beatroot.contracts.nutrition import NutritionFacts

CUES: dict[str, tuple[str, ...]] = {
    "kcal": ("kcal", "calorie", "calories"),
    "protein_g": ("protein",),
    "carbs_g": ("carb", "carbs", "carbohydrate", "carbohydrates"),
    "fat_g": ("fat",),
    "sodium_mg": ("sodium", "salt"),
    "fibre_g": ("fibre", "fiber"),
}
WINDOW = 30
_NUM = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class DriftFinding:
    field: str
    stated: float
    computed: float
    delta_pct: float


# A number counts as a CLAIM about a nutrient only when it is grammatically
# attached to that nutrient's cue word — "661.2 kcal", "19.8 g of protein",
# "sodium: 282.4". Anything else merely sharing a sentence with the cue is not
# a claim about it.
#
# This replaced a +/-30 character WINDOW that swept up every number near the
# cue, which made the whole check unfixable: with a window, "661.2 kcal per
# 100 g" offers {661.2, 100} for kcal and NEITHER selection rule works —
# taking the closest lets "661.2 kcal, though some say 9000 kcal" pass
# (fail-open, the bug this fixes), and taking the worst flags the innocent
# "per 100 g" as a 100 kcal claim (fail-closed, unusable). The selection rule
# was never the problem; binding numbers that were never claims was.
_UNIT = r"(?:kcal|calories?|g|gs|grams?|mg|milligrams?)"
_VALUE = r"(\d+(?:\.\d+)?)"


def _claims_for_cue(text: str, cue: str) -> list[float]:
    """Numbers grammatically bound to `cue`, in either order."""
    c = re.escape(cue)
    patterns = (
        # "9000 kcal", "76.9 carbs" — the value sits directly on the cue.
        rf"{_VALUE}\s*{c}\b",
        # "19.8 g of protein", "282.4 mg sodium" — a unit intervenes. Kept
        # separate from the form above so the optional unit cannot swallow a
        # cue that IS a unit ("kcal"), which silently matched nothing.
        rf"{_VALUE}\s*{_UNIT}\s*(?:of\s+)?{c}\b",
        # "sodium: 282.4", "protein is 19.8", "carbs = 76.9".
        rf"\b{c}\b\s*(?:of|is|are|at|:|=)?\s*{_VALUE}",
    )
    out: list[float] = []
    for pattern in patterns:
        out.extend(float(m.group(1)) for m in re.finditer(pattern, text))
    return out


def detect_drift(
    text: str, nutrition: NutritionFacts, tolerance: float = 0.05
) -> list[DriftFinding]:
    """Diff every number the model stated against catalog truth.

    Only numbers near a nutrient cue word count — "ready in 35 minutes" is not
    a nutrition claim. For each field the CLOSEST nearby number is compared, so
    a correct mention elsewhere in the sentence does not mask a wrong one.
    `tolerance` should come from `load_thresholds().verifiers.nutrition_drift_pct`
    at the call site, never a literal. Spec §9.
    """
    lowered = text.lower()
    findings: list[DriftFinding] = []

    for field, cues in CUES.items():
        computed = float(getattr(nutrition, field))
        if computed == 0:
            continue
        candidates = [n for cue in cues for n in _claims_for_cue(lowered, cue)]
        if not candidates:
            continue

        # Flag the WORST nearby claim, not the closest one.
        #
        # This previously took `min(candidates, key=|v - computed|)` — the
        # number nearest the truth — under a docstring saying it existed "so
        # a correct mention elsewhere in the sentence does not mask a wrong
        # one". It did exactly the opposite, and fails OPEN:
        #
        #   "This meal has 661.2 kcal, though some sources say 9000 kcal."
        #
        # passed the ledger clean, because 661.2 was closest to computed and
        # the 9000 was never scored. Hedging like that is a normal thing for
        # a model to emit, so this was reachable, not theoretical.
        #
        # Taking the worst means a diner is never shown prose containing a
        # fabricated figure just because a correct figure sits beside it. The
        # cost is the mirror risk — an incidental number ("per 100 g")
        # scoring as a claim — which is why `_claims_for_cue` binds only
        # numbers grammatically attached to the cue, never everything nearby.
        stated = max(candidates, key=lambda v: abs(v - computed))
        delta = abs(stated - computed) / computed
        if delta > tolerance:
            findings.append(DriftFinding(field, stated, computed, round(delta, 4)))

    return findings
