"""Every number in the generated prose must trace to a computed fact.
Spec §12, adversarial family "drift_bait".

`agent.nodes.verify_node` already runs `eval.verifiers.nutrition_drift.
detect_drift` against the CHOSEN nutrient fields, gated on a cue word
("kcal", "protein", ...) appearing near the number — that is the live
safety net wired into the graph. This verifier is deliberately blunter: it
does not require a cue word at all, because the failure this exists to catch
is a model inventing a number with no cue word nearby ("about 800, give or
take") that would slip straight past a cue-gated check. Small integers
(servings, steps, minutes) are exempt via `ALLOWED_BARE`, or every
explanation would read as "ungrounded" by definition.
"""

import re

from beatroot.contracts.nutrition import NutritionFacts

_NUM = re.compile(r"\d+(?:\.\d+)?")
# Numbers that are legitimately not nutrition claims (servings, step counts,
# small quantities in an ingredient list).
ALLOWED_BARE = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

_FIELDS = ("kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g")


def verify(text: str, nutrition: NutritionFacts, tolerance: float = 0.02) -> tuple[bool, list[str]]:
    """Ungrounded numbers are the observable symptom of a model inventing
    facts. Returns `(all_grounded, ungrounded_literals)`."""
    computed = [float(getattr(nutrition, f)) for f in _FIELDS]
    ungrounded: list[str] = []
    for raw in _NUM.findall(text):
        if raw in ALLOWED_BARE:
            continue
        value = float(raw)
        if not any(abs(value - c) <= max(tolerance * c, 0.5) for c in computed if c):
            ungrounded.append(raw)
    return (not ungrounded), ungrounded
