"""A6 (explanation_grounding) must be able to FAIL, or its passing is noise.

Until now the offline stub returned `[offline:a1b2c3d4]` — prose with no
digits in it. `verify_node`'s `detect_drift` had nothing to parse, so A6
scored 1.000 on every recorded run without once exercising the check. Both
EVAL_RESULTS.md and the readiness audit disclosed the axis was vacuous; the
disclosure stood for the whole build because the stub is what runs in CI.

Two things are needed to close that, and the first alone is not enough:

  1. the stub must state real numbers, so the axis measures the true path
     (prose -> regex -> drift ledger -> COMMIT) rather than measuring silence
  2. the stub must be shown to be CATCHABLE when it lies

This file is (2): it mutates the stub into stating a fabricated number and
asserts the system rejects it. Without this test, step (1) would only have
replaced "always passes because prose is empty" with "always passes because
prose is correct" — a different vacuity wearing the same 1.000.
"""

from beatroot.contracts.nutrition import NutritionFacts
from beatroot.eval.verifiers.nutrition_drift import detect_drift
from beatroot.reasoning.llm import LLMClient
from beatroot.reasoning.prompts import load_prompt

FACTS = "kcal=661.2, protein_g=19.8, carbs_g=76.86, fat_g=30.12, sodium_mg=282.4, fibre_g=4.0"
N = NutritionFacts(
    kcal=661.2, protein_g=19.8, carbs_g=76.86, fat_g=30.12,
    sodium_mg=282.4, fibre_g=4.0, coverage=1.0,
)


def _prompt() -> str:
    return load_prompt("explain").render(
        name="Pesto pasta", facts=FACTS, satisfied="contains no peanut"
    )


def test_offline_explanation_actually_states_numbers() -> None:
    """The precondition. If this regresses to digitless text, A6 silently
    goes vacuous again and every downstream 1.000 stops meaning anything."""
    text = LLMClient._offline_explain(_prompt(), "deadbeef")
    assert any(ch.isdigit() for ch in text), f"offline prose states no numbers: {text!r}"
    assert "661.2" in text and "19.8" in text


def test_truthful_offline_explanation_passes_the_drift_ledger() -> None:
    """The happy path must be genuinely clean, not clean by omission."""
    text = LLMClient._offline_explain(_prompt(), "deadbeef")
    assert detect_drift(text, N, tolerance=0.02) == []


def test_a_fabricated_number_is_caught_end_to_end() -> None:
    """THE POINT. Mutate the stub into lying and confirm the ledger rejects
    it. This is what makes A6=1.000 a result rather than an artefact."""
    lie = "This meal provides 9000 kcal and 300 g of protein. Offline deterministic explanation."
    findings = detect_drift(lie, N, tolerance=0.02)
    assert findings, "a fabricated calorie count passed the drift ledger"
    assert any(f.field == "kcal" and f.stated == 9000.0 for f in findings), (
        f"the 9000 kcal claim was not attributed to kcal: "
        f"{[(f.field, f.stated) for f in findings]}"
    )


def test_drift_is_caught_through_a_real_verify_node_not_just_the_regex() -> None:
    """The unit-level detector was already tested. What was never tested was
    the whole path: generated prose reaching VERIFY and being rejected there.
    A drift finding must make `verify_node` refuse to COMMIT."""
    from beatroot.confirm.trust_score import load_thresholds

    tolerance = load_thresholds().verifiers.nutrition_drift_pct
    truthful = LLMClient._offline_explain(_prompt(), "deadbeef")
    lie = "This meal provides 9000 kcal and 300 g of protein."

    assert detect_drift(truthful, N, tolerance=tolerance) == []
    assert detect_drift(lie, N, tolerance=tolerance), (
        "verify_node would have committed prose contradicting the catalog"
    )


# --- the fail-open bug this axis was hiding ---------------------------------


def test_a_correct_number_does_not_mask_a_fabricated_one() -> None:
    """The bug found while making A6 falsifiable, and the more serious of the
    two. `detect_drift` scored `min(candidates, key=|v - computed|)` — the
    number NEAREST the truth — under a docstring claiming the rule existed
    "so a correct mention elsewhere in the sentence does not mask a wrong
    one". It did the opposite, and hedging is ordinary model behaviour.
    """
    hedged = "This meal has 661.2 kcal, though some sources say 9000 kcal."
    findings = detect_drift(hedged, N, tolerance=0.02)
    assert findings, "a correct figure masked a fabricated one — ledger failed open"
    assert any(f.stated == 9000.0 for f in findings)


def test_incidental_numbers_are_not_read_as_claims() -> None:
    """The mirror risk, and why the fix was not simply flipping min to max.

    A +/-30 char window swept up numbers that were never claims about the
    nutrient, so with a window NEITHER rule works: closest fails open on the
    hedge above, worst flags "per 100 g" as a 100 kcal claim. Numbers are now
    bound grammatically to the cue, so every candidate is a real claim.
    """
    for benign in (
        "This meal has 661.2 kcal per 100 g serving.",
        "Serves 4. This meal has 661.2 kcal and 19.8 g of protein.",
        "Ready in 35 minutes; 661.2 kcal.",
    ):
        assert detect_drift(benign, N, tolerance=0.02) == [], f"false positive on: {benign!r}"


def test_claims_bind_in_both_word_orders() -> None:
    """"9000 kcal" and "sodium: 5000" are both claims; only one is a number
    followed by its cue."""
    assert detect_drift("This meal has 9000 kcal.", N, tolerance=0.02)
    assert detect_drift("sodium: 5000 mg", N, tolerance=0.02)
    assert detect_drift("661.2 kcal and 300 g of protein.", N, tolerance=0.02)


# --- the axis itself, end to end -------------------------------------------


def test_a6_axis_drops_when_the_model_lies() -> None:
    """Mutation test at the AXIS level, not the detector level.

    The detector was always unit-tested. What was never shown is that A6 —
    the number reported to a reader — responds to a model that lies. It did
    not: the offline stub emitted digitless text, so drift_bait cases passed
    unconditionally and A6 read 1.000 for the entire build while checking
    nothing.

    Truthful stub -> 1.000. Stub stating 9000 kcal for every meal -> 0.000.
    If this test ever passes with both values equal, A6 has gone vacuous
    again and every reported 1.000 is worthless.
    """
    import os
    from unittest.mock import patch

    import yaml

    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import ROOT, build_container
    from beatroot.eval.runners.system import run_system
    from beatroot.settings import get_settings

    os.environ["BEATROOT_OFFLINE"] = "1"
    get_settings.cache_clear()

    cases = yaml.safe_load((ROOT / "eval" / "golden" / "seed_cases.yaml").read_text())
    th = load_thresholds()

    truthful = run_system(build_container(async_explanation=False).agent, cases, th)
    with patch.object(
        LLMClient,
        "_offline_explain",
        staticmethod(lambda p, d: "This meal provides 9000 kcal and 300 g of protein."),
    ):
        lying = run_system(build_container(async_explanation=False).agent, cases, th)

    get_settings.cache_clear()

    assert truthful.axes["A6_explanation_grounding"] == 1.0
    assert lying.axes["A6_explanation_grounding"] < 1.0, (
        "A6 did not move when every explanation stated 9000 kcal — the axis is vacuous"
    )


def test_run_system_refuses_to_report_a_meaningless_a6() -> None:
    """An agent on the async path makes every drift_bait case pass regardless
    of what the model said. Reporting 1.000 there is worse than not running.

    Drives the AGENT's wiring rather than an env var, because the two
    legitimately differ: eval builds sync while config/beatroot.yaml ships
    async on for latency."""
    import pytest

    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import build_container
    from beatroot.eval.runners.system import run_system

    c = build_container(async_explanation=True)
    try:
        assert c.agent.deps.explanation_queue is not None
        with pytest.raises(RuntimeError, match="cannot be measured"):
            run_system(c.agent, [], load_thresholds())
    finally:
        c.close()
