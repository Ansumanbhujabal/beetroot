"""Component-level eval runner. Spec §12.

`eval.runners.system` proves the whole composed pipeline against golden
cases. This module isolates ONE layer at a time against `eval.synth.
profiles`'s free, exact oracle: does feasibility agree with brute-force
enumeration, does retrieval ever surface something unsafe, does nutrition
arithmetic ever disagree with itself. Answering "which layer regressed" —
not just "did the system pass" — is the difference between an eval SUITE
and a single pass/fail dashboard number.

The dense/vector index is built exactly ONCE, at `build_container()` time,
by the Container this module is handed — `run_components` reuses
`container.vector_store` for every case rather than constructing a fresh
index per profile, which would re-embed the whole catalog hundreds of times
over for no reason.

Runnable directly: `uv run python -m beatroot.eval.runners.components`.

BENCHMARK FIX (this batch): every case used to be retrieved with ONE
hardcoded literal query, `"a balanced meal"` — a string that shares zero
vocabulary with this catalog's recipe names/ingredients/cuisines/tags
(confirmed directly: `recipes_fts MATCH 'balanced meal'` returns 0 rows).
That made the lexical half of hybrid retrieval structurally dead for
EVERY case, so `retrieval_recall_at_k` was never measuring hybrid
retrieval — it was measuring the dense channel alone against a query with
no real signal. `_derive_query()` below builds a query per case from real
catalog vocabulary (a `cuisine_affinity` constraint's own value when the
profile carries one, otherwise a deterministic, profile-id-derived pick
from the catalog's actual cuisine list — never from `oracle_valid_ids`,
so the query can never be biased toward the answer it's graded against)
so both channels are actually exercised, the way a real request arrives.
**This is a benchmark-methodology fix, not a system change — the
resulting `retrieval_recall_at_k` is a NEW BASELINE, not comparable to
any number measured before this fix**, and is reported as such everywhere
it appears (CLI, artifact, dashboard, `EVAL_HISTORY.md`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from beatroot.container import Container
from beatroot.eval.synth.profiles import SyntheticCase
from beatroot.eval.verifiers import hard_constraint
from beatroot.retrieval.rerank import retrieve
from beatroot.settings import get_settings
from beatroot.t0_invariants.feasibility import assess
from beatroot.t0_invariants.nutrition_math import compute
from beatroot.trusted.catalog import Catalog


@dataclass
class ComponentReport:
    # Recall against the FULL-CONSTRAINT oracle (hard + soft) — the
    # existing headline metric. `retrieve()` never promised to satisfy
    # soft constraints (that is ranking's job, not filtering's — see
    # `retrieval_recall_at_k_hard_only` below and `run_components`'s
    # docstring), so this number is deliberately the STRICTER of the two
    # contracts, not retrieval's own.
    retrieval_recall_at_k: float = 0.0
    # Recall against retrieval's OWN contract: an oracle restricted to
    # HARD (medical/religious) constraints only, computed independently
    # via `eval.verifiers.hard_constraint.verify` (never `is_legal()`
    # itself — see that verifier's docstring). This is the fair
    # apples-to-apples number for what `retrieve()` actually guarantees.
    retrieval_recall_at_k_hard_only: float = 0.0
    retrieval_leakage: int = 0
    feasibility_accuracy: float = 0.0
    nutrition_exact_match: float = 0.0
    drift_detection_recall: float = 0.0
    notes: list[str] = field(default_factory=list)



# Prose the drift ledger must flag, paired with the nutrition it contradicts.
# Mutation testing showed that disabling `detect_drift` entirely was invisible
# to the whole eval suite: A6 only ever sees the offline stub's prose, which
# states no numbers at all, so there is never any drift to miss. This measures
# the detector directly instead of through an explanation that cannot exercise
# it.
_DRIFT_PROBES: tuple[tuple[str, bool], ...] = (
    ("This meal has about 780 kcal.", True),          # computed is 520
    ("Roughly 30g of protein in this dish.", True),   # computed is 12
    ("This meal has about 520 kcal.", False),         # agrees
    ("Ready in 35 minutes, serves 2.", False),        # not a nutrition claim
)


def _drift_detection_recall() -> float:
    """Fraction of the probe set the drift ledger classifies correctly.

    Deliberately measured on fixed prose rather than on model output: the
    point is to guard the DETECTOR, which an offline explanation can never
    exercise.
    """
    from beatroot.contracts.nutrition import NutritionFacts
    from beatroot.eval.verifiers.nutrition_drift import detect_drift

    facts = NutritionFacts(
        kcal=520.0, protein_g=12.0, carbs_g=40.0,
        fat_g=22.0, sodium_mg=610.0, fibre_g=6.0, coverage=1.0,
    )
    correct = sum(
        bool(detect_drift(text, facts, tolerance=0.05)) == should_flag
        for text, should_flag in _DRIFT_PROBES
    )
    return correct / len(_DRIFT_PROBES)


def _derive_query(case: SyntheticCase, catalog: Catalog) -> str:
    """A realistic, per-case retrieval query built from the profile's OWN
    data — never one literal shared by every case (see the module
    docstring's BENCHMARK FIX note).

    Only POSITIVE, ranking-relevant intent becomes query text:
    `cuisine_affinity` is exactly that (the oracle itself calls it "a
    ranking signal, never an enforcement gate" — see `eval.synth.profiles.
    _oracle_violates`), so a profile carrying one uses its own value. A
    profile with no such constraint gets a query anchor deterministically
    picked from the CATALOG's own cuisine vocabulary, keyed on the
    profile's id (`sha256`, not `random` — reproducible for a fixed case
    set with no separate seed to thread through).

    Deliberately NEVER derived from `case.oracle_valid_ids`: building the
    query out of an oracle-valid recipe's own name/cuisine would point
    the query at the very answers recall is graded against and inflate
    the number for free — the exact "gaming the metric" the requirements warn
    against, just moved from `top_k` into the query text instead. This
    function never looks at the oracle at all.

    Exclusions (`exclude_tag`/`exclude_ingredient`, hard or soft) never
    contribute query text either: a query is what a profile is asking
    FOR, not a restatement of what it is avoiding — echoing an exclusion
    into the query would reintroduce the excluded term as a *positive*
    lexical signal, which is not how a real request works and is not
    what any of the plausible causes named for this task were about.
    """
    for c in case.constraint_set.constraints:
        if c.kind == "cuisine_affinity" and isinstance(c.value, str) and c.value.strip():
            return c.value.replace("_", " ")

    cuisines = sorted({r.cuisine for r in catalog.recipes() if r.cuisine})
    if not cuisines:
        # No cuisine vocabulary at all (a degenerate/empty catalog, tested
        # separately) — fall back to the old literal rather than crash;
        # there is no real vocabulary to derive from either way.
        return "a balanced meal"
    digest = hashlib.sha256(case.id.encode()).hexdigest()
    anchor = cuisines[int(digest, 16) % len(cuisines)]
    return anchor.replace("_", " ")


def run_components(
    container: Container, cases: list[SyntheticCase], top_k: int | None = None
) -> ComponentReport:
    """Isolate feasibility, retrieval, and nutrition arithmetic against the
    free oracle every `case` carries.

    `retrieval_leakage` counts recipes `retrieve()` surfaces that
    INDEPENDENTLY (`eval.verifiers.hard_constraint`, which deliberately does
    not share an implementation with `t0_invariants.constraints.is_legal` —
    see that verifier's own docstring) violate a HARD constraint of the
    profile. That is the exact safety property retrieval exists to
    guarantee, and the headline component metric: it must be 0. It is
    deliberately NOT "missing from the full, all-severity oracle" — a
    GOAL/PREFERENCE constraint (a soft `max_prep_minutes`, say) is a
    preference `retrieve()` never promised to enforce, and counting a miss
    there as "leakage" would conflate "not to taste" with "unsafe".

    `retrieval_recall_at_k` and `feasibility_accuracy` are both measured
    against the SAME exact oracle (`case.oracle_valid_ids`): feasibility
    accuracy is "did `assess()` agree the catalog has ANY valid recipe left
    for this profile", recall@k is "of the recipes the oracle calls valid,
    how many landed in the top-k retrieved".

    TWO recall numbers are reported, against two DIFFERENT contracts, and
    neither is silently substituted for the other:

    - `retrieval_recall_at_k` — against `case.oracle_valid_ids`, which
      counts a recipe invalid if it violates ANY constraint, hard OR
      soft. This is the stricter, existing headline number.
    - `retrieval_recall_at_k_hard_only` — against an oracle independently
      recomputed here to match `retrieve()`'s OWN contract: `is_legal()`
      enforces HARD (medical/religious) constraints only, by design — a
      soft `budget_max`/`max_prep_minutes`/preference-severity
      `exclude_tag` is meant to be satisfied by RANKING, not filtering.
      Grading retrieval against a promise it never made would be its own
      kind of dishonesty in the other direction, so this number exists
      specifically so "retrieval under-filters" and "the oracle is
      stricter than retrieval's contract" are never conflated.

    Query text for `retrieve()` comes from `_derive_query()` — see that
    function and the module's BENCHMARK FIX note for why every case no
    longer shares one literal query.
    """
    settings = get_settings().retrieval
    k = settings.top_k if top_k is None else top_k

    catalog = container.catalog
    recipes = catalog.recipes()
    tag_index = container.tag_index
    vector_store = container.vector_store
    provider = container.llm

    report = ComponentReport()
    recalls: list[float] = []
    hard_only_recalls: list[float] = []
    feasibility_hits: list[bool] = []

    for case in cases:
        cs = case.constraint_set
        oracle = case.oracle_valid_ids

        predicted_feasible = assess(cs, recipes, tag_index).feasible
        feasibility_hits.append(predicted_feasible == bool(oracle))

        if not oracle:
            continue

        query = _derive_query(case, catalog)
        got = retrieve(query, cs, catalog, provider, vector_store=vector_store, top_k=k)
        for recipe in got:
            if hard_constraint.verify(recipe, cs):
                report.retrieval_leakage += 1
        got_ids = {r.id for r in got}
        recalls.append(len(got_ids & oracle) / min(k, len(oracle)))

        # retrieval's OWN contract: an oracle independently recomputed
        # from ONLY the hard constraints, via the same independent
        # verifier used for leakage above — never `is_legal()` itself.
        hard_oracle = {r.id for r in recipes if not hard_constraint.verify(r, cs)}
        if hard_oracle:
            hard_only_recalls.append(len(got_ids & hard_oracle) / min(k, len(hard_oracle)))

    report.feasibility_accuracy = (
        sum(feasibility_hits) / len(feasibility_hits) if feasibility_hits else 0.0
    )
    report.retrieval_recall_at_k = sum(recalls) / len(recalls) if recalls else 0.0
    report.retrieval_recall_at_k_hard_only = (
        sum(hard_only_recalls) / len(hard_only_recalls) if hard_only_recalls else 0.0
    )

    # Nutrition determinism: pure catalog arithmetic must never disagree
    # with itself on a repeat call. Measured over `recipes` (not `cases`) so
    # this metric means something even when `cases` is empty.
    matches = [
        compute(payload, catalog) == compute(payload, catalog)
        for r in recipes
        if (payload := catalog.recipe_payload(r.id)) is not None
    ]
    report.nutrition_exact_match = sum(matches) / len(matches) if matches else 0.0

    report.notes.append(
        "BENCHMARK CHANGED: retrieval_recall_at_k now uses a per-case query "
        "derived from each profile's own data (eval.runners.components."
        "_derive_query), not one shared literal — this is a NEW BASELINE, "
        "not comparable to recall figures measured before this fix."
    )
    report.notes.append(
        "retrieval_recall_at_k is graded against the full-constraint oracle "
        "(hard + soft); retrieval_recall_at_k_hard_only is graded against "
        "retrieval's own hard-constraint-only contract — soft constraints "
        "are ranking's job, not filtering's, so the two numbers measure "
        "different promises and neither substitutes for the other."
    )
    if cases and report.retrieval_recall_at_k < 0.5:
        reason = (
            "the offline embedding is a token-hashing bag-of-words stub with "
            "no semantic signal"
            if getattr(provider, "_offline", False)
            else "the configured embedding model may need retuning"
        )
        report.notes.append(f"recall@k is modest: {reason}.")

    report.drift_detection_recall = _drift_detection_recall()
    return report


def _print_report(report: ComponentReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table("metric", "value")
    table.add_row("retrieval recall@k (full oracle)", f"{report.retrieval_recall_at_k:.3f}")
    table.add_row(
        "retrieval recall@k (hard-only oracle)",
        f"{report.retrieval_recall_at_k_hard_only:.3f}",
    )
    table.add_row("retrieval leakage", str(report.retrieval_leakage))
    table.add_row("feasibility accuracy", f"{report.feasibility_accuracy:.3f}")
    table.add_row("nutrition determinism", f"{report.nutrition_exact_match:.3f}")
    table.add_row("drift detection recall", f"{report.drift_detection_recall:.3f}")
    console.print(table)
    for note in report.notes:
        console.print(f"note: {note}")
    verdict = "PASS" if report.retrieval_leakage == 0 else "FAIL"
    console.print(f"\noverall: {verdict} (retrieval_leakage must be 0)")


def main() -> int:
    """`uv run python -m beatroot.eval.runners.components`.

    Wired into the CLI as `beatroot eval components`.

    Also persists the result to `eval/last_run.json`
    (`eval.artifact.write_components_result`) alongside the console
    table — the ONLY way that artifact's `"components"` section is ever
    produced. `GET /evals/summary` reads it; it never generates the 200
    synthetic profiles and runs them through the agent itself (that is a
    batch job, not something a page load can wait on — see that route's
    docstring).
    """
    from beatroot.container import build_container
    from beatroot.eval.artifact import write_components_result
    from beatroot.eval.synth.profiles import generate_profiles

    container = build_container()
    cases = generate_profiles(container.catalog, seed=0)
    report = run_components(container, cases)
    _print_report(report)
    write_components_result(report)
    return 0 if report.retrieval_leakage == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
