# Eval results

The current-state snapshot: what the suite measures, what it reads, and what
it is not evidence for. Gates live in `eval/thresholds.yaml`; the historical
ledger of every run and every reverted experiment is `EVAL_HISTORY.md`, which
is generated and optional reading.

**What it runs against:** 174 recipes, 137 ingredients, 21 preset profiles,
9 constraint kinds, 5 severities (`MEDICAL`/`RELIGIOUS`/`DIETARY` are hard),
33 hand-authored golden cases across seven adversarial families.

```
uv run python -m beatroot.eval.runners.system        # or: beatroot eval system
uv run python -m beatroot.eval.runners.components    # or: beatroot eval components
uv run python -m beatroot.eval.runners.simulation    # generated adversarial families
uv run python -m beatroot.eval.calibration
```

Every table below is the offline run (`BEATROOT_OFFLINE=1` — deterministic
stub provider, no credentials, no network, $0.0000). The live-provider
comparison is in its own section.

---

## System eval — the six safety axes

33 golden cases through the real `MealPlanningAgent`. Every COMMIT is
re-checked by `eval/verifiers/hard_constraint.py`, which deliberately does not
call `t0_invariants.constraints.is_legal()` — the function the agent itself
used to pick the recipe.

| axis | score | gate | status |
|---|---|---|---|
| A1_allergen_safety | 1.000 | 1.00 | PASS |
| A2_religious_integrity | 1.000 | 1.00 | PASS |
| A3_injection_resistance | 1.000 | 0.95 | PASS |
| A4_infeasibility_detection | 1.000 | 0.95 | PASS |
| A5_escalation_correctness | 1.000 | 0.90 | PASS |
| A6_explanation_grounding | 1.000 | 0.95 | PASS |

Hard-constraint violations: **0** (gate 0 — a count, not a rate; the right
unit for irreversible harm). Latency p50 ≈50ms / p95 ≈400ms, cost $0.0000.
Overall: PASS.

Latency is gated too, per EXECUTION MODE — 2s/8s p50/p95 offline, 15s/25s
live. One budget cannot describe both: offline puts no network on the path and
a case finishes in tens of milliseconds, while a live case makes two or three
SERIAL model calls at roughly two seconds each, so a 7-9s p50 is the healthy
number. The shared budget failed every live run on latency alone while all six
safety axes read 1.000. `run_system` selects the pair by `settings.offline` and
the report names which one it applied, because a latency verdict is meaningless
without saying what it was measured against. Breaches are reported separately
from correctness failures — slow-but-correct and fast-but-unsafe are different
problems.

## Component eval — one layer at a time

Ground truth is computed, not labelled: the catalog is finite and constraints
are typed, so `eval/synth/profiles.py` brute-forces the exact valid set per
synthetic profile.

| metric | value |
|---|---|
| retrieval recall@5, full oracle (hard + soft) | 0.988 |
| retrieval recall@5, hard-only oracle | 1.000 |
| **retrieval leakage** | **0** |
| feasibility accuracy | 1.000 |
| nutrition determinism | 1.000 |
| drift detection recall | 1.000 |

Overall: PASS (the runner's exit code is `retrieval_leakage == 0`).

Two recall numbers, two different contracts, neither substituted for the
other. `retrieve()` promises to filter on hard constraints only — a soft
`budget_max` or `max_prep_minutes` is ranking's job. The hard-only oracle
grades that promise (1.000); the full oracle grades a stricter one it never
made (0.988). Leakage is the safety metric of the two: a recipe surfaced by
retrieval that the independent verifier says violates a hard constraint.

### The recall benchmark was wrong before it was low

Every case used to retrieve with one hardcoded literal, `"a balanced meal"` —
a string that returns zero FTS5 rows against this catalog, confirmed directly.
The lexical half of hybrid retrieval was structurally dead for every case, so
recall was measuring the dense channel alone against a query carrying no
signal.

Finding it was a null result read as a signal. Four retrieval configs were
swept; three of them — doubling `lexical_weight`, dropping `rrf_k` from 60 to
10, doubling `candidate_limit` — moved recall by **exactly 0.0000**. Three
orthogonal knobs with identical zero deltas is not four weak experiments, it
is one diagnostic: nothing was reaching the fusion stage to weight. The fourth
sweep (dense weight to zero) moved it, which located the only live channel.

`_derive_query` now builds a query per case from the profile's own data, and
is provably independent of `oracle_valid_ids` — two cases with identical
constraints but different oracles derive the identical query
(`test_derive_query_never_looks_at_the_oracle`), so the fix cannot inflate the
number by aiming the query at its own answers. The result is a new baseline,
not comparable to any recall figure recorded before it.

## Live provider

The same golden suite against Azure `gpt-4o`: **6/6 axes 1.000, 0 violations**,
p50 7932ms / p95 9894ms, **$0.0836** for the 33-case run. A single live
`POST /recommend` costs roughly $0.005, split across `compile`,
`rewrite_query` and `rerank`, with the async explanation adding ~$0.0009 when
the queue completes it. `/metrics` reconciles with the Langfuse trace costs.

---

## Why "1.000 everywhere" is not the evidence

A green row is a claim. What makes it credible is proof the row can go red.

**A6 read 1.000 while checking nothing.** `explanation_grounding` scores the
`drift_bait` family by diffing numbers in the generated prose against catalog
truth. The offline stub's prose stated no numbers at all, so there was nothing
to catch, and the axis reported a perfect score for the whole build without
ever evaluating anything. Closed by mutation, not by argument: a stub that
states `9000 kcal` for every meal drops A6 to **0.000** on the synchronous
path — and leaves it at 1.000 on the async path, because VERIFY sees an empty
explanation there. `run_system` now **refuses to run** against an
async-wired agent rather than report a meaningless 1.000, and
`eval/runners/components.py:_drift_detection_recall` tests the detector
directly against fixed probe prose instead of through an explanation that
cannot exercise it.

**The suite was itself mutation-tested, and failed.** Six safety properties
were disabled one at a time to see whether the suite noticed. The first pass
caught **2 of 6**. Disabling the entire trust gate was invisible because every
golden case had full catalog coverage, so nothing ever drove trust low enough
to refuse; disabling drift detection was invisible for the A6 reason above.
Both were closed with targeted artifacts — golden cases `g31`–`g33`, which
reach the *conjunctive veto* rather than the score (composite 0.575 sits
*above* the 0.55 threshold, so a weighted average alone would have missed
them), and the drift probe set — taking it to **5 of 6**. Six mutations is not
a proof of catching power; it is the difference between having checked and
not.

**Two oracles were tautologies and were rebuilt.** The A5
(`escalation_correctness`) oracle decided its verdict by calling
`t0_invariants.vocabulary.unknown_vocabulary` — the exact production predicate
the agent uses — so a bug would move the answer and the oracle together and
A5 would keep reading 1.000 with nothing left to disagree. The feasibility
oracle had the same shape one layer down: it computed ground truth by calling
`check_recipe`, the function under test, and "verified" itself by calling it
again, which proves determinism and not correctness. Both were rewritten from
scratch importing nothing from `t0_invariants`, and both cross-checks were
then proven capable of failing by monkeypatching production broken and
watching them disagree.

The independence those rewrites buy is **call-graph independence, not
assumption independence** — a distinction this project learned the expensive
way; see `docs/PRODUCTION_READINESS.md`.

---

## What these numbers do not prove

- **The dense embedder is a token-hashing stub in every run here, offline and
  live alike.** The Azure resource has a chat deployment but no embedding
  deployment, so `embedding_model` stays `local`. "Hybrid retrieval" is real
  as an architecture; its dense channel is not semantic. Every recall figure
  above measures the stub.
- **Calibration is effectively unmeasured.** COMMIT-only sampling puts every
  pair in one high-confidence bin, so the ECE that comes out says "when it
  commits it is confident and right" — never anything about calibration across
  the confidence range. Offline it is worse still: `model_self_assessment` is
  pinned at a constant, so the number describes only the deterministic share
  of the composite.
- **Offline safety scores mostly grade deterministic Python.** Families that
  terminate in `t0_invariants.feasibility` or `t0_invariants.vocabulary` never
  reach a model at all. That is exactly why their 1.000 is trustworthy, and
  exactly why it says nothing about live-model behaviour.
- **Generated is not exhaustive.** The simulation runner's families are
  parametrised templates with per-family floors in `thresholds.yaml`; a novel
  attack shape outside those templates is untested by construction.
- **A live model changing *which* legal recipe ranks highest is not
  measured.** The safety property is re-verified independently of anything the
  model says, so prose or ranking drift cannot flip a PASS to an unsafe
  COMMIT — but preference quality within the already-safe candidate set has no
  number here.
