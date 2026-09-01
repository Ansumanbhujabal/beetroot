---
title: beatroot
emoji: 🥁
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Meal planning with enforced model/validation boundaries
---

# beatroot

**Deterministic exactly where failure is irreversible; hybrid everywhere else.**

Not "deterministic because models can't be trusted." Hybrid retrieval beats pure
keyword search, and a rules-only planner is a worse product. But you cannot
un-eat a peanut. Allergen enforcement, religious-restriction enforcement, and
nutrition arithmetic never touch a model. Retrieval, ranking, and explanation
prose do — under a filter the model cannot see past. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the tier map and the two tests that
enforce it as code, not documentation.

**This ran well past the brief's suggested 2-3-hour local-prototype window —
197 files, 34,900+ insertions.** Beyond the brief: enforcement tests for the
T0 boundary instead of a documented convention, two complete features
(Infeasibility Negotiator, Trust-Gated Escalation with a healing loop)
instead of one stub, an eval harness with an independently-derived oracle
that has caught three real bugs in this codebase (see `CUT_LIST.md`), and a
written scale discussion instead of a paragraph of speculation. The brief's
four questions — where are the boundaries, what would you build, what would
you cut, how would this scale — are answered by, in order: the T0 tier and
the two tests in `ARCHITECTURE.md` §2; the two features below; `CUT_LIST.md`;
and `ARCHITECTURE.md` §5.

## Quick start

```bash
uv sync
uv run beatroot recommend "something warm with rice" --medical peanut
```

This runs against a **blank `.env`** — no keys, no network call. `beatroot.settings`
detects the absence of provider credentials, logs it at `WARNING`, and falls
back to a deterministic offline provider (SHA-256 hashing trick embeddings,
templated completions). Every code path — retrieval, trust scoring, escalation,
the state machine — runs for real; only the model's own opinion is a stub.

```
COMMIT — Jeera rice
  kcal=316 protein=4.21g carbs=26.97g fat=21.29g ... (coverage 1.00, provenance=computed)
  trust 0.88  (catalog_coverage=1.00, constraint_completeness=1.00, model_self_assessment=0.50)
  constraints satisfied: med0
```

`--medical`/`--exclude` build typed, structured constraints directly — the
free-text `query` argument feeds retrieval only. `compile_constraints`
(parsing free text into `ConstraintSet` entries) runs as a `compile` node
before `feasibility` and is reachable via `--preferences` (CLI), the
`preferences` field on `POST /recommend`, and the dashboard's free-text box.
It can only ever ADD `PREFERENCE`-severity constraints — never medical or
religious, never a removal of one that arrived structured — enforced in
code (`agent.nodes.compile_node`), not just asserted in the prompt. Don't
type an allergy into the query string and expect it enforced; pass it as a
constraint (or as `--preferences`, where it becomes one at `PREFERENCE`
severity, not `MEDICAL`).

To use a real model: copy `.env.example` to `.env`, fill in
`AZURE_API_KEY`/`AZURE_API_BASE` (or point `OLLAMA_API_BASE` at a local
Ollama), and re-run — no code change, same command.

## The 30-second proof

```bash
uv run python -m beatroot.eval.runners.system
```

Zero credentials required. This is the system-level safety suite: six axes,
each independently verified against a catalog-derived oracle that never calls
the code it's checking. Measured on this branch, this run:

```
A1_allergen_safety          1.000 / 1.0   PASS   (a count: 0 violations)
A2_religious_integrity      1.000 / 1.0   PASS
A3_injection_resistance     1.000 / 0.95  PASS
A4_infeasibility_detection  1.000 / 0.95  PASS
A5_escalation_correctness   1.000 / 0.9   PASS
A6_explanation_grounding    1.000 / 0.95  PASS
hard constraint violations: 0 (threshold 0)
p50 9ms  p95 11ms  cost $0.0000
overall: PASS
```

**This suite did not pass on its first honest run, and that is the point of
building it.** `A5_escalation_correctness` scored **0.000** the first time it
ran for real: a constraint naming a tag or ingredient absent from the catalog
vocabulary ("I'm allergic to sulfites," and the catalog has no `sulfite` tag)
was silently treated as **always satisfied** instead of triggering an
escalation. The user would have received a confident recommendation the
system never actually checked. Fixed by validating constraint vocabulary
against the real catalog before feasibility runs, at zero token cost. Full
story, and two later findings of the same shape, in `CUT_LIST.md` and
`ARCHITECTURE.md`.

**`A6_explanation_grounding` refuses to run at all against the shipped
config.** `config/beatroot.yaml` now ships `async_explanation: true` by
default (it used to be off). Under that path `verify_node` sees an empty
explanation string and finds no numbers to diff, so the drift-bait cases A6
scores against would pass unconditionally — `run_system` raises instead of
reporting a meaningless `1.000`, and the command above must build its
container with `async_explanation=False` to get the table shown. A6's
falsifiability is proven by mutation: a stub telling the truth scores
`1.000`; the same stub claiming 9000 kcal for every meal scores `0.000`. The
async path's own grounding guarantee is a separate, always-on check —
`tests/agent/test_async_explanation_is_grounded.py` — not this axis. See
`ARCHITECTURE.md` §3.

**A5's own oracle was found, in review, to be a tautology; it no longer
is.** The line above claims every axis is checked against an oracle that
never calls the code it's checking — `_oracle_has_valid_meal` broke that
for A5 specifically: it called `t0_invariants.vocabulary.
unknown_vocabulary`, the exact production predicate `feasibility` uses, to
decide the verdict. A bug there would have moved the agent's answer and the
oracle's answer together, and A5 would have kept reading 1.000 with nothing
left in the eval to disagree with it. Rebuilt from scratch in
`eval.verifiers.vocabulary` (reads catalog tags, ingredient ids, and
synonyms directly; imports nothing from `t0_invariants`), and proven
capable of disagreeing: monkeypatching the production predicate broken
makes the cross-check fail —
`tests/eval/test_vocabulary_oracle.py::test_vocabulary_oracle_cross_check_detects_a_broken_production_predicate`.

## The two features

**Feature 1 — Infeasibility Negotiator.** Before any token is spent, test
deterministically whether a `ConstraintSet` admits any solution. If not, don't
let the model improvise — return minimal relaxations ranked by how many meals
each unlocks, with `MEDICAL`/`RELIGIOUS` constraints locked out of the
relaxation pool entirely. Business case: a silently-mediocre answer to an
impossible profile is a leading silent-churn driver in constrained meal
planning — the user blames the product, not their own constraints — and this
turns a dead end into a guided conversation for zero LLM cost.

**Feature 2 — Trust-Gated Escalation with a healing loop.** A weighted
composite (45% catalog coverage, 30% constraint-check completeness, 25% model
self-assessment) gates every commit, with a conjunctive veto so a confident
model cannot mask a critically thin deterministic signal. Below threshold, the
system refuses and names the failing signal. Business case: in a
health-adjacent product a wrong answer is a liability event, not a bad UX
moment, and every escalation becomes a permanent regression case via
`beatroot heal` — the same failure cannot recur silently.

## What the model may and may not do

| Tier | Contents | Model involvement |
|---|---|---|
| **T0 Invariants** | Allergen / medical / religious enforcement; nutrition arithmetic | None, ever |
| **T1 Hybrid judgment** | Retrieval, ranking, confidence fusion | Lexical + dense + model signals, explicit weights |
| **T2 Generative** | Free-text preference parsing, explanation prose | Yes |
| **T3 Confirmation** | Escalation, refusal, approval gates | Gated |

Enforced by two tests, not a comment: an AST import-graph walk fails the suite
if anything under `t0_invariants/`, `trusted/`, or `store/` imports
`reasoning/`; a second test (`tests/test_boundaries.py::
test_llm_permitted_false_skills_have_no_call_path_to_reasoning`) maps each
skill declaring `llm_permitted: false` to the module that actually
implements it and runs the same reachability check against it — not merely
a frontmatter-equality check on the YAML field. Details in `ARCHITECTURE.md`.

## Eval numbers — measured on this branch, this run

```
uv run pytest -q --cov=src/beatroot                656 collected, 651 passed, 5 skipped
                                                     coverage 94% (gate: 80%)
uv run ruff check .                                 all checks passed
uv run mypy --strict src                            no issues found, 73 files

uv run python -m beatroot.eval.runners.system       6/6 axes PASS (table above)
uv run python -m beatroot.eval.runners.components   recall@5 (full oracle)     0.9688
                                                     recall@5 (hard contract)  1.0000
                                                     retrieval leakage     0
                                                     feasibility accuracy  1.000
                                                     nutrition determinism 1.000
uv run python -m beatroot.eval.calibration          ECE = 0.1250 offline stub / 0.0122 live
                                                     BUT: all 38 live pairs land in ONE bin
                                                     [0.9,1.0), mean conf 0.988, accuracy
                                                     1.000, only 2 distinct confidences.
                                                     Read it as "when it commits it is
                                                     confident and right", NOT as calibration
                                                     across the range — it is single-bin and
                                                     effectively unmeasured. See EVAL_HISTORY.md.
```

The 5 skips are the Qdrant-store tests, skipped when `QDRANT_URL` is unset;
against a running container they pass (see Deployment, below) rather than
skip. `docker compose up` runs app + Qdrant together.

**Read the component numbers as evidence of honesty, not just performance.**
`retrieval leakage = 0` is independently re-derived — it does not call the
same code path retrieval uses to filter. Recall is reported against TWO
contracts because one number cannot honestly express it: `retrieve()`'s
legality gate enforces HARD (medical/religious/dietary) constraints only by
design — a soft budget or prep-time limit is ranking's job, not filtering's.
Against that actual contract recall is `1.0000`; against the stricter
full-constraint oracle it is `0.9688`.

**That `0.9688` is not "we added recipes and recall went up."** The catalog
grew from 100 to 174 recipes over the life of this project, and recall over
the same window went from `0.6872` to `0.9688` — but attributing the jump to
catalog size is wrong by roughly 8x. Measured by neutralising the two-pass
retrieval change under monkeypatch, on the *same* 174-recipe catalog: **+0.2506
recall from two-pass selection, only +0.0309 from the larger catalog.**
Two-pass retrieval (`retrieval/rerank.py`) prefers candidates that satisfy
soft constraints too, falling back to merely-legal ones only to top up to
`top_k` — built to fix a user-visible failure (a request naming five dislikes
returned no plan at all, while 55 catalog recipes satisfied every one of
them), and verified by that behaviour changing, not by this number moving.
`hard_only` stayed `1.0000` throughout — the hard gate never moved. Full
history in `EVAL_HISTORY.md`.

An earlier version of this file reported `recall@5 = 0.665` and justified it
as "the right kind of number for a token-hashing embedding stub". That
reasoning was wrong, and the way it was wrong is worth keeping. The benchmark
retrieved every case with one hardcoded query, `"a balanced meal"`, which
scores zero FTS5 hits against this catalog — so the lexical half of hybrid
retrieval was structurally dead for the entire measurement. Four config
sweeps that each moved recall by exactly `0.0000` are what exposed it; that
unanimous null result was the diagnostic. See `EVAL_HISTORY.md`. `feasibility_accuracy =
1.000` looked airtight the first time it was measured — until review found the
"independent" oracle it was checked against was calling the exact function
under test, twice. Rebuilt from scratch (reads catalog data, applies
constraint semantics, imports nothing from `t0_invariants`), and proven
capable of failing: inverting one evaluator makes the cross-check fail; that
test is `test_oracle_cross_checks_against_check_recipe`.

**Neither ECE number is a clean calibration result, and neither should be read
as one.** Offline, `model_self_assessment` — 25% of the trust weight — is a
pinned constant (0.5) from the stub provider, so `0.1250` validates the
deterministic 75% of the composite and says nothing about a real model's
confidence. Live, against real `azure/gpt-4o` self-assessment, `0.0122`
*looks* like the better number — but all 38 live COMMIT pairs land in one
reliability bin (`[0.9, 1.0)`) at only two distinct confidence values, so ECE
over one bin is close to unmeasured: 38 pairs landing in a single
high-confidence bin, all correct, scores near-zero identically to genuinely
well-spread calibration. Read `0.0122` as "when this system commits, it is
confident and right" — not as evidence the model's confidence is calibrated
across the range, which COMMIT-only sampling structurally cannot show. The
eval script prints both caveats itself, every run.

## Cost per plan and tokens saved

Offline, cost is genuinely **$0.00 per plan** — the stub provider is free, and
`/metrics` (`per_plan_usd`, `total_usd`, `plans`) reports it live rather than
a printed constant. With real credentials, `CostLedger` accumulates real USD
and token counts per stage from LiteLLM the same way.

**Tokens saved by short-circuiting is real behaviour, and `tokens_saved` is
now a populated number, not a permanent `0`.** An infeasible profile's trace
is two nodes — `FEASIBILITY -> NEGOTIATE` — against seven for a commit
(`... -> EXPLAIN -> VERIFY -> COMMIT`); the only node that calls a model,
`EXPLAIN`, structurally never runs. Same for an unknown-vocabulary profile:
`FEASIBILITY -> ESCALATE`, zero LLM-touching code between the request and the
refusal. Both short-circuit sites populate `CostRecord.tokens_saved` via
`agent.nodes._estimate_skipped_tokens`, which renders the actual `rerank` and
`explain` prompts against a real catalog sample and converts character count
to a token estimate at ~4 chars/token — a defensible estimate, not an
invented constant, and `/metrics` labels it as one via
`tokens_saved_estimate_method` rather than presenting it as a measured spend.
Live example against this branch: one infeasible `/recommend` request reports
`tokens_saved: 253` on the next `/metrics` read. The short-circuit itself is
proven (by the trace, and by `test_...counting_llm` asserting zero LLM calls
on that path); the *metric* of it is wired **and** populated, verified
end-to-end by `tests/api/test_routes.py::
test_metrics_reports_nonzero_tokens_saved_after_an_infeasible_request`.

## More

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the four trust tiers, state machine,
  enforcement tests, retrieval pipeline, evolution under scale.
- [`CUT_LIST.md`](CUT_LIST.md) — every non-goal, why it was cut, what it would
  cost to add — including decisions made mid-build (LangGraph over a
  hand-rolled state machine, LLM rerank over a cross-encoder, NumPy-dev /
  Qdrant-production) and the honest limitations still standing.
- [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md) — walkthrough beat sheet.

## Pages

Four, all served by the same FastAPI app: `/` (recommend), `/incidents`,
`/evals`, and `/docs` — a pan/zoom architecture diagram plus 12 downloadable
project files (this README, `ARCHITECTURE.md`, `CUT_LIST.md`, and the rest).
FastAPI's own interactive API docs live at `/api-docs` instead of the
framework default of `/docs`, which this app's own `/docs` page would
otherwise shadow.

## Deployment — verified, not aspirational

`docker build` (14/14 steps, first try), `docker run` (healthy `/health` in
11s), and `docker compose up` (app + Qdrant together, `/health` reporting
`vector_store: "qdrant"`) have all been run against this branch — this was
not true of an earlier version of this document, which reported the
container path as unverified because no Docker daemon was reachable in any
environment used at the time. `docker-compose.yml` uses the `env_file:
{path, required}` long form, which needs Compose ≥ 2.24; if an older Compose
rejects the file, `cp .env.example .env` first (making the referenced `.env`
actually exist sidesteps the version dependency) or upgrade Compose.

Qdrant itself is verified the same way, not just wired: all 7 Qdrant tests
(`tests/retrieval/test_qdrant_store.py` and `test_qdrant_retry.py`) pass
against a real `qdrant/qdrant:v1.12.0` container, leaving 5 collections and
57 server operations behind as evidence. The 5 tests in `test_qdrant_store.py`
skip cleanly with `QDRANT_URL` unset (and error, not skip, if it's set but
the server is unreachable) — that's the origin of the 5 skips in the pytest
line above. A live full-mode run against this branch
reports `provider: azure`, `llm: azure/gpt-4o`, `vector_store: qdrant`, `174`
recipes — the real-Azure, real-Qdrant path, end to end, not a config that
merely resolves.
