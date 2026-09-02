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

Not "deterministic because models can't be trusted" — hybrid retrieval beats
pure keyword search, and a rules-only planner is a worse product. But you cannot
un-eat a peanut. Allergen enforcement, religious-restriction enforcement, and
nutrition arithmetic never touch a model. Retrieval, ranking, and explanation
prose do — under a filter the model cannot see past.

## What it is

A meal-planning agent that takes a constraint profile, recommends a dish from a
local catalog, validates it deterministically, and explains the choice. Each
request runs a state machine — `COMPILE → FEASIBILITY → RETRIEVE → SCORE → TRUST
→ EXPLAIN → VERIFY → COMMIT` — that can terminate early in `NEGOTIATE` (no meal
satisfies the profile) or `ESCALATE` (the system cannot vouch for an answer)
rather than improvise one. Catalog: 174 recipes, 137 ingredients, 21 preset
profiles, 9 constraint kinds across 5 severities. Ships as a CLI, a FastAPI
service, and a dashboard.

## Quick start

No keys, no network:

```bash
uv sync                          # add --extra qdrant for the Qdrant vector store
uv run beatroot recommend "something warm with rice" --medical peanut
```

With no provider credentials, `beatroot.settings` logs that at `WARNING` and
falls back to a deterministic offline provider (hashing-trick embeddings,
templated completions). Every code path still runs for real; only the model's own
opinion is a stub. `BEATROOT_OFFLINE=1` forces it even with keys present.

```
trace: FEASIBILITY -> RETRIEVE -> SCORE -> TRUST -> EXPLAIN -> VERIFY -> COMMIT
COMMIT — Vegan ragi dosa with coconut chutney
  kcal=527.57 protein=10.98g carbs=72.77g fat=22.63g fibre=9.04g  (coverage 1.00, provenance=computed)
  trust 0.88  (catalog_coverage=1.00, constraint_completeness=1.00, model_self_assessment=0.50)
  constraints satisfied: med0
```

`--medical` / `--exclude` build typed constraints; `query` only feeds retrieval
ranking, so don't type an allergy into it and expect enforcement. Free text goes
through `--preferences` (or `preferences` on `POST /recommend`, or the
dashboard's text box) and can only ever *add* `PREFERENCE`-severity constraints —
never medical or religious, never a removal.

`uv run beatroot serve` starts the same app — API plus dashboard — on
<http://localhost:7860>, keyless too.

## Full mode — Docker, Qdrant, a real provider, Langfuse

Six steps, in order. Everything below has been run end to end from a clean
clone.

**1. Prerequisites.** Docker with Compose ≥ 2.24 (`docker compose version`),
and `uv` if you also want to run the CLI or the tests outside the container:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Credentials.** `.env` is gitignored and is never in the clone — create it:

```bash
cp .env.example .env
```

Fill in `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` and set
`BEATROOT_OFFLINE=0`. For Langfuse prompt management and tracing, add
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` — the host is
**not** optional once the keys are set, because it defaults to the US cloud and
EU-region keys then authenticate against nothing, silently. Leave the Langfuse
values blank and everything still runs; prompts resolve from `prompts/*.md` and
tracing is a clean no-op.

**3. Start the stack.** Compose brings up the app and Qdrant together and waits
for Qdrant's healthcheck before starting the app:

```bash
docker compose up -d --build          # first build ~4 min
```

Port 7860 already taken? `BEATROOT_PORT=7861 docker compose up -d --build`
overrides the *host* port; the container port stays 7860.

**4. Confirm what actually answered.** A health check that only says "ok" is
decoration — this one names each dependency:

```bash
curl -s localhost:7860/health | python3 -m json.tool
```

Expect `provider: azure`, `vector_store: qdrant`, `recipes: 174`,
`skills_locked: true`, and `tracing.langfuse_configured: true`. Then prove the
Langfuse half specifically, which credentials alone do not:

```bash
docker compose exec beatroot beatroot obs check       # real authenticated API call
docker compose exec beatroot beatroot prompts status  # 4/4 @langfuse:vN, or @local:vN
```

`prompts push` publishes `prompts/*.md` to Langfuse at the `production` label;
`prompts status` reports where each one *actually* resolved from, so a fallback
to the local file is visible rather than assumed.

**5. Populate the dashboards.** `/evals` and `/incidents` are **empty on a
fresh install, by design.** They render runtime artifacts, not shipped data: an
eval result is something you produce by running the suite, and an incident is
something the system records when it refuses. Generate both:

```bash
docker compose exec beatroot beatroot eval system      # writes eval/last_run.json
docker compose exec beatroot beatroot eval components
```

That single run also drives 33 golden cases through the real graph, so the
infeasible and unverifiable ones leave incidents behind and `/incidents` and the
drift ledger fill in too. Re-run either command at any time; the pages read the
latest result.

**6. Look at it.** `/` recommend · `/incidents` · `/evals` · `/docs`
(pan/zoom architecture diagram plus downloadable documents). FastAPI's own API
explorer is at `/api-docs`, moved so it does not shadow this project's `/docs`.

Shut down with `docker compose down`.

## What the model may and may not do

| Tier | Contents | Model involvement |
|---|---|---|
| **T0 Invariants** | Allergen / medical / religious enforcement; nutrition arithmetic; feasibility | None, ever |
| **T1 Hybrid judgment** | Retrieval, ranking, confidence fusion | Lexical + dense + affinity at explicit weights; the model reranks only the already-legal top-k |
| **T2 Generative** | Free-text preference parsing, explanation prose | Yes |
| **T3 Confirmation** | Escalation, refusal, approval gates | Gated |

Two tests enforce this as code, not convention: an AST import-graph walk fails
the suite if anything under `t0_invariants/`, `trusted/`, or `store/` imports
`reasoning/`; a second maps every skill declaring `llm_permitted: false` to the
module that implements it and runs the same reachability check there, rather than
checking the YAML field against itself. Detail:
[`ARCHITECTURE.md`](ARCHITECTURE.md) §1–§2.

## The two chosen features

**Infeasibility Negotiator.** Before any token is spent, test deterministically
whether a `ConstraintSet` admits any solution. If it doesn't, return minimal
relaxations ranked by how many meals each unlocks, with `MEDICAL` and `RELIGIOUS`
locked out of the relaxation pool entirely. A silently mediocre answer to an
impossible profile is a quiet churn driver — the user blames the product, not
their own constraints. This turns a dead end into a guided conversation at zero
LLM cost.

**Trust-Gated Escalation with a healing loop.** A weighted composite — 45%
catalog coverage, 30% constraint-check completeness, 25% model self-assessment —
gates every commit, with a conjunctive veto so a confident model cannot mask a
critically thin deterministic signal. Below threshold the system refuses and
names the failing signal instead of guessing. In a health-adjacent product a
wrong answer is a liability event, not a bad UX moment, so every escalation
becomes a permanent regression case via `beatroot heal`.

## Observability and prompt management

All 4 prompts are published to Langfuse under the `production` label —
`beatroot prompts status` shows each resolving as `<name>@langfuse:v1`. Without
credentials they resolve `<name>@local:vN` and everything still runs. Rendering
stays local (`str.format`) deliberately, and a fetched template whose
placeholders are not all declared in the local file is refused at load, so a bad
remote edit is a startup log line rather than a mid-request `KeyError`.

Model calls are instrumented directly with the Langfuse SDK: one generation span
per stage, grouped into a session by request id, each carrying `promptName` and
`promptVersion` (native linkage), token usage, and cost.

LiteLLM's `langfuse_otel` callback was tried first and dropped: it exported
traces from short-lived processes and nothing from the long-running server, while
every diagnostic — the SDK's own `auth_check()` included — reported healthy. It
logs asynchronously and builds its tracer provider lazily, so a flush from
response middleware ran before the span existed. It is now deliberately
unregistered, pinned by a named constant and a test; a 5s periodic flusher in the
API lifespan keeps a long-running server's spans moving.

`GET /health` from a running full-mode container:

```
provider: azure          vector_store: qdrant     recipes: 174
llm_model: azure/gpt-4o  skills_locked: true
tracing: {langfuse_configured: true, host: https://us.cloud.langfuse.com,
          instrumentation: "langfuse-sdk (direct generation spans)"}
```

Cost is metered per stage and served live by `/metrics` (`per_plan_usd`,
`total_usd`, `plans`), not printed as a constant. A live `POST /recommend`
(peanut-allergic profile, free text "no dairy this week please") committed at
**$0.004745** — compile `$0.002427`, rewrite_query `$0.001254`, rerank
`$0.001064`, plus ~`$0.00086` when the async explanation completes — and the
trace costs reconcile with `/metrics` exactly.

## Numbers

Offline is the default: no credentials, no network, deterministic — and the
figures below are all **NumPy-store** measurements, which is what an unset
`QDRANT_URL` selects. That is stated because it is a variable, not a detail:
re-running the component suite against Qdrant inside the container gives
`recall@k` **0.972** rather than 0.988, on the same catalog and the same cases.
Neither number is wrong; they measure two different vector stores, and the
safety-critical ones do not move — `hard-only recall` stays 1.000 and retrieval
leakage stays 0 either way, because the legality gate runs before ranking in
both paths.

```
uv run python -m beatroot.eval.runners.system                33 golden cases
  A1 allergen 1.000/1.0 · A2 religious 1.000/1.0 · A3 injection 1.000/0.95
  A4 infeasibility 1.000/0.95 · A5 escalation 1.000/0.9 · A6 grounding 1.000/0.95
  0 hard-constraint violations (threshold 0) · p50 50ms · p95 395ms · $0.0000

uv run python -m beatroot.eval.runners.components
  recall@k 0.988 full oracle / 1.000 hard-only oracle · retrieval leakage 0
  feasibility accuracy 1.000 · nutrition determinism 1.000 · drift recall 1.000

uv run pytest              679 passed, 5 skipped, coverage ~91% (gate 80%)
uv run mypy --strict src   no issues, 73 source files
uv run ruff check .        all checks passed
```

The 5 skips are the Qdrant store tests, which skip when `QDRANT_URL` is unset.
Against a running container all 7 Qdrant tests pass:

```bash
docker compose up -d qdrant
QDRANT_URL=http://localhost:6333 uv run pytest tests/retrieval/test_qdrant_store.py \
                                               tests/retrieval/test_qdrant_retry.py
``` The suite is hermetic — a session fixture hides
the repo `.env` and strips provider credentials, so it cannot silently run
against a live provider on a machine holding real keys. The same system suite
also passes against live Azure: 6/6 axes 1.000, 0 violations, p50 7932ms, p95
9894ms, $0.0836 for the run. Eval design, oracle construction, and what these
numbers do *not* prove: [`EVAL_RESULTS.md`](EVAL_RESULTS.md).

## Documentation

| Document | What it covers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Trust tiers, enforcement tests, state machine, retrieval pipeline, evolution under scale |
| [`EVAL_RESULTS.md`](EVAL_RESULTS.md) | Eval design, the independent oracles, and the limits of every score |
| [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) | Hands-on walkthrough — scenarios run against a live server with observed output |
| [`CUT_LIST.md`](CUT_LIST.md) | What was deliberately not built, why, and what it would cost to add |
| [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) | Requirements coverage and the gap between "ready as a prototype" and "ready to serve users with allergies" |

## Honest limitations

- **The offline embedder is a stub, and stays one in full mode too.** The
  provider resource has a chat deployment but no embedding deployment, so
  `embedding_model` resolves to `local` — a token-level hashing-trick embedder,
  so dense retrieval is not semantic.
- **Calibration (ECE) is effectively unmeasured.** COMMIT-only sampling puts
  every pair in one high-confidence bin. Read it as "when it commits it is
  confident and right", not as calibration across the range.
- **The free-text compiler can over-classify.** In a live run, "no dairy this
  week please" was categorised `medical`. The severity ratchet floors severity
  but never caps it, so this errs safe — and makes a preference non-relaxable.
- **A6 refuses to run against the async explanation path.** `verify_node` sees
  an empty explanation and no numbers to diff, so `run_system` raises rather
  than reporting a meaningless 1.000.
- **The two enforcement tests cannot see `importlib.import_module`.** Dynamic
  imports are invisible to AST analysis; a boundary crossed that way is missed.
