
## What it is

A meal-planning agent that takes a constraint profile, recommends a dish from a
local catalog, validates it deterministically, and explains the choice. Each
request runs a state machine — `COMPILE → FEASIBILITY → RETRIEVE → SCORE → TRUST
→ EXPLAIN → VERIFY → COMMIT` — that can terminate early in `NEGOTIATE` (no meal
satisfies the profile) or `ESCALATE` (the system cannot vouch for an answer)
rather than improvise one. Catalog: 174 recipes, 137 ingredients, 21 preset
profiles, 9 constraint kinds across 5 severities. Ships as a CLI, a FastAPI
service, and a dashboard.

## Quick start — 30 seconds, no keys, no network

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv
git clone https://github.com/Ansumanbhujabal/beetroot.git && cd beetroot
uv sync
uv run beatroot recommend "something warm with rice" --medical peanut
```

```
trace: FEASIBILITY -> RETRIEVE -> SCORE -> TRUST -> EXPLAIN -> VERIFY -> COMMIT
COMMIT — Vegan ragi dosa with coconut chutney
  kcal=527.57 protein=10.98g carbs=72.77g fat=22.63g fibre=9.04g  (coverage 1.00, provenance=computed)
  trust 0.88  (catalog_coverage=1.00, constraint_completeness=1.00, model_self_assessment=0.50)
  constraints satisfied: med0
```

With no provider credentials, `beatroot.settings` says so at `WARNING` and falls
back to a deterministic offline provider. Every code path still runs for real —
retrieval, feasibility, trust scoring, verification — only the model's own
opinion is a stub.

`--medical` / `--exclude` build typed constraints; `query` only feeds retrieval
ranking, so don't type an allergy into it and expect enforcement. Free text goes
through `--preferences` and can only ever *add* constraints, never remove one.

---

## Two ways to run the full application

Both serve the identical app — same API, same four dashboard pages. They differ
in one dependency and one deployment shape. Everything below was run end to end
from a clean clone.

| | Option 1 — local | Option 2 — Docker |
|---|---|---|
| Vector store | NumPy, in-process | **Qdrant**, real service |
| Runs | `uvicorn` on your machine | two containers |
| Needs | `uv` | Docker Compose ≥ 2.24 |
| Start-up | seconds | ~4 min first build, then seconds |
| Provider + Langfuse | yes | yes |
| Use it for | development, the fastest loop | the production-shaped run |

### Shared step — credentials

`.env` is gitignored and is **never in the clone**. Create it once; both options
read the same file.

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `BEATROOT_OFFLINE=0` | use the real provider rather than the offline stub |
| `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` | the chat model |
| `BEATROOT_LLM__MODEL` | e.g. `azure/gpt-4o` |
| `BEATROOT_LLM__EMBEDDING_MODEL` | `local` unless you have an embedding deployment |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | prompt management + tracing |
| `LANGFUSE_HOST` | `https://us.cloud.langfuse.com` or `https://cloud.langfuse.com` |

`LANGFUSE_HOST` is **not optional once the keys are set**: it defaults to the US
cloud, so EU-region keys authenticate against nothing and every trace vanishes
in silence. Leave all the Langfuse values blank and everything still runs —
prompts resolve from `prompts/*.md`, tracing is a clean no-op.

---

### Option 1 — Local, no Docker, no Qdrant

```bash
uv sync
uv run beatroot serve                 # http://localhost:7860
```

Add `--port 7899` if 7860 is taken. Confirm what actually answered:

```bash
curl -s localhost:7860/health | python3 -m json.tool
```

```json
{ "provider": "azure", "llm_model": "azure/gpt-4o", "vector_store": "numpy",
  "recipes": 174, "skills_locked": true,
  "tracing": { "langfuse_configured": true, "host": "https://us.cloud.langfuse.com" } }
```

`vector_store: numpy` is correct here — with `QDRANT_URL` unset the in-process
store is selected, and it needs no service. Then prove Langfuse specifically,
which credentials alone do not:

```bash
uv run beatroot obs check         # real authenticated API call, names the project
uv run beatroot prompts status    # 4/4 as @langfuse:vN, or @local:vN on fallback
uv run beatroot prompts push      # publish prompts/*.md at the `production` label
```

Now jump to **[Populate the dashboards](#populate-the-dashboards)** below.

---

### Option 2 — Docker + Qdrant + everything

```bash
docker compose up -d --build          # app + Qdrant; first build ~4 min
```

Compose waits for Qdrant's own healthcheck before starting the app, so there is
no start-up race. If port 7860 is taken:

```bash
BEATROOT_PORT=7861 docker compose up -d --build
```

That overrides the **host** port only; the container port stays 7860. Confirm:

```bash
curl -s localhost:7860/health | python3 -m json.tool
```

Identical to Option 1 except **`"vector_store": "qdrant"`** — that one field is
how you know the real service answered rather than the fallback. Same two
checks, run inside the container:

```bash
docker compose exec beatroot beatroot obs check
docker compose exec beatroot beatroot prompts status
```

Stop with `docker compose down`.

---

### Populate the dashboards

`/evals` and `/incidents` are **empty on a fresh install, by design.** They
render runtime artifacts, not shipped data: an eval result is something you
produce by running the suite, and an incident is something the system records
when it refuses. Neither is baked into the repo or the image.

Option 1 (local):

```bash
uv run beatroot eval system
uv run beatroot eval components
```

Option 2 (Docker):

```bash
docker compose exec beatroot beatroot eval system
docker compose exec beatroot beatroot eval components
```

One `eval system` run drives 33 golden cases through the real graph, so the
infeasible and unverifiable ones leave incidents behind — `/evals`, `/incidents`
and the drift ledger all fill in together. Re-run at any time; the pages read
the latest result.

### Where to look

| URL | What it is | Worth looking at |
|---|---|---|
| `/` | Recommend | Build a profile or pick one of 21 presets. The result card shows the trust composite broken into its three signals, the constraints satisfied in plain language, and the full ingredient list. Try the Vegan preset with a chicken query. |
| `/incidents` | Incident feed + drift ledger | Every refusal, infeasibility and nutrition-drift finding, with the constraint set that caused it. Empty until step above. |
| `/evals` | Eval scores | The six safety axes against their thresholds, the component metrics, and per-axis pass/fail. Empty until step above. |
| `/docs` | Architecture | Pan/zoom architecture diagram plus every project document as a download. |
| `/api-docs` | OpenAPI explorer | FastAPI's own docs — moved here so it does not shadow `/docs`. |
| `/health` | Dependency report | Names the provider, model, vector store, catalog size, skill-lock state and tracing config. Not a bare "ok". |
| `/metrics` | Cost + caches | Cost per plan and per stage, token counts, feasibility and embedding cache hit rates. |
| `/profiles` | Preset profiles | The 21 dietary presets as JSON, the same shape `POST /recommend` accepts. |

Two refusals worth demonstrating, because they are the point of the system: an
impossible profile returns `NEGOTIATE` with a ranked relaxation ladder and the
medical constraints locked out of it, and a constraint naming something the
catalog has never heard of returns `ESCALATE` rather than a confident answer it
never actually checked. Both cost zero model tokens.

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
<img width="1638" height="825" alt="Screenshot from 2026-09-02 14-23-43" src="https://github.com/user-attachments/assets/6bd106f0-2cdc-4902-a5a7-84026a0e1f3a" />

<img width="1638" height="825" alt="Screenshot from 2026-09-02 14-24-08" src="https://github.com/user-attachments/assets/b93986ea-1ff9-4ce5-b57b-52af9266bfe7" />


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
also passes against live Azure: 6/6 axes 1.000, 0 violations, p50 ~7.5-9s, p95
~11-13s, about $0.085 for the run. Latency is gated per execution mode —
2s/8s offline, 15s/25s live — because a live case makes two or three serial
model calls at roughly two seconds each, so one shared budget would either
fail every live run or let an offline regression through unnoticed. Eval design, oracle construction, and what these
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
