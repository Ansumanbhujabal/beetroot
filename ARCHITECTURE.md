# Architecture

For a reader who has not seen `docs/specs/2026-08-30-beatroot-design.md`. That
spec is the design intent; this document, and the numbers in `README.md`, are
what the code on this branch actually does.

## 1. The four trust tiers

| Tier | Contents | Model involvement |
|---|---|---|
| **T0 Invariants** | Allergen / medical / religious enforcement (`t0_invariants/constraints.py`); nutrition arithmetic (`t0_invariants/nutrition_math.py`); catalog feasibility (`t0_invariants/feasibility.py`) | None, ever |
| **T1 Hybrid judgment** | Retrieval, ranking, confidence fusion (`retrieval/`) | Lexical (BM25) + dense (cosine) + preference affinity, fused by explicit-weight RRF; LLM reranks only the already-legal top-k |
| **T2 Generative** | Free-text preference parsing (`agent.nodes.compile_node`, wired as the graph's `compile` node), explanation prose (`reasoning/`) | Yes |
| **T3 Confirmation** | Escalation, refusal, approval gates (`confirm/`) | Gated — the model's signal is 25% of a composite it cannot single-handedly clear |

T0 is deliberately small and boring on purpose: it is the only tier that has
to be. `NutritionFacts.provenance` is typed as the literal `"computed"` — the
type system, not a convention, refuses to represent a model-generated
nutrition value.

**Severity has four hard tiers, not two.** `Severity` (`contracts/core.py`) is
`MEDICAL`, `RELIGIOUS`, `DIETARY`, `GOAL`, `PREFERENCE`, and
`HARD_SEVERITIES = {MEDICAL, RELIGIOUS, DIETARY}` — the set `is_legal()`
enforces at the filtering gate; `GOAL` and `PREFERENCE` are ranking's job,
not filtering's, by design. `DIETARY` exists because its absence caused a
real incident: a user on the Vegan preset was served chicken. The taxonomy
had only `MEDICAL` and `RELIGIOUS` as hard tiers, and an ethical vegan is
neither, so veganism was encoded as `PREFERENCE` — which `is_legal()` does
not enforce. The engine behaved exactly as specified; the vocabulary had
nowhere honest to put the requirement. The same bug had a visible twin: Jain
was `RELIGIOUS` and therefore enforced (a Jain user never saw root
vegetables), while vegan — the same class of categorical dietary rule — was
not, for no principled reason beyond which tier happened to fit. Adding
`DIETARY` as a fourth hard tier, floored by the severity ratchet for
identity-shaped constraint kinds, removes the inconsistency rather than
patching the one visible symptom.

**Constraint vocabulary is registry-driven, not an if/elif chain.**
`@evaluator(kind, shape=..., parse=...)` (`t0_invariants/constraints.py`) is
the single source of truth for a constraint `kind`'s entire vocabulary in one
registration: how to evaluate it against a recipe, the one-line model-facing
description of its value shape (`kind_shapes()`, rendered straight into the
compile-constraints prompt), and the parser that turns a raw model value
into a validated `ConstraintValue` (`get_parser()`). Both `shape` and `parse`
are required — a kind registered without either fails at import time. This
exists because of a real incident: `exclude_cuisine` was advertised to the
model in the prompt but had no registered parser, so the model would name a
cuisine to avoid and the parser silently dropped it, producing a constraint
that was never enforced. "Advertised but silently dropped" is now impossible
by construction, not just tested against — nine kinds are registered today:
`budget_max`, `cuisine_affinity`, `exclude_cuisine`, `exclude_ingredient`,
`exclude_tag`, `max_prep_minutes`, `nutrient_range`, `require_any_tag`,
`require_tag`. One thing is deliberately *not* registry-driven: the severity
ratchet (a model-proposed category becoming a `Severity`, and the `DIETARY`
floor for identity-shaped kinds like veganism) stays hardcoded in
`agent.nodes` — an evaluator declares how to check and parse a kind, never
how hard it gets to consider itself, since that decision should not be
delegable to whoever adds the next kind. Line-level detail in
`docs/ARCHITECTURE_DEEP.md`.

## 2. The two enforcement tests

The tier boundary is a property the suite checks, not a claim in a document.

**`tests/test_boundaries.py` — import-graph test.** Walks the AST of every
module under `t0_invariants/`, `trusted/`, and `store/`; fails if any of them
imports `beatroot.reasoning`, directly or transitively. This took four fix
rounds during the build to get right, and each round found a real hole:
relative imports (`from . import reasoning`) evaded the first version
entirely because the module-name guard skipped a `None` module; the graph
walk was one hop deep, so `t0_invariants -> contracts -> reasoning` passed
silently; `__init__.py` module-naming produced a key mismatch that reopened
the relative-import hole one level down after the first fix. The final
version does a canonicalised BFS over the whole internal import graph and is
proven against reconstructed pre-fix code, not just read.

**Skill-manifest test.** Every skill file under `skills/` declares `tier`
and `llm_permitted` in YAML frontmatter; `tests/agent/test_skills_registry.
py::test_t0_skills_declare_no_llm` checks the frontmatter says what it
should, but a frontmatter field equalling `False` is a claim about a
string, not about reachability. The actual boundary proof is
`tests/test_boundaries.py::
test_llm_permitted_false_skills_have_no_call_path_to_reasoning`: it maps
each `llm_permitted: false` skill to the module that actually implements
it (`check_feasibility` -> `t0_invariants/feasibility.py`,
`compute_nutrition` -> `t0_invariants/nutrition_math.py`) and runs the
same canonicalised-BFS reachability check the import-graph test above uses,
against `beatroot.reasoning`, directly on that module. `skills-lock.json`
hashes each skill's full content (including `tier`/`llm_permitted`, so
changing either without touching the body still trips drift detection) and
`beatroot`'s container **refuses to start** if a skill file has drifted
from its lock — a false provenance claim (the audit log names skill
versions that never actually ran) is treated as worse than a crash.

Documented limitation on both: `importlib.import_module("beatroot.reasoning")`
is invisible to AST analysis. Neither test claims to catch a dynamic import.

## 3. The state machine

Built with **LangGraph's `StateGraph`**, not the hand-rolled machine the
original design spec called for. See `CUT_LIST.md` for the reversal and why.

```
START -> compile --[internal error]--> escalate [terminal]
              `--> feasibility --[infeasible]--> negotiate [terminal]
                        |          [unknown vocabulary]--> escalate [terminal]
                        `--[feasible]--> retrieve --[no legal candidates]--> escalate [terminal]
                                              `--> score --[uncheckable]--> escalate [terminal]
                                                       `--> trust --[below threshold / grey-band pause]--> escalate [terminal]
                                                                `--> explain --> verify --[drift/violation]--> escalate [terminal]
                                                                                      `--[clean]--> commit [terminal]
```

`compile` (`agent.nodes.compile_node`) is the graph's real entry node —
`g.add_edge(START, "compile")` in `agent/graph.py` — not `feasibility`. An
empty `preferences` field makes it a zero-cost no-op (no trace entry, no
model call); free text compiled into `PREFERENCE`-only constraints is what
feeds `feasibility` next. Every node past `compile` is exception-guarded the
same way (any unhandled exception routes to `escalate`), but `compile` is
the only one drawn with that edge above, since it is the only node with no
other conditional branch of its own on the happy path.

Three terminal states — `negotiate`, `escalate`, `commit` — plus a fourth
suspended state, `PENDING_REVIEW`, when trust lands in the medical grey band:
`interrupt_before=["commit"]` pauses the graph, `SqliteSaver` checkpoints it
to a **file-backed** database (not `:memory:` — an early version used an
in-memory checkpointer, which meant `beatroot resume` run as a second process
could never find the paused thread; this is exactly the class of bug the
durability claim exists to prevent, and it was found by the resumability test
actually forcing a process boundary rather than reusing one connection), and
`beatroot resume <thread_id> --approved/--rejected` or `POST
/recommend/{thread_id}/resume` continues it.

**`verify` runs after the model and can still route to `escalate`.** Every
number the model states in its explanation is diffed against catalog-computed
truth by the drift ledger; every constraint is re-checked against the same
`ConstraintSet` that produced the candidate. The model never gets the last
word — a generation-stage failure downgrades the outcome, it does not get
silently accepted.

**`settings.async_explanation` is ON by default** (`config/beatroot.yaml`;
it used to ship off). When it's on, `explain_node` submits the explanation
job and returns immediately with `state["explanation"] == ""` —
`verify_node` sees empty prose, so `detect_drift` finds nothing to check on
the synchronous path (there is nothing false in stating no numbers). The
real explanation is generated later by `ExplanationQueue._work` and served
via `GET /recommend/{id}/explanation`.

**This used to be a real gap — it closed (`d7910d4`).** Enabling
`async_explanation` for latency moved prose generation past the `verify`
gate, and for a while nothing downstream picked the check back up: the queue
worker stored the generated prose as `"ready"` with no grounding check at
all, so a live model stating a wrong calorie count would have reached the
diner unchallenged on the one surface a diner actually reads. Fixed by
having `_work` run `detect_drift` itself, against the same nutrition facts
and threshold `verify_node` uses, and land the finding as `"failed"` — not
`"ready"` — when drift is detected. Refusing to serve the prose is the right
call, not an overreaction: the recommendation itself (nutrition, trust,
constraint checks) was already fully determined and verified before the
explanation job was ever queued, so a failed explanation costs only the
prose. Tested both directions in
`tests/agent/test_async_explanation_is_grounded.py`: grounded prose is
served; a stub claiming 9000 kcal is not. The COMMIT itself was never
affected either way.

Every node is exception-guarded; an unhandled exception writes an incident
and an audit record rather than leaving a checkpoint stuck mid-graph with
neither a commit nor a refusal on record — the one outcome this system's
whole posture is built to avoid.

## 4. Retrieval pipeline, filter pushdown

```
ConstraintSet -> T0 structural filter (meta-tags, transitive expansion) -> legal candidate set
                                                                                 |
                         BM25 (SQLite FTS5, NOT clause)         --+             |
                         Dense (cosine, NumPy dev / Qdrant prod) --+-- RRF (k=60) -> two-pass select -> LLM rerank top-k
                         Preference affinity (per-tag EMA)       --+
```

**The ordering is the argument: hybrid retrieval never sees an illegal
candidate.** Filtering happens before ranking, not after — post-hoc filtering
forces safety and ranking quality to compete for the same top-k slots;
pre-filtering removes the tradeoff entirely. RRF fuses on rank position only
(scores are discarded), so no single signal can dominate by having a wider
numeric range than another. The hard gate (`is_legal`, HARD_SEVERITIES only)
still runs first and is unchanged by anything below.

**Two-pass selection (`retrieval/rerank.py`).** `is_legal()` enforces HARD
constraints only, by design — a soft budget or prep-time or `require_tag`
constraint is ranking's job, not filtering's. But treating "soft" as
"ignorable" produced a real user-visible failure: a request naming five
dislikes returned no plan at all, even though 55 catalog recipes satisfied
every one of the five. `select_top_k` now does two passes over the fused
ranking — first collecting only candidates that also satisfy every soft
constraint, then falling back to merely-legal candidates to top up to
`top_k` only if the first pass came up short. The hard gate never changes;
this changes which *legal* candidates get preferred. It was verified by that
behaviour changing, not by a metric — but it also moved `recall@5` (full
oracle) from `0.6872` to `0.9688`, and the catalog's growth from 100 to 174
recipes over the same period accounts for only `+0.0309` of that, measured
by neutralising two-pass selection under monkeypatch on today's catalog. The
remaining `+0.2506` is the algorithm. See `README.md` and `EVAL_HISTORY.md`.

**Filter pushdown, concretely.** In the NumPy dev path, illegal ids are
masked to `-inf` before `argsort`, so they cannot surface even under a
ranking bug. In the Qdrant production path (`retrieval/qdrant_store.py`),
the same exclusion is expressed as `Filter(must_not=[FieldCondition(...)])`
passed as `query_filter` into the index's own search call — the exclusion
happens *inside* the vector index, not by the caller enumerating and
discarding results in Python afterward. Both implementations are asserted to
rank identically given the same inputs (`tests/retrieval/test_qdrant_store.
py::test_numpy_and_qdrant_agree_on_top_result`). **This is now verified by an
execution, not only by a test that would run:** all 7 Qdrant tests (that one
plus 4 more in `test_qdrant_store.py`, plus 2 in `test_qdrant_retry.py`) have
passed against a real `qdrant/qdrant:v1.12.0` container, and `docker compose
up` brings up the app against Qdrant with `/health` reporting `vector_store:
"qdrant"`. Absent `QDRANT_URL` the 5 store tests still skip cleanly (and
error, not skip, if it's set but the server is unreachable) — that remains
the right default for a bare clone with no Docker host.

**Dense retrieval is a stub offline, and this is disclosed rather than
hidden — including in this full-mode deployment.** The offline provider
originally hashed the whole query string with SHA-256, so semantically
similar text produced unrelated vectors — dense retrieval was pure noise,
and it dragged a good lexical ranking down when fused (querying the literal
dish name "jeera rice" once ranked that recipe 93rd of ~100, back when the
catalog was that size). Replaced with a token-level hashing-trick embedding
(tokenise, hash each token into one of 1024 dimensions, tf-weight,
L2-normalise) — still zero dependencies and fully deterministic, but now
`cos(jeera rice, jeera rice) = 1.0` and `cos(jeera rice, chicken burrito) =
0.0`. It is a stand-in for a real embedding model, not a real one, and this
stays true even in the verified full-mode deployment: the Azure resource
behind it has a chat deployment but no embedding deployment, so dense
retrieval runs the same local hash embedder regardless of provider mode.
`recall@5` (full oracle) is `0.9688` (see `README.md` for the two-pass
attribution behind that number, up from an earlier `0.665`/`0.687`) — a
number this stub earns despite still being a stub, because two-pass
selection and the larger catalog both operate downstream of it.

## 5. Evolution under scale (spec §17)

**What bottlenecks first is not the LLM — it's the T0 feasibility scan.**
Naively it is `O(recipes x constraints)` per request: fine at the catalog's
current 174 recipes, fatal at 100,000. The fix already built: `t0_invariants` compiles tag-based
constraints into a bitmask over precomputed per-tag inverted indices
(`trusted/index.py`'s `TagIndex`), so checking a tag constraint is a bitwise
AND with a cardinality check, and the residual scan (for non-tag constraints
like `budget_max`) walks survivors via `mask & -mask` bit-iteration —
`O(popcount)`, not `O(catalog)`. This was itself a finding, not a plan: the
first version's docstring claimed cost scaled with constraint count, which
was false as written — the residual pass walked the *whole catalog* every
call. A test now counts `check_recipe` invocations directly so the
complexity claim is guarded, not merely asserted.

**Retrieval.** In-memory NumPy gives way to Qdrant with HNSW (already built
behind the `VectorStore` protocol, selected by `QDRANT_URL`, not a hard
dependency — see `CUT_LIST.md`); per-user pre-filters become Qdrant payload
filters so the vector search still never sees an illegal candidate. FTS5
gives way to OpenSearch.

**Caching.** `FeasibilityCache` keys on `ConstraintSet.fingerprint()`, not
`profile_id` — many users share constraint *shapes* even with different
profile ids, so this is a shared cache, not a per-user one. `EmbeddingCache`
keys on `(embedder_id, content_hash)` — the `embedder_id` half was added
after a real bug: keying on text alone meant a real-provider run could
silently reuse vectors an earlier offline run produced with a completely
different embedding, corrupting retrieval invisibly while every test stayed
green. Both caches are wired and exercised — `/metrics` reports live hit
rates for both, not fixed strings.

**Asynchrony.** Explanation generation moves off the request path (built,
opt-in via `settings.async_explanation`, `agent/async_explain.py`): the
verified recommendation returns immediately — nutrition, trust, and
constraint checks are already fully determined without the model — and prose
streams in after via `GET /recommend/{id}/explanation`. This is only possible
*because* the model never makes the decision, which is the tier boundary
paying an operational dividend rather than costing one. `settings.
async_explanation: true` is now the shipped default in `config/beatroot.yaml`
(it was off during earlier development, when this section's numbers came
from a simulated 700ms provider latency because no real provider ran).
Against the real deployment now verified — `azure/gpt-4o`, Qdrant — a
`/recommend` request is **4.1s** on the local NumPy store and **~5.2s**
against Qdrant. This is not primarily a network- or token-latency story: a
trivial `gpt-4o` call round-trips in **1.92s** on this deployment
*regardless of prompt size*, so end-to-end latency here is set by the
**number of serial model calls** in the graph (compile, rerank, explain),
not by how much text each one carries. That is what async explanation
removes from the request path — one of those serial calls, not a large
one — and is why it earns its place as the default despite the tradeoff
below.

**The tradeoff this used to carry was real and is now closed, not just
disclosed (`d7910d4` — see §3).** `verify` still can't drift-check the
explanation actually served on the async path, because it runs before the
queue-generated prose exists. What changed is that the queue itself now
runs the same `detect_drift` check `verify_node` would have, against the
same nutrition facts and threshold, and serves the finding as `"failed"`
rather than the ungrounded prose if drift is detected — so enabling this
path no longer trades away the safety check, only which component performs
it and when.

**Evaluation.** Component evals run per PR; the system suite runs nightly
against a growing synthetic corpus. Shadow mode scores a candidate ranking
change against live traffic without serving it.

**Observability.** Per-tier latency and cost attribution via LiteLLM-native
callbacks (`obs/tracing.py`, no-op with no Langfuse key — verified with a
fully blank environment, `env -i`); alert on trust-score distribution shift,
the leading indicator of catalog drift.

**Safety.** The healing loop's proposal queue gets a review UI and an
approval audit trail. The allergen axis (A1) stays a **count**, at any
scale — see `README.md` for why a percentage is the wrong unit for
irreversible harm.

## 6. Module map

```
beatroot/
  contracts/        Pydantic models crossing every boundary
  t0_invariants/     T0 — hard constraint enforcement, feasibility solver, nutrition arithmetic
  trusted/           catalog access, canonicalisation, meta-tag expansion, TagIndex
  retrieval/         T1 — BM25 + dense (NumPy/Qdrant) + RRF fusion + rerank
  reasoning/         T2 — LiteLLM-backed provider interface, prompts as content
  confirm/           T3 — trust scoring, escalation, refusal, approval gates
  agent/             LangGraph StateGraph wiring the tiers, SqliteSaver checkpoints, async explain
  skills/            *.skill.md declarative skill definitions + skills-lock.json provenance
  eval/              component + system runners, synth generation (independent oracle), verifiers, calibration
  store/             SQLite persistence, audit log, incidents, preference memory, caches
  heal/              incident clustering, proposal generation
  obs/               LiteLLM-native tracing, structured JSON logging with redaction, cost ledger
  api/               FastAPI adapter
  cli/               Typer adapter (recommend, resume, serve, incidents, heal, eval, synth)
  web/               single static HTML dashboard, no build step, no CDN
```

Three adapters (`api`, `cli`, `web`) over one core (`agent.MealPlanningAgent`).
The agent is a library; the API is one client and the CLI is another.
