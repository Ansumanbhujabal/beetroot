# Architecture

beatroot is a single Python process. One core — `agent.graph.MealPlanningAgent` — sits under
three thin adapters: a FastAPI HTTP API, a Typer CLI, and a static HTML dashboard served by that
same API. No adapter holds decision logic; each builds a `Container`
(`container.build_container`) and calls `container.agent.run(...)` / `.resume(...)`.

This is the architecture reference for what the code on this branch does.
`docs/specs/2026-08-30-beatroot-design.md` is the design intent; `CUT_LIST.md` records where the
build diverged from it; `EVAL_RESULTS.md` and `README.md` carry the measured numbers.

Scale of the thing described: 174 recipes, 137 ingredients and 21 preset profiles; 9 registered
constraint kinds; 6 skill files; 4 prompts; 33 golden eval cases; ~11.1k lines under `src/`
against ~11.5k under `tests/`.

---

## 1. The four trust tiers

| Tier | Contents | Model involvement |
|---|---|---|
| **T0 Invariants** | Constraint enforcement (`t0_invariants/constraints.py`), catalog feasibility (`feasibility.py`), vocabulary validation (`vocabulary.py`), nutrition arithmetic (`nutrition_math.py`), catalog access (`trusted/`), persistence (`store/`) | **None, structurally** — no module here can import `beatroot.reasoning`, directly or transitively (§4) |
| **T1 Hybrid judgment** | Retrieval and ranking (`retrieval/`) | Yes, but only over a set T0 already proved legal: embeddings, query rewrite and LLM rerank all operate inside a filter the model cannot see past |
| **T2 Generative** | Free-text compilation (`agent.nodes.compile_node`), explanation prose (`explain_node`, `agent/async_explain.py`), provider wrapper and prompts (`reasoning/`) | Yes — this is the model's tier |
| **T3 Confirmation** | Trust scoring and the escalation gate (`confirm/`) | Gated: the model's self-assessment is 25% of a composite it cannot single-handedly clear (§7) |

The boundary sits here because the T0 questions are irreversible-if-wrong. An allergen exclusion,
a religious rule and a nutrition total are total functions of catalog data; a model is a
probabilistic function. T1 and T2 are the opposite kind of problem — a slightly-off ranking or a
clumsy sentence is a UX defect, not a safety incident.

**T0 is deliberately small.** It is the only tier that has to be correct, so it is the only tier
kept boring: pure functions over plain data, no catalog loaders, no I/O, and no settings lookups
— `feasibility.py` takes `max_relaxation_subset_size` as a parameter rather than reading global
config, so an invariant's answer never depends on ambient process state.

The type system carries the most important half. `contracts/nutrition.py:NutritionFacts.provenance`
is typed `Literal["computed"]` — one legal value, so a model-generated nutrition value is not
forbidden by convention, it is **unrepresentable**.
`tests/test_boundaries.py::test_nutrition_facts_cannot_be_constructed_from_model_output` asserts
the annotation's args are exactly `("computed",)`, so widening that `Literal` later breaks a test
rather than quietly opening a door.

---

## 2. The severity taxonomy

`contracts/core.py:Severity` has five members; `HARD_SEVERITIES` — the set `is_legal()` filters
on — has three.

| Severity | Hard | Enforced by | Relaxable |
|---|---|---|---|
| `MEDICAL` | yes | `is_legal()`, at the retrieval gate | never |
| `RELIGIOUS` | yes | `is_legal()`, at the retrieval gate | never |
| `DIETARY` | yes | `is_legal()`, at the retrieval gate | never |
| `GOAL` | no | ranking (`retrieve()`'s first pass, §6) and `check_recipe()` at `verify` | yes |
| `PREFERENCE` | no | ranking (`retrieve()`'s first pass, §6) and `check_recipe()` at `verify` | yes |

`rank_relaxations` (`t0_invariants/feasibility.py`) iterates `cs.soft()` only, so no hard
constraint is reachable by the relaxation ladder — a property of the loop, not a check on its
output. `Negotiation.locked` carries the hard constraint ids so a user sees they were protected,
not ignored.

**Why `DIETARY` is a fourth hard tier.** A user on the Vegan preset was served chicken. Veganism
was a seven-tag `PREFERENCE` denylist (dairy, egg, honey, gelatin, …); the catalog carries no
`chicken` tag, so nothing matched, and the dishes it did block were blocked incidentally, for
containing dairy. That is the general failure: **a denylist over an open world fails open**, since
anything nobody enumerated is admitted by default and the catalog keeps growing.
`require_tag` / `require_any_tag` invert the direction — a dish is excluded *until* positively
tagged — so an allowlist fails closed (`require_any_tag` covers a disjunctive identity one tag
cannot express: a pescetarian eats vegetarian food *or* fish, and those tag sets are disjoint).
But the primitive was only half of it: the taxonomy had no honest hard tier for an ethical vegan,
who is neither `MEDICAL` nor `RELIGIOUS`, so `PREFERENCE` was all that was left — and `is_legal()`
does not enforce soft constraints. The inconsistency was already in the same data: `jain`, the
same class of categorical rule, was `RELIGIOUS` and therefore enforced, so a Jain user never saw a
root vegetable while a vegan user saw chicken, for no principled reason beyond which pre-existing
tier happened to fit. `DIETARY` closes both gaps at once.

---

## 3. The constraint registry

`t0_invariants/constraints.py` dispatches evaluation through a registry, never an `if`/`match`
chain. `@evaluator(kind, shape=..., parse=...)` registers a kind's **entire vocabulary** in one
call:

| Declared | Via | Consumed by |
|---|---|---|
| how to check a `Constraint` against a `Recipe` | the decorated function | `_evaluate` → `check_recipe` / `is_legal` |
| the one-line, model-facing description of the value shape | `shape=` | `kind_shapes()` → rendered into `prompts/compile_constraints.md` by `compile_node` |
| how to turn one raw model item into a validated `(value, nutrient)` pair, or reject it | `parse=` | `get_parser(kind)` → `agent.nodes._parse_compiled_constraint` |

Both `shape` and `parse` are **required keyword arguments with no default**, so a kind registered
with an evaluator alone fails at import time. That forecloses a bug class that had already
shipped: `exclude_cuisine` was registered and advertised to the model, but the separate
hand-maintained `if`/`elif` parser chain had no branch for it, so a correctly proposed
`exclude_cuisine` constraint fell through the implicit `else` and was **silently dropped** — no
error, no log line, no failing test. From outside that reads as "the model ignored the user,"
far harder to chase than "the parser has a missing branch." Advertised-but-dropped is now
impossible by construction rather than tested against.

`_parse_compiled_constraint` no longer knows any kind's value shape; it looks up
`get_parser(kind)` and calls it, passing a `ConstraintVocabulary` (known tags, known cuisines) as
plain data so a parser can validate a real tag name without T0 importing a catalog loader. An
**unregistered** kind evaluates to `"uncheckable"`, never `"satisfied"` — and an uncheckable *hard*
constraint makes a recipe illegal, because a meal that cannot be proved safe is not served.
Registered today: `budget_max`, `cuisine_affinity`, `exclude_cuisine`,
`exclude_ingredient`, `exclude_tag`, `max_prep_minutes`, `nutrient_range`, `require_any_tag`,
`require_tag`.

**Deliberately not registry-driven: the severity ratchet.** It stays in `agent/nodes.py`.
`_CATEGORY_SEVERITY` maps the model's proposed `category` string to a `Severity` — the model
never writes a raw severity — and `_IDENTITY_KINDS` floors `require_tag` / `require_any_tag` at
`DIETARY` regardless of the category supplied. An evaluator declares how to check and parse a
kind, **never how hard it gets to consider itself**; letting a registration set its own severity
would let whoever adds the next kind ship a lenient default for something that should be able to
carry a `MEDICAL` constraint. The ratchet keys on the constraint's *kind*, not on any phrase in
its value, so it generalises to identities nobody wrote a test for: a model mislabelling an
identity claim as a preference cannot make it unenforced, and an unrecognised category drops the
item rather than defaulting it soft (under-enforcing an allergy) or hard (over-refusing a
dislike).

`contracts/core.py:ConstraintKind` stays a hand-maintained `Literal` of the nine strings — a
disclosed compromise, since generating it from `registered_kinds()` would create the exact import
cycle the tiering forbids.

---

## 4. The two enforcement tests

**Import-graph reachability** (`tests/test_boundaries.py`). `_internal_graph` parses the AST of
every `.py` file under `src/beatroot/` and builds a directed internal-import graph, resolving
three things a text search or a one-hop check would miss: relative imports (level-aware),
`__init__.py`-as-its-own-package naming (an `__init__.py` module *is* its package, not a child of
it), and `from X import Y` possibly naming a submodule rather than a symbol; a module-name
collision raises rather than silently dropping one file's edges. `_reaches` then runs a BFS from
every module under `t0_invariants/`, `trusted/` and `store/` for a path of any length to
`beatroot.reasoning`, returning the offending chain so a failure names it. Two synthetic file
trees built in `tmp_path` prove the BFS catches indirect violations rather than only running clean
against the real repo.

**Skill-manifest reachability** (`test_llm_permitted_false_skills_have_no_call_path_to_reasoning`).
Every file under `skills/` declares `tier` and `llm_permitted` in YAML frontmatter. Asserting the
frontmatter says `false` is a claim about a string —
`tests/agent/test_skills_registry.py::test_t0_skills_declare_no_llm` does that much. The boundary
proof maps each `llm_permitted: false` skill to the module that implements it
(`check_feasibility` → `beatroot.t0_invariants.feasibility`, `compute_nutrition` →
`beatroot.t0_invariants.nutrition_math`) and runs the same BFS directly against that module. It
also asserts the *set* of `llm_permitted: false` skills still equals the mapping's keys, so
adding one without extending the mapping fails loudly instead of going unchecked.

Provenance rides alongside. `Skill.digest()` (`agent/skills_registry.py`) hashes `tier`,
`llm_permitted`, `priority` and `body` — never `id`, never a path or mtime — into
`skills-lock.json`, so flipping `llm_permitted` without touching the body still trips drift
detection. `container.build_container` **refuses to start** on drift: an audit record naming a
skill version that never ran is a false provenance claim, worse than a crash.

**Honest limitation, in the test module's own docstring:**
`importlib.import_module("beatroot.reasoning")` is invisible to AST analysis. Both tests are
call-graph proofs, not runtime instrumentation; neither claims to catch a dynamic import.

---

## 5. The state machine

Built with LangGraph's `StateGraph` (`agent/graph.py:build_graph`) over ten node functions from
`agent/nodes.py:make_nodes`. `CUT_LIST.md` records the reversal from the spec's hand-rolled
machine and why.

```
START -> compile --[internal error / out_of_scope]--> escalate [terminal]
              `--> feasibility --[infeasible]--> negotiate [terminal]
                        |          [unknown vocabulary]--> escalate [terminal]
                        `--[feasible]--> retrieve --[no legal candidates]--> escalate [terminal]
                                              `--> score --[uncheckable]--> escalate [terminal]
                                                       `--> trust --[below threshold / weak-signal veto]--> escalate [terminal]
                                                                `--> explain --> verify --[drift / violation]--> escalate [terminal]
                                                                                      `--[clean]--> commit [terminal]
```

`compile` is the real entry node (`g.add_edge(START, "compile")`), not `feasibility`. With empty
`preferences` it is a genuine no-op — no model call, no cost, no trace entry. With free text, one
call does two jobs: it emits an open-vocabulary `constraints` list validated against
`registered_kinds()` and the catalog's live tag/cuisine vocabulary read fresh at call time (a new
tag or cuisine widens what the compiler can express with zero code and zero prompt change), and it
judges `in_scope`. `false` short-circuits to `ESCALATE(reason="out_of_scope")` in the same call —
no second round trip, no keyword blocklist; anything else defaults to in-scope, because refusing a
genuine food request is the worse failure. Compiled constraints are **appended**
(`[*existing, *added]`): no existing constraint's id, value or severity is ever read, edited or
removed, so "ignore my allergy" has no representable action here.

**Three terminals — `negotiate`, `escalate`, `commit` — plus one suspended state.** The graph
compiles with `interrupt_before=["commit"]`, so every run stops before committing and persists
full `PlanState` to `SqliteSaver` keyed by `thread_id`. Whether the pause is *shown* is decided
per run by `MealPlanningAgent._settle` from `needs_approval`, which `trust_node` sets only in the
medical grey band (§7); otherwise `_settle` re-invokes immediately and the caller never sees it.
When it does pause, the API reports `terminal_state: "PENDING_REVIEW"` — distinct from all three
graph terminals — and an audit record with `terminal_state="AWAITING_APPROVAL"` is written *at the
moment of pausing*, since retrieve/score/explain already spent real tokens and an abandoned thread
must not be the one path where spend goes unrecorded.
`container._build_checkpointer` supplies a **file-backed** connection on its own connection, which
is load-bearing: each CLI command is its own process, so `beatroot resume` can only find a thread
whose checkpoint outlived the process that paused it. `thread_state` classifies a thread as
`unknown` / `paused` / `resolved` before either adapter calls `.resume()`, and
`resume(approved=False)` does not branch by hand — `update_state(..., as_node="verify")` makes the
checkpoint look like `verify` produced an `ESCALATE`, so the graph's own conditional edge routes
to the real `escalate` node.

**Every node is exception-guarded.** `_guarded` wraps the seven non-terminal nodes: an unhandled
exception becomes `Escalation(reason="constraint_uncheckable", failing_signal="internal_error")`
and the following conditional edge routes it to `escalate`. `_guarded_terminal` wraps
`commit`/`negotiate`/`escalate`, whose edges run unconditionally to `END` with nothing downstream
to catch them, so it records the incident and audit itself. This rules out the outcome the system
cannot afford: a thread stuck mid-graph with neither a commit nor a refusal on record.

**`verify` runs after the model and can still escalate.** It re-runs `check_recipe` on the
*chosen* recipe against the same `ConstraintSet` that produced the candidates, and runs
`eval.verifiers.nutrition_drift.detect_drift` over the explanation against catalog-computed
`NutritionFacts` at the `verifiers.nutrition_drift_pct` tolerance. Either a re-violation or a
drift finding produces `Escalation(reason="verification_failed")` carrying the spend so far. The
model never gets the last word.

**The async-explanation grounding check.** `async_explanation: true` is the shipped default. On
that path `explain_node` submits the rendered prompt to `ExplanationQueue` and returns with
`explanation == ""`, so `verify_node` has no prose to diff and skips `detect_drift` explicitly
rather than relying on a regex finding nothing in an empty string. Moving prose generation off the
request path for latency moved it **past the one gate that checks its numbers**. The check follows
the prose to where it is now produced: `ExplanationQueue._work` runs the identical `detect_drift`
call, against the same nutrition facts and tolerance, before the entry is marked ready. Drift
lands the entry as `"failed"` — never `"ready"` — so `.get()` returns `None` and
`GET /recommend/{id}/explanation` serves no prose rather than ungrounded prose. That costs only
the prose: the recommendation was fully determined and verified before the job was queued, so a
failed explanation can never turn a COMMIT into an ESCALATE
(`tests/agent/test_async_explanation_is_grounded.py` covers both directions). The consequence:
**"the explanation is grounded" is a property of a pair of code paths** — `verify_node` when sync,
`ExplanationQueue._work` when async — which is also why the A6 eval axis refuses to score an
async-wired agent rather than report a meaningless 1.000.

Two supporting properties of the same design: `explain_node` renders the explain prompt once (via
`_describe_satisfied`, which turns constraint ids into human language) and hands the text over
verbatim, so the queue cannot leak raw ids onto the surface a diner reads; and `PlanState` accrues
`trace` and `cost` through reducers (`operator.add`, `merge_cost`), so both are the framework's
bookkeeping rather than conventions each node must honour.

---

## 6. Retrieval

```
ConstraintSet -> hard exclude_tag values pushed into both stores
                                 |
    rewrite_query()  (LLM; skipped on empty query; degrades to the original on any failure)
                                 |
        BM25 (SQLite FTS5, NOT clause)     --+
        Dense (NumPy dev / Qdrant prod)    --+-- RRF (k=60) -> two-pass select -> llm_rerank
        Preference affinity (per-tag EMA)  --+
```

**Filter pushdown, concretely.** Hard `exclude_tag` values are compiled into each store before
either ranks anything:

- `retrieval/lexical.py:lexical_search` appends a `NOT ("tag" OR …)` clause to the FTS5 `MATCH`
  expression itself, so an excluded row is never scored. Query text is stripped to alphanumeric
  tokens and tags quoted with FTS5's doubling rule, so neither is readable as query syntax.
- `retrieval/dense.py:DenseIndex.search` masks excluded rows to `-inf` *before* `np.argsort`, so a
  ranking bug cannot resurface them.
- `retrieval/qdrant_store.py` expresses the same exclusion as
  `Filter(must_not=[FieldCondition(key="tags", match=MatchAny(...))])` passed as `query_filter`
  into the index's own search call — filtered *inside* the index, not enumerated and discarded in
  Python afterward. Both stores are asserted to agree on the top result given identical inputs.

The ordering is the argument: hybrid retrieval never sees an illegal candidate. Post-hoc filtering
makes safety and ranking quality compete for the same top-k slots; pre-filtering removes the
tradeoff entirely.

**RRF fuses on rank position only.** `retrieval/fusion.py:rrf` adds `weight / (k + rank + 1)` per
source and discards the incoming score once it has established order. BM25 scores, cosine
similarities and a summed affinity live on incomparable scales, so no signal can dominate by
having a wider numeric range. `k` (60) and the weights (lexical 1.0, dense 1.0, affinity 0.5) come
from `settings.retrieval`, never literals in the module; ties break on id, so the order is
deterministic.

**Two-pass selection** (`retrieval/rerank.py:retrieve`). `is_legal()` gates hard constraints only
— soft constraints are ranking's job by design. But nothing was doing that job: a profile stating
five dislikes got back a dish violating two of them, and `verify_node`'s `check_recipe` recheck
(which evaluates *every* constraint) then escalated. "Ranking's job failing" was not a softer
outcome for that user; it was **no plan at all**, on a catalog holding recipes that satisfied all
five. `retrieve` now walks the fused list once, hydrating lazily, sorting survivors into
`preferred` (legal *and* `check_recipe(...).ok`) and `fallback` (legal, violates a soft
constraint), and returns `preferred` topped up from `fallback` only if it falls short of `top_k`.
The hard gate is unchanged and still runs first. It is deliberately a preference, not a second
filter: a diner whose dislikes cannot all be met still gets a plan with the violation surfaced at
`verify`, rather than the refusal promoting soft constraints to hard would produce.

`llm_rerank` reorders that list and nothing else: a missing, negative, out-of-range or non-integer
`choice_index` falls back to the fused top rank rather than raising or indexing blindly, and a
missing `self_assessment` degrades to `trust.weak_signal_floor`.

Two component metrics are reported separately and deliberately not conflated:
`retrieval_recall_at_k` against a full oracle (hard *and* soft) is **0.988**;
`retrieval_recall_at_k_hard_only`, against the contract `retrieve()` actually promises to filter
for, is **1.000**. `retrieval_leakage` — a candidate an independently re-derived verifier
(`eval/verifiers/hard_constraint.py`, which never calls `is_legal()`) finds violates a hard
constraint — is the headline safety number and reads **0**.

**The dense embedder is a stub, and stays one even in full mode.** `LLMClient._offline_vector` is
a token-level hashing trick: tokenise, SHA-256 each token into one of 1024 buckets, add ±1 by a
sign bit so collisions cancel rather than compound, L2-normalise. It replaced a whole-string hash
that gave near-identical text unrelated vectors — dense retrieval as anti-signal rather than
merely weak — and rewards lexical overlap in a dense-vector shape, but it is not semantic. This
holds in the verified full-mode deployment: the Azure resource has a chat deployment and no
embedding deployment, so `embedding_model` resolves to `local` and the same hash embedder runs
regardless of provider. A real embedding model is a config change.

---

## 7. Trust and escalation

**The composite** (`confirm/trust_score.py:score`). Weights come from `settings.trust.weights`,
validated at load to sum to exactly 1.0, so a misconfigured weighting is rejected at process start
rather than silently rescaling every score:

| Signal | Weight | Definition | Deterministic |
|---|---|---|---|
| `catalog_coverage` | 0.45 | `NutritionFacts.coverage` — mass-weighted fraction of the recipe's ingredient grams with usable nutrition data | yes |
| `constraint_completeness` | 0.30 | `(satisfied + violated) / total` — how many constraints were *conclusively* evaluated; `1.0` when there are none | yes |
| `model_self_assessment` | 0.25 | the model's reported confidence, clamped to `[0,1]`; `0.5` when missing or `NaN` | no |

75% deterministic is the point: a confident model must never rescue an answer the catalog does not
support. A missing self-assessment is neutral, not confident, so a silent model cannot inflate its
own score by omission. `NaN` maps to neutral rather than clamping, because comparisons against
`NaN` are always false and `min`/`max` would pass it into `TrustReport.composite`, whose own
`Field(ge=0, le=1)` would crash the gate that exists to contain exactly this output.

**The weak-signal veto.** `score()` also names `failing_signal`: whichever *deterministic* signal
is weakest, set whenever it falls below `weak_signal_floor` (0.5), independent of where the
composite lands. `model_self_assessment` is never eligible, so a confident model can never be
blamed for a refusal it did not cause.

**The gate is conjunctive** (`confirm/escalation.py:gate`):

```python
if trust.composite >= limit and trust.failing_signal is None:
    return None  # pass
```

The arithmetic is why both halves are needed, and it is exact. `constraint_completeness` (0.30)
plus `model_self_assessment` (0.25) sum to **0.55 — precisely the default `refusal_threshold`** —
and the threshold check is inclusive at the boundary. So a fully-checked constraint set plus a
maximally confident model reaches the gate *no matter how thin catalog coverage is*: a numeric
gate alone would let a confident model rescue an answer with near-zero catalog support. The veto
closes it, because `catalog_coverage` below the floor sets `failing_signal` regardless of the
composite, and the gate requires both. A refusal names which signal failed — a generic apology
teaches the user nothing and teaches the healing loop nothing.

**The medical grey band is a separate, narrower gate** (`agent.nodes.trust_node`). A composite
that clears the refusal gate but lands within `medical_review_band` (0.15) above it, **on a
profile carrying a `MEDICAL` hard constraint**, sets `needs_approval` and routes to the
human-confirmation pause of §5. Every other profile in the identical numeric band auto-resumes —
an agent that pauses on every request is not a safety gate. The condition checks `Severity.MEDICAL`
specifically, not `c.is_hard`, so a grey-band profile whose only hard constraint is `DIETARY` or
`RELIGIOUS` auto-resumes. A deliberate scope decision: the cost of human review is calibrated
against an allergy or condition being got wrong, not against every category enforced as
non-relaxable.

(`neutral_model_default` and `weak_signal_floor` both read 0.5 today but are distinct knobs; they
must not be retuned together.)

---

## 8. Observability and prompt management

**How a prompt resolves** (`reasoning/prompts.py`). Two sources in a fixed order: Langfuse when
configured, fetched by name at the `production` label; the local `prompts/*.md` file always, as
the fallback for every failure mode — no credentials, no network, a fetch error, an unpublished
prompt, or a remote prompt this code refuses. The local file is not a bootstrap copy: it is the
authoritative declaration of the prompt's contract (its `inputs` and `stage`) and what a bare
clone runs on. `beatroot prompts push` publishes; `beatroot prompts status` shows which source
each prompt resolved from — all four read `<name>@langfuse:v1` with credentials,
`<name>@local:vN` without.

**Rendering stays local**, deliberately. Langfuse can compile a prompt itself with mustache
`{{var}}`; this code uses `str.format`, because `prompts/compile_constraints.md` ends with a
literal JSON example whose braces are escaped for `str.format`. Under mustache the same characters
mean interpolation, so a second templating engine would silently corrupt the one prompt the whole
free-text path depends on. Langfuse is the versioned *store*; rendering is one engine with one set
of escaping rules, online and off.

**A fetched prompt can be refused.** A remote prompt is text someone can edit in a web UI outside
code review, and `str.format` fails at *render* time — inside a graph node, mid-request, after
tokens are already spent. So every `{placeholder}` in a fetched template must be declared in the
local file's `inputs`; an undeclared one is rejected at load, logged at `WARNING`, and the local
file used. A bad remote edit becomes a startup log line, not a mid-request `KeyError`.

**Tracing** (`obs/tracing.py`). Model calls are instrumented directly with the Langfuse SDK:
`observe_generation` opens a generation span before the request and closes it after, on the
calling thread. `propagate_attributes(session_id=…)` groups every generation serving one request
into one Langfuse session keyed by request id, so "what did answering this question cost" is a
number you read rather than reassemble. `prompt_ref_client` passes the raw Langfuse prompt object
into the span for native prompt-version linkage rather than a metadata string, and
`record_generation_result` attaches token usage and the cost this system already computed. Tracing
never takes a request down — a span that cannot open logs a warning and the call runs untraced.
Since the SDK flushes only on `flush()` or at process exit, `_PeriodicFlusher` (5s, started in the
API lifespan, flushing again on shutdown) is what stops a server from holding spans forever.

**Why LiteLLM's callback is not used** (full detail in the module docstring). `success_callback`
exported from short-lived scripts and nothing from the server, with every diagnostic healthy: a
callback name driving the removed v2 SDK failed non-blockingly inside LiteLLM; `auth_check()`
returns True against both Langfuse regions, hiding a wrong host; and LiteLLM logs success
asynchronously while OpenTelemetry force-flushes only at process exit, so middleware flushing ran
before the span existed. Registering it now would also double-count cost, so
`UNUSED_LITELLM_CALLBACK` and a test pin the decision.

Cost is measured, not estimated, on paths that spend — `/metrics` reports `per_plan_usd`, which
trace costs reconcile with — and the one estimate is labelled as one
(`tokens_saved_estimate_method`).

---

## 9. Evolution under scale

**What bottlenecks first is not the LLM — it's the T0 feasibility scan.** Naively
`O(recipes x constraints)` per request: invisible at 174 recipes, fatal at 100,000, and it sits on
the path most requests reach *before* any model call. The fix is already built.
`trusted/index.py:TagIndex` is an inverted index of `tag -> bitmap over recipe positions`, built
once at catalog load, so excluding a tag is one bitwise AND-NOT independent of catalog size and
`iter_ids` walks survivors with `mask & -mask` at `O(popcount)`. Be precise about what that buys:
only `exclude_tag` contributes to the mask. Every other kind goes through a residual `check_recipe`
pass over the mask's survivors — `O(survivors x non-tag constraints)`, not
`O(catalog x constraints)`: a real win once the mask has narrowed the field, but not free, and the
module says so. That distinction was itself a finding — the first docstring claimed cost scaled
with constraint count while the residual pass walked the whole catalog every call — so a test now
counts `check_recipe` invocations directly. `TagIndex.require_tags` exists for the inclusion-bitmap
optimisation `_survivors` does not yet take.

The vocabulary check runs before any of it (`t0_invariants/vocabulary.py`): membership tests make
a value naming nothing in the catalog vacuously "satisfied" for every recipe, silently certifying
a profile nobody verified. It checks every severity — a `PREFERENCE`-level unknown must escalate
exactly like a `MEDICAL` one, or the hole reopens one severity down.

**Retrieval.** In-memory NumPy gives way to Qdrant with HNSW — already built behind the
`VectorStore` protocol, selected by `QDRANT_URL`, not a hard dependency. Per-user pre-filters
become Qdrant payload filters, so the vector search still never sees an illegal candidate. FTS5
gives way to OpenSearch.

**Caching.** `FeasibilityCache` keys on `ConstraintSet.fingerprint()`, not `profile_id` — many
users share constraint *shapes*, so it is a shared cache, and an empty cached survivor list is a
legitimate answer rather than a miss. `EmbeddingCache` keys on `(embedder_id, content_hash)`; the
`embedder_id` half exists because keying on text alone let a real-provider run reuse vectors an
offline run produced with a different embedder, corrupting retrieval invisibly while every test
stayed green. Both report live hit rates on `/metrics`.

**Asynchrony.** Explanation generation is already off the request path (§5), and only possible
*because* the model never makes the decision — the tier boundary paying an operational dividend
rather than costing one. `ExplanationQueue`'s `ThreadPoolExecutor` is a small implementation of a
bigger idea: nothing upstream talks to an executor, only `.submit()` / `.status()` / `.get()`, so
a Celery task with a Redis-backed status store swaps in untouched. Two details a naive version
gets wrong are handled — context vars are copied at submit time, so log lines emitted after the
request returned still carry its id; and `submit` after `shutdown()` lands as `failed` rather than
raising `RuntimeError` into the request path of a graceful drain.

**Evaluation.** Component evals per PR; the system suite nightly against a growing synthetic
corpus, with the oracle re-derived independently of the code under test. Shadow mode scores a
candidate ranking change against live traffic without serving it.

**Safety.** The healing loop clusters incidents (`heal/cluster.py`) into repeated patterns and
turns them into permanent regression cases (`heal/proposals.py`), so a failure caught once cannot
silently recur; its proposal queue gets a review UI and an approval audit trail. The allergen axis
stays a **count**, at any scale — a percentage is the wrong unit for irreversible harm.

---

## 10. Module map

```
beatroot/
  contracts/       Pydantic models crossing every boundary — Severity, Constraint, ConstraintSet,
                   NutritionFacts, the terminal result types, TrustReport, CostRecord
  t0_invariants/   T0 — constraint registry and enforcement, feasibility + relaxation lattice,
                   vocabulary validation, nutrition arithmetic
  trusted/         catalog access, synonym canonicalisation, tag derivation, TagIndex bitmap
  retrieval/       T1 — FTS5/BM25, dense (NumPy dev / Qdrant prod behind VectorStore), RRF fusion,
                   query rewrite, two-pass select + LLM rerank
  reasoning/       T2 — the single LiteLLM wrapper, prompts as versioned content
  confirm/         T3 — composite trust scoring, conjunctive escalation gate
  agent/           LangGraph StateGraph wiring the tiers, typed state with reducers, SqliteSaver
                   checkpoints, async explanation queue, skills registry
  skills/          *.skill.md declarative definitions + skills-lock.json provenance
  eval/            component + system runners, synthetic generation with an independent oracle,
                   verifiers, calibration
  store/           SQLite persistence — schema/seed, audit, incidents, preference memory, caches
  heal/            incident clustering, proposal generation
  obs/             Langfuse tracing, structured JSON logging with redaction, cost ledger
  api/, cli/, web/ the three adapters — FastAPI, Typer, and a no-build static dashboard
  container.py     composition root: one connection, one write lock, one vector store, skill-lock
                   verification before anything expensive runs
  settings.py      the only module permitted to read the environment
```

Three adapters over one core. The agent is a library; the API is one client and the CLI another,
and both construct the same `Container`/`Deps` pair. `build_container` creates a single
`threading.Lock` and hands the same instance to every store that writes through the shared
`sqlite3.Connection` (`AuditLog`, `IncidentLog`, `FeasibilityCache`, `EmbeddingCache`,
`PreferenceMemory`), each holding it across its own `execute()` + `commit()` pair — required
because sync FastAPI routes run in threadpool workers and the connection is opened
`check_same_thread=False`.
