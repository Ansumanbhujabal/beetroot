# Production Readiness Audit — beatroot

**Branch:** `build` · **HEAD:** `3e91ad1` · **Audited:** 2026-08-31
**Method:** static reading of every module cited below, `git log`/`git show` on the
real diffs (not commit messages alone), a live `pytest`/`ruff`/`mypy`/`coverage`
run against this exact HEAD (offline, no LLM calls, no server restart — the
in-progress Azure eval was left untouched), and cross-checking every claim in
README.md / ARCHITECTURE.md / EVAL_RESULTS.md / CUT_LIST.md / `.sdd/progress.md`
against the code that supposedly backs it.

**Verified gate numbers on this exact HEAD, this run** (differ slightly from the
numbers printed in README.md/EVAL_RESULTS.md, which were measured on earlier
commits — see §6):

```
pytest:    577 tests, 0 failed, 0 errors, 5 skipped   (42.4s, offline)
ruff:      all checks passed
mypy --strict src:  no issues, 73 source files
coverage:  94% (3463 stmts, 208 missed; gate is 80%)
```

The 5 skipped are all Qdrant-dependent (`tests/retrieval/test_qdrant_store.py`,
`tests/retrieval/test_qdrant_retry.py`, one in `tests/api/test_routes.py`) —
confirmed by grep, not assumed.

---

## Part 1 — Requirements coverage

The take-home brief itself is not in the repo as a standalone file; the closest
artifact is `docs/specs/2026-08-30-beatroot-design.md`, which quotes it directly
("We care about the boundaries between model reasoning, deterministic
validation, trusted data, and actions that should require confirmation") and is
treated throughout `.sdd/` as the transcription of record. Requirements below
are drawn from that spec plus the brief's four closing questions as restated in
README.md ("where are the boundaries, what would you build, what would you cut,
how would this scale").

| # | Requirement | Verdict | Evidence (file : function) |
|---|---|---|---|
| 1 | Take a user profile with dietary constraints and recommend a meal | **Met** | `src/beatroot/agent/graph.py` (StateGraph), `cli/main.py:recommend`, `api/main.py:POST /recommend` |
| 2 | Validate the recommendation deterministically | **Met** | `t0_invariants/constraints.py:check_recipe`/`is_legal`, re-verified post-generation by `agent/nodes.py:verify_node` |
| 3 | Explain the recommendation, grounded in trusted data | **Partially met** | `reasoning/` + `agent/nodes.py:explain_node` generates prose; grounding is enforced by `eval/verifiers/nutrition_drift.py:detect_drift`, but A6 (explanation_grounding) is **vacuous under the offline stub** used for every eval run in this repo (§3) — the guarantee is proven only for the drift *detector*, not for real model prose, and is **was** un-enforced on the async-explanation path, which became the DEFAULT in `e75c9a7` when async explanation was switched on for latency — silently disabling this requirement on the only surface a diner reads. Fixed in `d7910d4`: `ExplanationQueue._work` now runs `detect_drift` against the same nutrition facts and threshold `verify_node` uses, and lands a drift finding as `failed` rather than serving ungrounded prose. Note the failure mode of the original disclosure: it was accurate, but its safety rested on the parenthetical "(off by default)" — a caveat conditional on a config value will outlive the config |
| 4 | Identify and enforce the boundary between model reasoning, deterministic validation, trusted data, and confirmation-gated actions | **Met** | `tests/test_boundaries.py` — AST import-graph BFS forbidding `t0_invariants/`, `trusted/`, `store/` → `reasoning/`; `tests/agent/test_skills_registry.py` + skill frontmatter (`tier`, `llm_permitted`) checked for reachability, not just declared |
| 5 | State what you would build with more time; what you would cut | **Met** | `CUT_LIST.md` — reversals (LangGraph, LiteLLM, pydantic-settings), non-goals with named cost-to-add, and an honestly-labelled "standing limitations" section (not just a wishlist) |
| 6 | Explain how the system evolves at scale | **Met** | ARCHITECTURE.md §5 / spec §17 — bottleneck analysis (T0 feasibility scan → bitmask index, `trusted/index.py:TagIndex`), Qdrant path (§2 below), async explanation (`agent/async_explain.py`) |
| 7 | Lightweight tool-calling flow / state machine / agent framework | **Met, but changed mid-build** | Design spec called for a hand-rolled ~80-line machine; the shipped system uses LangGraph `StateGraph` + `SqliteSaver` instead (`agent/graph.py`) — a deliberate, disclosed reversal (`CUT_LIST.md`), not a silent substitution |
| 8 | Grounded in a trusted nutrition/ingredient dataset; no fabricated numbers | **Met** | `contracts/nutrition.py:NutritionFacts.provenance: Literal["computed"]` — the type system, not a convention, forbids a model-authored value; `t0_invariants/nutrition_math.py:compute` |
| 9 | Free-text preferences as an untrusted input / injection surface | **Partially met, closed late** | `.sdd/progress.md` records that `compile_constraints` was **entirely unwired** for most of the build (a graded skill/prompt describing a capability that didn't exist); wired in commit `fbf038b` (`agent/nodes.py:compile_node`), **superseded**: parsed text can now produce HARD constraints too. The model states an open `category` (medical/religious/dietary/goal/preference) and CODE alone maps it to a `Severity` (`agent/nodes.py:_CATEGORY_SEVERITY`), with a ratchet flooring the identity kinds `require_tag`/`require_any_tag` at DIETARY regardless of what the model claimed. That is stricter, not looser: the old PREFERENCE-only rule meant a stated allergy compiled to something `is_legal()` does not enforce — the same class of bug that let the vegan preset serve chicken (commit `d773717`). Verify this specific commit is genuinely on `build` before treating the requirement as closed — it is present in `git log` on this branch. |
| 10 | Escalate/refuse when the system lacks trustworthy information | **Met, after one real safety gap was found and fixed** | `confirm/escalation.py`, `confirm/trust_score.py:gate` (conjunctive veto). `.sdd/progress.md` (Task 13) documents A5 originally scoring **0.000** — an unknown tag/ingredient was silently treated as satisfied rather than escalated — fixed across FOUR independent implementations that each had to be separately patched (production enforcer, oracle, verifier, docs) because they shared the same unstated assumption. This pattern (independent code, shared blind spot) recurred three times in this build — see §3. |
| 11 | Component + system eval harness, some LLM-free | **Met** | `eval/runners/components.py`, `eval/runners/system.py`, `beatroot eval --offline` equivalent (`BEATROOT_OFFLINE=1`); "free oracle" (`eval/synth/profiles.py`) — but see §3 for its own tautology-and-fix history |
| 12 | Deployable (Docker / HF Space) | **Met — verified 2026-09-01** | `Dockerfile`, `docker-compose.yml`, HF frontmatter in `README.md` all present and internally consistent (`docker compose config` validated) — and all three now **have** executed. `docker build` succeeded on its first real run (14/14 steps, `beatroot:local`, 1.89GB); `docker run` came up healthy in 11s serving all pages and a real recommendation; `docker compose up` brought up app+Qdrant with `/health` reporting `vector_store: "qdrant"` — the production path, end to end. The blocker was never the config: two orphaned `limactl usernet` processes held the colima disk lock (`failed to attach disk "colima", in use by instance "colima"`), which is why two earlier attempts failed. |
| 13 | Observability (tracing, cost) | **Met** | `obs/logging.py` (structured JSON, correlation ids, recursive redaction — see §2 for its own fix history), Langfuse via `litellm.success_callback` (no-op without keys, `obs/__init__.py`), `CostLedger` wired into `/metrics` (was built-but-dead for a full review round — see progress.md Task 19) |

**Rollup:** 10 met, 3 partially met, 0 flatly unmet — but three of the "met" verdicts (rows 4, 8, 10) only reached that state after a documented near-miss where the *first* shipped version silently failed the exact property being graded. That history is itself evidence about production readiness, not just development trivia — see Part 3.

---

## Part 2 — Production bar

### Correctness and safety invariants — **Strong, with one class of residual risk**
T0 (`t0_invariants/`) is genuinely model-free — enforced by an AST reachability
test, not a docstring. The allergen-safety metric is a **count**, not a rate
(`eval/thresholds.yaml: hard_constraint_violations: 0`), which is the right
unit for irreversible harm. But the project's own history shows the invariant
layer has been *conceptually* wrong three separate times (unknown-vocabulary
silent-pass, oracle tautology, ingredient-synonym enforcement gap) while every
test stayed green, because independently-written implementations shared an
unstated assumption rather than a bug. There is no structural guard against a
*fourth* instance of this pattern — the only defense that has worked so far is
someone writing an additional test they "assumed would be redundant."

### Test coverage, and whether tests would catch a regression — **Coverage is high; catching power was empirically checked and found wanting until fixed**
94% line coverage, 577 tests, 0 failing on this exact HEAD (verified live, not
taken from a doc). The more important number: `git show 1ac5327` is a real
mutation-testing pass — six safety properties were each disabled one at a time
and the suite was watched. **Before that commit, only 2 of 6 mutations were
caught**: disabling the entire trust gate (`confirm/trust_score.py:gate`) was
invisible because every golden case had full catalog coverage; disabling drift
detection was invisible because the offline stub's prose never states a
number. Both are now fixed with targeted cases/probes
(`eval/golden/seed_cases.yaml: g31-g33`, `eval/runners/components.py:_drift_detection_recall`).
This is the single strongest piece of evidence in the repo that testing here
is honest rather than decorative — but it also means, by the project's own
admission, that a majority of its safety mutations were undetected as recently
as one commit before HEAD, and there is no reason to believe all failure modes
have now been mutation-tested (only 6 were tried).

Separately, `.sdd/progress.md` documents at least four "decorative tests" found
during the build — regression tests that passed against the exact broken code
they claimed to catch. All were found and fixed, but only because a specific
review discipline (run the pre-fix code, observe the failure, don't just
reason about it) was applied by hand each time; it is not automated.

### Observability — **Real, with an honest gap on nested config leaks now closed**
Structured JSON logging with correlation-id contextvars (`obs/logging.py`).
Langfuse tracing via `litellm.success_callback`/`failure_callback` — genuinely
no-op without both keys present (`ObsConfig.langfuse_enabled`), so a keyless
reviewer never hits a credential wall. Redaction history is worth knowing: the
first shipped version matched only 5 literal key names and missed nested
dicts entirely (`Api-Key`, `x-api-key`, `openai_api_key` all leaked in
plaintext in the pre-fix version per `.sdd/progress.md` Task 19 review) — now
fixed to normalized substring matching + recursive dict/list walk, depth-capped
at 6 (`obs/logging.py:_redact`, `_MAX_REDACT_DEPTH`). **Documented, not fixed,
residual: a secret interpolated directly into a log message string (not passed
via `extra=`) bypasses redaction unconditionally** — this is stated in code
comments as a known limitation, not silently true.

### Configuration and secrets — **Sound design, one real operational hazard**
Single source of truth (`settings.py`), enforced by an AST-walk test (not a
grep — the grep version was proven to both false-positive on a docstring and
false-negative on `from os import environ`, both fixed). `.env` is
git-ignored and confirmed absent from git tracking (`git ls-files` scope; the
repo *is* a real git repo despite the environment banner claiming otherwise —
verified with `git rev-parse --is-inside-work-tree` → `true`). **This working
directory's `.env` currently holds live Azure API credentials** (verified by
key presence, values not printed here) alongside an actively-running eval
process. Gitignore protects against `git add`/`git commit`, but **nothing in
this repo protects against a naive `zip -r submission.zip .`** picking up
`.env` regardless of gitignore — there is no packaging script, Makefile
target, or `.gitattributes`-driven `git archive` step that a submitter is
forced through. This is a real, live risk right now, not a hypothetical one.

### Failure modes — **Handled for the cases tested, one now-obsolete crash fixed**
- **No API key / no embedding deployment:** `settings.py:_provider_credentials_present`
  + a `local` embedding provider (added in `d5b2db0`) let a real chat model run
  without requiring an embedding deployment on the same Azure resource —
  directly relevant to the live eval this session was told not to disturb.
  Falls back to `offline=True` at settings-load time, logged at WARNING, not a
  runtime crash. This *was* broken earlier in the build (`lifespan` eagerly
  embedded the whole catalog before `/health` was reachable, crashing on a
  genuinely blank environment) — CUT_LIST.md documents the fix and, notably,
  that the *first* "it works keyless" verification was itself contaminated by
  leftover local cache state and gave a false negative.
- **Malformed model output:** `reasoning/llm.py:Completion` bounds/guards
  self-assessment; `t0_invariants` evaluators degrade to `"uncheckable"`
  rather than crash on non-numeric/bool/malformed constraint values
  (Task 24 fix, per progress.md).
- **Provider outage/rate limiting:** delegated entirely to LiteLLM's
  `num_retries`/`fallbacks` — never independently tested against a real
  outage; the `constraint_flooding` adversarial threshold is deliberately
  floored at **0.99, not 1.0**, explicitly to leave headroom for "a transient
  infra failure that is not a defect in this codebase" (`eval/thresholds.yaml`
  comment) — an honest acknowledgment that this failure mode is *reasoned
  about*, not *observed*, since every measured run has been offline.

### Concurrency and data integrity — **A real bug found and partially fixed; one gap left open by design, not oversight**
`store/db.py` uses one shared `sqlite3.Connection` with `check_same_thread=False`
across a threadpool (FastAPI sync routes). A real corruption bug was found and
fixed: two threads' `execute()`+`commit()` pairs could interleave and merge an
unrelated write into another thread's still-open transaction — reproduced
standalone, then fixed with one shared `threading.Lock` held across each
write's execute+commit (`container.py`'s `db_lock`, shared by `AuditLog`,
`IncidentLog`, `FeasibilityCache`, `EmbeddingCache`). **What remains explicitly
unfixed and documented in code**: a concurrent *read* on the shared connection
can still observe another thread's write after `execute()` but before the
lock-holder's `commit()` releases — Python's sqlite3 module makes an
uncommitted write visible to other cursors on the same connection immediately.
Fixing this needs per-thread connections or a WAL-backed reader connection,
neither built. This is the "Read-can-observe-uncommitted-write concurrency
residual" the task named — confirmed still present in current code, not
historical.

### Persistence, checkpointing, resumability — **Real, after a bug that would have made the headline feature nonfunctional**
LangGraph `SqliteSaver`, file-backed (`<db>.checkpoints.db`, confirmed present
as untracked files in this working tree right now — consistent with the live
eval in progress). A real bug was caught late: the agent originally defaulted
to an **in-memory** checkpointer, so `beatroot resume` run as a separate CLI
invocation after `beatroot recommend` could never find its own paused thread —
the `interrupt_before` medical-grey-band approval gate, a headline feature,
would have passed every test (which reuse one process) and been completely
non-functional for the actual multi-process use case. Fixed
(`container.py:_build_checkpointer`); durability now proven by a test that
forces a real process-boundary-equivalent (a distinct `:memory:` connection
fails the same test, confirmed as a live differential check).

### Deployment — **Config is internally consistent; the actual boot has never been observed**
Dockerfile and docker-compose.yml are coherent with each other (Qdrant extras
installed, `env_file: {required: false}` so a fresh clone with no `.env`
doesn't refuse to start, a `service_healthy` dependency to avoid a startup
race against Qdrant). None of `docker build`, `docker run`, or
`docker compose up` has ever been executed anywhere in this project's
development — reconfirmed in this audit (no Docker daemon available here
either). The keyless-boot claim was **verified false once** mid-build (a
contaminated local-state false positive, see above) before being fixed and
re-verified at the application layer only (`uvicorn` boot, not a container
boot). CI (`.github/workflows/ci.yml`) exists and would run these gates plus
the offline safety evals on push — but this repo has **no git remote
configured** (`git remote -v` empty), so that workflow has never actually
executed on GitHub Actions or anywhere else; it is unexercised YAML, not a
running gate.

### Performance and cost — **Reasonable for the stated scale, entirely unmeasured under a live provider**
p50/p95 latency (9ms/14ms offline) and $0.00 cost are real numbers for the
offline path but say nothing about a live model's latency or LiteLLM retry
behavior under load. `CostLedger`/`/metrics` are wired (after a full review
round where they were built-but-dead) and would report real numbers the
moment a live provider runs — but no live-provider load test exists in this
repo, offline or otherwise.

### Security: prompt injection, credential leakage, PII — **Injection resistance is structurally real but narrower than advertised; leakage is fixed; PII handling is essentially absent**
- **Injection:** free text can only ever produce `PREFERENCE`-severity
  constraints (`agent/nodes.py:compile_node`, enforced by
  `tests/agent/test_compile.py`), and the safety property for the four
  adversarial families that do reach an LLM on a COMMIT path is independently
  re-verified by `eval/verifiers/hard_constraint.py`, not trusted to the
  model's behavior. This is real. It is *narrower* than the README's framing
  suggests in one respect: for most of the build, `compile_constraints` was
  entirely unwired, so "injection resistance" was trivially true because free
  text never reached the constraint layer at all — a materially weaker claim
  than the one the shipped skill file made. This was found and fixed
  (`fbf038b`), but it is exactly the kind of gap this audit is designed to
  surface: verify it holds on `build`, don't take the skill doc's word.
- **Credential leakage:** fixed after a real defect (plaintext secrets in
  logs, see Observability above); the interpolated-string bypass remains, by
  design/documentation not oversight.
- **PII:** there is no PII-specific handling anywhere in this codebase —
  profiles carry dietary/medical data (allergies, medical conditions) which
  is sensitive but is stored in plain SQLite with no encryption at rest, no
  access control, no data-retention policy, and no anonymization in the
  incident/audit log (`store/db.py`, `store/incidents.py`). This is
  appropriate for a local take-home prototype and would not be appropriate
  for a system serving real users' medical/allergy data. Not addressed
  anywhere in CUT_LIST.md as a named limitation — an omission worth flagging
  on its own.

---

## Part 3 — The honest gap list

| Gap | Why it exists | Blast radius | Cost to close |
|---|---|---|---|
| **Qdrant path — RESOLVED 2026-09-01** | Was: no Docker daemon in any build environment. Root cause of the daemon failure was two orphaned `limactl usernet` processes holding the colima disk lock, not the config. | **All 7 Qdrant tests now pass against a real server** (`QDRANT_URL=http://localhost:6533`, qdrant/qdrant:v1.12.0), and they demonstrably exercised it rather than passing vacuously: the run left 5 collections (`beatroot_test`, `beatroot_test_filter`, `beatroot_test_no_wipe`, `beatroot_test_agree`, `beatroot_recipes`) and 57 server operations. Separately, the compose stack served a live recommendation with `/health` reporting `vector_store: "qdrant"`. The claim no longer rests on static typing. | Closed. Re-run with `QDRANT_URL=... uv run pytest tests/retrieval/` whenever the client version moves. |
| **`docker build`/`run`/`compose up` — RESOLVED 2026-09-01** | Same root cause, same fix. | Build succeeded first try (14/14 steps, 1.89GB). `docker run` healthy in 11s; all pages 200; a recommendation returned with ingredients and readable constraints. `docker compose up` brought up app+Qdrant, warm request latency 5.0-5.9s against real Azure (vs 4.1s locally on the NumPy store). Verified on non-default host ports so the developer's OTHER project's containers, already bound to 6333, were never touched. | Closed. |
| **A6 (explanation_grounding) is vacuous offline** | Offline stub prose never states a nutrition number, so the drift ledger has nothing to catch on that axis in the system eval | The system-eval A6=1.000 is honest about what it measures (confirmed: the eval script and README both disclose this) but proves nothing about a live model confidently stating a wrong number in prose. The unit-level `tests/eval/test_drift.py` and the new `_drift_detection_recall` probe do test the detector directly — but never through a real generated explanation. | Run the golden `drift_bait` cases against a live provider once (exactly what the in-progress Azure eval may be doing right now) and confirm the system-level A6 stays 1.0 there too, not just offline. |
| **`exclude_tag` vs `exclude_ingredient` case/whitespace asymmetry** | `_exclude_ingredient` canonicalises through `resolve_ingredient_id`; `_exclude_tag` compares the raw string against catalog-canonical tags with no normalisation, because tags are asserted to have "no synonym layer" | Confirmed still present in `t0_invariants/constraints.py`. Both sides are *safe* (a mismatch never reads as satisfied — a tag-case mismatch escalates rather than silently passing), so this is a UX/precision gap, not a safety hole: a case-inconsistent `exclude_tag` from an upstream caller degrades to an unnecessary refusal rather than being resolved. | `_exclude_tag` could normalise through the same `trusted.canonical` machinery; named as a known follow-up in EVAL_RESULTS.md already, not fixed. Small — under an hour, plus a regression test. |
| **`constraint_flooding` threshold floored at 0.99, not 1.0** | Deliberate headroom for network I/O flakiness under a live provider; the measured offline rate is 1.000 | Means a single flaky call in a future live run would not fail CI even though the code is unchanged — correct engineering judgment, but it is a threshold set by *reasoning about* a live failure mode that has never actually been observed, since every recorded run is offline. | No action needed; re-validate the 0.99 choice once real live-provider run data exists. |
| **Read-can-observe-uncommitted-write concurrency residual** | Single shared sqlite3 connection; writes are now lock-serialized but reads are deliberately left unlocked for hot-path cost reasons | A concurrent read (e.g. `/metrics`, a second `/recommend`) can observe a write mid-flight, before its transaction commits. Documented in code and in `.sdd/progress.md`, not hidden — but still live in current code, confirmed by reading `store/db.py`. | Per-thread connections (`threading.local()`) or a WAL-backed reader connection. Non-trivial — a half-day-plus change touching every store class. |
| **Dense retrieval is a token-hashing stub; no embedding deployment on the live Azure resource** | The live Azure resource configured in `.env` has a chat deployment but no embedding deployment; `settings.py`'s `local` provider was added specifically so a real chat model can run without one | Even against the currently-running live eval, the "hybrid retrieval" dense signal is **not semantic** — it is a deterministic hashing-trick bag-of-words vector, same as offline. Recall@5 (0.665-0.687 depending on which commit) is honestly disclosed as measuring the stub, not real embeddings. This materially weakens the "hybrid retrieval" claim under the exact live conditions this session is operating under. | Provision an embedding deployment on the Azure resource (or point `embedding_model` at Ollama's `qwen3-embedding`), flip `local` off. Infra change, not a code change — hours, not minutes, and outside this session's control per the hard constraints. |
| **`.env` holds a live key; nothing prevents it shipping in a naive zip** | `.gitignore` covers `git`-mediated packaging, not filesystem-level zipping | If the submission process is "zip the working directory" rather than `git archive`, a live Azure key ships. Confirmed live and present in this exact working tree right now. | Delete/blank `.env` (or `cp .env.example .env`) immediately before packaging, or add a `make submission-zip` target that uses `git archive HEAD` (which correctly excludes gitignored files). Under an hour. |

---

## Part 4 — The verdict

**Production-ready as a take-home demonstration: yes, and unusually so.** The
codebase does something most take-homes don't: it caught and fixed several of
its own real safety defects mid-build (the A5 unknown-vocabulary silent-pass,
the allergen-synonym enforcement gap, the trust-gate mutation blind spot, the
oracle tautology, the credential-leak logging bug), documented each honestly
rather than papering over it, and left a legible trail (`.sdd/progress.md`,
`CUT_LIST.md`, EVAL_RESULTS.md's "what these numbers do NOT prove" section) that
makes the audit above possible at all. The gates are genuinely green on this
exact HEAD (verified live in this session, not copied from a doc), and the
mutation-testing commit is real evidence the test suite has teeth, not just
coverage percentage.

**Production-ready to serve real users with real dietary-safety consequences: no.**
Specifically, not because the safety *logic* is weak — it is the strongest part
of this codebase — but because:

1. The thing standing between "allergen enforcement works" and "allergen
   enforcement works in production" has failed *conceptually*, identically,
   in independently-written code, three separate times in this project's own
   history. There is no reason to believe a fourth instance doesn't exist
   today; the only defense that has worked is one more test someone thought
   to write. That is a process risk, not a code risk, and it does not go away
   at HEAD.
2. The concurrency model has a documented, live, unfixed data-race (reads
   observing uncommitted writes) in the exact store that holds the audit
   trail this system's whole safety story depends on for liability purposes.
3. The one "production" retrieval path (Qdrant) has never executed, anywhere,
   and a real bug in it was found only by a type checker, not by running it —
   because nothing has ever run it.
4. The container deployment path — the actual thing "production" means for a
   service — has never booted, anywhere, in this project's history.
5. There is no PII/medical-data handling posture at all: plaintext storage, no
   access control, no retention policy, for a system whose profiles carry
   allergy and medical-condition data.
6. A live credential sits in this exact working directory with no packaging
   guardrail against it shipping.

None of these are subtle or hidden — the codebase's own documentation is more
honest about most of them than a typical production readiness review would be.
But "honestly documented as not done" and "done" are different states, and the
question asked was the latter.

---

## Part 5 — Prioritised remediation (risk reduction per unit effort)

1. **< 1 hour — Remove the live key from the packaging surface.** Blank
   `.env` or add a `git archive`-based zip target before any submission.
   Highest risk-reduction-per-minute item in this entire audit: a leaked live
   Azure key is an immediate, concrete, external-facing incident, unlike
   every other gap here which is an internal-quality risk.
2. **< 1 hour — Normalise `_exclude_tag` the same way `_exclude_ingredient`
   is normalised.** Small, well-scoped, already diagnosed in EVAL_RESULTS.md;
   closes the one specifically-named asymmetry cleanly.
3. **1-2 hours — Verify `docker build && docker compose up` on any machine
   with a Docker daemon.** This project has already found one showstopper bug
   (`QdrantVectorStore.search()` calling a removed API method) purely from
   `mypy`, without ever running the code it was checking. There is no reason
   to believe the container path is clean until someone actually boots it —
   this is the single highest-value unblocked verification step left, and it
   was blocked only by environment, not difficulty.
4. **Half a day — Actually run the Qdrant tests against a live Qdrant
   container.** Same shape as #3: a concrete, bounded, already-written test
   suite that has simply never executed. High confidence payoff for
   moderate effort.
5. **Half a day+ — Fix the read-uncommitted-write race** with per-thread
   connections or WAL. Matters specifically because the thing at risk is the
   audit log this system's liability story depends on.
6. **Not fixable within this engagement, but worth stating precisely for
   whoever owns the next phase:** a PII/medical-data handling review (at-rest
   encryption, access control, retention policy for allergy/medical-condition
   data) before this ever touches a real user, and provisioning a real
   embedding deployment before "hybrid retrieval" is a true claim under live
   conditions.
