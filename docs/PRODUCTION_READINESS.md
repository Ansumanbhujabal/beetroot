# Production readiness

Two questions that get answered as one and should not be. "Is this a strong
take-home?" and "would I let this tell a peanut-allergic person what to eat?"
are different bars, and the honest answer differs. Part 3 is the split.

---

## Part 1 — Requirements coverage

Against the original requirements brief.

| Requirement | Verdict | Where |
|---|---|---|
| Backend API plus a frontend, CLI, or dashboard | Met | `api/main.py`, four served pages (`/`, `/incidents`, `/evals`, `/docs`), and a full CLI (`recommend`, `resume`, `serve`, `incidents`, `heal`, `eval`, `synth`, `prompts`, `obs`) |
| Small local dataset of ingredients, recipes, nutrition facts | Met | `data/` — 174 recipes, 137 ingredients, 21 preset profiles |
| Create a profile with several dietary/preference constraints | Met | 9 constraint kinds × 5 severities; presets loadable and then freely editable |
| Generate or select a recommendation from the profile | Met | `agent/graph.py`, hard filter → hybrid retrieval → LLM rerank |
| Show enough structured information to understand why it was chosen | Met | Ingredients with grams, computed nutrition, human-readable satisfied constraints, a three-part trust breakdown, and the full node trace |
| Persist useful state locally | Met | SQLite: audit log, incident log, feasibility and embedding caches, preference memory, LangGraph checkpoints |
| Keep it local — no cloud, no production-scale infra required | Met, and exceeded optionally | Default run is zero services. Docker, Qdrant and hosted tracing are opt-in paths, not requirements |
| One agent with ≥2 clearly defined skills | Met | 6 skills with `tier` / `llm_permitted` frontmatter, hash-locked, and checked for *reachability* rather than mere declaration |
| A lightweight tool-calling flow, state machine, or agent framework | Met | LangGraph `StateGraph` — a documented reversal of the spec's hand-rolled design (`CUT_LIST.md`) |
| Boundaries between model reasoning, deterministic validation, trusted data, and confirmation-gated actions | Met | An AST import-graph test forbids `t0_invariants/`, `trusted/` and `store/` from reaching `reasoning/`. The boundary is a failing build, not a convention |
| Grounded in trusted data; no fabricated numbers | Met | `NutritionFacts.provenance: Literal["computed"]` — the type system refuses to represent a model-authored nutrition value |
| One validation/eval mechanism, with a named failure mode | Met, and then some | Six adversarial axes plus a component suite with a computed oracle; the guarded failure mode is stated as "a hard constraint violated on a COMMIT", gated as a count of 0 |
| Two features of your choosing | Met | Feature 1: infeasibility negotiation with a ranked relaxation ladder that never offers a medical or religious constraint. Feature 2: trust-gated escalation with a conjunctive veto and a human approval gate on the medical grey band |
| Conflicting constraints, unsupported requests, wrong nutrition, low-confidence output | Met | NEGOTIATE, ESCALATE (`out_of_scope` / `unknown_ingredient`), the drift ledger, and PENDING_REVIEW respectively |
| Evolution at scale | Met | Spec §17 and `ARCHITECTURE.md` — T0 feasibility scan as first bottleneck, bitmask over inverted indices, Qdrant + payload filters, async explanation |

Free-text input reaches the constraint layer, which makes it an injection
surface rather than a decorative field. The safety rule is in code, not in the
prompt: the model proposes an open *category*, code alone maps category to
`Severity`, the merge is a pure append, and no path can remove, relax or
downgrade a constraint that arrived through the trusted structured channel.

---

## Part 2 — Where it stands against a production bar

| Area | Assessment |
|---|---|
| **Safety invariants** | Strongest part of the system. T0 is model-free by enforced import graph; the allergen gate is a violation *count*, not a rate. Residual risk is process, not code — see below |
| **Test catching power** | Coverage is high and the gate is 80%, but the load-bearing evidence is that the suite was mutation-tested and initially failed (`EVAL_RESULTS.md`). Six mutations is a start, not a proof |
| **Observability** | Structured JSON logs with correlation ids and recursive redaction; one generation span per model call carrying prompt name, version, tokens and cost; trace costs reconcile with `/metrics` exactly. Known and documented residual: a secret interpolated directly into a log message string, rather than passed as structured data, bypasses redaction |
| **Config and secrets** | Single source of truth, enforced by an AST-walk test. `.env` is gitignored and untracked. `.env.example` — a tracked file — previously contained real tracing keys and a variable name no SDK reads; it has been scrubbed to placeholders. Rotate anything that was ever committed, and prefer `git archive` over zipping a working directory, since gitignore does not constrain a filesystem-level zip |
| **Failure modes** | Missing credentials resolve to the offline provider as a first-class logged decision at settings load, not a runtime crash. Malformed model output is bounded; unevaluable constraints degrade to "uncheckable" and escalate rather than raising. Provider outage and rate limiting are delegated to LiteLLM and have never been tested against a real outage |
| **Concurrency** | One shared SQLite connection across a threadpool. Writes are serialized by a shared lock after a real interleaving corruption bug was reproduced and fixed. **Unfixed and documented:** a concurrent read can observe another thread's write between `execute()` and `commit()`. The store at risk is the audit log |
| **Persistence** | File-backed LangGraph checkpoints. Caught late: the agent originally defaulted to an *in-memory* checkpointer, so `resume` in a second process could never find its own paused thread — the approval gate would have passed every same-process test while being non-functional for its only real use case |
| **Deployment** | Verified running, not merely configured: image builds, container comes up healthy, compose brings up app plus Qdrant, `/health` reports the Qdrant path, and a live request returns a recommendation. Host port is overridable, because a hardcoded one fails outright when anything else holds it |
| **CI** | `.github/workflows/ci.yml` would run the gates and the offline safety evals on push. The repo has no remote configured, so it has never executed. Unexercised YAML, not a running gate |
| **Cost and latency** | Per-request cost is measured, attributed per stage, and reconciles between the ledger and the traces. There is no load test, at any concurrency, offline or live |
| **PII** | Absent as a posture. Profiles carry allergy and medical-condition data, stored in plain SQLite with no encryption at rest, no access control, no retention policy, and no anonymisation in the incident log |

### The residual risk that matters most

Three safety-relevant defects in this project shared one shape, and it is the
shape most likely to produce a fourth:

**Independence of implementation is not independence of assumption.**

1. The first component oracle computed ground truth by calling the function
   under test, then "verified" itself by calling it again.
2. The independent hard-constraint verifier was written deliberately *not* to
   call `is_legal()` — and independently reproduced the exact same
   unknown-vocabulary blind spot, because both implementations shared an
   unstated premise (a constraint value absent from the vocabulary is
   vacuously satisfiable), not any code.
3. The allergen synonym bug hit the same wall a third time. A `MEDICAL`
   exclusion on "groundnut oil" — the common name for peanut oil, and a
   synonym recorded against `ing_peanut_oil` in this project's own data — was
   **accepted** by validation, which canonicalises to confirm the term names a
   real ingredient, and then **never enforced**, because enforcement compared
   the raw string against canonical ingredient ids. The system told the user
   their allergy was understood and then served the allergen: worse than a
   missing check, because of the false assurance. The identical gap shipped in
   four separately-written places at once — the production enforcer, the
   independent verifier, the oracle, and the golden `synonym_evasion` family,
   which was named for the case it did not test.

Two implementations that share no code can still share a blind spot if they
share a mental model. Call-graph independence defends against a dispatch or
comparison bug local to one implementation; it does nothing for a premise
everyone who wrote these modules held at once. The only countermeasure that
worked was writing it independently **and then deliberately breaking one side
to watch the other catch it** — several regression tests in this project
initially passed against the exact pre-fix code they claimed to catch, and a
decorative test is worse than no test, because it reports safety that was never
checked. That discipline is applied by hand. It is not automated, and nothing
structural prevents a fourth instance.

---

## Part 3 — The verdict, split

**As a take-home submission: yes, and unusually so.** The system does the thing
most submissions do not — it caught and fixed several of its own real safety
defects, documented each rather than papering over it, and left evidence that
makes an audit like this one possible. The gates are green, the safety
invariants are enforced by tests rather than by convention, the eval suite has
been mutation-tested against itself, and every headline number in
`EVAL_RESULTS.md` carries the caveat that limits it. Deployment, the Qdrant
path, live-provider evals and end-to-end tracing are all verified running, not
described.

**To serve real users with real allergy consequences: no.** Not because the
safety logic is weak — it is the strongest part of the codebase — but because:

1. The gap between "allergen enforcement works" and "allergen enforcement works
   in production" has been crossed by a *conceptual* failure, identically, in
   independently-written code, three times in this project's own history.
   There is no reason to believe a fourth does not exist today. The only
   defense that has ever worked is one more test someone thought to write.
2. There is a documented, unfixed read-uncommitted-write race in the exact
   store that holds the audit trail the safety story depends on for liability.
3. There is no PII or medical-data posture at all — plaintext at rest, no
   access control, no retention policy — for a system whose profiles carry
   allergy and medical-condition data.
4. Dense retrieval is a hashing stub in every configuration, live included, so
   "hybrid retrieval" is architecturally true and semantically not.
5. The free-text compiler over-classifies severity. It errs safe, but a
   preference silently promoted to `medical` becomes non-relaxable, and the
   user is not told why.
6. Nothing has been load-tested at any concurrency, and CI has never run.

None of this is hidden — the project's own documentation is more candid about
most of it than a typical readiness review would be. But "honestly documented
as not done" and "done" are different states, and the second question asks for
the second one.

---

## Part 4 — Remediation, by risk reduction per unit effort

1. **Under an hour.** Rotate any credential that was ever committed to
   `.env.example`, and package with `git archive` rather than zipping the
   working directory — gitignore does not constrain a zip.
2. **Under an hour.** Normalise `exclude_tag` through the same canonicalisation
   `exclude_ingredient` uses, closing the one named asymmetry.
3. **Half a day.** Per-thread connections or a WAL reader to close the
   read-uncommitted-write race. Prioritised by *what* is at risk — the audit
   log — not by likelihood.
4. **Half a day.** Mutation-test the remaining safety properties, and automate
   the pass so the next conceptual gap is found by CI rather than by intuition.
   This is the single item that addresses the residual risk in Part 2 rather
   than one instance of it.
5. **Days, and mostly not code.** Provision an embedding deployment before
   "hybrid retrieval" is a true claim under live conditions; cap the free-text
   severity ratchet or surface the promotion to the user; add a load test.
6. **A precondition, not a task.** A PII and medical-data handling review —
   encryption at rest, access control, retention — before this touches a real
   user's allergy data.
