# Cut list

The brief asks directly what shortcuts and tradeoffs were made because this is
a 2–3 hour local prototype. This is that answer, stated as judgment rather than
omission: what was deliberately not built and what it would cost to add, which
design decisions were reversed once they met the code, and what remains
limited on purpose.

---

## Decisions taken against the design spec

The design spec (`docs/specs/2026-08-30-beatroot-design.md`) is the original
intent. Several of its calls did not survive contact with the build.

**Hand-rolled state machine → LangGraph.** The spec argued for ~80 lines of
typed hand-rolled transitions, on the grounds that a framework "would hide the
transitions inside its runtime, and the transitions are the thing being
demonstrated." That argument held until it was tested against what a
hand-rolled version would have to *fake* to look production-grade: durable
checkpointing across a process restart, and an `interrupt_before` approval gate
for the medical grey band. Simulating both is a worse signal in a submission
judged on production judgment than using the real thing. Shipped with
`StateGraph` plus a file-backed `SqliteSaver`; the state names in
`agent/graph.py` are exactly as visible and testable as hand-rolled ones would
have been, and `beatroot resume` finding a paused thread in a *fresh process*
is now a property the code has rather than one it claims. Reversing it is a
rewrite of `graph.py` alone — the nine nodes behind it are plain functions that
do not know they are wired by LangGraph.

**Hand-rolled provider clients → LiteLLM.** Retries, fallbacks, cost tracking
and Azure/Ollama/offline routing are solved problems. Hand-rolling them is the
amateur signal, not the sophisticated one. Cost: one dependency, in exchange
for provider selection, retry policy and cost accounting in less code than
provider selection alone would have taken.

**LiteLLM's Langfuse callback → direct SDK instrumentation.** Adopted, then
removed. The callback exported traces from short-lived processes and nothing at
all from the long-running server, while every diagnostic reported healthy; and
running it alongside direct instrumentation double-counts cost on every call.
Model calls are now instrumented directly (`obs.tracing.observe_generation`),
one generation span per stage, and a named constant plus a test pin the
decision that the callback stays unregistered.

**Hardcoded thresholds → `pydantic-settings` + `config/beatroot.yaml`.** No
`os.getenv` scattered through modules, enforced by an AST-walk test rather than
a convention. The AST version exists because a grep-based check was proven to
both false-positive on a docstring that merely *described* the rule and
false-negative on `from os import environ`.

Two decisions the spec made that were re-examined and kept:

- **LLM rerank over a cross-encoder.** At this catalog size, reranking an
  already-legal top-k with the model reuses the same provider seam as
  explanation generation instead of adding a second model dependency and a
  second failure mode.
- **NumPy in dev, Qdrant in production, behind one `VectorStore` protocol
  selected by `QDRANT_URL`.** A hard Qdrant dependency would break the property
  that a reviewer clones and runs with zero services; building only NumPy would
  leave the production path unproven. Both exist, and the Qdrant path is not
  theoretical — the compose stack runs it, `/health` reports
  `vector_store: qdrant`, and live requests are served through it. The tests
  that assert the two stores rank identically skip unless `QDRANT_URL` is set,
  so run them against a real server (`QDRANT_URL=... uv run pytest
  tests/retrieval/`) whenever the client version moves.

---

## Not built, and what it would cost

| Cut | Why | Cost to add |
|---|---|---|
| **Multi-day / multi-meal planning** | Single-meal recommendation already exercises every trust-tier boundary the brief asks about. A planning horizon adds combinatorics, not architectural insight. | A `PlanState` wrapping N single-meal runs plus a cross-meal nutrition-budget constraint kind. The T0–T3 boundary is untouched. 1–2 days. |
| **Auth, multi-tenancy, rate limiting** | Out of scope for a local prototype graded on architecture. | FastAPI middleware plus a tenant column on every table. ~1 day, no core-logic change. |
| **A hand-authored adversarial suite in the hundreds** | 33 reviewed golden cases carry the contract; generated families carry the statistical power. Hand-authoring hundreds of near-duplicates buys neither. | The generators scale to arbitrary `n` for free. The real cost is *golden* cases — reviewed and locked — at roughly 5 minutes each. |
| **Fine-tuning** | The catalog is the grounding mechanism. A fine-tuned model would still need every T0/T1 check already here, for no safety gain. | Needs a labelled preference dataset that does not exist. Weeks, and a data problem before a code problem. |
| **Real-time inventory, pricing, delivery logistics** | Product-integration surface, orthogonal to the AI-reliability question being graded. | An adapter per feed plus a staleness check on catalog data used in T0. The trust tiers absorb it — feed freshness becomes another `catalog_coverage` input. |
| **Cross-encoder reranking** | See above. | A `sentence-transformers` cross-encoder behind the same `top_k` cut. Half a day *including* an eval to confirm it beats LLM rerank here — which at this catalog size it plausibly does not. |
| **Everyday-vs-aspirational practicality ranking** | Genuinely interesting; needs labels for which meals people actually cook versus save. | A data-acquisition problem, not a scoping one. No code until the dataset exists. |
| **Triples in the relaxation lattice** | Feasibility explores dropping one constraint, then each pair (`O(n + n²)`). If no single or pairwise relaxation unlocks a meal, the answer is a profile review, not a deeper search. | `O(n³)` over a bounded soft-constraint count (rarely > 6) is still cheap here — an extension of the existing pairwise loop in `feasibility.py`, ~1 hour, inheriting the medical/religious lockout unchanged. |
| **PII posture: encryption at rest, access control, retention** | A local prototype whose data never leaves the machine. | Not a scoping decision so much as a precondition for real users — see `docs/PRODUCTION_READINESS.md`, where it is the gating item, not a nice-to-have. |

---

## Standing limitations

**The dense embedder is a token-hashing stub, in every configuration.** Not
only offline: the live deployment's Azure resource has a chat deployment and no
embedding deployment, so `embedding_model` resolves to `local`. Dense retrieval
is deterministic bag-of-words similarity, not semantic. This is an infra
provisioning gap, not a code one — the `VectorStore` seam and the embedding
cache are indifferent to which embedder fills them.

**The free-text compiler can over-classify.** A stated preference ("no dairy
this week please") has been observed compiling to `medical` category. The
severity ratchet only ever *floors* severity and never caps it, so the error
lands in the safe direction — but it makes a preference non-relaxable, which is
a real cost to the user. Worth stating rather than hiding: a stricter system
than asked for is still a system doing the wrong thing.

**`exclude_tag` and `exclude_ingredient` normalise differently.**
`_exclude_ingredient` canonicalises its value through the catalog's synonym
index before comparing; `_exclude_tag` compares the raw string against
catalog-canonical tags. Both fail *safe* — a case-inconsistent
`exclude_tag: "Peanut"` escalates rather than silently reading as satisfied —
so this is a precision gap, not a safety hole: an upstream caller with
inconsistent casing gets an unnecessary refusal where the ingredient path would
have resolved it. Under an hour to close, plus a regression test.

**The boundary and settings enforcement tests are blind to dynamic imports.**
Both walk the AST; `importlib.import_module("beatroot.reasoning")` is invisible
to static analysis by construction. The tests catch the mistake anyone actually
makes and would not catch a deliberate evasion.

**A6 cannot be measured on the async explanation path**, which is the default
for request latency. The runner refuses to score rather than report a
meaningless 1.000; the async path's own grounding guarantee is covered by a
dedicated test instead. See `EVAL_RESULTS.md`.

**Calibration is effectively unmeasured**, for reasons that are a property of
the sampling and not of the metric. See `EVAL_RESULTS.md` rather than a second
account here.

**The catalog is closed-world by design.** Anything outside its tag, cuisine or
ingredient vocabulary — sulfite, MSG, a brand name — is unverifiable, and the
system refuses rather than guesses. That is the intended behaviour, and it also
means the system has no opinion at all outside 174 recipes and 137 ingredients.
