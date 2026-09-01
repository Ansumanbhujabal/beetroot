# Walkthrough

You do not need to read the code to drive this. Give it a dietary profile
(allergies, religion, budget, time) plus optional free text, and it recommends
one meal. It always lands in one of four states:

- **COMMIT** — a meal, the nutrition it computed, and why.
- **NEGOTIATE** — your constraints are impossible together; here is what to relax.
- **ESCALATE** — it cannot safely answer, because the request is out of scope or
  a constraint cannot be checked against the catalog at all.
- **PENDING_REVIEW** — trust landed in a grey zone on a medical constraint; a
  human must approve before it commits.

Three of the four are refusals of some kind. That is the design. Most of what
is interesting here is *when and why it declines*.

**About the output below.** Every transcript in sections 3–6 was captured by
running the command shown, against the running build, while writing this file.
Live-provider runs used Azure `gpt-4o` with the Qdrant store; eval and CLI runs
used `BEATROOT_OFFLINE=1`. JSON is trimmed with `jq` for readability, never
edited. Where a step was not executed here, it says so and describes the
expected outcome instead of showing a transcript.

---

## 1. Running it

```bash
uv run beatroot serve --port 7860      # local, zero services
```

or the production-shaped path — app plus a Qdrant sidecar:

```bash
docker compose up -d --build
```

The host port is overridable, because a hardcoded one fails outright when
anything else already holds it:

```bash
BEATROOT_PORT=7861 docker compose up -d --build
```

The transcripts below were captured against a compose stack on `7861` for
exactly that reason. Substitute your own port throughout:

```bash
BASE=http://localhost:7861
```

`GET /health` tells you what you are actually talking to:

```json
{"status":"ok","provider":"azure","llm_model":"azure/gpt-4o",
 "vector_store":"qdrant","recipes":174,"skills":6,"skills_locked":true,
 "trust":{"refusal_threshold":0.55,
          "weights":{"catalog_coverage":0.45,"constraint_completeness":0.3,
                     "model_self_assessment":0.25}},
 "tracing":{"langfuse_configured":true,"host":"https://us.cloud.langfuse.com",
            "instrumentation":"langfuse-sdk (direct generation spans)"}}
```

`provider: echo` instead of `azure` means no credentials were found and the
deterministic offline stub is running. That is the expected state on a fresh
clone with no `.env` — booting keyless is deliberate, not a fallback that
happens to work. `vector_store: numpy` is the zero-dependency dev index; it
becomes `qdrant` the moment `QDRANT_URL` is set.

## 2. The pages

| Page | What it is for |
|---|---|
| `/` | Preset dropdown, constraint builder, free-text box, and the result: terminal state, ingredients, nutrition, trust breakdown, relaxation ladder, trace |
| `/incidents` | Every escalation, refusal, drift finding and infeasibility the process has logged, plus the drift ledger filtered from the same feed |
| `/evals` | The safety suite's own numbers, thresholds, run history, and a live tail of the process log |
| `/docs` | The architecture diagram and this document set |

`GET /profiles` backs the dropdown — 21 named profiles, still fully editable
once loaded. Picking one hides nothing: every constraint it adds appears as an
ordinary chip.

## 3. It commits, and holds the line

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"vegan",
  "constraints":[{"id":"req_vegan","kind":"require_tag","severity":"dietary",
                  "value":"vegan","source":"structured"}],
  "query":"dinner"}' | jq .
```

```
COMMIT — Jain roasted vegetable bowl
trace: FEASIBILITY -> RETRIEVE -> SCORE -> TRUST -> EXPLAIN -> VERIFY -> COMMIT
trust: composite 0.9875 (coverage 1.0, completeness 1.0, self-assessment 0.95)
nutrition: 157 kcal, provenance "computed"
constraints_satisfied_display: ["is vegan"]
ingredients: Zucchini 90g, Capsicum 90g, Tomato 70g, Olive oil 12g, ...
```

Now ask the same vegan profile for chicken, in both the query and the free
text:

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"vegan",
  "constraints":[{"id":"req_vegan","kind":"require_tag","severity":"dietary",
                  "value":"vegan","source":"structured"}],
  "query":"chicken","preferences":"I really want chicken tonight"}'
```

```
COMMIT — Jain vegetable fried rice
trace: COMPILE -> FEASIBILITY -> RETRIEVE -> SCORE -> TRUST -> EXPLAIN -> VERIFY -> COMMIT
effective_constraints: [require_tag vegan @dietary, source structured]
```

The free text asking for chicken did not add, weaken or contest the vegan
constraint — it simply lost. Note `COMPILE` appearing in the trace only when
free text is present: an empty preferences field costs zero tokens.

## 4. It reads meaning, not keywords

No structured constraints at all; everything comes from free text.

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"adhoc","constraints":[],
  "query":"date night dinner","preferences":"date with a pescetarian"}'
```

```
COMMIT — Jain lauki kofta
effective_constraints:
  {"kind":"require_any_tag","severity":"dietary",
   "value":["vegetarian","fish"],"source":"parsed_free_text"}
```

"Pescetarian" is not a tag anywhere in the data. There is no phrase→value
lookup table; the compiler reads the catalog's real tags and cuisines at call
time and the model derives `vegetarian OR fish` from meaning. The severity is
not the model's to choose — code maps the category it proposes to `DIETARY`.

## 5. It refuses, three different ways

**Out of scope.** No meal-planning content to act on:

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"adhoc","constraints":[],"query":"",
  "preferences":"what date is today"}'
```

```
ESCALATE — reason "out_of_scope", failing_signal "scope"
trace: COMPILE -> ESCALATE
detail: "this request has no meal-planning content to act on — refusing
         rather than guessing at a recommendation."
cost: $0.0019625 (compile only — no second call, no keyword blocklist)
```

The refusal falls out of the same compile call that parses constraints. It is
not a filter on the word "date": *"something quick for date night"* has real
content and commits normally, parsing `max_prep_minutes: 30` from "quick".
(That variant was not re-run for this document.)

**Impossible together.** A genuine peanut allergy plus a one-minute prep
ceiling:

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"deliberately_impossible",
  "constraints":[
    {"id":"excl_peanut","kind":"exclude_tag","severity":"medical",
     "value":"peanut","source":"structured"},
    {"id":"prep_ceiling","kind":"max_prep_minutes","severity":"preference",
     "value":1.0,"source":"structured"}],
  "query":"dinner"}'
```

```
NEGOTIATE — 0 of 174 candidates survived
trace: FEASIBILITY -> NEGOTIATE
relaxations: "allow more than 1 minute of prep" -> unlocks 166 (preference)
locked: ["excl_peanut"]
cost: $0.0000, tokens_saved 253
```

The model is never called — an impossible profile costs nothing — and the
allergy is never offered as something to give up, even though dropping it would
also unlock meals. The CLI prints the same ladder:

```bash
BEATROOT_OFFLINE=1 uv run beatroot recommend "dinner" --max-prep 1 --medical peanut
```

```
trace: FEASIBILITY -> NEGOTIATE (audit_id=7c22c8c6-...)
NEGOTIATE — no meal recommended. 0/174 recipes survived.
  relax                              unlocks   severity
  allow more than 1 minute of prep   166       preference
  LOCKED (never relaxed — medical/religious): med0
```

**Unverifiable.** A real allergen the catalog has no data on:

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"unverifiable",
  "constraints":[{"id":"excl_sulfite","kind":"exclude_tag","severity":"medical",
                  "value":"sulfite","source":"structured"}],
  "query":"dinner"}'
```

```
ESCALATE — reason "unknown_ingredient"
detail: "cannot verify constraint(s) against the catalog vocabulary:
         excl_sulfite (medical) names an unrecognised value: 'sulfite'"
```

Sulfite is real; this catalog cannot check it. "No data" is never treated as
"none present".

**The fourth state, PENDING_REVIEW**, was not reachable here. It fires when a
composite trust score lands in `[0.55, 0.70)` *and* the profile carries a hard
medical constraint. Every medical preset in this catalog scores far above that
band against the live model, so the gate is exercised in the test suite with a
forced grey-band score instead: it yields `terminal_state: PENDING_REVIEW` and
a `thread_id`, resumable via `POST /recommend/{thread_id}/resume` with
`{"approved": true|false}` or `beatroot resume`.

## 6. What to look for in a response

```bash
curl -s -X POST $BASE/recommend -H 'Content-Type: application/json' -d '{
  "profile_id":"severe_nut_allergy",
  "constraints":[{"id":"excl_peanut","kind":"exclude_tag","severity":"medical",
                  "value":"peanut","source":"structured"}],
  "query":"something warm with rice"}'
```

```
COMMIT — Vegan sambar rice
trust: composite 0.9875
explanation: null, explanation_status "pending"
cost: $0.0023375  (rewrite_query $0.001254, rerank $0.0010835)
```

- **`nutrition.provenance: "computed"`** — the tell that nutrition was
  calculated from the ingredient catalog, not generated. The type refuses to
  hold any other value.
- **`ingredients`** carry names and grams, not bare ids.
- **`constraints_satisfied_display`** is human language ("is vegan"); the raw
  ids are kept alongside it for the audit trail.
- **`trust.composite`** and its three named parts, weighted 0.45 / 0.30 / 0.25.
  The model's own confidence is deliberately the minority signal.
- **Prose is generated off the request path.** Poll for it:

```bash
curl -s $BASE/recommend/<thread_id>/explanation
```

```json
{"explanation_status":"ready",
 "explanation":"This vegan sambar rice is a nutrient-dense meal that provides
  a balanced combination of protein, fiber, and carbohydrates, making it both
  satisfying and wholesome. Additionally, it meets the user's dietary
  constraint by being free of peanuts."}
```

Ready within a few seconds. Before serving it, the queue runs drift detection
against the same verified facts the prompt was given; a finding lands the
explanation as `failed` rather than serving ungrounded prose.

`GET /metrics` accounts for all of it:

```json
{"feasibility_cache":{"hits":2,"misses":5,"hit_rate":0.286},
 "embedding_cache":{"hits":2,"misses":4,"hit_rate":0.333},
 "incidents":4,
 "cost":{"per_stage_usd":{"compile":0.010623,"rewrite_query":0.00724075,
                          "rerank":0.00667425},
         "total_usd":0.024538,"plans":9,"per_plan_usd":0.00272644,
         "tokens":8132,"tokens_saved":506,
         "tokens_saved_estimate_method":"chars(rendered rerank + explain
          prompts, real catalog sample) / 4 chars-per-token"}}
```

`tokens_saved` is labelled an estimate everywhere it appears, because it is
one — it is what the skipped rerank and explain prompts would have cost.

## 7. Tests and evals

```bash
uv run pytest              # 5 skips are the Qdrant tests, gated on QDRANT_URL
uv run ruff check .
uv run mypy --strict src
```

The suite is hermetic: a session fixture hides the repo `.env` and strips
provider credentials, so a machine holding real keys cannot accidentally run
the whole suite against a live provider. With a Qdrant server reachable, the
skips run:

```bash
QDRANT_URL=http://localhost:6333 uv run pytest tests/retrieval/
```

The safety suites, offline and free:

```bash
BEATROOT_OFFLINE=1 uv run python -m beatroot.eval.runners.system
```

```
A1_allergen_safety          1.000 / 1.0    PASS
A2_religious_integrity      1.000 / 1.0    PASS
A3_injection_resistance     1.000 / 0.95   PASS
A4_infeasibility_detection  1.000 / 0.95   PASS
A5_escalation_correctness   1.000 / 0.9    PASS
A6_explanation_grounding    1.000 / 0.95   PASS
hard constraint violations: 0 (threshold 0)
overall: PASS
```

```bash
BEATROOT_OFFLINE=1 uv run python -m beatroot.eval.runners.components
```

```
retrieval recall@k (full oracle)       0.988
retrieval recall@k (hard-only oracle)  1.000
retrieval leakage                      0
feasibility accuracy                   1.000
nutrition determinism                  1.000
drift detection recall                 1.000
overall: PASS (retrieval_leakage must be 0)
```

Read `EVAL_RESULTS.md` before trusting any of those 1.000s — particularly the
section on why a green row is a claim rather than the evidence.

## 8. Known limits when driving it

- **Free-text parsing is model-driven, so exact wording matters.** The same
  intent phrased two ways can land on different constraint kinds, and one of
  them may name something outside the catalog vocabulary and legitimately
  escalate. Rephrase and retry before treating a free-text escalation as a bug.
- **The catalog is closed-world.** Anything outside its tags, cuisines and
  ingredients is unverifiable and refuses rather than guesses.
- **One meal at a time.** No multi-day planning, no accounts, no history beyond
  the audit and incident logs the system keeps about itself.
- **Dense retrieval is not semantic** in any configuration here. See
  `CUT_LIST.md`.
