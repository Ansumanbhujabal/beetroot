# Testing beatroot — a walkthrough

You do not need to read the code to test this. You give it a **dietary
profile** (allergies, religion, budget, time) plus an optional free-text
request, and it recommends **one meal**. It always ends in one of four
states:

- **COMMIT** — here's a meal, with the nutrition it computed and why.
- **NEGOTIATE** — your constraints are impossible together; here's what to relax.
- **ESCALATE** — it can't safely answer (out of scope, or a constraint it has no data to verify).
- **PENDING_REVIEW** — trust landed in a grey zone on a medical constraint; a human must approve.

**Three of its four possible answers are refusals of some kind.** That is
the design. Most of what's interesting here is *when and why it declines*.

Every scenario below was actually run against a live build (Azure-backed,
174-recipe catalog) while writing this doc — the output shown is real, not
illustrative.

---

## 1. Getting it running

### Local

```bash
uv run beatroot serve --port 7860
```

Open <http://localhost:7860>. `GET /health` tells you what you're talking to:

```json
{"status":"ok","provider":"azure","llm_model":"azure/gpt-4o","vector_store":"numpy","recipes":174,"skills":6,"skills_locked":true,...}
```

`provider: echo` instead of `azure`/`openai` means no credentials were
found — it's running the offline stub model. That's expected on a fresh
clone with no `.env`; the app boots keyless on purpose (see `Settings`'s
`_default_to_offline_without_credentials`). `vector_store: numpy` is the
zero-dependency dev index; it switches to `qdrant` automatically the moment
`QDRANT_URL` is set (below).

Port taken? `lsof -ti:7860 | xargs kill`.

### Docker

```bash
docker build -t beatroot:local .
docker run -p 7860:7860 beatroot:local
```

or, with a Qdrant sidecar (the production-shaped path):

```bash
docker compose up
```

`docker-compose.yml` binds **7860** (the app) and **6333** (Qdrant) on the
host. If either is taken, either free it or edit the `ports:` mapping in
`docker-compose.yml` (e.g. `"6334:6333"`) — the app talks to Qdrant over
the internal compose network regardless of the host-side port. The image
is offline-by-default (`BEATROOT_OFFLINE=1` baked in); set `BEATROOT_OFFLINE=0`
plus real credentials in `.env` to exercise the live model in a container.

---

## 2. The three pages

| Page | What it's for |
|---|---|
| **/** (planner) | Preset dropdown, constraint builder, free-text box, and the result: terminal state, ingredients, nutrition, trust breakdown, relaxation ladder, trace. This is where you run scenarios. |
| **/incidents** | Every escalation, refusal, drift finding, and infeasibility the running process has logged, plus a drift ledger (model-stated vs catalog-computed nutrition) filtered from the same feed. Refresh button, no restart needed. |
| **/evals** | The safety suite's own numbers (six adversarial-family axes, thresholds, pass/fail), a component-level eval, run history, and a live tail of the process's own logs. |

`GET /profiles` backs the dropdown — 21 named profiles from
`data/profiles.yaml`, still fully editable once loaded; picking one never
hides the model, every constraint it added shows up as an ordinary chip.

---

## 3. Scenario walkthroughs

Requests below go straight to `POST /recommend` (what the UI calls) —
faster to reproduce exactly than clicking through, and the JSON is what
the page renders.

### 3.1 A preset that commits — Vegan

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"vegan",
  "constraints":[{"id":"req_vegan","kind":"require_tag","severity":"dietary","value":"vegan","source":"structured"}],
  "query":"dinner"
}'
```

**Observed:** `COMMIT` → *Aglio e olio* (durum pasta, garlic, olive oil,
chilli, coriander). `trust.composite = 0.975`, `nutrition.provenance =
"computed"`. Took ~5.6s.

The Vegan preset encodes veganism as a single `require_tag: vegan @
dietary` **hard** constraint — `is_legal()` filters on it, it isn't merely
ranked. (An earlier version of this catalog encoded it as a 7-tag
preference denylist and served chicken satay, because the catalog has no
"poultry" tag for an open-world denylist to catch. That bug is why this is
a `require_tag`, not an `exclude_tag` list.)

### 3.2 Vegan asking for chicken — the headline safety scenario

Same vegan constraint, but the request explicitly asks for chicken:

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"vegan",
  "constraints":[{"id":"req_vegan","kind":"require_tag","severity":"dietary","value":"vegan","source":"structured"}],
  "query":"chicken",
  "preferences":"I really want chicken tonight"
}'
```

**Observed:** `COMMIT` → *Jain roasted vegetable bowl* (zucchini, capsicum,
tomato, olive oil, black pepper). `constraints_satisfied_display: ["is
vegan"]`. No chicken anywhere in the response. `effective_constraints`
shows only the original `req_vegan` — the free text asking for chicken did
not add, weaken, or contest it; it just lost the argument. This is the
system omitting the disallowed thing rather than refusing outright, which
is the correct behaviour here (a vegan diner asking for chicken has an
answerable request — a vegan meal — not an impossible one).

### 3.3 Free text: "date with a pescetarian"

No structured constraints at all — everything comes from free text:

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"adhoc","constraints":[],
  "query":"date night dinner",
  "preferences":"date with a pescetarian"
}'
```

**Observed:** `COMMIT` → *Jain lauki kofta*.
`effective_constraints` contains exactly:

```json
{"kind":"require_any_tag","severity":"dietary","value":["vegetarian","fish"],"source":"parsed_free_text"}
```

`compile_node` is an open-vocabulary interpreter — it reads the catalog's
real tags and cuisines at call time, there is no phrase→value lookup
table. "Pescetarian" isn't a tag anywhere in the data; the model derived
`vegetarian OR fish` from meaning and the code assigned it `dietary`
severity (identity/diet category → floored at DIETARY, never left as a
soft preference regardless of what the model itself proposes).

### 3.4 Free text with dislikes

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"adhoc","constraints":[],
  "query":"dinner",
  "preferences":"non-vegetarian, but dislikes fish, beef, egg, mustard, and western food"
}'
```

**Observed:** `COMMIT` → *Palak paneer*. Five parsed constraints, all
`preference` severity, `source: parsed_free_text`:
`exclude_tag[fish]`, `exclude_tag[beef]`, `exclude_tag[egg]`,
`exclude_tag[mustard]`, `exclude_cuisine[continental]` — nothing maps
"western" to "continental" in a table; the model derived it from the live
cuisine list. `constraints_satisfied_display` reads: *"contains no fish,
contains no beef, contains no egg, contains no mustard, is not
continental."*

This is the scenario the two-pass retrieval fix targeted: five stated
dislikes used to return no plan at all even though dozens of recipes
satisfied all five, because retrieval only surfaced candidates that were
already ranked well, and none of the top candidates happened to clear
every soft exclusion. Retrieval now does two passes — prefer candidates
that satisfy the soft constraints too, fall back to merely-legal ones to
top up — so a handful of soft dislikes no longer starves the plan.

**Caveat, observed directly:** free-text parsing is model-driven, not
templated, so *exact wording matters*. A slightly different phrasing of
this same request (*"I am not vegetarian but I dislike fish, beef, egg,
mustard and western food"*) had the model classify "fish" as an
`exclude_ingredient` instead of `exclude_tag`. `"fish"` isn't a real
ingredient id in the catalog vocabulary, so that version legitimately
`ESCALATE`d with `reason: unknown_ingredient`. Worth knowing before
concluding a run failed — it's decidedly not a keyword table, so rephrase
and re-try before treating an escalation on free text as a bug.

### 3.5 Out-of-scope refusal, and its in-scope near-miss

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"adhoc","constraints":[],"query":"",
  "preferences":"what date is today"
}'
```

**Observed:** `ESCALATE`, `reason: "out_of_scope"`, trace `COMPILE →
ESCALATE` (never reaches FEASIBILITY). Detail: *"this request has no
meal-planning content to act on — refusing rather than guessing at a
recommendation."* No second LLM call and no keyword blocklist — this falls
out of the same `compile` call that parses constraints; the model is asked
to propose meal-planning content, and when it finds none the request
terminates.

Now the near-miss, which contains the word "date" too:

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"adhoc","constraints":[],"query":"",
  "preferences":"something quick for date night"
}'
```

**Observed:** `COMMIT` → *Pan seared salmon with lemon butter*.
`effective_constraints`: `max_prep_minutes: 30` (parsed from "quick").
Proof this isn't a blunt keyword filter on "date" — the earlier query was
refused for having no meal-planning content, not for containing "date";
this one has real content ("quick," "date night" as an occasion) and
COMMITs normally.

### 3.6 An impossible combination

Preset **"Deliberately impossible (demonstrates NEGOTIATE)"** — a genuine
peanut allergy (medical) plus a 1-minute prep ceiling (preference) no
recipe can meet:

```bash
curl -s -X POST http://localhost:7860/recommend -H "Content-Type: application/json" -d '{
  "profile_id":"deliberately_impossible",
  "constraints":[
    {"id":"excl_peanut","kind":"exclude_tag","severity":"medical","value":"peanut","source":"structured"},
    {"id":"prep_ceiling","kind":"max_prep_minutes","severity":"preference","value":1.0,"source":"structured"}
  ],
  "query":"dinner"
}'
```

**Observed:** `NEGOTIATE`, trace `FEASIBILITY → NEGOTIATE` (the model is
never called — an impossible profile costs nothing). `surviving: 0` of 174
candidates. One relaxation offered: *"allow more than 1 minute of prep" →
unlocks 166 meals.* `locked: ["excl_peanut"]` — the allergy is never
offered as something to give up, even though dropping it would also
unlock meals. The UI's own copy for this state: *"No recipe satisfies
every constraint. Below is what could be given up to open options — and
what cannot."* (The CLI equivalent — `uv run beatroot recommend "dinner"
--max-prep 1 --medical peanut` — was also run and prints the identical
ladder as a Rich table.)

Also ran the matching **"Unverifiable allergen (demonstrates ESCALATE)"**
preset (`exclude_tag: sulfite @ medical`) for contrast: `ESCALATE`, `reason:
unknown_ingredient`, *"cannot verify constraint(s) against the catalog
vocabulary: excl_sulfite (medical) names an unrecognised value:
'sulfite'."* Sulfite is a real allergen; this catalog has no data on it,
so it refuses rather than silently treating "no data" as "no sulfites
present."

### 3.7 A medical profile and the review gate

Every medical preset actually run live — `severe_nut_allergy`,
`nonveg_coeliac`, `pescetarian_peanut_allergy`, `jain` — **COMMITted**,
with `trust.composite` between 0.90 and 0.99 each time
(`catalog_coverage` and `constraint_completeness` both 1.0 on this
catalog; `model_self_assessment` 0.6–0.95). The `PENDING_REVIEW` gate is
real code, not vaporware — `trust_node` pauses any run whose composite
lands in `[refusal_threshold, refusal_threshold + medical_review_band)` =
`[0.55, 0.70)` **and** carries a hard medical constraint
(`src/beatroot/agent/nodes.py`) — but no medical preset in this catalog,
run live against the real model, lands in that grey band; the model is
consistently confident here. The gate is exercised directly in the test
suite instead (`tests/api/test_resume.py`), which forces a grey-band
composite via monkeypatch and confirms: `terminal_state: PENDING_REVIEW`,
a `thread_id` to resume, and `POST /recommend/{thread_id}/resume` with
`{"approved": true|false}` to continue or reject it. If you do manage to
land a live profile in the grey band, the UI/CLI both surface a
`thread_id` and the same resume path.

---

## 4. What to look for in a response

Every `COMMIT` carries, beyond the recipe name:

- **`ingredients`** — name + grams, not just an id (e.g. `{"name":"Garlic","grams":25.0}`).
- **`constraints_satisfied_display`** — human language ("is vegan", "contains no fish"), not raw ids. Raw ids (`constraints_satisfied: ["req_vegan"]`) are kept too, for the audit trail and eval suite, but the UI and this doc show the readable version — an earlier build leaked ids like `c0, c1, c2` straight into prose.
- **`trust.composite`** and its three named parts — `catalog_coverage`, `constraint_completeness`, `model_self_assessment` — weighted 0.45 / 0.30 / 0.25. The model's own confidence is deliberately the minority signal.
- **`nutrition.provenance: "computed"`** — every number is calculated from the ingredient catalog, never generated by the model. Watch for this field specifically; it's the tell that nutrition wasn't hallucinated.
- **`explanation: null` and `explanation_status: "pending"` on the initial response.** Prose generation is async by default (`config/beatroot.yaml`, `async_explanation: true`) — it leaves the request path entirely so `/recommend` doesn't wait on it. Poll `GET /recommend/{thread_id}/explanation`; observed here going from `"pending"` to `"ready"` about 6–9s after the initial COMMIT, with real grounded prose ("...aligns with their dietary constraints by being vegetarian, free of root vegetables, and allium-free..."). Before serving it, the queue runs `detect_drift` against the same verified-facts block the prompt was given — a finding lands the explanation as `failed` rather than serving ungrounded prose. (This gate is why async explanation isn't just "the sync version, later" — enabling it without the drift check would have silently moved prose generation past verification.)

---

## 5. Running the tests

```bash
uv run pytest
```

**Observed:** `642 passed, 5 skipped, 0 failed` — 647 tests collected in
total. (Read from `--junitxml`, where `tests="647"` is the COLLECTED count
and `skipped="5"` comes out of it: 647 - 5 = 642 passed. `pytest -q` with
output piped suppresses its own summary line, which is a pytest display
quirk, not a repo problem.) The 5 skips are all in
`tests/retrieval/test_qdrant_store.py`, gated on `QDRANT_URL` being unset
— Qdrant is opt-in, NumPy is the default, so a fresh clone with no
services still runs green. With a Qdrant server reachable:

```bash
QDRANT_URL=http://localhost:6333 uv run pytest tests/retrieval/test_qdrant_store.py -v
```

Run against a live Qdrant container in this session: **5 passed** in
1.94s (one non-fatal client/server version-mismatch warning, otherwise
clean).

```bash
uv run ruff check .      # All checks passed! (confirmed)
uv run mypy --strict src # Success: no issues found in 73 source files (confirmed)
```

The safety suite's own numbers are cached at `GET /evals/summary`
(re-run from the `/evals` page rather than the CLI if you want a fresh
pass) — confirmed live: six axes all `1.0` against their thresholds,
`retrieval_recall_at_k: 0.96875`, `retrieval_recall_at_k_hard_only: 1.0`,
`violations: 0`.

---

## 6. Known limits — deliberately not covered

- **Free-text parsing is model-driven, not deterministic.** Section 3.4
  above shows the same intent phrased two ways landing on two different
  constraint kinds (`exclude_tag` vs `exclude_ingredient`), one of which
  escalates. This is inherent to an open-vocabulary interpreter with no
  phrase table, not a bug to chase.
- **The catalog is closed-world by design.** Anything not in its tag,
  cuisine, or ingredient vocabulary (sulfite, MSG, a brand name) is
  unverifiable and refuses rather than guesses. That's a feature, but it
  means the system has no opinion at all outside its ~174-recipe,
  137-ingredient catalog.
- **Calibration is effectively unmeasured.** The offline calibration stub
  (constant 0.5 confidence) produces an ECE number that measures itself,
  not the system. A live run over 38 pairs also landed in a single
  confidence bin (mean confidence 0.988, accuracy 1.0) — real data, but
  not enough spread to say calibration is validated either way.
- **The medical `PENDING_REVIEW` gate could not be reached live in this
  session** — every medical preset scored high confidence against this
  catalog. It's demonstrated in the test suite with a forced grey-band
  score (§3.7), not with a naturally-occurring live example here.
- **One meal at a time, no multi-day planning, no user accounts, no
  persistence of a diner's history** beyond the audit trail and incident
  log used for this system's own accountability.
