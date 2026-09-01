# Start here

An index. Every document below is current as of the latest commit; where a
number appears twice it has been reconciled, and where something is unproven
it says so rather than staying quiet.

---

## If you have 15 minutes and want to drive the system

**[`TESTING.md`](TESTING.md)** — the reviewer's walkthrough. Nine scenarios,
every one of them actually executed against a running server with the real
observed output pasted in, not described from memory. Covers the vegan-asks-
for-chicken case, free-text planning, the out-of-scope refusal (with an
in-scope near-miss so you can see it is not a keyword filter), and an
impossible combination.

Fastest path: `uv run beatroot serve`, then open <http://localhost:7860>.

---

## If you want to understand how it works

| Document | What it gives you |
| --- | --- |
| **[`README.md`](README.md)** | The project in one page: the thesis, the two invented features, how to run it. |
| **[`ARCHITECTURE.md`](ARCHITECTURE.md)** | The tier map and the request path, at a level you can hold in your head. |
| **[`docs/ARCHITECTURE_DEEP.md`](docs/ARCHITECTURE_DEEP.md)** | The real thing — every flow traced to `module:function`, 8 diagrams, and the reasoning behind each boundary. Read this if you are going to ask hard questions. |
| **[`docs/diagrams/`](docs/diagrams/) · `/docs` in the app** | The architecture diagram as `.drawio`, PNG, SVG and PDF. The running app serves it at `/docs` with pan and zoom, and the same page has download links for every document here. |

---

## If you are preparing to be questioned on it

**[`docs/INTERVIEW_PREP.md`](docs/INTERVIEW_PREP.md)** — the main preparation
document. Q&A covering what, why, and *the alternative that was rejected*;
nine war stories told as problem → investigation → root cause → fix → what it
generalises to; hostile questions answered honestly rather than defensively;
and a rapid-fire table of every number with its source file.

**[`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md)** — a timed 8–10 minute beat
sheet, plus the numbers to keep on screen and two lines worth saying verbatim.

The one thing to internalise before either: **every score in this project is
1.000, and a week ago most of them were also 1.000 while three real safety
bugs sat underneath.** The defensible claim is not the scores — it is that the
evals can now be made to fail on demand.

---

## If you want to know what is measured, and what is not

| Document | What it gives you |
| --- | --- |
| **[`EVAL_RESULTS.md`](EVAL_RESULTS.md)** | The eval design: system axes, component metrics, adversarial families, the free oracle — and a section on what the numbers do **not** prove. |
| **[`EVAL_HISTORY.md`](EVAL_HISTORY.md)** | Every eval run, each attributed to a named change, **including the iterations that were reverted**. The recall jump is broken out as +0.2506 algorithm vs +0.0309 data, because the obvious reading of it is wrong. |
| **[`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md)** | Requirements coverage and the honest gap list. Splits "ready as a take-home" from "ready to serve real users with allergy consequences" — those are different bars and the answer differs. |
| **[`CUT_LIST.md`](CUT_LIST.md)** | What was deliberately not built, what it would cost to add, and the standing limitations. |

---

## Running it

```bash
uv run beatroot serve                 # http://localhost:7860
uv run pytest                         # 656 collected, 651 passed, 5 skipped
uv run ruff check . && uv run mypy --strict src

docker compose up                     # app + Qdrant, verified working
```

The 5 skips are the Qdrant tests. They skip cleanly when `QDRANT_URL` is
unset. With a Qdrant container running:

```bash
docker run -d -p 6533:6333 qdrant/qdrant:v1.12.0
QDRANT_URL=http://localhost:6533 uv run pytest     # 656 passed, 0 skipped
```

**Note:** those tests *error* rather than skip if `QDRANT_URL` is set but the
server is unreachable. Either start the container or unset the variable.

---

## Before sending this anywhere

```bash
./scripts/package_submission.sh
```

Builds the archive from `git archive`, so only tracked files can enter, then
re-scans the extracted result for credential patterns and **fails closed**.
`.env` is untracked, but a naive `zip -r` would still sweep it in — this is
the guard against that. It has been verified against a planted copy of a real
key.

---

## The honest summary

Nine real bugs surfaced during the final week of this build: a vegan served
chicken, egg admitted by vegetarian profiles, asafoetida missing its gluten
tag (12 dishes legal for a coeliac), a constraint kind advertised to the model
and silently dropped by the parser, internal ids leaking into user-facing
prose, a latency change that disabled a grounding check, a crash on shutdown,
a drift ledger that failed open on hedged prose, and a socket leaked per model
call.

**None of them were caught by a failing test.** The suite was green
throughout. Every one came from someone using the application or reading its
logs — which is the most useful thing this project has to say about the limits
of its own test suite, and it is written down rather than left to be
discovered.
