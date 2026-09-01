---
id: retrieve_candidates
name: Retrieve Candidates
tier: T1
llm_permitted: true
triggers_on: ["after FEASIBILITY"]
priority: 30
---

## When to use

After `check_feasibility` has confirmed at least one meal survives the full
ConstraintSet, before trust is scored. This skill turns "some meals are
legal" into "here are the best few legal meals for this query," using BM25,
dense similarity, and a model-scored rerank of the fused top slice.

## The pattern

1. Compile the exclude set (every hard constraint — MEDICAL and RELIGIOUS
   included) into each store's own filter mechanism: an FTS5 `NOT (...)`
   clause for lexical search, a Qdrant payload `must_not` filter for the
   production vector store, or an in-memory tag mask for the NumPy dev
   fallback. The filter runs INSIDE the store's own search call, not around
   it — the same `-inf` masking / `NOT` clause shape either way.
2. Run lexical and dense search independently against the already-filtered
   corpus. Each returns its own ranked `(id, score)` list on its own scale
   — BM25 (bounded above, lower-is-better before negation) and cosine
   similarity (`[-1, 1]`) — and neither is comparable to the other yet.
3. Fuse the two rankings with RRF (reciprocal rank fusion), by rank
   POSITION, never by raw score: each item earns `weight / (k + rank + 1)`
   per source, so an item ranked well by both signals beats one ranked
   well by only one, with no normalisation step in between to get wrong.
4. Only now, on the fused top-k, is the model allowed near the data: it
   reranks by preference fit — cuisine, texture, stated likes — using the
   rerank prompt, which states outright that every candidate it sees has
   already been verified safe and legal. It is never asked to reason about
   safety, only about fit.
5. Return the reranked list to `compute_nutrition`. Retrieval's job ends at
   "these are the best legal candidates for this query," not at "this is
   the meal" — trust and verification still have to run.

## Pitfalls

- **Filtering after scoring instead of before.** Retrieving broadly and
  discarding illegal rows afterward means an illegal candidate was scored,
  ranked, and held in memory next to legal ones — and at catalog scale,
  "retrieve everything then filter" simply stops finishing in time. The
  filter has to be a query-time clause or a payload condition evaluated
  inside the store, never a Python-side `if tag not in excluded` applied to
  an already-ranked list.
- **Normalising BM25 and cosine onto a shared scale before combining them.**
  It looks like the careful thing to do and it is not: a BM25 score of 8
  and a cosine of 0.8 carry no relative meaning to each other, so whatever
  normalisation you pick quietly decides a winner by construction. RRF's
  entire point is not needing that step.
- **Letting the reranker see anything but preference signals.** The rerank
  prompt is deliberately narrow. If the rerank step starts taking allergen
  tags, budget, or constraint status into its context "for a better
  ranking," it has quietly reopened a safety question a model is not
  allowed to answer — even if it happens to answer it correctly this time.
- **Shipping the NumPy dev fallback as the production path.** `DenseIndex`
  holds the whole embedding matrix in memory and masks exclusions with a
  Python tag scan; it exists so a fresh clone runs with no external
  services, not because it's the design target. It stops scaling well
  before this system's target catalog size — `QdrantVectorStore` is the
  actual answer, and it's selected automatically the moment `QDRANT_URL`
  is set, so there's no excuse to be running the fallback anywhere that
  matters.
