"""Hybrid retrieval pipeline: fuse lexical + dense, then rerank with the model.

Constraints are compiled into the retrieval itself, not applied around it.
Hard-constraint exclusion tags are pushed DOWN into both stores — a `NOT`
clause in FTS5 (`lexical_search`), a score mask in the vector store
(`VectorStore.search`) — so no illegal candidate is ever scored, and ranking
quality never competes with safety the way it would if filtering happened
after the fact. The residual `is_legal` pass below covers only what a tag
filter cannot express (nutrient ranges, budget, prep time) over the already
small fused set, not the whole catalog. Spec §10, §17.

`llm_rerank` never lets a model escape the legal set it is handed: it only
reorders `retrieve()`'s output, and any choice it can't trust (an
out-of-range, negative, non-integer, or missing `choice_index`) falls back to
the fused top rank rather than raising or guessing.
"""

from pydantic import BaseModel

from beatroot.contracts.core import ConstraintSet
from beatroot.contracts.trust import CostRecord
from beatroot.reasoning.llm import LLMClient
from beatroot.reasoning.prompts import load_prompt
from beatroot.retrieval.dense import VectorStore, get_vector_store
from beatroot.retrieval.fusion import rrf
from beatroot.retrieval.lexical import lexical_search
from beatroot.settings import get_settings
from beatroot.t0_invariants.constraints import check_recipe, is_legal
from beatroot.trusted.catalog import Catalog, Recipe


class _Choice(BaseModel):
    choice_index: int


def retrieve(
    query: str,
    cs: ConstraintSet,
    catalog: Catalog,
    provider: LLMClient,
    vector_store: "VectorStore | None" = None,
    top_k: int | None = None,
    affinity: dict[str, float] | None = None,
) -> list[Recipe]:
    """Hybrid retrieval: FTS5 + dense, fused by RRF, filtered for legality.

    Every weight, limit, and RRF `k` comes from `settings.retrieval` — never
    a literal here. `top_k` defaults to `settings.retrieval.top_k`.
    """
    settings = get_settings().retrieval
    if top_k is None:
        top_k = settings.top_k

    # `exclude_tag` constraints always carry a str tag; see the identical
    # guard in `t0_invariants.feasibility._survivors` for why this filters
    # on `isinstance` rather than trusting the union type.
    excluded = [c.value for c in cs.hard() if c.kind == "exclude_tag" and isinstance(c.value, str)]

    lex = lexical_search(catalog.conn, query, limit=settings.candidate_limit, exclude_tags=excluded)
    index = vector_store or get_vector_store(provider, catalog)
    den = index.search(query, limit=settings.candidate_limit, exclude_tags=excluded)

    rankings = [lex, den]
    weights = [settings.lexical_weight, settings.dense_weight]

    if affinity:
        candidate_ids = {i for i, _ in lex} | {i for i, _ in den}
        pref = sorted(
            (
                (r.id, sum(affinity.get(t, 0.0) for t in r.tags))
                for r in catalog.recipes()
                if r.id in candidate_ids
            ),
            key=lambda kv: -kv[1],
        )
        rankings.append(pref)
        weights.append(settings.affinity_weight)

    fused = rrf(rankings, k=settings.rrf_k, weights=weights)

    # Residual check: constraints no tag filter can express (nutrient_range,
    # budget_max, max_prep_minutes), evaluated over the small fused set —
    # not the whole catalog. Hydration (nutrition/cost) happens lazily, one
    # candidate at a time, and stops as soon as top_k legal recipes are found.
    # TWO PASSES, and the second one is the point.
    #
    # `is_legal()` gates HARD constraints only — soft ones are ranking's job
    # by design. But ranking was not actually doing that job: a request
    # stating five dislikes (fish, beef, egg, mustard, western food) returned
    # a dish violating two of them and ESCALATED at verify, while 55 recipes
    # in the catalog satisfied all five. `check_recipe()` evaluates every
    # constraint, so `verify_node` escalates on unmet soft constraints —
    # which means "ranking's job" failing is not a softer outcome for the
    # user, it is no plan at all.
    #
    # So: prefer candidates that satisfy the soft constraints too, and fall
    # back to merely-legal ones only to top up. Preference order within each
    # pass stays the fused RRF order. This never admits anything `is_legal()`
    # rejects — the safety gate is unchanged and still runs first — it only
    # decides which of the already-legal candidates to hand onward.
    #
    # Deliberately a preference, not a filter: a diner whose dislikes cannot
    # all be met still gets a plan (with the violation surfaced at verify)
    # rather than a refusal, which is what treating soft constraints as hard
    # would produce.
    preferred: list[Recipe] = []
    fallback: list[Recipe] = []
    for item_id, _score in fused:
        recipe = catalog.recipe(item_id)
        if recipe is None:
            continue
        recipe = catalog.hydrate(recipe)
        if not is_legal(recipe, cs):
            continue
        if check_recipe(recipe, cs).ok:
            preferred.append(recipe)
            if len(preferred) >= top_k:
                return preferred
        elif len(fallback) < top_k:
            fallback.append(recipe)
    return (preferred + fallback)[:top_k]


def llm_rerank(
    query: str, candidates: list[Recipe], provider: LLMClient, preferences: str = ""
) -> tuple[Recipe | None, str, float, CostRecord]:
    """Reorder already-legal candidates by preference fit. Returns
    `(chosen_recipe, rationale, self_assessment, cost)`.

    Operates only over candidates `retrieve()` already proved legal — this
    function cannot make anything unsafe, and it must never let a model pick
    its way outside that set. An out-of-range, negative, non-integer, or
    missing `choice_index` degrades to the fused top rank (`candidates[0]`)
    instead of raising or trusting the model's arithmetic.

    `cost` is the `Completion.cost` this call actually spent — a COMMIT
    spends real money on two model calls (rerank here, explain later), and
    dropping this one silently understated `per_plan_usd` by roughly half
    against a real provider. The no-candidate and single-candidate short
    circuits below never call the model, so they return `CostRecord()`
    (all zeros), not a fabricated cost.
    """
    if not candidates:
        return None, "", 0.0, CostRecord()
    if len(candidates) == 1:
        return candidates[0], "only legal candidate", 1.0, CostRecord()

    listing = "\n".join(
        f"{i}. {r.name} ({r.cuisine}, {r.prep_minutes} min)" for i, r in enumerate(candidates)
    )
    rerank_prompt = load_prompt("rerank")
    completion = provider.complete(
        rerank_prompt.render(query=query, preferences=preferences, candidates=listing),
        schema=_Choice,
        stage="rerank",
        prompt_ref=rerank_prompt,
    )
    parsed = completion.parsed or {}
    idx = parsed.get("choice_index", 0)
    if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(candidates):
        idx = 0  # out-of-range/negative/non-integer model output falls back
        # to the fused top rank rather than raising or escaping the legal set.
    # A missing self-assessment is treated as a weak signal, not a confident
    # one — reuse the same floor the trust layer uses for "unknown", rather
    # than a locally invented number.
    fallback = get_settings().trust.weak_signal_floor
    return (
        candidates[idx],
        parsed.get("rationale", ""),
        float(completion.self_assessment) if completion.self_assessment is not None else fallback,
        completion.cost,
    )
