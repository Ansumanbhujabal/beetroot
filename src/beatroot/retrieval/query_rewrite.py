"""Query rewriting: expand a short natural-language query into better
retrieval terms BEFORE it reaches the lexical/dense stores.

This is an optimisation over ranking, never a safety decision — every
candidate `retrieve()` returns is still filtered for legality exactly as
it always was; a bad or missing rewrite can only make ranking worse, never
make an unsafe recipe rankable. Three properties make that true structurally,
not just by convention:

- **Skipped entirely on an empty query.** No model call, no trace entry, no
  cost — mirrors the same zero-token posture `compile_node` (GAP 1) already
  established for empty free text.
- **Degrades to the original query on ANY failure** — a non-JSON reply, a
  missing/blank `rewritten_query`, a raised exception. A rewrite is
  optional; a recommendation is not, so nothing here is allowed to turn a
  rewrite failure into a failed recommendation.
- **Both the original and the rewritten text are always returned**
  (`QueryRewrite.original` / `.rewritten`), so the effect — or the absence
  of one — is visible to a caller rather than silently applied.
"""

from pydantic import BaseModel, Field

from beatroot.contracts.trust import CostRecord
from beatroot.reasoning.llm import LLMClient
from beatroot.reasoning.prompts import load_prompt


class QueryRewrite(BaseModel):
    """What actually happened to one query. `applied=False` means
    `rewritten == original` — either there was nothing to expand or the
    rewrite step degraded — never a silent no-op a caller can't see."""

    original: str
    rewritten: str
    terms: list[str] = Field(default_factory=list)
    applied: bool = False
    cost: CostRecord = Field(default_factory=CostRecord)


def rewrite_query(query: str, llm: LLMClient) -> QueryRewrite:
    """Expand `query` into retrieval terms via `prompts/rewrite_query.md`.

    Deterministic offline (`LLMClient.offline()` — see
    `reasoning.llm.LLMClient._offline_rewrite_query`): a small, fixed
    expansion table keyed on tokens actually present in the query, so a
    demo run with no credentials still shows a real, reproducible effect
    rather than the identity function.
    """
    text = (query or "").strip()
    if not text:
        # Zero-token guarantee: an empty query must not spend a call.
        return QueryRewrite(original=text, rewritten=text, terms=[], applied=False)

    try:
        rewrite_prompt = load_prompt("rewrite_query")
        completion = llm.complete(
            rewrite_prompt.render(query=text),
            stage="rewrite_query",
            prompt_ref=rewrite_prompt,
        )
    except Exception:
        # A rewrite is an optimisation, never a dependency — any failure
        # (network, provider, timeout) degrades to the original query
        # rather than costing the recommendation itself.
        return QueryRewrite(original=text, rewritten=text, terms=[], applied=False)

    parsed = completion.parsed or {}
    rewritten = parsed.get("rewritten_query")
    raw_terms = parsed.get("terms")
    terms = [t for t in raw_terms if isinstance(t, str)] if isinstance(raw_terms, list) else []

    if not isinstance(rewritten, str) or not rewritten.strip():
        # A malformed or empty reply is the same "degrade, don't guess"
        # posture as everywhere else in this codebase — the call's own
        # cost still rides along, since it genuinely was spent.
        return QueryRewrite(
            original=text, rewritten=text, terms=[], applied=False, cost=completion.cost
        )

    rewritten = rewritten.strip()
    return QueryRewrite(
        original=text,
        rewritten=rewritten,
        terms=terms,
        applied=rewritten != text,
        cost=completion.cost,
    )
