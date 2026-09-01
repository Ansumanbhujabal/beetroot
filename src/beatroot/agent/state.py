"""Typed LangGraph state. Spec §6.

Every field a node contributes is declared here, and `trace` uses an append
reducer (`operator.add`) so the path a run took through the graph is a
property of the framework's own state-merging, not of hand-rolled
bookkeeping a node author has to remember to update.

Every Pydantic model that crosses into this state (`ConstraintSet`,
`CheckResult`, `NutritionFacts`, `TrustReport`, `Negotiation`, `Escalation`,
`Recommendation`) is stored as a validated dict (`.model_dump()`), never the
live object and never `.__dict__`. `SqliteSaver`'s serializer *can* round-trip
a bare model instance, but only by falling back to an "unregistered type"
path it warns is headed for removal — a validated dump reconstructed with
`.model_validate(...)` at the point of use is what actually "serialises
cleanly," and it is forward-compatible with that removal.
"""

from operator import add
from typing import Annotated, Any, Literal, TypedDict

Terminal = Literal["NEGOTIATE", "ESCALATE", "COMMIT"]


def merge_cost(existing: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for `PlanState.cost`: SUMS every node's CostRecord
    contribution across the run, instead of the default TypedDict merge
    (last writer overwrites the running total) or operator.add (correct for
    trace's list-append, wrong here). Inputs are each a validated CostRecord
    dump, or empty/None before anything has spent anything — a node that
    never calls the model simply never contributes a cost key, and the
    channel is left untouched."""
    existing = existing or {}
    new = new or {}
    per_stage: dict[str, float] = dict(existing.get("per_stage") or {})
    for stage, usd in (new.get("per_stage") or {}).items():
        per_stage[stage] = per_stage.get(stage, 0.0) + usd
    return {
        "prompt_tokens": existing.get("prompt_tokens", 0) + new.get("prompt_tokens", 0),
        "completion_tokens": existing.get("completion_tokens", 0) + new.get("completion_tokens", 0),
        "usd": existing.get("usd", 0.0) + new.get("usd", 0.0),
        "tokens_saved": existing.get("tokens_saved", 0) + new.get("tokens_saved", 0),
        "per_stage": per_stage,
    }


class PlanState(TypedDict, total=False):
    """LangGraph state. `trace` uses an append reducer so the path through the
    graph is recorded by the framework rather than by bookkeeping. `cost`
    uses a summing reducer (`merge_cost`) for the same reason: cost-per-plan
    is a property of every node that spent tokens, not of whichever one ran
    last."""

    constraint_set: dict[str, Any]  # ConstraintSet.model_dump()
    query: str
    # Optional free text compiled into constraints — at whatever severity
    # actually enforces them (medical/religious/dietary land hard,
    # goal/preference stay soft) — by the `compile` node (agent.nodes,
    # GAP 1 + FREE-TEXT/NLP) before FEASIBILITY ever runs. `compile` can
    # also terminate the run straight to ESCALATE(reason="out_of_scope")
    # when this text has no meal-planning content at all. Distinct from
    # `query`, which only ever feeds retrieval ranking — this is the field
    # that actually reaches `compile_constraints`.
    preferences: str
    trace: Annotated[list[str], add]
    cost: Annotated[dict[str, Any], merge_cost]  # summed CostRecord.model_dump(mode="json")
    # Task 23: the same id `MealPlanningAgent.run()` generates for
    # `SqliteSaver` checkpointing, threaded into the state itself so
    # `explain_node` can key an async explanation job (`agent.
    # async_explain.ExplanationQueue`) on something the API already knows
    # how to hand back to a caller — no separate id invented for this.
    thread_id: str

    surviving_ids: list[str]
    # QUERY REWRITE task: `retrieval.query_rewrite.QueryRewrite.model_dump()`
    # — set by `retrieve_node` regardless of whether the rewrite step
    # actually changed anything, so a caller can always show the original
    # and rewritten query side by side. Absent when RETRIEVE never ran
    # (FEASIBILITY already routed to negotiate/escalate).
    query_rewrite: dict[str, Any]
    candidates: list[str]
    chosen_id: str
    nutrition: dict[str, Any]  # NutritionFacts.model_dump()
    check: dict[str, Any]  # CheckResult.model_dump()
    trust: dict[str, Any]  # TrustReport.model_dump()
    explanation: str
    self_assessment: float

    negotiation: dict[str, Any]  # Negotiation.model_dump()
    escalation: dict[str, Any]  # Escalation.model_dump()
    recommendation: dict[str, Any]  # Recommendation.model_dump()
    terminal: Terminal
    needs_approval: bool
    audit_id: str  # id returned by AuditLog.record() for this terminal — surfaced to
    # callers via MealPlanningAgent.last_audit_id so the API/CLI can hand it back
    # without threading an audit id through the domain result models themselves.
