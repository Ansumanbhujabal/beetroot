"""The meal-planning agent as a LangGraph `StateGraph`. Spec §6.

Three terminal nodes — `negotiate`, `escalate`, `commit` — and two of them
decline to produce a meal. That posture is a property of the graph's shape
(`test_graph_declares_exactly_three_terminal_nodes`), not something buried in
a conditional a reader has to trace by hand.

What this buys over a hand-rolled state machine:

- **`SqliteSaver` checkpointing** — every node's output is persisted keyed by
  `thread_id`; a run interrupted mid-flight (process killed, request timed
  out) is recoverable from its last completed node, not lost.
- **`interrupt_before=["commit"]`** — the fourth boundary zone, "actions that
  should require confirmation," is a real pause in a real graph: the graph
  literally stops with durable state and waits. `MealPlanningAgent` decides,
  per run, whether that pause is actually shown to anyone
  (`needs_approval`) — grey-band trust on a MEDICAL profile stops; every
  other profile auto-resumes past the same interrupt point in the same call.
- **Typed state with an append reducer for `trace`** — the path through the
  graph is the framework's own bookkeeping (`operator.add` over `trace`), so
  `test_trace_is_produced_by_the_graph_not_by_bookkeeping` is checking a
  structural fact, not a convention nodes have to honour by hand. `cost` gets
  the same treatment with a summing reducer (`agent.state.merge_cost`), so
  cost-per-plan is accumulated by the graph, not reconstructed after the
  fact from whichever node happened to run last.

Every node (`agent.nodes.make_nodes`) is wrapped so an unhandled exception
routes to ESCALATE instead of propagating out of `graph.invoke()` and
stranding a thread's checkpoint mid-graph with no terminal and no incident. A
thread PAUSED at `interrupt_before` and never resumed still gets an audit
record (`terminal_state="AWAITING_APPROVAL"`) at the moment it pauses — real
tokens (retrieve/score/explain) may already be spent by then, and an
abandoned thread must not be the one path spend goes unaccounted for.
"""

import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Literal, cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from beatroot.agent.nodes import Deps, make_nodes, skill_versions
from beatroot.agent.state import PlanState
from beatroot.contracts.core import ConstraintSet
from beatroot.contracts.result import Escalation, Negotiation, Recommendation
from beatroot.contracts.trust import CostRecord, TrustReport


def _after_compile(state: PlanState) -> str:
    # `compile` (agent.nodes, GAP 1) never intentionally sets `terminal` on
    # its own happy path — but it is `_guarded` like every other
    # non-terminal node, so an unhandled exception here DOES set
    # `terminal == "ESCALATE"`. Without this conditional edge that would
    # silently fall through to `feasibility`, which reads `state[
    # "constraint_set"]` unconditionally and would work anyway today, but
    # would mask a real internal failure as if COMPILE had simply produced
    # no parsed constraints. Same posture as every other guarded stage.
    return "escalate" if state.get("terminal") == "ESCALATE" else "feasibility"


def _after_feasibility(state: PlanState) -> str:
    terminal = state.get("terminal")
    if terminal == "ESCALATE":
        # The vocabulary check (agent.nodes.feasibility, t0_invariants.
        # vocabulary.unknown_vocabulary) can escalate straight from this
        # first node — zero tokens spent, same posture as the NEGOTIATE
        # branch below.
        return "escalate"
    return "negotiate" if terminal == "NEGOTIATE" else "retrieve"


def _after_retrieve(state: PlanState) -> str:
    return "escalate" if state.get("terminal") == "ESCALATE" else "score"


def _after_score(state: PlanState) -> str:
    return "escalate" if state.get("terminal") == "ESCALATE" else "trust"


def _after_trust(state: PlanState) -> str:
    return "escalate" if state.get("terminal") == "ESCALATE" else "explain"


def _after_explain(state: PlanState) -> str:
    # `explain_node` itself never sets `terminal` on the happy path — but
    # every node is exception-guarded (agent.nodes._guarded), and a guarded
    # exception here DOES set `terminal == "ESCALATE"`. Without this
    # conditional edge that escalation would silently fall through to
    # `verify`, which would then crash on the `explanation` key `explain`
    # never got to write — replacing one uncaught failure with another.
    return "escalate" if state.get("terminal") == "ESCALATE" else "verify"


def _after_verify(state: PlanState) -> str:
    return "escalate" if state.get("terminal") == "ESCALATE" else "commit"


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver[str] | None = None
) -> CompiledStateGraph[PlanState, None, PlanState, PlanState]:
    """Compile the agent graph. `checkpointer` is optional here (structural
    tests only inspect graph shape) but `MealPlanningAgent` always supplies
    one — durability is not opt-in for anything that actually runs."""
    nodes = make_nodes(deps)
    g = StateGraph(PlanState)
    for name, fn in nodes.items():
        # mypy cannot resolve add_node's NodeInputT-bound overload set
        # through a dict[str, Callable[...]] value (confirmed in isolation:
        # the identical call with a literal function reference type-checks
        # cleanly) — a stub/inference gap, not a real type mismatch; the
        # 31 tests in tests/agent/ exercise this exact call at runtime.
        g.add_node(name, fn)  # type: ignore[call-overload]

    g.add_edge(START, "compile")
    g.add_conditional_edges(
        "compile", _after_compile, {"escalate": "escalate", "feasibility": "feasibility"}
    )
    g.add_conditional_edges(
        "feasibility",
        _after_feasibility,
        {"negotiate": "negotiate", "retrieve": "retrieve", "escalate": "escalate"},
    )
    g.add_conditional_edges("retrieve", _after_retrieve, {"escalate": "escalate", "score": "score"})
    g.add_conditional_edges("score", _after_score, {"escalate": "escalate", "trust": "trust"})
    g.add_conditional_edges("trust", _after_trust, {"escalate": "escalate", "explain": "explain"})
    g.add_conditional_edges("explain", _after_explain, {"escalate": "escalate", "verify": "verify"})
    g.add_conditional_edges("verify", _after_verify, {"escalate": "escalate", "commit": "commit"})
    for terminal in ("negotiate", "escalate", "commit"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["commit"])


def _default_checkpointer() -> SqliteSaver:
    """An in-process, in-memory durable store by default — every run still
    goes through real `SqliteSaver` checkpointing (not a hand-rolled dict),
    it just isn't pinned to a file unless the caller asks for that."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    return SqliteSaver(conn)


_Graph = CompiledStateGraph[PlanState, None, PlanState, PlanState]


def _invoke(graph: _Graph, payload: PlanState | None, cfg: RunnableConfig) -> PlanState:
    """`Pregel.invoke`'s default ("v1") overload returns `dict[str, Any] |
    Any`, not `PlanState` — the precisely-typed `GraphOutput[PlanState]`
    return only comes from passing `version="v2"`, a different streaming/
    output contract this project has not adopted (and switching to it is a
    behavioural change, not a typing fix). The runtime value is always a
    `PlanState`-shaped dict either way; this cast documents that once,
    instead of every call site losing precision silently.
    """
    result = graph.invoke(payload, config=cfg)
    return cast(PlanState, result)


@dataclass
class MealPlanningAgent:
    """Facade over `build_graph`. `.run()` drives a fresh thread to either a
    terminal result or a genuine pause; `.resume()` is the only way a paused
    thread ever continues. Spec §6.

    The graph is (re)built from `self.deps` at the top of every call — "closed
    over at graph build time" (Spec brief) means build time, not process
    start: swapping `.deps` (a test double, a hot-reloaded catalog) takes
    effect on the next call with no separate re-wiring step.
    """

    deps: Deps
    checkpointer: BaseCheckpointSaver[str] | None = None
    trace: list[str] = field(default_factory=list)
    graph: _Graph | None = field(init=False, default=None, repr=False)
    last_thread_id: str | None = field(init=False, default=None, repr=False)
    # The AuditLog id for whatever terminal (or AWAITING_APPROVAL pause) the
    # most recent .run()/.resume() call produced — set alongside `trace` so
    # a caller (the API's response envelope, the CLI) can hand it back
    # without contracts/result.py's domain models carrying an audit id.
    last_audit_id: str | None = field(init=False, default=None, repr=False)
    # QUERY REWRITE task: `retrieval.query_rewrite.QueryRewrite.model_dump()`
    # off the most recent run — `None` when RETRIEVE never ran (an
    # infeasible or vocabulary-escalated profile never reaches it). Set
    # from the final graph state, never recomputed here, so it always
    # reflects exactly what `retrieve_node` actually did.
    last_query_rewrite: dict[str, object] | None = field(init=False, default=None, repr=False)
    # FREE-TEXT/NLP task: the FULL, already-merged `ConstraintSet.
    # model_dump()` off the most recent run — structured constraints as
    # submitted, plus whatever `compile_node` (GAP 1) appended from free
    # text (`source == "parsed_free_text"`). Never recomputed here; read
    # straight from final graph state so it can never drift from what
    # FEASIBILITY/RETRIEVE/VERIFY actually enforced.
    last_constraint_set: dict[str, object] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.checkpointer is None:
            self.checkpointer = _default_checkpointer()
        self.graph = build_graph(self.deps, self.checkpointer)

    @staticmethod
    def _cfg(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def run(
        self,
        cs: ConstraintSet,
        query: str = "",
        preferences: str = "",
        thread_id: str | None = None,
    ) -> Negotiation | Escalation | Recommendation | None:
        """Drive one run to a terminal, or to a pause awaiting `.resume()`.

        `preferences` is optional free text compiled into constraints — at
        whatever severity actually enforces them — by the `compile` node
        before FEASIBILITY runs (GAP 1 + FREE-TEXT/NLP), and can itself
        terminate the run to ESCALATE(reason="out_of_scope") when the text
        has no meal-planning content at all. Distinct from `query`, which
        only ever feeds retrieval ranking. Leaving it empty costs zero
        tokens: `compile` (scope check included) degrades to a no-op.

        Returns `None` only when the run stopped at `interrupt_before` and
        `needs_approval` is true — durable state sits at the checkpoint under
        `thread_id`; nothing is lost, and `.resume(thread_id, approved)` is
        how it continues. Every other outcome — including an auto-resumed
        pause — returns the typed terminal result directly.
        """
        graph = build_graph(self.deps, self.checkpointer)
        self.graph = graph
        thread_id = thread_id or str(uuid.uuid4())
        self.last_thread_id = thread_id
        # QUERY REWRITE task: reset before every run — `self` (via
        # Container) is a shared, long-lived instance across requests, so
        # a profile that never reaches RETRIEVE this time (infeasible,
        # vocabulary-escalated) must not surface a PREVIOUS run's
        # query_rewrite instead of `None`.
        self.last_query_rewrite = None
        self.last_constraint_set = None
        cfg = self._cfg(thread_id)
        payload: PlanState = {
            "constraint_set": cs.model_dump(mode="json"),
            "query": query,
            "preferences": preferences,
            "trace": [],
            "cost": {},
            # Task 23: the SAME id used for checkpointing, threaded into
            # PlanState so `explain_node` can key an async explanation job
            # on it — see `agent.state.PlanState.thread_id`.
            "thread_id": thread_id,
        }
        state = _invoke(graph, payload, cfg)
        self.trace = list(state.get("trace", []))
        return self._settle(graph, state, cfg)

    def _settle(
        self, graph: _Graph, state: PlanState, cfg: RunnableConfig
    ) -> Negotiation | Escalation | Recommendation | None:
        # QUERY REWRITE task: reflects the LATEST state seen, which on the
        # auto-resume recursive call below is the post-commit state —
        # `query_rewrite` itself was written earlier (RETRIEVE) and simply
        # rides along unchanged, since no later node overwrites that key.
        if "query_rewrite" in state:
            self.last_query_rewrite = state["query_rewrite"]
        # FREE-TEXT/NLP task: `constraint_set` is written into the payload
        # before the FIRST node ever runs (see `.run()` below) and never
        # removed, so this is always present for every real terminal.
        if "constraint_set" in state:
            self.last_constraint_set = state["constraint_set"]
        terminal = state.get("terminal")
        if terminal in ("COMMIT", "NEGOTIATE", "ESCALATE"):
            self.last_audit_id = state.get("audit_id")
        if terminal == "COMMIT":
            return Recommendation.model_validate(state["recommendation"])
        if terminal == "NEGOTIATE":
            return Negotiation.model_validate(state["negotiation"])
        if terminal == "ESCALATE":
            return Escalation.model_validate(state["escalation"])

        pending = graph.get_state(cfg)
        if pending.next == ("commit",):
            if state.get("needs_approval"):
                # The fourth boundary zone, for real: durable, paused, and
                # waiting on a person. Spec §6. RETRIEVE/SCORE/EXPLAIN have
                # already run by this point — real tokens may already be
                # spent — so the pause gets its own audit record now,
                # rather than accounting for that spend only if and when
                # someone eventually calls .resume(). A thread abandoned
                # here forever is still accounted for.
                cs = ConstraintSet.model_validate(state["constraint_set"])
                cost = CostRecord.model_validate(state.get("cost") or {})
                self.last_audit_id = self.deps.audit.record(
                    cs.profile_id,
                    "AWAITING_APPROVAL",
                    {"chosen_id": state.get("chosen_id"), "trust": state.get("trust")},
                    skill_versions(self.deps),
                    cost.usd,
                )
                return None
            # Auto-resume: this pause exists for the graph's own durability
            # story, not because this particular run needs a human.
            state = _invoke(graph, None, cfg)
            self.trace = list(state.get("trace", []))
            return self._settle(graph, state, cfg)
        return None

    def thread_state(self, thread_id: str) -> Literal["unknown", "paused", "resolved"]:
        """Where `thread_id` stands, without mutating anything.

        `"unknown"`: no checkpoint exists for this thread at all.
        `"paused"`: genuinely stopped at `interrupt_before`, waiting on
        `.resume()`. `"resolved"`: the thread exists but is not paused —
        either it never needed approval or `.resume()` already ran on it.
        The API's resume route and the CLI's resume command both call this
        first, so an unknown or already-settled thread never reaches
        `.resume()` and never has a chance to produce a 500.
        """
        graph = build_graph(self.deps, self.checkpointer)
        pending = graph.get_state(self._cfg(thread_id))
        if not pending.values:
            return "unknown"
        if pending.next == ("commit",):
            return "paused"
        return "resolved"

    def resume(
        self, thread_id: str, approved: bool
    ) -> Negotiation | Escalation | Recommendation | None:
        """Continue a thread paused at `interrupt_before=["commit"]`.

        `approved=True` lets the pending `commit` run as planned.
        `approved=False` reroutes to `escalate` — `update_state(...,
        as_node="verify")` makes the checkpoint look like `verify` (the real
        predecessor of the pending `commit`) just produced an ESCALATE
        terminal, so the graph's own conditional edge (not a hand-rolled
        branch here) sends it to `escalate`, and the real `escalate` node
        writes the incident/audit records exactly as it would for any other
        escalation.
        """
        graph = build_graph(self.deps, self.checkpointer)
        self.graph = graph
        cfg = self._cfg(thread_id)
        if approved:
            state = _invoke(graph, None, cfg)
        else:
            pending = graph.get_state(cfg)
            trust = (
                TrustReport.model_validate(pending.values["trust"])
                if pending.values.get("trust")
                else None
            )
            spent = CostRecord.model_validate(pending.values.get("cost") or {})
            esc = Escalation(
                reason="low_trust",
                failing_signal="human_review_declined",
                trust=trust,
                cost=spent,
                detail="A human reviewer did not approve this recommendation.",
            )
            values = {"terminal": "ESCALATE", "escalation": esc.model_dump(mode="json")}
            graph.update_state(cfg, values, as_node="verify")
            state = _invoke(graph, None, cfg)
        self.trace = list(state.get("trace", []))
        return self._settle(graph, state, cfg)
