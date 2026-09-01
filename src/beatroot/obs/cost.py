"""Cost accounting across plans. Spec §13, §17.

`beatroot.contracts.trust.CostRecord` already carries per-stage cost for a
SINGLE plan (accumulated across a graph run by `agent.state.merge_cost`).
`CostLedger` is the layer above that: aggregation ACROSS plans, so
cost-per-plan is a headline metric that is computed, not eyeballed from a
pile of individual `CostRecord`s.

`tokens_saved` is the argument for filter-before-generate made measurable:
an infeasible profile short-circuits before any model call, and a
feasibility-cache hit skips even the lattice walk. Every token that a call
never had to spend is recorded here as evidence that the short-circuit paid
for itself.
"""

import threading

from pydantic import BaseModel, Field, PrivateAttr

from beatroot.contracts.trust import CostRecord

# ~4 characters per token is the standard rule-of-thumb for English text
# under BPE-style tokenizers (OpenAI's own cookbook guidance: "a helpful
# rule of thumb is that one token generally corresponds to ~4 characters
# of text"). No tokenizer is actually invoked here — there is no
# completion to count usage off, because the whole point is that the call
# never happened — so this is a STATED, defensible approximation, never an
# invented number. Every caller that uses `estimate_tokens` must surface
# the result as an estimate (see `api.main`'s `/metrics` response), never
# as a measured spend.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Character-count-based token estimate for a prompt that was never
    actually sent to a model — see `CHARS_PER_TOKEN` for the method. Empty
    text costs nothing; any non-empty text rounds to at least 1 token
    rather than truncating to 0, which would silently under-count a short
    but real prompt."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


class CostLedger(BaseModel):
    """Running total of spend per stage, and of tokens never spent.

    Mutated from more than one thread: request handlers fold in a terminal
    result's cost, and `agent.async_explain.ExplanationQueue`'s worker folds
    in the explanation's cost whenever that job finishes — which is after
    the response has already been sent. `self.per_stage[k] = ... + usd` is a
    read-modify-write, so the lock is not decoration.
    """

    per_stage: dict[str, float] = Field(default_factory=dict)
    tokens: int = 0
    tokens_saved: int = 0
    plans: int = 0

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add(self, stage: str, usd: float, tokens: int = 0) -> None:
        """Record spend for one stage of one plan."""
        with self._lock:
            self.per_stage[stage] = round(self.per_stage.get(stage, 0.0) + usd, 8)
            self.tokens += tokens

    def record_short_circuit(self, estimated_tokens: int) -> None:
        """Record tokens that a short-circuit (infeasibility, cache hit)
        avoided spending entirely."""
        with self._lock:
            self.tokens_saved += estimated_tokens

    def record_plan(self) -> None:
        """Mark one plan as complete, so `per_plan_usd` has a denominator."""
        with self._lock:
            self.plans += 1

    def fold(self, cost: "CostRecord") -> None:
        """Add one `CostRecord`'s spend to this ledger, WITHOUT counting a
        new plan.

        Exists for cost that arrives after its plan already finished — the
        async explanation is generated once the response is out the door, so
        its spend has no terminal-route handler left to account for it. Until
        this, `per_plan_usd` silently under-reported every COMMIT by the
        whole explanation call: real money spent, on the request path's own
        behalf, that no metric ever saw.
        """
        for stage, usd in cost.per_stage.items():
            self.add(stage, usd)
        total_tokens = cost.prompt_tokens + cost.completion_tokens
        if total_tokens:
            with self._lock:
                self.tokens += total_tokens

    @property
    def total_usd(self) -> float:
        return round(sum(self.per_stage.values()), 8)

    @property
    def per_plan_usd(self) -> float:
        """Cost-per-plan — the headline metric. Zero, not a division error,
        before any plan has been recorded."""
        return round(self.total_usd / self.plans, 8) if self.plans else 0.0
