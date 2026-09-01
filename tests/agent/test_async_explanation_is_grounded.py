"""Async-generated prose must be drift-checked before a diner ever sees it.

`verify_node` diffs every number an explanation states against catalog truth.
With `async_explanation` OFF that check covers the prose, because the prose
exists by the time VERIFY runs. With it ON, `state["explanation"]` is `""` at
VERIFY and the prose is generated afterwards, in the queue — where nothing
checked it at all.

That gap was dormant while async was opt-in, and became the DEFAULT path when
async_explanation was switched on for latency (e75c9a7). A latency change
silently disabled the assignment's "grounded in trusted data" guarantee on
the only surface a diner reads.
"""

from beatroot.agent.async_explain import ExplanationQueue
from beatroot.contracts.nutrition import NutritionFacts

N = NutritionFacts(
    kcal=520.0,
    protein_g=28.0,
    carbs_g=40.0,
    fat_g=22.0,
    sodium_mg=610.0,
    fibre_g=6.0,
    coverage=1.0,
)


class _StubLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.model = "stub"

    def complete(
        self, prompt: str, *, schema: object = None, stage: str = "", prompt_ref: object = None
    ):
        from beatroot.contracts.trust import Completion, CostRecord

        return Completion(text=self.text, parsed=None, cost=CostRecord())


def _run(text: str) -> ExplanationQueue:
    q = ExplanationQueue(_StubLLM(text))
    q.submit("t1", recipe=None, nutrition=N, check=None, prompt="ignored by the stub")
    q.drain() if hasattr(q, "drain") else None
    # block until the worker finishes
    import time

    for _ in range(200):
        if q.status("t1") != "pending":
            break
        time.sleep(0.01)
    return q


def test_grounded_prose_is_served() -> None:
    q = _run("This meal has about 520 kcal and 28g protein.")
    assert q.status("t1") == "ready"


def test_fabricated_number_is_not_served_to_the_diner() -> None:
    """The headline: prose stating a number the catalog contradicts must not
    reach the reader. Failing the entry is correct — the recommendation itself
    was fully determined and verified before this job was queued, so losing
    the prose costs nothing but the prose."""
    q = _run("This meal has roughly 9000 calories and 300g of protein.")
    assert q.status("t1") == "failed", "ungrounded explanation was served as ready"


def test_submit_after_shutdown_does_not_raise_into_the_request_path() -> None:
    """A SIGTERM / rolling restart shuts the executor down while requests are
    still in flight. Unguarded, `ThreadPoolExecutor.submit` raises "cannot
    schedule new futures after shutdown" into the caller — a 500 for a diner
    whose recommendation was already fully computed and verified.

    Surfaced as a logged RuntimeError during a real test teardown, which is
    harmless there and would not have been in production.
    """
    q = ExplanationQueue(_StubLLM("fine"))
    q.shutdown(wait=True)
    q.submit("t-shutdown", recipe=None, nutrition=N, check=None, prompt="x")
    assert q.status("t-shutdown") == "failed"
