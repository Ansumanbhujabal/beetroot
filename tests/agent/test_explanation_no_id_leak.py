"""The explanation a diner reads must never contain internal constraint ids.

Regression. A user on the async path was shown: "aligning with constraints
c0, c1, and c2". `_describe_satisfied` exists precisely to stop that, and the
SYNC path used it — but `ExplanationQueue._render_prompt` had a second
renderer that joined the raw `check.satisfied` ids, and its docstring claimed
it produced "the exact prompt explain_node renders synchronously". It did not.

The leak was dormant until `async_explanation` was switched on for latency,
which routed every request onto the unfixed renderer. Two renderers that must
agree eventually will not, so `explain_node` now renders once and hands the
finished prompt to the queue.
"""

from beatroot.agent.nodes import _describe_satisfied, _satisfied_display

CS = {
    "profile_id": "p",
    "constraints": [
        {"id": "c0", "kind": "exclude_tag", "severity": "preference",
         "value": "peanut", "source": "structured"},
        {"id": "c1", "kind": "max_prep_minutes", "severity": "preference",
         "value": 30, "source": "structured"},
        {"id": "c2", "kind": "exclude_cuisine", "severity": "preference",
         "value": ["italian"], "source": "structured"},
    ],
}
IDS = ["c0", "c1", "c2"]


def test_description_never_contains_a_constraint_id() -> None:
    text = _describe_satisfied(CS, IDS)
    for cid in IDS:
        assert cid not in text, f"internal id {cid!r} leaked into user-facing prose: {text!r}"
    assert "peanut" in text and "30" in text


def test_every_satisfied_constraint_is_described_not_silently_dropped() -> None:
    """`_describe_satisfied` skipped kinds it had no branch for — exclude_cuisine
    and cuisine_affinity among them — so a constraint could be enforced,
    reported as satisfied, and then vanish from the explanation entirely."""
    assert len(_satisfied_display(CS, IDS)) == len(IDS)


def test_async_and_sync_paths_render_the_same_prompt() -> None:
    """The structural fix: `explain_node` builds the prompt and passes it to
    the queue, so there is one renderer rather than two that must agree."""
    import inspect

    from beatroot.agent import nodes

    src = inspect.getsource(nodes)
    submit = src[src.index("explanation_queue.submit(") :][:600]
    assert "prompt=" in submit, "async submit must hand over a rendered prompt"
    assert "_describe_satisfied" in submit, "and it must be rendered with the descriptions"
