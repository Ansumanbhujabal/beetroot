"""The real deliverable of the registry-driven vocabulary refactor.

Before this refactor, adding a constraint kind meant editing THREE places by
hand: `prompts/compile_constraints.md`'s static KIND SHAPES section,
`agent/nodes.py::_parse_compiled_constraint`'s if/elif chain, and
`contracts/core.py::ConstraintKind`. `exclude_cuisine` was added as a T0
evaluator and duly advertised via `registered_kinds()`, but the parser had no
branch for it — every one was silently dropped, and the symptom looked like
the model ignoring the user.

This test registers a brand-new kind with nothing touched outside a single
`@evaluator(...)` call, and proves it is immediately usable: it shows up in
the rendered model-facing prompt, AND its registered parser — the exact
`get_parser(kind)` lookup that replaced the old if/elif chain in
`agent.nodes._parse_compiled_constraint` — validates a raw model value for
it. No edit to the prompt file or the parser function was needed.

`_parse_compiled_constraint` itself is exercised up to that same lookup (it
resolves the id/severity/vocabulary-membership machinery the same as any
pre-existing kind); what it does NOT do for a genuinely novel kind is
construct a final `Constraint` — `ConstraintKind` stays a hand-maintained
pydantic `Literal` on purpose (see this refactor's brief: contracts/ cannot
import t0_invariants/ without a cycle), so a kind this test invents is
correctly rejected at that last, static boundary. `test_constraints.py::
test_every_declared_constraint_kind_has_an_evaluator` is what keeps that
Literal from drifting out of sync with the registry for every REAL kind.
"""

from beatroot.agent.nodes import _render_kind_shapes
from beatroot.reasoning.prompts import load_prompt
from beatroot.t0_invariants import constraints as t0_constraints


def test_a_new_kind_needs_only_one_registration_to_be_advertised_and_parsed():
    kind = "scratch_test_kind"
    assert kind not in t0_constraints.registered_kinds(), "test kind must not already exist"

    def _parse_scratch(item: dict, vocab: t0_constraints.ConstraintVocabulary):
        raw = item.get("value")
        if not isinstance(raw, str) or not raw.strip():
            return None
        return raw.strip(), None

    def _eval_scratch(recipe, c):  # pragma: no cover - never actually evaluated here
        return "satisfied"

    try:
        # This is the ENTIRE addition — one call, nothing else edited.
        t0_constraints.evaluator(
            kind,
            shape="a scratch value used only by this test, to prove the wiring.",
            parse=_parse_scratch,
        )(_eval_scratch)

        # 1. Advertised: registered_kinds() carries it immediately.
        assert kind in t0_constraints.registered_kinds()

        # 2. The rendered KIND SHAPES text (agent.nodes._render_kind_shapes,
        #    generated from t0_constraints.kind_shapes()) carries it too.
        rendered_shapes = _render_kind_shapes()
        assert kind in rendered_shapes
        assert "scratch value used only by this test" in rendered_shapes

        # 3. It reaches the actual prompt template the model is sent.
        prompt_text = load_prompt("compile_constraints").render(
            free_text="irrelevant",
            known_tags="",
            known_cuisines="",
            known_kinds=", ".join(t0_constraints.registered_kinds()),
            kind_shapes=rendered_shapes,
        )
        assert kind in prompt_text

        # 4. It PARSES — via the exact registry lookup
        #    `_parse_compiled_constraint` now uses instead of an if/elif
        #    chain. This is the precise mechanism whose absence caused the
        #    exclude_cuisine incident: a kind advertised through
        #    known_kinds/registered_kinds() with no branch to validate its
        #    value, silently dropped.
        parser = t0_constraints.get_parser(kind)
        assert parser is not None, "a registered kind must always have a parser"
        item = {"kind": kind, "value": "hello", "category": "preference"}
        vocabulary = t0_constraints.ConstraintVocabulary(
            known_tags=frozenset(), known_cuisines=frozenset()
        )
        parsed = parser(item, vocabulary)
        assert parsed == ("hello", None)
    finally:
        # Leave the registry exactly as it was found.
        t0_constraints._REGISTRY.pop(kind, None)
        t0_constraints._SHAPES.pop(kind, None)
        t0_constraints._PARSERS.pop(kind, None)

    assert kind not in t0_constraints.registered_kinds()
    assert kind not in t0_constraints.kind_shapes()
    assert t0_constraints.get_parser(kind) is None
