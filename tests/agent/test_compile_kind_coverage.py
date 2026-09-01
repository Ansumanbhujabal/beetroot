"""Every T0 evaluator kind must be reachable from free text.

`compile_node` tells the model it may emit any kind in `registered_kinds()`,
but `_parse_compiled_constraint` validates value shapes with a per-kind
branch chain whose `else` silently drops anything it has no branch for.
Those two lists are coupled and nothing checked they agreed.

That gap was not hypothetical: `exclude_cuisine` was added as an evaluator
and advertised to the model, which duly emitted it — and every one was
dropped on the floor. The symptom was a diner who said "no western food"
being served Aglio e olio, with `effective_constraints` showing nothing at
all, which reads like the model ignoring the request rather than the parser
discarding its answer.
"""

import pytest

from beatroot.agent.nodes import _parse_compiled_constraint
from beatroot.t0_invariants.constraints import registered_kinds

TAGS = {"vegan", "vegetarian", "fish"}
CUISINES = {"italian", "north_indian"}

# One well-formed example per kind. A kind added to the registry with no
# example here fails the coverage test below rather than passing silently.
EXAMPLES: dict[str, dict] = {
    "require_tag": {"value": "vegan"},
    "exclude_tag": {"value": "fish"},
    "require_any_tag": {"value": ["vegetarian", "fish"]},
    "exclude_ingredient": {"value": "mustard oil"},
    "cuisine_affinity": {"value": "italian"},
    "exclude_cuisine": {"value": ["italian"]},
    "nutrient_range": {"value": [0, 60], "nutrient": "carbs_g"},
    "max_prep_minutes": {"value": 30},
    "budget_max": {"value": 200},
}


def test_every_registered_kind_has_a_parser_example() -> None:
    missing = sorted(set(registered_kinds()) - set(EXAMPLES))
    assert not missing, (
        f"kinds registered in T0 but with no parser example here: {missing}. "
        "Add one, and confirm _parse_compiled_constraint has a branch for it."
    )


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_every_advertised_kind_survives_parsing(kind: str) -> None:
    """The model is told it may emit this kind; the parser must accept it."""
    item = {"kind": kind, "category": "preference", **EXAMPLES[kind]}
    parsed = _parse_compiled_constraint(0, item, set(registered_kinds()), TAGS, CUISINES)
    assert parsed is not None, (
        f"kind {kind!r} is advertised to the model by registered_kinds() but "
        "_parse_compiled_constraint drops it — the model's answer is discarded silently."
    )
    assert parsed.kind == kind
