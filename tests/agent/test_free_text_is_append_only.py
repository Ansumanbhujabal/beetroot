"""Free text may ADD constraints. It may never edit or remove one.

`ARCHITECTURE.md` used to justify this by saying compiled free text
could only ever produce PREFERENCE-severity constraints, so it was
"structurally incapable of lifting a MEDICAL exclusion". That reasoning stopped
being true at commit e75c9a7: the open-vocabulary compiler can now emit
MEDICAL, RELIGIOUS and DIETARY constraints, because encoding a stated allergy
as a soft preference was itself the bug that served chicken to a vegan.

The conclusion survives, for a different reason — and a right answer resting on
a wrong reason is one refactor from being wrong. The real invariant is the
merge in `compile_node`:

    merged = cs.model_copy(update={"constraints": [*cs.constraints, *added]})

Strictly append-only. Every constraint the caller supplied is carried through
verbatim, so no amount of free text can weaken one.
"""

import inspect

from beatroot.agent import nodes
from beatroot.contracts.core import Constraint, ConstraintSet, Severity


def test_merge_is_append_only_in_source() -> None:
    """Structural: the merge must splat the ORIGINAL constraints first and only
    concatenate. A future edit that filters or rewrites `cs.constraints` here
    would let parsed text weaken a medical exclusion, and nothing else in the
    suite would notice."""
    src = inspect.getsource(nodes)
    assert "[*cs.constraints, *added]" in src, (
        "compile_node's merge is no longer a plain append of the caller's "
        "constraints — free text may now be able to edit or drop one"
    )


def test_every_caller_supplied_constraint_survives_the_merge() -> None:
    """Behavioural, without a model: whatever the compiler proposes, the
    original set must still be present unchanged afterwards."""
    original = ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(
                id="med0", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"
            ),
            Constraint(
                id="rel0", kind="exclude_tag", severity=Severity.RELIGIOUS, value="beef"
            ),
        ],
    )
    added = [
        Constraint(
            id="parsed_0", kind="exclude_tag", severity=Severity.PREFERENCE, value="fish"
        )
    ]

    merged = original.model_copy(
        update={"constraints": [*original.constraints, *added]}
    )

    by_id = {c.id: c for c in merged.constraints}
    assert by_id["med0"].severity == Severity.MEDICAL
    assert by_id["med0"].value == "peanut"
    assert by_id["rel0"].severity == Severity.RELIGIOUS
    assert len(merged.constraints) == 3, "the merge dropped or replaced a constraint"
