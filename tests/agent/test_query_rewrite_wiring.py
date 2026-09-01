"""QUERY REWRITE task: `retrieve_node` (agent.nodes) actually wires
`retrieval.query_rewrite.rewrite_query` into the graph, and
`MealPlanningAgent.last_query_rewrite` reflects it end to end — never a
stale value from a previous run on the same (Container-shared) agent
instance.
"""

from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Recommendation


def test_commit_path_surfaces_a_real_query_rewrite(agent):
    cs = ConstraintSet(profile_id="p", constraints=[])
    result = agent.run(cs, query="something warm and comforting")
    assert isinstance(result, Recommendation)

    qr = agent.last_query_rewrite
    assert qr is not None
    assert qr["original"] == "something warm and comforting"
    assert qr["applied"] is True
    assert "hearty" in qr["terms"]
    assert qr["rewritten"] != qr["original"]
    assert "RETRIEVE" in agent.trace


def test_empty_query_never_applies_a_rewrite(agent):
    cs = ConstraintSet(profile_id="p", constraints=[])
    result = agent.run(cs, query="")
    assert isinstance(result, Recommendation)

    qr = agent.last_query_rewrite
    assert qr is not None
    assert qr["original"] == ""
    assert qr["applied"] is False


def test_infeasible_profile_never_reaches_retrieve_and_clears_stale_rewrite(agent):
    # First, a run that DOES reach RETRIEVE and sets last_query_rewrite.
    warm_cs = ConstraintSet(profile_id="p1", constraints=[])
    agent.run(warm_cs, query="something warm and comforting")
    assert agent.last_query_rewrite is not None

    # Then, on the SAME (shared) agent instance, an impossible profile that
    # never reaches RETRIEVE at all — FEASIBILITY routes straight to
    # NEGOTIATE. `last_query_rewrite` must not leak the previous run's
    # value forward.
    impossible_cs = ConstraintSet(
        profile_id="p2",
        constraints=[
            Constraint(id="c1", kind="max_prep_minutes", severity="preference", value=0),
        ],
    )
    result = agent.run(impossible_cs, query="dinner")
    assert result is not None and not isinstance(result, Recommendation)
    assert "RETRIEVE" not in agent.trace
    assert agent.last_query_rewrite is None


def test_free_text_constraints_are_visible_on_last_constraint_set(agent):
    """FREE-TEXT/NLP task: the effective constraint set the API surfaces
    (`last_constraint_set`) must include anything `compile_node` parsed out
    of free text, tagged `source == "parsed_free_text"` — never silently
    folded in as if the user had set it themselves."""
    cs = ConstraintSet(profile_id="p", constraints=[])
    agent.run(cs, query="dinner", preferences="no peanut please")

    effective = agent.last_constraint_set["constraints"]
    parsed = [c for c in effective if c["source"] == "parsed_free_text"]
    assert parsed, "the offline compile stub should extract a known tag from the free text"
    assert all(c["severity"] == "preference" for c in parsed)
