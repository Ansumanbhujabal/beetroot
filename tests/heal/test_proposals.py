"""Task 16: the healing loop. Incidents become proposals.

The load-bearing test here is `test_propose_never_modifies_config_or_data`:
it is what actually guards the design position that rule proposals are
reviewable diffs, never auto-applied mutations.
"""

from __future__ import annotations

from datetime import UTC, datetime

from beatroot.confirm.trust_score import load_thresholds
from beatroot.container import ROOT, THRESHOLDS_PATH, build_container
from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.contracts.result import Incident
from beatroot.eval.runners.system import load_cases, run_system
from beatroot.heal.cluster import cluster
from beatroot.heal.proposals import propose


def _inc(kind: str, detail: str, payload: dict | None = None, i: int = 0) -> Incident:
    return Incident(
        id=f"i{i}",
        kind=kind,
        profile_id="p",
        detail=detail,
        payload=payload or {},
        created_at=datetime.now(UTC),
    )


def test_similar_incidents_cluster_together():
    incidents = [_inc("drift", "kcal stated 780 vs computed 520", i=i) for i in range(3)]
    clusters = cluster(incidents)
    assert len(clusters) == 1
    assert clusters[0].count == 3


def test_similar_incidents_with_different_numbers_still_cluster():
    incidents = [
        _inc("drift", "kcal stated 780 vs computed 520", i=0),
        _inc("drift", "kcal stated 640 vs computed 430", i=1),
    ]
    clusters = cluster(incidents)
    assert len(clusters) == 1
    assert clusters[0].count == 2


def test_different_kinds_do_not_cluster():
    assert len(cluster([_inc("drift", "a", i=0), _inc("escalation", "a", i=1)])) == 2


def test_different_detail_shapes_do_not_cluster():
    incidents = [
        _inc("drift", "kcal stated 780 vs computed 520", i=0),
        _inc("drift", "totally different failure", i=1),
    ]
    assert len(cluster(incidents)) == 2


def test_clusters_sorted_by_descending_count():
    incidents = [_inc("drift", "a 1", i=0)] + [_inc("escalation", "b", i=j) for j in range(1, 4)]
    clusters = cluster(incidents)
    assert clusters[0].count == 3
    assert clusters[0].kind == "escalation"


def test_eval_case_proposals_are_auto_applied(tmp_path):
    """Additive test cases can only tighten the suite, so they are safe."""
    clusters = cluster([_inc("escalation", "low trust", i=i) for i in range(3)])
    proposals = propose(clusters, tmp_path)
    eval_props = [p for p in proposals if p.kind == "eval_case"]
    assert eval_props
    assert all(p.auto_applied for p in eval_props)


def test_eval_case_proposal_emitted_even_for_a_singleton_cluster(tmp_path):
    """A single failure is still worth a permanent regression test."""
    proposals = propose(cluster([_inc("drift", "one off", i=0)]), tmp_path)
    eval_props = [p for p in proposals if p.kind == "eval_case"]
    assert len(eval_props) == 1
    assert eval_props[0].auto_applied
    assert eval_props[0].path.exists()


def test_rule_proposals_are_never_auto_applied(tmp_path):
    """An agent that silently rewrites its own allergen rules is a liability."""
    clusters = cluster([_inc("drift", "kcal drift", {"delta_pct": 0.5}, i=i) for i in range(5)])
    proposals = propose(clusters, tmp_path)
    rule_props = [p for p in proposals if p.kind in ("threshold", "meta_tag")]
    assert rule_props, "repeated drift should propose a threshold change"
    assert not any(p.auto_applied for p in rule_props)
    for p in rule_props:
        assert p.path.exists(), "proposal must be written to disk as a reviewable diff"


def test_meta_tag_proposal_for_repeated_escalations(tmp_path):
    clusters = cluster([_inc("escalation", "unrecognised ingredient foo", i=i) for i in range(4)])
    proposals = propose(clusters, tmp_path)
    rule_props = [p for p in proposals if p.kind == "meta_tag"]
    assert rule_props
    assert not any(p.auto_applied for p in rule_props)
    assert rule_props[0].path.suffix == ".md"


def test_singleton_incidents_do_not_generate_rule_changes(tmp_path):
    proposals = propose(cluster([_inc("drift", "one off", i=0)]), tmp_path)
    assert not [p for p in proposals if p.kind == "threshold"]
    assert not [p for p in proposals if p.kind == "meta_tag"]


def test_below_min_cluster_yields_no_rule_proposal_but_two_singletons_do_not_merge(tmp_path):
    """Two DIFFERENT singleton clusters must not be conflated into one
    'repeated' pattern just because propose() is called with both at once."""
    clusters = cluster([_inc("drift", "one off A", i=0), _inc("drift", "totally unrelated B", i=1)])
    assert len(clusters) == 2
    proposals = propose(clusters, tmp_path)
    assert not [p for p in proposals if p.kind == "threshold"]


def test_rule_proposal_body_is_self_explanatory(tmp_path):
    """A human reading the proposal weeks later, cold, needs: what was
    observed, how many times, and what the suggested change is."""
    clusters = cluster([_inc("drift", "kcal drift", i=i) for i in range(4)])
    proposals = propose(clusters, tmp_path)
    body = next(p.body for p in proposals if p.kind == "threshold")
    assert "4" in body  # how many times
    assert "kcal drift" in body  # what was observed
    assert "nutrition_drift_pct" in body  # the suggested change


def test_generated_eval_case_is_loadable_by_the_system_runner(tmp_path):
    """A generated case the runner cannot parse is worthless."""
    incidents = [_inc("unknown_ingredient", "cannot verify constraint xyz", i=i) for i in range(2)]
    clusters = cluster(incidents)
    proposals = propose(clusters, tmp_path)
    case_path = next(p.path for p in proposals if p.kind == "eval_case")

    loaded = load_cases(case_path)
    assert len(loaded) == 1
    case = loaded[0]
    assert case["id"]
    assert case["family"] == "regression"
    assert isinstance(case["constraints"], list)
    assert isinstance(case["expect_terminal"], list) and case["expect_terminal"]

    # "regression" must actually be a mapped axis, or run_system raises.
    from beatroot.settings import get_settings

    assert "regression" in get_settings().eval.axis_by_family


def test_propose_creates_proposals_and_generated_dirs(tmp_path):
    propose(cluster([_inc("drift", "x", i=0)]), tmp_path)
    assert (tmp_path / "proposals").is_dir()
    assert (tmp_path / "generated").is_dir()


def test_propose_is_idempotent_across_runs(tmp_path):
    """Re-running heal over an unchanged incident log must not pile up
    duplicate files run after run — filenames are derived deterministically
    from the cluster signature, not from a per-process salt."""
    incidents = [_inc("drift", "kcal drift", i=i) for i in range(4)]
    first = propose(cluster(incidents), tmp_path)
    second = propose(cluster(incidents), tmp_path)
    assert {p.path for p in first} == {p.path for p in second}


def test_propose_never_modifies_config_or_data(tmp_path):
    """The test that guards the design position: propose() must never
    touch a live config or data file, only out_dir."""
    config_path = ROOT / "config" / "beatroot.yaml"
    thresholds_path = ROOT / "eval" / "thresholds.yaml"
    before_config = config_path.read_bytes()
    before_thresholds = thresholds_path.read_bytes()
    before_config_mtime = config_path.stat().st_mtime_ns
    before_thresholds_mtime = thresholds_path.stat().st_mtime_ns

    clusters = cluster(
        [_inc("drift", "kcal drift", i=i) for i in range(5)]
        + [_inc("escalation", "unknown thing", i=i) for i in range(5, 10)]
    )
    propose(clusters, tmp_path)

    assert config_path.read_bytes() == before_config
    assert thresholds_path.read_bytes() == before_thresholds
    assert config_path.stat().st_mtime_ns == before_config_mtime
    assert thresholds_path.stat().st_mtime_ns == before_thresholds_mtime


# ---------------------------------------------------------------------------
# Fix round: a generated eval case must be able to fail on BEHAVIOUR, not
# only on a parse error / crash. See heal/proposals.py's module docstring.
# ---------------------------------------------------------------------------


def _cs_payload(constraints):
    return ConstraintSet(profile_id="p", constraints=constraints).model_dump(mode="json")


def test_eval_case_replays_the_real_constraint_set_when_recorded(tmp_path):
    """When the triggering incident carries a constraint_set + terminal
    (every current agent.nodes emission site records both), the generated
    case must replay the REAL constraints and assert the REAL terminal —
    not an empty ConstraintSet and 'any terminal will do'."""
    constraints = [
        Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")
    ]
    inc = _inc(
        "unknown_ingredient",
        "med1 (medical) names an unrecognised value: 'peanut'",
        {"constraint_set": _cs_payload(constraints), "terminal": "ESCALATE"},
        i=0,
    )
    proposals = propose(cluster([inc]), tmp_path)
    case_path = next(p.path for p in proposals if p.kind == "eval_case")
    case = load_cases(case_path)[0]

    assert case["verified"] is True
    assert case["expect_terminal"] == ["ESCALATE"]
    assert case["constraints"], "must replay the real constraints, not []"
    assert case["constraints"][0]["value"] == "peanut"
    # Must be reconstructible by run_system's own Constraint(**c) call.
    assert Constraint(**case["constraints"][0]).value == "peanut"


def test_eval_case_infeasible_incident_asserts_negotiate(tmp_path):
    """kind-based fallback terminal: infeasible -> NEGOTIATE, when the
    incident didn't separately record its own terminal."""
    constraints = [
        Constraint(id="g1", kind="exclude_tag", severity=Severity.MEDICAL, value="dairy")
    ]
    inc = _inc(
        "infeasible",
        "no meal satisfies the profile",
        {"constraint_set": _cs_payload(constraints)},  # no explicit "terminal"
        i=0,
    )
    case = load_cases(
        next(p.path for p in propose(cluster([inc]), tmp_path) if p.kind == "eval_case")
    )[0]
    assert case["verified"] is True
    assert case["expect_terminal"] == ["NEGOTIATE"]


def test_eval_case_without_constraint_set_is_honestly_marked_unverified(tmp_path):
    """An incident recorded before this fix (or from a site that never
    learns to carry constraint_set) must not masquerade as a strong case."""
    case = load_cases(
        next(
            p.path
            for p in propose(cluster([_inc("drift", "one off", i=0)]), tmp_path)
            if p.kind == "eval_case"
        )
    )[0]
    assert case["verified"] is False
    assert case["constraints"] == []
    assert set(case["expect_terminal"]) == {"COMMIT", "NEGOTIATE", "ESCALATE"}
    assert "LOADER SMOKE TEST" in case["note"]


def test_generated_case_is_a_real_regression_test_for_unknown_ingredient(tmp_path, monkeypatch):
    """The claim the healing loop exists to support: a generated case must
    actually be able to catch a regression, not just parse.

    Drives a real `unknown_ingredient` incident through the real agent,
    generates its case, confirms the case currently PASSES, then
    monkeypatches away the exact safety check that produced the incident
    (simulating a revert, without touching any file on disk — other agents
    share this tree) and confirms the SAME case now FAILS. Reverting the
    monkeypatch restores a PASS.
    """
    import beatroot.agent.nodes as nodes_module

    container = build_container(tmp_path / "heal_proof.db", async_explanation=False)
    try:
        cs = ConstraintSet(
            profile_id="proof",
            constraints=[
                Constraint(id="med1", kind="exclude_tag", severity=Severity.MEDICAL, value="kiwi")
            ],
        )
        container.agent.run(cs, query="anything, but exclude kiwi")

        incidents = [i for i in container.incidents.all() if i.kind == "unknown_ingredient"]
        assert incidents, "expected a real unknown_ingredient incident to have been recorded"

        proposals = propose(cluster(incidents), tmp_path / "healing")
        case_path = next(p.path for p in proposals if p.kind == "eval_case")
        cases = load_cases(case_path)
        assert cases[0]["verified"] is True
        assert cases[0]["expect_terminal"] == ["ESCALATE"]

        thresholds = load_thresholds(THRESHOLDS_PATH)

        report_before = run_system(container.agent, cases, thresholds)
        assert report_before.passed, report_before.failures

        # Simulate reverting the unknown-vocabulary safety check.
        monkeypatch.setattr(nodes_module, "unknown_vocabulary", lambda cs, catalog: [])
        report_broken = run_system(container.agent, cases, thresholds)
        assert not report_broken.passed
        assert report_broken.failures
        assert report_broken.failures[0]["case"] == cases[0]["id"]

        monkeypatch.undo()
        report_after = run_system(container.agent, cases, thresholds)
        assert report_after.passed, report_after.failures
    finally:
        container.close()
