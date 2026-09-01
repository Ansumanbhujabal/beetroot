"""API tests. Spec §15.

Node names in `trace` come from `agent.nodes.make_nodes` (Task 11) —
`FEASIBILITY`, not the brief's stale `INTAKE` — so assertions below match
the real graph.
"""

from typing import ClassVar

from beatroot.settings import get_settings


def test_health_reports_the_active_provider(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "echo"  # offline mode in this fixture
    assert body["vector_store"] in {"numpy", "qdrant"}
    assert body["recipes"] > 0
    assert body["skills"] > 0
    assert body["skills_locked"] is True


def test_health_exposes_the_live_trust_config_not_a_frontend_literal(client):
    """The dashboard (Task 18) reads its refusal-threshold line and 75/25
    weight labels from here — `beatroot.settings`, never a number baked
    into `web/index.html`. This is the one place that has to agree with
    `get_settings().trust` for that promise to hold."""
    r = client.get("/health")
    assert r.status_code == 200
    trust = r.json()["trust"]
    live = get_settings().trust
    assert trust["refusal_threshold"] == live.refusal_threshold
    assert trust["weights"] == live.weights.model_dump()
    assert abs(sum(trust["weights"].values()) - 1.0) < 1e-9


def test_recommend_returns_a_terminal_state_and_trust_breakdown(client):
    r = client.post(
        "/recommend",
        json={
            "profile_id": "p1",
            "constraints": [
                {"id": "c1", "kind": "exclude_tag", "severity": "medical", "value": "peanut"}
            ],
            "query": "something with rice",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["terminal_state"] in {"COMMIT", "NEGOTIATE", "ESCALATE"}
    if body["terminal_state"] == "COMMIT":
        assert body["result"]["nutrition"]["provenance"] == "computed"
        assert set(body["result"]["trust"]) >= {
            "composite",
            "catalog_coverage",
            "constraint_completeness",
            "model_self_assessment",
        }
    elif body["terminal_state"] == "ESCALATE":
        assert body["result"]["failing_signal"]


def test_impossible_profile_returns_a_relaxation_ladder(client):
    r = client.post(
        "/recommend",
        json={
            "profile_id": "p2",
            "constraints": [
                {"id": "c1", "kind": "max_prep_minutes", "severity": "preference", "value": 0}
            ],
            "query": "anything",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["terminal_state"] == "NEGOTIATE"
    assert "relaxations" in body["result"]
    assert body["result"]["locked"] == []  # preference severity, not hard


def test_trace_is_returned_for_inspection(client):
    r = client.post(
        "/recommend",
        json={"profile_id": "p3", "constraints": [], "query": "rice"},
    )
    assert r.status_code == 200
    assert r.json()["trace"][0] == "FEASIBILITY"


def test_recommend_rejects_oversized_profile_id(client):
    r = client.post(
        "/recommend",
        json={"profile_id": "x" * 200, "constraints": [], "query": "rice"},
    )
    assert r.status_code == 422


def test_recommend_rejects_too_many_constraints(client):
    constraints = [
        {"id": f"c{i}", "kind": "exclude_tag", "severity": "preference", "value": "x"}
        for i in range(51)
    ]
    r = client.post(
        "/recommend",
        json={"profile_id": "p4", "constraints": constraints, "query": "rice"},
    )
    assert r.status_code == 422


def test_recommend_rejects_oversized_query(client):
    r = client.post(
        "/recommend",
        json={"profile_id": "p5", "constraints": [], "query": "x" * 5000},
    )
    assert r.status_code == 422


def test_metrics_reports_cache_hit_rates_and_incident_count(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    for cache in ("feasibility_cache", "embedding_cache"):
        assert set(body[cache]) == {"hits", "misses", "hit_rate"}
    assert isinstance(body["incidents"], int)


def test_metrics_reports_cost_ledger_shape(client):
    """Task 19: cost-per-plan is the headline observability metric — it
    must actually be reachable from the running app, not just computable
    from a `CostLedger` object nobody surfaces."""
    r = client.get("/metrics")
    assert r.status_code == 200
    cost = r.json()["cost"]
    assert set(cost) == {
        "per_stage_usd",
        "total_usd",
        "plans",
        "per_plan_usd",
        "tokens",
        "tokens_saved",
        "tokens_saved_estimate_method",
    }
    assert isinstance(cost["per_stage_usd"], dict)


def test_metrics_reports_nonzero_tokens_saved_after_an_infeasible_request(client):
    """GAP 2: an infeasible profile short-circuits straight to NEGOTIATE —
    RETRIEVE/SCORE/EXPLAIN never run — and `feasibility` (agent.nodes)
    records a defensible estimate of what those stages would have cost via
    `_estimate_skipped_tokens`. `/metrics` must surface that as a real,
    non-zero, accumulating number, not a permanent 0."""
    r = client.post(
        "/recommend",
        json={
            "profile_id": "tokens-saved-check",
            "constraints": [
                {"id": "p1", "kind": "max_prep_minutes", "severity": "preference", "value": 0}
            ],
            "query": "dinner",
        },
    )
    assert r.status_code == 200
    assert r.json()["terminal_state"] == "NEGOTIATE"
    metrics = client.get("/metrics").json()
    assert metrics["cost"]["tokens_saved"] > 0
    assert metrics["cost"]["tokens_saved_estimate_method"]


def test_recommend_increments_the_cost_ledgers_plan_count(client, test_container):
    """Every terminal `/recommend` result folds its CostRecord into the
    process-wide CostLedger, so cost-per-plan actually accumulates over
    the life of the process rather than staying permanently at zero."""
    plans_before = test_container.cost_ledger.plans
    r = client.post(
        "/recommend",
        json={"profile_id": "cost-ledger-check", "constraints": [], "query": "rice"},
    )
    assert r.status_code == 200
    assert test_container.cost_ledger.plans == plans_before + 1
    metrics_plans = client.get("/metrics").json()["cost"]["plans"]
    assert metrics_plans == test_container.cost_ledger.plans


def test_audit_lookup_returns_the_record_and_404_for_unknown(client, test_container):
    client.post(
        "/recommend",
        json={"profile_id": "p6", "constraints": [], "query": "rice"},
    )
    rows = test_container.audit.all()
    assert rows, "every terminal node writes an audit record"
    audit_id = rows[-1]["id"]

    r = client.get(f"/audit/{audit_id}")
    assert r.status_code == 200
    assert r.json()["id"] == audit_id
    assert isinstance(r.json()["skill_versions"], dict)

    missing = client.get("/audit/does-not-exist")
    assert missing.status_code == 404


def test_incidents_endpoint_lists_incidents(client):
    # An infeasible profile always records an "infeasible" incident.
    client.post(
        "/recommend",
        json={
            "profile_id": "p7",
            "constraints": [
                {"id": "c1", "kind": "max_prep_minutes", "severity": "preference", "value": 0}
            ],
            "query": "anything",
        },
    )
    r = client.get("/api/incidents")
    assert r.status_code == 200
    kinds = {i["kind"] for i in r.json()["incidents"]}
    assert "infeasible" in kinds


def test_correlation_id_is_generated_and_echoed(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers

    r2 = client.get("/health", headers={"x-request-id": "caller-supplied-id"})
    assert r2.headers["x-request-id"] == "caller-supplied-id"


def test_unhandled_exception_is_logged_and_never_leaks_a_traceback(client):
    class _Boom:
        class agent:
            trace: ClassVar = []

            @staticmethod
            def run(cs, query=""):
                raise RuntimeError("boom: something this process did not anticipate")

    from beatroot.api.main import app
    from beatroot.api.main import container as container_dependency

    app.dependency_overrides[container_dependency] = lambda: _Boom()
    try:
        r = client.post(
            "/recommend",
            json={"profile_id": "p8", "constraints": [], "query": "rice"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal_error"
    assert "boom" not in body["detail"]  # never leak the exception text
    assert "Traceback" not in r.text
    assert "x-request-id" in r.headers
