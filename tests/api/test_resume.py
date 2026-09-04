"""Tests for `POST /recommend/{thread_id}/resume`. Spec §6, §15.

`MealPlanningAgent.resume(thread_id, approved)` (Task 11) had no route —
a MEDICAL grey-band profile reached `PENDING_REVIEW` with no way forward on
either surface. These tests drive the real approval gate end to end through
the API: a profile pauses, then gets resumed both approved and rejected,
plus the three failure modes a caller can hit (unknown thread_id, a thread
that never paused, a thread already resumed).

Builds its own `Container` from the same tiny, coverage-pinned catalog
`tests/agent/conftest.py` uses for the grey-band graph tests (`agent`
fixture there proves `.resume()` itself works; this proves the HTTP/CLI
surface reaches it) — the real ~100-recipe catalog offers no reliable way
to land trust in the grey band on demand.
"""

import pytest
import yaml
from fastapi.testclient import TestClient

from beatroot.agent.batch_plan import WeeklyPlanner
from beatroot.agent.graph import MealPlanningAgent
from beatroot.agent.nodes import Deps
from beatroot.agent.skills_registry import load_skills
from beatroot.api.main import app
from beatroot.api.main import container as container_dependency
from beatroot.confirm.trust_score import load_thresholds
from beatroot.container import THRESHOLDS_PATH, Container
from beatroot.obs.cost import CostLedger
from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.dense import DenseIndex
from beatroot.settings import get_settings
from beatroot.store.audit import AuditLog
from beatroot.store.cache import EmbeddingCache, FeasibilityCache
from beatroot.store.db import connect, seed
from beatroot.store.incidents import IncidentLog
from beatroot.trusted.catalog import Catalog
from beatroot.trusted.index import TagIndex
from tests.agent.conftest import _GREY_INGREDIENTS, _GREY_RECIPES


def _grey_band_container(tmp_path) -> Container:
    data_dir = tmp_path / "grey_data"
    data_dir.mkdir()
    (data_dir / "ingredients.yaml").write_text(yaml.dump(_GREY_INGREDIENTS))
    (data_dir / "recipes.yaml").write_text(yaml.dump(_GREY_RECIPES))

    conn = connect(tmp_path / "grey.db")
    seed(conn, data_dir)
    conn.execute("DELETE FROM ingredients WHERE id = 'gb_dal'")
    conn.commit()

    catalog = Catalog(conn)
    llm = LLMClient.offline()
    embedding_cache = EmbeddingCache(conn)
    vector_store = DenseIndex(llm, catalog, embedding_cache=embedding_cache)
    tag_index = TagIndex(catalog.recipes())
    incidents = IncidentLog(conn)
    audit = AuditLog(conn)
    feasibility_cache = FeasibilityCache(conn)
    skills = load_skills()

    deps = Deps(
        catalog=catalog,
        llm=llm,
        vector_store=vector_store,
        tag_index=tag_index,
        incidents=incidents,
        audit=audit,
        skills=skills,
        preferences=None,
        feasibility_cache=feasibility_cache,
    )
    agent = MealPlanningAgent(deps)
    return Container(
        conn=conn,
        catalog=catalog,
        llm=llm,
        vector_store=vector_store,
        tag_index=tag_index,
        incidents=incidents,
        audit=audit,
        preferences=None,
        feasibility_cache=feasibility_cache,
        embedding_cache=embedding_cache,
        skills=skills,
        thresholds=load_thresholds(THRESHOLDS_PATH),
        agent=agent,
        cost_ledger=CostLedger(),
        planner=WeeklyPlanner(agent, catalog),
    )


@pytest.fixture
def grey_client(tmp_path, monkeypatch):
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    c = _grey_band_container(tmp_path)
    app.dependency_overrides[container_dependency] = lambda: c
    yield TestClient(app), c
    app.dependency_overrides.clear()
    c.close()
    get_settings.cache_clear()


def _medical_body(profile_id: str) -> dict:
    return {
        "profile_id": profile_id,
        "constraints": [
            {"id": "med", "kind": "exclude_tag", "severity": "medical", "value": "peanut"}
        ],
        "query": "Grey Band Rice Bowl",
    }


def test_medical_profile_pauses_at_pending_review(grey_client):
    client, _ = grey_client
    r = client.post("/recommend", json=_medical_body("p-pause"))
    assert r.status_code == 200
    body = r.json()
    assert body["terminal_state"] == "PENDING_REVIEW"
    assert body["result"]["thread_id"]
    assert body["audit_id"], "the AWAITING_APPROVAL audit record's id must be surfaced"


def test_resume_approved_commits(grey_client):
    client, _ = grey_client
    paused = client.post("/recommend", json=_medical_body("p-approve")).json()
    thread_id = paused["result"]["thread_id"]

    r = client.post(f"/recommend/{thread_id}/resume", json={"approved": True})
    assert r.status_code == 200
    body = r.json()
    assert body["terminal_state"] == "COMMIT"
    assert body["result"]["recipe_id"] == "gb_target"
    assert body["audit_id"]
    assert body["audit_id"] != paused["audit_id"], "resume writes its own, later audit record"


def test_resume_rejected_escalates(grey_client):
    client, _ = grey_client
    paused = client.post("/recommend", json=_medical_body("p-reject")).json()
    thread_id = paused["result"]["thread_id"]

    r = client.post(f"/recommend/{thread_id}/resume", json={"approved": False})
    assert r.status_code == 200
    body = r.json()
    assert body["terminal_state"] == "ESCALATE"
    assert body["result"]["reason"] == "low_trust"
    assert body["result"]["failing_signal"] == "human_review_declined"


def test_resume_unknown_thread_id_is_404_not_500(grey_client):
    client, _ = grey_client
    r = client.post("/recommend/does-not-exist/resume", json={"approved": True})
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_resume_on_a_thread_that_never_paused_is_a_conflict_not_500(grey_client):
    """A non-medical (or otherwise not-grey-band) run reaches a terminal
    directly, without ever pausing. Resuming that thread_id must be
    reported cleanly, not attempted."""
    client, container = grey_client
    body = {
        "profile_id": "p-never-paused",
        "constraints": [
            {"id": "pref", "kind": "exclude_tag", "severity": "preference", "value": "peanut"}
        ],
        "query": "Grey Band Rice Bowl",
    }
    committed = client.post("/recommend", json=body).json()
    assert committed["terminal_state"] == "COMMIT"
    thread_id = container.agent.last_thread_id

    r = client.post(f"/recommend/{thread_id}/resume", json={"approved": True})
    assert r.status_code == 409
    assert r.json()["error"] == "already_resolved"


def test_resume_twice_on_the_same_thread_is_a_conflict_not_500(grey_client):
    client, _ = grey_client
    paused = client.post("/recommend", json=_medical_body("p-twice")).json()
    thread_id = paused["result"]["thread_id"]

    first = client.post(f"/recommend/{thread_id}/resume", json={"approved": True})
    assert first.status_code == 200

    second = client.post(f"/recommend/{thread_id}/resume", json={"approved": True})
    assert second.status_code == 409
    assert second.json()["error"] == "already_resolved"
