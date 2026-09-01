"""API tests for Task 23's async explanation split. Spec §15, §17.

Builds its own `Container` with `BEATROOT_ASYNC_EXPLANATION=1` — deliberately
NOT the shared `client`/`test_container` fixture from `tests/api/conftest.py`
(no `async_explanation` override there), so this module never leaves that
env var set for any other test file to accidentally inherit, and every
other API test module keeps exercising the default, synchronous behaviour
completely unchanged.
"""

import time

import pytest
from fastapi.testclient import TestClient

from beatroot.api.main import app
from beatroot.api.main import container as container_dependency
from beatroot.container import build_container
from beatroot.settings import get_settings


@pytest.fixture(scope="module")
def async_container(tmp_path_factory, monkeypatch_module_async):
    # Kept SET for the whole module, not just for this build call:
    # `explain_node`'s async-vs-sync branch reads `settings.
    # async_explanation` fresh on every `MealPlanningAgent.run()` (a new
    # graph is built per call), not only once at container-construction
    # time — undoing this env var early would make every /recommend in
    # this module silently fall back to the synchronous branch even
    # though `async_container.explanation_queue` is already built and
    # non-None. `test_explanation_route_404s_when_async_mode_is_off`
    # below builds its OWN throwaway sync container instead of sharing
    # this module's env, so it is unaffected by this staying set.
    monkeypatch_module_async.setenv("BEATROOT_OFFLINE", "1")
    monkeypatch_module_async.setenv("BEATROOT_ASYNC_EXPLANATION", "1")
    get_settings.cache_clear()
    db_path = tmp_path_factory.mktemp("api-async") / "async.db"
    c = build_container(db_path)
    yield c
    c.close()
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module_async():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def async_client(async_container):
    app.dependency_overrides[container_dependency] = lambda: async_container
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_recommend_returns_immediately_with_prose_deferred(async_container, async_client):
    assert async_container.explanation_queue is not None
    r = async_client.post(
        "/recommend",
        json={"profile_id": "async-p1", "constraints": [], "query": "rice"},
    )
    assert r.status_code == 200
    body = r.json()
    if body["terminal_state"] != "COMMIT":
        pytest.skip(f"this profile did not COMMIT ({body['terminal_state']})")
    result = body["result"]
    # The card is already complete — nothing here waited on the model.
    assert result["recipe_id"]
    assert result["nutrition"]["kcal"] > 0
    assert result["trust"]["composite"] > 0
    assert result["explanation"] is None
    # The offline stub is a hash, not a network call — it can genuinely
    # finish before this response body is even assembled. Either status
    # proves the same thing: the route never blocked on it either way.
    assert result["explanation_status"] in {"pending", "ready"}
    assert body["thread_id"]


def test_explanation_route_fills_in_once_the_worker_finishes(async_container, async_client):
    r = async_client.post(
        "/recommend",
        json={"profile_id": "async-p2", "constraints": [], "query": "dal"},
    )
    body = r.json()
    if body["terminal_state"] != "COMMIT":
        pytest.skip(f"this profile did not COMMIT ({body['terminal_state']})")
    thread_id = body["thread_id"]

    deadline = time.monotonic() + 5.0
    status = "pending"
    text = None
    while time.monotonic() < deadline:
        poll = async_client.get(f"/recommend/{thread_id}/explanation")
        assert poll.status_code == 200
        status = poll.json()["explanation_status"]
        text = poll.json()["explanation"]
        if status == "ready":
            break
        time.sleep(0.05)

    assert status == "ready"
    assert text


def test_explanation_route_404s_when_async_mode_is_off(tmp_path, monkeypatch):
    """A container built with the default (synchronous) settings has no
    `explanation_queue` at all — the route is a clean 404, not a silent
    always-pending stub. Builds its own throwaway sync container rather
    than the shared `client` fixture, so this test can never be sensitive
    to fixture build order relative to `async_container` above."""
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    # Explicitly OFF, not merely unset. Deleting the env var used to be
    # enough because the shipped default was false; once
    # `config/beatroot.yaml` set async_explanation: true for latency, an
    # unset env var no longer meant "off" and this test built an async
    # container while claiming to build a sync one.
    monkeypatch.setenv("BEATROOT_ASYNC_EXPLANATION", "0")
    get_settings.cache_clear()
    sync_container = build_container(tmp_path / "sync.db")
    try:
        assert sync_container.explanation_queue is None
        app.dependency_overrides[container_dependency] = lambda: sync_container
        client = TestClient(app)
        r = client.get("/recommend/some-thread-id/explanation")
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()
        sync_container.close()
        get_settings.cache_clear()
