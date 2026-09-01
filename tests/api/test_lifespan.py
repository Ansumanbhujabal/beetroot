"""Tests that the app's real `lifespan` — and the `build_container()` call
inside it — actually runs. Spec §15.

`tests/api/conftest.py`'s `client` fixture deliberately never enters
`TestClient(app)` as a context manager (its own docstring says so): it
overrides the `container` dependency directly so route handlers never call
`request.app.state.container`, and `lifespan` never runs. That is the right
choice for the other 12 route tests — a second, unused Container built
against the process-default `beatroot.db` for every test would be wasteful
and pointless — but it means nothing in the suite proves `lifespan` itself
works, or that a genuinely drifted skill lock stops the app the same way it
stops the CLI. These two tests are that proof: both actually enter
`with TestClient(app) as c:`, which runs Starlette's real startup event,
which calls the real `build_container()`, exactly like `uvicorn` does.
"""

import shutil

import pytest
from fastapi.testclient import TestClient

import beatroot.container as container_module
from beatroot.api.main import app
from beatroot.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_boot_db(tmp_path, monkeypatch):
    """Every test in this file points the real `build_container()` — called
    with no arguments, exactly as `lifespan` calls it — at a throwaway
    sqlite file instead of the process-default `ROOT / "beatroot.db"`.
    `DEFAULT_DB` is read from this module's own namespace at call time
    (`db_path or DEFAULT_DB`, not a bound default), so patching it here
    changes what `lifespan`'s parameterless `build_container()` call does
    without needing `lifespan` itself to take any arguments."""
    monkeypatch.setattr(container_module, "DEFAULT_DB", tmp_path / "boot.db")
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_lifespan_actually_builds_the_container_and_serves_a_request():
    """The production boot path — `uvicorn` -> `lifespan` ->
    `build_container()` — covered somewhere in the suite, not just the fast
    dependency-override path every other route test uses."""
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["skills_locked"] is True

        r2 = client.post(
            "/recommend",
            json={"profile_id": "lifespan-p1", "constraints": [], "query": "rice"},
        )
        assert r2.status_code == 200
        assert r2.json()["terminal_state"] in {"COMMIT", "NEGOTIATE", "ESCALATE"}


def test_lifespan_fails_loudly_when_the_skill_lock_has_drifted(tmp_path, monkeypatch):
    """A drifted skill lock must stop the app from ever accepting a
    request — not just stop the CLI, not just be provable by calling
    `build_container` directly. Mirrors `tests/test_container.py`'s drift
    fixture, but drives it through the actual ASGI startup event."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for src in container_module.SKILLS_DIR.glob("*.skill.md"):
        shutil.copy(src, skills_dir / src.name)
    drifted = skills_dir / "assess_trust.skill.md"
    drifted.write_text(drifted.read_text() + "\nTHIS BODY WAS MUTATED FOR A TEST.\n")

    monkeypatch.setattr(container_module, "SKILLS_DIR", skills_dir)

    with pytest.raises(RuntimeError, match="assess_trust"), TestClient(app):
        pass  # pragma: no cover — startup must raise before this body runs
