"""Fixtures for the API tests.

`test_container` is a real `Container` built once per test module, against
a throwaway sqlite file — never the process default `beatroot.db`. `client`
wires it into the FastAPI app the same way any test should swap a
dependency here: `app.dependency_overrides[container]`, never monkeypatching
an import. That override means the app's own `lifespan` never has to run
for these tests to be meaningful — the route layer never reaches for
`request.app.state.container` once the dependency is overridden.
"""

import pytest
from fastapi.testclient import TestClient

from beatroot.api.main import app
from beatroot.api.main import container as container_dependency
from beatroot.container import build_container
from beatroot.settings import get_settings


@pytest.fixture(scope="module")
def test_container(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    db_path = tmp_path_factory.mktemp("api") / "api.db"
    c = build_container(db_path)
    yield c
    c.close()
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    """`monkeypatch` is function-scoped by default; this module-scoped
    variant lets `test_container` (module-scoped, so the catalog is only
    embedded once for the whole file) still control the offline env var
    without leaking it into every other test module in the suite."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def client(test_container):
    """No `with TestClient(app) as c:` here on purpose: entering that
    context runs the app's real `lifespan`, which would build a SECOND,
    unused Container against the process-default `beatroot.db`. The
    dependency override below means route handlers never call
    `request.app.state.container` at all, so lifespan never needs to run
    for these tests to exercise the real routes end to end."""
    app.dependency_overrides[container_dependency] = lambda: test_container
    yield TestClient(app)
    app.dependency_overrides.clear()
