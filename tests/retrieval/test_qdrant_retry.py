"""`QdrantVectorStore._get_collections_with_retry` — the bounded retry
around the cold-start race with `docker compose up` (Task 20 fix round:
qdrant has no HEALTHCHECK guarantee at the exact moment `build_container()`
calls this, so a transient connection failure must retry, not crash the
whole process).

Unlike `tests/retrieval/test_qdrant_store.py`, these need only the
`qdrant_client` PACKAGE (for the exception type it wraps connection
failures in) — never a live server — so they run whenever the `qdrant`
extra is installed, independent of `QDRANT_URL`.
"""

import importlib.util

import pytest
import tenacity

_QDRANT_INSTALLED = importlib.util.find_spec("qdrant_client") is not None

pytestmark = pytest.mark.skipif(
    not _QDRANT_INSTALLED, reason="qdrant_client not installed (optional `qdrant` extra)"
)

if _QDRANT_INSTALLED:
    from qdrant_client.http import exceptions as qdrant_http_exceptions


def _bare_store():
    """A `QdrantVectorStore` with `__init__` never run — this test drives
    `_get_collections_with_retry` directly, in isolation, with no real
    network, no catalog, no embedding."""
    from beatroot.retrieval.qdrant_store import QdrantVectorStore

    return QdrantVectorStore.__new__(QdrantVectorStore)


def test_retries_past_transient_connection_failures_then_succeeds(monkeypatch):
    import beatroot.retrieval.qdrant_store as mod

    real_wait_fixed = tenacity.wait_fixed
    # Zero-second waits so this test costs nothing — the retry COUNT is
    # what's under test, not real elapsed time.
    monkeypatch.setattr(mod.tenacity, "wait_fixed", lambda seconds: real_wait_fixed(0))

    store = _bare_store()
    calls = {"n": 0}

    class _FlakyClient:
        def get_collections(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise qdrant_http_exceptions.ResponseHandlingException("connection refused")
            return "collections-response"

    store._client = _FlakyClient()
    assert store._get_collections_with_retry() == "collections-response"
    assert calls["n"] == 3


def test_gives_up_after_the_bound_reraising_the_real_error(monkeypatch):
    """Bounded, not infinite: a genuinely dead Qdrant must fail loudly, not
    hang forever waiting for a server that will never come up."""
    import beatroot.retrieval.qdrant_store as mod

    real_wait_fixed = tenacity.wait_fixed
    monkeypatch.setattr(mod.tenacity, "wait_fixed", lambda seconds: real_wait_fixed(0))

    store = _bare_store()
    calls = {"n": 0}

    class _AlwaysDownClient:
        def get_collections(self):
            calls["n"] += 1
            raise qdrant_http_exceptions.ResponseHandlingException("connection refused")

    store._client = _AlwaysDownClient()
    with pytest.raises(qdrant_http_exceptions.ResponseHandlingException):
        store._get_collections_with_retry()
    assert calls["n"] == 15, "must stop at the configured attempt bound, not retry forever"
