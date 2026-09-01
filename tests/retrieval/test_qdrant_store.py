"""Qdrant is opt-in: these tests SKIP cleanly when QDRANT_URL is unset, so a
fresh clone with no services still runs the full suite green. Run with a
local server via `docker compose up -d qdrant` and
`QDRANT_URL=http://localhost:6333 uv run pytest tests/retrieval/test_qdrant_store.py -v`.

None of these have ever executed in this environment (no Docker daemon
available here) — they skip, which is correct behaviour for an opt-in path,
but that also means "numpy and qdrant agree" below is an untested claim
until someone runs this suite with a live Qdrant server.
"""

import os
from pathlib import Path

import pytest

qdrant_url = os.getenv("QDRANT_URL")
pytestmark = pytest.mark.skipif(
    not qdrant_url, reason="QDRANT_URL not set — Qdrant is opt-in, NumPy is the default"
)

DATA = Path(__file__).parents[2] / "data"


def test_qdrant_store_satisfies_the_vector_store_protocol(tmp_path):
    from beatroot.reasoning.llm import LLMClient
    from beatroot.retrieval.dense import VectorStore
    from beatroot.retrieval.qdrant_store import QdrantVectorStore
    from beatroot.store.db import connect, seed
    from beatroot.trusted.catalog import Catalog

    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    cat = Catalog(conn)
    store = QdrantVectorStore(LLMClient.offline(), cat, url=qdrant_url, collection="beatroot_test")
    store.reindex(cat)  # explicit populate — construction alone must not reindex
    assert isinstance(store, VectorStore)
    results = store.search("rice", limit=5)
    assert results and all(isinstance(i, str) and isinstance(s, float) for i, s in results)


def test_constructing_the_store_again_does_not_wipe_an_existing_collection(tmp_path):
    """The bug this guards against: __init__ used to unconditionally call
    recreate_collection, so merely building a second store instance against
    the same collection silently emptied it out from under any concurrent
    search — in the path the whole task exists to make production-safe."""
    from beatroot.reasoning.llm import LLMClient
    from beatroot.retrieval.qdrant_store import QdrantVectorStore
    from beatroot.store.db import connect, seed
    from beatroot.trusted.catalog import Catalog

    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    cat, provider = Catalog(conn), LLMClient.offline()
    collection = "beatroot_test_no_wipe"

    first = QdrantVectorStore(provider, cat, url=qdrant_url, collection=collection)
    first.reindex(cat)
    before = first.search("rice", limit=100)
    assert before

    # A second instance built against the SAME collection must reuse it —
    # not drop and rebuild it — since __init__ only creates when absent.
    second = QdrantVectorStore(provider, cat, url=qdrant_url, collection=collection)
    after = second.search("rice", limit=100)
    assert after == before


def test_qdrant_store_applies_exclude_tags_as_a_payload_filter(tmp_path):
    """The whole point of the seam: an excluded-tag recipe is dropped inside
    the index, not filtered out of an already-ranked Python list."""
    from beatroot.reasoning.llm import LLMClient
    from beatroot.retrieval.qdrant_store import QdrantVectorStore
    from beatroot.store.db import connect, seed
    from beatroot.trusted.catalog import Catalog

    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    cat = Catalog(conn)
    store = QdrantVectorStore(
        LLMClient.offline(), cat, url=qdrant_url, collection="beatroot_test_filter"
    )
    store.reindex(cat)
    banned = next(iter(next(r.tags for r in cat.recipes() if r.tags)))
    results = store.search("rice", limit=100, exclude_tags=[banned])
    tags_by_id = {r.id: r.tags for r in cat.recipes()}
    assert all(banned not in tags_by_id.get(rid, set()) for rid, _ in results)


def test_numpy_and_qdrant_agree_on_top_result(tmp_path):
    """The seam is only honest if both implementations rank the same way."""
    from beatroot.reasoning.llm import LLMClient
    from beatroot.retrieval.dense import DenseIndex
    from beatroot.retrieval.qdrant_store import QdrantVectorStore
    from beatroot.store.db import connect, seed
    from beatroot.trusted.catalog import Catalog

    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    cat, provider = Catalog(conn), LLMClient.offline()
    numpy_top = DenseIndex(provider, cat).search("rice", limit=1)[0][0]
    store = QdrantVectorStore(provider, cat, url=qdrant_url, collection="beatroot_test_agree")
    store.reindex(cat)
    qdrant_top = store.search("rice", limit=1)[0][0]
    assert numpy_top == qdrant_top


def test_get_vector_store_selects_qdrant_when_url_set(tmp_path, monkeypatch):
    from beatroot.reasoning.llm import LLMClient
    from beatroot.retrieval.dense import get_vector_store, reset_vector_store
    from beatroot.retrieval.qdrant_store import QdrantVectorStore
    from beatroot.settings import get_settings
    from beatroot.store.db import connect, seed
    from beatroot.trusted.catalog import Catalog

    reset_vector_store()
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    get_settings.cache_clear()  # settings is lru_cached; the env change above
    # is invisible to it otherwise.
    try:
        conn = connect(tmp_path / "t.db")
        seed(conn, DATA)
        store = get_vector_store(LLMClient.offline(), Catalog(conn))
        assert isinstance(store, QdrantVectorStore)
    finally:
        reset_vector_store()
        get_settings.cache_clear()
