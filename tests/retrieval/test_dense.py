from pathlib import Path

import pytest

from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.dense import DenseIndex
from beatroot.store.db import connect, seed
from beatroot.trusted.catalog import Catalog

DATA = Path(__file__).parents[2] / "data"


@pytest.fixture
def catalog(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    return Catalog(conn)


def test_dense_ranks_a_named_dish_above_an_unrelated_one(catalog):
    """Guards against the dense path being noise.

    Every other retrieval test asserts shape and safety, all of which hold
    with a RANDOM dense ranking. This one asserts signal: a query naming a
    dish must rank that dish above an unrelated one, through the dense store
    alone.
    """
    index = DenseIndex(LLMClient.offline(), catalog)
    # The limit must cover the WHOLE catalog, derived rather than typed: a
    # hardcoded 100 silently rotted once the catalog grew past it, and the
    # failure was a `ValueError: 'rec_tabbouleh' is not in list` from
    # `.index()` — a test that stops testing what it claims to, and reports
    # it as a crash rather than a ranking regression.
    everything = len(catalog.recipes())
    ranked = [rid for rid, _ in index.search("jeera rice", limit=everything)]
    assert len(ranked) == everything, "the search window must span the whole catalog"
    assert ranked.index("rec_jeera_rice") < ranked.index("rec_tabbouleh")


def test_dense_puts_a_named_dish_in_the_top_ten(catalog):
    index = DenseIndex(LLMClient.offline(), catalog)
    top = [rid for rid, _ in index.search("jeera rice", limit=10)]
    assert "rec_jeera_rice" in top


def test_dense_index_populates_the_embedding_cache(catalog):
    """EmbeddingCache must actually be exercised by DenseIndex, not merely
    exist unused — Task 22 review."""
    from beatroot.store.cache import EmbeddingCache

    DenseIndex(LLMClient.offline(), catalog)
    cache = EmbeddingCache(catalog.conn)
    text = DenseIndex._text(catalog.recipes()[0], catalog)
    assert cache.get(LLMClient.offline().embedder_id, text) is not None


def test_get_vector_store_builds_only_once_per_process(catalog):
    """The bug this guards against: get_vector_store() used to construct a
    fresh store on every call, so the 'production' Qdrant path re-embedded
    and re-indexed the whole catalog on every retrieve() — Task 22 review."""
    from beatroot.retrieval.dense import get_vector_store

    provider = LLMClient.offline()
    first = get_vector_store(provider, catalog)
    second = get_vector_store(provider, catalog)
    assert first is second


def test_get_vector_store_force_new_rebuilds(catalog):
    from beatroot.retrieval.dense import get_vector_store

    provider = LLMClient.offline()
    first = get_vector_store(provider, catalog)
    second = get_vector_store(provider, catalog, force_new=True)
    assert first is not second


def test_reset_vector_store_clears_the_cache(catalog):
    from beatroot.retrieval.dense import get_vector_store, reset_vector_store

    provider = LLMClient.offline()
    first = get_vector_store(provider, catalog)
    reset_vector_store()
    second = get_vector_store(provider, catalog)
    assert first is not second
