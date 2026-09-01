from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.store.cache import EmbeddingCache, FeasibilityCache
from beatroot.store.db import connect


def _cs(value: str) -> ConstraintSet:
    return ConstraintSet(
        profile_id="p",
        constraints=[
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value=value)
        ],
    )


def test_feasibility_cache_keys_on_constraints_not_profile(tmp_path):
    """Many users share constraint SHAPES. Keying on profile_id would miss
    every one of those hits. Spec §17."""
    cache = FeasibilityCache(connect(tmp_path / "t.db"))
    a = ConstraintSet(profile_id="alice", constraints=_cs("peanut").constraints)
    b = ConstraintSet(profile_id="bob", constraints=_cs("peanut").constraints)
    cache.put(a, ["rec_1", "rec_2"])
    assert cache.get(b) == ["rec_1", "rec_2"]


def test_constraint_order_does_not_change_the_key(tmp_path):
    cache = FeasibilityCache(connect(tmp_path / "t.db"))
    c1 = Constraint(id="a", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")
    c2 = Constraint(id="b", kind="exclude_tag", severity=Severity.MEDICAL, value="dairy")
    forward = ConstraintSet(profile_id="p", constraints=[c1, c2])
    reverse = ConstraintSet(profile_id="p", constraints=[c2, c1])
    cache.put(forward, ["rec_1"])
    assert cache.get(reverse) == ["rec_1"]


def test_miss_returns_none_not_empty_list(tmp_path):
    """An empty list is a legitimate cached answer (infeasible profile). It must
    be distinguishable from a miss, or every infeasible profile recomputes."""
    cache = FeasibilityCache(connect(tmp_path / "t.db"))
    assert cache.get(_cs("peanut")) is None
    cache.put(_cs("peanut"), [])
    assert cache.get(_cs("peanut")) == []


def test_embedding_cache_keys_on_content_hash(tmp_path):
    cache = EmbeddingCache(connect(tmp_path / "t.db"))
    cache.put("m1", "paneer butter masala", [0.1, 0.2, 0.3])
    assert cache.get("m1", "paneer butter masala") == [0.1, 0.2, 0.3]
    assert cache.get("m1", "dal tadka") is None


def test_embedding_cache_invalidates_on_content_change(tmp_path):
    """Recipe text changed means the old vector is wrong, not stale-but-usable."""
    cache = EmbeddingCache(connect(tmp_path / "t.db"))
    cache.put("m1", "paneer masala", [0.1])
    assert cache.get("m1", "paneer masala v2") is None


def test_embedding_cache_keys_on_embedder_identity_not_just_text(tmp_path):
    """The bug this guards against: two embedders sharing configured text
    must never share a cache entry. An offline hash-of-tokens vector and a
    real model's vector for the identical text are not interchangeable —
    keying on text alone would let one silently answer for the other."""
    cache = EmbeddingCache(connect(tmp_path / "t.db"))
    cache.put("echo", "paneer masala", [0.1])
    cache.put("azure/text-embedding-3-small", "paneer masala", [0.9, 0.9])
    assert cache.get("echo", "paneer masala") == [0.1]
    assert cache.get("azure/text-embedding-3-small", "paneer masala") == [0.9, 0.9]
    assert cache.get("ollama/other-model", "paneer masala") is None


def test_cache_stats_are_reported(tmp_path):
    cache = FeasibilityCache(connect(tmp_path / "t.db"))
    cache.get(_cs("peanut"))
    cache.put(_cs("peanut"), ["r"])
    cache.get(_cs("peanut"))
    assert cache.hits == 1
    assert cache.misses == 1


def test_invalidate_all_clears_feasibility_cache(tmp_path):
    """A reseed changes recipe ids underneath any cached answer — the cache
    must be emptied, not left holding stale ids."""
    cache = FeasibilityCache(connect(tmp_path / "t.db"))
    cache.put(_cs("peanut"), ["rec_1"])
    cache.invalidate_all()
    assert cache.get(_cs("peanut")) is None


def test_reseed_invalidates_the_feasibility_cache(tmp_path):
    """seed() must call FeasibilityCache.invalidate_all() itself — cached
    entries hold recipe ids that a reseed can change underneath them."""
    from pathlib import Path

    from beatroot.store.db import seed

    data_dir = Path(__file__).parents[2] / "data"
    conn = connect(tmp_path / "t.db")
    seed(conn, data_dir)

    cache = FeasibilityCache(conn)
    cache.put(_cs("peanut"), ["rec_1"])
    assert cache.get(_cs("peanut")) == ["rec_1"]

    seed(conn, data_dir)
    assert cache.get(_cs("peanut")) is None


def test_embed_with_cache_preserves_order_and_only_embeds_misses(tmp_path):
    """Order must survive regardless of which positions were cache hits —
    a scrambled order would silently corrupt every consumer's index."""
    from beatroot.store.cache import EmbeddingCache, embed_with_cache

    cache = EmbeddingCache(connect(tmp_path / "t.db"))
    cache.put("test-model", "b", [9.0])

    calls: list[list[str]] = []

    class _Provider:
        embedder_id = "test-model"

        def embed(self, texts):
            calls.append(list(texts))
            return [[float(len(t)) + 100] for t in texts]

    result = embed_with_cache(_Provider(), ["a", "b", "c"], cache)
    assert result == [[101.0], [9.0], [101.0]]
    assert calls == [["a", "c"]], "must only call provider.embed() for the misses"


def test_embed_with_cache_none_is_a_passthrough(tmp_path):
    class _Provider:
        embedder_id = "test-model"

        def embed(self, texts):
            return [[1.0] for _ in texts]

    from beatroot.store.cache import embed_with_cache

    assert embed_with_cache(_Provider(), ["x"], None) == [[1.0]]
