"""Two caches for the production retrieval path. Spec §17.

Both are the first things the spec names to add under load: feasibility
computation is O(catalog) per profile, and embedding is a network call per
text. Neither cache is optional polish — without them, the two most
expensive operations in the system recompute on every request.

Both report `hits`/`misses`/`hit_rate` so caching is measurable (Task 12
surfaces this at `/metrics`) rather than merely asserted.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Protocol

from beatroot.contracts.core import ConstraintSet


class _EmbeddingProvider(Protocol):
    """Structural type for the embedding call `embed_with_cache` needs.

    Deliberately NOT `beatroot.reasoning.llm.LLMClient` — `store/` must never
    import `beatroot.reasoning` (tests/test_boundaries.py enforces this
    tier boundary transitively). A Protocol lets this module stay typed
    without crossing it; `LLMClient` satisfies it structurally.

    `embedder_id` matters as much as `embed()` itself: it is what
    `EmbeddingCache` keys on alongside the text (see below) so vectors
    produced by two different embedders — the offline hash-of-tokens stub
    vs a real model, or two different real models — can never collide in
    the cache. `LLMClient.embedder_id` returns `"echo"` when offline
    (regardless of the configured `embedding_model` string, since the
    offline stub ignores it entirely) or the concrete model string
    otherwise.
    """

    @property
    def embedder_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _Cache:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock | None = None) -> None:
        self.conn = conn
        # See `store.audit.AuditLog.__init__` — a Container-owned lock,
        # shared with every other store writing this same `conn`, held
        # across each write's execute()+commit() pair so two threadpool
        # requests' writes can never interleave into one transaction.
        self._lock = lock or threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


class FeasibilityCache(_Cache):
    """Keyed on the ConstraintSet fingerprint, NOT the profile id.

    Many users share constraint shapes — "vegetarian, no nuts, under 30 min" is
    one key serving thousands of profiles. Keying on profile_id would give a
    hit rate near zero, since almost no two profiles are the same person.
    Spec §17.

    A cache MISS returns `None`. An empty list is a legitimate cached answer
    for a profile with no feasible recipes — conflating the two would make
    every infeasible profile recompute forever, exactly the case this cache
    exists for.
    """

    def get(self, cs: ConstraintSet) -> list[str] | None:
        # Same shared `_lock` every writer through this `conn` holds (see
        # `_Cache.__init__`). sqlite3 does not tolerate simultaneous
        # unlocked `execute()` calls from different threads on one
        # connection at all — an unlocked `SELECT` here could race a
        # writer's execute()+commit() pair into the same `InterfaceError`
        # that lock exists to prevent.
        with self._lock:
            row = self.conn.execute(
                "SELECT recipe_ids FROM feasibility_cache WHERE fingerprint = ?",
                (cs.fingerprint(),),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        recipe_ids: list[str] = json.loads(row["recipe_ids"])
        return recipe_ids

    def put(self, cs: ConstraintSet, recipe_ids: list[str]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO feasibility_cache (fingerprint, recipe_ids, created_at)"
                " VALUES (?,?,?)",
                (cs.fingerprint(), json.dumps(recipe_ids), datetime.now(UTC).isoformat()),
            )
            self.conn.commit()

    def invalidate_all(self) -> None:
        """Called from `store.db.seed()` — cached entries hold recipe ids,
        so a reseed must clear them or answers can point at ids that no
        longer exist (or now mean something different)."""
        with self._lock:
            self.conn.execute("DELETE FROM feasibility_cache")
            self.conn.commit()


class EmbeddingCache(_Cache):
    """Keyed on a hash of BOTH `embedder_id` and the exact text embedded.

    Text alone used to be the whole key. That silently let a cold-start
    offline run's hash-of-tokens vectors satisfy a later real-provider run's
    cache lookups (or vice versa) for identical text — two completely
    different vector spaces, indistinguishable to a cache keyed on content
    alone. `embedder_id` scopes every entry to the embedder that actually
    produced it, so a real model's vectors are never handed back to a
    different embedder's caller, and changed text still misses exactly as
    before.
    """

    @staticmethod
    def _key(embedder_id: str, text: str) -> str:
        # NUL-separated: neither field can forge a collision by containing
        # the other's boundary the way plain concatenation could.
        return hashlib.sha256(f"{embedder_id}\x00{text}".encode()).hexdigest()

    def get(self, embedder_id: str, text: str) -> list[float] | None:
        # See `FeasibilityCache.get` — same shared `_lock`, same reason: an
        # unlocked `SELECT` racing a writer's execute()+commit() pair on
        # this shared `conn` can surface the same `InterfaceError` the lock
        # was introduced to prevent.
        with self._lock:
            row = self.conn.execute(
                "SELECT vector FROM embedding_cache WHERE content_hash = ?",
                (self._key(embedder_id, text),),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        vector: list[float] = json.loads(row["vector"])
        return vector

    def put(self, embedder_id: str, text: str, vector: list[float]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, vector, created_at)"
                " VALUES (?,?,?)",
                (self._key(embedder_id, text), json.dumps(vector), datetime.now(UTC).isoformat()),
            )
            self.conn.commit()


def embed_with_cache(
    provider: _EmbeddingProvider, texts: list[str], cache: EmbeddingCache | None
) -> list[list[float]]:
    """Embed `texts`, calling `provider.embed()` only for the entries that
    miss `cache`. Order-preserving: `result[i]` is always the embedding of
    `texts[i]`, regardless of which positions were cache hits vs misses — a
    scrambled order here would silently corrupt every consumer's index
    (recipe i built with recipe j's vector), so hits and misses are written
    back into their original positions rather than concatenated.

    `cache=None` is a plain passthrough for callers with no connection to
    cache against.

    No eviction here: entries never expire. `feasibility_cache` is bounded
    by distinct constraint fingerprints (low risk); this table grows with
    every distinct text ever embedded, so a production deployment needs a
    TTL or LRU bound eventually — noted, not fixed, here.
    """
    if not texts:
        return []
    if cache is None:
        return provider.embed(texts)

    embedder_id = provider.embedder_id
    results: list[list[float] | None] = [None] * len(texts)
    miss_positions: list[int] = []
    miss_texts: list[str] = []
    for i, text in enumerate(texts):
        cached = cache.get(embedder_id, text)
        if cached is None:
            miss_positions.append(i)
            miss_texts.append(text)
        else:
            results[i] = cached

    if miss_texts:
        embedded = provider.embed(miss_texts)
        for pos, text, vector in zip(miss_positions, miss_texts, embedded, strict=True):
            results[pos] = vector
            cache.put(embedder_id, text, vector)

    return results  # type: ignore[return-value]
