"""Dense vector retrieval.

`VectorStore` is the seam. It is shaped for where this system is going, not
for what a 100-recipe catalog lets you get away with: `QdrantVectorStore`
(Task 22) is the production path, applying `exclude_tags` as a payload filter
inside the index. `DenseIndex` below is the zero-dependency, in-memory NumPy
fallback that keeps a fresh clone runnable with no external services — it is
a dev convenience, not the design target.

Either implementation pushes constraint filtering DOWN into the store rather
than letting the caller enumerate the catalog and filter afterwards. That is
the same `O(catalog)` mistake the naive feasibility scan made: at scale the
legal set is too large to pull into Python and filter by hand, so the filter
has to live where the index lives. Spec §10, §17.
"""

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from beatroot.reasoning.llm import LLMClient
from beatroot.settings import get_settings
from beatroot.store.cache import EmbeddingCache, embed_with_cache
from beatroot.trusted.catalog import Catalog, Recipe

log = logging.getLogger("beatroot")


@runtime_checkable
class VectorStore(Protocol):
    """The dense-retrieval seam. `DenseIndex` (dev fallback, NumPy) and
    `QdrantVectorStore` (Task 22, production) both implement this."""

    name: str

    def search(
        self, query: str, limit: int | None = None, exclude_tags: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Constraint filtering is pushed DOWN into the store, never applied
        by the caller afterwards. At scale the legal set is too large to
        enumerate in Python, so the filter has to live where the index is."""
        ...


class DenseIndex:
    """In-memory NumPy cosine similarity — the DEV FALLBACK `VectorStore`.

    Exists so a fresh clone runs with no external services. It is explicitly
    NOT the design target: it holds the whole embedding matrix in memory and
    masks excluded rows with a Python id/tag lookup, both of which stop
    scaling well before the catalog sizes this system is designed for.
    `QdrantVectorStore` (Task 22) gets the identical `VectorStore` interface
    and is the production path. Spec §10, §17.
    """

    name = "numpy"

    def __init__(
        self, provider: LLMClient, catalog: Catalog, embedding_cache: EmbeddingCache | None = None
    ) -> None:
        # Defaults to a cache on the catalog's own connection so embedding
        # is actually cached without waiting on Task 12's Container to hand
        # one in — see store.cache.EmbeddingCache / embed_with_cache.
        self._embedding_cache = (
            embedding_cache if embedding_cache is not None else EmbeddingCache(catalog.conn)
        )
        recipes = catalog.recipes()
        self._ids = [r.id for r in recipes]
        self._tags = [set(r.tags) for r in recipes]
        texts = [self._text(r, catalog) for r in recipes]
        vectors = embed_with_cache(provider, texts, self._embedding_cache)
        matrix = (
            np.asarray(vectors, dtype=np.float32) if texts else np.zeros((0, 1), dtype=np.float32)
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.where(norms == 0, 1.0, norms)
        self._provider = provider

    @staticmethod
    def _text(recipe: Recipe, catalog: Catalog) -> str:
        names = " ".join(
            (catalog.ingredient_payload(i) or {}).get("name", "") for i in recipe.ingredient_ids
        )
        return f"{recipe.name}. {recipe.cuisine}. {names}. {' '.join(sorted(recipe.tags))}"

    def search(
        self, query: str, limit: int | None = None, exclude_tags: list[str] | None = None
    ) -> list[tuple[str, float]]:
        if limit is None:
            limit = get_settings().retrieval.candidate_limit
        if not self._ids or not query:
            return []

        q = np.asarray(
            embed_with_cache(self._provider, [query], self._embedding_cache)[0], dtype=np.float32
        )
        n = float(np.linalg.norm(q)) or 1.0
        sims = self._matrix @ (q / n)

        if exclude_tags:
            # Never scored, never returned — masked to -inf before ranking,
            # not filtered out of an already-ranked list.
            banned = set(exclude_tags)
            for i, tags in enumerate(self._tags):
                if tags & banned:
                    sims[i] = -np.inf

        order = [int(i) for i in np.argsort(-sims)[:limit] if np.isfinite(sims[i])]
        return [(self._ids[i], float(sims[i])) for i in order]


# Process-global: see get_vector_store's docstring for why this must not be
# rebuilt per call. reset_vector_store() exists so tests (and, eventually,
# a hot reseed) can clear it deliberately.
_STORE: VectorStore | None = None


def _build_vector_store(provider: LLMClient, catalog: Catalog) -> VectorStore:
    """The actual construction work — embeds the whole catalog and, for
    Qdrant, touches a remote collection. Callers should go through
    `get_vector_store()`, which caches the result, not this directly."""
    url = get_settings().qdrant_url
    if url:
        from beatroot.retrieval.qdrant_store import QdrantVectorStore

        log.info("vector store: qdrant at %s", url)
        return QdrantVectorStore(provider, catalog, url=url)

    log.warning("vector store: numpy in-memory fallback (set QDRANT_URL for the production path)")
    return DenseIndex(provider, catalog)


def get_vector_store(
    provider: LLMClient, catalog: Catalog, *, force_new: bool = False
) -> VectorStore:
    """Select and cache the `VectorStore` implementation for this process.

    Qdrant is the production path and is used whenever it is configured —
    `docker compose up` sets `QDRANT_URL` automatically, so the containerised
    run (the one that mirrors production) gets Qdrant with no extra step.
    `DenseIndex` (NumPy, in-memory) is the fallback for a bare `uv run` with
    no external services.

    Building a store embeds the whole catalog and, for Qdrant, touches a
    remote collection — that is startup work, not per-request work. Cached
    per process here so `retrieve()` (which calls this on every request
    that doesn't pass an explicit `vector_store`) does not rebuild — for
    Qdrant, does not re-embed the whole catalog and drop/repopulate the
    live collection — on every call. Task 12's `Container` will own this
    lifecycle properly (build once at startup, inject); until then this
    module-level cache is what keeps the production path from being slower
    than the dev fallback it exists to replace.

    Which store answered is always logged, at INFO for the production path
    and WARNING for the fallback, so it is never ambiguous which one is
    serving a given process.
    """
    global _STORE
    if _STORE is None or force_new:
        _STORE = _build_vector_store(provider, catalog)
    return _STORE


def reset_vector_store() -> None:
    """Test-only escape hatch. Without this, a store built from one test's
    catalog/connection would silently keep answering a later test's
    queries against different data — its own bug, and a confusing one."""
    global _STORE
    _STORE = None
