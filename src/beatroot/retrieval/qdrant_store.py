"""Qdrant-backed `VectorStore` — the production retrieval path. Spec §10, §17.

`DenseIndex` (Task 8) holds the whole embedding matrix in NumPy and masks
excluded rows with a Python id/tag scan; that is fine for a ~100-recipe demo
catalog and falls over well before the catalog sizes this system is designed
for. `QdrantVectorStore` implements the identical `VectorStore` protocol but
stores each recipe's derived tags in the Qdrant payload, so at scale the hard
constraint filter becomes a **payload filter evaluated inside the index**
(`Filter(must_not=[...])` below) rather than something the caller enumerates
the catalog in Python to apply. That is the whole point of the seam: an
illegal candidate is never scored, never returned — filtered before ranking,
not after — and this holds however large the legal set gets, because the
filtering happens where the index lives, not where Python does.

Constructing this class does NOT reindex. `__init__` only creates the
collection if it is missing; an already-populated collection is reused as
is. Reindexing — dropping and repopulating from the catalog's current
embeddings — is the explicit, deliberate `reindex()` method, never a side
effect of building a client. `get_vector_store()` in `dense.py` also caches
the instance itself, once per process, precisely so this constructor is
startup work, not per-request work — a client built (and reused) on every
`retrieve()` call must never drop a collection another request might be
searching concurrently.

`qdrant_client` is an OPTIONAL dependency (`uv add --optional qdrant`) and is
imported only inside methods, never at module import time, so a fresh clone
with `QDRANT_URL` unset never needs it installed.
"""

import logging
import uuid
from typing import TYPE_CHECKING

import tenacity

from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.dense import DenseIndex
from beatroot.settings import get_settings
from beatroot.store.cache import EmbeddingCache, embed_with_cache
from beatroot.trusted.catalog import Catalog

if TYPE_CHECKING:
    # Type-only — see the module docstring: qdrant_client is never imported
    # at module level so a fresh clone with QDRANT_URL unset never needs it
    # installed.
    from qdrant_client.http.models.models import CollectionsResponse

log = logging.getLogger("beatroot.retrieval.qdrant")


class QdrantVectorStore:
    """Qdrant-backed `VectorStore`. Opt-in via `QDRANT_URL` — the production
    path, selected automatically once a Qdrant server is reachable (docker
    compose sets `QDRANT_URL`, so the containerised run gets it by default).

    HNSW index, cosine distance. Payload carries the derived tags, cuisine,
    and prep time so the hard-constraint filter runs as a Qdrant payload
    filter inside the index rather than a Python-side scan after retrieval.
    """

    name = "qdrant"

    def __init__(
        self,
        provider: LLMClient,
        catalog: Catalog,
        url: str | None = None,
        collection: str = "beatroot_recipes",
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self._provider = provider
        self._client = QdrantClient(url=url or get_settings().qdrant_url)
        self._collection = collection
        # Defaults to a cache on the catalog's own connection so embedding
        # is actually cached without waiting on Task 12's Container to hand
        # one in — see store.cache.EmbeddingCache / embed_with_cache.
        self._embedding_cache = (
            embedding_cache if embedding_cache is not None else EmbeddingCache(catalog.conn)
        )

        existing = {c.name for c in self._get_collections_with_retry().collections}
        if collection not in existing:
            self.reindex(catalog)
        # else: reuse the live collection as is. Reindexing is an explicit
        # operation, never a side effect of constructing a client — see the
        # module docstring and the task-22 review that caught this.

    def _get_collections_with_retry(self) -> "CollectionsResponse":
        """The cold-start race this bounds: `docker compose up` starts
        `beatroot` and `qdrant` together, and this call happens during
        `build_container()`'s eager startup — before qdrant's HTTP server
        is necessarily accepting connections yet, healthcheck/
        `depends_on: condition: service_healthy` notwithstanding (belt-
        and-braces, not a substitute for this). `ResponseHandlingException`
        is what qdrant-client wraps a connection failure in; retried with a
        short fixed wait and a hard bound (~30s total) so a genuinely dead
        Qdrant still fails loudly rather than hanging forever.
        """
        from qdrant_client.http.exceptions import ResponseHandlingException

        @tenacity.retry(
            retry=tenacity.retry_if_exception_type(ResponseHandlingException),
            wait=tenacity.wait_fixed(2),
            stop=tenacity.stop_after_attempt(15),
            before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
            reraise=True,
        )
        def _connect() -> "CollectionsResponse":
            return self._client.get_collections()

        return _connect()

    def reindex(self, catalog: Catalog) -> None:
        """Destructive: drops and repopulates the collection from the
        catalog's current embeddings. Call this deliberately — e.g. after a
        reseed — never implicitly from `__init__`: every call that merely
        builds or looks up a store must not empty a collection out from
        under a concurrent search.
        """
        from qdrant_client.models import Distance, PointStruct, VectorParams

        recipes = catalog.recipes()
        texts = [DenseIndex._text(r, catalog) for r in recipes]
        vectors = embed_with_cache(self._provider, texts, self._embedding_cache)
        if not vectors:
            return

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_OID, r.id)),
                    vector=v,
                    payload={
                        "recipe_id": r.id,
                        "tags": sorted(r.tags),
                        "cuisine": r.cuisine,
                        "prep_minutes": r.prep_minutes,
                    },
                )
                for r, v in zip(recipes, vectors, strict=True)
            ],
        )

    def search(
        self, query: str, limit: int | None = None, exclude_tags: list[str] | None = None
    ) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        if limit is None:
            limit = get_settings().retrieval.candidate_limit
        if not query:
            return []

        # The exclusion set becomes a payload filter inside Qdrant, so the
        # illegal set is dropped before ranking — the index never scores a
        # point that carries a banned tag. This is what "filter-before-rank
        # in the index" means once the catalog is too large to enumerate.
        query_filter = None
        if exclude_tags:
            query_filter = Filter(
                must_not=[FieldCondition(key="tags", match=MatchAny(any=list(exclude_tags)))]
            )

        query_vector = embed_with_cache(self._provider, [query], self._embedding_cache)[0]
        # `.search()` was removed from qdrant-client (this project pins
        # >=1.12; `.search()` is gone by 1.19) in favour of `.query_points()`,
        # which wraps the same hits in a `QueryResponse.points` list rather
        # than returning them directly. Calling the old `.search()` here
        # would raise `AttributeError` on any real Qdrant-backed request —
        # this path is only exercised with `QDRANT_URL` set, so the offline
        # test suite could never have caught it.
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        # `payload` is optional in the client's own typing even though this
        # class always upserts one (see `reindex`); skip rather than crash
        # on a hit Qdrant somehow returned with none.
        return [(h.payload["recipe_id"], float(h.score)) for h in response.points if h.payload]
