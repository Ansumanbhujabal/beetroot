"""The composition root. Spec §15.

Every dependency the CLI and API need is constructed exactly once, here, and
injected into whatever needs it — `MealPlanningAgent` via `agent.nodes.Deps`,
route handlers and CLI commands via the returned `Container`. Nothing outside
this module reaches for a global to get a catalog, an LLM client, or a
vector store. A test swaps any one of them by constructing its own
`Container` (or calling `build_container` with different arguments), never
by monkeypatching an import somewhere else in the tree.

Three things this module owns on purpose because they used to leak:

1. **The vector store lifecycle.** `retrieval.dense.get_vector_store` caches
   its result in a bare module global, unkeyed by catalog or connection — a
   second `Container` (a second catalog, a second db_path, a second test)
   would silently be handed the first one's store. `_build_vector_store`
   below never calls `get_vector_store` at all: it does the same
   Qdrant-vs-NumPy provider selection `retrieval.dense._build_vector_store`
   does, but builds a fresh instance for THIS catalog/connection every time
   and hands it straight to `Deps.vector_store`. `retrieve()` always
   receives that instance explicitly (`vector_store=deps.vector_store`), so
   the module global in `dense.py` is simply never consulted by anything
   built through this Container. It is not deleted — another task owns that
   file — this module just stops depending on it.

2. **Skill-lock verification at startup.** `build_container` raises if
   `skills-lock.json` no longer matches the skill files on disk. An audit
   record names the skill versions that produced a recommendation
   (`agent.nodes.skill_versions`); if those versions were never actually
   locked, that record is a false provenance claim, which is worse than
   having no audit record at all. Failing loudly at startup, before a
   single request is served, is the only place this can be caught for free.

3. **One write lock for the one shared connection.** Every route is sync
   `def`, so Starlette runs it in a threadpool worker; the Container holds
   exactly ONE `sqlite3.Connection` (`store.db.connect`, `check_same_thread
   =False`), so every in-flight request's writes share it. `build_container`
   creates one `threading.Lock` and hands the SAME instance to every store
   that writes (`AuditLog`, `IncidentLog`, `FeasibilityCache`,
   `EmbeddingCache`, `PreferenceMemory`), which each hold it across their
   own execute()+commit() pair — so two threads' writes can never interleave
   into one merged transaction. See `build_container`'s body for exactly
   what this does and does not guarantee.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Deliberately TYPE_CHECKING-only, not top-level: every real import
    # below (agent.graph, reasoning.llm, retrieval.*, store.*, ...) is
    # loaded lazily, inside the functions that need it, so importing this
    # composition-root module for its `Container`/`build_container` names
    # alone never pays for langgraph/litellm/etc. Types-only imports here
    # keep that lazy-startup property while still giving `Container` and
    # `_build_vector_store` real static types instead of bare `object`.
    # `sqlite3` is a real top-level import above (cheap stdlib, unlike the
    # heavy optional deps this block exists to defer).
    from langgraph.checkpoint.sqlite import SqliteSaver

    from beatroot.agent.async_explain import ExplanationQueue
    from beatroot.agent.graph import MealPlanningAgent
    from beatroot.agent.skills_registry import Skill
    from beatroot.confirm.trust_score import EvalThresholds
    from beatroot.obs.cost import CostLedger
    from beatroot.reasoning.llm import LLMClient
    from beatroot.retrieval.dense import VectorStore
    from beatroot.store.audit import AuditLog
    from beatroot.store.cache import EmbeddingCache, FeasibilityCache
    from beatroot.store.incidents import IncidentLog
    from beatroot.store.preferences import PreferenceMemory
    from beatroot.trusted.catalog import Catalog
    from beatroot.trusted.index import TagIndex

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
SKILLS_DIR = ROOT / "skills"
SKILLS_LOCK_PATH = ROOT / "skills-lock.json"
PROMPTS_DIR = ROOT / "prompts"
THRESHOLDS_PATH = ROOT / "eval" / "thresholds.yaml"
DEFAULT_DB = ROOT / "beatroot.db"


def _provider_name(llm: object) -> str:
    """The short name of whatever is actually answering completions —
    'echo' for the offline stub, otherwise the LiteLLM provider prefix
    ('azure', 'ollama', ...) off the configured model string. A health
    check that can only say 'ok' is decoration; this is the fact a
    reviewer actually wants when something looks wrong."""
    if getattr(llm, "_offline", False):
        return "echo"
    model = getattr(llm, "model", "") or ""
    return model.split("/", 1)[0] if "/" in model else (model or "unknown")


def verify_lock_drift(skills: dict[str, Skill]) -> list[str]:
    """Sorted skill ids whose on-disk content no longer matches
    `skills-lock.json`. Empty means clean."""
    from beatroot.agent.skills_registry import verify_lock

    return verify_lock(skills, SKILLS_LOCK_PATH)


def _build_vector_store(
    llm: LLMClient, catalog: Catalog, embedding_cache: EmbeddingCache | None
) -> VectorStore:
    """Container-owned construction — see module docstring, point 1.

    Mirrors `retrieval.dense._build_vector_store`'s provider selection
    (Qdrant when `QDRANT_URL` is configured, the in-memory NumPy fallback
    otherwise) but, unlike that helper, never touches `dense._STORE` and
    always injects the ONE `EmbeddingCache` this Container owns — so
    `/metrics` reports hit/miss activity from the cache that actually did
    the embedding work, not a second instance nobody ever calls.
    """
    from beatroot.settings import get_settings

    settings = get_settings()
    if settings.qdrant_url:
        from beatroot.retrieval.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            llm, catalog, url=settings.qdrant_url, embedding_cache=embedding_cache
        )

    from beatroot.retrieval.dense import DenseIndex

    return DenseIndex(llm, catalog, embedding_cache=embedding_cache)


def _build_checkpointer(db_path: Path) -> SqliteSaver:
    """A durable, file-backed checkpointer — never `MealPlanningAgent`'s
    in-memory default (`agent.graph._default_checkpointer`), which a fresh
    `sqlite3.connect(":memory:", ...)` per Container and dies with it.

    Without this, a thread that pauses in one process — a CLI `beatroot
    recommend` invocation, say — is unrecoverable by `.resume()` in any
    OTHER process, including a separate `beatroot resume` invocation: the
    only way the CLI is ever actually used, since each Typer command is its
    own `build_container()` call and its own process lifetime. The API
    doesn't strictly need this (`lifespan` builds one Container for the
    whole server process, so `/recommend` and `/resume` already share one
    agent's in-memory checkpointer) but gets it anyway for the same
    resume-after-restart durability `SqliteSaver` exists for in the first
    place.

    Lives on ITS OWN connection, a sibling file next to the main db —
    deliberately not sharing `conn` above. `SqliteSaver` already manages
    its own internal `threading.Lock` around whatever connection it gets;
    sharing `conn` would put that lock and `db_lock` (point 3) guarding two
    unrelated write paths on the SAME connection, which is exactly the kind
    of un-synchronized-pair-of-writers bug this fix round just closed.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    checkpoint_path = db_path.with_name(f"{db_path.stem}.checkpoints.db")
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    return SqliteSaver(conn)


@dataclass
class Container:
    """Composition root. Everything is constructed once, here, and injected.

    Nothing else in the codebase reaches for a global. A test swaps any
    dependency by constructing a Container, not by monkeypatching imports.
    """

    conn: sqlite3.Connection
    catalog: Catalog
    llm: LLMClient
    vector_store: VectorStore
    tag_index: TagIndex
    incidents: IncidentLog
    audit: AuditLog
    preferences: PreferenceMemory
    # Always set by build_container(); non-optional here (unlike
    # agent.nodes.Deps.feasibility_cache, which stays optional to support a
    # future cache-less deployment) because every consumer of Container —
    # /metrics included — reads it unconditionally.
    feasibility_cache: FeasibilityCache
    embedding_cache: EmbeddingCache
    skills: dict[str, Skill]
    thresholds: EvalThresholds
    agent: MealPlanningAgent
    # Task 19: process-wide cost accounting across every plan this
    # Container serves (not the single-plan CostRecord each terminal
    # result already carries) — route handlers fold each result's
    # CostRecord into this ledger so /metrics can report a real,
    # accumulating cost-per-plan rather than a per-request number nobody
    # aggregates.
    cost_ledger: CostLedger
    # Task 23: `None` unless `settings.async_explanation` is on — the SAME
    # instance handed to `agent.nodes.Deps.explanation_queue`, so a route
    # handler (`GET /recommend/{id}/explanation`) reads exactly the job
    # `explain_node` submitted, not a second, disconnected queue.
    explanation_queue: ExplanationQueue | None = None

    def close(self) -> None:
        self.conn.close()
        # `self.agent.checkpointer` is a SECOND, separate connection (see
        # `_build_checkpointer`) — `getattr` because `BaseCheckpointSaver`
        # itself has no `.conn`; only the concrete `SqliteSaver` this
        # Container actually builds does.
        checkpointer_conn = getattr(self.agent.checkpointer, "conn", None)
        if checkpointer_conn is not None:
            checkpointer_conn.close()
        if self.explanation_queue is not None:
            # Don't block process shutdown on in-flight explanations — the
            # recommendations they belong to have already been served
            # (Task 23's whole point); nothing waits on this thread pool
            # to answer a request.
            self.explanation_queue.shutdown(wait=False)

    def health(self) -> dict[str, Any]:
        """Reports what each dependency actually IS, not just 'ok'. A health
        check that cannot tell you which vector store answered, or whether
        the skill rules it ran under are the ones it claims to have run
        under, is decoration.

        Also the ONE place the dashboard (Task 18) can read the live trust
        threshold and weights from — `beatroot.settings`, never a literal
        baked into `web/index.html`. A screen recording showing a stale
        number against a retuned config would be actively misleading, so
        the frontend has no excuse to hardcode what this already exposes.
        """
        from beatroot.obs.tracing import INSTRUMENTATION
        from beatroot.settings import get_settings

        drift = verify_lock_drift(self.skills)
        settings = get_settings()
        trust = settings.trust
        return {
            "tracing": {
                # Whether credentials resolved at all — NOT whether spans are
                # arriving. Those are different claims and conflating them is
                # how tracing stays dark while every check reads healthy.
                "langfuse_configured": settings.obs.langfuse_enabled,
                "host": settings.obs.langfuse_host or None,
                "instrumentation": INSTRUMENTATION,
            },
            "status": "ok",
            "provider": _provider_name(self.llm),
            "llm_model": getattr(self.llm, "model", "offline"),
            "vector_store": getattr(self.vector_store, "name", "unknown"),
            "recipes": len(self.catalog.recipes()),
            "skills": len(self.skills),
            "skills_locked": not drift,
            "trust": {
                "refusal_threshold": trust.refusal_threshold,
                "weights": trust.weights.model_dump(),
            },
        }


def build_container(
    db_path: Path | None = None,
    seed_data: bool = True,
    skills_dir: Path | None = None,
    async_explanation: bool | None = None,
) -> Container:
    """Construct every dependency once and wire them together.

    Raises `RuntimeError` if the skills on disk no longer match
    `skills-lock.json` — see the module docstring, point 2. This check runs
    before anything expensive (embedding the catalog, opening a vector
    store) so a stale lock fails fast, at startup, not partway through
    serving a request.

    `skills_dir` defaults to `SKILLS_DIR` (the repo's real `skills/`); a
    test points it at a throwaway copy of the skill files (one deliberately
    mutated) to prove the raise above actually fires — see
    `tests/test_container.py`.
    """
    from beatroot.agent.async_explain import ExplanationQueue
    from beatroot.agent.graph import MealPlanningAgent
    from beatroot.agent.nodes import Deps
    from beatroot.agent.skills_registry import load_skills
    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.obs.cost import CostLedger
    from beatroot.obs.logging import configure_logging
    from beatroot.obs.tracing import configure_observability
    from beatroot.reasoning.llm import get_llm_client
    from beatroot.settings import get_settings
    from beatroot.store.audit import AuditLog
    from beatroot.store.cache import EmbeddingCache, FeasibilityCache
    from beatroot.store.db import connect, seed
    from beatroot.store.incidents import IncidentLog
    from beatroot.store.preferences import PreferenceMemory
    from beatroot.trusted.catalog import Catalog
    from beatroot.trusted.index import TagIndex

    # Task 19: both the CLI (every Typer command) and the API (the
    # `lifespan` handler) reach this function exactly once per process, so
    # this is the one place `configure_logging`/`configure_observability`
    # need to be called from — never at import time, which would make test
    # output depend on import order. Both are idempotent no-ops on a repeat
    # call and clean no-ops with no Langfuse credentials configured.
    configure_logging()
    configure_observability()

    skills = load_skills(skills_dir or SKILLS_DIR)
    drift = verify_lock_drift(skills)
    if drift:
        raise RuntimeError(
            "skills-lock.json does not match the skill files on disk for: "
            f"{', '.join(drift)}. An audit record naming these skill "
            "versions would be a false provenance claim. Regenerate the "
            "lock (skills_registry.write_lock) before starting beatroot."
        )

    resolved_db_path = db_path or DEFAULT_DB
    conn = connect(resolved_db_path)

    # ONE lock, owned here, handed to every store that writes through the
    # ONE shared `conn` above — see the module docstring, point 3, and
    # `store.db.connect`'s docstring for why a lock is needed at all now
    # that `check_same_thread=False` lets a threadpool worker use `conn`.
    # Every write path (AuditLog.record, IncidentLog.record,
    # FeasibilityCache.put, EmbeddingCache.put, PreferenceMemory.record,
    # store.db.seed's own FeasibilityCache.invalidate_all) holds this SAME
    # Lock across its own execute()+commit() pair, so two requests running
    # concurrently in Starlette's threadpool can never interleave: whichever
    # thread acquires the lock first fully commits before the next thread's
    # write is even issued. This restores atomicity — it does not make
    # `conn` itself safe for concurrent USE beyond that (see the docstring
    # for what is still not guaranteed). Constructed BEFORE `seed()` runs
    # (rather than after, as a plain "nothing else is using `conn` yet"
    # afterthought) precisely so `seed()` never has to fall back to a
    # private, unshared lock of its own for the one write it does.
    db_lock = threading.Lock()

    if seed_data:
        seed(conn, DATA_DIR, lock=db_lock)

    catalog = Catalog(conn)
    llm = get_llm_client()
    embedding_cache = EmbeddingCache(conn, lock=db_lock)
    vector_store = _build_vector_store(llm, catalog, embedding_cache)
    tag_index = TagIndex(catalog.recipes())
    incidents = IncidentLog(conn, lock=db_lock)
    audit = AuditLog(conn, lock=db_lock)
    feasibility_cache = FeasibilityCache(conn, lock=db_lock)
    preferences = PreferenceMemory(conn, lock=db_lock)

    # Task 23: one queue, owned here, shared by `Deps.explanation_queue`
    # (what `explain_node` submits to) and `Container.explanation_queue`
    # (what `GET /recommend/{id}/explanation` reads from) — never two
    # separate instances that could disagree about a job's status. `None`
    # unless explicitly turned on; see `settings.Settings.async_explanation`
    # for why this defaults off.
    # `async_explanation=False` forces the SYNCHRONOUS path regardless of
    # config. The eval runners need this: A6 (explanation_grounding) is
    # scored from `verify_node` diffing the explanation's numbers against
    # catalog truth, and under async that explanation is still "" at VERIFY,
    # so every drift_bait case passes unconditionally and A6 reports a
    # meaningless 1.000 (see `eval.runners.system.run_system`'s guard).
    #
    # An explicit parameter rather than an env var: `settings.py` is the only
    # module permitted to read the environment, enforced by an AST test
    # (`tests/test_settings.py`). Poking `os.environ` from a runner to force
    # this was the first attempt and that test correctly rejected it.
    use_async = get_settings().async_explanation if async_explanation is None else async_explanation
    # One ledger, shared by the request handlers and by the explanation
    # queue's worker — the queue spends after the response has been sent, so
    # without this hand-off `/metrics` under-reports every COMMIT by the
    # whole explanation call.
    cost_ledger = CostLedger()
    explanation_queue = ExplanationQueue(llm, on_complete=cost_ledger.fold) if use_async else None

    deps = Deps(
        catalog=catalog,
        llm=llm,
        vector_store=vector_store,
        tag_index=tag_index,
        incidents=incidents,
        audit=audit,
        skills=skills,
        preferences=preferences,
        feasibility_cache=feasibility_cache,
        explanation_queue=explanation_queue,
    )

    return Container(
        conn=conn,
        catalog=catalog,
        llm=llm,
        vector_store=vector_store,
        tag_index=tag_index,
        incidents=incidents,
        audit=audit,
        preferences=preferences,
        feasibility_cache=feasibility_cache,
        embedding_cache=embedding_cache,
        skills=skills,
        thresholds=load_thresholds(THRESHOLDS_PATH),
        agent=MealPlanningAgent(deps, checkpointer=_build_checkpointer(resolved_db_path)),
        cost_ledger=cost_ledger,
        explanation_queue=explanation_queue,
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Process-wide singleton over `build_container()`'s defaults. Exists
    for callers (e.g. a REPL, a script) that want the default, shared
    Container without wiring one up by hand — the API and CLI entrypoints
    do not use this: they each build (or receive) their own Container so
    they can be pointed at a different `db_path` in tests."""
    return build_container()
