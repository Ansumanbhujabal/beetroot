"""Concurrent-write safety for the one shared `sqlite3.Connection` a
Container hands to every store. Spec §15, task-12 review round 1.

Every route is sync `def`, so Starlette runs each one in a threadpool
worker thread; the Container holds exactly ONE connection, shared across
every in-flight request. `AuditLog.record`, `IncidentLog.record`,
`FeasibilityCache.put` and `EmbeddingCache.put` each do `execute()` then a
separate `commit()` — with no lock around the pair, two threads' writes can
interleave and merge into a single transaction instead of committing
independently (proved directly, without any of this project's code, in the
fix-round report for this task: thread A's own `rollback()` discarded
thread B's unrelated, already-committed write, because B's `commit()` had
swept A's still-open insert in with its own). `container.build_container`
now hands the SAME `threading.Lock` to every store that writes the shared
connection; these tests exercise that fix through real threads, the same
way Starlette's threadpool does, not through the graph or the API.

These are deterministic, not flaky-by-construction: the lock forces full
serialization, so the assertions below hold on every run regardless of
however the OS happens to schedule the threads — they are not trying to
reproduce the race itself (that needs artificial delays to force a specific
interleaving, which is not something a fast unit test should carry), only
to prove concurrent load through the fixed code no longer loses or
corrupts a single write.
"""

from concurrent.futures import ThreadPoolExecutor

from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.store.audit import AuditLog
from beatroot.store.cache import EmbeddingCache, FeasibilityCache
from beatroot.store.db import connect
from beatroot.store.incidents import IncidentLog

N_THREADS = 40


def test_concurrent_audit_writes_through_a_threadpool_lose_nothing(tmp_path):
    conn = connect(tmp_path / "concurrency.db")
    audit = AuditLog(conn)

    def write(i: int) -> str:
        return audit.record(f"profile-{i}", "COMMIT", {"i": i}, {"s": "v"}, 0.0)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        ids = list(pool.map(write, range(N_THREADS)))

    assert len(ids) == N_THREADS
    assert len(set(ids)) == N_THREADS, "every write must get its own id, none dropped or merged"

    rows = audit.all()
    assert len(rows) == N_THREADS
    profile_ids = {r["profile_id"] for r in rows}
    assert profile_ids == {f"profile-{i}" for i in range(N_THREADS)}


def test_concurrent_writes_across_every_store_sharing_one_connection_and_lock(tmp_path):
    """Mirrors `container.build_container`'s wiring directly: one
    connection, one `threading.Lock`, four stores that all write through
    it, hit concurrently — not just one store in isolation."""
    import threading

    conn = connect(tmp_path / "concurrency.db")
    lock = threading.Lock()
    audit = AuditLog(conn, lock=lock)
    incidents = IncidentLog(conn, lock=lock)
    feasibility = FeasibilityCache(conn, lock=lock)
    embedding = EmbeddingCache(conn, lock=lock)

    def audit_write(i: int) -> None:
        audit.record(f"p{i}", "COMMIT", {"i": i}, {}, 0.0)

    def incident_write(i: int) -> None:
        incidents.record("infeasible", f"p{i}", f"detail-{i}")

    def feasibility_write(i: int) -> None:
        cs = ConstraintSet(
            profile_id="p",
            constraints=[
                Constraint(id="c", kind="exclude_tag", severity=Severity.PREFERENCE, value=f"t{i}")
            ],
        )
        feasibility.put(cs, [f"rec_{i}"])

    def embedding_write(i: int) -> None:
        embedding.put("test-model", f"text-{i}", [float(i)])

    jobs = (
        [(audit_write, i) for i in range(N_THREADS)]
        + [(incident_write, i) for i in range(N_THREADS)]
        + [(feasibility_write, i) for i in range(N_THREADS)]
        + [(embedding_write, i) for i in range(N_THREADS)]
    )

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(fn, i) for fn, i in jobs]
        for f in futures:
            f.result()  # re-raises anything a worker thread raised

    assert len(audit.all()) == N_THREADS
    assert len(incidents.all()) == N_THREADS
    row_count = conn.execute("SELECT COUNT(*) AS n FROM feasibility_cache").fetchone()["n"]
    assert row_count == N_THREADS
    row_count = conn.execute("SELECT COUNT(*) AS n FROM embedding_cache").fetchone()["n"]
    assert row_count == N_THREADS
