"""Preference memory: the self-learning half of Feature 2. Spec §9.

Deliberately deterministic and inspectable — a per-tag affinity updated by
exponential moving average (EMA) toward +1.0 on acceptance or -1.0 on
rejection, never a trained model. There is no gradient, no drift, and no
opaque internal state: the weights are just numbers in the `preferences`
table, one row per `(profile_id, tag)`, and can be printed and explained to
a user at any time. The same feedback, applied in the same order, always
produces the same numbers. That is the property this approach was chosen
for over anything learned.

An EMA step is a convex combination of a prior already in `[-1, 1]` and a
target of exactly `+-1.0` — `updated = (1 - alpha) * prior + alpha *
target` always lies between `prior` and `target`, so repeated updates move
monotonically toward the bound and can never cross or exceed it.

`affinity()` feeds `retrieval.rerank.retrieve()` as a THIRD ranking,
weighted by `settings.retrieval.affinity_weight` in RRF fusion, over
candidates that ranking has already proven legal — affinity can only
reorder that set, never add to it. See
`tests/retrieval/test_pipeline.py::test_affinity_cannot_promote_an_illegal_candidate`
for the test that pins this safety property down.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable


class PreferenceMemory:
    """Per-profile, per-tag affinity in `[-1.0, 1.0]`, updated by EMA."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        alpha: float | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        from beatroot.settings import get_settings

        self.conn = conn
        self.alpha = alpha if alpha is not None else get_settings().preferences.ema_alpha
        # Same Container-owned lock every other writer through this shared
        # `conn` takes (`store.audit.AuditLog`, `store.incidents.IncidentLog`,
        # `store.cache._Cache`) — held across the execute()+commit() pair so
        # two threadpool workers' writes can never interleave into one
        # merged transaction. A standalone `PreferenceMemory(conn)` still
        # gets a private Lock, safe on its own but not cross-object-shared.
        self._lock = lock or threading.Lock()

    def record(self, profile_id: str, tags: Iterable[str], accepted: bool) -> None:
        """Move every tag in `tags` one EMA step toward +1 (accepted) or -1
        (rejected) for this profile. Profiles are isolated: this only ever
        reads and writes rows keyed by `profile_id`."""
        target = 1.0 if accepted else -1.0
        # Read-modify-write, both under the SAME lock as the write below —
        # not just the write. Computing the EMA step here needs the prior
        # value read and the new value written to be one atomic unit: two
        # `record()` calls racing on the same profile/tag must not both
        # read the same prior and silently drop one update (a lost-update
        # race). Calls `_affinity_locked` (not `affinity()`) so this thread
        # doesn't try to re-acquire its own lock — `threading.Lock` is not
        # reentrant, and `affinity()` now takes the same lock on every read
        # (see its docstring) so a naive call here would deadlock.
        with self._lock:
            current = self._affinity_locked(profile_id)
            for tag in tags:
                prior = current.get(tag, 0.0)
                updated = (1 - self.alpha) * prior + self.alpha * target
                self.conn.execute(
                    "INSERT INTO preferences (profile_id, tag, affinity) VALUES (?,?,?)"
                    " ON CONFLICT(profile_id, tag) DO UPDATE SET affinity = excluded.affinity",
                    (profile_id, tag, updated),
                )
            self.conn.commit()

    def affinity(self, profile_id: str) -> dict[str, float]:
        """Tag affinities for this profile only — an unknown profile (or
        one with no feedback yet) returns an empty mapping, never another
        profile's data."""
        # Takes the SAME shared lock every writer through this `conn`
        # holds. sqlite3 does not tolerate simultaneous unlocked
        # `execute()` calls from different threads on one connection at
        # all — that is precisely the bug `record()`'s lock was introduced
        # to fix, and an unlocked SELECT here could still race a writer's
        # execute()+commit() pair into the same InterfaceError. A plain
        # SELECT is fast enough that holding the lock for it is not a
        # measured cost on the hot path (uncontended-lock overhead was
        # measured at ~5ns/call, ~0.5% relative, against store.cache's
        # identical pattern).
        with self._lock:
            return self._affinity_locked(profile_id)

    def _affinity_locked(self, profile_id: str) -> dict[str, float]:
        """The actual `SELECT`, assuming the caller already holds `_lock`."""
        rows = self.conn.execute(
            "SELECT tag, affinity FROM preferences WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()
        return {r["tag"]: float(r["affinity"]) for r in rows}
