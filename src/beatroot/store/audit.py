"""Audit log. Spec §13.

Every terminal state writes one audit record naming the skill versions that
produced it, so a recommendation or a refusal is replayable against the
exact rules that generated it — a skill digest from the Task 10 lock, never
a version string typed by hand.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any


class AuditLog:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock | None = None) -> None:
        self.conn = conn
        # Shared with every other store the same Container built against
        # this `conn` (see `container.build_container`) so their execute()
        # +commit() pairs can never interleave on the one connection every
        # threadpool-worker request shares. Standalone construction (tests
        # that call `AuditLog(conn)` directly) still gets a private Lock,
        # which is safe — it just isn't cross-object-shared, which matters
        # only when several store objects write the same `conn` at once.
        self._lock = lock or threading.Lock()

    def record(
        self,
        profile_id: str,
        terminal_state: str,
        payload: dict[str, Any],
        skill_versions: dict[str, str],
        cost_usd: float,
    ) -> str:
        aid = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                "INSERT INTO audit (id, profile_id, terminal_state, payload,"
                " skill_versions, cost_usd, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    aid,
                    profile_id,
                    terminal_state,
                    json.dumps(payload, default=str),
                    json.dumps(skill_versions),
                    cost_usd,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self.conn.commit()
        return aid

    def get(self, audit_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM audit WHERE id = ?", (audit_id,)).fetchone()
        return dict(row) if row else None

    def all(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM audit ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]
