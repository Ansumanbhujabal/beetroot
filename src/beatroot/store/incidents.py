"""Incident log. Spec §9.

Every escalation, refusal, drift detection and infeasibility lands here.
This is the input side of the healing loop (Task 16).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from beatroot.contracts.result import Incident


class IncidentLog:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock | None = None) -> None:
        self.conn = conn
        # See `AuditLog.__init__` — same reasoning, same Container-owned
        # lock shared across every store writing this one `conn`.
        self._lock = lock or threading.Lock()

    def record(
        self, kind: str, profile_id: str, detail: str, payload: dict[str, Any] | None = None
    ) -> Incident:
        inc = Incident(
            id=str(uuid.uuid4()),
            kind=kind,
            profile_id=profile_id,
            detail=detail,
            payload=payload or {},
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self.conn.execute(
                "INSERT INTO incidents (id, kind, profile_id, detail, payload, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    inc.id,
                    inc.kind,
                    inc.profile_id,
                    inc.detail,
                    json.dumps(inc.payload),
                    inc.created_at.isoformat(),
                ),
            )
            self.conn.commit()
        return inc

    def all(self) -> list[Incident]:
        rows = self.conn.execute("SELECT * FROM incidents ORDER BY created_at").fetchall()
        return [
            Incident(
                id=r["id"],
                kind=r["kind"],
                profile_id=r["profile_id"],
                detail=r["detail"],
                payload=json.loads(r["payload"]),
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
