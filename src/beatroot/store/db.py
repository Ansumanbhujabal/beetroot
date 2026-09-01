"""SQLite connection and schema management for the catalog and audit stores."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import yaml

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingredients (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, cuisine TEXT,
    prep_minutes INTEGER, payload TEXT NOT NULL, tags TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5(
    id UNINDEXED, name, cuisine, ingredient_names, tags
);
CREATE TABLE IF NOT EXISTS audit (
    id TEXT PRIMARY KEY, profile_id TEXT, terminal_state TEXT,
    payload TEXT, skill_versions TEXT, cost_usd REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY, kind TEXT, profile_id TEXT,
    detail TEXT, payload TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS preferences (
    profile_id TEXT, tag TEXT, affinity REAL,
    PRIMARY KEY (profile_id, tag)
);
CREATE TABLE IF NOT EXISTS feasibility_cache (
    fingerprint TEXT PRIMARY KEY, recipe_ids TEXT NOT NULL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY, vector TEXT NOT NULL, created_at TEXT
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    # check_same_thread=False: Task 12's Container builds one connection at
    # startup and shares it across every request. A sync FastAPI route runs
    # in a starlette threadpool worker thread, never the thread that called
    # `connect()`, so the default same-thread check would raise a
    # ProgrammingError on literally the first request.
    #
    # That alone does NOT make concurrent writers safe: two threadpool
    # workers each doing execute() then a separate commit() on this one
    # connection, with nothing serializing the pair, can interleave — one
    # thread's commit() can land both threads' writes in a single merged
    # transaction. `container.build_container` closes that gap by handing
    # every write path (`AuditLog`, `IncidentLog`, `FeasibilityCache`,
    # `EmbeddingCache`) the SAME `threading.Lock`, held across each
    # execute()+commit() pair — see the container module docstring, point
    # 3. `connect()` itself stays a plain single connection; the lock lives
    # with whoever wires the stores together, not here.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def seed(conn: sqlite3.Connection, data_dir: Path, lock: threading.Lock | None = None) -> None:
    """Idempotent. Derives recipe tags at seed time — never hand-set.

    Also invalidates the feasibility cache: its entries hold recipe ids, and
    a reseed can change what those ids mean (or remove them outright).

    `lock` should be the SAME `threading.Lock` `container.build_container`
    hands to every other store writing through this `conn`
    (`FeasibilityCache.invalidate_all` holds it across its own
    execute()+commit() pair, exactly like every other write path) —
    passed here rather than constructing a private `FeasibilityCache(conn)`
    so this write is never a silent bypass of the one lock every other
    writer on this connection honours. Defaults to `None` (a private,
    unshared `Lock`) for callers with no shared lock to pass — every
    existing caller, this module's own test suite included, seeds a
    single-threaded, freshly-`connect()`ed connection with nothing else
    writing to it yet, so a private lock is exactly as safe there as no
    lock at all.
    """
    import json

    from beatroot.store.cache import FeasibilityCache
    from beatroot.t0_invariants.nutrition_math import FIELDS, _usable
    from beatroot.trusted.tags import derive_recipe_tags

    ingredients = yaml.safe_load((data_dir / "ingredients.yaml").read_text())
    recipes = yaml.safe_load((data_dir / "recipes.yaml").read_text())
    by_id = {i["id"]: i for i in ingredients}

    for ing in ingredients:
        # Validate per_100g has all required fields with usable values
        per100 = ing.get("per_100g")
        if per100 is not None:
            missing = [f for f in FIELDS if f not in per100]
            unusable = {
                f: per100.get(f) for f in FIELDS if f in per100 and not _usable(per100.get(f))
            }
            if missing or unusable:
                msg = f"Ingredient '{ing['id']}' has incomplete per_100g:"
                if missing:
                    msg += f" missing {missing}"
                if unusable:
                    msg += f" unusable {unusable}"
                raise ValueError(msg)
        conn.execute(
            "INSERT OR REPLACE INTO ingredients (id, name, payload) VALUES (?,?,?)",
            (ing["id"], ing["name"], json.dumps(ing)),
        )

    for rec in recipes:
        tags = sorted(derive_recipe_tags(rec, by_id))
        names = " ".join(by_id[r["ingredient_id"]]["name"] for r in rec["ingredients"])
        conn.execute(
            "INSERT OR REPLACE INTO recipes (id, name, cuisine, prep_minutes, payload, tags)"
            " VALUES (?,?,?,?,?,?)",
            (
                rec["id"],
                rec["name"],
                rec.get("cuisine"),
                rec.get("prep_minutes"),
                json.dumps(rec),
                json.dumps(tags),
            ),
        )
        conn.execute("DELETE FROM recipes_fts WHERE id = ?", (rec["id"],))
        conn.execute(
            "INSERT INTO recipes_fts (id, name, cuisine, ingredient_names, tags)"
            " VALUES (?,?,?,?,?)",
            (rec["id"], rec["name"], rec.get("cuisine", ""), names, " ".join(tags)),
        )
    conn.commit()
    FeasibilityCache(conn, lock=lock).invalidate_all()
