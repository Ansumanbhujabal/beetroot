"""SQLite FTS5 lexical search — BM25 with zero extra dependencies.

Constraint filtering is pushed DOWN into the query itself: excluded tags
become a `NOT (...)` clause inside the FTS5 MATCH expression, so an excluded
row is never scored or ranked, let alone filtered out afterwards. Spec §10.
"""

import sqlite3

from beatroot.settings import get_settings


def _sanitize_terms(text: str) -> list[str]:
    """Strip FTS5 operator punctuation out of free text, keeping only
    alphanumeric tokens. Prevents user query text from being interpreted as
    FTS5 query syntax (AND/OR/NOT, quoting, column filters, ...)."""
    return [t for t in "".join(c if c.isalnum() else " " for c in text).split() if t]


def _quote(tag: str) -> str:
    """Quote an FTS5 string literal, doubling any embedded `"` per FTS5's
    escaping rule. Tag values come from constraint data, not from our own
    controlled vocabulary elsewhere, so they are never trusted to be
    syntax-safe tokens on their own."""
    return '"' + tag.replace('"', '""') + '"'


def lexical_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int | None = None,
    exclude_tags: list[str] | None = None,
) -> list[tuple[str, float]]:
    """FTS5 MATCH against `recipes_fts`, ranked by SQLite's built-in bm25().

    `bm25()` returns lower-is-better; it is negated here so callers get the
    conventional higher-is-better score used everywhere else in retrieval.

    `exclude_tags` compiles hard-constraint exclusions into the MATCH
    expression as a `NOT (...)` clause — the row is never scored, never
    ranked, and never appears in the result set. This is the lexical half of
    "constraint filtering is pushed down into every store, never applied
    around it." Spec §10.
    """
    if limit is None:
        limit = get_settings().retrieval.candidate_limit

    terms = _sanitize_terms(query)
    if not terms:
        return []

    match = " OR ".join(terms)
    if exclude_tags:
        excluded = " OR ".join(_quote(t) for t in exclude_tags)
        match = f"({match}) NOT ({excluded})"

    rows = conn.execute(
        "SELECT id, bm25(recipes_fts) AS score FROM recipes_fts"
        " WHERE recipes_fts MATCH ? ORDER BY score LIMIT ?",
        (match, limit),
    ).fetchall()
    return [(r["id"], -float(r["score"])) for r in rows]
