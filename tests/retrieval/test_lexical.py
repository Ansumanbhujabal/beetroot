from pathlib import Path

import pytest

from beatroot.retrieval.lexical import lexical_search
from beatroot.store.db import connect, seed

DATA = Path(__file__).parents[2] / "data"


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    seed(c, DATA)
    return c


def test_matches_something_real(conn):
    results = lexical_search(conn, "rice")
    assert results
    ids = [r for r, _ in results]
    assert "rec_jeera_rice" in ids


def test_scores_are_descending_and_conventional(conn):
    results = lexical_search(conn, "rice")
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_exclude_tags_removes_matching_rows_via_not_clause(conn):
    """FTS5 NOT clause pushdown against the real seeded catalog — not a mock.
    `peanut` is a real allergen tag on real recipes in data/recipes.yaml."""
    unfiltered = {r for r, _ in lexical_search(conn, "rice")}
    filtered = {r for r, _ in lexical_search(conn, "rice", exclude_tags=["peanut"])}
    assert filtered < unfiltered
    assert filtered  # still finds legal matches

    # Verify none of the surviving rows actually carry the excluded tag.
    # Safe despite the templated SQL: only the `?` placeholder COUNT is
    # interpolated (never a value), and every actual value is still bound
    # through the parameterized `list(filtered)` below.
    placeholders = ",".join("?" * len(filtered))
    rows = conn.execute(
        f"SELECT id, tags FROM recipes WHERE id IN ({placeholders})",  # noqa: S608
        list(filtered),
    ).fetchall()
    for row in rows:
        assert "peanut" not in row["tags"]


def test_exclude_multiple_tags_with_underscore(conn):
    """`root_vegetable` exercises a tag containing an underscore — FTS5
    boolean syntax is fussy, so this is run against the real database rather
    than assumed to work."""
    unfiltered = {r for r, _ in lexical_search(conn, "rice")}
    filtered = {
        r for r, _ in lexical_search(conn, "rice", exclude_tags=["peanut", "root_vegetable"])
    }
    assert filtered <= unfiltered
    assert filtered < unfiltered


def test_empty_query_returns_empty(conn):
    assert lexical_search(conn, "") == []
    assert lexical_search(conn, "   ") == []


def test_empty_exclusion_list_behaves_like_no_exclusion(conn):
    assert lexical_search(conn, "rice", exclude_tags=[]) == lexical_search(conn, "rice")


def test_query_matching_nothing_returns_empty(conn):
    assert lexical_search(conn, "zzzznonexistentzzzz") == []


def test_limit_is_respected(conn):
    assert len(lexical_search(conn, "rice", limit=2)) <= 2
