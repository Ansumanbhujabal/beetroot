"""Smoke tests for the `/` (recommend) page. Task 18, spec §15; PAGES task.

The page itself is one self-contained file (`beatroot/web/index.html`) with
no build step and no external network calls — the whole project's
premise is that it runs offline. These tests assert exactly that plus the
four terminal-state names a viewer must be able to see without reading any
code: `COMMIT`, `NEGOTIATE`, `ESCALATE`, and `PENDING_REVIEW` (the medical
review gate — not one of the three real terminals, but a state the UI must
still name plainly rather than hide).

The incident feed and drift ledger moved off this page to `/incidents`
(`tests/api/test_pages.py` covers that page and `/evals`).
"""

from fastapi.testclient import TestClient

from beatroot.api.main import app

client = TestClient(app)


def test_index_serves_and_is_self_contained():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<html" in body.lower()
    for external in (
        "http://cdn",
        "https://cdn",
        "unpkg",
        "jsdelivr",
        "googleapis",
        "gstatic",
        "http://",
        "https://",
    ):
        assert external not in body, f"dashboard must work offline, found {external!r}"


def test_index_surfaces_all_four_state_names():
    body = client.get("/").text
    for state in ("COMMIT", "NEGOTIATE", "ESCALATE", "PENDING_REVIEW"):
        assert state in body


def test_index_labels_the_free_text_box_as_untrusted():
    body = client.get("/").text
    assert "untrusted" in body.lower()


def test_index_shows_locked_constraints_never_as_clickable_options():
    body = client.get("/").text
    assert "never offered" in body.lower()


def test_index_reads_trust_config_from_health_not_a_literal():
    """Fix round: the refusal threshold and 75/25 weight labels must come
    from `GET /health` (which itself reads `beatroot.settings`), never a
    hardcoded number — a stale value on a screen recording is worse than
    the number being absent. `0.55` is the config's current value; if it
    is ever baked back into the page as a literal, this fails the moment
    the config is retuned without a matching page edit."""
    body = client.get("/").text
    assert 'api("/health")' in body
    assert "0.55" not in body
    assert "trustConfig" in body


def test_index_points_to_the_incidents_and_evals_pages():
    """PAGES task: the incident feed and drift ledger moved to `/incidents`
    (see test_pages.py), and eval results/logs live on `/evals` — this page
    must still say where to find them rather than going silent."""
    body = client.get("/").text
    assert '/incidents"' in body
    assert '/evals"' in body


def test_index_shows_the_free_text_safety_boundary_and_parsed_constraints():
    """FREE-TEXT/NLP task: the free-text box must say plainly that it only
    ever produces SOFT preferences and can never lift a medical/religious
    rule, and the page must render which constraints came from it
    (`source: parsed_free_text`) distinctly from ones set explicitly."""
    body = client.get("/").text
    assert "parsed_free_text" in body
    assert "soft preference" in body.lower()
    assert "from free text" in body.lower()


def test_index_has_a_query_rewrite_panel():
    """QUERY REWRITE task: both the original and rewritten query must be
    shown, so the rewrite's effect (or lack of one) is visible."""
    body = client.get("/").text
    assert "query rewrite" in body.lower()
    assert "no expansion applied" in body.lower()


def test_index_offers_the_preset_profile_dropdown():
    """PRESET PROFILES task: picking a preset loads it into the SAME
    builder — GET /profiles is fetched, never a hardcoded list."""
    body = client.get("/").text
    assert 'api("/profiles")' in body
    assert "presetSelect" in body


def test_builder_offers_every_registered_constraint_kind():
    """The dashboard must be able to express what the engine can enforce.

    Three kinds were missing — `require_tag`, `require_any_tag` and
    `exclude_cuisine` — and they are not incidental ones: the first two are
    the allowlist primitives that exist because a denylist served chicken to
    a vegan, and the third exists because a missing "not this cuisine"
    primitive made the compiler emit an affinity TOWARD the disliked cuisine.
    A UI that cannot express the fixes is a UI that cannot demonstrate them.
    """
    from beatroot.t0_invariants.constraints import registered_kinds

    body = client.get("/").text
    missing = [k for k in registered_kinds() if f">{k}<" not in body]
    assert not missing, f"constraint kinds the builder cannot express: {missing}"


def test_builder_severity_options_match_the_contract():
    """Every `Severity` must be selectable, and the page's own hard-severity
    set must mirror `HARD_SEVERITIES`.

    `dietary` was absent from both, which drew a constraint the engine
    enforces as hard as though it were soft — the same class of mismatch,
    one layer up, as the bug that caused `dietary` to be introduced.
    """
    from beatroot.contracts.core import HARD_SEVERITIES, Severity

    body = client.get("/").text
    missing = [s.value for s in Severity if f'value="{s.value}"' not in body]
    assert not missing, f"severities the builder cannot select: {missing}"

    hard_line = next(line for line in body.splitlines() if "const HARD" in line)
    for severity in HARD_SEVERITIES:
        assert severity.value in hard_line, f"{severity.value} missing from the page's HARD set"
