"""Smoke tests for the `/incidents` and `/evals` pages, plus the JSON data
routes that back them (`/api/incidents`, `/profiles`, `/evals/summary`,
`/evals/logs`). PAGES / PRESET PROFILES / EVALS PAGE tasks.

Each page is one self-contained file with no build step and no external
network calls — the same offline premise `test_index.py` already asserts
for `/`.

Uses the module-scoped `client`/`test_container` fixtures from
`tests/api/conftest.py` (dependency-override, never a bare
`TestClient(app)`) — the `app` object is a process-wide singleton shared
across every test MODULE, and a bare client hitting a container-dependent
route can pick up a stale `app.state.container` left behind by
`test_lifespan.py`'s real `with TestClient(app) as c:` entry elsewhere in
the session. The HTML-only page routes don't strictly need a container, but
using the same fixture everywhere keeps this file from being a landmine the
next person copies.
"""

import logging
from typing import get_args

from beatroot.contracts.core import ConstraintKind, Severity

_EXTERNAL = (
    "http://cdn",
    "https://cdn",
    "unpkg",
    "jsdelivr",
    "googleapis",
    "gstatic",
    "http://",
    "https://",
)


def _assert_self_contained(body: str) -> None:
    assert "<html" in body.lower()
    for external in _EXTERNAL:
        assert external not in body, f"page must work offline, found {external!r}"


def test_incidents_page_serves_and_is_self_contained(client):
    r = client.get("/incidents")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    _assert_self_contained(r.text)


def test_incidents_page_has_a_kind_filter_and_drift_ledger(client):
    body = client.get("/incidents").text
    assert "kindfilter" in body.lower()
    assert "drift ledger" in body.lower()
    assert "no nutrition drift detected" in body.lower()
    assert "/api/incidents" in body


def test_evals_page_serves_and_is_self_contained(client):
    r = client.get("/evals")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    _assert_self_contained(r.text)


def test_evals_page_says_logs_are_bounded_not_full_history(client):
    body = client.get("/evals").text
    assert "not full history" in body.lower()
    assert "/evals/summary" in body
    assert "/evals/logs" in body


def test_shared_nav_present_on_every_page(client):
    """Four pages now — /docs joined them. Kept as an explicit tuple rather
    than discovered, so ADDING a page without wiring it into the nav is a
    deliberate choice someone has to make here, not an omission."""
    for path in ("/", "/incidents", "/evals", "/docs"):
        body = client.get(path).text
        assert '<nav class="topnav">' in body
        assert 'href="/"' in body
        assert 'href="/incidents"' in body
        assert 'href="/evals"' in body
        assert 'href="/docs"' in body


def test_profiles_endpoint_returns_every_validated_preset(client):
    r = client.get("/profiles")
    assert r.status_code == 200
    profiles = r.json()["profiles"]
    # Explicit count, not len(yaml) — that would be tautological and would not
    # notice a profile silently disappearing. Update it deliberately when a
    # preset is added; 21 as of the Hindu non-vegetarian preset.
    assert len(profiles) == 21
    ids = [p["id"] for p in profiles]
    assert len(ids) == len(set(ids))
    for p in profiles:
        assert p["name"] and p["description"]
        assert isinstance(p["constraints"], list) and p["constraints"]
        for c in p["constraints"]:
            # Derived from the contract, not a hand-copied duplicate of it.
            # These were literal tuples, so adding `require_tag` (the
            # allowlist primitive the vegan-served-chicken fix needed) failed
            # here for no reason beyond the copy having drifted. A test that
            # restates a Literal is a second source of truth that can only
            # ever go stale.
            assert c["kind"] in get_args(ConstraintKind)
            assert c["severity"] in {s.value for s in Severity}

    # The two profiles that exist specifically to demonstrate a refusal
    # must say so in their own description, so a user never reads a
    # NEGOTIATE/ESCALATE result as a bug.
    by_id = {p["id"]: p for p in profiles}
    assert "not a bug" in by_id["deliberately_impossible"]["description"].lower()
    assert "not a bug" in by_id["unverifiable_allergen"]["description"].lower()


def test_profiles_endpoint_matches_data_file_ids(client):
    import yaml

    from beatroot.container import DATA_DIR

    raw = yaml.safe_load((DATA_DIR / "profiles.yaml").read_text())
    file_ids = {p["id"] for p in raw}
    api_ids = {p["id"] for p in client.get("/profiles").json()["profiles"]}
    assert file_ids == api_ids


def test_api_incidents_still_returns_json_data(client):
    r = client.get("/api/incidents")
    assert r.status_code == 200
    assert isinstance(r.json()["incidents"], list)


def _use_scratch_artifact(monkeypatch, tmp_path):
    """Point `eval.artifact.ARTIFACT_PATH` at a throwaway file for the
    duration of one test — `/evals/summary` must never be exercised
    against whatever the real repo's `eval/last_run.json` happens to hold
    (present, absent, or mid-write from a CLI run elsewhere), and a test
    must never leave that real file behind either."""
    from beatroot.eval import artifact

    scratch = tmp_path / "last_run.json"
    monkeypatch.setattr(artifact, "ARTIFACT_PATH", scratch)
    return scratch


def test_evals_summary_responds_fast_with_no_artifact(monkeypatch, tmp_path, client):
    """THE property that broke: `/evals/summary` must never compute
    anything, so it must never be slow — pin the timing budget directly
    rather than only asserting the response shape."""
    import time

    _use_scratch_artifact(monkeypatch, tmp_path)

    started = time.perf_counter()
    r = client.get("/evals/summary")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"/evals/summary took {elapsed:.3f}s with no artifact present"
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_computed"
    assert body["system"] is None
    assert body["components"] is None
    assert body["computed_at"] is None
    assert any("eval system" in cmd for cmd in body["commands"])
    assert any("eval components" in cmd for cmd in body["commands"])


def test_evals_summary_responds_fast_when_populated(monkeypatch, tmp_path, client):
    """Same timing budget once a real artifact IS present — reading and
    serialising a small JSON file must never approach the >100s the old
    in-handler computation hit."""
    import json
    import time

    from beatroot.eval.artifact import write_components_result, write_system_result

    _use_scratch_artifact(monkeypatch, tmp_path)

    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import THRESHOLDS_PATH
    from beatroot.eval.runners.components import ComponentReport
    from beatroot.eval.runners.system import SystemReport

    thresholds = load_thresholds(THRESHOLDS_PATH)
    write_system_result(SystemReport(axes={"a1": 1.0}, passed=True), thresholds)
    write_components_result(ComponentReport(retrieval_recall_at_k=0.9, retrieval_leakage=0))

    started = time.perf_counter()
    r = client.get("/evals/summary")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"/evals/summary took {elapsed:.3f}s with an artifact present"
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["system"]) >= {"axes", "thresholds", "violations", "passed", "p50_ms", "p95_ms"}
    assert set(body["components"]) >= {
        "retrieval_recall_at_k",
        "retrieval_leakage",
        "feasibility_accuracy",
        "nutrition_exact_match",
        "drift_detection_recall",
    }
    assert body["system"]["axes"] == {"a1": 1.0}
    assert body["computed_at"] is not None
    assert isinstance(body["age_seconds"], (int, float))
    # both writes happened moments ago in this same test
    assert body["age_seconds"] < 30

    # Never computes on demand — reads exactly what was persisted.
    on_disk = json.loads((tmp_path / "last_run.json").read_text())
    assert body["system"]["axes"] == on_disk["system"]["axes"]


def test_evals_summary_reports_partial_when_only_one_section_is_written(
    monkeypatch, tmp_path, client
):
    from beatroot.confirm.trust_score import load_thresholds
    from beatroot.container import THRESHOLDS_PATH
    from beatroot.eval.artifact import write_system_result
    from beatroot.eval.runners.system import SystemReport

    _use_scratch_artifact(monkeypatch, tmp_path)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    write_system_result(SystemReport(axes={"a1": 1.0}, passed=True), thresholds)

    body = client.get("/evals/summary").json()
    assert body["status"] == "partial"
    assert body["system"] is not None
    assert body["components"] is None
    assert any("eval components" in cmd for cmd in body["commands"])
    assert not any("eval system" in cmd for cmd in body["commands"])


def test_evals_logs_endpoint_is_bounded_and_honest(client):
    logging.getLogger("beatroot.test_pages").info("evals-logs-marker", extra={"stage": "test"})
    r = client.get("/evals/logs?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["ring_buffer_size"] >= 5
    assert len(body["lines"]) <= 5
    assert isinstance(body["lines"], list)


def test_evals_logs_endpoint_redacts_like_every_other_log_line(client):
    logging.getLogger("beatroot.test_pages").info(
        "evals-logs-secret-check", extra={"api_key": "sk-should-never-appear"}
    )
    r = client.get("/evals/logs?limit=200")
    body = r.json()
    marker = next(line for line in body["lines"] if line["message"] == "evals-logs-secret-check")
    assert marker["api_key"] == "[redacted]"


def _use_scratch_history(monkeypatch, tmp_path):
    """Point `eval.history.HISTORY_DIR` at a throwaway directory for the
    duration of one test — same reasoning as `_use_scratch_artifact` above,
    for the sibling `eval/history/` snapshots `GET /evals/history` reads."""
    from beatroot.eval import history

    scratch = tmp_path / "history"
    monkeypatch.setattr(history, "HISTORY_DIR", scratch)
    return scratch


def test_evals_history_is_empty_list_when_nothing_recorded(monkeypatch, tmp_path, client):
    _use_scratch_history(monkeypatch, tmp_path)
    r = client.get("/evals/history")
    assert r.status_code == 200
    assert r.json() == {"entries": []}


def test_evals_history_responds_fast_and_reads_persisted_snapshots(monkeypatch, tmp_path, client):
    """Same `/evals/summary`-style timing budget: this route only ever
    reads what `beatroot eval iterate` already wrote."""
    import time

    from beatroot.eval.history import build_snapshot, write_snapshot

    scratch = _use_scratch_history(monkeypatch, tmp_path)
    entry = build_snapshot(
        label="baseline",
        note="initial",
        offline=True,
        config={"retrieval": {"lexical_weight": 1.0}},
        metrics={"components": {"retrieval_recall_at_k": 0.665}},
    )
    write_snapshot(entry, directory=scratch)

    started = time.perf_counter()
    r = client.get("/evals/history")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"/evals/history took {elapsed:.3f}s"
    assert r.status_code == 200
    body = r.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["label"] == "baseline"


def test_evals_history_limit_keeps_only_the_most_recent(monkeypatch, tmp_path, client):
    from beatroot.eval.history import build_snapshot, write_snapshot

    scratch = _use_scratch_history(monkeypatch, tmp_path)
    import datetime as dt

    for i in range(5):
        entry = build_snapshot(
            label=f"run{i}",
            note="",
            offline=True,
            config={},
            metrics={},
            now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(hours=i),
        )
        when = dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(hours=i)
        write_snapshot(entry, directory=scratch, now=when)

    body = client.get("/evals/history?limit=2").json()
    assert [e["label"] for e in body["entries"]] == ["run3", "run4"]


def test_evals_page_mentions_run_history_and_history_endpoint(client):
    body = client.get("/evals").text
    assert "run history" in body.lower()
    assert "/evals/history" in body
    assert "metric trends" in body.lower()


def test_evals_page_history_metrics_read_the_nested_metrics_object(client):
    """Regression guard: `GET /evals/history` entries carry every metric
    under a top-level `metrics` key (`{"metrics": {"axes": ..., "components":
    ...}}`), never at the entry's own top level. A prior version of this
    page's JS called `getPath(entry, "axes.A1_...")` directly on the raw
    entry instead of `entry.metrics` and silently rendered every cell as
    empty with no error — caught only by executing the JS, not by curl.
    This pins the fix at the source level: every dashboard lookup into a
    history entry must read through `.metrics`."""
    body = client.get("/evals").text
    assert "getPath(e.metrics" in body
    assert "getPath(e, path)" not in body
    assert "getPath(prev.metrics" in body
    assert "getPath(prev, path)" not in body
