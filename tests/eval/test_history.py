"""`eval.history` — persisted iteration snapshots and the regenerated
`EVAL_HISTORY.md` changelog. EVAL ITERATION LOOP task.

Every test points `HISTORY_DIR`/`HISTORY_MD_PATH`-equivalent paths at a
throwaway `tmp_path`, never the real repo `eval/history/` (gitignored,
written for real by `beatroot eval iterate`) or `EVAL_HISTORY.md`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from beatroot.eval.history import (
    build_snapshot,
    load_history,
    regenerate_history_md,
    render_history_markdown,
    write_snapshot,
)

_NOW = dt.datetime(2026, 8, 30, 12, 0, 0, tzinfo=dt.UTC)


def _entry(label: str, recall: float, ece: float, when: dt.datetime, offline: bool = True) -> dict:
    return build_snapshot(
        label=label,
        note=f"note for {label}",
        offline=offline,
        config={"retrieval": {"lexical_weight": 1.0}},
        metrics={
            "axes": {"A1_allergen_safety": 1.0},
            "system_passed": True,
            "hard_constraint_violations": 0,
            "components": {
                "retrieval_recall_at_k": recall,
                "retrieval_leakage": 0,
                "feasibility_accuracy": 1.0,
                "nutrition_exact_match": 1.0,
                "drift_detection_recall": 1.0,
            },
            "adversarial": {"injection": 1.0},
            "adversarial_passed": True,
            "calibration": {"ece": ece, "pairs": 10},
        },
        git_sha_value="deadbeef1234",
        now=when,
    )


def test_build_snapshot_has_every_required_top_level_field() -> None:
    entry = _entry("baseline", 0.665, 0.125, _NOW)
    assert entry["label"] == "baseline"
    assert entry["note"] == "note for baseline"
    assert entry["git_sha"] == "deadbeef1234"
    assert entry["offline"] is True
    assert entry["timestamp"] == _NOW.isoformat()
    assert entry["metrics"]["components"]["retrieval_recall_at_k"] == 0.665
    assert entry["verdict"] == ""
    assert entry["reason"] == ""


def test_git_sha_returns_a_real_commit_in_this_checkout() -> None:
    from beatroot.eval.history import git_sha

    sha = git_sha()
    assert sha != "unknown"
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_git_sha_falls_back_to_unknown_when_git_fails(monkeypatch) -> None:
    import subprocess

    from beatroot.eval import history as history_mod

    def _raise(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(history_mod.subprocess, "run", _raise)
    assert history_mod.git_sha() == "unknown"


def test_build_snapshot_accepts_an_explicit_git_sha_without_calling_git() -> None:
    entry = build_snapshot(
        label="x", note="", offline=True, config={}, metrics={}, git_sha_value="explicit", now=_NOW
    )
    assert entry["git_sha"] == "explicit"


def test_write_snapshot_round_trips_through_json(tmp_path: Path) -> None:
    entry = _entry("baseline", 0.665, 0.125, _NOW)
    path = write_snapshot(entry, directory=tmp_path, now=_NOW)
    assert path.exists()
    assert path.parent == tmp_path
    on_disk = json.loads(path.read_text())
    assert on_disk == entry


def test_load_history_is_empty_for_a_directory_that_does_not_exist_yet(tmp_path: Path) -> None:
    assert load_history(directory=tmp_path / "never_written") == []


def test_load_history_sorts_oldest_first_and_skips_corrupt_files(tmp_path: Path) -> None:
    later = _entry("second", 0.70, 0.10, _NOW + dt.timedelta(hours=1))
    earlier = _entry("first", 0.665, 0.125, _NOW)
    write_snapshot(later, directory=tmp_path, now=_NOW + dt.timedelta(hours=1))
    write_snapshot(earlier, directory=tmp_path, now=_NOW)
    (tmp_path / "corrupt.json").write_text("{not valid json")
    (tmp_path / "not_a_dict.json").write_text("[1, 2, 3]")

    entries = load_history(directory=tmp_path)
    assert [e["label"] for e in entries] == ["first", "second"]


def test_render_history_markdown_on_empty_history_is_well_defined() -> None:
    text = render_history_markdown([])
    assert "no iterations recorded yet" in text.lower()
    assert "still below target" in text.lower()


def test_render_history_markdown_shows_delta_between_consecutive_runs() -> None:
    entries = [
        _entry("baseline", 0.60, 0.20, _NOW),
        _entry("tuned", 0.70, 0.10, _NOW + dt.timedelta(hours=1)),
    ]
    text = render_history_markdown(entries)
    assert "`baseline`" in text
    assert "`tuned`" in text
    # recall went up (0.60 -> 0.70): a genuine improvement, delta shown.
    assert "+0.1000" in text
    # ECE went down (0.20 -> 0.10): also an improvement for a lower-is-better
    # metric, shown as a negative delta.
    assert "-0.1000" in text
    assert "still below target" in text.lower()


def test_render_history_markdown_first_run_has_no_delta() -> None:
    entries = [_entry("baseline", 0.60, 0.20, _NOW)]
    text = render_history_markdown(entries)
    assert "(first run)" in text


def test_regenerate_history_md_writes_the_file(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    md_path = tmp_path / "EVAL_HISTORY.md"
    write_snapshot(_entry("baseline", 0.60, 0.20, _NOW), directory=history_dir, now=_NOW)

    out = regenerate_history_md(directory=history_dir, md_path=md_path)

    assert out == md_path
    assert md_path.exists()
    assert "baseline" in md_path.read_text()


def test_verdict_and_reason_survive_the_round_trip(tmp_path: Path) -> None:
    entry = build_snapshot(
        label="lexical_weight_2x",
        note="doubled lexical_weight",
        offline=True,
        config={},
        metrics={"components": {"retrieval_recall_at_k": 0.665}},
        verdict="reverted",
        reason="zero measured delta; lexical channel was empty for this query",
        now=_NOW,
    )
    path = write_snapshot(entry, directory=tmp_path, now=_NOW)
    on_disk = json.loads(path.read_text())
    assert on_disk["verdict"] == "reverted"
    assert "zero measured delta" in on_disk["reason"]

    md = render_history_markdown([entry])
    assert "reverted" in md
    assert "zero measured delta" in md
