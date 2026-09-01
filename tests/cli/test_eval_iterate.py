"""`beatroot eval iterate` — the full-suite, attributed snapshot command.
EVAL ITERATION LOOP task.

Points `eval.history`'s module-level paths at `tmp_path` for the duration
of each test (same pattern `tests/eval/test_artifact.py` uses for
`ARTIFACT_PATH`) so a test run never writes into the real repo's
`eval/history/`/`EVAL_HISTORY.md`. Every invocation uses small `--n-*`
values — this drives the REAL agent end to end, offline, and does not need
hundreds of cases to prove the wiring works.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from beatroot.cli.main import app
from beatroot.eval import history as history_mod

runner = CliRunner()


@pytest.fixture(autouse=True)
def _scratch_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    history_dir = tmp_path / "history"
    md_path = tmp_path / "EVAL_HISTORY.md"
    monkeypatch.setattr(history_mod, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(history_mod, "HISTORY_MD_PATH", md_path)
    return history_dir


def _invoke(label: str = "baseline", note: str = "initial measurement") -> object:
    return runner.invoke(
        app,
        [
            "eval",
            "iterate",
            "--label",
            label,
            "--note",
            note,
            "--n-adversarial",
            "20",
            "--calibration-n",
            "15",
        ],
    )


def test_eval_iterate_runs_end_to_end_offline() -> None:
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert "baseline" in result.output
    assert "calibration ECE" in result.output


def test_eval_iterate_writes_a_history_snapshot(_scratch_history: Path) -> None:
    _invoke(label="baseline", note="initial measurement")
    files = list(_scratch_history.glob("*.json"))
    assert len(files) == 1

    import json

    data = json.loads(files[0].read_text())
    assert data["label"] == "baseline"
    assert data["note"] == "initial measurement"
    assert data["offline"] is True
    assert data["git_sha"] != ""
    assert "retrieval_recall_at_k" in data["metrics"]["components"]
    assert "ece" in data["metrics"]["calibration"]
    assert set(data["metrics"]["axes"]) >= {"A1_allergen_safety", "A6_explanation_grounding"}


def test_eval_iterate_regenerates_eval_history_md() -> None:
    _invoke(label="baseline", note="initial measurement")
    md_path = history_mod.HISTORY_MD_PATH
    assert md_path.exists()
    text = md_path.read_text()
    assert "baseline" in text
    assert "initial measurement" in text


def test_eval_iterate_verdict_and_reason_are_recorded(_scratch_history: Path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "iterate",
            "--label",
            "lexical_2x",
            "--note",
            "doubled lexical_weight",
            "--verdict",
            "reverted",
            "--reason",
            "zero measured delta",
            "--n-adversarial",
            "20",
            "--calibration-n",
            "15",
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    data = json.loads(next(_scratch_history.glob("*.json")).read_text())
    assert data["verdict"] == "reverted"
    assert data["reason"] == "zero measured delta"


def test_second_iteration_appends_rather_than_overwrites(_scratch_history: Path) -> None:
    _invoke(label="first", note="a")
    _invoke(label="second", note="b")
    files = list(_scratch_history.glob("*.json"))
    assert len(files) == 2
    text = history_mod.HISTORY_MD_PATH.read_text()
    assert "first" in text
    assert "second" in text
