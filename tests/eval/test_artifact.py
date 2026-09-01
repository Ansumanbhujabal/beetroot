"""`eval.artifact` — the persisted `eval/last_run.json` that `GET
/evals/summary` reads and NEVER computes on demand. EVAL SUMMARY FIX task.

Every test points `ARTIFACT_PATH` at a throwaway `tmp_path` file (never the
real repo `eval/last_run.json`, which the CLI writes for real on every
`beatroot eval system`/`beatroot eval components` invocation).
"""

from pathlib import Path

import pytest

from beatroot.confirm.trust_score import load_thresholds
from beatroot.container import THRESHOLDS_PATH
from beatroot.eval import artifact
from beatroot.eval.runners.components import ComponentReport
from beatroot.eval.runners.system import SystemReport

_THRESHOLDS = load_thresholds(THRESHOLDS_PATH)


@pytest.fixture
def scratch_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "last_run.json"
    monkeypatch.setattr(artifact, "ARTIFACT_PATH", path)
    return path


def test_read_artifact_is_none_when_never_written(scratch_path: Path) -> None:
    assert artifact.read_artifact() is None


def test_read_artifact_is_none_on_corrupt_json(scratch_path: Path) -> None:
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_path.write_text("{not valid json")
    assert artifact.read_artifact() is None


def test_write_system_result_then_read_round_trips(scratch_path: Path) -> None:
    report = SystemReport(axes={"a1": 0.95}, violations=0, passed=True)
    written = artifact.write_system_result(report, _THRESHOLDS)
    assert written["axes"] == {"a1": 0.95}
    assert "computed_at" in written

    data = artifact.read_artifact()
    assert data is not None
    assert data["system"]["axes"] == {"a1": 0.95}
    assert data.get("components") is None


def test_write_components_result_preserves_existing_system_section(scratch_path: Path) -> None:
    system_report = SystemReport(axes={"a1": 1.0}, passed=True)
    artifact.write_system_result(system_report, _THRESHOLDS)

    component_report = ComponentReport(retrieval_recall_at_k=0.8, retrieval_leakage=0)
    artifact.write_components_result(component_report)

    data = artifact.read_artifact()
    assert data is not None
    assert data["system"]["axes"] == {"a1": 1.0}, "components write must not clobber system"
    assert data["components"]["retrieval_recall_at_k"] == 0.8


def test_write_system_result_preserves_existing_components_section(scratch_path: Path) -> None:
    component_report = ComponentReport(retrieval_recall_at_k=0.8, retrieval_leakage=0)
    artifact.write_components_result(component_report)

    system_report = SystemReport(axes={"a1": 1.0}, passed=True)
    artifact.write_system_result(system_report, _THRESHOLDS)

    data = artifact.read_artifact()
    assert data is not None
    assert data["components"]["retrieval_recall_at_k"] == 0.8, "system write clobbered components"
    assert data["system"]["axes"] == {"a1": 1.0}


def test_a_second_system_write_overwrites_the_first(scratch_path: Path) -> None:
    artifact.write_system_result(SystemReport(axes={"a1": 0.1}, passed=False), _THRESHOLDS)
    artifact.write_system_result(SystemReport(axes={"a1": 0.99}, passed=True), _THRESHOLDS)

    data = artifact.read_artifact()
    assert data is not None
    assert data["system"]["axes"] == {"a1": 0.99}
    assert data["system"]["passed"] is True
