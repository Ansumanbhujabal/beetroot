"""Persisted eval results — the artifact `beatroot eval system` and
`beatroot eval components` write to disk, and the only thing `GET
/evals/summary` is ever allowed to read.

The bug this module exists to fix: `evals_summary` used to run the real
system suite AND generate-and-run 200 synthetic profiles through the whole
agent, INSIDE the request handler — a batch job disguised as a page load,
observed to never return (>100s, unresolved). A route handler must never
run a suite; it may only read what a suite already wrote. Every route that
wants eval numbers now reads `ARTIFACT_PATH` (`eval/last_run.json`,
gitignored — generated output, same as `eval/synthetic/` and
`eval/healing/`) and returns immediately, even when the file has never been
written.

`write_system_result`/`write_components_result` do a read-modify-write of
the SAME file so each CLI command can be run independently (system today,
components tomorrow) without clobbering the other's last result — the file
holds up to two top-level sections, `"system"` and `"components"`, each
timestamped on its own. There is no locking around that read-modify-write:
both writers are CLI commands a human runs one at a time, never concurrent
request handlers, so the ordinary "two processes racing the same file"
hazard a web server would have to worry about does not apply here.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beatroot.container import ROOT

if TYPE_CHECKING:
    from beatroot.confirm.trust_score import EvalThresholds
    from beatroot.eval.runners.components import ComponentReport
    from beatroot.eval.runners.system import SystemReport

ARTIFACT_PATH = ROOT / "eval" / "last_run.json"


def system_report_to_dict(report: SystemReport, thresholds: EvalThresholds) -> dict[str, Any]:
    """The JSON-safe shape of one system-suite run — shared by the artifact
    writer and (previously) the request handler, so the two can never drift
    apart on what a "system" section actually contains."""
    return {
        "axes": report.axes,
        "thresholds": dict(thresholds.axes),
        "violations": report.violations,
        "hard_constraint_violation_threshold": thresholds.verifiers.hard_constraint_violations,
        "passed": report.passed,
        "p50_ms": round(report.p(0.50), 1),
        "p95_ms": round(report.p(0.95), 1),
        "cost_usd": round(report.cost_usd, 6),
        "failures": report.failures,
    }


def components_report_to_dict(report: ComponentReport) -> dict[str, Any]:
    """The JSON-safe shape of one component-suite run — see
    `system_report_to_dict` above for why this is a shared function rather
    than something inlined at each call site."""
    return {
        "retrieval_recall_at_k": report.retrieval_recall_at_k,
        "retrieval_recall_at_k_hard_only": report.retrieval_recall_at_k_hard_only,
        "retrieval_leakage": report.retrieval_leakage,
        "feasibility_accuracy": report.feasibility_accuracy,
        "nutrition_exact_match": report.nutrition_exact_match,
        "drift_detection_recall": report.drift_detection_recall,
        "notes": report.notes,
    }


def _read_raw(path: Path) -> dict[str, Any]:
    """The artifact's raw on-disk content, or an empty mapping for both
    "never written" and "unreadable" — corrupt output is always safe to
    treat as absent and regenerate, never a crash."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_system_result(
    report: SystemReport, thresholds: EvalThresholds, path: Path | None = None
) -> dict[str, Any]:
    """Merge a fresh system-suite result into the artifact and write it
    back, preserving whatever `"components"` section (if any) is already
    there. Returns the `"system"` section just written.

    `path` defaults to the CURRENT value of module-level `ARTIFACT_PATH`,
    looked up at call time rather than baked into the signature at def
    time — a test can `monkeypatch.setattr(artifact, "ARTIFACT_PATH", ...)`
    and every caller that doesn't pass its own `path` picks that up, with
    no risk of silently writing to the real repo path from a test.
    """
    resolved = path if path is not None else ARTIFACT_PATH
    data = _read_raw(resolved)
    section = {
        "computed_at": dt.datetime.now(dt.UTC).isoformat(),
        **system_report_to_dict(report, thresholds),
    }
    data["system"] = section
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2, default=str))
    return section


def write_components_result(report: ComponentReport, path: Path | None = None) -> dict[str, Any]:
    """Merge a fresh component-suite result into the artifact and write it
    back, preserving whatever `"system"` section (if any) is already
    there. Returns the `"components"` section just written. `path`
    resolves against the current `ARTIFACT_PATH` the same way
    `write_system_result` does — see its docstring."""
    resolved = path if path is not None else ARTIFACT_PATH
    data = _read_raw(resolved)
    section = {
        "computed_at": dt.datetime.now(dt.UTC).isoformat(),
        **components_report_to_dict(report),
    }
    data["components"] = section
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2, default=str))
    return section


def read_artifact(path: Path | None = None) -> dict[str, Any] | None:
    """The persisted artifact, or `None` when neither CLI command has ever
    written to it (or the file is unreadable) — the "not computed yet"
    signal `GET /evals/summary` turns into a structured, honest response
    instead of ever trying to compute the numbers itself. `path` resolves
    against the current `ARTIFACT_PATH` the same way `write_system_result`
    does — see its docstring."""
    resolved = path if path is not None else ARTIFACT_PATH
    data = _read_raw(resolved)
    return data or None
