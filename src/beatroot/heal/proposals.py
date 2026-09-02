"""Incidents become proposals. Spec §9, Task 16 — the healing loop closes.

**The design position this module exists to hold:** beatroot runs in a
domain carrying allergen and medical constraints. An agent that silently
rewrites its own safety rules — a drift tolerance, a meta-tag mapping — in
that domain is a liability, not a feature. So `propose()` NEVER writes to a
live config or data file, and NEVER mutates one on the caller's behalf.
Every `threshold` / `meta_tag` proposal it produces is written to
`out_dir/proposals/` as an inert, human-readable file: what was observed,
how many times, what the options are, and what change is suggested. A human
reads it, judges it, and applies it by hand (or doesn't). This is the
fourth boundary zone (actions that require confirmation) taken literally —
there is no code path in this module, or anywhere it calls, that opens
`config/beatroot.yaml` or `eval/thresholds.yaml` for writing.

**The single exception is additive eval-case generation.** Every cluster,
regardless of size, also yields an `eval_case` proposal written to
`out_dir/generated/` and marked `auto_applied=True`. That's safe in a way a
rule change never is: a new regression case can only ever ADD a check the
suite runs — it can loosen nothing, relax nothing, and cannot itself
introduce a false negative into an existing case. A single failure is still
worth a permanent regression test, so this happens even for a cluster of
one; a rule change is not, so that stays gated on
`healing.rule_proposal_min_cluster` — a one-off must not retune a threshold
that every future request is judged against.

**Fix round: a generated case must be able to fail on behaviour, not just
on a parse error.** The first pass generated every case with `constraints:
[]` and a permissive `expect_terminal` listing all three terminals — that
can never fail except by crashing the loader, which made "the same
incident cannot recur" false. `agent.nodes` now stashes the triggering
`ConstraintSet` and the terminal it actually reached on every incident it
records (Task 9's `Incident.payload` was always free-form for exactly
this); `_eval_case_proposal` below replays that real `ConstraintSet` and
asserts that real terminal whenever it's present, and is honest —
`case["verified"] = False` plus a "LOADER SMOKE TEST ONLY" note — when a
triggering incident predates this and doesn't carry one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from beatroot.heal.cluster import Cluster


@dataclass
class Proposal:
    """One output of `propose()`. `auto_applied` is True only for
    `kind == "eval_case"` — see the module docstring."""

    kind: str
    auto_applied: bool
    path: Path
    body: str


def _fingerprint(signature: str) -> str:
    """Stable, deterministic id derived from a cluster's signature.

    Deliberately `hashlib.sha256`, not Python's builtin `hash()`: `hash()`
    is salted per-process (PYTHONHASHSEED) unless explicitly disabled, so
    two `heal` runs over the same incidents would mint two different
    filenames for the same pattern, silently piling up duplicate proposals
    and duplicate generated cases run after run instead of re-writing the
    same file. sha256 makes `heal` idempotent: re-running it over an
    unchanged incident log reproduces the same paths.
    """
    return hashlib.sha256(signature.encode()).hexdigest()[:8]


# Deterministic expected terminal per incident `kind`, used only as a
# fallback when an incident's payload carries a `constraint_set` but (for
# whatever reason — an older row) no explicit `terminal`. Every current
# emission site (`agent/nodes.py`) now records BOTH, so this exists purely
# as a safety net, not the primary source of truth — see
# `_eval_case_proposal`, which always prefers the recorded terminal.
_TERMINAL_BY_KIND = {
    "escalation": "ESCALATE",
    "drift": "ESCALATE",
    "unknown_ingredient": "ESCALATE",
    "verification_failure": "ESCALATE",
    "infeasible": "NEGOTIATE",
}


def _eval_case_proposal(cluster_: Cluster, generated_dir: Path) -> Proposal:
    """Always-safe half of `propose()`: turn one cluster into one additive,
    auto-applied regression case. Loadable by
    `beatroot.eval.runners.system.load_cases` — same shape as
    `eval/golden/seed_cases.yaml` — and scored under the `regression`
    family (`config/beatroot.yaml`'s `eval.axis_by_family.regression`,
    mapped to A5_escalation_correctness).

    **Fix round (review finding):** a case with `constraints: []` and
    `expect_terminal` listing all three terminals cannot fail on behaviour
    — it only proves the loader can parse it. Every `agent.nodes`
    incident-emission site now stashes `payload["constraint_set"]`
    (`ConstraintSet.model_dump(mode="json")`, the exact shape
    `run_system` already reconstructs via `Constraint(**c)`) and
    `payload["terminal"]` (the terminal that run actually reached)
    alongside whatever it already recorded. When the triggering
    incident carries both, this case replays the REAL `ConstraintSet`
    and asserts the REAL terminal — a genuine behavioural regression
    check: break the code path that produced it, and this case now
    fails. `case["verified"]` records which happened, and the `note`
    says so in prose, so nobody mistakes a weak case for a strong one.

    Older incidents (recorded before this fix, or from a future emission
    site that doesn't call `deps.incidents.record` with a
    `constraint_set`) have no such payload — for those this case falls
    back to `constraints: []` and every terminal accepted, same as
    before: a loader smoke test only, never a behavioural guarantee.
    """
    fp = _fingerprint(cluster_.signature)
    inc = cluster_.incidents[0]
    payload = inc.payload or {}
    raw_cs = payload.get("constraint_set")
    constraints = raw_cs.get("constraints") if isinstance(raw_cs, dict) else None

    if constraints is not None:
        terminal = payload.get("terminal") or _TERMINAL_BY_KIND.get(cluster_.kind)
    else:
        terminal = None
        constraints = []

    verified = terminal is not None
    expect_terminal = [terminal] if terminal else ["COMMIT", "NEGOTIATE", "ESCALATE"]

    if verified:
        note = (
            f"Auto-generated by the healing loop from {cluster_.count} '{cluster_.kind}' "
            f"incident(s) matching: {cluster_.signature}. Replays the ORIGINAL "
            f"ConstraintSet recorded on the triggering incident and asserts the terminal "
            f"it actually reached ({terminal}) — a real behavioural regression check: "
            f"if a future change makes this scenario resolve differently, this case fails."
        )
    else:
        note = (
            f"Auto-generated by the healing loop from {cluster_.count} '{cluster_.kind}' "
            f"incident(s) matching: {cluster_.signature}. LOADER SMOKE TEST ONLY — the "
            f"triggering incident's payload carries no constraint_set (an older row, or an "
            f"emission site that predates this), so this case cannot replay the original "
            f"scenario and accepts any terminal. A pass here proves the case still parses "
            f"and the run still completes; it is NOT proof the original failure cannot "
            f"recur."
        )

    case: dict[str, Any] = {
        "id": f"gen_{fp}",
        "family": "regression",
        "verified": verified,
        "note": note,
        "source_signature": cluster_.signature,
        "observed_count": cluster_.count,
        "query": inc.detail[:120],
        "constraints": constraints,
        "expect_terminal": expect_terminal,
    }
    path = generated_dir / f"gen_{fp}.yaml"
    path.write_text(yaml.safe_dump([case], sort_keys=False))
    return Proposal(kind="eval_case", auto_applied=True, path=path, body=yaml.safe_dump(case))


def _threshold_proposal(cluster_: Cluster, proposals_dir: Path) -> Proposal:
    """Human-gated: repeated nutrition drift, suggesting the explanation
    prompt or the drift tolerance itself needs tuning. Never auto-applied —
    `verifiers.nutrition_drift_pct` in `eval/thresholds.yaml` governs every
    future COMMIT, so no automated process gets to retune it unattended."""
    body = (
        f"# PROPOSAL (requires human approval) — kind: threshold\n"
        f"# Generated by the healing loop. Not applied automatically.\n"
        f"#\n"
        f"# Observed: {cluster_.count} '{cluster_.kind}' incident(s) matching:\n"
        f"#   {cluster_.signature}\n"
        f"#\n"
        f"# Example (of {cluster_.count}): {cluster_.incidents[0].detail}\n"
        f"#\n"
        f"# The model is repeatedly stating nutrition numbers that do not\n"
        f"# match the catalog by more than the current drift tolerance. Two\n"
        f"# options, either of which a maintainer should choose deliberately:\n"
        f"#   1. Tighten the explanation prompt (prompts/explain.md) to\n"
        f"#      forbid restating computed numbers in prose.\n"
        f"#   2. Lower verifiers.nutrition_drift_pct in eval/thresholds.yaml\n"
        f"#      so smaller drift is caught sooner.\n"
        f"#\n"
        f"# To apply option 2, edit eval/thresholds.yaml by hand:\n"
        f"verifiers:\n"
        f"  nutrition_drift_pct: 0.03  # was 0.05 — tighten only after review\n"
    )
    path = proposals_dir / f"threshold_{_fingerprint(cluster_.signature)}.yaml"
    path.write_text(body)
    return Proposal(kind="threshold", auto_applied=False, path=path, body=body)


def _meta_tag_proposal(cluster_: Cluster, proposals_dir: Path) -> Proposal:
    """Human-gated: repeated escalations / infeasibility / unrecognised
    vocabulary on the same signature, suggesting the catalog is missing
    data rather than that a threshold is miscalibrated. Never auto-applied
    — a meta-tag or catalog edit changes which recipes a future request is
    even allowed to see."""
    body = (
        f"# PROPOSAL (requires human approval) — kind: meta_tag\n"
        f"# Generated by the healing loop. Not applied automatically.\n"
        f"#\n"
        f"# Observed: {cluster_.count} '{cluster_.kind}' incident(s) matching:\n"
        f"#   {cluster_.signature}\n"
        f"#\n"
        f"# Example (of {cluster_.count}): {cluster_.incidents[0].detail}\n"
        f"#\n"
        f"# Repeated incidents on the same signature usually mean the\n"
        f"# catalog is missing coverage rather than that the guard rejecting\n"
        f"# it is wrong — an unrecognised ingredient/tag, or a constraint\n"
        f"# combination nothing in the catalog satisfies. Review the\n"
        f"# ingredients/recipes involved (data/ingredients.yaml,\n"
        f"# data/recipes.yaml) and add the missing nutrition data or\n"
        f"# meta-tags before considering any threshold change.\n"
    )
    path = proposals_dir / f"meta_tag_{_fingerprint(cluster_.signature)}.md"
    path.write_text(body)
    return Proposal(kind="meta_tag", auto_applied=False, path=path, body=body)


# Kinds whose repeated occurrence points at the catalog being incomplete
# rather than at a numeric threshold being wrong. Data, not an if/elif
# chain over kinds growing without bound — adding a kind here is a one-line
# change, not a new branch. "verification_failure" is a real `Incident.kind`
# (Spec §9's contract) that no running code path emits yet; it is listed
# here so a future emitter is covered for free rather than silently falling
# through to "no rule proposal at all".
_META_TAG_KINDS = frozenset({"escalation", "unknown_ingredient", "infeasible"})
_THRESHOLD_KINDS = frozenset({"drift", "verification_failure"})


def propose(clusters: list[Cluster], out_dir: Path) -> list[Proposal]:
    """Turn each cluster into proposals. See the module docstring for the
    asymmetry this enforces: `eval_case` proposals are always emitted and
    always `auto_applied=True`; `threshold`/`meta_tag` proposals are only
    emitted for clusters at or above `healing.rule_proposal_min_cluster`
    and are NEVER `auto_applied` — they are written to disk for a human to
    read and apply by hand, never mutated back into a live config or data
    file by this function or anything it calls.
    """
    from beatroot.settings import get_settings

    rule_threshold = get_settings().healing.rule_proposal_min_cluster
    out_dir = Path(out_dir)
    proposals_dir = out_dir / "proposals"
    generated_dir = out_dir / "generated"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    out: list[Proposal] = []
    for c in clusters:
        out.append(_eval_case_proposal(c, generated_dir))

        if c.count < rule_threshold:
            continue
        if c.kind in _THRESHOLD_KINDS:
            out.append(_threshold_proposal(c, proposals_dir))
        elif c.kind in _META_TAG_KINDS:
            out.append(_meta_tag_proposal(c, proposals_dir))

    return out
