from beatroot.confirm.escalation import gate
from beatroot.contracts.trust import TrustReport
from beatroot.settings import get_settings
from beatroot.store.db import connect
from beatroot.store.incidents import IncidentLog


def _trust(composite: float, failing: str | None = "catalog_coverage") -> TrustReport:
    return TrustReport(
        composite=composite,
        catalog_coverage=composite,
        constraint_completeness=1.0,
        model_self_assessment=0.5,
        failing_signal=failing,
    )


def test_high_trust_passes_the_gate():
    assert gate(_trust(0.9, None)) is None


def test_low_trust_escalates_and_names_the_signal():
    esc = gate(_trust(0.3))
    assert esc is not None
    assert esc.reason == "low_trust"
    assert esc.failing_signal == "catalog_coverage"
    assert "0.3" in esc.detail or "0.30" in esc.detail


def test_boundary_value_is_inclusive_pass():
    """The gate is read from settings, not a raw dict — the threshold below
    comes straight from get_settings().trust.refusal_threshold."""
    threshold = get_settings().trust.refusal_threshold
    assert gate(_trust(threshold, None)) is None
    assert gate(_trust(threshold - 0.0001)) is not None


def test_weak_deterministic_signal_escalates_even_above_the_numeric_threshold():
    """constraint_completeness (0.30) + model_self_assessment (0.25) alone
    sum to exactly the default refusal threshold, so a fully-checked
    constraint set plus a maximally confident model can carry the composite
    to threshold regardless of how thin catalog coverage is. A confident
    model must not be able to rescue an answer the catalog does not support
    — so a set failing_signal escalates even when the numeric composite
    would otherwise pass."""
    thin_catalog_but_high_composite = TrustReport(
        composite=0.64,
        catalog_coverage=0.2,
        constraint_completeness=1.0,
        model_self_assessment=1.0,
        failing_signal="catalog_coverage",
    )
    esc = gate(thin_catalog_but_high_composite)
    assert esc is not None
    assert esc.failing_signal == "catalog_coverage"


def test_gate_uses_the_passed_settings_override():
    """gate() must honour an explicit settings argument rather than always
    falling back to get_settings()."""
    strict = get_settings().model_copy(deep=True)
    strict.trust.refusal_threshold = 0.9
    assert gate(_trust(0.8, None), settings=strict) is not None
    assert gate(_trust(0.95, None), settings=strict) is None


def test_incident_log_roundtrips(tmp_path):
    log = IncidentLog(connect(tmp_path / "t.db"))
    inc = log.record("escalation", "p1", "low trust", {"composite": 0.3})
    assert inc.id
    all_ = log.all()
    assert len(all_) == 1
    assert all_[0].kind == "escalation"
    assert all_[0].payload["composite"] == 0.3
    assert all_[0].created_at.tzinfo is not None
