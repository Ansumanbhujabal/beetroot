"""Tests for `store.preferences.PreferenceMemory`. Spec §9.

EMA convergence is tested over many updates, not just one — a single
acceptance moving affinity off zero proves the arithmetic runs, but the
load-bearing property is that repeated feedback converges toward the bound
without ever crossing it.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from beatroot.settings import get_settings
from beatroot.store.db import connect
from beatroot.store.preferences import PreferenceMemory


def test_acceptance_raises_affinity(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    m.record("p1", {"spicy", "vegan"}, accepted=True)
    a = m.affinity("p1")
    assert a["spicy"] > 0
    assert a["vegan"] > 0


def test_rejection_lowers_affinity(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    m.record("p1", {"spicy"}, accepted=False)
    assert m.affinity("p1")["spicy"] < 0


def test_ema_converges_and_stays_bounded(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    for _ in range(50):
        m.record("p1", {"spicy"}, accepted=True)
    v = m.affinity("p1")["spicy"]
    assert 0.9 < v <= 1.0, "EMA must converge toward 1 without exceeding it"


def test_ema_converges_downward_and_stays_bounded(tmp_path):
    """The same convergence property, mirrored for rejection — must not
    exceed -1.0 either."""
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    for _ in range(50):
        m.record("p1", {"spicy"}, accepted=False)
    v = m.affinity("p1")["spicy"]
    assert -1.0 <= v < -0.9


def test_ema_never_exceeds_bounds_across_many_mixed_updates(tmp_path):
    """Alternating feedback still never crosses [-1, 1] at any point, over
    a long run — not just the final value."""
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    pattern = [True, True, False, True, False, False, True, True, True, False]
    for i in range(200):
        m.record("p1", {"spicy"}, accepted=pattern[i % len(pattern)])
        v = m.affinity("p1")["spicy"]
        assert -1.0 <= v <= 1.0


def test_profiles_are_isolated(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    m.record("p1", {"spicy"}, accepted=True)
    assert m.affinity("p2") == {}


def test_second_profiles_feedback_never_moves_the_first(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    m.record("p1", {"spicy"}, accepted=True)
    before = m.affinity("p1")["spicy"]
    m.record("p2", {"spicy"}, accepted=False)
    assert m.affinity("p1")["spicy"] == before


def test_alpha_defaults_from_settings_not_a_literal(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"))
    assert m.alpha == get_settings().preferences.ema_alpha


def test_explicit_alpha_overrides_settings_default(tmp_path):
    m = PreferenceMemory(connect(tmp_path / "t.db"), alpha=0.9)
    m.record("p1", {"spicy"}, accepted=True)
    # alpha=0.9 moves almost all the way to the target on one update.
    assert m.affinity("p1")["spicy"] > 0.85


def test_record_takes_the_shared_lock(tmp_path):
    """A writer that ignores the shared `db_lock` reintroduces the
    interleaving bug `store.audit`/`store.incidents`/`store.cache` were
    already fixed for — `record()` must actually acquire whatever lock it
    is given, not merely accept the parameter."""
    conn = connect(tmp_path / "t.db")
    lock = threading.Lock()
    m = PreferenceMemory(conn, lock=lock)
    assert m._lock is lock

    def write(i: int) -> None:
        m.record(f"p{i}", {"spicy"}, accepted=True)

    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(write, range(20)))

    for i in range(20):
        assert m.affinity(f"p{i}")["spicy"] > 0
