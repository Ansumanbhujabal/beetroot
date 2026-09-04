"""Tests for `agent.batch_plan.WeeklyPlanner`.

The planner fans one `ConstraintSet` out into N days across its own pool
and rolls the results back up, so these drive it end to end against the
real agent rather than a stub: what is worth proving here is that a week
submitted is a week that lands, that a partially finished week is readable,
and that a finished week's spend reaches whoever is accounting for it.

`.submit()` returns before any day has run, so every test that wants a
finished week polls `.status()` — there is deliberately no `.join()` on
this class, because no caller of it in production has one either.
"""

import time

import pytest

from beatroot.agent.batch_plan import WeeklyPlanner
from beatroot.contracts.trust import CostRecord

# Generous by a factor of ~50: a 7-day offline week measures at well under
# a second. Large enough that a loaded CI box never flakes, small enough
# that a genuinely stuck week fails the test instead of hanging it.
_TIMEOUT_S = 30.0


def _await_ready(planner: WeeklyPlanner, job_id: str) -> None:
    deadline = time.monotonic() + _TIMEOUT_S
    while planner.status(job_id) == "running":
        if time.monotonic() > deadline:
            completed = planner.completed(job_id)
            pytest.fail(f"week {job_id} still running after {_TIMEOUT_S}s ({completed} days done)")
        time.sleep(0.02)


@pytest.fixture
def planner(agent, agent_deps) -> WeeklyPlanner:
    return WeeklyPlanner(agent, agent_deps.catalog)


def test_a_week_lands_a_day_for_every_day_requested(planner, preset_cs):
    job_id = planner.submit(preset_cs("eggetarian"), days=7)
    _await_ready(planner, job_id)

    plan = planner.get(job_id)
    assert planner.status(job_id) == "ready"
    assert len(plan.days) == 7
    assert [d.day for d in plan.days] == list(range(7))
    assert plan.profile_id == "eggetarian"


def test_every_day_reports_the_terminal_it_actually_reached(planner, preset_cs):
    """Not "every day is a meal" — a day that negotiates or escalates is a
    legitimate day, and the planner's contract is that it says so rather
    than dropping the day or dressing it up."""
    job_id = planner.submit(preset_cs("eggetarian"), days=3)
    _await_ready(planner, job_id)

    for day in planner.get(job_id).days:
        assert day.terminal_state
        if day.terminal_state == "COMMIT":
            assert day.recipe_id and day.recipe_name and day.nutrition
        else:
            assert day.recipe_id is None


def test_totals_cover_only_the_days_that_produced_a_meal(planner, preset_cs):
    job_id = planner.submit(preset_cs("eggetarian"), days=5)
    _await_ready(planner, job_id)

    plan = planner.get(job_id)
    with_a_meal = [d for d in plan.days if d.nutrition is not None]
    assert plan.totals.days_counted == len(with_a_meal)
    assert plan.totals.kcal == pytest.approx(sum(d.nutrition.kcal for d in with_a_meal))


def test_an_id_the_planner_never_issued_has_no_status_and_no_plan(planner):
    assert planner.status("00000000-0000-0000-0000-000000000000") == "unknown"
    assert planner.get("00000000-0000-0000-0000-000000000000") is None
    assert planner.completed("00000000-0000-0000-0000-000000000000") == 0


def test_a_finished_week_hands_its_spend_to_the_completion_hook(agent, agent_deps, preset_cs):
    """The `202` has already gone out by the time a week finishes, so no
    request handler is left to fold this cost into the ledger. If this
    hook stops firing, a week's tokens stop appearing in `/metrics` and
    nothing else in the system notices."""
    banked: list[CostRecord] = []
    planner = WeeklyPlanner(agent, agent_deps.catalog, on_complete=banked.append)

    job_id = planner.submit(preset_cs("eggetarian"), days=3)
    _await_ready(planner, job_id)

    assert len(banked) == 1
    assert banked[0] == planner.get(job_id).cost
