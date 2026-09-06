"""Contracts for the weekly plan: one day, the week, and the job that builds it.

A weekly plan is N single-meal recommendations plus an aggregate nutrition
roll-up. Each day is whatever terminal the agent actually reached for that
day — `PlanDay.terminal_state` is reported verbatim, so a week containing a
NEGOTIATE or an ESCALATE reads as exactly that rather than as a gap. Two of
the three terminals are not a meal, and a weekly plan has no more licence to
dress them up than `POST /recommend` does.

`PlanJob` is the mutable half: the planner writes into it as days land, and
`WeeklyPlan` is the immutable view a caller is handed once the week is
complete.
"""

from typing import Literal

from pydantic import BaseModel, Field

from beatroot.contracts.nutrition import NutritionFacts
from beatroot.contracts.trust import CostRecord

JobStatus = Literal["running", "ready", "failed", "unknown"]

# How each day's recipe was chosen. "agent" is a real run through the graph;
# "variety" is a day the planner replaced so the week does not read as the
# same dish N times (`agent.batch_plan.WeeklyPlanner._backfill_for_variety`).
# Carried on the day itself so a caller can always tell the two apart instead
# of inferring it from the shape of the week.
DaySource = Literal["agent", "variety"]


class PlanDay(BaseModel):
    """One day of the week.

    `recipe_id`/`recipe_name`/`nutrition` are populated only for a day that
    actually reached COMMIT. A NEGOTIATE or ESCALATE day carries its
    `terminal_state` and a `detail` string instead — the same posture the
    CLI and the API already take for a single recommendation.
    """

    day: int
    terminal_state: str
    recipe_id: str | None = None
    recipe_name: str | None = None
    nutrition: NutritionFacts | None = None
    detail: str | None = None
    source: DaySource = "agent"


class PlanTotals(BaseModel):
    """Summed nutrition across the days that produced a meal.

    `days_counted` is on the model rather than left to the reader because a
    week where three days escalated has totals over three days, and a total
    that does not say how many days it covers invites being read as a weekly
    figure when it is not. Deliberately NOT a `NutritionFacts`: that model
    means "the computed facts for one dish", carries a `coverage` in [0,1]
    and a `provenance` literal, and neither claim survives summation.
    """

    kcal: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    sodium_mg: float = 0.0
    fibre_g: float = 0.0
    days_counted: int = 0


class WeeklyPlan(BaseModel):
    """The finished week, as a caller receives it."""

    profile_id: str
    days: list[PlanDay]
    totals: PlanTotals
    cost: CostRecord = Field(default_factory=CostRecord)


class PlanJob(BaseModel):
    """The planner's own working state for one submitted week.

    Not a `WeeklyPlan`: the days arrive out of order and some of them are
    still `None`, which is a shape a finished plan should not be able to
    express. `.plan()` is the one place the two are bridged.
    """

    job_id: str
    profile_id: str
    requested_days: int
    status: JobStatus = "running"
    completed: int = 0
    days: list[PlanDay | None] = Field(default_factory=list)
    totals: PlanTotals = Field(default_factory=PlanTotals)
    cost: CostRecord = Field(default_factory=CostRecord)
    error: str | None = None

    def plan(self) -> WeeklyPlan:
        """The week so far, with unfinished days dropped. Safe to call at
        any point — a caller polling a half-finished job gets the days that
        have landed, which is the whole reason this is a job and not a
        blocking call."""
        return WeeklyPlan(
            profile_id=self.profile_id,
            days=[d for d in self.days if d is not None],
            totals=self.totals,
            cost=self.cost,
        )
