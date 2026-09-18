"""V2-4: replay budgets as first-class plan config, enforced before spend.

Plans declare replay budgets — max replays per proposal, per case family,
and per run — and the queue refuses a schedule that would exceed any of
them *before* the first replay is spent. A run never discovers an exceeded
budget on the bill after the fact: queue-time refusal is all-or-nothing,
and nothing is reconciled afterwards.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from evogine.bench.replay import REPEAT_FLOOR, ReplayRequest, ReplaySchedule


class ReplayBudgets(BaseModel):
    """Declared replay budget caps.

    All three knobs are required: an optional knob would reintroduce the
    ambient-budget failure mode V2-4 exists to fix.
    """

    model_config = ConfigDict(extra="forbid")

    max_per_proposal: int = Field(ge=1)
    max_per_case_family: int = Field(ge=1)
    max_per_run: int = Field(ge=1)


class BenchPlan(BaseModel):
    """First-class bench plan config: repeats, budgets, optional case subset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    repeats: int = Field(ge=REPEAT_FLOOR)
    budgets: ReplayBudgets
    case_ids: list[str] | None = None
    """Omitted means the full corpus; a subset must name existing case ids."""


class BudgetScope(str, Enum):
    """Which declared budget a schedule exceeded."""

    RUN = "run"
    CASE_FAMILY = "case_family"
    PROPOSAL = "proposal"


class BudgetExceededError(Exception):
    """Queue-time refusal: the schedule exceeds a declared budget before spend."""

    def __init__(self, scope: BudgetScope, key: str, required: int, allowed: int) -> None:
        self.scope = scope
        self.key = key
        self.required = required
        self.allowed = allowed
        super().__init__(
            f"replay budget exceeded at queue time (scope={scope.value}, key={key!r}): "
            f"schedule requires {required} replays, budget allows {allowed} — "
            "refusing before spend"
        )


def queue_replays(
    requests: list[ReplayRequest], budgets: ReplayBudgets
) -> ReplaySchedule:
    """Enforce the declared budgets on the full schedule BEFORE any replay is spent.

    Check order is per-run, then per-family, then per-proposal. Refusal is
    all-or-nothing: an exceeding schedule spends nothing and is never
    reconciled after the fact.
    """
    schedule = ReplaySchedule(requests=requests)
    total = schedule.total()
    if total > budgets.max_per_run:
        raise BudgetExceededError(BudgetScope.RUN, "*", total, budgets.max_per_run)
    for family, count in sorted(schedule.by_family().items()):
        if count > budgets.max_per_case_family:
            raise BudgetExceededError(
                BudgetScope.CASE_FAMILY, family, count, budgets.max_per_case_family
            )
    for proposal, count in sorted(schedule.by_proposal().items()):
        if count > budgets.max_per_proposal:
            raise BudgetExceededError(
                BudgetScope.PROPOSAL, proposal, count, budgets.max_per_proposal
            )
    return schedule


def load_plan(path: str | Path) -> BenchPlan:
    """Parse and validate a bench plan from a YAML file.

    Unknown fields are refused (config drift), the repeat count must meet
    the 4-repeat floor, and budgets must be present and positive.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BenchPlan.model_validate(data)
