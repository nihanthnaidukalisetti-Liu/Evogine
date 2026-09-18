"""Replay scheduling vocabulary: the schedule a bench run will spend.

The harness queues the full replay schedule — every case × arm × repeat —
before executing anything, so spend is always a checked, inspectable plan
rather than a side effect. V2-4 budget enforcement
(:mod:`evogine.bench.budgets`) refuses at this queue step when the schedule
exceeds the budgets declared in plan config.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evogine.bench.cases import BenchCase

REPEAT_FLOOR = 4
"""Minimum repeats per case per arm for any bench run (the 4-repeat floor).

Below four repeats, per-case variance is too thin to distinguish a stable
effect from replay noise (evaluation discipline).
"""

Arm = Literal["cold", "warm"]

ARMS: tuple[Arm, ...] = ("cold", "warm")


class ReplayRequest(BaseModel):
    """One planned replay: a single case execution in one arm."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    family: str
    proposal_id: str
    arm: Arm
    repeat: int = Field(ge=0)


class ReplaySchedule(BaseModel):
    """A queued replay schedule with its per-scope replay counts."""

    model_config = ConfigDict(extra="forbid")

    requests: list[ReplayRequest]

    def total(self) -> int:
        """Return the total number of replays the schedule will spend."""
        return len(self.requests)

    def by_family(self) -> dict[str, int]:
        """Return replay counts per case family."""
        return dict(Counter(request.family for request in self.requests))

    def by_proposal(self) -> dict[str, int]:
        """Return replay counts per proposal."""
        return dict(Counter(request.proposal_id for request in self.requests))


def build_replay_requests(
    cases: Sequence[BenchCase], repeats: int
) -> list[ReplayRequest]:
    """Build the full replay schedule for ``cases``: every case × arm × repeat."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    return [
        ReplayRequest(
            case_id=case.id,
            family=case.family,
            proposal_id=case.proposal(),
            arm=arm,
            repeat=repeat,
        )
        for case in cases
        for arm in ARMS
        for repeat in range(repeats)
    ]
