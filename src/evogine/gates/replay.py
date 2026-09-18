"""Replay seam: where the memory gate obtains cold/warm verdicts.

The gate never runs models itself — it consumes replay results through the
:class:`ReplayRunner` protocol, so the unit suite stays offline (stub
runners over deterministic fixtures) and a real harness plugs in later as
configuration, not code. Each run reports its arm — ``baseline`` (cold,
learning absent) or ``warm`` (learning applied) — and its per-case-instance
verdicts.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from evogine.gates.materiality import Verdict

Arm = Literal["baseline", "warm"]


class ReplayRun(BaseModel):
    """One replay run's verdicts: which arm it ran and what it observed."""

    run_id: str
    arm: Arm
    verdicts: list[Verdict]


class ReplayRunner(Protocol):
    """Source of recorded replay results, keyed by run id."""

    def run(self, run_id: str) -> ReplayRun: ...
