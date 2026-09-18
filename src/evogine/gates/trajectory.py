"""Per-decision trajectory capture: the full path to every gate verdict.

V1's promotion pass decided eleven proposals and left behind no trace of
*how* — the verdict existed, the path to it did not. Telemetry (V2-1)
records the decision; anti-gaming (V2-8) records the trajectory: every
step a gate review took, tool calls included — which replay runs were
fetched, what they returned, which evidence was written, and which stage
stopped (or cleared) the proposal. One trajectory file per decision,
written alongside the :class:`~evogine.gates.telemetry.PromotionDecision`
evidence under ``runs/``, so a verdict is always reconstructible from what
actually happened, never only from what was claimed.

Captured content is sanitized at the recorder boundary: replay results and
error text can carry candidate-influenced content, and the security
review's rule — no unfiltered candidate output in gate artifacts — applies
to trajectories too.

Like the decision log, the store keys by proposal id and holds the latest
trajectory per proposal (a gate-error retry overwrites); full lifecycle
history stays in the store's append-only audit trail.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.gates.telemetry import ID_PATTERN
from evogine.sandbox.sanitize import sanitize_output

#: Per-field capture cap. Trajectories are audit artifacts, not transcripts:
#: large tool results are summarized by truncation, with the drop recorded.
_FIELD_CAP = 2_000


def _now() -> datetime:
    return datetime.now(UTC)


class TrajectoryError(ValueError):
    """A trajectory capture or storage rule was violated."""


class ToolCall(BaseModel):
    """One tool invocation inside a gate review, sanitized for the audit."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    tool: str = Field(min_length=1)
    arguments: str
    """Sanitized JSON rendering of the call arguments."""
    result: str
    """Sanitized summary of what the tool returned (or how it failed)."""
    at: datetime = Field(default_factory=_now)


class TrajectoryStep(BaseModel):
    """One step of a gate review: a tool call, an observation, or the verdict."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    kind: Literal["tool_call", "observation", "decision"]
    detail: str
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def _tool_call_shape(self) -> TrajectoryStep:
        if (self.tool_call is not None) != (self.kind == "tool_call"):
            raise ValueError("tool_call must be set exactly when kind is 'tool_call'")
        return self


class Trajectory(BaseModel):
    """The ordered capture of one gate review, tool calls included."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    gate: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=_now)
    decided_at: datetime | None = None
    steps: list[TrajectoryStep] = Field(default_factory=list)

    def tool_calls(self) -> list[ToolCall]:
        """Every tool call in review order — the "tool calls included" surface."""
        return [step.tool_call for step in self.steps if step.tool_call is not None]


class TrajectoryRecorder:
    """Collects the steps of one gate review; :meth:`freeze` hands back the record.

    The recorder sanitizes at capture time, so callers cannot forget: every
    argument, result, and observation passes through
    :func:`~evogine.sandbox.sanitize.sanitize_output` before it is stored.
    After :meth:`freeze` the recorder is closed — further steps raise
    :class:`TrajectoryError`, so one recorder is exactly one review.
    """

    def __init__(self, *, gate: str, proposal_id: str) -> None:
        self._trajectory = Trajectory(proposal_id=proposal_id, gate=gate)
        self._frozen = False

    def tool_call(
        self, tool: str, arguments: Mapping[str, object], *, result: str
    ) -> None:
        """Record one tool invocation and its sanitized outcome."""
        seq = self._next_seq()
        call = ToolCall(
            seq=seq,
            tool=tool,
            arguments=sanitize_output(
                json.dumps(arguments, sort_keys=True, default=str), max_chars=_FIELD_CAP
            ),
            result=sanitize_output(result, max_chars=_FIELD_CAP),
        )
        self._append(
            TrajectoryStep(
                seq=seq,
                kind="tool_call",
                detail=f"{tool} -> {call.result}",
                tool_call=call,
            )
        )

    def observe(self, detail: str) -> None:
        """Record a non-tool step of the review (a check, a stage note)."""
        self._append(
            TrajectoryStep(
                seq=self._next_seq(),
                kind="observation",
                detail=sanitize_output(detail, max_chars=_FIELD_CAP),
            )
        )

    def decide(self, stage: str, *, promoted: bool, reason: str) -> None:
        """Record the verdict step and stamp when the decision was reached."""
        self._trajectory.decided_at = _now()
        self._append(
            TrajectoryStep(
                seq=self._next_seq(),
                kind="decision",
                detail=sanitize_output(
                    f"{stage}: {reason} (promoted={promoted})", max_chars=_FIELD_CAP
                ),
            )
        )

    def freeze(self) -> Trajectory:
        """Close the review and return the trajectory (defensive copy)."""
        if self._frozen:
            raise TrajectoryError("trajectory recorder already frozen")
        self._frozen = True
        return self._trajectory.model_copy(deep=True)

    def _append(self, step: TrajectoryStep) -> None:
        if self._frozen:
            raise TrajectoryError("trajectory recorder is frozen")
        self._trajectory.steps.append(step)

    def _next_seq(self) -> int:
        return len(self._trajectory.steps) + 1


class TrajectoryStore:
    """File-backed trajectory store: one trajectory file per gate decision.

    Trajectories live under ``<root>/runs/trajectory/`` beside the gate
    evidence at ``<root>/runs/gate/``, inside the engine store, so the next
    store commit versions them alongside the decisions they explain. Files
    are named by proposal id (store-root-relative ref:
    ``runs/trajectory/<proposal_id>.json``) with the same filesystem-safe
    id rule as every other run-artifact sink.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.trajectories_dir = self.root / "runs" / "trajectory"

    def write(self, proposal_id: str, trajectory: Trajectory) -> str:
        """Persist ``trajectory`` atomically and return its store-relative ref."""
        if not ID_PATTERN.fullmatch(proposal_id):
            raise TrajectoryError(
                f"proposal id {proposal_id!r} must match {ID_PATTERN.pattern}"
            )
        if trajectory.proposal_id != proposal_id:
            raise TrajectoryError(
                f"trajectory proposal_id {trajectory.proposal_id!r} does not match"
                f" file id {proposal_id!r}"
            )
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(proposal_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(trajectory.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return f"runs/trajectory/{proposal_id}.json"

    def read(self, proposal_id: str) -> Trajectory:
        """Load the trajectory recorded for ``proposal_id``."""
        if not ID_PATTERN.fullmatch(proposal_id):
            raise TrajectoryError(
                f"proposal id {proposal_id!r} must match {ID_PATTERN.pattern}"
            )
        path = self._path(proposal_id)
        if not path.is_file():
            raise TrajectoryError(f"no trajectory recorded for proposal {proposal_id!r}")
        return Trajectory.model_validate_json(path.read_text(encoding="utf-8"))

    def all(self) -> list[Trajectory]:
        """Every recorded trajectory, ordered by proposal id."""
        if not self.trajectories_dir.exists():
            return []
        return [
            Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.trajectories_dir.glob("*.json"))
        ]

    def _path(self, proposal_id: str) -> Path:
        return self.trajectories_dir / f"{proposal_id}.json"
