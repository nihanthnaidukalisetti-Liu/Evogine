"""Learning records: the durable knowledge atom of the engine.

A :class:`LearningRecord` is the single versioned, git-backed payload the
engine stores, gates, and retrieves. The v1 lifecycle moves a record through
``ephemeral -> candidate -> promoted`` (or ``dropped``); the memory gate
promotes a candidate only when counterfactual replay demonstrates a material
verdict change. Every record carries provenance and confidence so later
phases can audit *why* the engine believes something.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class RecordBody(BaseModel):
    """Structured failure analysis: what was seen, why, how found, how fixed."""

    signal: str
    failure_mode: str
    detection: str
    fix_strategy: str


class RecordProvenance(BaseModel):
    """Where a record came from and who proposed it."""

    proposed_by: str
    origin: str
    source: str
    note: str


class RecordCounterfactual(BaseModel):
    """Replay evidence attached by the memory gate.

    ``verdict_change`` stays ``None`` until a counterfactual replay has
    actually run; ``evidence`` points at the replay artifacts that justify
    a promotion (or rejection).
    """

    replay_run_ids: list[str] = Field(default_factory=list)
    verdict_change: bool | None = None
    evidence: str | None = None


class LearningRecord(BaseModel):
    """A versioned unit of engine knowledge.

    Exactly one of ``failure-pattern``, ``playbook``, or ``check``, with a
    confidence in the closed interval ``[0, 1]`` and a lifecycle status.
    """

    id: str
    type: Literal["failure-pattern", "playbook", "check"]
    title: str
    body: RecordBody
    status: Literal["ephemeral", "candidate", "promoted", "dropped"]
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: RecordProvenance
    counterfactual: RecordCounterfactual
    created: datetime
    promoted_at: datetime | None = None

    @classmethod
    def sample(cls) -> LearningRecord:
        """Return a realistic, fully-populated record for tests and docs.

        Mirrors the seed-pack shape: a failure-pattern proposed by the
        reflector from a Darwin phase-0 degradation rollout, still ephemeral
        because no counterfactual replay has run yet.
        """
        return cls(
            id="lr-2026-09-17-0001",
            type="failure-pattern",
            title="Unhandled 429 backoff snowballs into retry storms",
            body=RecordBody(
                signal="Latency spikes with rising 429 counts across consecutive rollouts",
                failure_mode="Bounded retries without jitter amplify rate-limit pressure",
                detection="Counterfactual replay flips the verifier verdict on the degraded case",
                fix_strategy="Add jittered exponential backoff with a per-rollout retry budget",
            ),
            status="ephemeral",
            confidence=0.62,
            provenance=RecordProvenance(
                proposed_by="reflector",
                origin="darwin-phase0-rollout",
                source="rollout-2791/degradation=rate_limiter",
                note="Proposed from the seed-pack degradation taxonomy; not yet replayed",
            ),
            counterfactual=RecordCounterfactual(
                replay_run_ids=[],
                verdict_change=None,
                evidence=None,
            ),
            created=datetime(2026, 9, 17, tzinfo=UTC),
            promoted_at=None,
        )
