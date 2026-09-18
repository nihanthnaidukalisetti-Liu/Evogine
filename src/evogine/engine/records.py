"""Learning records: the durable knowledge atom of the engine.

A :class:`LearningRecord` is the single versioned, git-backed payload the
engine stores, gates, and retrieves. The v1 lifecycle moves a record through
``ephemeral -> candidate -> promoted`` (or ``dropped``); the memory gate
promotes a candidate only when counterfactual replay demonstrates a material
verdict change. Every record carries provenance and confidence so later
phases can audit *why* the engine believes something.
"""

from __future__ import annotations

import math
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


class MergedSource(BaseModel):
    """Provenance of one duplicate absorbed by this record on merge.

    Stored verbatim — the absorbed record's provenance is never rewritten
    or summarized, so consolidation loses no lineage. V2-5's locked
    non-goal is recursive summarization; this is the record-level
    alternative: full provenance travels with the survivor.
    """

    record_id: str
    title: str
    provenance: RecordProvenance
    confidence: float = Field(ge=0.0, le=1.0)
    merged_at: datetime


class DecayConfig(BaseModel):
    """Configuration for the V2-5 confidence-decay curve.

    Exponential half-life decay with an asymptotic floor: confidence is
    never pushed below ``floor``, so even fully-aged knowledge stays a
    visible, auditable belief. ``half_life_days > 0`` keeps the curve
    well-defined; ``floor < 1`` keeps decay a decay.
    """

    half_life_days: float = Field(default=30.0, gt=0.0)
    floor: float = Field(default=0.05, ge=0.0, lt=1.0)


def decayed_confidence(
    confidence: float,
    *,
    since: datetime,
    now: datetime,
    config: DecayConfig,
) -> float:
    """Confidence after exponential half-life decay with an asymptotic floor.

    ``floor + (confidence - floor) * 2 ** (-age / half_life)``: unchanged at
    zero age, halved above the floor at one half-life, never below the
    configured floor. Decay only lowers — a confidence already under the
    floor is left alone rather than lifted back toward it, and a ``now``
    before ``since`` changes nothing.
    """
    if now <= since:
        return confidence
    age_days = (now - since).total_seconds() / 86400
    decayed = config.floor + (confidence - config.floor) * math.pow(
        2.0, -age_days / config.half_life_days
    )
    return min(confidence, decayed)


class LearningRecord(BaseModel):
    """A versioned unit of engine knowledge.

    Exactly one of ``failure-pattern``, ``playbook``, or ``check``, with a
    confidence in the closed interval ``[0, 1]`` and a lifecycle status.

    V2-5 extends the v1 lifecycle with new states without breaking the
    round-trip contract: ``merged`` marks a record absorbed into a
    duplicate-free survivor (``merged_into`` names it; body and provenance
    stay verbatim), and ``retired`` shelves a stale record reversibly —
    ``retired_from`` remembers its prior status so reinstatement restores
    it exactly. All new fields default, so v1 JSON validates unchanged.
    """

    id: str
    type: Literal["failure-pattern", "playbook", "check"]
    title: str
    body: RecordBody
    status: Literal[
        "ephemeral", "candidate", "promoted", "dropped", "merged", "retired"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: RecordProvenance
    counterfactual: RecordCounterfactual
    created: datetime
    promoted_at: datetime | None = None
    # --- v2 lifecycle extensions (all defaulted: v1 round-trips intact) ---
    merged_sources: list[MergedSource] = Field(default_factory=list)
    merged_into: str | None = None
    confidence_updated_at: datetime | None = None
    evergreen: bool = False
    retired_at: datetime | None = None
    retired_from: str | None = None

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
