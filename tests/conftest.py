"""Shared test fixtures: deterministic, fully-populated record factories.

The suite is offline by contract — no live-model calls anywhere; these
factories are pure pydantic construction.
"""

from datetime import UTC, datetime

import pytest

from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)


def make_record(record_id: str, topic: str = "429 backoff") -> LearningRecord:
    """A realistic ephemeral draft; ``topic`` varies body text for retrieval."""
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=f"Unhandled {topic} escalates across rollouts",
        body=RecordBody(
            signal=f"rising {topic} counts across consecutive rollouts",
            failure_mode=f"{topic} amplifies without jittered backoff",
            detection="counterfactual replay flips the verifier verdict on the degraded case",
            fix_strategy="add jittered exponential backoff with a per-rollout retry budget",
        ),
        status="ephemeral",
        confidence=0.6,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="test-fixture",
            source="unit-test",
            note="deterministic fixture; never touches a live model",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 18, tzinfo=UTC),
        promoted_at=None,
    )


def replay_evidence(run_ids: list[str]) -> RecordCounterfactual:
    """Evidence the memory gate would pass after a material verdict flip."""
    return RecordCounterfactual(
        replay_run_ids=run_ids,
        verdict_change=True,
        evidence="runs/replay/cold-vs-warm.json",
    )


@pytest.fixture
def record_factory():
    """Factory for deterministic, fully-populated ephemeral drafts."""
    return make_record


@pytest.fixture
def evidence_factory():
    """Factory for passing (material verdict change) replay evidence."""
    return replay_evidence
