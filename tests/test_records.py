"""LearningRecord contract tests: JSON round-trip and shape enforcement."""

import pytest
from pydantic import ValidationError

from evogine.engine.records import LearningRecord


def test_learning_record_json_round_trip() -> None:
    """A sample record must survive a JSON serialize/validate round trip unchanged."""
    record = LearningRecord.sample()
    raw = record.model_dump_json()
    restored = LearningRecord.model_validate_json(raw)
    assert restored == record


def test_learning_record_rejects_invalid_type() -> None:
    """``type`` accepts only failure-pattern, playbook, or check."""
    payload = LearningRecord.sample().model_dump()
    payload["type"] = "gossip"
    with pytest.raises(ValidationError, match="type"):
        LearningRecord.model_validate(payload)


def test_confidence_bounds_enforced() -> None:
    """``confidence`` is pinned to the closed interval [0, 1]."""
    payload = LearningRecord.sample().model_dump()
    payload["confidence"] = 1.5
    with pytest.raises(ValidationError, match="confidence"):
        LearningRecord.model_validate(payload)
