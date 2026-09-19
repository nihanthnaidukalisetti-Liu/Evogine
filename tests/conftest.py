"""Shared test fixtures: deterministic, fully-populated record factories.

The suite is offline by contract — no live-model calls anywhere; these
factories are pure pydantic construction.
"""

import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import settings

from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)

# Hypothesis profiles: the ``dev`` default keeps local runs fast; CI exports
# ``HYPOTHESIS_PROFILE=ci`` (see .github/workflows/ci.yml) to raise the
# example budget. Tests that pay for real processes per example (git commits,
# fresh interpreters) pin their own lower max_examples explicitly — the
# profile raises the in-memory suites, not the process-heavy ones.
settings.register_profile("dev", max_examples=30, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


def temp_store_root(prefix: str) -> Path:
    """Create a temp dir holding an initialized repo for a test store.

    ``git commit`` can spawn a detached background ``git gc``
    (``gc.autoDetach`` defaults to true) that keeps mutating ``.git/objects``
    after the commit returns — racing ``shutil.rmtree`` cleanup with
    ``OSError [Errno 39] Directory not empty`` in slower CI environments.
    Repo-local config wins over the ambient default, so the helper inits the
    repo and disables auto-gc there (production behavior untouched; the
    Engine skips its own init when ``.git`` already exists).
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    subprocess.run(
        ["git", "init", "-q"], cwd=root, check=True, capture_output=True, text=True
    )
    config = root / ".git" / "config"
    with config.open("a", encoding="utf-8") as fh:
        fh.write("[gc]\n\tauto = 0\n\tautoDetach = false\n")
    return root


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
