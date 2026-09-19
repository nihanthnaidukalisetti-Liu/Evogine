"""Engine wiring: transition metrics and the schema gate ride the store."""

from __future__ import annotations

from pathlib import Path

import pytest

from evogine.engine.store import Engine, StoreError
from evogine.ops.metrics import (
    STORE_TRANSITIONS_TOTAL,
    metrics,
    reset_metrics,
)
from evogine.ops.migrations import (
    CURRENT_SCHEMA_VERSION,
    SchemaState,
    read_schema_version,
    write_schema,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    reset_metrics()


def test_propose_counts_one_transition(tmp_path: Path, record_factory) -> None:
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    snapshot = metrics().snapshot()
    assert (
        snapshot.counter(
            STORE_TRANSITIONS_TOTAL,
            {"event": "propose", "from_status": "none", "to_status": "ephemeral"},
        )
        == 1
    )


def test_full_lifecycle_counts_match_the_audit_trail(
    tmp_path: Path, record_factory, evidence_factory
) -> None:
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-a", "run-b"])
    engine.promote("lr-1", evidence_factory(["run-a", "run-b"]))
    engine.demote("lr-1", "obsolete")
    audit_lines = engine.audit_path.read_text(encoding="utf-8").strip().splitlines()
    snapshot = metrics().snapshot()
    counted = sum(
        series.value for series in snapshot.counters if series.name == STORE_TRANSITIONS_TOTAL
    )
    assert counted == len(audit_lines) == 4


def test_decay_counts_its_actual_transitions(tmp_path: Path, record_factory) -> None:
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))  # ephemeral: decay never touches it
    engine.decay()
    assert (
        metrics()
        .snapshot()
        .counter(
            STORE_TRANSITIONS_TOTAL,
            {"event": "decay", "from_status": "promoted", "to_status": "promoted"},
        )
        == 0
    )


def test_schema_gate_blocks_engine_open(tmp_path: Path) -> None:
    root = tmp_path / "store"
    Engine(root)
    write_schema(root, SchemaState(version=CURRENT_SCHEMA_VERSION + 3))
    with pytest.raises(StoreError, match="forward-only"):
        Engine(root)
    assert read_schema_version(root) == CURRENT_SCHEMA_VERSION + 3
