"""Retirement tests: reversible shelving that never hard-deletes.

V2-5 retire control: stale records move to a ``retired`` state — out of
retrieval, on disk, auditable — and ``reinstate`` restores them exactly.
The invariant under test: retirement is reversible, nothing is deleted.
"""

import json
import subprocess

import pytest

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine, StoreError


def _promoted(engine: Engine, record_factory, evidence_factory, record_id: str) -> LearningRecord:
    """Propose, replay, and promote one deterministic record."""
    engine.propose(record_factory(record_id))
    engine.queue_replay(record_id, ["run-1"])
    return engine.promote(record_id, evidence_factory(["run-1"]))


def _context_for(record: LearningRecord) -> str:
    return f"{record.title} {record.body.signal} {record.body.failure_mode}"


def test_retire_moves_promoted_to_retired(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    retired = engine.retire("lr-1", "stale: superseded by the jittered-backoff playbook")

    assert retired.status == "retired"
    assert retired.retired_from == "promoted"
    assert retired.retired_at is not None
    stored = engine.get("lr-1")
    assert stored.status == "retired"
    assert stored.retired_from == "promoted"


def test_retired_record_leaves_retrieval(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    record = _promoted(engine, record_factory, evidence_factory, "lr-1")
    context = _context_for(record)
    assert [r.id for r in engine.retrieve(context)] == ["lr-1"]  # live and retrieved

    engine.retire("lr-1", "stale: no longer applicable")

    assert engine.retrieve(context) == []  # shelved, out of warm runs


def test_retired_record_stays_on_disk_and_readable(tmp_path, record_factory, evidence_factory):
    """Auditable, never deleted: the record file and its lineage survive."""
    engine = Engine(tmp_path / "store")
    root = tmp_path / "store"
    record = _promoted(engine, record_factory, evidence_factory, "lr-1")

    engine.retire("lr-1", "stale")

    assert (root / "records" / "lr-1.json").exists()
    stored = engine.get("lr-1")
    assert stored.body == record.body
    assert stored.provenance == record.provenance
    assert stored.counterfactual == record.counterfactual


def test_reinstate_restores_the_record_exactly(tmp_path, record_factory, evidence_factory):
    """The retirement-reversibility invariant: reinstated == pre-retirement."""
    engine = Engine(tmp_path / "store")
    record = _promoted(engine, record_factory, evidence_factory, "lr-1")
    context = _context_for(record)

    engine.retire("lr-1", "stale")
    reinstated = engine.reinstate("lr-1")

    assert reinstated == record  # field-for-field, stamps cleared
    assert reinstated.retired_at is None
    assert reinstated.retired_from is None
    assert [r.id for r in engine.retrieve(context)] == ["lr-1"]  # back in warm runs


def test_reinstate_restores_the_prior_status(tmp_path, record_factory):
    """A retired candidate reinstates to candidate, not to some default."""
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])

    engine.retire("lr-1", "stale while awaiting replay")
    reinstated = engine.reinstate("lr-1")

    assert reinstated.status == "candidate"
    assert reinstated.counterfactual.replay_run_ids == ["run-1"]


def test_reinstate_requires_retired_state(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="retired record"):
        engine.reinstate("lr-1")


def test_retire_requires_live_state(tmp_path, record_factory, evidence_factory):
    """Dropped, merged, and already-retired records cannot retire again."""
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.demote("lr-1", "rejected at the gate")
    with pytest.raises(StoreError, match="live record"):
        engine.retire("lr-1", "rejected is already terminal")

    engine.propose(record_factory("lr-2"))
    engine.propose(record_factory("lr-3"))
    engine.merge("lr-2", "lr-3", "duplicate")
    with pytest.raises(StoreError, match="live record"):
        engine.retire("lr-3", "absorbed is already terminal")

    engine.retire("lr-2", "stale")
    with pytest.raises(StoreError, match="live record"):
        engine.retire("lr-2", "retired is already terminal")


def test_retire_requires_a_reason(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="reason"):
        engine.retire("lr-1", "   ")


def test_retire_and_reinstate_are_audited(tmp_path, record_factory, evidence_factory):
    root = tmp_path / "store"
    engine = Engine(root)
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    engine.retire("lr-1", "stale: superseded")
    engine.reinstate("lr-1")

    events = [
        json.loads(line)
        for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events][-2:] == ["retire", "reinstate"]
    assert events[-2]["from_status"] == "promoted"
    assert events[-2]["to_status"] == "retired"
    assert events[-2]["reason"] == "stale: superseded"
    assert events[-1]["from_status"] == "retired"
    assert events[-1]["to_status"] == "promoted"


def test_retire_and_reinstate_each_commit_and_leave_the_store_clean(
    tmp_path, record_factory, evidence_factory
):
    root = tmp_path / "store"
    engine = Engine(root)
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return proc.stdout

    engine.retire("lr-1", "stale")
    engine.reinstate("lr-1")

    # propose, queue, promote, retire, reinstate — one commit each.
    assert int(git("rev-list", "--count", "HEAD").strip()) == 5
    assert git("status", "--porcelain").strip() == ""


def test_retired_record_round_trips_json(tmp_path, record_factory, evidence_factory):
    """A retired record survives a JSON round trip with its stamps intact."""
    engine = Engine(tmp_path / "store")
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    retired = engine.retire("lr-1", "stale")
    raw = retired.model_dump_json()

    assert LearningRecord.model_validate_json(raw) == retired
    assert engine.get("lr-1").retired_from == "promoted"
