"""Lifecycle-transition unit tests for the git-backed engine store."""

import pytest

from evogine.engine.store import Engine, StoreError


def test_propose_stores_record_as_ephemeral(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    record = engine.propose(record_factory("lr-1"))
    assert record.status == "ephemeral"
    assert engine.get("lr-1") == record


def test_store_persists_across_engine_instances(tmp_path, record_factory):
    root = tmp_path / "store"
    engine = Engine(root)
    engine.propose(record_factory("lr-1"))
    reopened = Engine(root)
    assert reopened.get("lr-1").status == "ephemeral"


def test_propose_rejects_duplicate_id(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="already exists"):
        engine.propose(record_factory("lr-1"))


def test_propose_rejects_non_ephemeral_draft(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    draft = record_factory("lr-1").model_copy(update={"status": "promoted"})
    with pytest.raises(StoreError, match="ephemeral"):
        engine.propose(draft)


def test_propose_rejects_draft_carrying_replay_evidence(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    draft = record_factory("lr-1").model_copy(
        update={"counterfactual": evidence_factory(["run-1"])}
    )
    with pytest.raises(StoreError, match="replay evidence"):
        engine.propose(draft)


def test_propose_rejects_path_traversing_id(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    with pytest.raises(StoreError, match="id"):
        engine.propose(record_factory("../../etc/passwd"))
    with pytest.raises(StoreError, match="unknown learning id"):
        engine.get("../../etc/passwd")
    assert list((tmp_path / "store" / "records").iterdir()) == []


def test_queue_replay_moves_ephemeral_to_candidate(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-a", "run-b"])
    record = engine.get("lr-1")
    assert record.status == "candidate"
    assert record.counterfactual.replay_run_ids == ["run-a", "run-b"]


def test_queue_replay_rejects_non_ephemeral(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-a"])
    with pytest.raises(StoreError, match="ephemeral"):
        engine.queue_replay("lr-1", ["run-b"])


def test_queue_replay_rejects_empty_run_ids(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="replay run id"):
        engine.queue_replay("lr-1", [])


def test_promote_requires_candidate(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="candidate"):
        engine.promote("lr-1", evidence_factory(["run-1"]))


def test_promote_stamps_evidence_and_promoted_at(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    record = engine.promote("lr-1", evidence_factory(["run-1"]))
    assert record.status == "promoted"
    assert record.promoted_at is not None
    assert record.counterfactual.verdict_change is True
    assert record.counterfactual.replay_run_ids == ["run-1"]
    assert engine.get("lr-1").promoted_at == record.promoted_at


def test_promote_requires_material_verdict_change(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    for evidence in (
        {"replay_run_ids": ["run-1"], "verdict_change": False, "evidence": "runs/r.json"},
        {"replay_run_ids": ["run-1"], "verdict_change": None, "evidence": "runs/r.json"},
    ):
        with pytest.raises(StoreError, match="verdict change"):
            engine.promote("lr-1", evidence)


def test_promote_requires_evidence_run_ids(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    with pytest.raises(StoreError, match="at least one replay run"):
        engine.promote("lr-1", {"replay_run_ids": [], "verdict_change": True})


def test_promote_evidence_must_cover_queued_runs(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-a", "run-b"])
    with pytest.raises(StoreError, match="queued replay runs"):
        engine.promote("lr-1", evidence_factory(["run-a"]))


def test_demote_moves_to_dropped_and_keeps_record(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    root = tmp_path / "store"
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    dropped = engine.demote("lr-1", "gate rejected: no verdict flip")
    assert dropped.status == "dropped"
    # Rejected knowledge stays in the store, auditable, never deleted.
    assert engine.get("lr-1").status == "dropped"
    assert (root / "records" / "lr-1.json").exists()


def test_demote_from_promoted_is_a_recorded_retraction(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    engine.promote("lr-1", evidence_factory(["run-1"]))
    assert engine.demote("lr-1", "retracted: evidence invalid").status == "dropped"


def test_demote_twice_raises(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.demote("lr-1", "first rejection")
    with pytest.raises(StoreError, match="already dropped"):
        engine.demote("lr-1", "second rejection")


def test_demote_requires_reason(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="reason"):
        engine.demote("lr-1", "   ")


def test_get_unknown_id_raises(tmp_path):
    engine = Engine(tmp_path / "store")
    with pytest.raises(StoreError, match="unknown learning id"):
        engine.get("lr-missing")
