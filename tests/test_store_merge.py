"""Duplicate-merge tests: consolidation with provenance preserved.

V2-5 merge control: exact duplicates consolidate at the record level —
the survivor keeps its body verbatim, the absorbed record's provenance is
preserved in full on ``merged_sources``, and nothing is ever recursively
summarized. The absorbed record stays in the store, readable, never deleted.
"""

import json
import subprocess

import pytest

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine, StoreError


def propose_pair(engine: Engine, record_factory, primary_id: str, duplicate_id: str) -> None:
    """Two exact duplicates: same body, different ids and titles."""
    engine.propose(record_factory(primary_id))
    engine.propose(
        record_factory(duplicate_id).model_copy(update={"title": f"duplicate of {primary_id}"})
    )


def test_merge_keeps_survivor_body_verbatim(tmp_path, record_factory, evidence_factory):
    """The survivor's analysis is never rewritten: consolidation is record-level."""
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-1", "lr-2")
    engine.queue_replay("lr-1", ["run-1"])
    engine.promote("lr-1", evidence_factory(["run-1"]))
    original_body = engine.get("lr-1").body

    survivor = engine.merge("lr-1", "lr-2", "reflector proposed the same lesson twice")

    assert survivor.body == original_body
    assert survivor.status == "promoted"  # live state untouched by consolidation
    assert engine.get("lr-2").body == original_body  # absorbed record unchanged too


def test_merge_preserves_duplicate_provenance_verbatim(tmp_path, record_factory):
    """The absorbed record's provenance travels to the survivor, unmodified."""
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-1", "lr-2")
    duplicate = engine.get("lr-2")

    survivor = engine.merge("lr-1", "lr-2", "duplicate proposal")

    assert len(survivor.merged_sources) == 1
    source = survivor.merged_sources[0]
    assert source.record_id == "lr-2"
    assert source.provenance == duplicate.provenance  # verbatim, not summarized
    assert source.title == duplicate.title
    assert source.confidence == duplicate.confidence
    assert source.merged_at is not None


def test_merge_transfers_provenance_from_an_absorbing_duplicate(tmp_path, record_factory):
    """A duplicate that had itself absorbed records hands that lineage on.

    Never a summary of a summary: the survivor ends up with one verbatim
    provenance entry per absorbed record.
    """
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-2", "lr-3")
    engine.merge("lr-2", "lr-3", "first consolidation")
    engine.propose(record_factory("lr-1"))

    survivor = engine.merge("lr-1", "lr-2", "second consolidation")

    assert [source.record_id for source in survivor.merged_sources] == ["lr-2", "lr-3"]


def test_merge_takes_max_confidence(tmp_path, record_factory):
    """Corroboration strengthens the survivor; it never weakens it."""
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.propose(record_factory("lr-2").model_copy(update={"confidence": 0.9}))

    survivor = engine.merge("lr-1", "lr-2", "same lesson, stronger evidence")

    assert survivor.confidence == 0.9


def test_merge_stamps_confidence_updated_at(tmp_path, record_factory):
    """A later decay pass anchors on the merge moment, not record creation."""
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-1", "lr-2")

    survivor = engine.merge("lr-1", "lr-2", "duplicate proposal")

    assert survivor.confidence_updated_at is not None


def test_find_duplicates_reports_exact_duplicate_pairs(tmp_path, record_factory):
    """Detection is deterministic: same type + identical body fields."""
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-a", "lr-b")
    engine.propose(record_factory("lr-c", topic="deadlock contention"))

    assert engine.find_duplicates() == [("lr-a", "lr-b")]

    engine.merge("lr-a", "lr-b", "duplicate proposal")
    assert engine.find_duplicates() == []  # the merged record is no longer live


def test_merge_requires_identical_body(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.propose(record_factory("lr-2", topic="deadlock contention"))
    with pytest.raises(StoreError, match="identical body"):
        engine.merge("lr-1", "lr-2", "not actually duplicates")


def test_merge_requires_same_type(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    playbook = record_factory("lr-2").model_copy(update={"type": "playbook"})
    engine.propose(playbook)
    with pytest.raises(StoreError, match="same type"):
        engine.merge("lr-1", "lr-2", "bodies match but types differ")


def test_merge_rejects_self_merge(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    with pytest.raises(StoreError, match="into itself"):
        engine.merge("lr-1", "lr-1", "one record")


def test_merge_requires_a_reason(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-1", "lr-2")
    with pytest.raises(StoreError, match="reason"):
        engine.merge("lr-1", "lr-2", "   ")


def test_merge_requires_live_records(tmp_path, record_factory):
    """Dropped and already-merged records are terminal: they cannot merge."""
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-1", "lr-2")
    engine.demote("lr-2", "rejected at the gate")
    with pytest.raises(StoreError, match="live duplicate"):
        engine.merge("lr-1", "lr-2", "rejected records do not merge")

    propose_pair(engine, record_factory, "lr-3", "lr-4")
    engine.merge("lr-3", "lr-4", "first merge")
    with pytest.raises(StoreError, match="live duplicate"):
        engine.merge("lr-1", "lr-4", "absorbed records do not merge again")


def test_merged_duplicate_stays_readable_and_undeleted(tmp_path, record_factory):
    """Absorption never deletes: the record file survives with full lineage."""
    engine = Engine(tmp_path / "store")
    root = tmp_path / "store"
    propose_pair(engine, record_factory, "lr-1", "lr-2")

    engine.merge("lr-1", "lr-2", "duplicate proposal")

    assert (root / "records" / "lr-2.json").exists()
    absorbed = engine.get("lr-2")
    assert absorbed.status == "merged"
    assert absorbed.merged_into == "lr-1"
    assert absorbed.provenance == record_factory("lr-2").provenance


def test_merge_is_audited_and_committed(tmp_path, record_factory):
    engine = Engine(tmp_path / "store")
    root = tmp_path / "store"
    propose_pair(engine, record_factory, "lr-1", "lr-2")

    engine.merge("lr-1", "lr-2", "duplicate proposal")

    events = [
        json.loads(line)
        for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events] == ["propose", "propose", "merge"]
    assert events[2]["record_id"] == "lr-2"
    assert events[2]["from_status"] == "ephemeral"
    assert events[2]["to_status"] == "merged"
    assert events[2]["reason"] == "merged into lr-1: duplicate proposal"

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return proc.stdout

    assert int(git("rev-list", "--count", "HEAD").strip()) == 3
    assert git("status", "--porcelain").strip() == ""


def test_merged_sources_round_trip_json(tmp_path, record_factory):
    """A survivor with absorbed provenance survives a JSON round trip intact."""
    engine = Engine(tmp_path / "store")
    propose_pair(engine, record_factory, "lr-1", "lr-2")
    survivor = engine.merge("lr-1", "lr-2", "duplicate proposal")

    restored = engine.get("lr-1")
    assert restored == survivor
    raw = survivor.model_dump_json()
    assert type(survivor).model_validate_json(raw) == survivor


def test_v1_record_json_still_validates_with_v2_fields(tmp_path, record_factory):
    """A v1-shaped record (no v2 fields) validates: extensions are additive."""
    payload = record_factory("lr-1").model_dump()
    assert "merged_sources" not in payload or payload["merged_sources"] == []
    for v2_field in ("merged_sources", "merged_into", "confidence_updated_at"):
        payload.pop(v2_field, None)  # strip back to the v1 wire shape

    record = LearningRecord.model_validate(payload)

    assert record.status == "ephemeral"
    assert record.merged_sources == []
    assert record.merged_into is None
    assert record.confidence_updated_at is None
