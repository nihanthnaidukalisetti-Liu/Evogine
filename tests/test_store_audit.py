"""Append-only audit trail tests for the engine store."""

import json
import subprocess
from itertools import pairwise

from evogine.engine.store import Engine


def _audit_events(root):
    lines = (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_audit_records_every_transition(tmp_path, record_factory, evidence_factory):
    root = tmp_path / "store"
    engine = Engine(root)
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-a", "run-b"])
    engine.promote("lr-1", evidence_factory(["run-a", "run-b"]))
    engine.demote("lr-1", "verdict regression on the held-out family")

    events = _audit_events(root)
    assert [event["event"] for event in events] == [
        "propose",
        "queue_replay",
        "promote",
        "demote",
    ]
    assert [event["seq"] for event in events] == [1, 2, 3, 4]
    assert events[1]["run_ids"] == ["run-a", "run-b"]
    assert events[2]["from_status"] == "candidate"
    assert events[2]["to_status"] == "promoted"
    assert events[3]["from_status"] == "promoted"
    assert events[3]["to_status"] == "dropped"
    assert events[3]["reason"] == "verdict regression on the held-out family"


def test_audit_log_is_append_only(tmp_path, record_factory, evidence_factory):
    root = tmp_path / "store"
    engine = Engine(root)
    transitions = [
        lambda: engine.propose(record_factory("lr-1")),
        lambda: engine.queue_replay("lr-1", ["run-1"]),
        lambda: engine.promote("lr-1", evidence_factory(["run-1"])),
        lambda: engine.demote("lr-1", "rejected at the memory gate"),
    ]

    snapshots = []
    for transition in transitions:
        transition()
        snapshots.append((root / "audit.jsonl").read_bytes())

    # Every earlier snapshot must be a byte-prefix of every later one: the
    # trail is only ever appended to, never truncated or rewritten.
    for earlier, later in pairwise(snapshots):
        assert later.startswith(earlier), "audit log was rewritten, not appended"


def test_audit_seq_continues_across_engine_instances(tmp_path, record_factory):
    root = tmp_path / "store"
    first = Engine(root)
    first.propose(record_factory("lr-1"))
    second = Engine(root)
    second.propose(record_factory("lr-2"))

    events = _audit_events(root)
    assert [event["seq"] for event in events] == [1, 2]
    assert second.get("lr-2").status == "ephemeral"


def test_every_transition_is_a_git_commit(tmp_path, record_factory, evidence_factory):
    root = tmp_path / "store"
    engine = Engine(root)
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    engine.promote("lr-1", evidence_factory(["run-1"]))
    engine.demote("lr-1", "rejected at the memory gate")

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return proc.stdout

    # One commit per lifecycle transition, and the store always committed
    # clean — history is the append-only backstop for the audit trail.
    assert int(git("rev-list", "--count", "HEAD").strip()) == 4
    assert git("status", "--porcelain").strip() == ""
