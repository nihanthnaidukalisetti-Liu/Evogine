"""Compaction: the derived index is exact, self-invalidating, and audited."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import make_record, temp_store_root
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.ops.compaction import (
    CompactionError,
    build_index,
    compact_store,
    index_path,
    load_index_if_fresh,
)


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _head(root: Path) -> str | None:
    """Current HEAD sha, or None before the store's first commit."""
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _build_populated_store(tmp_path: Path) -> Engine:
    engine = Engine(tmp_path / "store")
    for i in range(6):
        engine.propose(make_record(f"lr-{i}"))
        engine.queue_replay(f"lr-{i}", [f"run-{i}-a", f"run-{i}-b"])
    return engine


def test_build_index_writes_header_and_entries(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    engine.propose(make_record("lr-1"))
    store_index = build_index(engine.root)
    assert store_index.record_count == 1
    assert index_path(engine.root).exists()
    fresh = load_index_if_fresh(engine.root)
    assert fresh is not None
    assert fresh.record_count == 1
    assert fresh.entries[0].id == "lr-1"
    assert fresh.entries[0].status == "ephemeral"


def test_index_is_stale_after_a_write(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    build_index(engine.root)
    assert load_index_if_fresh(engine.root) is not None
    engine.propose(make_record("lr-new"))
    assert load_index_if_fresh(engine.root) is None  # one new file: whole index drops


def test_index_is_stale_after_a_record_changes(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    build_index(engine.root)
    engine.demote("lr-0", "obsolete")  # status rewrite touches the record file
    assert load_index_if_fresh(engine.root) is None


def test_corrupt_index_is_absence_never_a_failure(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    index_path(engine.root).write_text("not json\nnope\n", encoding="utf-8")
    assert load_index_if_fresh(engine.root) is None
    # Reads still work — the honest full scan answers instead.
    assert engine.promoted_records() == []


def test_index_is_stale_after_a_schema_bump(tmp_path: Path) -> None:
    from evogine.ops.migrations import SchemaState, write_schema

    engine = Engine(tmp_path / "store")
    build_index(engine.root)
    write_schema(engine.root, SchemaState(version=99))
    assert load_index_if_fresh(engine.root) is None


def test_compact_store_preserves_head_audit_and_records(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    head_before = _git(engine.root, "git", "rev-parse", "HEAD")
    audit_before = engine.audit_path.read_bytes()
    records_before = {
        path.name: path.read_bytes() for path in sorted(engine.records_dir.glob("*.json"))
    }

    result = compact_store(engine.root)

    assert result.record_count == 6
    # Pre-compaction history is preserved and extended by exactly the index
    # commit: HEAD moves (the audit at work) but everything before it holds.
    assert result.head_before == head_before
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head_before, result.head_after],
        cwd=engine.root,
        check=True,
    )
    assert engine.audit_path.read_bytes() == audit_before
    records_after = {
        path.name: path.read_bytes() for path in sorted(engine.records_dir.glob("*.json"))
    }
    assert records_after == records_before
    # The index is committed to the store's git history like any other change.
    assert "ops: rebuild store index (6 records)" in _git(engine.root, "git", "log", "--oneline")


def test_compact_store_refuses_a_non_store(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-store"
    bare.mkdir()
    with pytest.raises(CompactionError, match="not a store"):
        compact_store(bare)


def test_index_fast_paths_match_full_scan(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    for i in range(4):
        engine.propose(make_record(f"dup-{i}"))
    # Two exact duplicate bodies (same type + all four body fields).
    twin_a = make_record("twin-a")
    twin_b = make_record("twin-b").model_copy(update={"body": twin_a.body})
    engine.propose(twin_a)
    engine.propose(twin_b)
    engine.queue_replay("twin-a", ["run-1", "run-2"])

    slow_promoted = [r.id for r in engine.promoted_records()]
    slow_candidates = [r.id for r in engine.candidates()]
    slow_duplicates = engine.find_duplicates()

    build_index(engine.root)  # all three reads now take the index fast path

    assert [r.id for r in engine.promoted_records()] == slow_promoted
    assert [r.id for r in engine.candidates()] == slow_candidates
    assert engine.find_duplicates() == slow_duplicates

    # With no index at all, the full scan gives the same answers.
    index_path(engine.root).unlink()
    assert [r.id for r in engine.promoted_records()] == slow_promoted
    assert [r.id for r in engine.candidates()] == slow_candidates
    assert engine.find_duplicates() == slow_duplicates


# --------------------------------------------------------------- properties


@settings(max_examples=20, deadline=None)  # git commits per op: budget pinned
@given(
    ops=st.lists(
        st.tuples(st.sampled_from(["propose", "queue", "retire"]), st.integers(0, 7)),
        max_size=12,
    )
)
def test_property_compaction_never_corrupts_the_audit_trail(
    ops: list[tuple[str, int]],
) -> None:
    """Whatever lifecycle fuzz builds, compaction leaves the audit trail
    byte-identical and pre-compaction history intact as an ancestor."""
    root = temp_store_root("evogine-compaction-fuzz-")
    try:
        engine = Engine(root / "store")
        for op, i in ops:
            record_id = f"lr-{i % 8}"
            record_file = engine.root / "records" / f"{record_id}.json"
            if op == "propose":
                if not record_file.exists():
                    engine.propose(make_record(record_id))
                continue
            if not record_file.exists():
                continue
            record = LearningRecord.model_validate_json(record_file.read_text(encoding="utf-8"))
            if op == "queue" and record.status == "ephemeral":
                engine.queue_replay(record_id, [f"run-{i}"])
            elif op == "retire" and record.status in ("ephemeral", "candidate", "promoted"):
                engine.retire(record_id, "property test retire")
        head_before = _head(engine.root)
        audit_before = engine.audit_path.read_bytes() if engine.audit_path.exists() else None

        compact_store(engine.root)

        if audit_before is not None:
            assert engine.audit_path.read_bytes() == audit_before
        head_after = _head(engine.root)
        if head_before is not None:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", head_before, head_after],
                cwd=engine.root,
                check=True,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)
