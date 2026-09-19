"""Backup/restore round-trips: byte-identical trees, history included."""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import pytest
from conftest import make_record

from evogine.engine.records import RecordCounterfactual
from evogine.engine.store import Engine
from evogine.ops.backup import (
    BACKUP_SUFFIX,
    BackupError,
    create_backup,
    restore_backup,
)


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Every file under root, by relative path — the byte-identity oracle."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _build_populated_store(tmp_path: Path) -> Engine:
    engine = Engine(tmp_path / "store")
    for i in range(4):
        engine.propose(make_record(f"lr-{i}"))
        engine.queue_replay(f"lr-{i}", [f"run-{i}-a", f"run-{i}-b"])
    # Promote one record through the full boundary so the store spans
    # several lifecycle states and commit types.
    engine.promote(
        "lr-0",
        RecordCounterfactual(replay_run_ids=["run-0-a", "run-0-b"], verdict_change=True),
    )
    return engine


def test_backup_restore_round_trip_is_byte_identical(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    archive = tmp_path / f"store{BACKUP_SUFFIX}"

    result = create_backup(engine.root, archive)

    assert archive.is_file()
    assert result.file_count > 0
    assert result.archive_bytes == archive.stat().st_size

    restored_root = restore_backup(archive, tmp_path / "restored")

    # Byte identity across every file — records, audit, schema, index, .git.
    assert _tree_bytes(restored_root) == _tree_bytes(engine.root)
    # The restored store IS a working store: same HEAD, intact history.
    assert _git(restored_root, "git", "rev-parse", "HEAD") == _git(
        engine.root, "git", "rev-parse", "HEAD"
    )
    subprocess.run(["git", "fsck", "--no-progress"], cwd=restored_root, check=True)
    # And it behaves as one: reads see the same records and audit trail.
    restored_engine = Engine(restored_root)
    assert [r.id for r in restored_engine.promoted_records()] == ["lr-0"]
    assert restored_engine.audit_path.read_bytes() == engine.audit_path.read_bytes()


def test_backup_refuses_a_destination_inside_the_store(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    inside = engine.root / "backups" / "store.tar"
    with pytest.raises(BackupError, match="outside the store"):
        create_backup(engine.root, inside)


def test_backup_refuses_a_non_store(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-store"
    bare.mkdir()
    with pytest.raises(BackupError, match="not a store"):
        create_backup(bare, tmp_path / "x.tar")


def test_restore_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    archive = tmp_path / "store.tar"
    create_backup(engine.root, archive)

    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "keep.txt").write_text("precious", encoding="utf-8")
    with pytest.raises(BackupError, match="non-empty"):
        restore_backup(archive, dest)
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "precious"


def test_restore_rejects_traversal_members(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    archive = tmp_path / "store.tar"
    create_backup(engine.root, archive)

    # Rewrite the archive with an escaped member name appended.
    evil = tmp_path / "evil.tar"
    with tarfile.open(archive, "r") as src, tarfile.open(evil, "w") as dst:
        for member in src.getmembers():
            fileobj = src.extractfile(member)
            if fileobj is not None:
                dst.addfile(member, fileobj)
        payload = b"owned"
        escaped = tarfile.TarInfo(name="../escaped.txt")
        escaped.size = len(payload)
        dst.addfile(escaped, io.BytesIO(payload))

    dest = tmp_path / "restored-evil"
    with pytest.raises(tarfile.OutsideDestinationError):
        restore_backup(evil, dest)
    assert not (tmp_path / "escaped.txt").exists()


def test_restore_reports_a_truncated_store(tmp_path: Path) -> None:
    engine = _build_populated_store(tmp_path)
    archive = tmp_path / "store.tar"
    create_backup(engine.root, archive)

    # An archive missing the audit trail must not restore as a store.
    stripped = tmp_path / "stripped.tar"
    with tarfile.open(archive, "r") as src, tarfile.open(stripped, "w") as dst:
        for member in src.getmembers():
            if member.name == "audit.jsonl":
                continue
            fileobj = src.extractfile(member)
            if fileobj is not None:
                dst.addfile(member, fileobj)

    with pytest.raises(BackupError, match="missing store file"):
        restore_backup(stripped, tmp_path / "restored-stripped")
