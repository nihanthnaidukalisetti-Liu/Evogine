"""Backup and restore: complete, byte-identical store snapshots.

The store root is self-contained — records, the append-only audit trail,
the schema stamp, the derived index, and the full ``.git`` history live
under one directory — so a complete backup is exactly a consistent copy
of that directory. An uncompressed tar keeps the archive inspectable
with stock tools, needs no new dependencies, and preserves file bytes
exactly, which is what makes the round-trip byte-identical.

Restore is fail-closed: it refuses to overwrite a non-empty destination,
extracts with Python's data-safe tar filter (absolute paths and ``..``
traversal are rejected), and verifies store post-conditions (git history,
schema stamp, records) before reporting success.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evogine.engine.store import StoreError, store_lock

BACKUP_SUFFIX = ".evogine-backup.tar"
"""Conventional archive extension (uncompressed tar)."""


class BackupError(StoreError):
    """Raised when a backup or restore cannot be completed safely."""


class BackupResult(BaseModel):
    """What one backup run wrote, for the CLI to echo and tests to assert."""

    model_config = ConfigDict(frozen=True)

    archive: str
    """Absolute path of the written archive."""
    file_count: int
    """Number of files captured, including everything under ``.git``."""
    archive_bytes: int
    """Size of the written archive in bytes."""


def _require_store(root: Path) -> None:
    if not (root / ".git").is_dir():
        raise BackupError(f"not a store (missing .git): {root}")


def create_backup(root: Path | str, dest: Path | str) -> BackupResult:
    """Snapshot the store at ``root`` into an uncompressed tar at ``dest``.

    Holds the store's advisory lock for the snapshot window so the
    documented read-modify-write critical sections cannot interleave
    with archive construction. The archive captures every file under the
    store root, including ``.git`` — restoring it yields a byte-identical
    store with its complete git history.
    """
    root_path = Path(root).resolve()
    _require_store(root_path)
    dest_path = Path(dest).resolve()
    if dest_path == root_path or root_path in dest_path.parents:
        raise BackupError(f"backup destination must live outside the store: {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with store_lock(root_path):
        files = sorted(path for path in root_path.rglob("*") if path.is_file())
        with tarfile.open(dest_path, "w") as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(root_path)))

    return BackupResult(
        archive=str(dest_path),
        file_count=len(files),
        archive_bytes=dest_path.stat().st_size,
    )


def restore_backup(archive: Path | str, dest: Path | str) -> Path:
    """Restore a backup archive into the (empty or absent) directory ``dest``.

    Returns the restored store root. Refuses a non-empty destination and
    any archive member that would extract outside it; verifies the
    extracted tree is a complete store before returning.
    """
    archive_path = Path(archive).resolve()
    dest_path = Path(dest).resolve()
    if not archive_path.is_file():
        raise BackupError(f"backup archive not found: {archive_path}")
    if dest_path.exists() and any(dest_path.iterdir()):
        raise BackupError(f"refusing to restore into a non-empty directory: {dest_path}")

    dest_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        # ``data`` rejects absolute paths, ``..`` traversal, and devices —
        # the archive-slip guard; members outside dest never hit disk.
        tar.extractall(dest_path, filter="data")

    _require_store(dest_path)
    for required in ("schema.json", "records", "audit.jsonl"):
        if not (dest_path / required).exists():
            raise BackupError(f"restored tree is missing store file: {required}")
    return dest_path


__all__ = [
    "BACKUP_SUFFIX",
    "BackupError",
    "BackupResult",
    "create_backup",
    "restore_backup",
]
