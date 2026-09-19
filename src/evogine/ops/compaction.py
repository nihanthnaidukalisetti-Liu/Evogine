"""Store compaction: a derived, self-invalidating index for hot read paths.

A store's reads are honest the hard way: every query parses every record
file (O(n) pydantic parses per lookup), because correctness is the source
file and nothing else. At retrieval-scale that is fine; at 10k records it
is the store's only superlinear cost. :mod:`evogine.ops.compaction` adds
a **derived** index — ``index.jsonl``, one line per record plus a header —
so the hot read paths (:meth:`Engine.promoted_records`,
:meth:`Engine.candidates`, :meth:`Engine.find_duplicates`) can skip
parsing records they don't need.

The index never becomes a second source of truth:

- **Derived**: it is rebuilt from the record files and says so — a header
  pins the index format version and the store schema version it was built
  against.
- **Self-invalidating**: every entry carries the stat fingerprint (size,
  ``mtime_ns``) of the file it mirrors. On read, the index is trusted
  only if the file count matches AND every fingerprint matches; ANY
  drift — one new record, one touched file, a schema migration — drops
  the whole index and the read falls back to the honest full scan. Stale
  answers are impossible; the fallback is the pre-index behavior.
- **Corruption = absence**: a malformed index line is a fallback, never a
  failed read. Derived state has no right to break the data it was
  derived from.
- **Audited**: compaction commits the index like any other store change
  and runs git's garbage collection — the one place repacking is safe,
  because the commit graph's meaning is preserved byte-for-byte at HEAD.

Index operations assume the store's single-writer discipline (the same
advisory lock every store operation takes); they are read-only on the
records and never rewrite the audit trail.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evogine.engine.records import LearningRecord
from evogine.ops.migrations import read_schema_version

#: Derived-state module: warnings here mean a rebuild (not data loss).
logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.jsonl"
"""The derived index, one JSON line per record after a header line."""

INDEX_FORMAT_VERSION = 1
"""Bumped when the index layout changes; older indexes are simply stale."""


class CompactionError(Exception):
    """Compaction refuses: not a store, or git would not commit the result."""


class IndexEntry(BaseModel):
    """One record's summary plus the stat fingerprint that guards it."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: str
    type: str
    signal: str
    failure_mode: str
    detection: str
    fix_strategy: str
    size: int
    """The record file's byte size when the index was built."""
    mtime_ns: int
    """The record file's mtime in nanoseconds when the index was built."""


class StoreIndex(BaseModel):
    """The full parsed index: header facts plus one entry per record."""

    model_config = ConfigDict(frozen=True)

    format_version: int
    schema_version: int
    record_count: int
    entries: list[IndexEntry] = Field(default_factory=list)


class CompactionResult(BaseModel):
    """What one compaction pass did, with before/after evidence."""

    record_count: int
    index_path: str
    index_bytes: int
    store_bytes_before: int
    store_bytes_after: int
    head_before: str
    head_after: str
    """HEAD is unchanged by compaction — asserted by the property tests."""


def index_path(root: Path | str) -> Path:
    """The store's derived index path."""
    return Path(root) / INDEX_FILENAME


def _stat_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _record_path(root: Path, record_id: str) -> Path:
    return root / "records" / f"{record_id}.json"


def build_index(root: Path | str) -> StoreIndex:
    """Rebuild the index from the record files (the honest source) and write it.

    Entries are ordered by record id — the same order ``records/`` glob
    order produces — so index-driven reads match full-scan reads exactly.
    """
    root_path = Path(root)
    records_dir = root_path / "records"
    entries: list[IndexEntry] = []
    for path in sorted(records_dir.glob("*.json")):
        record = LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))
        size, mtime_ns = _stat_fingerprint(path)
        entries.append(
            IndexEntry(
                id=record.id,
                status=record.status,
                type=record.type,
                signal=record.body.signal,
                failure_mode=record.body.failure_mode,
                detection=record.body.detection,
                fix_strategy=record.body.fix_strategy,
                size=size,
                mtime_ns=mtime_ns,
            )
        )
    store_index = StoreIndex(
        format_version=INDEX_FORMAT_VERSION,
        schema_version=read_schema_version(root_path),
        record_count=len(entries),
        entries=entries,
    )
    lines = [store_index.model_dump_json()] + [entry.model_dump_json() for entry in entries]
    index = index_path(root_path)
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return store_index


def load_index_if_fresh(root: Path | str) -> StoreIndex | None:
    """The parsed index, or ``None`` when absent, stale, or malformed.

    ``None`` means "do not trust it" — callers fall back to the full scan.
    Freshness: the header must match this build's index format and the
    store's current schema version, the record-file count must equal the
    entry count, and every entry's stat fingerprint must match its file.
    """
    path = index_path(root)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    try:
        header = StoreIndex.model_validate_json(lines[0])
        entries = [IndexEntry.model_validate_json(line) for line in lines[1:]]
    except ValueError:
        # Corrupt derived state: the fallback is the honest full scan.
        logger.warning("store index %s is malformed; ignoring it", path)
        return None
    if header.format_version != INDEX_FORMAT_VERSION:
        return None
    if header.schema_version != read_schema_version(root):
        return None
    if header.record_count != len(entries):
        return None
    root_path = Path(root)
    record_files = list((root_path / "records").glob("*.json"))
    if len(record_files) != len(entries):
        return None
    for entry in entries:
        record_file = _record_path(root_path, entry.id)
        if not record_file.exists():
            return None
        if _stat_fingerprint(record_file) != (entry.size, entry.mtime_ns):
            return None
    return StoreIndex(
        format_version=header.format_version,
        schema_version=header.schema_version,
        record_count=header.record_count,
        entries=entries,
    )


def _store_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _git_head(root: Path) -> str:
    """Current HEAD sha, or "" before the store's first commit (unborn)."""
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git(root: Path, *args: str) -> None:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evogine Engine",
            "-c",
            "user.email=engine@evogine.local",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CompactionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")


def compact_store(root: Path | str) -> CompactionResult:
    """Rebuild the index, commit it, and garbage-collect the store repo.

    The audit trail and record files are untouched: compaction adds one
    derived file and repacks git's object store. HEAD and the commit
    graph's meaning are preserved (property-tested); the win is fewer
    parsed files on hot reads and a smaller ``.git``.
    """
    root_path = Path(root)
    if not (root_path / ".git").exists():
        raise CompactionError(f"not a store (no git repo at {root_path})")
    head_before = _git_head(root_path)
    store_bytes_before = _store_bytes(root_path)
    store_index = build_index(root_path)
    _git(root_path, "add", "-A", ".")
    _git(
        root_path,
        "commit",
        "-q",
        "-m",
        f"ops: rebuild store index ({store_index.record_count} records)",
    )
    _git(root_path, "gc", "-q", "--auto")
    store_bytes_after = _store_bytes(root_path)
    return CompactionResult(
        record_count=store_index.record_count,
        index_path=str(index_path(root_path)),
        index_bytes=index_path(root_path).stat().st_size,
        store_bytes_before=store_bytes_before,
        store_bytes_after=store_bytes_after,
        head_before=head_before,
        head_after=_git_head(root_path),
    )


__all__ = [
    "INDEX_FILENAME",
    "INDEX_FORMAT_VERSION",
    "CompactionError",
    "CompactionResult",
    "IndexEntry",
    "StoreIndex",
    "build_index",
    "compact_store",
    "index_path",
    "load_index_if_fresh",
]
