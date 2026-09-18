"""Per-run artifact directories with crash-safe resume.

Every run gets its own directory under a runs root, named by a unique run
id (UTC timestamp plus a random token). Artifacts land inside that
directory and never outside it, so reruns can never clobber results: a
completed run directory is immutable — reopening one is refused — and new
work always starts in a fresh directory.

An incomplete run directory (a crashed or interrupted process) resumes in
place: reopening it by id keeps every recorded artifact, so a rerun
redoes only the work that is genuinely missing. Resume is explicit —
callers pass the run id; opening without one always starts fresh.

Writes are atomic (temporary file, then rename within the same
directory), so a crash mid-write leaves either the previous artifact or
no artifact — never a half-written one. Completion is marked by writing
``complete.marker`` last: a run directory without the marker is
resumable; with it, immutable.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path

#: Run ids and artifact names become path segments; hold them to a safe
#: shape so a hostile id can never traverse out of the runs root.
_SAFE_SEGMENT = re.compile(r"[0-9A-Za-z][0-9A-Za-z_-]*\Z")

_COMPLETION_MARKER = "complete.marker"


class RunCompleteError(ValueError):
    """A completed run directory was reopened — completed runs are immutable."""


def _checked_segment(name: str, kind: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(name):
        raise ValueError(f"unsafe {kind} {name!r}: must match {_SAFE_SEGMENT.pattern}")
    return name


class RunDirectory:
    """One run's artifact directory: unique, resumable, never clobbered."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = root
        self._run_id = _checked_segment(run_id, "run id")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def directory(self) -> Path:
        return self._root / self._run_id

    def artifact_path(self, name: str) -> Path:
        safe = _checked_segment(name, "artifact name")
        return self.directory / "artifacts" / f"{safe}.json"

    def read_artifact(self, name: str) -> str | None:
        """Return the artifact's content, or ``None`` when not yet recorded."""
        try:
            return self.artifact_path(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def write_artifact(self, name: str, content: str) -> None:
        """Record an artifact atomically; a crash never leaves a half write."""
        path = self.artifact_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def is_complete(self) -> bool:
        return (self.directory / _COMPLETION_MARKER).exists()

    def mark_complete(self) -> None:
        """Freeze the run: the marker lands last, atomically, and is final."""
        if self.is_complete():
            raise RunCompleteError(f"run {self._run_id!r} is already complete and immutable")
        self.directory.mkdir(parents=True, exist_ok=True)
        marker = self.directory / _COMPLETION_MARKER
        tmp = marker.with_name(f".{_COMPLETION_MARKER}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_text(f"{datetime.now(UTC).isoformat()}\n", encoding="utf-8")
            os.replace(tmp, marker)
        finally:
            tmp.unlink(missing_ok=True)


def open_run(root: Path, run_id: str | None = None) -> RunDirectory:
    """Open a run directory: start a fresh one, or resume an incomplete one.

    ``run_id=None`` always creates a new, uniquely named run directory —
    new work never writes into an old directory. Passing a run id resumes
    that run: the directory and its recorded artifacts are reused
    untouched (and created if the earlier process died before creating
    it). Reopening a completed run is refused — its evidence is immutable,
    and rerunning it means starting a new run.
    """
    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        fresh = RunDirectory(root, f"{stamp}-{secrets.token_hex(4)}")
        fresh.directory.mkdir(parents=True, exist_ok=True)
        return fresh
    handle = RunDirectory(root, run_id)
    if handle.is_complete():
        raise RunCompleteError(
            f"run {run_id!r} is complete and immutable — start a new run instead of reopening it"
        )
    handle.directory.mkdir(parents=True, exist_ok=True)
    return handle


__all__ = ["RunCompleteError", "RunDirectory", "open_run"]
