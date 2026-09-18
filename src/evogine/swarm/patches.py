"""Safe application of builder-proposed unified diffs to a workspace tree.

Model output is untrusted text, so this module accepts only well-formed
unified diffs and applies them only inside the given workspace root:
absolute paths, ``..`` traversal, and symlink escapes are rejected; file
deletions and ``\\ No newline at end of file`` hunks are refused rather than
misapplied. Application is two-phase — every file's new content is computed
in memory first — so a malformed or non-matching hunk raises before a single
byte is written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    """Base class for patch parsing and application failures."""


class DiffFormatError(PatchError):
    """The diff text is not a well-formed unified diff."""


class UnsupportedDiffError(PatchError):
    """The diff is well-formed but uses a feature this applier refuses."""


class UnsafeDiffError(PatchError):
    """The diff touches paths outside the workspace or through traversal."""


class DiffApplicationError(PatchError):
    """The diff does not cleanly match the workspace content."""


@dataclass(frozen=True)
class _Hunk:
    """One ``@@`` hunk: the old lines to find and the new lines to write."""

    old_start: int
    old_lines: list[str]
    new_lines: list[str]


@dataclass(frozen=True)
class _FilePatch:
    """All hunks for one target file."""

    path: str
    create: bool
    hunks: list[_Hunk]


def apply_unified_diff(workspace: Path, diff: str) -> list[Path]:
    """Apply a unified ``diff`` inside ``workspace``; return the touched paths.

    Two-phase: every file's new content is computed in memory before anything
    is written, so any parse, safety, or context failure raises
    :class:`PatchError` with the workspace untouched.
    """
    if not workspace.is_dir():
        raise DiffApplicationError(f"workspace does not exist: {workspace}")
    file_patches = _parse_unified_diff(diff)
    seen: set[str] = set()
    for patch in file_patches:
        if patch.path in seen:
            raise UnsupportedDiffError(f"diff patches {patch.path!r} more than once")
        seen.add(patch.path)
    planned = [_plan_file(workspace, patch) for patch in file_patches]
    touched: list[Path] = []
    for target, text, create in planned:
        if create:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        touched.append(target)
    return touched


def _parse_unified_diff(diff: str) -> list[_FilePatch]:
    """Parse a unified diff into per-file patches, verifying hunk counts."""
    lines = diff.splitlines()
    file_patches: list[_FilePatch] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1  # tolerate git's 'diff --git' / 'index' framing lines
            continue
        old_path = _split_header_path(lines[i][4:])
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise DiffFormatError(f"+++ header missing after --- at line {i + 1}")
        new_path = _split_header_path(lines[i + 1][4:])
        i += 2
        if new_path is None:
            raise UnsupportedDiffError("file deletions are not supported")
        if old_path is None:
            create = True
            path = new_path
        else:
            create = False
            if old_path != new_path:
                raise DiffFormatError(
                    f"--- and +++ name different files: {old_path!r} vs {new_path!r}"
                )
            path = old_path
        hunks, i = _parse_hunks(lines, i)
        if not hunks:
            raise DiffFormatError(f"no hunks found for {path!r}")
        file_patches.append(_FilePatch(path=path, create=create, hunks=hunks))
    if not file_patches:
        raise DiffFormatError("diff contains no file patches")
    return file_patches


def _split_header_path(raw: str) -> str | None:
    """Return a header path stripped of timestamp and ``a/``/``b/`` prefix.

    ``None`` means the ``/dev/null`` sentinel.
    """
    path = raw.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _parse_hunks(lines: list[str], start: int) -> tuple[list[_Hunk], int]:
    """Parse ``@@`` hunks from ``lines[start:]`` until the first non-hunk line."""
    hunks: list[_Hunk] = []
    i = start
    while i < len(lines):
        header = _HUNK_HEADER.match(lines[i])
        if header is None:
            break
        old_start = int(header.group(1))
        old_count = int(header.group(2)) if header.group(2) is not None else 1
        new_count = int(header.group(4)) if header.group(4) is not None else 1
        i += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        seen_old = 0
        seen_new = 0
        while i < len(lines) and (seen_old < old_count or seen_new < new_count):
            line = lines[i]
            if line.startswith("\\"):
                raise UnsupportedDiffError(
                    "'\\ No newline at end of file' diffs are not supported"
                )
            if not line or line[0] not in " +-":
                raise DiffFormatError(f"malformed hunk body at line {i + 1}: {line!r}")
            tag, content = line[0], line[1:]
            if tag == " ":
                old_lines.append(content)
                new_lines.append(content)
                seen_old += 1
                seen_new += 1
            elif tag == "-":
                old_lines.append(content)
                seen_old += 1
            else:
                new_lines.append(content)
                seen_new += 1
            i += 1
        if seen_old != old_count or seen_new != new_count:
            raise DiffFormatError(
                f"hunk at old line {old_start} promises {old_count}/{new_count} "
                f"old/new lines, body has {seen_old}/{seen_new}"
            )
        hunks.append(_Hunk(old_start=old_start, old_lines=old_lines, new_lines=new_lines))
    return hunks, i


def _plan_file(workspace: Path, patch: _FilePatch) -> tuple[Path, str, bool]:
    """Compute the post-patch content for one file without writing it."""
    target = _safe_target(workspace, patch.path)
    if patch.create:
        if target.exists():
            raise DiffApplicationError(f"cannot create {patch.path}: it already exists")
        original = ""
        trailing_newline = True  # files this applier creates end with a newline
    else:
        if not target.is_file():
            raise DiffApplicationError(f"cannot patch missing file {patch.path}")
        original = target.read_text(encoding="utf-8")
        trailing_newline = original.endswith("\n")
    lines = original.splitlines()
    offset = 0
    for hunk in patch.hunks:
        position = _locate(lines, hunk, offset)
        lines[position : position + len(hunk.old_lines)] = hunk.new_lines
        offset += len(hunk.new_lines) - len(hunk.old_lines)
    text = "\n".join(lines)
    if trailing_newline and text:
        text += "\n"
    return target, text, patch.create


def _locate(lines: list[str], hunk: _Hunk, offset: int) -> int:
    """Return the 0-based offset where ``hunk`` applies within ``lines``.

    Trusts the hunk header first, falls back to a unique whole-file search;
    ambiguous or missing context is refused rather than guessed at.
    """
    if not hunk.old_lines:
        # '-l,0' insertion hunks insert after old line l (0-based index l).
        return max(0, min(hunk.old_start + offset, len(lines)))
    position = hunk.old_start - 1 + offset
    if _matches(lines, position, hunk.old_lines):
        return position
    occurrences = [
        i
        for i in range(len(lines) - len(hunk.old_lines) + 1)
        if _matches(lines, i, hunk.old_lines)
    ]
    if len(occurrences) == 1:
        return occurrences[0]
    if not occurrences:
        raise DiffApplicationError(f"context not found for hunk at old line {hunk.old_start}")
    raise DiffApplicationError(
        f"context for hunk at old line {hunk.old_start} matches {len(occurrences)} "
        "places; refusing to guess"
    )


def _matches(lines: list[str], position: int, expected: list[str]) -> bool:
    if position < 0 or position + len(expected) > len(lines):
        return False
    return lines[position : position + len(expected)] == expected


def _safe_target(workspace: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside ``workspace``, refusing anything unsafe."""
    if not rel_path or rel_path.startswith(("/", "\\")) or ":" in rel_path:
        raise UnsafeDiffError(f"unsafe path in diff: {rel_path!r}")
    if ".." in Path(rel_path).parts:
        raise UnsafeDiffError(f"path traversal in diff: {rel_path!r}")
    candidate = workspace / rel_path
    if not candidate.resolve().is_relative_to(workspace.resolve()):
        raise UnsafeDiffError(f"path escapes the workspace: {rel_path!r}")
    return candidate
