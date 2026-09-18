"""Fuzz and adversarial corpus for the sandbox diff applier (V3 seam).

``apply_unified_diff`` is the trust boundary for builder-proposed diffs
(PR #6): model output is untrusted text, and the applier's contract is that
every refusal is clean — computed in memory first, so a malformed, unsafe,
or non-matching diff raises :class:`PatchError` with the workspace
byte-for-byte unchanged. This suite fuzzes that contract three ways:

- **Adversarial corpus** — path traversals, absolute and Windows-style
  paths, symlink escapes, file deletions, ``\\ No newline at end of file``
  hunks, duplicate targets, ambiguous context (a hunk whose context matches
  more than one place), missing context, create-on-existing,
  patch-a-missing-file, malformed framing, and a missing workspace. Every
  case must end in the expected clean refusal, and the tree signature must
  be identical before and after: NEVER a partial write.
- **Two-phase atomicity** — a diff mixing sections that would apply cleanly
  with one poisoned section, in shuffled order, is refused whole: the valid
  sections must not land.
- **Benign control** — well-formed modifications and creations still apply
  exactly (guarding the corpus against degenerating into all-refusal), and
  a creation lands with the promised trailing newline.

All offline: files and strings only, no Docker, no network.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.swarm.patches import (
    DiffApplicationError,
    DiffFormatError,
    PatchError,
    UnsafeDiffError,
    UnsupportedDiffError,
    apply_unified_diff,
)


def _tree_signature(workspace: Path) -> str:
    """A digest over the whole tree: paths, symlink targets, and file bytes."""
    digest = hashlib.sha256()
    for path in sorted(workspace.rglob("*")):
        rel = path.relative_to(workspace).as_posix()
        if path.is_symlink():
            digest.update(f"{rel} -> {os.readlink(path)}\n".encode())
        elif path.is_dir():
            digest.update(f"{rel}/\n".encode())
        else:
            digest.update(f"{rel}\n".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _AdversarialCase:
    """One adversarial input: the diff, the refusal it must end in, the setup."""

    label: str
    diff: str
    expected: type[PatchError]
    setup_kind: str = "empty"  # dispatched by _prepare
    setup_payload: str = ""


# --------------------------------------------------------------- the corpus

_ESCAPE_PATHS = [
    "../../escape.txt",
    "../sibling-escape.txt",
    "sub/../../../escape.txt",
    "a/../../../escape.txt",
    "/etc/evogine-escape.txt",
    "C:\\tmp\\escape.txt",
    "C:escape.txt",
]


@st.composite
def _traversal_case(draw: st.DrawFn) -> _AdversarialCase:
    escape = draw(st.sampled_from(_ESCAPE_PATHS))
    prefixed = draw(st.booleans())
    old = f"a/{escape}" if prefixed else escape
    new = f"b/{escape}" if prefixed else escape
    create = draw(st.booleans())
    if create:
        diff = f"--- /dev/null\n+++ {new}\n@@ -0,0 +1,1 @@\n+owned\n"
    else:
        diff = f"--- {old}\n+++ {new}\n@@ -1,1 +1,1 @@\n-content\n+owned\n"
    return _AdversarialCase(
        label=f"traversal: {escape}",
        diff=diff,
        expected=UnsafeDiffError,
        setup_kind="plain_file",
    )


@st.composite
def _deletion_case(draw: st.DrawFn) -> _AdversarialCase:
    target = draw(st.sampled_from(["file.txt", "sub/dir/file.txt"]))
    return _AdversarialCase(
        label=f"deletion: {target}",
        diff=f"--- a/{target}\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-content\n",
        expected=UnsupportedDiffError,
        setup_kind="plain_file",
    )


@st.composite
def _no_newline_case(draw: st.DrawFn) -> _AdversarialCase:
    """'\\ No newline at end of file' hunks are refused, never misapplied."""
    marker_first = draw(st.booleans())
    body = (
        "@@ -1,1 +1,1 @@\n\\ No newline at end of file\n"
        if marker_first
        else "@@ -1,1 +1,1 @@\n-old\n\\ No newline at end of file\n+new\n"
    )
    return _AdversarialCase(
        label="no-newline marker",
        diff=f"--- a/file.txt\n+++ b/file.txt\n{body}",
        expected=UnsupportedDiffError,
        setup_kind="plain_file",
    )


@st.composite
def _duplicate_target_case(draw: st.DrawFn) -> _AdversarialCase:
    name = draw(st.sampled_from(["dup.txt", "sub/dir/dup.txt"]))
    return _AdversarialCase(
        label=f"duplicate target: {name}",
        diff=(
            f"--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,1 @@\n-one\n+uno\n"
            f"--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,1 @@\n-one\n+dos\n"
        ),
        expected=UnsupportedDiffError,
        setup_kind="plain_file",
    )


@st.composite
def _ambiguous_context_case(draw: st.DrawFn) -> _AdversarialCase:
    """Context matching two places is refused, never guessed."""
    first = draw(st.sampled_from(["alpha", "ready"]))
    second = draw(st.sampled_from(["beta", "set"]))
    far = draw(st.integers(min_value=90, max_value=500))
    noise = draw(st.sampled_from(["gamma", "tail"]))
    content = f"{first}\n{second}\n{first}\n{second}\n{noise}\n"
    return _AdversarialCase(
        label=f"ambiguous context at old line {far}",
        diff=(
            f"--- a/repeated.txt\n+++ b/repeated.txt\n"
            f"@@ -{far},2 +{far},2 @@\n {first}\n-{second}\n+PATCHED\n"
        ),
        expected=DiffApplicationError,
        setup_kind="repeated_context",
        setup_payload=content,
    )


def _missing_context_case() -> st.SearchStrategy[_AdversarialCase]:
    return st.just(
        _AdversarialCase(
            label="context not found",
            diff="--- a/file.txt\n+++ b/file.txt\n@@ -1,1 +1,1 @@\n-absent\n+new\n",
            expected=DiffApplicationError,
            setup_kind="plain_file",
        )
    )


@st.composite
def _malformed_case(draw: st.DrawFn) -> _AdversarialCase:
    name = draw(st.sampled_from(["one.txt", "sub/dir/two.txt"]))
    shape = draw(
        st.sampled_from(
            [
                "missing_plus",
                "no_hunks",
                "count_mismatch",
                "bad_body",
                "name_mismatch",
                "empty_diff",
                "not_a_diff",
            ]
        )
    )
    diffs = {
        "missing_plus": f"--- a/{name}\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        "no_hunks": f"--- a/{name}\n+++ b/{name}\n",
        "count_mismatch": f"--- a/{name}\n+++ b/{name}\n@@ -1,3 +1,3 @@\n-only\n+only\n",
        "bad_body": f"--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,1 @@\nx bogus\n",
        "name_mismatch": f"--- a/{name}\n+++ b/other.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        "empty_diff": "",
        "not_a_diff": "hello world\njust text\n",
    }
    return _AdversarialCase(
        label=f"malformed: {shape}", diff=diffs[shape], expected=DiffFormatError
    )


@st.composite
def _boundary_case(draw: st.DrawFn) -> _AdversarialCase:
    kind = draw(st.sampled_from(["create_existing", "patch_missing", "symlink_escape"]))
    cases = {
        "create_existing": _AdversarialCase(
            label="create over an existing file",
            diff="--- /dev/null\n+++ b/target.txt\n@@ -0,0 +1,1 @@\n+content\n",
            expected=DiffApplicationError,
            setup_kind="existing_create_target",
        ),
        "patch_missing": _AdversarialCase(
            label="patch a missing file",
            diff="--- a/ghost.txt\n+++ b/ghost.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n",
            expected=DiffApplicationError,
            setup_kind="empty",
        ),
        "symlink_escape": _AdversarialCase(
            label="symlink escape",
            diff=(
                "--- a/link/file.txt\n+++ b/link/file.txt\n"
                "@@ -1,1 +1,1 @@\n-outside\n+owned\n"
            ),
            expected=UnsafeDiffError,
            setup_kind="symlink_escape",
        ),
    }
    return cases[kind]


def _refusal_cases() -> st.SearchStrategy[_AdversarialCase]:
    return st.one_of(
        _traversal_case(),
        _deletion_case(),
        _no_newline_case(),
        _duplicate_target_case(),
        _ambiguous_context_case(),
        _missing_context_case(),
        _malformed_case(),
        _boundary_case(),
    )


def _prepare(case: _AdversarialCase, workroot: Path, workspace: Path) -> None:
    """Build the workspace state the case assumes (never outside workroot)."""
    if case.setup_kind == "plain_file":
        (workspace / "file.txt").write_text("content\n", encoding="utf-8")
    elif case.setup_kind == "existing_create_target":
        (workspace / "target.txt").write_text("already here\n", encoding="utf-8")
    elif case.setup_kind == "repeated_context":
        (workspace / "repeated.txt").write_text(case.setup_payload, encoding="utf-8")
    elif case.setup_kind == "symlink_escape":
        outside = workroot / "outside"
        outside.mkdir()
        (outside / "file.txt").write_text("outside\n", encoding="utf-8")
        (workspace / "link").symlink_to(outside, target_is_directory=True)


def test_missing_workspace_refuses_cleanly() -> None:
    """No workspace, no write: the applier refuses before parsing."""
    workroot = Path(tempfile.mkdtemp(prefix="evogine-patch-fuzz-"))
    try:
        with pytest.raises(DiffApplicationError, match="workspace does not exist"):
            apply_unified_diff(
                workroot / "no-such-dir", "--- a/x\n+++ b/x\n@@\n-a\n+b\n"
            )
    finally:
        shutil.rmtree(workroot)


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(case=_refusal_cases())
def test_adversarial_corpus_is_refused_with_the_workspace_untouched(
    case: _AdversarialCase,
) -> None:
    """Every adversarial input ends in its expected refusal — zero writes."""
    workroot = Path(tempfile.mkdtemp(prefix="evogine-patch-fuzz-"))
    workspace = workroot / "workspace"
    workspace.mkdir()
    try:
        _prepare(case, workroot, workspace)
        before = _tree_signature(workspace)

        with pytest.raises(case.expected):
            apply_unified_diff(workspace, case.diff)

        assert _tree_signature(workspace) == before, (
            f"{case.label}: the refusal was not clean — the workspace changed"
        )
    finally:
        shutil.rmtree(workroot)


# ------------------------------------------------------- two-phase atomicity


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(
    valid_indices=st.lists(
        st.integers(min_value=0, max_value=2), min_size=1, max_size=3, unique=True
    ),
    poison_position=st.integers(min_value=0, max_value=3),
)
def test_one_poisoned_section_refuses_the_whole_diff(
    valid_indices: list[int], poison_position: int
) -> None:
    """A diff with valid sections and one poisoned section writes NOTHING.

    The two-phase contract: every file's new content is computed in memory
    before the first byte is written, so a poisoned last section cannot ride
    in on cleanly-applied earlier sections.
    """
    workroot = Path(tempfile.mkdtemp(prefix="evogine-patch-two-phase-"))
    workspace = workroot / "workspace"
    workspace.mkdir()
    try:
        for index in range(3):
            (workspace / f"f{index}.txt").write_text(
                f"line-{index}-0\nline-{index}-1\n", encoding="utf-8"
            )
        (workspace / "poison.txt").write_text("unrelated content\n", encoding="utf-8")
        before = _tree_signature(workspace)

        sections = [
            f"--- a/f{index}.txt\n+++ b/f{index}.txt\n"
            f"@@ -1,1 +1,1 @@\n-line-{index}-0\n+line-{index}-PATCHED\n"
            for index in valid_indices
        ]
        poison = (
            "--- a/poison.txt\n+++ b/poison.txt\n"
            "@@ -1,1 +1,1 @@\n-absent-line\n+PATCHED\n"
        )
        sections.insert(min(poison_position, len(sections)), poison)

        with pytest.raises(DiffApplicationError):
            apply_unified_diff(workspace, "\n".join(sections))

        assert _tree_signature(workspace) == before, (
            "a valid section was written before the poisoned section refused"
        )
    finally:
        shutil.rmtree(workroot)


# ------------------------------------------------------------ benign control


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(
    target_index=st.integers(min_value=0, max_value=2),
    replacement=st.sampled_from(["patched line", "fixed handler", "jittered backoff"]),
)
def test_wellformed_modifications_apply_exactly(
    target_index: int, replacement: str
) -> None:
    """Control: the applier still applies — exactly, and only where told."""
    workroot = Path(tempfile.mkdtemp(prefix="evogine-patch-benign-"))
    workspace = workroot / "workspace"
    workspace.mkdir()
    try:
        for index in range(3):
            (workspace / f"f{index}.txt").write_text(
                f"line-{index}-0\nline-{index}-1\n", encoding="utf-8"
            )
        target = workspace / f"f{target_index}.txt"
        diff = (
            f"--- a/f{target_index}.txt\n+++ b/f{target_index}.txt\n"
            f"@@ -1,1 +1,1 @@\n-line-{target_index}-0\n+{replacement}\n"
        )

        touched = apply_unified_diff(workspace, diff)

        assert touched == [target]
        assert target.read_text(encoding="utf-8") == (
            f"{replacement}\nline-{target_index}-1\n"
        )
        for index in range(3):
            if index != target_index:
                other = workspace / f"f{index}.txt"
                assert other.read_text(encoding="utf-8") == (
                    f"line-{index}-0\nline-{index}-1\n"
                )
    finally:
        shutil.rmtree(workroot)


_WORDS = ["created line", "owned by the diff", "alpha beta"]


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(first=st.sampled_from(_WORDS), second=st.sampled_from(_WORDS))
def test_creations_land_with_a_trailing_newline(first: str, second: str) -> None:
    """Control: creations write the hunk's lines and the promised newline."""
    workroot = Path(tempfile.mkdtemp(prefix="evogine-patch-create-"))
    workspace = workroot / "workspace"
    workspace.mkdir()
    try:
        diff = f"--- /dev/null\n+++ b/new-file.txt\n@@ -0,0 +1,2 @@\n+{first}\n+{second}\n"

        touched = apply_unified_diff(workspace, diff)

        assert touched == [workspace / "new-file.txt"]
        assert (workspace / "new-file.txt").read_text(encoding="utf-8") == (
            f"{first}\n{second}\n"
        )
    finally:
        shutil.rmtree(workroot)
