"""Patch application tests: safety, atomicity, and hunk placement."""

from pathlib import Path

import pytest

from evogine.swarm.patches import (
    DiffApplicationError,
    DiffFormatError,
    UnsafeDiffError,
    UnsupportedDiffError,
    apply_unified_diff,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_applies_creation(tmp_path: Path) -> None:
    diff = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n"

    touched = apply_unified_diff(tmp_path, diff)

    assert touched == [tmp_path / "new.py"]
    assert (tmp_path / "new.py").read_text() == "a = 1\nb = 2\n"


def test_applies_creation_in_new_subdirectory(tmp_path: Path) -> None:
    diff = "--- /dev/null\n+++ b/pkg/mod.py\n@@ -0,0 +1 @@\n+x = 1\n"

    apply_unified_diff(tmp_path, diff)

    assert (tmp_path / "pkg" / "mod.py").read_text() == "x = 1\n"


def test_applies_modification(tmp_path: Path) -> None:
    _write(tmp_path / "app.py", "def f():\n    return 1\n")
    diff = (
        "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n"
        " def f():\n-    return 1\n+    return 2\n"
    )

    apply_unified_diff(tmp_path, diff)

    assert (tmp_path / "app.py").read_text() == "def f():\n    return 2\n"


def test_relocates_hunk_when_header_has_drifted(tmp_path: Path) -> None:
    _write(tmp_path / "app.py", "one\ntwo\nthree\nfour\nfive\n")
    # Header claims old line 1; the context actually sits at line 3.
    diff = (
        "--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,3 @@\n"
        " three\n-four\n+FOUR\n five\n"
    )

    apply_unified_diff(tmp_path, diff)

    assert (tmp_path / "app.py").read_text() == "one\ntwo\nthree\nFOUR\nfive\n"


def test_preserves_missing_trailing_newline(tmp_path: Path) -> None:
    _write(tmp_path / "app.py", "a\nb")  # no trailing newline
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n"

    apply_unified_diff(tmp_path, diff)

    assert (tmp_path / "app.py").read_text() == "a\nc"


def test_rejects_ambiguous_context(tmp_path: Path) -> None:
    _write(tmp_path / "app.py", "x\nx\n")
    # Header points past end of file and the old line matches two places:
    # the applier must refuse rather than guess.
    diff = "--- a/app.py\n+++ b/app.py\n@@ -3,1 +3,1 @@\n-x\n+y\n"

    with pytest.raises(DiffApplicationError, match="refusing to guess"):
        apply_unified_diff(tmp_path, diff)


def test_rejects_missing_context_and_leaves_file_untouched(tmp_path: Path) -> None:
    original = "a\nb\n"
    _write(tmp_path / "app.py", original)
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n a\n-Z\n+b\n"

    with pytest.raises(DiffApplicationError, match="context not found"):
        apply_unified_diff(tmp_path, diff)

    assert (tmp_path / "app.py").read_text() == original


def test_rejects_path_traversal(tmp_path: Path) -> None:
    diff = "--- /dev/null\n+++ b/../escape.txt\n@@ -0,0 +1 @@\n+nope\n"

    with pytest.raises(UnsafeDiffError, match="traversal|escapes"):
        apply_unified_diff(tmp_path, diff)

    assert not (tmp_path.parent / "escape.txt").exists()


def test_rejects_absolute_paths(tmp_path: Path) -> None:
    diff = "--- /dev/null\n+++ /etc/passwd\n@@ -0,0 +1 @@\n+nope\n"

    with pytest.raises(UnsafeDiffError, match="unsafe path"):
        apply_unified_diff(tmp_path, diff)


def test_rejects_deletions(tmp_path: Path) -> None:
    diff = "--- a/app.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n"

    with pytest.raises(UnsupportedDiffError, match="deletion"):
        apply_unified_diff(tmp_path, diff)


def test_rejects_no_newline_markers(tmp_path: Path) -> None:
    _write(tmp_path / "app.py", "a\nb")
    diff = (
        "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n a\n-b\n"
        "\\ No newline at end of file\n+b\n"
    )

    with pytest.raises(UnsupportedDiffError, match="No newline"):
        apply_unified_diff(tmp_path, diff)


def test_rejects_count_mismatch(tmp_path: Path) -> None:
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n a\n"

    with pytest.raises(DiffFormatError, match="promises"):
        apply_unified_diff(tmp_path, diff)


def test_never_partially_applies_a_multi_file_diff(tmp_path: Path) -> None:
    _write(tmp_path / "b.txt", "target exists\n")
    diff = (
        "--- /dev/null\n+++ b/ok.txt\n@@ -0,0 +1 @@\n+fine\n"
        "--- /dev/null\n+++ b/b.txt\n@@ -0,0 +1 @@\n+clash\n"
    )

    with pytest.raises(DiffApplicationError, match="already exists"):
        apply_unified_diff(tmp_path, diff)

    assert not (tmp_path / "ok.txt").exists()  # first file was not written


def test_rejects_same_file_patched_twice(tmp_path: Path) -> None:
    diff = (
        "--- /dev/null\n+++ b/x.txt\n@@ -0,0 +1 @@\n+one\n"
        "--- /dev/null\n+++ b/x.txt\n@@ -0,0 +1 @@\n+two\n"
    )

    with pytest.raises(UnsupportedDiffError, match="more than once"):
        apply_unified_diff(tmp_path, diff)


def test_rejects_non_diff_input(tmp_path: Path) -> None:
    with pytest.raises(DiffFormatError, match="no file patches"):
        apply_unified_diff(tmp_path, "hello world\n")
