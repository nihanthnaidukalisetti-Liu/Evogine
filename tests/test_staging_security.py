"""Staging isolation probes: path traversal must never escape the staging root.

Security-review C1 acceptance evidence. All probes run offline — real git
operations in temp directories, no Docker, no models.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from evogine.sandbox.staging import (
    PatchRejected,
    StagingError,
    apply_patch,
    create_staging_root,
    extract_baseline,
    validate_patch_paths,
    validate_relative_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAVERSAL_PATCH = """diff --git a/../../escape.txt b/../../escape.txt
new file mode 100644
--- /dev/null
+++ b/../../escape.txt
@@ -0,0 +1 @@
+pwned
"""

#: Reaches far above any plausible parent — the historical canary probe.
DEEP_TRAVERSAL_PATCH = """diff --git a/../../../evogine-traversal-escape b/../../../evogine-traversal-escape
new file mode 100644
--- /dev/null
+++ b/../../../evogine-traversal-escape
@@ -0,0 +1 @@
+pwned
"""

ABSOLUTE_PATCH = """diff --git a/etc-probe b/etc-probe
new file mode 100644
--- /dev/null
+++ /etc/evogine-absolute-probe
@@ -0,0 +1 @@
+pwned
"""

ADD_FILE_PATCH = """diff --git a/new_probe.py b/new_probe.py
new file mode 100644
--- /dev/null
+++ b/new_probe.py
@@ -0,0 +1 @@
+PROBE = "ok"
"""


@pytest.mark.parametrize(
    "patch",
    [TRAVERSAL_PATCH, DEEP_TRAVERSAL_PATCH, ABSOLUTE_PATCH],
    ids=["parent-traversal", "deep-traversal", "absolute-path"],
)
def test_traversal_patches_are_rejected(patch: str) -> None:
    with pytest.raises(PatchRejected):
        validate_patch_paths(patch)


def test_quoted_path_rejected() -> None:
    patch = ADD_FILE_PATCH.replace("+++ b/new_probe.py", '+++ "b/new probe.py"')
    with pytest.raises(PatchRejected):
        validate_patch_paths(patch)


def test_binary_patch_rejected() -> None:
    patch = "diff --git a/blob b/blob\nBinary files a/blob and b/blob differ\n"
    with pytest.raises(PatchRejected):
        validate_patch_paths(patch)


def test_valid_patch_paths_returned() -> None:
    patch = ADD_FILE_PATCH + ADD_FILE_PATCH.replace("new_probe.py", "other.py")
    assert validate_patch_paths(patch) == ["new_probe.py", "other.py"]


@pytest.mark.parametrize(
    "path",
    ["../x", "/etc/passwd", "a/../../b", "C:/evil", "dir/", ""],
)
def test_validate_relative_path_rejects_unsafe(path: str) -> None:
    with pytest.raises(PatchRejected):
        validate_relative_path(path)


@pytest.mark.parametrize(
    "path",
    ["src/evogine/__init__.py", "README.md", "tests/test_smoke.py"],
)
def test_validate_relative_path_accepts_plain_paths(path: str) -> None:
    assert validate_relative_path(path) == path


def test_apply_patch_rejects_traversal_before_touching_disk(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    with pytest.raises(PatchRejected):
        apply_patch(DEEP_TRAVERSAL_PATCH, tree)
    assert not (tmp_path / "escape.txt").exists()
    assert not Path("/tmp/evogine-traversal-escape").exists()


def test_apply_patch_adds_file_in_tree(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    apply_patch(ADD_FILE_PATCH, tree)
    assert (tree / "new_probe.py").read_text() == 'PROBE = "ok"\n'


def test_apply_patch_rejects_malformed(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    with pytest.raises(PatchRejected):
        apply_patch("not a patch at all", tree)


def test_staging_root_is_private() -> None:
    root = create_staging_root()
    try:
        assert (root.stat().st_mode & 0o777) == 0o700
    finally:
        shutil.rmtree(root)


def test_extract_baseline_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "baseline"
    extract_baseline(REPO_ROOT, "HEAD", dest)
    assert (dest / "pyproject.toml").is_file()
    assert not (dest / ".git").exists()
    apply_patch(ADD_FILE_PATCH, dest)  # git apply needs no repository
    assert (dest / "new_probe.py").exists()


def test_extract_baseline_rejects_bad_refs(tmp_path: Path) -> None:
    for bad in ("../elsewhere", "a..b", "-web-diff", "/etc"):
        with pytest.raises(StagingError):
            extract_baseline(REPO_ROOT, bad, tmp_path / "x")
        assert not (tmp_path / "x").exists()
