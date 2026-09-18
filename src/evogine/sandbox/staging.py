"""Path-traversal-safe staging of baseline + candidate trees (review C1).

A candidate patch is arbitrary text from an untrusted builder. Staging turns
it into an isolated directory pair — baseline tree, candidate tree — without
ever letting a patch path escape the staging root. ``git apply``'s own refusals
are kept as a second layer, not the only one.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

# Refs: HEAD, v1.0.0, a sha, origin/main — never option-like or escaping.
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,198}$")

# Patch paths: plain relative POSIX paths inside the evaluated tree only.
# Spaces, quotes, backslashes, drives, and leading dashes are refused.
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._+][A-Za-z0-9._+/_-]{0,4096}$")


class PatchRejected(ValueError):
    """The patch is unsafe or malformed and must not be staged."""


class StagingError(RuntimeError):
    """Staging infrastructure failed (missing ref, broken git, timeout)."""


def validate_ref(baseline_ref: str) -> str:
    """Accept only plain git refs; anything else fails closed."""
    if not _SAFE_REF.fullmatch(baseline_ref) or ".." in baseline_ref:
        raise StagingError(f"baseline_ref rejected: {baseline_ref[:50]!r}")
    return baseline_ref


def validate_relative_path(path: str) -> str:
    """Reject absolute, parent-relative, and otherwise exotic paths."""
    if not _SAFE_RELATIVE.fullmatch(path) or ".." in path.split("/") or path.endswith("/"):
        raise PatchRejected(f"unsafe path in patch: {path[:100]!r}")
    return path


def _git_subcommand(args: list[str]) -> str:
    """The first non-flag subcommand — ``-c key=value`` pairs precede it."""
    index = 0
    while index + 1 < len(args) and args[index] == "-c":
        index += 2
    return args[index]


def _run_git(args: list[str], *, cwd: Path | None = None) -> bytes:
    """Run git hermetically: no repo/user config, no shell, no host env."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        # Config files are nulled above, so the env-config channel is the only
        # way to reach git's config. The gate stages trees as repos on the
        # host, then the container (uid nobody) runs git inside them; git's
        # ownership refusal would otherwise reject every such call. All git
        # invocations here target repos WE staged or the caller's own checkout,
        # so the ownership check protects nothing in this path.
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "*",
        "LC_ALL": "C",
    }
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        raise StagingError("git is not available on the host") from exc
    except subprocess.TimeoutExpired as exc:
        raise StagingError(f"git {_git_subcommand(args)} timed out") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip()
        raise StagingError(f"git {_git_subcommand(args)} failed: {message[:200]}")
    return proc.stdout


def create_staging_root(prefix: str = "evogine-gate-") -> Path:
    """Fresh 0700 scratch root; every evaluation stages inside its own."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def extract_baseline(repo_root: Path, baseline_ref: str, dest: Path) -> Path:
    """Materialize ``baseline_ref``'s tree into ``dest`` via ``git archive``.

    Extraction uses tarfile's ``data`` filter so even a hostile archive entry
    cannot traverse out of ``dest``. The ref must resolve in ``repo_root``;
    shallow checkouts limit which refs resolve.
    """
    ref = validate_ref(baseline_ref)
    _run_git(["rev-parse", "--verify", ref + "^{commit}"], cwd=repo_root)
    archive = _run_git(["archive", ref], cwd=repo_root)
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def initialize_staged_repo(tree: Path) -> None:
    """Make a staged tree a self-contained git repository with a HEAD.

    The gate container runs the candidate suite against an archive-extracted
    tree, which has no ``.git`` — but the suite's own tests exercise
    git-backed paths against their repo root. A fresh, identity-pinned init
    makes every staged tree hermetic: no host refs, no host config, and no
    host identity leaks into the tree or back out of it.
    """
    _run_git(["init", "-q", "-b", "main"], cwd=tree)
    _run_git(["add", "-A"], cwd=tree)
    _run_git(
        [
            "-c",
            "user.name=evogine-gate",
            "-c",
            "user.email=gate@invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "evogine-gate staged tree",
        ],
        cwd=tree,
    )


def validate_patch_paths(patch: str) -> list[str]:
    """Every path a patch writes, validated; raise on the first unsafe one.

    Only plain relative POSIX paths inside the evaluated tree are accepted;
    quoted (git-escaped) paths and binary patches are refused outright.
    """
    paths: list[str] = []
    for raw in patch.splitlines():
        if raw.startswith(("GIT binary patch", "Binary files")):
            raise PatchRejected("binary patches are not verifiable — rejected")
        if not raw.startswith(("--- ", "+++ ")):
            continue
        token = raw[4:].split("\t")[0].strip()
        if token == "/dev/null":
            continue
        if token.startswith('"'):
            raise PatchRejected("quoted paths in patches are not accepted")
        if token.startswith(("a/", "b/")):
            token = token[2:]
        validate_relative_path(token)
        if token not in paths:
            paths.append(token)
    if not paths:
        raise PatchRejected("patch writes no files — nothing to verify")
    return paths


def apply_patch(patch: str, tree: Path) -> None:
    """Apply ``patch`` inside ``tree`` via git apply (no repo index involved)."""
    validate_patch_paths(patch)
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=patch.encode(),
        cwd=tree,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip()
        raise PatchRejected(f"patch does not apply cleanly: {message[:300]}")
