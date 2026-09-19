"""The low-conflict merge surface: one JSON file per record, append-only audit.

Per-store audit order is preserved by git itself — a merge receives each
side's lines verbatim, never invents a global ordering across stores, and
treats the git history as the reconciliation record. This module pins the
merge driver's contract: disjoint appends interleave cleanly, duplicate
appends collapse, and any attempt to rewrite history refuses visibly —
both at the function level and through a real ``git merge`` with the
driver wired via ``.gitattributes``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from evogine.sync.merge import main as merge_main
from evogine.sync.merge import merge_audit_lines

BASE = '{"seq":1,"event":"propose","record_id":"lr-1"}\n'
OURS_APPEND = '{"seq":2,"event":"propose","record_id":"lr-ours"}\n'
THEIRS_APPEND = '{"seq":3,"event":"propose","record_id":"lr-theirs"}\n'

# The driver command is environment-proof: it re-injects PYTHONPATH pointing
# at this checkout's src/ tree and uses the running interpreter explicitly.
# Git-spawned merge drivers run with the REPO's working directory, so a bare
# `python3 -m evogine.sync.merge` loses the candidate tree whenever the
# ambient environment does not already carry an importable evogine (the
# hermetic evaluator runs the suite uninstalled, imports resolved via
# PYTHONPATH) — the driver then dies with ModuleNotFoundError and git
# reports a bare conflict. sys.executable is the same interpreter running
# the tests; SRC is this checkout's source root in any staging layout.
SRC = Path(__file__).resolve().parents[1] / "src"
DRIVER = f"env PYTHONPATH={SRC} {sys.executable} -m evogine.sync.merge %O %A %B"


def test_disjoint_appends_keep_each_sides_order() -> None:
    merged = merge_audit_lines(BASE, BASE + OURS_APPEND, BASE + THEIRS_APPEND)
    assert merged == BASE + OURS_APPEND + THEIRS_APPEND


def test_duplicate_append_appears_once() -> None:
    merged = merge_audit_lines(BASE, BASE + OURS_APPEND, BASE + OURS_APPEND + THEIRS_APPEND)
    assert merged == BASE + OURS_APPEND + THEIRS_APPEND


def test_identical_sides_are_stable() -> None:
    assert (
        merge_audit_lines(BASE + OURS_APPEND, BASE + OURS_APPEND, BASE + OURS_APPEND)
        == BASE + OURS_APPEND
    )


@pytest.mark.parametrize("side", ["ours", "theirs"])
def test_history_rewrite_refuses(side: str) -> None:
    rewritten = '{"seq":1,"event":"propose","record_id":"lr-REWRITTEN"}\n'
    ours = BASE + OURS_APPEND if side == "ours" else BASE
    theirs = BASE if side == "ours" else BASE + THEIRS_APPEND
    base = rewritten
    with pytest.raises(ValueError, match="append-only"):
        merge_audit_lines(base, ours, theirs)


def test_cli_merges_in_place(tmp_path: Path) -> None:
    base, ours, theirs = tmp_path / "base", tmp_path / "ours", tmp_path / "theirs"
    base.write_text(BASE, encoding="utf-8")
    ours.write_text(BASE + OURS_APPEND, encoding="utf-8")
    theirs.write_text(BASE + THEIRS_APPEND, encoding="utf-8")
    assert merge_main([str(base), str(ours), str(theirs)]) == 0
    assert ours.read_text(encoding="utf-8") == BASE + OURS_APPEND + THEIRS_APPEND


def test_cli_refuses_rewrite_with_visible_error(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    base, ours, theirs = tmp_path / "base", tmp_path / "ours", tmp_path / "theirs"
    base.write_text(BASE, encoding="utf-8")
    ours.write_text('{"seq":1,"event":"propose","record_id":"lr-REWRITTEN"}\n', encoding="utf-8")
    theirs.write_text(BASE, encoding="utf-8")
    assert merge_main([str(base), str(ours), str(theirs)]) == 1
    assert "append-only" in capsys.readouterr().err
    # The file is left as git delivered it — ours, untouched.
    assert "lr-REWRITTEN" in ours.read_text(encoding="utf-8")


def _git(*args: str, cwd: Path) -> None:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode:
        # Surface git's own output — a failed merge driver's stderr reaches
        # us only through here, and a bare rc hides the actual cause.
        raise AssertionError(
            f"git {' '.join(args)} failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _commit_all(cwd: Path, message: str) -> None:
    _git("add", "-A", cwd=cwd)
    _git("commit", "-m", message, cwd=cwd)


def test_real_git_merge_wires_the_driver(tmp_path: Path) -> None:
    """A real two-branch merge resolves the audit driver-side, cleanly."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "merge-test@example.com", cwd=repo)
    _git("config", "user.name", "Merge Test", cwd=repo)
    _git(
        "config",
        "merge.evogine-audit.driver",
        DRIVER,
        cwd=repo,
    )
    (repo / ".gitattributes").write_text("audit.jsonl merge=evogine-audit\n", encoding="utf-8")
    (repo / "audit.jsonl").write_text(BASE, encoding="utf-8")
    _commit_all(repo, "seed")

    _git("checkout", "-b", "side", cwd=repo)
    with open(repo / "audit.jsonl", "a", encoding="utf-8") as audit:
        audit.write(THEIRS_APPEND)
    _commit_all(repo, "side appends")

    _git("checkout", "main", cwd=repo)
    with open(repo / "audit.jsonl", "a", encoding="utf-8") as audit:
        audit.write(OURS_APPEND)
    _commit_all(repo, "main appends")

    _git("merge", "side", "-m", "merge side", cwd=repo)
    merged = (repo / "audit.jsonl").read_text(encoding="utf-8")
    assert merged == BASE + OURS_APPEND + THEIRS_APPEND


def test_real_git_merge_keeps_history_rewrite_visible(tmp_path: Path) -> None:
    """A rewritten base line must surface as a git conflict, never silently."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "merge-test@example.com", cwd=repo)
    _git("config", "user.name", "Merge Test", cwd=repo)
    _git(
        "config",
        "merge.evogine-audit.driver",
        DRIVER,
        cwd=repo,
    )
    (repo / ".gitattributes").write_text("audit.jsonl merge=evogine-audit\n", encoding="utf-8")
    (repo / "audit.jsonl").write_text(BASE, encoding="utf-8")
    _commit_all(repo, "seed")

    _git("checkout", "-b", "side", cwd=repo)
    (repo / "audit.jsonl").write_text(
        '{"seq":1,"event":"propose","record_id":"lr-REWRITTEN"}\n', encoding="utf-8"
    )
    _commit_all(repo, "side rewrites history")

    _git("checkout", "main", cwd=repo)
    with open(repo / "audit.jsonl", "a", encoding="utf-8") as audit:
        audit.write(OURS_APPEND)
    _commit_all(repo, "main appends")

    merge = subprocess.run(
        ["git", "merge", "side", "-m", "merge side"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge.returncode != 0
    assert "CONFLICT" in merge.stdout + merge.stderr
