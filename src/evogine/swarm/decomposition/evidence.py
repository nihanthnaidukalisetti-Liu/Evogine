"""Evidence collection: local git history and an existing ``coverage.json``.

Fully offline by contract — two local ``git`` queries (log, rev-parse; no
network, no store writes) and one optional file read. Evidence is additive:
a repository without git history, or without a ``coverage.json``, decomposes
with honest zeros (:func:`collect_git_evidence` returns empty counts,
``head_commit=None``) and scoring simply weights the remaining signals. A
``coverage.json`` that exists but cannot be parsed is different: it is
present evidence that contradicts itself, so it refuses loudly
(:class:`~evogine.swarm.decomposition.errors.DecompositionError`) rather
than silently decomposing as if the file were absent.

Churn counts come from ``git log --name-only`` over the current HEAD's
history, so a checked-out branch's history is the evidence base — the same
state a swarm run would act on.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evogine.swarm.decomposition.errors import DecompositionError

_FAILURE_SUBJECT_PATTERN = re.compile(
    r"\b("
    r"fix|fixes|fixed|bug|bugs|regress|regressed|regression|regressions"
    r"|crash|crashes|crashed|revert|reverted|broken|hotfix"
    r")\b"
)
"""Whole-word failure markers for a commit subject. Exact words only, with
boundaries on both sides: ``fixture`` is not ``fix``, and ``bugfix`` is one
word that matches none of the alternatives — only the listed forms count,
keeping the failure signal conservative."""


def is_failure_subject(subject: str) -> bool:
    """Whether a commit subject reads as a failure fix (whole words only)."""
    return _FAILURE_SUBJECT_PATTERN.search(subject.lower()) is not None


class GitEvidence(BaseModel):
    """What the local git history says, keyed by repo-relative posix path."""

    model_config = ConfigDict(extra="forbid")

    churn: dict[str, int] = Field(default_factory=dict)
    failures: dict[str, int] = Field(default_factory=dict)
    head_commit: str | None = None


# RS/US control-character sentinels: a commit subject can contain any
# printable text, so its boundaries must not be a printable delimiter. Paths
# cannot contain these control characters, and git quotes anything exotic by
# default, so splitting the log stream on \\x1e is unambiguous.
_LOG_COMMIT_FORMAT = "%x1e%H%x1f%s"


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes] | None:
    """Run a read-only git query; ``None`` when git refuses the repository."""
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        # Git missing or unreadable cwd: treat as no evidence, the honest zero.
        return None


def collect_git_evidence(repo: Path) -> GitEvidence:
    """Per-path churn and failure counts from the current HEAD's history."""
    log = _run_git(
        repo, "log", "--no-renames", "--name-only", "--pretty=format:" + _LOG_COMMIT_FORMAT
    )
    if log is None or log.returncode != 0:
        # Not a git repository (or no commits yet): no evidence, not a guess.
        return GitEvidence()
    churn: dict[str, int] = {}
    failures: dict[str, int] = {}
    text = log.stdout.decode("utf-8", errors="replace")
    for record in text.split("\x1e"):
        lines = record.splitlines()
        if not lines or "\x1f" not in lines[0]:
            continue
        _sha, subject = lines[0].split("\x1f", 1)
        failure_commit = is_failure_subject(subject)
        for path in lines[1:]:
            path = path.strip()
            if not path:
                continue  # the blank line that separates commits in the stream
            churn[path] = churn.get(path, 0) + 1
            if failure_commit:
                failures[path] = failures.get(path, 0) + 1
    head = _run_git(repo, "rev-parse", "HEAD")
    head_commit: str | None = None
    if head is not None and head.returncode == 0:
        candidate = head.stdout.decode("ascii", errors="replace").strip()
        if re.fullmatch(r"[0-9a-f]{40}", candidate):
            head_commit = candidate
    return GitEvidence(churn=churn, failures=failures, head_commit=head_commit)


def collect_coverage_gaps(repo: Path) -> dict[str, float]:
    """Percent of statements missed per path, from ``coverage.json`` if present.

    Reads the ``coverage json`` format (``files.<path>.summary.percent_covered``)
    from the repository root. Absent file → empty dict; present-but-unreadable
    file → :class:`DecompositionError`.
    """
    coverage_path = repo / "coverage.json"
    if not coverage_path.is_file():
        return {}
    try:
        payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecompositionError(
            f"coverage.json at {coverage_path} exists but cannot be read: {exc}"
        ) from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        raise DecompositionError(
            f"coverage.json at {coverage_path} has no 'files' mapping — "
            "is it really `coverage json` output?"
        )
    gaps: dict[str, float] = {}
    for file_path in sorted(files):
        info = files[file_path]
        summary = info.get("summary") if isinstance(info, dict) else None
        percent = summary.get("percent_covered") if isinstance(summary, dict) else None
        if isinstance(percent, (int, float)):
            # percent_covered can exceed 100 under exotic context combinations;
            # a gap is never negative.
            gaps[file_path] = max(0.0, 100.0 - float(percent))
    return gaps
