"""Tests for local evidence collection: git churn/failures and coverage gaps.

Every fixture here drives git directly with local commands only — no network,
no store writes — mirroring how the decomposer itself runs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evogine.swarm.decomposition import (
    DecompositionError,
    collect_coverage_gaps,
    collect_git_evidence,
    is_failure_subject,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A scratch git repository, isolated from the sandbox's own config."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def _commit(repo: Path, filename: str, content: str, message: str) -> None:
    path = repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repo, check=True, capture_output=True)
    # gc.auto=0 keeps fixture commits free of background maintenance work.
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=tester",
            "-c",
            "user.email=tester@example.com",
            "-c",
            "gc.auto=0",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_churn_and_failure_counts_come_from_history(git_repo):
    _commit(git_repo, "alpha.py", "A = 1\n", "initial setup")
    _commit(git_repo, "alpha.py", "A = 2\n", "fix: crash on empty input")
    _commit(git_repo, "beta.py", "B = 1\n", "docs: add beta notes")

    evidence = collect_git_evidence(git_repo)

    assert evidence.churn == {"alpha.py": 2, "beta.py": 1}
    assert evidence.failures == {"alpha.py": 1}


def test_head_commit_is_recorded_when_history_exists(git_repo):
    _commit(git_repo, "alpha.py", "A = 1\n", "initial setup")

    evidence = collect_git_evidence(git_repo)

    assert evidence.head_commit is not None
    assert len(evidence.head_commit) == 40
    assert all(char in "0123456789abcdef" for char in evidence.head_commit)


def test_non_git_directory_yields_honest_zero(tmp_path):
    evidence = collect_git_evidence(tmp_path)

    assert evidence.churn == {}
    assert evidence.failures == {}
    assert evidence.head_commit is None
    assert collect_coverage_gaps(tmp_path) == {}


def test_git_directory_without_commits_is_honest_zero(git_repo):
    evidence = collect_git_evidence(git_repo)

    assert evidence.churn == {}
    assert evidence.failures == {}
    assert evidence.head_commit is None


def test_failure_subjects_are_word_boundaries_and_case_insensitive():
    assert is_failure_subject("fix: handle empty input")
    assert is_failure_subject("Fix the crash")
    assert is_failure_subject('Revert "feat: flaky retry"')
    assert is_failure_subject("hotfix rollout")
    # Conservative on purpose: near-misses are not failures.
    assert not is_failure_subject("fixture handling improved")
    assert not is_failure_subject("fixt: typo")
    assert not is_failure_subject("bugfix for the retry loop")  # not the word "bug"
    assert not is_failure_subject("docs: add usage notes")


def test_coverage_gaps_round_trip_from_coverage_json(tmp_path):
    (tmp_path / "coverage.json").write_text(
        json.dumps(
            {
                "files": {
                    "src/a.py": {"summary": {"percent_covered": 82.5}},
                    "b.py": {"summary": {"percent_covered": 100.0}},
                }
            }
        ),
        encoding="utf-8",
    )

    assert collect_coverage_gaps(tmp_path) == {"src/a.py": 17.5, "b.py": 0.0}


def test_malformed_coverage_json_refuses(tmp_path):
    (tmp_path / "coverage.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(DecompositionError, match="cannot be read"):
        collect_coverage_gaps(tmp_path)


def test_coverage_json_without_a_files_mapping_refuses(tmp_path):
    (tmp_path / "coverage.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DecompositionError, match="no 'files' mapping"):
        collect_coverage_gaps(tmp_path)
