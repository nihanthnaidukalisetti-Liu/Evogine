"""Run-directory units: unique dirs, atomic artifacts, crash-safe resume.

Generic directory semantics are tested directly; the harness integration
proves a resumed rerun reuses recorded outcomes without re-executing
anything (the runner explodes if it is ever called) and that reruns never
clobber completed runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_harness import ScriptedRunner, StubRetriever

from evogine.bench.cases import BenchCase
from evogine.bench.harness import REPEAT_FLOOR, BenchHarness, RunArtifacts
from evogine.bench.rundirs import RunCompleteError, open_run
from evogine.engine.records import LearningRecord


def test_open_run_creates_unique_directories(tmp_path: Path) -> None:
    first = open_run(tmp_path)
    second = open_run(tmp_path)
    assert first.run_id != second.run_id
    assert first.directory.exists() and second.directory.exists()


def test_artifact_roundtrip_and_missing_reads(tmp_path: Path) -> None:
    run = open_run(tmp_path)
    assert run.read_artifact("missing") is None
    run.write_artifact("report", '{"verdict": "ok"}')
    assert run.read_artifact("report") == '{"verdict": "ok"}'
    assert run.artifact_path("report").exists()


def test_unsafe_run_ids_and_artifact_names_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe run id"):
        open_run(tmp_path, run_id="../escape")
    with pytest.raises(ValueError, match="unsafe run id"):
        open_run(tmp_path, run_id="a/b")
    run = open_run(tmp_path)
    with pytest.raises(ValueError, match="unsafe artifact name"):
        run.write_artifact("../../escape", "x")


def test_resume_keeps_recorded_artifacts(tmp_path: Path) -> None:
    run = open_run(tmp_path)
    run.write_artifact("spent", "evidence")
    resumed = open_run(tmp_path, run_id=run.run_id)
    assert resumed.directory == run.directory
    assert resumed.read_artifact("spent") == "evidence"


def test_resume_of_unknown_id_starts_fresh(tmp_path: Path) -> None:
    """A rerun whose first process died before creating the dir still works."""
    resumed = open_run(tmp_path, run_id="20260918T000000-deadbeef")
    assert resumed.directory.exists()
    assert not resumed.is_complete()


def test_completed_runs_are_immutable(tmp_path: Path) -> None:
    run = open_run(tmp_path)
    run.write_artifact("spent", "evidence")
    run.mark_complete()
    assert run.is_complete()
    with pytest.raises(RunCompleteError, match="immutable"):
        open_run(tmp_path, run_id=run.run_id)
    with pytest.raises(RunCompleteError, match="immutable"):
        run.mark_complete()
    assert run.read_artifact("spent") == "evidence"


def _stub_cases() -> list[BenchCase]:
    def invariant(output: str) -> bool:
        return "ok" in output

    return [
        BenchCase(
            id="alpha",
            family="fam-a",
            task_context="context-for-alpha",
            hidden_invariant=invariant,
        ),
        BenchCase(
            id="beta",
            family="fam-a",
            task_context="context-for-beta",
            hidden_invariant=invariant,
        ),
    ]


class ExplodingRunner:
    """Runner that must never be called on a fully-resumed rerun."""

    def execute(
        self,
        case: BenchCase,
        *,
        arm: str,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        raise AssertionError(f"resumed run re-executed {case.id}/{arm}/{repeat} — clobbered spend")


def _snapshot(run_dir: Path) -> dict[str, str]:
    artifacts = run_dir / "artifacts"
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(artifacts.iterdir())}


def test_run_records_outcomes_and_resume_never_reexecutes(tmp_path: Path) -> None:
    cases = _stub_cases()
    retriever = StubRetriever()
    runner = ScriptedRunner()

    run = open_run(tmp_path)
    first = BenchHarness(retriever, runner).run(
        [case.id for case in cases],
        corpus=cases,
        repeats=REPEAT_FLOOR,
        artifacts=RunArtifacts(run),
    )
    # Crash simulation: the run never completed.
    assert not run.is_complete()
    snapshot = _snapshot(run.directory)

    resumed = RunArtifacts(open_run(tmp_path, run_id=run.run_id))
    second = BenchHarness(retriever, ExplodingRunner()).run(
        [case.id for case in cases],
        corpus=cases,
        repeats=REPEAT_FLOOR,
        artifacts=resumed,
    )
    assert second == first
    assert _snapshot(run.directory) == snapshot


def test_completed_run_survives_a_new_run_untouched(tmp_path: Path) -> None:
    cases = _stub_cases()
    retriever = StubRetriever()
    runner = ScriptedRunner()

    completed = open_run(tmp_path)
    BenchHarness(retriever, runner).run(
        [case.id for case in cases],
        corpus=cases,
        repeats=REPEAT_FLOOR,
        artifacts=RunArtifacts(completed),
    )
    completed.mark_complete()
    snapshot = _snapshot(completed.directory)

    next_run = open_run(tmp_path)
    assert next_run.run_id != completed.run_id
    BenchHarness(retriever, runner).run(
        [case.id for case in cases],
        corpus=cases,
        repeats=REPEAT_FLOOR,
        artifacts=RunArtifacts(next_run),
    )
    assert _snapshot(completed.directory) == snapshot
    assert completed.is_complete()
