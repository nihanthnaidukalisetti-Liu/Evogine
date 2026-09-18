"""Offline wall-clock cross-check units: a too-fast pass fails closed.

The pure check and the production evaluator are both exercised without a
Docker daemon — the evaluator runs against a stub sandbox returning canned
:class:`SandboxResult` values.
"""

from __future__ import annotations

from pathlib import Path

from evogine.gates.code_gate import DockerCaseEvaluator, EvalCase
from evogine.gates.equivalence import CaseVerdict
from evogine.gates.wallclock import crosscheck_too_fast_pass
from evogine.sandbox.docker_runner import SandboxResult


def test_pass_above_the_floor_survives() -> None:
    assert (
        crosscheck_too_fast_pass(verdict=CaseVerdict.PASS, duration_ms=2500.0, min_pass_seconds=1.0)
        is None
    )


def test_pass_below_the_floor_is_rejected() -> None:
    detail = crosscheck_too_fast_pass(
        verdict=CaseVerdict.PASS, duration_ms=7.0, min_pass_seconds=1.0
    )
    assert detail is not None
    assert "wall-clock cross-check" in detail
    assert "failing closed" in detail


def test_fail_verdicts_are_exempt() -> None:
    """A too-fast FAIL is an honest fast failure — the check never touches it."""
    assert (
        crosscheck_too_fast_pass(verdict=CaseVerdict.FAIL, duration_ms=0.0, min_pass_seconds=60.0)
        is None
    )


def test_pass_without_a_recorded_duration_fails_closed() -> None:
    detail = crosscheck_too_fast_pass(
        verdict=CaseVerdict.PASS, duration_ms=None, min_pass_seconds=0.5
    )
    assert detail is not None
    assert "no recorded duration" in detail


class StubSandbox:
    """SandboxRun double: returns one canned result and records the call."""

    def __init__(self, result: SandboxResult) -> None:
        self._result = result
        self.calls: list[str] = []

    def run(self, command: str, **_: object) -> SandboxResult:
        self.calls.append(command)
        return self._result


def _sandbox(duration_ms: int, *, exit_code: int = 0) -> StubSandbox:
    return StubSandbox(
        SandboxResult(
            exit_code=exit_code,
            stdout="",
            stderr="",
            timed_out=False,
            duration_ms=duration_ms,
            truncated=False,
        )
    )


def test_evaluator_fails_closed_on_too_fast_pass() -> None:
    evaluator = DockerCaseEvaluator(_sandbox(12))
    case = EvalCase(case_id="suspicious", command="python -c 'print(1)'", min_pass_seconds=5.0)
    result = evaluator.run_case("candidate", Path("/tmp/unused"), case)
    assert result.verdict is CaseVerdict.FAIL
    assert "wall-clock cross-check" in result.detail
    assert result.duration_ms == 12.0


def test_evaluator_passes_an_honestly_timed_run() -> None:
    evaluator = DockerCaseEvaluator(_sandbox(5_000))
    case = EvalCase(case_id="honest", command="python -c 'print(1)'", min_pass_seconds=1.0)
    result = evaluator.run_case("candidate", Path("/tmp/unused"), case)
    assert result.verdict is CaseVerdict.PASS
    assert result.duration_ms == 5_000.0


def test_evaluator_without_floor_keeps_legacy_behavior() -> None:
    evaluator = DockerCaseEvaluator(_sandbox(0))
    case = EvalCase(case_id="unfloored", command="python -c 'print(1)'")
    result = evaluator.run_case("candidate", Path("/tmp/unused"), case)
    assert result.verdict is CaseVerdict.PASS


def test_evaluator_records_duration_on_failures() -> None:
    evaluator = DockerCaseEvaluator(_sandbox(1_500, exit_code=2))
    case = EvalCase(case_id="failing", command="false")
    result = evaluator.run_case("candidate", Path("/tmp/unused"), case)
    assert result.verdict is CaseVerdict.FAIL
    assert "exit 2" in result.detail
    assert result.duration_ms == 1_500.0
