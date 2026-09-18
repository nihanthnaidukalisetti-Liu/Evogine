"""Trajectory capture tests: one trajectory file per gate decision (V2-8).

The core acceptance: every decision the memory gate emits — knowledge
verdicts and gate errors alike — leaves a trajectory file beside the
PromotionDecision evidence, capturing the review's tool calls (replay
fetches, evidence writes) and the verdict step. Content that flows into a
trajectory is sanitized at the recorder boundary. All offline; replays
come from deterministic stub runners.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.gates.approvals import ApprovalLog
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink
from evogine.gates.trajectory import (
    ToolCall,
    Trajectory,
    TrajectoryError,
    TrajectoryRecorder,
    TrajectoryStep,
    TrajectoryStore,
)


class ScriptedRunner:
    """Deterministic replay runner; unknown run ids raise (gate_error path)."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise RuntimeError(f"replay executor unavailable for {run_id}")
        return self._runs[run_id]


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=o) for i, o in enumerate(outcomes)
        ],
    )


def material_runs() -> dict[str, ReplayRun]:
    """Cold fails, warm passes on every instance — a material flip."""
    return {
        "run-cold": binary_run("run-cold", "baseline", [False, False]),
        "run-warm": binary_run("run-warm", "warm", [True, True]),
    }


def candidate(record_id: str, run_ids: list[str] | None = None) -> LearningRecord:
    """A deterministic candidate record wired to the material replay runs."""
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=f"Candidate {record_id}",
        body=RecordBody(signal="s", failure_mode="f", detection="d", fix_strategy="x"),
        status="candidate",
        confidence=0.5,
        provenance=RecordProvenance(
            proposed_by="reflector", origin="test", source="unit-test", note=""
        ),
        counterfactual=RecordCounterfactual(
            replay_run_ids=run_ids or ["run-cold", "run-warm"]
        ),
        created=datetime(2026, 9, 18, tzinfo=UTC),
    )


def make_gate(tmp_path: Path, runs: dict[str, ReplayRun], **flags: bool) -> MemoryGate:
    kwargs: dict[str, object] = {
        # V2-6: the approval boundary is required wiring — a gate without
        # an approval log is misconfigured and clears nothing.
        "approvals": ApprovalLog(tmp_path),
    }
    if "with_evidence" in flags:
        kwargs["evidence"] = RunsEvidenceSink(tmp_path) if flags["with_evidence"] else None
    else:
        kwargs["evidence"] = RunsEvidenceSink(tmp_path)
    if "with_trajectories" in flags:
        kwargs["trajectories"] = (
            TrajectoryStore(tmp_path) if flags["with_trajectories"] else None
        )
    else:
        kwargs["trajectories"] = TrajectoryStore(tmp_path)
    return MemoryGate(
        ScriptedRunner(runs),  # type: ignore[arg-type]
        DecisionLog(tmp_path),
        **kwargs,  # type: ignore[arg-type]
    )


def test_trajectory_file_per_decision_over_promotion_pass(tmp_path: Path):
    """The V2-8 acceptance: one trajectory file per decision, every decision."""
    gate = make_gate(tmp_path, material_runs())
    first = candidate("first-proposal")
    second = candidate("second-proposal")

    class TwoCandidateStore:
        """Minimal store double: two candidates; promote/demote are recorded."""

        def __init__(self) -> None:
            self.promoted: list[str] = []
            self.demoted: list[tuple[str, str]] = []

        def candidates(self) -> list[LearningRecord]:
            return [first, second]

        def promote(self, record_id: str, counterfactual: object) -> None:
            self.promoted.append(record_id)

        def demote(self, record_id: str, reason: str) -> None:
            self.demoted.append((record_id, reason))

    decisions = run_promotion_pass(TwoCandidateStore(), gate)  # type: ignore[arg-type]

    assert len(decisions) == 2
    trajectories = TrajectoryStore(tmp_path)
    for decision in decisions:
        # Every decision names its trajectory, and the file exists.
        assert (
            decision.trajectory_ref
            == f"runs/trajectory/{decision.proposal_id}.json"
        )
        trajectory = trajectories.read(decision.proposal_id)
        assert trajectory.gate == "memory_gate"
        assert trajectory.decided_at is not None
    assert {t.proposal_id for t in trajectories.all()} == {
        "first-proposal",
        "second-proposal",
    }
    # The trajectory lives alongside the PromotionDecision evidence.
    for proposal_id in ("first-proposal", "second-proposal"):
        assert (tmp_path / "runs" / "gate" / f"{proposal_id}.json").is_file()
        assert (tmp_path / "runs" / "trajectory" / f"{proposal_id}.json").is_file()


def test_trajectory_captures_tool_calls_and_decision(tmp_path: Path):
    """Tool calls (replay fetches, evidence write) and the verdict are all steps."""
    record = candidate("proposal-1")
    gate = make_gate(tmp_path, material_runs())
    decision = gate.review(record, ["run-cold", "run-warm"])

    trajectory: Trajectory = TrajectoryStore(tmp_path).read(record.id)
    calls = trajectory.tool_calls()
    # Tool calls included: one per replay fetch, the evidence write, and
    # the V2-6 proposal render that the cleared verdict's effect now is.
    assert [call.tool for call in calls] == [
        "replay_runner.run",
        "replay_runner.run",
        "evidence_sink.write",
        "approvals.render",
    ]
    assert [call.arguments for call in calls[:2]] == [
        '{"run_id": "run-cold"}',
        '{"run_id": "run-warm"}',
    ]
    assert "baseline: 2 verdict(s)" in calls[0].result
    assert "warm: 2 verdict(s)" in calls[1].result
    assert "wrote runs/gate/proposal-1.json" in calls[2].result
    assert "rendered appr-proposal-1" in calls[3].result
    # The verdict step carries the stage outcome.
    decision_steps = [step for step in trajectory.steps if step.kind == "decision"]
    assert len(decision_steps) == 1
    assert "never_worse: cleared" in decision_steps[0].detail
    # V2-6: the cleared verdict renders a pending proposal; it does not promote.
    assert decision.promoted is False
    assert decision.stage is DecisionStage.APPROVAL
    assert [p.proposal_id for p in ApprovalLog(tmp_path).pending()] == [
        "appr-proposal-1"
    ]


def test_gate_error_decision_still_gets_trajectory(tmp_path: Path):
    """Every decision means every decision: infrastructure failures too."""
    gate = make_gate(tmp_path, runs={})
    record = candidate("proposal-err", ["run-missing"])
    decision = gate.review(record, ["run-missing"])

    assert decision.stage is DecisionStage.GATE_ERROR
    trajectory = TrajectoryStore(tmp_path).read(record.id)
    failed_calls = trajectory.tool_calls()
    assert len(failed_calls) == 1
    assert failed_calls[0].tool == "replay_runner.run"
    assert "failed" in failed_calls[0].result
    assert decision.trajectory_ref is not None


def test_trajectory_sanitizes_secret_shaped_tool_output(tmp_path: Path):
    """Candidate-influenced tool output is redacted before it is captured."""

    class HostileRunner:
        def run(self, run_id: str) -> ReplayRun:
            raise RuntimeError(f"boom sk-supersecret123 {run_id}")

    gate = MemoryGate(
        HostileRunner(),  # type: ignore[arg-type]
        DecisionLog(tmp_path),
        approvals=ApprovalLog(tmp_path),
        evidence=RunsEvidenceSink(tmp_path),
        trajectories=TrajectoryStore(tmp_path),
    )
    record = candidate("proposal-leak", ["run-hostile"])
    gate.review(record, ["run-hostile"])

    stored = (tmp_path / "runs" / "trajectory" / "proposal-leak.json").read_text()
    assert "sk-supersecret123" not in stored
    assert "<redacted>" in stored


def test_store_rejects_path_traversal_id(tmp_path: Path):
    """A hostile proposal id never traverses paths — same rule as telemetry."""
    store = TrajectoryStore(tmp_path)
    trajectory = TrajectoryRecorder(gate="memory_gate", proposal_id="x").freeze()
    with pytest.raises(TrajectoryError):
        store.write("../evil", trajectory)
    assert not (tmp_path.parent / "evil.json").exists()
    with pytest.raises(TrajectoryError):
        store.read("../evil")


def test_store_rejects_id_payload_mismatch(tmp_path: Path):
    """The stored trajectory must belong to the file it is written under."""
    store = TrajectoryStore(tmp_path)
    trajectory = TrajectoryRecorder(gate="memory_gate", proposal_id="a").freeze()
    with pytest.raises(TrajectoryError, match="does not match"):
        store.write("b", trajectory)


def test_store_write_roundtrip(tmp_path: Path):
    store = TrajectoryStore(tmp_path)
    recorder = TrajectoryRecorder(gate="memory_gate", proposal_id="proposal-rt")
    recorder.observe("checked materiality")
    trajectory = recorder.freeze()
    ref = store.write("proposal-rt", trajectory)
    assert ref == "runs/trajectory/proposal-rt.json"
    assert store.read("proposal-rt") == trajectory


def test_recorder_rejects_steps_after_freeze():
    """One recorder is exactly one review: freeze closes it."""
    recorder = TrajectoryRecorder(gate="memory_gate", proposal_id="p")
    recorder.freeze()
    with pytest.raises(TrajectoryError):
        recorder.observe("after freeze")
    with pytest.raises(TrajectoryError):
        recorder.tool_call("replay_runner.run", {"run_id": "r"}, result="x")
    with pytest.raises(TrajectoryError):
        recorder.freeze()


def test_trajectory_step_tool_call_shape():
    """tool_call content is bound to the tool_call kind, in both directions."""
    call = ToolCall(seq=1, tool="t", arguments="{}", result="r")
    with pytest.raises(ValueError, match="tool_call must be set"):
        TrajectoryStep(seq=1, kind="observation", detail="d", tool_call=call)
    with pytest.raises(ValueError, match="tool_call must be set"):
        TrajectoryStep(seq=1, kind="tool_call", detail="d", tool_call=None)


def test_decision_without_trajectory_store_has_no_ref(tmp_path: Path):
    """Backward compatible: no trajectory store wired, no trajectory_ref."""
    gate = make_gate(tmp_path, material_runs(), with_trajectories=False)
    record = candidate("proposal-plain")
    decision = gate.review(record, ["run-cold", "run-warm"])
    assert decision.trajectory_ref is None
    assert decision.evidence_ref is not None
