"""Memory gate tests: the counterfactual replay pipeline, fully offline.

Every test drives the gate through a stub :class:`ReplayRunner` over
deterministic fixtures — the suite never calls a live model. Each test
also asserts the telemetry side effect: a review that does not leave
exactly one decision record is the V1 bug V2-1 exists to kill.
"""

from pathlib import Path

import pytest

from evogine.engine.records import LearningRecord
from evogine.gates.approvals import ApprovalError, ApprovalLog, ApprovalState
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import (
    DecisionLog,
    DecisionStage,
    PromotionDecision,
    RunsEvidenceSink,
    TelemetryError,
)


class StubRunner:
    """Deterministic replay runner: run id -> scripted ReplayRun."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs
        self.calls: list[str] = []

    def run(self, run_id: str) -> ReplayRun:
        self.calls.append(run_id)
        if run_id not in self._runs:
            raise KeyError(f"unknown replay run: {run_id}")
        return self._runs[run_id]


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[Verdict(instance_id=f"case-{i}", passed=o) for i, o in enumerate(outcomes)],
    )


def scored_run(run_id: str, arm: str, scores: list[float]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[Verdict(instance_id=f"case-{i}", score=s) for i, s in enumerate(scores)],
    )


@pytest.fixture
def record(record_factory) -> LearningRecord:
    return record_factory("prop-0001")


@pytest.fixture
def log(tmp_path: Path) -> DecisionLog:
    return DecisionLog(tmp_path / "store")


@pytest.fixture
def approvals(tmp_path: Path) -> ApprovalLog:
    return ApprovalLog(tmp_path / "store")


@pytest.fixture
def gate_factory(log: DecisionLog, approvals: ApprovalLog):
    def factory(runner: StubRunner, **kwargs) -> MemoryGate:
        return MemoryGate(runner, log, approvals, **kwargs)

    return factory


# ------------------------------------------------------------- promote path


def test_material_binary_flip_renders_a_proposal(
    gate_factory, approvals: ApprovalLog, record: LearningRecord, tmp_path: Path
) -> None:
    """A cleared review proposes; it does not promote (V2-6)."""
    runner = StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
            "warm-1": binary_run("warm-1", "warm", [True, True, True]),
        }
    )
    decision = gate_factory(
        runner, evidence=RunsEvidenceSink(tmp_path / "store")
    ).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.APPROVAL
    assert decision.reason == "proposed: 3 binary flips"
    assert decision.replay_run_ids == ["cold-1", "warm-1"]
    assert decision.learning_id == decision.proposal_id == "prop-0001"

    # The proposal waits in the approval log with the cleared trail attached.
    pending = approvals.pending()
    assert [proposal.proposal_id for proposal in pending] == ["appr-prop-0001"]
    assert pending[0].learning_id == "prop-0001"
    assert pending[0].decision.reason == "cleared: 3 binary flips"
    assert pending[0].decision.promoted is True  # the gate's recommendation
    assert pending[0].state is ApprovalState.PENDING


def test_review_always_records_its_decision(
    gate_factory, log: DecisionLog, record: LearningRecord
) -> None:
    """Every verdict — pass or fail — leaves exactly one decision record."""
    runner = StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
            "warm-1": binary_run("warm-1", "warm", [True, True, True]),
        }
    )
    gate = gate_factory(runner)
    gate.review(record, ["cold-1", "warm-1"])

    recorded = log.all()
    assert len(recorded) == 1
    assert recorded[0].proposal_id == "prop-0001"
    assert recorded[0].stage is DecisionStage.APPROVAL
    assert recorded[0].decided_at.tzinfo is not None


# ------------------------------------------------------------ stage failures


def test_no_flip_stops_at_replay_stage(
    gate_factory, log: DecisionLog, record: LearningRecord
) -> None:
    runner = StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, True, False]),
            "warm-1": binary_run("warm-1", "warm", [False, True, False]),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.REPLAY
    assert decision.reason == "no_flip: 0/3 replays changed verdict"
    assert log.all()[0].stage is DecisionStage.REPLAY


def test_scored_no_change_stops_at_replay_stage(
    gate_factory, record: LearningRecord
) -> None:
    """Identical scored outcomes never flip, whatever the threshold."""
    runner = StubRunner(
        {
            "cold-1": scored_run("cold-1", "baseline", [0.5, 0.6, 0.7]),
            "warm-1": scored_run("warm-1", "warm", [0.5, 0.6, 0.7]),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.stage is DecisionStage.REPLAY
    assert decision.reason == "no_flip: 0/3 replays changed verdict"


def test_scored_sub_threshold_flip_stops_at_materiality(
    gate_factory, record: LearningRecord
) -> None:
    runner = StubRunner(
        {
            "cold-1": scored_run("cold-1", "baseline", [0.5, 0.5, 0.5]),
            "warm-1": scored_run("warm-1", "warm", [0.55, 0.55, 0.55]),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.MATERIALITY
    assert decision.reason == "not_material: scored delta 0.050 vs threshold 0.15"


def test_regression_stops_at_never_worse_after_materiality(
    gate_factory, record: LearningRecord
) -> None:
    """A material flip that also regresses a case is stopped at NEVER_WORSE.

    Case-2 regresses (pass -> fail) while case-1 improves — materiality
    passes (flips exist), never-worse refuses.
    """
    runner = StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, True]),
            "warm-1": binary_run("warm-1", "warm", [True, True, False]),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.NEVER_WORSE
    assert decision.reason == "regression: 1 case(s) regressed: case-2"


def test_scored_regression_stops_at_never_worse_after_materiality(
    gate_factory, record: LearningRecord
) -> None:
    """A material scored flip that also regresses a case stops at NEVER_WORSE.

    Deltas [0.8, 0.8, -0.1]: the paired median (0.8) clears materiality,
    but case-2 regressed — never-worse refuses, listing every regressed
    case.
    """
    runner = StubRunner(
        {
            "cold-1": scored_run("cold-1", "baseline", [0.1, 0.1, 0.1]),
            "warm-1": scored_run("warm-1", "warm", [0.9, 0.9, 0.0]),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.NEVER_WORSE
    assert decision.reason == "regression: 1 case(s) regressed: case-2"


# ------------------------------------------------------------- gate errors


def test_runner_crash_is_gate_error(gate_factory, record: LearningRecord) -> None:
    class ExplodingRunner:
        def run(self, run_id: str) -> ReplayRun:
            raise RuntimeError("replay executor crashed")

    decision = gate_factory(ExplodingRunner()).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.GATE_ERROR
    assert decision.reason == "gate_error: replay run cold-1 failed: replay executor crashed"


def test_unknown_run_id_is_gate_error(gate_factory, record: LearningRecord) -> None:
    runner = StubRunner(
        {"cold-1": binary_run("cold-1", "baseline", [False, False])}
    )
    decision = gate_factory(runner).review(record, ["cold-1", "missing-warm"])

    assert decision.stage is DecisionStage.GATE_ERROR
    assert decision.reason.startswith("gate_error: replay run missing-warm failed:")
    assert "unknown replay run" in decision.reason


def test_missing_warm_arm_is_gate_error(gate_factory, record: LearningRecord) -> None:
    runner = StubRunner({"cold-1": binary_run("cold-1", "baseline", [False, False])})
    decision = gate_factory(runner).review(record, ["cold-1"])

    assert decision.stage is DecisionStage.GATE_ERROR
    assert decision.reason == "gate_error: replay runs missing the warm arm"


def test_instance_mismatch_is_gate_error(gate_factory, record: LearningRecord) -> None:
    """Arms covering different case instances are an infrastructure defect."""
    runner = StubRunner(
        {
            "cold-1": ReplayRun(
                run_id="cold-1",
                arm="baseline",
                verdicts=[Verdict(instance_id="case-a", passed=False)],
            ),
            "warm-1": ReplayRun(
                run_id="warm-1",
                arm="warm",
                verdicts=[Verdict(instance_id="case-b", passed=True)],
            ),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.stage is DecisionStage.GATE_ERROR
    assert "different case instances" in decision.reason


def test_no_run_ids_is_gate_error(gate_factory, record: LearningRecord) -> None:
    decision = gate_factory(StubRunner({})).review(record, [])

    assert decision.stage is DecisionStage.GATE_ERROR
    assert decision.reason == "gate_error: no replay runs queued"


# ----------------------------------------------------------------- evidence


def test_evidence_sink_records_the_ref(
    gate_factory, record: LearningRecord, tmp_path: Path
) -> None:
    store_root = tmp_path / "store"
    evidence = RunsEvidenceSink(store_root)
    runner = StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
            "warm-1": binary_run("warm-1", "warm", [True, True, True]),
        }
    )
    gate = MemoryGate(
        runner, DecisionLog(store_root), ApprovalLog(store_root), evidence=evidence
    )
    decision = gate.review(record, ["cold-1", "warm-1"])

    assert decision.evidence_ref == "runs/gate/prop-0001.json"
    evidence_file = store_root / decision.evidence_ref
    assert evidence_file.exists()
    assert "cold-1" in evidence_file.read_text(encoding="utf-8")


def test_cleared_review_without_evidence_is_refused(
    gate_factory, approvals: ApprovalLog, record: LearningRecord
) -> None:
    """No proposal without replay evidence — regardless of the human (V2-6).

    The knowledge cleared, but with no evidence sink there is nothing a
    proposal could justify itself with: the review records the refusal and
    the approval log stays empty.
    """
    runner = StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
            "warm-1": binary_run("warm-1", "warm", [True, True, True]),
        }
    )
    decision = gate_factory(runner).review(record, ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.APPROVAL
    assert decision.reason.startswith("refused_no_evidence: no replay evidence")
    assert decision.evidence_ref is None
    assert approvals.all() == []


def test_render_refuses_a_proposal_without_evidence(approvals: ApprovalLog) -> None:
    """The approval log itself refuses evidence-free renders."""
    cleared = PromotionDecision(
        proposal_id="prop-9",
        learning_id="prop-9",
        promoted=True,
        stage=DecisionStage.NEVER_WORSE,
        reason="cleared: 2 binary flips",
    )
    with pytest.raises(ApprovalError, match="no replay evidence"):
        approvals.render("prop-9", cleared, None)
    assert approvals.all() == []


def test_render_rejects_a_trail_that_is_not_cleared(approvals: ApprovalLog) -> None:
    """A proposal must carry the gate's cleared verdict; anything else is a bug."""
    not_cleared = PromotionDecision(
        proposal_id="prop-9",
        learning_id="prop-9",
        promoted=False,
        stage=DecisionStage.REPLAY,
        reason="no_flip: 0/3 replays changed verdict",
    )
    with pytest.raises(TelemetryError, match="not a cleared gate verdict"):
        approvals.render("prop-9", not_cleared, "runs/gate/prop-9.json")
