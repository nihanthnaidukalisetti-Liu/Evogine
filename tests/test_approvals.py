"""Human-approval boundary tests (V2-6): approve, reject, and refuse.

Integration level: the gate renders, a human decides through the approval
log or the ``evogine approvals`` CLI, and the store only ever moves when a
recorded human decision says so. Fully offline — stub replay runner,
deterministic fixtures, no live model.
"""

from pathlib import Path

import pytest

from evogine.cli import main
from evogine.engine.store import Engine
from evogine.gates.approvals import (
    ApprovalError,
    ApprovalLog,
    ApprovalState,
    approve_proposal,
    reject_proposal,
)
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink
from evogine.gates.trajectory import TrajectoryStore

PROPOSAL_ID = "appr-prop-9"
LEARNING_ID = "prop-9"


class StubRunner:
    """Deterministic replay runner: run id -> scripted ReplayRun."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise KeyError(f"unknown replay run: {run_id}")
        return self._runs[run_id]


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=ok) for i, ok in enumerate(outcomes)
        ],
    )


# Stateless: every run returns the same scripted replays, so a module-level
# runner is safe to share across tests.
CLEARED_RUNNER = StubRunner(
    {
        "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
        "warm-1": binary_run("warm-1", "warm", [True, True, True]),
    }
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def store(root: Path) -> Engine:
    return Engine(root)


@pytest.fixture
def approvals(root: Path) -> ApprovalLog:
    return ApprovalLog(root)


@pytest.fixture
def pending(
    root: Path, store: Engine, approvals: ApprovalLog, record_factory
) -> str:
    """One reviewed record whose cleared knowledge is a pending proposal."""
    record = store.propose(record_factory(LEARNING_ID))
    store.queue_replay(record.id, ["cold-1", "warm-1"])
    gate = MemoryGate(
        CLEARED_RUNNER, DecisionLog(root), approvals, evidence=RunsEvidenceSink(root)
    )
    gate.review(store.get(record.id), ["cold-1", "warm-1"])
    return PROPOSAL_ID


# ------------------------------------------------------------------- approve


def test_approve_applies_the_promotion(
    store: Engine, approvals: ApprovalLog, pending: str
) -> None:
    """A human approval is what moves the record: candidate -> promoted."""
    proposal_before = approvals.get(pending)
    promoted = approve_proposal(
        store, approvals, pending, decided_by="nihanth", reason="replays convince me"
    )

    assert promoted.status == "promoted"
    assert promoted.promoted_at is not None
    assert promoted.counterfactual.replay_run_ids == ["cold-1", "warm-1"]
    assert promoted.counterfactual.evidence == proposal_before.evidence_ref

    decided = approvals.get(pending)
    assert decided.state is ApprovalState.APPLIED
    assert decided.decided_by == "nihanth"
    assert decided.decided_reason == "replays convince me"
    # Retrieval draws only from the promoted pool — it must now see it.
    assert [record.id for record in store.retrieve("429 backoff")] == [LEARNING_ID]


def test_approval_leaves_the_decision_log_untouched(
    store: Engine, approvals: ApprovalLog, pending: str, root: Path
) -> None:
    """The approval lives on the proposal; the review's decision record stands."""
    log = DecisionLog(root)
    before = log.all()
    approve_proposal(store, approvals, pending, decided_by="nihanth")

    after = log.all()
    assert len(after) == 1
    assert after == before


# -------------------------------------------------------------------- reject


def test_reject_refuses_and_demotes_the_record(
    store: Engine, approvals: ApprovalLog, pending: str
) -> None:
    """A rejection is a knowledge verdict: the record is demoted to dropped."""
    dropped = reject_proposal(
        store,
        approvals,
        pending,
        decided_by="nihanth",
        reason="warm arm never ran on the degraded case",
    )

    assert dropped.status == "dropped"
    decided = approvals.get(pending)
    assert decided.state is ApprovalState.REJECTED
    assert decided.decided_by == "nihanth"
    assert decided.decided_reason == "warm arm never ran on the degraded case"
    assert store.candidates() == []
    # Rejected knowledge stays auditable, never deleted.
    assert store.get(LEARNING_ID).status == "dropped"


def test_decided_proposals_are_final(
    store: Engine, approvals: ApprovalLog, pending: str
) -> None:
    """A decided proposal cannot be approved or rejected again."""
    approve_proposal(store, approvals, pending, decided_by="nihanth")

    with pytest.raises(ApprovalError, match="already applied"):
        approve_proposal(store, approvals, pending, decided_by="nihanth")
    with pytest.raises(ApprovalError, match="already applied"):
        reject_proposal(store, approvals, pending, decided_by="nihanth", reason="no")


# --------------------------------------------------------------- enforcement


def test_gate_cannot_promote_without_a_human_record(
    store: Engine, approvals: ApprovalLog, pending: str
) -> None:
    """The negative invariant: a cleared review alone promotes nothing."""
    assert store.get(LEARNING_ID).status == "candidate"
    assert store.promoted_records() == []
    assert store.retrieve("429 backoff") == []
    assert [proposal.proposal_id for proposal in approvals.pending()] == [pending]


def test_approve_refuses_a_proposal_that_lost_its_evidence(
    store: Engine, approvals: ApprovalLog, pending: str, root: Path
) -> None:
    """Defense in depth: the evidence must survive up to the human step."""
    path = root / "approvals" / f"{pending}.json"
    proposal = approvals.get(pending)
    path.write_text(
        proposal.model_copy(update={"evidence_ref": None}).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ApprovalError, match="carries no replay evidence"):
        approve_proposal(store, approvals, pending, decided_by="nihanth")

    # Nothing moved: the proposal is still pending, the record still a candidate.
    assert approvals.get(pending).state is ApprovalState.PENDING
    assert store.get(LEARNING_ID).status == "candidate"


def test_review_composes_trajectory_capture_with_the_approval_boundary(
    root: Path, record_factory
) -> None:
    """V2-6 x V2-8 composition, end to end.

    A cleared review leaves a trajectory file, a pending approval
    proposal, and exactly one PromotionDecision; the human approval
    promotes with the evidence attached; the rejection never promotes.
    """
    store = Engine(root)
    approvals = ApprovalLog(root)
    log = DecisionLog(root)
    gate = MemoryGate(
        CLEARED_RUNNER,
        log,
        approvals,
        evidence=RunsEvidenceSink(root),
        trajectories=TrajectoryStore(root),
    )

    # Approve arc: a cleared review renders — it does not promote.
    approved_record = store.propose(record_factory("prop-9"))
    store.queue_replay(approved_record.id, ["cold-1", "warm-1"])
    decision = gate.review(store.get(approved_record.id), ["cold-1", "warm-1"])

    assert decision.promoted is False
    assert decision.stage is DecisionStage.APPROVAL
    assert decision.evidence_ref == "runs/gate/prop-9.json"
    assert decision.trajectory_ref == "runs/trajectory/prop-9.json"
    assert (root / "runs" / "trajectory" / "prop-9.json").is_file()
    assert [d.learning_id for d in log.all()] == ["prop-9"]  # exactly one decision
    trajectory = TrajectoryStore(root).read("prop-9")
    assert [call.tool for call in trajectory.tool_calls()] == [
        "replay_runner.run",
        "replay_runner.run",
        "evidence_sink.write",
        "approvals.render",
    ]
    assert [proposal.proposal_id for proposal in approvals.pending()] == [PROPOSAL_ID]

    promoted = approve_proposal(
        store,
        approvals,
        PROPOSAL_ID,
        decided_by="nihanth",
        reason="replays convince me",
    )
    assert promoted.status == "promoted"
    assert promoted.counterfactual.evidence == decision.evidence_ref
    assert [record.id for record in store.retrieve("429 backoff")] == ["prop-9"]

    # Reject arc: the same composition, and the refusal never promotes.
    rejected_record = store.propose(record_factory("prop-10"))
    store.queue_replay(rejected_record.id, ["cold-1", "warm-1"])
    gate.review(store.get(rejected_record.id), ["cold-1", "warm-1"])
    rejected = reject_proposal(
        store, approvals, "appr-prop-10", decided_by="nihanth", reason="not convinced"
    )

    assert rejected.status == "dropped"
    assert [record.id for record in store.retrieve("429 backoff")] == ["prop-9"]
    assert (root / "runs" / "trajectory" / "prop-10.json").is_file()


def test_review_twice_keeps_one_pending_proposal(
    store: Engine, approvals: ApprovalLog, record_factory, root: Path
) -> None:
    """A fresh review of the same learning replaces, never duplicates, the trail."""
    record = store.propose(record_factory(LEARNING_ID))
    store.queue_replay(record.id, ["cold-1", "warm-1"])
    gate = MemoryGate(
        CLEARED_RUNNER, DecisionLog(root), approvals, evidence=RunsEvidenceSink(root)
    )
    gate.review(store.get(record.id), ["cold-1", "warm-1"])
    gate.review(store.get(record.id), ["cold-1", "warm-1"])

    pending_proposals = approvals.pending()
    assert [proposal.proposal_id for proposal in pending_proposals] == [PROPOSAL_ID]
    assert pending_proposals[0].learning_id == LEARNING_ID
    assert pending_proposals[0].decision.reason == "cleared: 3 binary flips"


# ----------------------------------------------------------------------- CLI


def test_cli_approve_applies_and_exits_zero(
    store: Engine, approvals: ApprovalLog, pending: str, capsys
) -> None:
    exit_code = main(
        [
            "approvals",
            "approve",
            pending,
            "--root",
            str(store.root),
            "--by",
            "nihanth",
            "--reason",
            "replays convince me",
        ]
    )

    assert exit_code == 0
    assert f"{pending} approved: record {LEARNING_ID} is promoted" in (
        capsys.readouterr().out
    )
    assert store.get(LEARNING_ID).status == "promoted"
    assert approvals.get(pending).state is ApprovalState.APPLIED


def test_cli_reject_refuses_and_exits_zero(
    store: Engine, approvals: ApprovalLog, pending: str, capsys
) -> None:
    exit_code = main(
        [
            "approvals",
            "reject",
            pending,
            "--root",
            str(store.root),
            "--by",
            "nihanth",
            "--reason",
            "warm arm never ran",
        ]
    )

    assert exit_code == 0
    assert f"{pending} rejected: record {LEARNING_ID} is dropped" in (
        capsys.readouterr().out
    )
    assert store.get(LEARNING_ID).status == "dropped"
    assert approvals.get(pending).state is ApprovalState.REJECTED


def test_cli_unknown_proposal_exits_one(store: Engine, pending: str, capsys) -> None:
    exit_code = main(
        [
            "approvals",
            "approve",
            "appr-nope",
            "--root",
            str(store.root),
            "--by",
            "nihanth",
        ]
    )

    assert exit_code == 1
    assert "error: unknown proposal: appr-nope" in capsys.readouterr().err


def test_cli_reject_requires_a_reason(store: Engine, pending: str) -> None:
    """argparse enforces the rejection reason before anything is touched."""
    with pytest.raises(SystemExit):
        main(
            [
                "approvals",
                "reject",
                pending,
                "--root",
                str(store.root),
                "--by",
                "nihanth",
            ]
        )
    assert store.get(LEARNING_ID).status == "candidate"


def test_cli_list_marks_pending_proposals(store: Engine, pending: str, capsys) -> None:
    exit_code = main(["approvals", "list", "--root", str(store.root)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "approval proposals: 1 total, 1 pending" in output
    assert f"{PROPOSAL_ID} [pending]" in output
    assert f"learning: {LEARNING_ID}" in output
    assert "trail: cleared: 3 binary flips (evidence: runs/gate/prop-9.json)" in output


def test_cli_list_renders_decided_proposals(
    store: Engine, approvals: ApprovalLog, pending: str, capsys
) -> None:
    reject_proposal(store, approvals, pending, decided_by="nihanth", reason="not yet")
    exit_code = main(["approvals", "list", "--root", str(store.root)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "approval proposals: 1 total, 0 pending" in output
    assert f"{PROPOSAL_ID} [rejected]" in output
    assert "decided by nihanth: not yet" in output
