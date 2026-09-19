"""Promotion telemetry tests: the completeness invariant and the report.

The core acceptance for V2-1: a promotion pass over stub replays decides
every proposal and leaves exactly one decision record per proposal — a
decided proposal with no decision record is a bug, and these tests fail
on it. The report test pins ``evogine decisions report`` rendering. All
offline; replays come from deterministic stub runners.
"""

from pathlib import Path

import pytest

from evogine.cli import main
from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.approvals import ApprovalLog, ApprovalState
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import (
    DecisionLog,
    DecisionStage,
    PromotionDecision,
    RunsEvidenceSink,
    TelemetryError,
    check_reason,
)
from evogine.ops.metrics import (
    FUNNEL_STAGE_TOTAL,
    GATE_DECISIONS_TOTAL,
    metrics,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def _isolate_metrics():
    """Metric wiring tests read process-wide counters — reset around each."""
    reset_metrics()
    yield
    reset_metrics()


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
        verdicts=[Verdict(instance_id=f"case-{i}", passed=o) for i, o in enumerate(outcomes)],
    )


def scored_run(run_id: str, arm: str, scores: list[float]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # fixture pins the literal
        verdicts=[Verdict(instance_id=f"case-{i}", score=s) for i, s in enumerate(scores)],
    )


def decision(**overrides) -> PromotionDecision:
    """A valid decision; overrides replace fields wholesale."""
    fields: dict[str, object] = {
        "proposal_id": "prop-a",
        "learning_id": "prop-a",
        "promoted": False,
        "stage": DecisionStage.REPLAY,
        "reason": "no_flip: 0/3 replays changed verdict",
    }
    fields.update(overrides)
    return PromotionDecision.model_validate(fields)


# ------------------------------------------------------------- decision log


def test_decision_log_round_trips_every_field(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    original = decision(
        proposal_id="prop-x",
        learning_id="prop-x",
        promoted=True,
        stage=DecisionStage.NEVER_WORSE,
        reason="cleared: 2 binary flips",
        replay_run_ids=["cold-1", "warm-1"],
        evidence_ref="runs/gate/prop-x.json",
    )

    log.record(original)
    loaded = log.all()

    assert loaded == [original]


def test_decision_log_rejects_free_form_reason(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    with pytest.raises(TelemetryError, match="must be '<label>: <evidence>'"):
        log.record(decision(reason="it did not flip at all"))


def test_decision_log_rejects_label_from_wrong_stage(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    with pytest.raises(TelemetryError, match="not valid for stage replay"):
        log.record(decision(reason="cleared: 2 binary flips"))


def test_decision_log_rejects_evidence_outside_the_template(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    with pytest.raises(TelemetryError, match="does not match the template"):
        log.record(decision(reason="no_flip: honestly nothing changed"))


def test_recorded_decisions_increment_gate_and_funnel_counters(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    log.record(
        decision(
            proposal_id="prop-a",
            learning_id="prop-a",
            stage=DecisionStage.REPLAY,
            reason="no_flip: 0/3 replays changed verdict",
        )
    )
    log.record(
        decision(
            proposal_id="prop-b",
            learning_id="prop-b",
            promoted=True,
            stage=DecisionStage.NEVER_WORSE,
            reason="cleared: 2 binary flips",
        )
    )

    snapshot = metrics().snapshot()
    assert snapshot.counter(GATE_DECISIONS_TOTAL, {"stage": "replay", "promoted": "false"}) == 1
    assert snapshot.counter(GATE_DECISIONS_TOTAL, {"stage": "never_worse", "promoted": "true"}) == 1
    assert snapshot.counter(FUNNEL_STAGE_TOTAL, {"stage": "replay"}) == 1
    assert snapshot.counter(FUNNEL_STAGE_TOTAL, {"stage": "never_worse"}) == 1


def test_refused_decision_records_are_never_counted(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    with pytest.raises(TelemetryError):
        log.record(decision(reason="it did not flip at all"))

    # Metrics count durable decisions only — a refused record is persisted
    # nowhere and counted nowhere.
    snapshot = metrics().snapshot()
    assert snapshot.counter(GATE_DECISIONS_TOTAL) == 0
    assert snapshot.counter(FUNNEL_STAGE_TOTAL) == 0


def test_check_reason_accepts_the_spec_example() -> None:
    check_reason(DecisionStage.REPLAY, "no_flip: 0/3 replays changed verdict")


def test_counts_by_stage_is_zero_filled(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    log.record(decision())
    log.record(
        decision(
            proposal_id="prop-err",
            learning_id="prop-err",
            stage=DecisionStage.GATE_ERROR,
            reason="gate_error: replay executor unavailable",
        )
    )

    counts = log.counts_by_stage()

    assert counts == {
        DecisionStage.REPLAY: 1,
        DecisionStage.MATERIALITY: 0,
        DecisionStage.NEVER_WORSE: 0,
        DecisionStage.APPROVAL: 0,
        DecisionStage.GATE_ERROR: 1,
    }


def test_decision_log_rejects_unsafe_proposal_ids(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    with pytest.raises(TelemetryError, match="must match"):
        log.record(decision(proposal_id="../escape", learning_id="../escape"))


# ---------------------------------------------------------- report rendering


def test_report_renders_counts_by_stage(tmp_path: Path, capsys) -> None:
    log = DecisionLog(tmp_path)
    log.record(decision())
    log.record(
        decision(
            proposal_id="prop-b",
            learning_id="prop-b",
            stage=DecisionStage.MATERIALITY,
            reason="not_material: scored delta 0.050 vs threshold 0.15",
        )
    )
    log.record(
        decision(
            proposal_id="prop-c",
            learning_id="prop-c",
            stage=DecisionStage.NEVER_WORSE,
            reason="regression: 1 case(s) regressed: case-2",
        )
    )
    log.record(
        decision(
            proposal_id="prop-d",
            learning_id="prop-d",
            stage=DecisionStage.GATE_ERROR,
            reason="gate_error: replay executor unavailable",
        )
    )
    log.record(
        decision(
            proposal_id="prop-e",
            learning_id="prop-e",
            promoted=True,
            stage=DecisionStage.NEVER_WORSE,
            reason="cleared: 3 binary flips",
        )
    )

    exit_code = main(["decisions", "report", "--root", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output == (
        "promotion decisions: 5 total\n"
        "  replay: 1\n"
        "  materiality: 1\n"
        "  never_worse: 2\n"
        "  approval: 0\n"
        "  gate_error: 1\n"
        "promoted: 1\n"
    )


def test_report_renders_an_empty_log_as_zeros(tmp_path: Path, capsys) -> None:
    exit_code = main(["decisions", "report", "--root", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert output == (
        "promotion decisions: 0 total\n"
        "  replay: 0\n"
        "  materiality: 0\n"
        "  never_worse: 0\n"
        "  approval: 0\n"
        "  gate_error: 0\n"
        "promoted: 0\n"
    )


# ----------------------------------------------------- completeness invariant


@pytest.fixture
def store(tmp_path: Path) -> Engine:
    return Engine(tmp_path / "store")


@pytest.fixture
def candidates_in_store(store: Engine, record_factory) -> list[LearningRecord]:
    """Five queued proposals whose stub replays hit every decision stage."""
    records = []
    for proposal_id in ("prop-a", "prop-b", "prop-c", "prop-d", "prop-e"):
        record = store.propose(record_factory(proposal_id))
        store.queue_replay(record.id, [f"{record.id}-cold", f"{record.id}-warm"])
        records.append(store.get(record.id))
    return records


def promotion_pass_over(store: Engine) -> list[PromotionDecision]:
    runner = ScriptedRunner(
        {
            # prop-a: full binary flip — promoted.
            "prop-a-cold": binary_run("prop-a-cold", "baseline", [False, False, False]),
            "prop-a-warm": binary_run("prop-a-warm", "warm", [True, True, True]),
            # prop-b: no flip — stopped at REPLAY.
            "prop-b-cold": binary_run("prop-b-cold", "baseline", [False, True, False]),
            "prop-b-warm": binary_run("prop-b-warm", "warm", [False, True, False]),
            # prop-c: scored flip below threshold — stopped at MATERIALITY.
            "prop-c-cold": scored_run("prop-c-cold", "baseline", [0.5, 0.5, 0.5]),
            "prop-c-warm": scored_run("prop-c-warm", "warm", [0.55, 0.55, 0.55]),
            # prop-d: material flip that regresses case-2 — NEVER_WORSE.
            "prop-d-cold": binary_run("prop-d-cold", "baseline", [False, False, True]),
            "prop-d-warm": binary_run("prop-d-warm", "warm", [True, True, False]),
            # prop-e: the replay executor is down — GATE_ERROR.
        }
    )
    gate = MemoryGate(
        runner,
        DecisionLog(store.root),
        ApprovalLog(store.root),
        evidence=RunsEvidenceSink(store.root),
    )
    return run_promotion_pass(store, gate)


def test_promotion_pass_records_exactly_one_decision_per_proposal(
    store: Engine, candidates_in_store
) -> None:
    """THE completeness invariant: every decided proposal has a record."""
    decisions = promotion_pass_over(store)
    decided_ids = {record.id for record in candidates_in_store}

    assert {d.proposal_id for d in decisions} == decided_ids
    assert len(decisions) == len(decided_ids)
    logged = DecisionLog(store.root).all()
    by_proposal = [d.proposal_id for d in logged]
    assert sorted(by_proposal) == sorted(decided_ids)
    assert len(by_proposal) == len(set(by_proposal))


def test_promotion_pass_hit_every_decision_stage(store: Engine, candidates_in_store) -> None:
    """Stages are enumerable: the pass corpus exercises all five."""
    decisions = promotion_pass_over(store)

    assert {d.stage for d in decisions} == set(DecisionStage)


def test_promotion_pass_reasons_all_match_their_stage_templates(
    store: Engine, candidates_in_store
) -> None:
    decisions = promotion_pass_over(store)

    for d in decisions:
        check_reason(d.stage, d.reason)  # raises if a reason left the vocabulary


def test_promotion_pass_applies_lifecycle_verdicts(store: Engine, candidates_in_store) -> None:
    """Cleared candidates render proposals (no promotion); rejects demote."""
    decisions = promotion_pass_over(store)
    by_id = {d.proposal_id: d for d in decisions}

    # prop-a cleared every knowledge stage: the pass rendered a pending
    # proposal and stopped — the store is untouched until a human decides.
    assert by_id["prop-a"].stage is DecisionStage.APPROVAL
    assert by_id["prop-a"].promoted is False
    assert by_id["prop-a"].reason == "proposed: 3 binary flips"
    proposed = store.get("prop-a")
    assert proposed.status == "candidate"

    proposal = ApprovalLog(store.root).get("appr-prop-a")
    assert proposal.state is ApprovalState.PENDING
    assert proposal.learning_id == "prop-a"
    assert proposal.decision.reason == "cleared: 3 binary flips"
    assert proposal.decision.promoted is True  # the gate's recommendation
    assert proposal.evidence_ref == "runs/gate/prop-a.json"

    for rejected in ("prop-b", "prop-c", "prop-d"):
        assert store.get(rejected).status == "dropped", by_id[rejected].reason

    # GATE_ERROR is infrastructure, not knowledge: the record stays a
    # candidate and its decision carries no evidence ref.
    assert by_id["prop-e"].stage is DecisionStage.GATE_ERROR
    assert store.get("prop-e").status == "candidate"
    assert by_id["prop-e"].evidence_ref is None


def test_promotion_pass_writes_evidence_for_replayed_proposals(
    store: Engine, candidates_in_store
) -> None:
    decisions = promotion_pass_over(store)

    for d in decisions:
        if d.stage is DecisionStage.GATE_ERROR:
            continue
        evidence_file = store.root / d.evidence_ref
        assert evidence_file.exists(), d.proposal_id
        assert d.replay_run_ids[0] in evidence_file.read_text(encoding="utf-8")


def test_engine_candidates_lists_only_queued_records(store: Engine, record_factory) -> None:
    store.propose(record_factory("lr-ephemeral"))
    store.propose(record_factory("lr-queued"))
    store.queue_replay("lr-queued", ["run-1"])
    store.propose(record_factory("lr-promoted"))
    store.queue_replay("lr-promoted", ["run-2"])
    store.promote(
        "lr-promoted",
        store.get("lr-promoted").counterfactual.model_copy(
            update={
                "replay_run_ids": ["run-2"],
                "verdict_change": True,
                "evidence": "runs/x.json",
            }
        ),
    )

    assert [record.id for record in store.candidates()] == ["lr-queued"]
