"""Proposal-funnel tests: hand-computed counts, vocabulary exactness, honesty.

The acceptance for the funnel report: the numbers it prints match counts
computed BY HAND from a fixture store's files — every expected number in
``test_funnel_matches_hand_computed_counts`` is a literal derived from the
fixture written by ``hand_computed_fixture`` (the arithmetic is commented
inline), never by calling the funnel implementation twice. The vocabulary
tests prove the report's reason rows are exactly the machine-checked labels
the gate enforces at record time, the live-evidence tests prove token counts
aggregate when present and vanish honestly when absent, and the CLI tests
pin both renderings. All offline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evogine.bench.gate_replay import replay_run_ids
from evogine.cli import main
from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.approvals import (
    ApprovalLog,
    ApprovalState,
    PromotionProposal,
    approve_proposal,
)
from evogine.gates.funnel import (
    FunnelReport,
    build_funnel_report,
    refusal_reasons,
)
from evogine.gates.telemetry import (
    STAGE_REASON_LABELS,
    DecisionStage,
    PassDecisionSummary,
    PassManifest,
    PromotionDecision,
    check_reason,
)

PASS_ONE = "pass-20260919T010101-aaaaaa01"
PASS_TWO = "pass-20260919T020202-bbbbbb02"

# The hand-computed fixture, decision by decision (this comment table IS the
# hand computation the acceptance requires — read the files, count, assert):
#
#   proposal | pass   | stage       | label
#   ---------+--------+-------------+--------------------------------------
#   alpha    | pass 1 | replay      | no_flip          (stopped at replay)
#   beta     | pass 1 | materiality | not_material     (cleared replay,
#            |        |             |                   stopped at materiality)
#   gamma    | pass 1 | approval    | proposed         (cleared everything;
#            |        |             |                   human approved + applied)
#   delta    | pass 1 | gate_error  | gate_error       (infrastructure)
#   epsilon  | pass 2 | approval    | refused_no_evidence
#   zeta     | (none) | never_worse | regression       (legacy, pre-#17)
#
# Cumulative (from these six decision files, the applied proposal, and the
# one promoted record file):
#   candidates          = 6
#   replay_cleared      = 4   (beta, gamma, epsilon, zeta)
#   materiality_cleared = 3   (gamma, epsilon, zeta)
#   approved            = 1   (gamma: proposal state applied)
#   promoted            = 1   (gamma: record status promoted)
#   gate_errors         = 1   (delta)
#   pending / rejected  = 0 / 0
#   reasons: replay no_flip=1; materiality not_material=1;
#            never_worse regression=1, cleared=0;
#            approval proposed=1, refused_no_evidence=1; gate_error=1
#
# Per pass (decision files stamped with that pass's id):
#   pass 1: candidates=4, decisions=4, replay-cleared=2 (beta, gamma),
#           materiality-cleared=1 (gamma), approved=1, promoted=1, errors=1
#   pass 2: candidates=1, decisions=1, replay-cleared=1,
#           materiality-cleared=1, approved=0, promoted=0, errors=0


def write_decision(
    root: Path,
    proposal_id: str,
    *,
    stage: DecisionStage,
    reason: str,
    promoted: bool = False,
    pass_id: str | None = None,
    learning_id: str | None = None,
) -> None:
    """Write one decision file in the store's production schema."""
    decision = PromotionDecision(
        proposal_id=proposal_id,
        learning_id=learning_id or proposal_id,
        promoted=promoted,
        stage=stage,
        reason=reason,
        pass_id=pass_id,
    )
    decisions_dir = root / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    (decisions_dir / f"{proposal_id}.json").write_text(
        decision.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def write_pass(
    root: Path,
    pass_id: str,
    candidates: list[str],
    decided: list[tuple[str, DecisionStage, bool]],
) -> None:
    """Write one pass manifest in the store's production schema."""
    manifest = PassManifest(
        pass_id=pass_id,
        started_at=datetime(2026, 9, 19, 1, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 19, 1, 5, 0, tzinfo=UTC),
        candidates=candidates,
        decisions=[
            PassDecisionSummary(proposal_id=pid, stage=stage, promoted=promoted)
            for pid, stage, promoted in decided
        ],
        counts_by_stage={},
    )
    passes_dir = root / "passes"
    passes_dir.mkdir(parents=True, exist_ok=True)
    (passes_dir / f"{pass_id}.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def write_proposal(
    root: Path, learning_id: str, state: ApprovalState, pass_id: str | None
) -> None:
    """Write one approval proposal whose cleared trail came from ``pass_id``."""
    decided = state is not ApprovalState.PENDING
    proposal = PromotionProposal(
        proposal_id=f"appr-{learning_id}",
        learning_id=learning_id,
        decision=PromotionDecision(
            proposal_id=learning_id,
            learning_id=learning_id,
            promoted=True,
            stage=DecisionStage.NEVER_WORSE,
            reason="cleared: 2 binary flips",
            pass_id=pass_id,
        ),
        evidence_ref=f"runs/gate/{learning_id}.json",
        state=state,
        decided_by="Nihanth" if decided else None,
        decided_at=datetime(2026, 9, 19, 2, 0, 0, tzinfo=UTC) if decided else None,
    )
    approvals_dir = root / "approvals"
    approvals_dir.mkdir(parents=True, exist_ok=True)
    (approvals_dir / f"{proposal.proposal_id}.json").write_text(
        proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def write_record(root: Path, record_id: str, status: str) -> None:
    """Write one record file with ``status`` (records/ production schema)."""
    record = LearningRecord.sample().model_copy(
        update={
            "id": record_id,
            "status": status,
            "promoted_at": (
                datetime(2026, 9, 19, 2, 0, 0, tzinfo=UTC) if status == "promoted" else None
            ),
        }
    )
    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / f"{record_id}.json").write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def hand_computed_fixture(tmp_path: Path) -> Path:
    """The six-decision, two-pass fixture store; counts commented above."""
    root = tmp_path / "store"
    root.mkdir()
    write_pass(
        root,
        PASS_ONE,
        ["alpha", "beta", "gamma", "delta"],
        [
            ("alpha", DecisionStage.REPLAY, False),
            ("beta", DecisionStage.MATERIALITY, False),
            ("gamma", DecisionStage.APPROVAL, False),
            ("delta", DecisionStage.GATE_ERROR, False),
        ],
    )
    write_decision(
        root,
        "alpha",
        stage=DecisionStage.REPLAY,
        reason="no_flip: 0/3 replays changed verdict",
        pass_id=PASS_ONE,
    )
    write_decision(
        root,
        "beta",
        stage=DecisionStage.MATERIALITY,
        reason="not_material: scored delta 0.020 vs threshold 0.15",
        pass_id=PASS_ONE,
    )
    write_decision(
        root,
        "gamma",
        stage=DecisionStage.APPROVAL,
        reason="proposed: 2 binary flips",
        pass_id=PASS_ONE,
    )
    write_decision(
        root,
        "delta",
        stage=DecisionStage.GATE_ERROR,
        reason="gate_error: replay run cold-x failed: boom",
        pass_id=PASS_ONE,
    )
    write_pass(
        root,
        PASS_TWO,
        ["epsilon"],
        [("epsilon", DecisionStage.APPROVAL, False)],
    )
    write_decision(
        root,
        "epsilon",
        stage=DecisionStage.APPROVAL,
        reason="refused_no_evidence: no replay evidence attached for epsilon",
        pass_id=PASS_TWO,
    )
    write_decision(
        root,
        "zeta",
        stage=DecisionStage.NEVER_WORSE,
        reason="regression: 1 case(s) regressed: case-9",
        pass_id=None,
    )
    write_proposal(root, "gamma", ApprovalState.APPLIED, pass_id=PASS_ONE)
    write_record(root, "gamma", "promoted")
    write_record(root, "epsilon", "candidate")
    return root


def test_funnel_matches_hand_computed_counts(hand_computed_fixture: Path) -> None:
    """THE acceptance: report == hand-counted literals from the fixture files."""
    report = build_funnel_report(hand_computed_fixture)

    # Cumulative — see the comment table on the fixture above.
    assert report.funnel.candidates == 6
    assert report.funnel.replay_cleared == 4
    assert report.funnel.materiality_cleared == 3
    assert report.funnel.approved == 1
    assert report.funnel.promoted == 1
    assert report.funnel.gate_errors == 1
    assert report.funnel.proposals_pending == 0
    assert report.funnel.proposals_rejected == 0

    # Refusal reasons by stage — one line per fixture decision.
    assert report.reasons_by_stage == {
        "replay": {"no_flip": 1},
        "materiality": {"not_material": 1},
        "never_worse": {"regression": 1, "cleared": 0},
        "approval": {"proposed": 1, "refused_no_evidence": 1},
        "gate_error": {"gate_error": 1},
    }

    # Per pass.
    assert [entry.pass_id for entry in report.passes] == [PASS_ONE, PASS_TWO]
    first, second = report.passes
    assert first.candidates == 4
    assert first.decisions == 4
    assert first.replay_cleared == 2
    assert first.materiality_cleared == 1
    assert first.approved == 1
    assert first.promoted == 1
    assert first.gate_errors == 1
    assert second.candidates == 1
    assert second.decisions == 1
    assert second.replay_cleared == 1
    assert second.materiality_cleared == 1
    assert second.approved == 0
    assert second.promoted == 0
    assert second.gate_errors == 0

    # No crash-window artifacts in this fixture.
    assert report.decisions_without_manifest == 0
    assert report.live_tokens is None  # no runs/ written — absent, not zero


def test_legacy_decision_without_pass_id_stays_cumulative_only(
    hand_computed_fixture: Path,
) -> None:
    """A pre-#17 decision (no pass_id) appears in the funnel, in no pass."""
    report = build_funnel_report(hand_computed_fixture)

    # zeta is the only decision with pass_id None: per-pass decisions sum to 5.
    assert sum(entry.decisions for entry in report.passes) == 5
    assert report.funnel.candidates == 6


def test_redecided_candidate_shows_the_honest_gap(hand_computed_fixture: Path) -> None:
    """A candidate re-decided by pass two: pass one shows fewer decisions."""
    # beta was re-decided in pass two (its decision file now carries pass two).
    write_decision(
        hand_computed_fixture,
        "beta",
        stage=DecisionStage.MATERIALITY,
        reason="not_material: scored delta 0.010 vs threshold 0.15",
        pass_id=PASS_TWO,
    )

    report = build_funnel_report(hand_computed_fixture)

    first, second = report.passes
    assert first.candidates == 4
    assert first.decisions == 3  # alpha, gamma, delta — beta's file moved on
    assert second.decisions == 2  # epsilon + the re-decided beta
    assert report.funnel.candidates == 6  # unchanged: distinct proposals
    assert report.decisions_without_manifest == 0


def test_crash_window_decisions_are_shown(tmp_path: Path) -> None:
    """A decision whose pass never finished (no manifest) is counted, not hidden."""
    root = tmp_path / "store"
    root.mkdir()
    write_decision(
        root,
        "alpha",
        stage=DecisionStage.REPLAY,
        reason="no_flip: 0/3 replays changed verdict",
        pass_id="pass-lost",
    )

    report = build_funnel_report(root)

    assert report.passes == []
    assert report.decisions_without_manifest == 1
    assert report.funnel.candidates == 1


def test_reasons_breakdown_covers_the_vocabulary_exactly(tmp_path: Path) -> None:
    """Rows == the machine-checked vocabulary: every stage, every label, zero-filled."""
    root = tmp_path / "store"
    root.mkdir()

    report = build_funnel_report(root)

    assert set(report.reasons_by_stage) == {stage.value for stage in DecisionStage}
    for stage in DecisionStage:
        assert sorted(report.reasons_by_stage[stage.value]) == sorted(
            STAGE_REASON_LABELS[stage]
        )
        assert all(count == 0 for count in report.reasons_by_stage[stage.value].values())


def test_every_vocabulary_label_is_aggregable() -> None:
    """Any reason check_reason accepts lands in exactly one (stage, label) row.

    The reasons below are the minimal template-valid evidence per vocabulary
    label — built here, not copied, so the test fails if either side drifts.
    """
    minimal_evidence: dict[str, str] = {
        "no_flip": "no_flip: 0/1 replays changed verdict",
        "not_material": "not_material: scored delta 0.000 vs threshold 0.15",
        "regression": "regression: 1 case(s) regressed",
        "cleared": "cleared: 1 binary flips",
        "proposed": "proposed: 1 binary flips",
        "refused_no_evidence": "refused_no_evidence: no replay evidence attached",
        "gate_error": "gate_error: replay unavailable",
    }
    decisions = []
    for stage in DecisionStage:
        for label in sorted(STAGE_REASON_LABELS[stage]):
            reason = minimal_evidence[label]
            check_reason(stage, reason)  # the gate's own record-time check
            decisions.append(
                PromotionDecision(
                    proposal_id=f"prop-{stage.value}-{label}",
                    learning_id=f"prop-{stage.value}-{label}",
                    promoted=False,
                    stage=stage,
                    reason=reason,
                )
            )

    breakdown = refusal_reasons(decisions)

    for stage in DecisionStage:
        for label in STAGE_REASON_LABELS[stage]:
            assert breakdown[stage.value][label] == 1


def test_prose_reason_fails_loudly(tmp_path: Path) -> None:
    """A hand-edited, off-vocabulary decision breaks the report — visibly."""
    from evogine.gates.telemetry import TelemetryError

    root = tmp_path / "store"
    root.mkdir()
    decisions_dir = root / "decisions"
    decisions_dir.mkdir(parents=True)
    (decisions_dir / "rogue.json").write_text(
        PromotionDecision(
            proposal_id="rogue",
            learning_id="rogue",
            promoted=False,
            stage=DecisionStage.REPLAY,
            reason="honestly nothing changed here",
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(TelemetryError, match="outside the machine-checked vocabulary"):
        build_funnel_report(root)

    assert main(["decisions", "funnel", "--root", str(root)]) == 1


def test_missing_root_is_refused(tmp_path: Path) -> None:
    """A mistyped --root fails closed instead of creating directories."""
    from evogine.gates.telemetry import TelemetryError

    missing = tmp_path / "no-such-store"

    with pytest.raises(TelemetryError, match="does not exist"):
        build_funnel_report(missing)

    assert not missing.exists()
    assert main(["decisions", "funnel", "--root", str(missing)]) == 1


def test_empty_store_reports_honest_empties(tmp_path: Path) -> None:
    """An offline, never-run store: zeros and none-found, never fabricated spend."""
    root = tmp_path / "store"
    root.mkdir()

    report = build_funnel_report(root)

    assert report.funnel.candidates == 0
    assert report.funnel.promoted == 0
    assert report.passes == []
    assert report.live_tokens is None
    assert report.decisions_without_manifest == 0


def write_call_record(path: Path, **fields: object) -> None:
    """Persist one live-call record in the runner's artifact schema."""
    payload: dict[str, object] = {
        "case_id": "case-a",
        "arm": "cold",
        "repeat": 0,
        "call_index": 1,
        "prompt_pack_fingerprint": "pack-001",
        "ok": True,
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "error": None,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_live_token_evidence_aggregates_by_model(tmp_path: Path) -> None:
    """Per-call artifacts sum by hand: calls, tokens, per model; corrupt shown."""
    root = tmp_path / "store"
    root.mkdir()
    # Two ok calls on claude-x: 100+120 in, 10+15 out.
    write_call_record(
        root / "runs" / "replay" / "run-1" / "artifacts" / "call-case-a-baseline-0000-0001.json",
        model="claude-x",
        input_tokens=100,
        output_tokens=10,
    )
    write_call_record(
        root / "runs" / "replay" / "run-1" / "artifacts" / "call-case-a-warm-0000-0002.json",
        arm="warm",
        call_index=2,
        model="claude-x",
        input_tokens=120,
        output_tokens=15,
    )
    # One ok call on claude-y: 50 in, 5 out.
    write_call_record(
        root / "runs" / "replay" / "run-2" / "artifacts" / "call-case-b-warm-0000-0003.json",
        case_id="case-b",
        arm="warm",
        model="claude-y",
        input_tokens=50,
        output_tokens=5,
    )
    # One ok call the provider answered without a model id: lands in the
    # "unknown" bucket rather than being dropped from the totals.
    write_call_record(
        root / "runs" / "replay" / "run-2" / "artifacts" / "call-case-b-warm-0000-0004.json",
        case_id="case-b",
        arm="warm",
        call_index=4,
    )
    # One failed call: counted, spent nothing billable (model/tokens None).
    write_call_record(
        root / "runs" / "replay" / "run-2" / "artifacts" / "call-case-b-baseline-0000-0005.json",
        case_id="case-b",
        call_index=5,
        ok=False,
        error="provider down",
    )
    # One file matching the artifact name that is not a valid call record.
    corrupt = root / "runs" / "replay" / "run-3" / "artifacts" / "call-corrupt-0000-0006.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("not a call record", encoding="utf-8")

    report = build_funnel_report(root)

    live = report.live_tokens
    assert live is not None
    assert live.call_records == 5
    assert live.ok == 4
    assert live.failed == 1
    assert live.input_tokens == 270
    assert live.output_tokens == 30
    assert live.by_model["claude-x"].calls == 2
    assert live.by_model["claude-x"].input_tokens == 220
    assert live.by_model["claude-x"].output_tokens == 25
    assert live.by_model["claude-y"].calls == 1
    assert live.by_model["claude-y"].input_tokens == 50
    assert live.by_model["unknown"].calls == 1
    assert live.by_model["unknown"].input_tokens == 0
    assert live.unreadable_records == 1


def test_live_token_evidence_absent_is_none(tmp_path: Path) -> None:
    """No runs tree, or a runs tree with no call artifacts — honestly None."""
    root = tmp_path / "store"
    root.mkdir()
    assert build_funnel_report(root).live_tokens is None

    (root / "runs" / "shadow" / "cand-1" / "artifacts").mkdir(parents=True)
    (root / "runs" / "shadow" / "cand-1" / "artifacts" / "profile.json").write_text("{}")
    assert build_funnel_report(root).live_tokens is None


def test_cli_funnel_table_matches_hand_computed_counts(
    hand_computed_fixture: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table rendering prints the same numbers the report computed."""
    assert main(["decisions", "funnel", "--root", str(hand_computed_fixture)]) == 0

    out = capsys.readouterr().out
    assert "candidates:          6" in out
    assert "replay-cleared:      4" in out
    assert "materiality-cleared: 3" in out
    assert "approved:            1" in out
    assert "promoted:            1" in out
    assert "gate errors:         1" in out
    assert "no_flip: 1" in out
    assert f"  {PASS_ONE}:" in out
    assert "live run token evidence: none found" in out


def test_cli_funnel_json_matches_the_report(
    hand_computed_fixture: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--format json`` emits the structured report for downstream surfaces."""
    assert (
        main(["decisions", "funnel", "--root", str(hand_computed_fixture), "--format", "json"])
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    report = FunnelReport.model_validate(payload)
    assert report.funnel.candidates == 6
    assert report.funnel.replay_cleared == 4
    assert len(report.passes) == 2
    assert report.live_tokens is None


def test_cli_funnel_end_to_end_promotes_and_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real pipeline: pass proposes, a human approves, the funnel counts it."""
    # A draft whose signal echoes the retry case's own context — the same
    # anchoring the CLI surface tests use to produce a material flip.
    from evogine.bench.cases.corpus import default_corpus
    from evogine.engine.records import RecordBody

    root = tmp_path / "store"
    case = next(c for c in default_corpus() if c.id == "retry-jitter-budget")
    draft = json.dumps(
        {
            "type": "failure-pattern",
            "title": "Retry policy for the rate-limited client",
            "body": {
                "signal": case.task_context,
                "failure_mode": "immediate tight retries amplify a small stall into an outage",
                "detection": "the retry policy ignores attempt bounds",
                "fix_strategy": "use jittered exponential backoff with a hard cap on attempts",
            },
            "confidence": 0.8,
        }
    )
    parsed = json.loads(draft)
    record = LearningRecord.sample().model_copy(
        update={
            "title": parsed["title"],
            "body": RecordBody.model_validate(parsed["body"]),
        }
    )
    engine = Engine(str(root))
    engine.propose(record)
    engine.queue_replay(record.id, replay_run_ids(record.id))

    assert main(["run", "--root", str(root)]) == 0
    capsys.readouterr()  # consume the pass's human-readable run output
    proposal = ApprovalLog(root).all()[0]
    assert proposal.state is ApprovalState.PENDING

    # The funnel BEFORE the human decides: cleared, proposed, promoted nothing.
    assert main(["decisions", "funnel", "--root", str(root), "--format", "json"]) == 0
    pending_report = FunnelReport.model_validate(json.loads(capsys.readouterr().out))
    assert pending_report.funnel.candidates == 1
    assert pending_report.funnel.approved == 0
    assert pending_report.funnel.promoted == 0
    assert pending_report.funnel.proposals_pending == 1

    approve_proposal(engine, ApprovalLog(root), proposal.proposal_id, decided_by="Nihanth")

    assert main(["decisions", "funnel", "--root", str(root), "--format", "json"]) == 0
    applied_report = FunnelReport.model_validate(json.loads(capsys.readouterr().out))
    assert applied_report.funnel.approved == 1
    assert applied_report.funnel.promoted == 1
    assert applied_report.funnel.proposals_pending == 0
    # The proposal belongs to the pass that cleared it — which ran above.
    assert applied_report.passes[0].promoted == 1


def test_decisions_report_still_works(tmp_path: Path) -> None:
    """The sibling ``decisions report`` command is untouched by the funnel."""
    root = tmp_path / "store"
    root.mkdir()
    write_decision(
        root,
        "alpha",
        stage=DecisionStage.REPLAY,
        reason="no_flip: 0/3 replays changed verdict",
    )

    assert main(["decisions", "report", "--root", str(root)]) == 0


def test_funnel_module_exports_are_complete() -> None:
    """The public surface the console panel (phase two) will reuse exists."""
    from evogine.gates import funnel

    for name in ("build_funnel_report", "FunnelReport", "funnel_counts", "pass_funnel"):
        assert getattr(funnel, name) is not None
