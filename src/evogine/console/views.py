"""Read-only view models over the store's telemetry files.

Every console page renders one of the typed views built here. The builders
are pure functions of the store root — no HTTP, no templates, no FastAPI —
so the route layer is a thin transport and this module is testable without
the web stack. The read path reuses the same file-backed logs the CLI uses
(:class:`DecisionLog`, :class:`ApprovalLog`, :class:`TrajectoryStore`, the
store's ``audit.jsonl``); the console never invents its own view of the
store, and nothing here writes.

Evidence pairing reuses :func:`align_verdicts` — replay verdicts pair by
case-instance identity, never by list position, and a malformed or
unpairable evidence file surfaces as a visible refusal state, not as a
wrong table.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from evogine.console.launch import ConsoleError
from evogine.engine.store import AuditEvent, Engine, StoreError
from evogine.gates.approvals import ApprovalLog
from evogine.gates.materiality import MaterialityVerdict, Verdict, align_verdicts
from evogine.gates.telemetry import ID_PATTERN, DecisionLog, PromotionDecision
from evogine.gates.trajectory import TrajectoryError, TrajectoryStore


class OverviewCounts(BaseModel):
    """The numbers `evogine decisions report` prints, plus the approval queue depth."""

    total_decisions: int
    by_stage: dict[str, int]
    promoted: int
    pending_approvals: int


def build_overview(root: Path | str) -> OverviewCounts:
    """Decision counts by stage, promoted total, and pending-approval count.

    Computed through the same :class:`DecisionLog` APIs the CLI's
    ``decisions report`` uses, so the overview page and the report cannot
    disagree without one of the two having changed.
    """
    log = DecisionLog(root)
    decisions = log.all()
    return OverviewCounts(
        total_decisions=len(decisions),
        by_stage={stage.value: count for stage, count in log.counts_by_stage().items()},
        promoted=sum(1 for decision in decisions if decision.promoted),
        pending_approvals=len(ApprovalLog(root).pending()),
    )


def _fmt(moment: datetime | None) -> str | None:
    """Render a timestamp for a table cell (second precision, offset kept)."""
    if moment is None:
        return None
    return moment.isoformat(timespec="seconds")


class DecisionRow(BaseModel):
    """One row of the decision list."""

    proposal_id: str
    learning_id: str
    decided_at: str | None
    promoted: bool
    stage: str
    reason: str
    evidence_ref: str | None
    trajectory_ref: str | None


def list_decisions(root: Path | str) -> list[DecisionRow]:
    """Every recorded decision, in the decision log's own order (by proposal id)."""
    return [
        DecisionRow(
            proposal_id=decision.proposal_id,
            learning_id=decision.learning_id,
            decided_at=_fmt(decision.decided_at),
            promoted=decision.promoted,
            stage=decision.stage.value,
            reason=decision.reason,
            evidence_ref=decision.evidence_ref,
            trajectory_ref=decision.trajectory_ref,
        )
        for decision in DecisionLog(root).all()
    ]


class VerdictPairView(BaseModel):
    """One baseline/warm pair, aligned by case-instance identity."""

    instance_id: str
    baseline: str
    warm: str


class EvidenceView(BaseModel):
    """The paired replay evidence behind one decision."""

    learning_id: str
    replay_run_ids: list[str] = Field(default_factory=list)
    material: bool
    materiality_detail: str
    pairs: list[VerdictPairView] = Field(default_factory=list)
    pairing_error: str | None = None
    """Set when the two arms cannot be paired (mismatched case sets): the
    file is shown as refused rather than silently half-rendered."""


class TrajectoryStepView(BaseModel):
    """One rendered step of a gate-review trajectory."""

    seq: int
    kind: str
    detail: str
    tool: str | None = None
    arguments: str | None = None
    result: str | None = None


class TrajectoryView(BaseModel):
    """The sanitized, ordered capture of one gate review."""

    gate: str
    started_at: str | None
    decided_at: str | None
    steps: list[TrajectoryStepView] = Field(default_factory=list)


class DecisionDetail(BaseModel):
    """Everything the drill-down page shows for one decision."""

    row: DecisionRow
    record_status: str | None = None
    """The learning's current lifecycle status, when the record exists."""
    record_status_note: str | None = None
    evidence: EvidenceView | None = None
    evidence_note: str | None = None
    """Why evidence is absent, when it is (gate errors write none)."""
    trajectory: TrajectoryView | None = None
    trajectory_note: str | None = None


def _fmt_verdict(verdict: Verdict) -> str:
    """Render one replay verdict: ``pass``/``fail`` or a scored value."""
    if verdict.passed is not None:
        return "pass" if verdict.passed else "fail"
    if verdict.score is not None:
        return f"score {verdict.score:.3f}"
    raise ConsoleError(f"verdict {verdict.instance_id!r} carries no outcome")


class _EvidencePayload(BaseModel):
    """The shape the memory gate's :class:`RunsEvidenceSink` writes."""

    learning_id: str
    replay_run_ids: list[str] = Field(default_factory=list)
    materiality: MaterialityVerdict
    baseline: list[Verdict] = Field(default_factory=list)
    warm: list[Verdict] = Field(default_factory=list)


def _load_evidence(root: Path, learning_id: str) -> EvidenceView | None:
    """Load and pair ``runs/gate/<learning_id>.json``; ``None`` when absent.

    The gate keys evidence files by the record's learning id (what the sink
    was handed), not by proposal id; the path is derived from the
    pattern-validated learning id, never from a stored ref string — a
    hand-crafted decision file cannot aim the read path outside the store.
    """
    path = Path(root) / "runs" / "gate" / f"{learning_id}.json"
    if not path.exists():
        return None
    try:
        payload = _EvidencePayload.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConsoleError(f"evidence file {path.name} is malformed: {exc}") from exc
    try:
        pairs = [
            VerdictPairView(
                instance_id=baseline.instance_id,
                baseline=_fmt_verdict(baseline),
                warm=_fmt_verdict(warm),
            )
            for baseline, warm in align_verdicts(payload.baseline, payload.warm)
        ]
        pairing_error = None
    except ValueError as exc:  # MaterialityError: mismatched arms are data, not a crash
        pairs = []
        pairing_error = str(exc)
    return EvidenceView(
        learning_id=payload.learning_id,
        replay_run_ids=payload.replay_run_ids,
        material=payload.materiality.material,
        materiality_detail=payload.materiality.detail,
        pairs=pairs,
        pairing_error=pairing_error,
    )


def _load_trajectory(
    root: Path, learning_id: str
) -> tuple[TrajectoryView | None, str | None]:
    """Load ``runs/trajectory/<learning_id>.json`` or explain its absence."""
    try:
        trajectory = TrajectoryStore(root).read(learning_id)
    except TrajectoryError as exc:
        return None, str(exc)
    view = TrajectoryView(
        gate=trajectory.gate,
        started_at=_fmt(trajectory.started_at),
        decided_at=_fmt(trajectory.decided_at),
        steps=[
            TrajectoryStepView(
                seq=step.seq,
                kind=step.kind,
                detail=step.detail,
                tool=step.tool_call.tool if step.tool_call else None,
                arguments=step.tool_call.arguments if step.tool_call else None,
                result=step.tool_call.result if step.tool_call else None,
            )
            for step in trajectory.steps
        ],
    )
    return view, None


def build_decision_detail(root: Path | str, proposal_id: str) -> DecisionDetail:
    """One decision plus its evidence, trajectory, and record status.

    ``proposal_id`` comes from the URL: it is checked against the store's
    filesystem-safe id rule before anything is read, so a traversal-shaped
    id is a visible refusal, never a path escape.
    """
    if not ID_PATTERN.fullmatch(proposal_id):
        raise ConsoleError(f"proposal id {proposal_id!r} is not a filesystem-safe id")
    store_root = Path(root)
    decision_path = DecisionLog(store_root).decisions_dir / f"{proposal_id}.json"
    if not decision_path.exists():
        raise ConsoleError(f"no decision recorded for proposal {proposal_id!r}")
    try:
        decision = PromotionDecision.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ConsoleError(f"decision file {decision_path.name} is malformed: {exc}") from exc
    # Evidence and trajectory files are keyed by the record's learning id
    # (what the sinks were handed), never by proposal id; both are checked
    # against the filesystem-safe rule before they touch a path.
    if not ID_PATTERN.fullmatch(decision.learning_id):
        raise ConsoleError(
            f"learning id {decision.learning_id!r} is not a filesystem-safe id"
        )

    row = DecisionRow(
        proposal_id=decision.proposal_id,
        learning_id=decision.learning_id,
        decided_at=_fmt(decision.decided_at),
        promoted=decision.promoted,
        stage=decision.stage.value,
        reason=decision.reason,
        evidence_ref=decision.evidence_ref,
        trajectory_ref=decision.trajectory_ref,
    )
    try:
        record_status: str | None = Engine(store_root).get(decision.learning_id).status
        record_status_note = None
    except StoreError:
        record_status = None
        record_status_note = f"no record {decision.learning_id} in this store"

    evidence = _load_evidence(store_root, decision.learning_id)
    evidence_note = None
    if evidence is None:
        evidence_note = (
            "no evidence file for this proposal — gate-error decisions write none"
        )
    trajectory, trajectory_note = _load_trajectory(store_root, decision.learning_id)
    return DecisionDetail(
        row=row,
        record_status=record_status,
        record_status_note=record_status_note,
        evidence=evidence,
        evidence_note=evidence_note,
        trajectory=trajectory,
        trajectory_note=trajectory_note,
    )


class ApprovalRow(BaseModel):
    """One row of the approvals queue."""

    proposal_id: str
    learning_id: str
    created_at: str | None
    state: str
    decided_by: str | None
    decided_at: str | None
    decided_reason: str | None
    evidence_ref: str | None
    trail_stage: str
    """The gate stage of the cleared verdict the proposal carries."""
    trail_reason: str
    """The machine-checked cleared verdict itself."""


def list_approvals(root: Path | str) -> list[ApprovalRow]:
    """Every proposal in the log, pending ones first, then by proposal id.

    The queue is rendered from the :class:`ApprovalLog`'s own read API —
    the same one ``evogine approvals list`` uses — so the page and the CLI
    show one queue, not two.
    """
    proposals = ApprovalLog(root).all()
    pending = [p for p in proposals if p.state.value == "pending"]
    decided = [p for p in proposals if p.state.value != "pending"]
    return [
        ApprovalRow(
            proposal_id=proposal.proposal_id,
            learning_id=proposal.learning_id,
            created_at=_fmt(proposal.created_at),
            state=proposal.state.value,
            decided_by=proposal.decided_by,
            decided_at=_fmt(proposal.decided_at),
            decided_reason=proposal.decided_reason,
            evidence_ref=proposal.evidence_ref,
            trail_stage=proposal.decision.stage.value,
            trail_reason=proposal.decision.reason,
        )
        for proposal in [*pending, *decided]
    ]


class AuditRow(BaseModel):
    """One entry of the append-only audit trail."""

    seq: int
    ts: str | None
    event: str
    record_id: str
    from_status: str | None
    to_status: str
    reason: str | None
    run_ids: list[str] = Field(default_factory=list)


def build_timeline(root: Path | str) -> list[AuditRow]:
    """The store's ``audit.jsonl`` as a timeline, newest event first.

    A malformed line is a corrupted audit trail — the console refuses
    visibly (names the line) instead of silently skipping what it cannot
    parse; an audit surface that skips lines is lying by omission.
    """
    path = Path(root) / "audit.jsonl"
    if not path.exists():
        return []
    rows: list[AuditRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = AuditEvent.model_validate_json(line)
        except ValueError as exc:
            raise ConsoleError(f"audit line {line_no} is malformed: {exc}") from exc
        rows.append(
            AuditRow(
                seq=event.seq,
                ts=_fmt(event.ts),
                event=event.event,
                record_id=event.record_id,
                from_status=event.from_status,
                to_status=event.to_status,
                reason=event.reason,
                run_ids=event.run_ids,
            )
        )
    rows.reverse()
    return rows


__all__ = [
    "ApprovalRow",
    "AuditRow",
    "DecisionDetail",
    "DecisionRow",
    "EvidenceView",
    "OverviewCounts",
    "TrajectoryStepView",
    "TrajectoryView",
    "VerdictPairView",
    "build_decision_detail",
    "build_overview",
    "build_timeline",
    "list_approvals",
    "list_decisions",
]
