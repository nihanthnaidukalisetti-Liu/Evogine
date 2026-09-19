"""Proposal-quality funnel: per-pass promotion telemetry, aggregated.

V3's proposal-quality question — of the proposals a pass reviewed, how many
cleared each stage, where the rest stopped, and why — is answerable from
records that already exist. This module adds no new store concepts; it reads
the four surfaces the engine already writes and computes the funnel:

- ``decisions/<proposal_id>.json`` — the gate's latest verdict per proposal
  (:class:`~evogine.gates.telemetry.PromotionDecision`, with its stage and a
  template-checked reason). The funnel's stage arithmetic reads these files:
  a proposal reached past replay exactly when its decision's stage is
  materiality or later, and past materiality exactly when it is never_worse
  or approval. ``pass_id`` (PR #17) attributes each decision to the pass
  that produced it, so the same arithmetic runs per pass and cumulatively.
- ``passes/<pass_id>.json`` — the pass manifests (:class:`~evogine.gates.telemetry.PassManifest`):
  each pass's candidate list and wall-clock bounds. A candidate that a later
  pass re-decided shows honestly here as a pass with fewer decisions than
  candidates — the manifest remains the immutable record of what that pass saw.
- ``approvals/<proposal_id>.json`` — the human boundary's state machine
  (:class:`~evogine.gates.approvals.PromotionProposal`): approved, applied,
  rejected, or pending. The approved stage of the funnel is the log's
  decision, not the gate's recommendation.
- ``records/*.json`` — the lifecycle truth: a record is promoted when its
  status says so. The promoted stage of the cumulative funnel counts these.

Refusal reasons are grouped by ``(stage, label)`` over the machine-checked
vocabulary (:data:`~evogine.gates.telemetry.STAGE_REASON_LABELS`) — the same
labels :func:`~evogine.gates.telemetry.check_reason` enforces at record time,
imported from there rather than copied. The breakdown is zero-filled over the
whole vocabulary so every stage and label is always visible, and a reason
that is not in the vocabulary raises rather than aggregates: telemetry that
drifted into prose must fail loudly here, not silently reshape the report.

Live-run token evidence (the per-call ``call-*.json`` artifacts the bench
live runner persists under a run directory, PR #15) is surfaced when present
and absent honestly when not: no call records means ``live_tokens=None``,
never a fabricated zero-spend story.

The pure computation (:func:`funnel_counts`, :func:`refusal_reasons`,
:func:`pass_funnel`) is separated from the file reads
(:func:`build_funnel_report`) so the numbers are hand-checkable against a
fixture store's files — the acceptance for this module — and so the console
panel (phase two, after the console PR merges) can render the same
:class:`FunnelReport` with parity to ``evogine decisions funnel``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from evogine.engine.records import LearningRecord
from evogine.gates.approvals import ApprovalLog, ApprovalState, PromotionProposal
from evogine.gates.telemetry import (
    STAGE_REASON_LABELS,
    DecisionLog,
    DecisionStage,
    PassManifest,
    PromotionDecision,
    TelemetryError,
)

#: Stages whose decision implies the replay stage flipped a verdict: only a
#: cleared replay reaches materiality, the never-worse check, or approval.
REPLAY_CLEARED_STAGES: frozenset[DecisionStage] = frozenset(
    {DecisionStage.MATERIALITY, DecisionStage.NEVER_WORSE, DecisionStage.APPROVAL}
)

#: Stages whose decision implies the materiality threshold was met: only a
#: material flip reaches the never-worse check or the approval boundary.
MATERIALITY_CLEARED_STAGES: frozenset[DecisionStage] = frozenset(
    {DecisionStage.NEVER_WORSE, DecisionStage.APPROVAL}
)


class FunnelCounts(BaseModel):
    """The cumulative funnel over a whole store, plus its context counts.

    ``approved`` counts proposals a human decided in favor of (state
    ``approved`` or ``applied`` — an approved-but-unapplied proposal is the
    boundary's crash-window state and stays visible here); ``promoted``
    counts records currently ``promoted`` — what warm retrieval can actually
    see. A gap between the two is exactly the state the approval boundary
    designed to be visible rather than silent.
    """

    candidates: int
    replay_cleared: int
    materiality_cleared: int
    approved: int
    promoted: int
    gate_errors: int
    proposals_pending: int
    proposals_rejected: int


class PassFunnel(BaseModel):
    """One pass's funnel: manifest bounds plus per-`pass_id` decision counts.

    ``candidates`` is the manifest's own candidate list (what the pass
    reviewed); the stage counts come from the decision files stamped with
    this pass's id — the same source the cumulative funnel reads, so both
    views can be hand-checked against the same files. ``approved``/``promoted``
    attribute proposals to the pass that *cleared* them (the pass id stamped
    on the cleared trail the proposal embeds): a proposal rendered by an
    earlier pass and decided by a human later still belongs, proposal-quality
    -wise, to the pass whose cleared verdict justified it.
    """

    pass_id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    candidates: int
    decisions: int
    replay_cleared: int
    materiality_cleared: int
    approved: int
    promoted: int
    gate_errors: int


class ModelTokenSummary(BaseModel):
    """Token totals for one model, from per-call live evidence."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class LiveTokenSummary(BaseModel):
    """Aggregated per-call token evidence from recorded live runs.

    Failed calls carry ``None`` for model and tokens by the live runner's
    own contract — they are counted as calls that spent nothing billable,
    never fabricated into usage data. ``unreadable_records`` counts files
    that matched the ``call-*.json`` artifact naming but failed validation:
    shown, not hidden, and never silently dropped.
    """

    call_records: int
    ok: int
    failed: int
    input_tokens: int
    output_tokens: int
    by_model: dict[str, ModelTokenSummary] = Field(default_factory=dict)
    unreadable_records: int = 0


class FunnelReport(BaseModel):
    """The full proposal-quality report: cumulative, per-pass, live tokens."""

    funnel: FunnelCounts
    reasons_by_stage: dict[str, dict[str, int]]
    passes: list[PassFunnel]
    live_tokens: LiveTokenSummary | None = None
    decisions_without_manifest: int = 0
    """Decisions stamped with a pass id that has no manifest — the crash
    window between a pass's last decision and its manifest write. Zero in a
    healthy store; nonzero is shown, never absorbed."""


def reason_label(reason: str) -> str:
    """The vocabulary label of a template-checked reason: ``"<label>: <..."``."""
    return reason.partition(": ")[0]


def refusal_reasons(
    decisions: Sequence[PromotionDecision],
) -> dict[str, dict[str, int]]:
    """Decision counts per ``(stage, label)``, zero-filled over the vocabulary.

    Every stage and label of :data:`STAGE_REASON_LABELS` appears, even at
    zero — the report shows the whole machine-checked vocabulary and where
    decisions landed in it, so absence reads as zero, not as a missing row.
    A reason outside the vocabulary raises :class:`TelemetryError`: only
    records written through :func:`~evogine.gates.telemetry.check_reason`
    can be aggregated, and a hand-edited store must fail loudly rather than
    quietly reshape the report.
    """
    table: dict[str, dict[str, int]] = {
        stage.value: {label: 0 for label in labels}
        for stage, labels in STAGE_REASON_LABELS.items()
    }
    for decision in decisions:
        stage_row = table.get(decision.stage.value)
        label = reason_label(decision.reason)
        if stage_row is None or label not in stage_row:
            raise TelemetryError(
                f"decision {decision.proposal_id!r} carries reason "
                f"{decision.reason!r} outside the machine-checked vocabulary "
                f"for stage {decision.stage.value}"
            )
        stage_row[label] += 1
    return table


def funnel_counts(
    decisions: Sequence[PromotionDecision],
    proposals: Sequence[PromotionProposal],
    promoted_ids: set[str],
) -> FunnelCounts:
    """The cumulative funnel from decisions, the approval log, and records.

    ``promoted_ids`` is the set of record ids currently ``promoted`` (read
    from the record files by the caller) — the store's own truth about what
    warm retrieval can see, independent of the approval log's state.
    """
    return FunnelCounts(
        candidates=len(decisions),
        replay_cleared=sum(d.stage in REPLAY_CLEARED_STAGES for d in decisions),
        materiality_cleared=sum(d.stage in MATERIALITY_CLEARED_STAGES for d in decisions),
        approved=sum(
            p.state in (ApprovalState.APPROVED, ApprovalState.APPLIED) for p in proposals
        ),
        promoted=len(promoted_ids),
        gate_errors=sum(d.stage is DecisionStage.GATE_ERROR for d in decisions),
        proposals_pending=sum(p.state is ApprovalState.PENDING for p in proposals),
        proposals_rejected=sum(p.state is ApprovalState.REJECTED for p in proposals),
    )


def pass_funnel(
    manifest: PassManifest,
    decisions: Sequence[PromotionDecision],
    proposals: Sequence[PromotionProposal],
) -> PassFunnel:
    """One pass's funnel: manifest candidates plus this pass's decisions.

    ``decisions`` are the decision files stamped with the manifest's pass
    id. ``approved``/``promoted`` attribute proposals to the pass that
    *cleared* them — the pass id stamped on the trail the proposal embeds
    — and count the promotion event itself: state ``applied`` means this
    pass's cleared verdict was approved and promoted, however the record
    evolved afterwards (a later retirement is a newer fact than the pass,
    and the cumulative funnel's ``promoted`` row owns current status).
    """
    cleared_via_this_pass = [
        p for p in proposals if p.decision.pass_id == manifest.pass_id
    ]
    return PassFunnel(
        pass_id=manifest.pass_id,
        started_at=manifest.started_at,
        finished_at=manifest.finished_at,
        candidates=len(manifest.candidates),
        decisions=len(decisions),
        replay_cleared=sum(d.stage in REPLAY_CLEARED_STAGES for d in decisions),
        materiality_cleared=sum(d.stage in MATERIALITY_CLEARED_STAGES for d in decisions),
        approved=sum(
            p.state in (ApprovalState.APPROVED, ApprovalState.APPLIED)
            for p in cleared_via_this_pass
        ),
        promoted=sum(
            p.state is ApprovalState.APPLIED for p in cleared_via_this_pass
        ),
        gate_errors=sum(d.stage is DecisionStage.GATE_ERROR for d in decisions),
    )


def load_live_token_evidence(root: Path | str) -> LiveTokenSummary | None:
    """Aggregate per-call token evidence under ``<root>/runs``, or ``None``.

    The bench live runner (PR #15) persists one ``call-<case>-<arm>-<repeat>-
    <index>.json`` artifact per model call into its run directory. The scan
    is deliberately generic over the whole runs tree — run directories are
    the runner's own evidence surface, wherever a caller rooted them — and
    returns ``None`` when nothing has been recorded: an offline store has
    no live evidence, and the report says so rather than implying a spend.
    """
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return None
    call_paths = sorted(runs_dir.rglob("call-*.json"))
    if not call_paths:
        return None
    # Deferred import: bench depends on gates (the runner seam), so gates
    # must not import bench at module level — the live record model is
    # loaded only when a report actually reads live evidence.
    from evogine.bench.live import LiveCallRecord

    ok = 0
    failed = 0
    unreadable = 0
    by_model: dict[str, ModelTokenSummary] = {}
    input_tokens = 0
    output_tokens = 0
    for path in call_paths:
        try:
            record = LiveCallRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            unreadable += 1
            continue
        if not record.ok:
            failed += 1
            continue
        ok += 1
        input_tokens += record.input_tokens or 0
        output_tokens += record.output_tokens or 0
        model = record.model or "unknown"
        summary = by_model.setdefault(model, ModelTokenSummary())
        summary.calls += 1
        summary.input_tokens += record.input_tokens or 0
        summary.output_tokens += record.output_tokens or 0
    return LiveTokenSummary(
        call_records=ok + failed,
        ok=ok,
        failed=failed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        by_model=by_model,
        unreadable_records=unreadable,
    )


def _read_passes(root: Path) -> list[PassManifest]:
    passes_dir = root / "passes"
    if not passes_dir.is_dir():
        return []
    return [
        PassManifest.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(passes_dir.glob("*.json"))
    ]


def _read_promoted_ids(root: Path) -> set[str]:
    records_dir = root / "records"
    if not records_dir.is_dir():
        return set()
    return {
        record.id
        for path in sorted(records_dir.glob("*.json"))
        if (record := LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))).status
        == "promoted"
    }


def build_funnel_report(root: Path | str) -> FunnelReport:
    """Read a store's existing telemetry and compute the full funnel report.

    Store-owned surfaces (decisions, manifests, approvals, records) fail
    closed: a malformed file is a store-integrity defect and the error
    propagates — the report never silently narrows to the readable subset.
    Only the foreign live-evidence artifacts degrade individually (counted
    as ``unreadable_records``), because they are a run directory's cargo,
    not the store's audited invariants.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise TelemetryError(f"store root does not exist: {root_path}")
    decisions = DecisionLog(root_path).all()
    proposals = ApprovalLog(root_path).all()
    promoted_ids = _read_promoted_ids(root_path)
    manifests = _read_passes(root_path)
    manifest_ids = {manifest.pass_id for manifest in manifests}
    by_pass: dict[str, list[PromotionDecision]] = {}
    for decision in decisions:
        if decision.pass_id is not None:
            by_pass.setdefault(decision.pass_id, []).append(decision)
    return FunnelReport(
        funnel=funnel_counts(decisions, proposals, promoted_ids),
        reasons_by_stage=refusal_reasons(decisions),
        passes=[
            pass_funnel(manifest, by_pass.get(manifest.pass_id, []), proposals)
            for manifest in manifests
        ],
        live_tokens=load_live_token_evidence(root_path),
        decisions_without_manifest=sum(
            decision.pass_id is not None and decision.pass_id not in manifest_ids
            for decision in decisions
        ),
    )


__all__ = [
    "MATERIALITY_CLEARED_STAGES",
    "REPLAY_CLEARED_STAGES",
    "FunnelCounts",
    "FunnelReport",
    "LiveTokenSummary",
    "ModelTokenSummary",
    "PassFunnel",
    "build_funnel_report",
    "funnel_counts",
    "load_live_token_evidence",
    "pass_funnel",
    "reason_label",
    "refusal_reasons",
]
