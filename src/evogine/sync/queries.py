"""Pure read models over a hosted store — the query half of the daemon.

The brief's scope: serve reads over records, decisions, approvals, funnel
aggregates, run evidence, shadow outputs, and spend. Every reader here
reads the same typed files the console reads, with the same refusal
discipline: a malformed store file names the offending path and refuses
visibly instead of fabricating an empty result.

Reuse over reimplementation: decisions, approvals, the overview, and the
funnel view come from the console's existing view builders
(``evogine.console.views``, ``evogine.console.funnel``) — one funnel
definition, so the daemon cannot drift from the console's. The console
view builders construct the engine's log objects, whose constructors
create their directories on the read path; that mkdir leaves an *empty*
directory behind, and empty directories are invisible to git (never
tracked, never in status, never blocking an ``updateInstead`` checkout),
so they cannot dirty a hosted store's sync state.

The readers added here — records, gate evidence, shadow outputs, spend —
are pure glob-and-parse over the store's layout, no engine construction at
all: they read existing evidence, they never generate it.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ValidationError

from evogine.bench.budgets import SpendReconciliation
from evogine.bench.shadow import ShadowReport
from evogine.console.views import (
    ApprovalRow,
    DecisionRow,
    build_overview,
    list_approvals,
    list_decisions,
)
from evogine.engine.records import LearningRecord
from evogine.engine.store import StoreError, split_qualified_id
from evogine.gates.materiality import MaterialityVerdict, Verdict
from evogine.sync.settings import SyncError

__all__ = [
    "GateEvidence",
    "ShadowOutput",
    "SpendRow",
    "SpendSummary",
    "read_approvals",
    "read_decisions",
    "read_evidence",
    "read_funnel",
    "read_overview",
    "read_record",
    "read_records",
    "read_shadow_outputs",
    "read_spend",
]

#: The record lifecycle's six statuses (engine.records). A status filter
#: that is not one of these is a typo the caller should see, not an
#: empty-looking result.
_RECORD_STATUSES = frozenset({"ephemeral", "candidate", "promoted", "dropped", "merged", "retired"})


class GateEvidence(BaseModel):
    """The persist-evidence payload for one materiality-gated promotion.

    Mirror of the telemetry sink's payload shape (the console's evidence
    table column), served as a query result for remote machines.
    """

    learning_id: str
    replay_run_ids: list[str]
    materiality: MaterialityVerdict
    baseline: list[Verdict]
    warm: list[Verdict]


class ShadowOutput(BaseModel):
    """One persisted shadow-execution report, keyed by its candidate."""

    candidate_id: str
    report: ShadowReport


class SpendRow(BaseModel):
    """One bench run's spend reconciliation."""

    run_id: str
    reconciliation: SpendReconciliation


class SpendSummary(BaseModel):
    """Spend across the hosted store's recorded bench runs.

    The aggregates are ``None`` when no run recorded spend — absence of
    evidence is reported as absence (``recorded_runs: 0``), never dressed
    up as a fabricated zero-spend story (the funnel's honesty rule).
    """

    recorded_runs: int
    rows: list[SpendRow]
    actual_cost_usd: Decimal | None = None
    overage_usd: Decimal | None = None
    within_ceiling: bool | None = None


def _parse_json(path: Path, kind: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError(f"malformed {kind} file {path}: {exc}", http_status=500) from exc
    if not isinstance(value, dict):
        raise SyncError(f"malformed {kind} file {path}: not a JSON object", http_status=500)
    return value


def read_overview(root: Path) -> object:
    """The console's overview aggregates for the hosted store."""
    return build_overview(root)


def read_records(root: Path, status: str | None = None) -> list[LearningRecord]:
    """All records in the store, sorted by id; optionally by lifecycle status.

    A malformed record file is a visible refusal naming the file — the
    store's durability contract makes every ``*.json`` under ``records/``
    a parseable record, so a bad one is corruption worth surfacing, not
    noise worth skipping.
    """
    if status is not None and status not in _RECORD_STATUSES:
        raise SyncError(
            f"unknown lifecycle status {status!r} (expected one of "
            f"{', '.join(sorted(_RECORD_STATUSES))})",
            http_status=400,
        )
    records_dir = root / "records"
    if not records_dir.is_dir():
        return []
    records: list[LearningRecord] = []
    for path in sorted(records_dir.glob("*.json")):
        payload = _parse_json(path, "record")
        try:
            record = LearningRecord.model_validate(payload)
        except ValidationError as exc:
            raise SyncError(
                f"record file {path.name} does not match the store schema: {exc}",
                http_status=500,
            ) from exc
        if status is None or record.status == status:
            records.append(record)
    return records


def read_record(root: Path, record_id: str, store_name: str) -> LearningRecord:
    """One record by its public id; a well-formed unknown id is a 404.

    The id is validated through the federation's public qualified-id
    helper — the record half of ``<store-name>:<record-id>`` must match
    the exact grammar the store enforces — so a malformed or
    traversal-shaped id is a visible 400 and can never touch a path.
    """
    try:
        _, record_id = split_qualified_id(f"{store_name}:{record_id}")
    except StoreError as exc:
        raise SyncError(str(exc), http_status=400) from exc
    path = root / "records" / f"{record_id}.json"
    if not path.is_file():
        raise SyncError(f"unknown record {record_id!r}", http_status=404)
    payload = _parse_json(path, "record")
    try:
        return LearningRecord.model_validate(payload)
    except ValidationError as exc:
        raise SyncError(
            f"record file {path.name} does not match the store schema: {exc}",
            http_status=500,
        ) from exc


def read_decisions(root: Path) -> list[DecisionRow]:
    """Promotion decisions (the store's decision log), newest last."""
    return list_decisions(root)


def read_approvals(root: Path) -> list[ApprovalRow]:
    """Approval queue and history (the store's approval log), pending first."""
    return list_approvals(root)


def read_funnel(root: Path) -> object:
    """The proposal funnel with stage rows — the console's own funnel view."""
    from evogine.console.funnel import build_funnel_view

    return build_funnel_view(root)


def read_evidence(root: Path) -> list[GateEvidence]:
    """Persisted gate evidence, one per materiality-gated promotion attempt."""
    gate_dir = root / "runs" / "gate"
    if not gate_dir.is_dir():
        return []
    evidence: list[GateEvidence] = []
    for path in sorted(gate_dir.glob("*.json")):
        payload = _parse_json(path, "evidence")
        try:
            evidence.append(GateEvidence.model_validate(payload))
        except ValidationError as exc:
            raise SyncError(
                f"evidence file {path.name} does not match the evidence schema: {exc}",
                http_status=500,
            ) from exc
    return evidence


def read_shadow_outputs(root: Path) -> list[ShadowOutput]:
    """Persisted shadow-execution reports, keyed by candidate."""
    shadow_dir = root / "runs" / "shadow"
    if not shadow_dir.is_dir():
        return []
    outputs: list[ShadowOutput] = []
    for report_path in sorted(shadow_dir.glob("*/artifacts/report.json")):
        payload = _parse_json(report_path, "shadow report")
        try:
            report = ShadowReport.model_validate(payload)
        except ValidationError as exc:
            raise SyncError(
                f"shadow report {report_path} does not match the shadow schema: {exc}",
                http_status=500,
            ) from exc
        outputs.append(ShadowOutput(candidate_id=report.candidate_id, report=report))
    return outputs


def read_spend(root: Path) -> SpendSummary:
    """Spend reconciliation across the store's recorded bench runs."""
    bench_dir = root / "runs" / "bench"
    rows: list[SpendRow] = []
    if bench_dir.is_dir():
        for report_path in sorted(bench_dir.glob("*/artifacts/spend-report.json")):
            payload = _parse_json(report_path, "spend report")
            try:
                reconciliation = SpendReconciliation.model_validate(payload)
            except ValidationError as exc:
                raise SyncError(
                    f"spend report {report_path} does not match the spend schema: {exc}",
                    http_status=500,
                ) from exc
            rows.append(SpendRow(run_id=report_path.parents[1].name, reconciliation=reconciliation))
    if not rows:
        return SpendSummary(recorded_runs=0, rows=[])
    return SpendSummary(
        recorded_runs=len(rows),
        rows=rows,
        actual_cost_usd=sum((row.reconciliation.actual_cost_usd for row in rows), Decimal(0)),
        overage_usd=sum((row.reconciliation.overage_usd for row in rows), Decimal(0)),
        within_ceiling=all(row.reconciliation.within_ceiling for row in rows),
    )
