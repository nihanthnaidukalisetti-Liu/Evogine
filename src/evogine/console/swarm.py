"""The swarm board and the learning loop: read-only pages over existing data.

The board reads the orchestrator's typed JSONL event feed — one
``events.jsonl`` per run directory, whose schema
(:mod:`evogine.swarm.orchestrator.events`) is the contract this module
consumes — through that module's own public read helpers
(:func:`~evogine.swarm.orchestrator.read_events`, :class:`RunDirectory`,
:class:`~evogine.swarm.orchestrator.CheckpointStore`). There is no second
parser here to drift from the writer's schema, and nothing in this module
writes: the pages are views over runs, decisions, approvals, and records
that already exist.

Honesty is the design constraint. A run that refused (spend ceiling,
missing key) renders its refusal as a first-class row state, never as an
error to hide; a run directory with no feed yet renders an explicit "no
events yet" state; a *corrupt* feed renders a visible ``corrupt`` row state
naming the run and the defect (the audit-timeline precedent: an audit
surface that skips lines is lying by omission) — while the rest of the
board keeps rendering, because one poisoned run must not hide the others.
Money stays exact ``Decimal`` from the feed to the HTML; the only
quantization is the labeled display percentage, computed in Decimal.

The routes live in this module (the funnel panel's precedent) so the two
pages are one reviewable unit: an :class:`~fastapi.APIRouter` factory that
takes the pages' ``render`` closure, keeping transport conventions in one
place while the view builders stay pure and testable without HTTP.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.responses import Response

from evogine.bench.rundirs import RunDirectory
from evogine.console.funnel import FunnelView, build_funnel_view
from evogine.console.launch import ConsoleError
from evogine.console.views import build_timeline
from evogine.engine.store import Engine, StoreError
from evogine.gates.approvals import ApprovalLog
from evogine.gates.telemetry import DecisionLog, TelemetryError
from evogine.swarm.orchestrator import (
    CheckpointError,
    CheckpointStore,
    EventLogError,
    RunEvent,
    read_events,
)
from evogine.swarm.orchestrator.events import (
    EVENTS_FILENAME,
    CompleteEvent,
    RefusalEvent,
    SpawnEvent,
    SpendEvent,
    StatusEvent,
)

#: The page's render closure (app.py builds it over the Jinja2 environment);
#: typed loosely because the route only forwards template payloads.
RenderFn = Callable[..., Response]

SWARM_RUNS_DIRNAME = "runs/swarm"
"""The swarm runs directory under the store root — the same default the
config's ``swarm.runs_dir`` declares, resolved against the same root, so
the writer and this board agree on one location by construction."""


def resolve_swarm_runs_root(
    store_root: Path | str, override: Path | str | None = None
) -> Path:
    """The runs root the board reads: the explicit override, or the derived
    default under the store root."""
    if override is not None:
        return Path(override)
    return Path(store_root) / SWARM_RUNS_DIRNAME


def discover_runs(runs_root: Path | str) -> list[RunDirectory]:
    """Run-directory handles under ``runs_root``, in run-id order.

    ``RunDirectory``'s constructor applies the same run-id validation
    ``open_run`` does, so a directory whose name is not a run id (a stray
    ``.git``, a scratch directory) is not a run and is skipped — never
    rendered as an empty row. Handles are constructed directly, not via
    :func:`open_run`, because reopening a completed run is refused to
    writers and the board only reads.
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    runs: list[RunDirectory] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            runs.append(RunDirectory(root, entry.name))
        except ValueError:
            continue  # not a run-id name — not a run, not an empty state to show
    return runs


class WorkerBoardRow(BaseModel):
    """One spawned worker's observable state, from the feed alone."""

    worker_id: int
    pool_size: int | None = None
    """From the worker's spawn event; ``None`` when no spawn was recorded."""
    calls: int = 0
    spent_usd: Decimal = Decimal(0)
    """The worker's own per-call cost sum — exact Decimal."""
    last_item_id: str | None = None
    last_status: str | None = None
    last_detail: str = ""


class RefusalRow(BaseModel):
    """A run-level refusal, rendered as a state — data, never an error."""

    reason: str
    detail: str
    spent_usd: Decimal
    projected_usd: Decimal | None = None
    ceiling_usd: Decimal


class RunBoardRow(BaseModel):
    """One orchestrator run as the board renders it."""

    run_id: str
    complete: bool
    """The run's ``complete.marker`` — the freeze, not the feed's opinion."""
    feed_state: Literal["ok", "empty", "corrupt"]
    feed_note: str | None = None
    """Why the feed is empty or corrupt, when it is."""
    started_at: str | None = None
    last_event_at: str | None = None
    events: int = 0
    workers: list[WorkerBoardRow] = Field(default_factory=list)
    item_counts: dict[str, int] = Field(default_factory=dict)
    """Current status per item (last transition wins), nonzero counts only."""
    items_total: int | None = None
    items_completed: int | None = None
    resumed_items: int | None = None
    """From the complete event when present, else the skipped-item count."""
    checkpoint_markers: int = 0
    spent_usd: Decimal | None = None
    ceiling_usd: Decimal | None = None
    spend_percent_display: str | None = None
    """Labeled display quantization (Decimal math); exact money sits beside it."""
    spend_percent_clamped: int | None = None
    refusal: RefusalRow | None = None


class SwarmBoardView(BaseModel):
    """Everything the swarm board page shows."""

    runs_root: str
    runs_present: bool
    """Whether the runs root exists at all — absence is an explicit empty
    state (nothing has run yet), never a fabricated board."""
    runs: list[RunBoardRow] = Field(default_factory=list)


class StoreHealthRow(BaseModel):
    """One store-health signal: a count and what it means."""

    label: str
    value: int
    meaning: str


class PromotionTelemetryRow(BaseModel):
    """One promotion decision with its pass identity and latency."""

    proposal_id: str
    learning_id: str
    decided_at: str
    promoted: bool
    stage: str
    reason: str
    pass_id: str | None = None
    review_latency_seconds: float | None = None


class LearningLoopView(BaseModel):
    """The learning-loop page: funnel stages, store health, promotion telemetry."""

    funnel: FunnelView
    """The funnel panel's own view — the same engine, never a second report."""
    health: list[StoreHealthRow] = Field(default_factory=list)
    promotions: list[PromotionTelemetryRow] = Field(default_factory=list)


def _fmt(moment: datetime | None) -> str | None:
    """Render a timestamp for a table cell (second precision, offset kept)."""
    if moment is None:
        return None
    return moment.isoformat(timespec="seconds")


def _spend_display(spent: Decimal, ceiling: Decimal) -> tuple[str, int]:
    """The spend fraction for display: a labeled percentage and a bar width.

    Decimal in, Decimal out until the last step: the percentage is quantized
    to one decimal place for the cell text, and clamped to ``0..100`` for
    the progress bar's width — the exact money values are rendered beside
    it, so the display rounding can never be the whole story.
    """
    fraction = (spent / ceiling * Decimal(100)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    clamped = max(0, min(100, int(fraction)))
    return f"{fraction}%", clamped


def _worker_rows(events: list[RunEvent]) -> list[WorkerBoardRow]:
    """Per-worker state aggregated from the feed, in worker-id order."""
    pool_size: dict[int, int] = {}
    calls: dict[int, int] = {}
    spent: dict[int, Decimal] = {}
    last_item: dict[int, str] = {}
    last_status: dict[int, str] = {}
    last_detail: dict[int, str] = {}
    for event in events:
        if isinstance(event, SpawnEvent):
            pool_size[event.worker_id] = event.pool_size
        elif isinstance(event, StatusEvent):
            if event.worker_id is None:
                continue  # run-level statement — no worker row to update
            last_item[event.worker_id] = event.item_id
            last_status[event.worker_id] = event.status
            last_detail[event.worker_id] = event.detail
        elif isinstance(event, SpendEvent):
            calls[event.worker_id] = calls.get(event.worker_id, 0) + 1
            spent[event.worker_id] = (
                spent.get(event.worker_id, Decimal(0)) + event.cost_usd
            )
    worker_ids = sorted(set(pool_size) | set(calls) | set(last_item))
    return [
        WorkerBoardRow(
            worker_id=worker_id,
            pool_size=pool_size.get(worker_id),
            calls=calls.get(worker_id, 0),
            spent_usd=spent.get(worker_id, Decimal(0)),
            last_item_id=last_item.get(worker_id),
            last_status=last_status.get(worker_id),
            last_detail=last_detail.get(worker_id, ""),
        )
        for worker_id in worker_ids
    ]


def _item_counts(events: list[RunEvent]) -> dict[str, int]:
    """Current status per item (last transition wins), nonzero counts only."""
    last: dict[str, str] = {}
    for event in events:
        if isinstance(event, StatusEvent):
            last[event.item_id] = event.status
    counts: dict[str, int] = {}
    for status in last.values():
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_run_row(run: RunDirectory) -> RunBoardRow:
    """One run's board row from its event feed and checkpoint markers.

    A corrupt feed or an ambiguous marker set is real corruption: the row
    renders the ``corrupt`` feed state with the defect named — visible,
    never skipped, never a fabricated board — while the rest of the board
    keeps rendering, because one poisoned run must not hide the others.
    """
    try:
        events = read_events(run)
        markers = CheckpointStore(run).completed()
    except (EventLogError, CheckpointError, ValueError) as exc:
        return RunBoardRow(
            run_id=run.run_id,
            complete=run.is_complete(),
            feed_state="corrupt",
            feed_note=str(exc),
        )

    started_at = events[0].ts if events else None
    last_event_at = events[-1].ts if events else None
    spent_usd: Decimal | None = None
    ceiling_usd: Decimal | None = None
    refusal: RefusalRow | None = None
    items_total: int | None = None
    items_completed: int | None = None
    resumed_items: int | None = None
    for event in events:
        if isinstance(event, SpendEvent | CompleteEvent | RefusalEvent):
            # The latest money-bearing event carries the ledger state.
            spent_usd = event.spent_usd
            ceiling_usd = event.ceiling_usd
        if isinstance(event, CompleteEvent):
            items_total = event.items_total
            items_completed = event.items_completed
            resumed_items = event.resumed_items
        if isinstance(event, RefusalEvent):
            refusal = RefusalRow(
                reason=event.reason,
                detail=event.detail,
                spent_usd=event.spent_usd,
                projected_usd=event.projected_usd,
                ceiling_usd=event.ceiling_usd,
            )
    counts = _item_counts(events)
    if resumed_items is None and counts.get("skipped"):
        resumed_items = counts["skipped"]
    spend_percent_display: str | None = None
    spend_percent_clamped: int | None = None
    if spent_usd is not None and ceiling_usd is not None:
        spend_percent_display, spend_percent_clamped = _spend_display(
            spent_usd, ceiling_usd
        )
    feed_state: Literal["ok", "empty"] = "ok" if events else "empty"
    return RunBoardRow(
        run_id=run.run_id,
        complete=run.is_complete(),
        feed_state=feed_state,
        feed_note=None if events else (
            "no events written yet — the run directory exists but its "
            f"{EVENTS_FILENAME} is absent or has no lines"
        ),
        started_at=_fmt(started_at),
        last_event_at=_fmt(last_event_at),
        events=len(events),
        workers=_worker_rows(events),
        item_counts=counts,
        items_total=items_total,
        items_completed=items_completed,
        resumed_items=resumed_items,
        checkpoint_markers=len(markers),
        spent_usd=spent_usd,
        ceiling_usd=ceiling_usd,
        spend_percent_display=spend_percent_display,
        spend_percent_clamped=spend_percent_clamped,
        refusal=refusal,
    )


def build_swarm_board(
    store_root: Path | str, swarm_runs_root: Path | str | None = None
) -> SwarmBoardView:
    """The swarm board's view model: every discovered run, in run-id order.

    Run discovery reads the runs root (the config's ``swarm.runs_dir``
    location — derived from the store root unless overridden); each run's
    state comes from its event feed through the orchestrator's public
    readers, and from nothing else about the run's interior.
    """
    runs_root = resolve_swarm_runs_root(store_root, swarm_runs_root)
    runs = discover_runs(runs_root)
    return SwarmBoardView(
        runs_root=str(runs_root),
        runs_present=runs_root.is_dir(),
        runs=[build_run_row(run) for run in runs],
    )


def build_learning_loop(root: Path | str) -> LearningLoopView:
    """The learning-loop page's view model, through the store's own engines.

    The funnel numbers come from the funnel panel's builder (the CLI's own
    funnel engine); store health and promotion telemetry come from the same
    read APIs the CLI commands use. Store defects surface as
    :class:`ConsoleError` — the refusal vocabulary every page shares.
    """
    store_root = Path(root)
    try:
        funnel = build_funnel_view(store_root)
    except TelemetryError as exc:
        raise ConsoleError(str(exc)) from exc
    engine = Engine(store_root)
    try:
        promoted = engine.promoted_records()
        candidates = engine.candidates()
        duplicates = engine.find_duplicates()
    except StoreError as exc:
        raise ConsoleError(str(exc)) from exc
    pending = ApprovalLog(store_root).pending()
    audit_rows = build_timeline(store_root)
    decisions = sorted(
        DecisionLog(store_root).all(),
        key=lambda decision: decision.decided_at,
        reverse=True,
    )
    return LearningLoopView(
        funnel=funnel,
        health=[
            StoreHealthRow(
                label="promoted records",
                value=len(promoted),
                meaning="live in the promoted pool — the only pool retrieval draws from",
            ),
            StoreHealthRow(
                label="candidate records",
                value=len(candidates),
                meaning="queued for replay — the memory gate reviews exactly these",
            ),
            StoreHealthRow(
                label="duplicate pairs",
                value=len(duplicates),
                meaning="unmerged duplicate record pairs found by the store's own scan",
            ),
            StoreHealthRow(
                label="audit events",
                value=len(audit_rows),
                meaning="append-only audit entries on the store's trail",
            ),
            StoreHealthRow(
                label="pending approvals",
                value=len(pending),
                meaning="proposals awaiting a human decision",
            ),
        ],
        promotions=[
            PromotionTelemetryRow(
                proposal_id=decision.proposal_id,
                learning_id=decision.learning_id,
                decided_at=_fmt(decision.decided_at) or "",
                promoted=decision.promoted,
                stage=decision.stage.value,
                reason=decision.reason,
                pass_id=decision.pass_id,
                review_latency_seconds=decision.review_latency_seconds,
            )
            for decision in decisions
        ],
    )


def swarm_routes(
    store_root: Path, swarm_runs_root: Path, render: RenderFn
) -> APIRouter:
    """Both pages' routes, bound to one store root, runs root, and render closure."""
    router = APIRouter()

    @router.get("/swarm")
    def swarm(request: Request) -> Response:
        """The swarm board; feed corruption refuses visibly, never 500s blind."""
        try:
            view = build_swarm_board(store_root, swarm_runs_root)
        except (ConsoleError, ValueError) as exc:
            return render(request, "error.html", {"status": 500, "message": str(exc)}, 500)
        return render(request, "swarm.html", {"view": view})

    @router.get("/loop")
    def loop(request: Request) -> Response:
        """The learning loop; store corruption refuses visibly, never 500s blind."""
        try:
            view = build_learning_loop(store_root)
        except (ConsoleError, ValueError) as exc:
            return render(request, "error.html", {"status": 500, "message": str(exc)}, 500)
        return render(request, "loop.html", {"view": view})

    return router


__all__ = [
    "SWARM_RUNS_DIRNAME",
    "LearningLoopView",
    "PromotionTelemetryRow",
    "RefusalRow",
    "RunBoardRow",
    "StoreHealthRow",
    "SwarmBoardView",
    "WorkerBoardRow",
    "build_learning_loop",
    "build_run_row",
    "build_swarm_board",
    "discover_runs",
    "resolve_swarm_runs_root",
    "swarm_routes",
]
