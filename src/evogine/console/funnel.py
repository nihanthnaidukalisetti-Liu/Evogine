"""The funnel panel: read-only console page over the store's funnel engine.

``evogine decisions funnel`` answers V3's proposal-quality question from the
terminal; this module renders the same answer in the console. Parity is by
construction, not by re-implementation: the panel calls
:func:`evogine.gates.funnel.build_funnel_report` — the exact function the CLI
calls — and only reshapes the returned :class:`~evogine.gates.funnel.FunnelReport`
into template rows. There is no second funnel here to drift out of sync.

Like every console page, the panel is honest about absence: an empty store
renders zero counts (the CLI zero-fills the same way), no pass manifests read
as "no passes recorded", and missing live-run token evidence reads as "none
found" — never as a fabricated zero-spend story. Store corruption refuses
visibly, with the offending file named.

The route lives in its own module (not inline in :mod:`evogine.console.app`)
so the panel is one reviewable unit: an :class:`~fastapi.APIRouter` factory
that takes the page's ``render`` closure, keeping transport conventions in
one place while the view builders here stay pure and testable without HTTP.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel
from starlette.responses import Response

from evogine.console.launch import ConsoleError
from evogine.gates.funnel import (
    FunnelCounts,
    LiveTokenSummary,
    build_funnel_report,
)
from evogine.gates.telemetry import TelemetryError

#: The page's render closure (app.py builds it over the Jinja2 environment);
#: typed loosely because the route only forwards template payloads.
RenderFn = Callable[..., Response]


class FunnelStageRow(BaseModel):
    """One row of the cumulative stage table: a count and what it means."""

    stage: str
    count: int
    meaning: str


class FunnelReasonRow(BaseModel):
    """One ``(stage, label)`` cell of the zero-filled refusal breakdown."""

    stage: str
    label: str
    count: int


class FunnelPassRow(BaseModel):
    """One pass's funnel, timestamps rendered for the table cells."""

    pass_id: str
    started_at: str | None
    finished_at: str | None
    candidates: int
    decisions: int
    replay_cleared: int
    materiality_cleared: int
    approved: int
    promoted: int
    gate_errors: int


class FunnelView(BaseModel):
    """Everything the funnel page shows, from one :class:`FunnelReport`."""

    counts: FunnelCounts
    stage_rows: list[FunnelStageRow]
    reasons: list[FunnelReasonRow]
    passes: list[FunnelPassRow]
    decisions_without_manifest: int
    live_tokens: LiveTokenSummary | None


def _fmt(moment: datetime | None) -> str | None:
    """Render a timestamp for a table cell (second precision, offset kept)."""
    if moment is None:
        return None
    return moment.isoformat(timespec="seconds")


def build_funnel_view(root: Path | str) -> FunnelView:
    """The funnel page's view model, computed through the funnel engine.

    Store defects surface as :class:`ConsoleError` — the console's own
    refusal vocabulary — so the route's error path needs no funnel-specific
    knowledge. The cumulative rows below access one report field each: the
    display strings mirror the CLI table's wording, the numbers are the
    engine's, and no arithmetic happens at this layer.
    """
    try:
        report = build_funnel_report(root)
    except TelemetryError as exc:
        raise ConsoleError(str(exc)) from exc
    counts = report.funnel
    return FunnelView(
        counts=counts,
        stage_rows=[
            FunnelStageRow(
                stage="candidates",
                count=counts.candidates,
                meaning="decided proposals (latest verdict each)",
            ),
            FunnelStageRow(
                stage="replay-cleared",
                count=counts.replay_cleared,
                meaning="replay flipped a verdict",
            ),
            FunnelStageRow(
                stage="materiality-cleared",
                count=counts.materiality_cleared,
                meaning="flip met the materiality threshold",
            ),
            FunnelStageRow(
                stage="approved",
                count=counts.approved,
                meaning="human approved (or applied) the proposal",
            ),
            FunnelStageRow(
                stage="promoted",
                count=counts.promoted,
                meaning="live in the promoted pool",
            ),
            FunnelStageRow(
                stage="gate errors",
                count=counts.gate_errors,
                meaning="infrastructure — undecided, retryable",
            ),
            FunnelStageRow(
                stage="proposals pending",
                count=counts.proposals_pending,
                meaning="awaiting a human decision",
            ),
            FunnelStageRow(
                stage="proposals rejected",
                count=counts.proposals_rejected,
                meaning="refused at the approval boundary",
            ),
        ],
        reasons=[
            FunnelReasonRow(stage=stage, label=label, count=count)
            for stage, labels in report.reasons_by_stage.items()
            for label, count in labels.items()
        ],
        passes=[
            FunnelPassRow(
                pass_id=entry.pass_id,
                started_at=_fmt(entry.started_at),
                finished_at=_fmt(entry.finished_at),
                candidates=entry.candidates,
                decisions=entry.decisions,
                replay_cleared=entry.replay_cleared,
                materiality_cleared=entry.materiality_cleared,
                approved=entry.approved,
                promoted=entry.promoted,
                gate_errors=entry.gate_errors,
            )
            for entry in report.passes
        ],
        decisions_without_manifest=report.decisions_without_manifest,
        live_tokens=report.live_tokens,
    )


def funnel_routes(store_root: Path, render: RenderFn) -> APIRouter:
    """The funnel panel's routes, bound to one store root and render closure."""
    router = APIRouter()

    @router.get("/funnel")
    def funnel(request: Request) -> Response:
        """The proposal funnel; store corruption refuses visibly, never 500s blind."""
        try:
            view = build_funnel_view(store_root)
        except (ConsoleError, ValueError) as exc:
            return render(request, "error.html", {"status": 500, "message": str(exc)}, 500)
        return render(request, "funnel.html", {"view": view})

    return router


__all__ = [
    "FunnelPassRow",
    "FunnelReasonRow",
    "FunnelStageRow",
    "FunnelView",
    "build_funnel_view",
    "funnel_routes",
]
