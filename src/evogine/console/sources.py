"""The stores-and-sync selector: one console, many store contexts.

The federation is per-project stores; the sync daemon hosts them and serves
a read/query API; the sync client
(:mod:`evogine.sync.client`) is the one reader of that API. This module puts
a selector on the dashboard: which store context this console's read-only
views target — the local store, or any sync remote the launch config
declares. The selector changes what pages *read* and nothing else: every
write route (approval decisions, the guarded import) stays bound to the
local store's own guarded paths, and no remote write exists to reach.

Per-source health is the page's data: reachability, the query API's
overview counts, the remote's own spend report, and the audit position.
Honesty rules inherited from every other console page:

- an unreachable or misconfigured remote renders as its refusal — the
  row names the remote and why, never a fabricated empty result (the sync
  client's fail-closed vocabulary, passed through verbatim);
- the local row reads the store's own files through the same builders the
  overview and audit pages use — never a second accounting;
- a remote's audit trail is *not* served by the query API, and the panel
  says so instead of leaving the cell blank: per-store trails are never
  stitched into one sequence (governance §3.3), so remote audit position
  is a structural absence, not a missing feature.

Only what the sync client already exposes is consumed here
(:func:`fleet_status`, :func:`fetch_overview`, :func:`fetch_spend`) — no
second HTTP reader to drift from the client's refusal discipline.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from evogine.config import SyncEndpoint
from evogine.console.launch import ConsoleError
from evogine.console.views import build_overview, build_timeline
from evogine.engine.store import Engine, StoreError
from evogine.sync.client import (
    RemoteOverview,
    RemoteSpend,
    RemoteStatus,
    ResolvedRemote,
    SyncClientError,
    fetch_overview,
    fetch_spend,
    fleet_status,
    resolve_endpoint,
)

#: The page's render closure (app.py builds it over the Jinja2 environment);
#: typed loosely because the route only forwards template payloads.
RenderFn = Callable[..., Response]

LOCAL_SOURCE_NAME = "local"
"""The selector's name for this console's own store."""

REMOTE_AUDIT_NOTE = (
    "not served by the remote's query API — audit trails stay per-store and "
    "are never stitched into one sequence (governance §3.3)"
)


class SourceLink(BaseModel):
    """One selector entry: where it leads and whether it is the current context."""

    label: str
    href: str
    selected: bool


class SourceRow(BaseModel):
    """One store context's health, as the sources page renders it.

    ``None`` fields are rendered as their explicit absence (a remote that
    serves no overview, a store with no audit events yet) — the row never
    dresses an absence up as a zero or a blank.
    """

    name: str
    kind: Literal["local", "remote"]
    store: str
    """The store identity this context names — the engine's own identity for
    the local row, the daemon's hosted store name for a remote."""
    url: str | None = None
    reachable: bool = True
    """The local row is reachable by construction (the files are on disk);
    a remote row is reachable only when its query API answered."""
    total_decisions: int | None = None
    promoted: int | None = None
    pending_approvals: int | None = None
    recorded_runs: int | None = None
    actual_cost_usd: Decimal | None = None
    audit_seq: int | None = None
    audit_ts: str | None = None
    audit_note: str | None = None
    """Why the audit position is absent, when it is."""
    note: str | None = None
    """A remote's refusal — why it is not reachable — verbatim from the client."""


class SourcesView(BaseModel):
    """The stores-and-sync page: every configured source's health."""

    sources: list[SourceRow] = Field(default_factory=list)
    selected: str
    """Which source the selector currently points at."""


class RemoteOverviewView(BaseModel):
    """A remote source's read view, served through the sync client's query API."""

    name: str
    store: str
    url: str
    overview: RemoteOverview
    spend: RemoteSpend


def build_local_row(store_root: Path | str) -> SourceRow:
    """The local store's health row, through the console's own read builders.

    The same overview engine as the overview page and the same audit-timeline
    reader as the audit page — the selector's local row cannot disagree with
    the pages it selects for. An unreadable audit trail is the row's note,
    not a crash: one corrupt file must not hide the store's whole health.
    """
    store_root_path = Path(store_root)
    try:
        engine = Engine(store_root_path)
        counts = build_overview(store_root_path)
    except (StoreError, ValueError) as exc:
        raise ConsoleError(str(exc)) from exc
    audit_seq: int | None = None
    audit_ts: str | None = None
    audit_note: str | None = None
    try:
        timeline = build_timeline(store_root_path)
    except ConsoleError as exc:
        audit_note = f"audit trail unreadable: {exc}"
    else:
        if timeline:
            audit_seq = timeline[0].seq
            audit_ts = timeline[0].ts
        else:
            audit_note = "no audit events yet — nothing has transitioned on this store"
    return SourceRow(
        name=LOCAL_SOURCE_NAME,
        kind="local",
        store=engine.store_name,
        total_decisions=counts.total_decisions,
        promoted=counts.promoted,
        pending_approvals=counts.pending_approvals,
        audit_seq=audit_seq,
        audit_ts=audit_ts,
        audit_note=audit_note,
    )


def build_remote_row(status: RemoteStatus) -> SourceRow:
    """One remote's health row from the client's own status result.

    The client already classified the outcome (answered, or refused with
    the reason) — this only reshapes it for the page, keeping the refusal
    text verbatim so the console and ``evogine sync status`` say the same
    thing about the same remote.
    """
    if not status.ok or status.remote is None or status.overview is None or status.spend is None:
        return SourceRow(
            name=status.label,
            kind="remote",
            store=status.remote.store if status.remote is not None else "—",
            url=status.remote.base_url if status.remote is not None else None,
            reachable=False,
            note=status.refusal,
        )
    return SourceRow(
        name=status.label,
        kind="remote",
        store=status.remote.store,
        url=status.remote.base_url,
        total_decisions=status.overview.total_decisions,
        promoted=status.overview.promoted,
        pending_approvals=status.overview.pending_approvals,
        recorded_runs=status.spend.recorded_runs,
        actual_cost_usd=status.spend.actual_cost_usd,
        audit_note=REMOTE_AUDIT_NOTE,
    )


def build_sources_view(
    store_root: Path | str, endpoints: Sequence[SyncEndpoint], *, selected: str
) -> SourcesView:
    """The page's view model: the local row plus every configured remote.

    Remotes are statused through :func:`fleet_status`, which never aborts
    the scan on one refused machine — each remote's state (or refusal) is
    its own row, so the page shows every source's state and hides none.
    """
    return SourcesView(
        sources=[
            build_local_row(store_root),
            *(build_remote_row(status) for status in fleet_status(endpoints)),
        ],
        selected=selected,
    )


def resolve_source(source: str | None, endpoints: Sequence[SyncEndpoint]) -> ResolvedRemote | None:
    """The remote a ``source`` selector value names; ``None`` means local.

    Anything absent, blank, or literally ``local`` selects the local store.
    Any other value must name a configured endpoint — an unknown selector
    value is a mistake the page refuses by name (with what *is*
    configured), never a silent fall-back to local reads.
    """
    if source is None or not source.strip() or source == LOCAL_SOURCE_NAME:
        return None
    for endpoint in endpoints:
        if endpoint.name == source:
            try:
                return resolve_endpoint(endpoint)
            except SyncClientError as exc:
                raise ConsoleError(str(exc)) from exc
    if not endpoints:
        raise ConsoleError(
            f"unknown store context {source!r} — no sync remotes are configured; "
            "the only source is the local store"
        )
    configured = ", ".join(endpoint.name for endpoint in endpoints)
    raise ConsoleError(
        f"unknown store context {source!r} (configured sources: local, {configured})"
    )


def build_remote_overview(remote: ResolvedRemote) -> RemoteOverviewView:
    """A remote source's overview and spend, through the client's query API.

    Any client refusal (unreachable, schema mismatch) propagates as
    :class:`SyncClientError` — the caller renders it as the page's refusal
    state; a query result is never fabricated.
    """
    return RemoteOverviewView(
        name=remote.name,
        store=remote.store,
        url=remote.base_url,
        overview=fetch_overview(remote),
        spend=fetch_spend(remote),
    )


def sources_bar(selected: str, endpoints: Sequence[SyncEndpoint], *, base: str) -> list[SourceLink]:
    """The selector strip: one link per source, the current one marked.

    ``base`` is the page the strip is rendered on (``/`` or ``/sources``) —
    the selector switches that page's own context, so links stay on the
    page the operator is reading.
    """
    links = [
        SourceLink(
            label=LOCAL_SOURCE_NAME,
            href=f"{base}?source={LOCAL_SOURCE_NAME}",
            selected=selected == LOCAL_SOURCE_NAME,
        )
    ]
    links.extend(
        SourceLink(
            label=endpoint.name,
            href=f"{base}?source={urllib.parse.quote(endpoint.name)}",
            selected=selected == endpoint.name,
        )
        for endpoint in endpoints
    )
    return links


def sources_routes(
    store_root: Path, endpoints: Sequence[SyncEndpoint], render: RenderFn
) -> APIRouter:
    """The sources page's route, bound to one store root and render closure."""
    router = APIRouter()

    @router.get("/sources")
    def sources(request: Request) -> Response:
        """Per-source health; the selector is the page's own UI."""
        selected = request.query_params.get("source") or LOCAL_SOURCE_NAME
        try:
            view = build_sources_view(store_root, endpoints, selected=selected)
        except (ConsoleError, ValueError) as exc:
            return render(request, "error.html", {"status": 500, "message": str(exc)}, 500)
        return render(
            request,
            "sources.html",
            {
                "view": view,
                "sources_bar": sources_bar(selected, endpoints, base="/sources"),
                "remotes_configured": bool(endpoints),
            },
        )

    return router


__all__ = [
    "LOCAL_SOURCE_NAME",
    "REMOTE_AUDIT_NOTE",
    "RemoteOverviewView",
    "SourceLink",
    "SourceRow",
    "SourcesView",
    "build_local_row",
    "build_remote_overview",
    "build_remote_row",
    "build_sources_view",
    "resolve_source",
    "sources_bar",
    "sources_routes",
]
