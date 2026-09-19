"""The FastAPI application factory: routes are a thin transport over views.

Eight read-only pages — overview, decisions, one decision's drill-down, the
approvals queue, the audit timeline, the proposal funnel, the swarm board,
and the learning loop — render the typed views from
:mod:`evogine.console.views`, :mod:`evogine.console.funnel`, and
:mod:`evogine.console.swarm`. The write path
(:mod:`evogine.console.write`) registers the two POST routes that decide
one approval proposal each; every write goes through the approval boundary
functions under the store lock, and there is deliberately no auth stack:
the server binds loopback only (machine possession is the trust anchor),
so there is nothing exposed to authenticate.

Templates are server-rendered Jinja2 with autoescaping on — decision
reasons, trajectories, and record content are store data and are never
trusted as HTML — and no client framework is loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from evogine.console import views
from evogine.console.funnel import funnel_routes
from evogine.console.launch import ConsoleError, validate_operator
from evogine.console.swarm import resolve_swarm_runs_root, swarm_routes
from evogine.console.write import register_write_routes

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app(
    root: Path | str,
    operator: str,
    swarm_runs_root: Path | str | None = None,
) -> FastAPI:
    """Build the console for one store root and operator identity.

    The factory is the only construction path: every page is served for
    exactly one root and one operator, both shown in the page header so the
    operator always knows which store they are looking at. The swarm board
    reads run directories under ``swarm_runs_root`` when given, else the
    derived default under the store root (the config's ``swarm.runs_dir``
    location).
    """
    identity = validate_operator(operator)
    store_root = Path(root)
    runs_root = resolve_swarm_runs_root(store_root, swarm_runs_root)
    app = FastAPI(title="Evogine Console", docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(
        env=jinja2.Environment(
            loader=jinja2.FileSystemLoader(_TEMPLATES_DIR),
            # Store data (reasons, trajectories, record content) is never
            # trusted as HTML; autoescaping is the console's XSS posture.
            autoescape=True,
        )
    )

    def render(
        request: Request, name: str, extra: dict[str, Any], status_code: int = 200
    ) -> Response:
        response: Response = templates.TemplateResponse(
            request,
            name,
            {"operator": identity, "root": str(store_root), **extra},
            status_code=status_code,
        )
        return response

    def refusal(request: Request, status: int, message: str) -> Response:
        """A visible refusal state: the console explains itself, never 500s blind."""
        return render(request, "error.html", {"status": status, "message": message}, status)

    def page(
        request: Request, build: Callable[[], dict[str, Any]], name: str
    ) -> Response:
        """Render a list page, or refuse visibly when the store data is unreadable.

        A malformed telemetry file behind a LIST is store corruption — the
        page refuses with the file's name instead of a blind 500 or a
        silently half-empty table.
        """
        try:
            extra = build()
        except (ConsoleError, ValueError) as exc:
            return refusal(request, 500, str(exc))
        return render(request, name, extra)

    @app.exception_handler(StarletteHTTPException)
    def http_exception(request: Request, exc: StarletteHTTPException) -> Response:
        return refusal(request, exc.status_code, str(exc.detail))

    @app.get("/")
    def overview(request: Request) -> Response:
        return page(
            request, lambda: {"counts": views.build_overview(store_root)}, "overview.html"
        )

    @app.get("/decisions")
    def decisions(request: Request) -> Response:
        return page(
            request, lambda: {"rows": views.list_decisions(store_root)}, "decisions.html"
        )

    @app.get("/decisions/{proposal_id}")
    def decision_detail(request: Request, proposal_id: str) -> Response:
        try:
            detail = views.build_decision_detail(store_root, proposal_id)
        except ConsoleError as exc:
            return refusal(request, 404, str(exc))
        return render(request, "decision_detail.html", {"detail": detail})

    @app.get("/approvals")
    def approvals(request: Request) -> Response:
        return page(
            request, lambda: {"rows": views.list_approvals(store_root)}, "approvals.html"
        )

    @app.get("/audit")
    def audit(request: Request) -> Response:
        try:
            timeline = views.build_timeline(store_root)
        except ConsoleError as exc:
            return refusal(request, 500, str(exc))
        return render(request, "audit.html", {"rows": timeline, "empty": not timeline})

    app.include_router(funnel_routes(store_root, render))
    app.include_router(swarm_routes(store_root, runs_root, render))

    # The write path is its own module so the read-only surface stays a
    # separate, reviewable change; registration is the only seam.
    register_write_routes(app, store_root, identity, refuse=refusal)

    return app
