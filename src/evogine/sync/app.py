"""The FastAPI app: git transport routes, query routes, binding guards.

Route inventory, by trust surface:

- ``GET/POST /git/{store}/...`` — the git smart-HTTP protocol. The
  mutating half (``receive-pack``) is refused for non-loopback peers with
  a visible 403 **regardless of any flag**; the read half (``upload-pack``,
  clone/fetch) is subject only to the daemon's bind address.
- ``GET /health``, ``GET /api/stores[...]`` — read-only query surface over
  the hosted stores' typed files. Reads are gated by the bind address
  (loopback unless ``--expose-lan``), never by per-request checks: there
  is no auth stack by scope decision, so the bind is the read gate.

Every refusal is a JSON body naming the reason — visible refusal states,
not silent 500s. Store lifecycle writes run under the engine's existing
per-store advisory lock (``review.lock``), held across the receive-pack
subprocess, so a push and a local lifecycle transition can never
interleave within one store. Reads and fetches take no lock: git's own
atomicity covers them.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from evogine.engine.store import store_lock
from evogine.sync import queries
from evogine.sync.gitbackend import (
    CgiResponse,
    receive_pack_requested,
    run_git_http_backend,
)
from evogine.sync.settings import SyncError, SyncSettings, is_loopback_peer
from evogine.sync.stores import StoreRegistry


def _peer(request: Request) -> str | None:
    return request.client.host if request.client else None


def create_app(settings: SyncSettings) -> FastAPI:
    """Build the daemon app over one stores root. Binds no sockets."""
    registry = StoreRegistry(settings.stores_root)
    app = FastAPI(title="Evogine Sync", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(SyncError)
    async def sync_error(request: Request, exc: SyncError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"error": str(exc)})

    @app.api_route("/git/{store_name}/{git_path:path}", methods=["GET", "POST"])
    async def git_transport(request: Request, store_name: str, git_path: str) -> Response:
        store_root = registry.resolve_git(store_name)
        # The path converter strips the leading slash; the CGI contract
        # keeps it (git_path arrives as "/info/refs", not "info/refs").
        git_path = f"/{git_path}"
        query_string = request.url.query
        body = await request.body() if request.method == "POST" else b""
        peer = _peer(request)
        mutating = receive_pack_requested(git_path, query_string)
        if mutating and not is_loopback_peer(peer):
            return JSONResponse(
                status_code=403,
                content={
                    "error": (
                        f"store mutation is loopback-only regardless of --expose-lan: "
                        f"peer {peer!r} may not push to {store_name!r} (the approval "
                        "boundary never trusts a network peer)"
                    )
                },
            )
        content_type = request.headers.get("content-type")

        def backend_call() -> CgiResponse:
            if request.method == "POST" and mutating:

                def locked_push() -> CgiResponse:
                    with store_lock(store_root):
                        return run_git_http_backend(
                            registry.stores_root,
                            store_name,
                            git_path,
                            method=request.method,
                            query_string=query_string,
                            content_type=content_type,
                            body=body,
                            remote_addr=peer,
                        )

                return locked_push()
            return run_git_http_backend(
                registry.stores_root,
                store_name,
                git_path,
                method=request.method,
                query_string=query_string,
                content_type=content_type,
                body=body,
                remote_addr=peer,
            )

        cgi = await run_in_threadpool(backend_call)
        return Response(
            content=cgi.body,
            status_code=cgi.status,
            headers={name: value for name, value in cgi.headers},
            media_type=None,
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "stores": registry.names(),
            "bind_policy": {
                "host": settings.host,
                "expose_lan": settings.expose_lan,
                "mutations": "loopback-only regardless of any flag",
            },
        }

    @app.get("/api/stores")
    def stores_list() -> dict[str, list[str]]:
        return {"stores": registry.names()}

    @app.get("/api/stores/{name}/overview")
    def store_overview(name: str) -> object:
        return queries.read_overview(registry.resolve(name))

    @app.get("/api/stores/{name}/records")
    def store_records(name: str, status: str | None = None) -> object:
        return queries.read_records(registry.resolve(name), status=status)

    @app.get("/api/stores/{name}/records/{record_id}")
    def store_record(name: str, record_id: str) -> object:
        return queries.read_record(registry.resolve(name), record_id, name)

    @app.get("/api/stores/{name}/decisions")
    def store_decisions(name: str) -> object:
        return queries.read_decisions(registry.resolve(name))

    @app.get("/api/stores/{name}/approvals")
    def store_approvals(name: str) -> object:
        return queries.read_approvals(registry.resolve(name))

    @app.get("/api/stores/{name}/funnel")
    def store_funnel(name: str) -> object:
        return queries.read_funnel(registry.resolve(name))

    @app.get("/api/stores/{name}/evidence")
    def store_evidence(name: str) -> object:
        return queries.read_evidence(registry.resolve(name))

    @app.get("/api/stores/{name}/shadow")
    def store_shadow(name: str) -> object:
        return queries.read_shadow_outputs(registry.resolve(name))

    @app.get("/api/stores/{name}/spend")
    def store_spend(name: str) -> object:
        return queries.read_spend(registry.resolve(name))

    return app
