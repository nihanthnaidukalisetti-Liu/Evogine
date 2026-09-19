"""The console's write path: POST approve/reject over the approval boundary.

Transport only — the routes here add HTTP and the operator identity to
:func:`evogine.gates.approvals.approve_proposal` /
:func:`evogine.gates.approvals.reject_proposal`, which own every rule the
write enforces: the store's review lock across the whole decision window,
the pending-state re-check, the evidence requirement, and the
nonblank-reason requirement for rejections. The console never re-implements
or bypasses them, and it records nothing on its own: a proposal file and
the store's audit trail are the only ledgers.

Two routes, each deciding exactly one proposal; there is no batch
primitive, by design — a promotion is one human decision on one proposal,
and a console that approves in bulk invites decisions no one made.

Every refusal is a visible state (the error page names the proposal and
why) and leaves the store untouched; success redirects back to the queue
(PRG — a browser refresh cannot re-post a decision).
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from evogine.engine.store import Engine, StoreError
from evogine.gates.approvals import (
    ApprovalError,
    ApprovalLog,
    approve_proposal,
    operator_identity,
    reject_proposal,
)
from evogine.gates.telemetry import ID_PATTERN

type RefuseFn = Callable[[Request, int, str], Response]
"""Render the console's visible refusal state for a write that is refused."""


def parse_reason_field(body: bytes) -> str:
    """The ``reason`` field of a urlencoded form POST body ('' when absent).

    Form bodies are parsed by hand — urlencoded is the browser's default
    enctype, so the console needs no client framework and no multipart
    dependency. A missing field and an empty one are the same thing here:
    the reason gate below decides whether that is acceptable.
    """
    fields = urllib.parse.parse_qs(
        body.decode("utf-8", errors="replace"), keep_blank_values=True
    )
    return fields.get("reason", [""])[0]


def is_safe_proposal_id(proposal_id: str) -> bool:
    """Whether ``proposal_id`` is a filesystem-safe id, before any path use.

    The boundary's log keys proposal files by id; a traversal-shaped id
    must be refused at the transport before it can aim the lookup outside
    the store (the read path applies the same rule — one id rule for both).
    """
    return ID_PATTERN.fullmatch(proposal_id) is not None


def register_write_routes(
    app: FastAPI,
    store_root: Path,
    operator: str,
    *,
    refuse: RefuseFn,
) -> None:
    """Register the two write routes on the console app.

    ``operator`` is the identity the console was launched with (validated
    nonblank at launch); it is recorded as ``decided_by`` on every decision
    this console makes. The boundary re-validates the format at record
    time — the transport pre-flights it so a launch identity the boundary
    will never accept refuses visibly with the fix, not as a surprise 409
    on the operator's first action.
    """
    try:
        identity = operator_identity(operator)
    except ApprovalError as exc:
        raise RuntimeError(
            f"console operator identity cannot be recorded by the approval "
            f"boundary: {exc}; relaunch with a matching --operator"
        ) from exc

    def decide(
        request: Request,
        proposal_id: str,
        *,
        reject: bool,
        reason: str | None,
    ) -> Response:
        """One proposal, one decision, through the boundary under the lock."""
        if not is_safe_proposal_id(proposal_id):
            return refuse(
                request, 400, f"proposal id {proposal_id!r} is not a filesystem-safe id"
            )
        log = ApprovalLog(store_root)
        try:
            # Refuse an unknown or malformed proposal as its own state (404 /
            # 500) before the boundary call, which would lump both into the
            # decided-or-refused conflict answer.
            log.get(proposal_id)
        except ApprovalError as exc:
            return refuse(request, 404, str(exc))
        except ValueError as exc:
            return refuse(
                request, 500, f"proposal file {proposal_id}.json is malformed: {exc}"
            )
        try:
            if reject:
                checked = (reason or "").strip()
                if not checked:
                    return refuse(
                        request,
                        400,
                        "a rejection requires a nonblank reason — it is recorded on "
                        "the proposal and the store's audit trail",
                    )
                reject_proposal(
                    engine=Engine(store_root),
                    log=log,
                    proposal_id=proposal_id,
                    decided_by=identity,
                    reason=checked,
                )
            else:
                approve_proposal(
                    engine=Engine(store_root),
                    log=log,
                    proposal_id=proposal_id,
                    decided_by=identity,
                    # An approval's reason is optional (the note field): a
                    # blank one is absent, matching the CLI transport.
                    reason=(reason or "").strip() or None,
                )
        except (ApprovalError, StoreError) as exc:
            # The boundary refused (decided proposal, lost evidence, non-
            # candidate record, dropped record) — a visible refusal state,
            # the store untouched by this request.
            return refuse(request, 409, str(exc))
        return RedirectResponse("/approvals", status_code=303)

    async def decide_in_threadpool(
        request: Request, proposal_id: str, *, reject: bool
    ) -> Response:
        """Read the small form body, then run the blocking boundary off the loop.

        The boundary commits through git subprocesses and blocks on the
        store's review lock — none of that may hold the event loop, or a
        second concurrent decision could not even reach its own lock wait.
        """
        reason = parse_reason_field(await request.body())
        return await run_in_threadpool(
            decide, request, proposal_id, reject=reject, reason=reason
        )

    @app.post("/approvals/{proposal_id}/approve")
    async def approve_route(request: Request, proposal_id: str) -> Response:
        return await decide_in_threadpool(request, proposal_id, reject=False)

    @app.post("/approvals/{proposal_id}/reject")
    async def reject_route(request: Request, proposal_id: str) -> Response:
        return await decide_in_threadpool(request, proposal_id, reject=True)
