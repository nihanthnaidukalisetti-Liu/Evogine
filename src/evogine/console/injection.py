"""The knowledge-injection panel: reviewed import bundles, guarded landing.

``evogine memory export`` writes a bundle of a source store's promoted pool;
``evogine memory import`` is the human-reviewed landing (dry run, then an
explicit confirm). This panel puts that same flow on the dashboard — it adds
no import path of its own. Bundle reading, validation, and the landing all
run through :mod:`evogine.engine.transfer`'s public functions
(:func:`read_bundle`, :func:`import_preview`, :func:`apply_import`), so the
console cannot drift from the CLI's contract: same grammar, same refusals,
same all-or-nothing guarantee.

What the operator sees, in order:

- the bundles directory (the export flow's output location: the store's
  ``imports/`` directory, ``--imports-dir`` to override) with every bundle's
  manifest summary — source store, exported-at, schema version, record
  count, qualified ids — rendered *before* any action;
- a bundle's detail: the same summary plus the dry run (exactly the
  landings a confirmed import would create, nothing written) and, only when
  the dry run is clean, the one guarded confirm button;
- refusals as first-class panels: an invalid manifest, a wrong schema
  version, duplicate qualified ids, a bundle that conflicts with the
  destination, and an unapproved injection (a confirm POST that skipped the
  reviewed dry run) each render explicitly, and nothing is written.

Landed records enter the destination as **candidates** —
:meth:`~evogine.engine.store.Engine.import_candidate` is the only write path,
and the records then face this store's memory gate and a human approval in
the approvals queue, exactly like swarm output. The console never writes
promoted knowledge, and there is no path around the gate.

The one write route follows the approval write path's discipline
(:mod:`evogine.console.write`): the landing runs off the event loop, under
the store's advisory review lock (the same lock class the sync daemon holds
across a receive — one writer at a time per store root), and every refusal
is a visible state that leaves the store untouched.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from evogine.console.launch import ConsoleError
from evogine.engine.store import Engine, StoreError, store_lock
from evogine.engine.transfer import (
    ExportBundle,
    ImportLanding,
    ImportRefused,
    apply_import,
    import_preview,
    read_bundle,
)
from evogine.gates.telemetry import ID_PATTERN

#: The page's render closure (app.py builds it over the Jinja2 environment);
#: typed loosely because the route only forwards template payloads.
RenderFn = Callable[..., Response]

type RefuseFn = Callable[[Request, int, str], Response]
"""Render the console's visible refusal state for a write that is refused."""

IMPORTS_DIRNAME = "imports"
"""The bundles directory under the store root — where the export flow's
``--out`` is pointed so the panel finds bundles by construction, like the
swarm board's ``runs/swarm``."""


def resolve_imports_dir(store_root: Path | str, override: Path | str | None = None) -> Path:
    """The bundles directory the panel reads: the explicit override, or the
    derived default under the store root."""
    if override is not None:
        return Path(override)
    return Path(store_root) / IMPORTS_DIRNAME


def is_safe_bundle_name(name: str) -> bool:
    """Whether ``name`` is a filesystem-safe bundle filename, before any path use.

    The same id grammar the console applies to proposal ids, plus the
    ``.json`` suffix real bundles carry: no separators (so the name cannot
    escape the bundles directory), nothing hidden, nothing traversal-shaped.
    """
    return ID_PATTERN.fullmatch(name) is not None and name.endswith(".json")


def discover_bundles(imports_dir: Path | str) -> list[str]:
    """Bundle filenames in ``imports_dir``, sorted — not a listing of junk.

    Only ``*.json`` files are candidates (``write_bundle``'s atomic
    ``.tmp`` leftover is never a bundle); anything else in the directory is
    not shown, and a missing directory is the caller's empty state.
    """
    directory = Path(imports_dir)
    if not directory.is_dir():
        return []
    return [
        path.name
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name.endswith(".json")
    ]


class BundleSummary(BaseModel):
    """A bundle's manifest summary — everything the operator reviews first."""

    filename: str
    source_store: str
    exported_at: str
    schema_version: int
    record_count: int
    qualified_ids: list[str] = Field(default_factory=list)


class BundleRow(BaseModel):
    """One discovered bundle on the panel, as the list page renders it.

    A bundle that refuses to read is a first-class refusal row — named,
    with the reason — while the rest of the panel keeps rendering; one
    poisoned bundle must not hide the others.
    """

    filename: str
    summary: BundleSummary | None = None
    refusal: str | None = None


class InjectionView(BaseModel):
    """The knowledge-injection page: the bundles directory and its bundles."""

    imports_dir: str
    dir_present: bool
    """Whether the bundles directory exists at all — absence is an explicit
    empty state (nothing was exported yet), never a fabricated panel."""
    bundles: list[BundleRow] = Field(default_factory=list)


class BundleDetailView(BaseModel):
    """One bundle's review page: summary, dry run, and the guarded action."""

    filename: str
    summary: BundleSummary | None = None
    refusal: str | None = None
    """Set when the bundle itself refuses to read (invalid manifest, wrong
    schema version, duplicate qualified ids) — the panel's refusal state."""
    landings: list[ImportLanding] = Field(default_factory=list)
    """The dry run's landings — exactly what a confirmed import creates."""
    conflicts: list[str] = Field(default_factory=list)
    """The dry run's destination conflicts — non-empty means the import
    cannot apply at all, and the panel says so before any write."""
    applied_count: int | None = None
    """How many records the confirmed import landed, when the operator just
    applied this bundle (the redirect carries it). ``None`` otherwise."""


def _summary(filename: str, bundle: ExportBundle) -> BundleSummary:
    """The reviewed face of a validated bundle, for the summary table."""
    manifest = bundle.manifest
    return BundleSummary(
        filename=filename,
        source_store=manifest.source_store,
        exported_at=manifest.exported_at.isoformat(timespec="seconds"),
        schema_version=bundle.schema_version,
        record_count=len(manifest.records),
        qualified_ids=[entry.qualified_id for entry in manifest.records],
    )


def build_injection_view(imports_dir: Path | str) -> InjectionView:
    """The bundles list: every discovered bundle summarized or refusing.

    Reading is per bundle and isolated — a bundle that refuses (invalid
    manifest, wrong schema version, duplicate qualified ids) becomes its
    own refusal row with the reason from the transfer module's validator,
    and the rest of the panel still renders.
    """
    directory = Path(imports_dir)
    rows: list[BundleRow] = []
    for filename in discover_bundles(directory):
        try:
            bundle = read_bundle(directory / filename)
        except ImportRefused as exc:
            rows.append(BundleRow(filename=filename, refusal=str(exc)))
            continue
        rows.append(BundleRow(filename=filename, summary=_summary(filename, bundle)))
    return InjectionView(imports_dir=str(directory), dir_present=directory.is_dir(), bundles=rows)


def build_bundle_detail(
    store_root: Path | str,
    imports_dir: Path | str,
    filename: str,
    *,
    applied_count: int | None = None,
) -> BundleDetailView:
    """One bundle's review page: summary, destination dry run, guarded action.

    The dry run is the import command's own :func:`import_preview` against
    this store's engine — the same conflicts the CLI refuses on, rendered
    here as the refusal panel. A clean dry run is what makes the confirm
    button exist; nothing in this builder writes.
    """
    if not is_safe_bundle_name(filename):
        raise ConsoleError(f"bundle name {filename!r} is not a filesystem-safe bundle filename")
    directory = Path(imports_dir)
    try:
        bundle = read_bundle(directory / filename)
    except ImportRefused as exc:
        return BundleDetailView(filename=filename, refusal=str(exc))
    engine = Engine(store_root)
    preview = import_preview(bundle, engine)
    return BundleDetailView(
        filename=filename,
        summary=_summary(filename, bundle),
        landings=preview.landings,
        conflicts=preview.conflicts,
        applied_count=applied_count,
    )


def parse_form_fields(body: bytes) -> dict[str, str]:
    """The fields of a urlencoded form POST body (missing fields become '').

    Parsed by hand for the same reason the approval write path parses its
    own: urlencoded is the browser's default enctype, so the console needs
    no client framework and no multipart dependency.
    """
    fields = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    return {key: values[0] for key, values in fields.items()}


def injection_routes(
    store_root: Path,
    imports_dir: Path,
    render: RenderFn,
    *,
    refuse: RefuseFn,
) -> APIRouter:
    """The panel's routes, bound to one store root, bundles dir, and render."""
    router = APIRouter()

    @router.get("/injection")
    def injection(request: Request) -> Response:
        """The bundles panel; an unreadable directory refuses visibly."""
        try:
            view = build_injection_view(imports_dir)
        except (ConsoleError, ValueError) as exc:
            return render(request, "error.html", {"status": 500, "message": str(exc)}, 500)
        return render(request, "injection.html", {"view": view})

    @router.get("/injection/bundles/{filename}")
    def bundle_detail(request: Request, filename: str) -> Response:
        """One bundle's review page; its refusals render as the page's panels."""
        applied_raw = request.query_params.get("applied")
        applied_count: int | None = None
        if applied_raw is not None:
            try:
                applied_count = int(applied_raw)
            except ValueError:
                applied_count = None  # not this route's redirect — no banner
        try:
            view = build_bundle_detail(
                store_root, imports_dir, filename, applied_count=applied_count
            )
        except ConsoleError as exc:
            return refuse(request, 400, str(exc))
        return render(request, "bundle.html", {"view": view})

    def _apply(request: Request, fields: dict[str, str]) -> Response:
        """The one guarded write: reviewed bundle + confirmed dry run + lock."""
        bundle_name = fields.get("bundle", "")
        confirmed = fields.get("confirmed", "")
        if not is_safe_bundle_name(bundle_name):
            return refuse(
                request,
                400,
                f"bundle name {bundle_name!r} is not a filesystem-safe bundle filename",
            )
        try:
            bundle = read_bundle(imports_dir / bundle_name)
        except ImportRefused as exc:
            return refuse(
                request,
                422,
                f"import refused for {bundle_name!r} — nothing was applied: {exc}",
            )
        if confirmed != "yes":
            # The dry run's confirmation IS the approval boundary for the
            # landing: without it this is an unattended injection, and the
            # console refuses instead of writing.
            return refuse(
                request,
                403,
                f"unapproved injection — {bundle_name!r} was not applied. An import lands "
                "only through the reviewed dry run's explicit confirmation; review the "
                "bundle first, then confirm on its page.",
            )
        try:
            engine = Engine(store_root)
            preview = import_preview(bundle, engine)
        except (StoreError, ValueError) as exc:
            return refuse(
                request, 500, f"could not review {bundle_name!r} against this store: {exc}"
            )
        if preview.conflicts:
            return refuse(
                request,
                409,
                "import refused — nothing was applied:\n"
                + "\n".join(f"  - {conflict}" for conflict in preview.conflicts),
            )
        try:
            # The store's advisory review lock across the whole apply — the
            # same one-writer-per-root discipline the approval decisions and
            # the daemon's receives hold, so a landing cannot interleave
            # with a concurrent lifecycle write.
            with store_lock(store_root):
                landed = apply_import(bundle, engine)
        except ImportRefused as exc:
            return refuse(request, 409, str(exc))
        except StoreError as exc:
            # Outside the refusal vocabulary: a contract violation mid-apply
            # (e.g. a concurrent writer broke the one-writer-per-root rule).
            return refuse(request, 500, f"import failed after landing began — {exc}")
        return RedirectResponse(
            f"/injection/bundles/{bundle_name}?applied={len(landed)}", status_code=303
        )

    async def _apply_in_threadpool(request: Request) -> Response:
        """Read the small form body, then run the blocking apply off the loop.

        The landing commits through git subprocesses and holds the store's
        review lock — none of that may hold the event loop.
        """
        return await run_in_threadpool(_apply, request, parse_form_fields(await request.body()))

    @router.post("/injection/apply")
    async def apply_route(request: Request) -> Response:
        return await _apply_in_threadpool(request)

    return router


__all__ = [
    "IMPORTS_DIRNAME",
    "BundleDetailView",
    "BundleRow",
    "BundleSummary",
    "InjectionView",
    "build_bundle_detail",
    "build_injection_view",
    "discover_bundles",
    "injection_routes",
    "is_safe_bundle_name",
    "parse_form_fields",
    "resolve_imports_dir",
]
