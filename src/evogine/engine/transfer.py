"""Cross-project import/export: the explicit, human-reviewed transfer flow.

The V4 federation's movement layer (docs/cross-project-memory.md, Option B,
borrowing-with-provenance): bundles of promoted knowledge travel between
per-project stores, and every landing is a reviewed event. The store stays
the enforcement point — this module never persists a row itself, it hands
records to :meth:`~evogine.engine.store.Engine.import_candidate` — so the
four governance invariants hold structurally:

- **§3.1 scoping at the storage layer** — the bundle is built from the
  source store's own promoted pool, and import lands rows only through the
  destination store's write path (with its write-time reference validation).
- **§3.2 promotion-only, per scope** — exports carry promoted records;
  imports land them as *candidates* in the destination. Source-scope
  promotion evidence never masquerades as destination evidence.
- **§3.3 per-store audit, no invented global order** — import events
  reference the source trail by qualified id only; no source sequence
  numbers, no timestamps stitched into one clock.
- **§3.4 qualified ids before cross-scope reference** — export qualifies
  every source-local reference in a record's ``merged_sources`` lineage
  (the ``MergedSource`` pattern extended across stores: provenance content
  travels verbatim, references travel qualified); import refuses a bundle
  that arrives with unqualified cross-scope references.

Refusal discipline: an import is all-or-nothing. Every validation — bundle
grammar, schema version, provenance completeness, destination conflicts —
runs before the first write, and a refused bundle persists nothing (a
visible refusal, not a partial application). Once validation passes, the
confirmed apply lands each record through its own audited, committed
lifecycle transition, matching the store's one-commit-per-transition
convention.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine, StoreError, split_qualified_id

#: Bundle schema version. Import refuses any other value — fail-closed:
#: a newer bundle is not silently misread by an older reader.
BUNDLE_SCHEMA_VERSION: Literal[1] = 1

#: Marker that a JSON document is an Evogine memory export bundle.
BUNDLE_KIND: Literal["evogine-memory-export"] = "evogine-memory-export"

# Provenance fields that identify where knowledge came from. The note is
# commentary (may be empty); these three may not — an import without them
# would land belief that cannot be traced to its source.
_IDENTITY_FIELDS = ("proposed_by", "origin", "source")


class ImportRefused(Exception):
    """Malformed bundle or unlandable import: refused, nothing applied."""


class ExportBundleEntry(BaseModel):
    """One exported record: its qualified source id plus the full payload.

    The payload is the source record verbatim — provenance and merged
    lineage travel whole, never summarized. ``qualified_id`` is the
    cross-scope name the record answers to outside its store (§3.4).
    """

    qualified_id: str
    record: LearningRecord


class ExportManifest(BaseModel):
    """The reviewed face of a bundle: where it came from and what it holds."""

    source_store: str
    exported_at: datetime
    records: list[ExportBundleEntry]


class ExportBundle(BaseModel):
    """A memory export: envelope (schema identity) plus the manifest.

    ``schema_version`` and ``kind`` are pinned Literals — a document that
    is not this exact bundle grammar refuses at parse time, before any
    manifest field is trusted.
    """

    schema_version: Literal[1]
    kind: Literal["evogine-memory-export"]
    manifest: ExportManifest


class ImportLanding(BaseModel):
    """One row a confirmed import would land, as the reviewer sees it."""

    qualified_id: str
    record_id: str
    type: str
    title: str
    confidence: float


class ImportPreview(BaseModel):
    """The dry-run summary: what would land, and what refuses it.

    ``conflicts`` non-empty means the import cannot apply at all — the
    list is the visible refusal state, and nothing has been written.
    """

    landings: list[ImportLanding]
    conflicts: list[str]


def _qualify_reference(reference: str, source_store: str) -> str:
    """Qualify one lineage reference for cross-scope travel (§3.4).

    A source-local (unqualified) merged-source reference names a record
    that lives in the source store — abroad it is ambiguous by
    construction, so it is qualified with the source identity. An already
    qualified reference is foreign lineage (this store imported from a
    third scope earlier) and passes through verbatim: qualification is
    never double-applied, and nothing else about the reference is touched.
    """
    if ":" in reference:
        return reference
    return f"{source_store}:{reference}"


def _for_export(record: LearningRecord, source_store: str) -> LearningRecord:
    """The cross-scope form of a record: verbatim payload, qualified lineage.

    The only transformation export ever applies — unqualified
    ``merged_sources`` references gain the source-store qualifier. Every
    provenance-bearing field travels byte-for-byte.
    """
    return record.model_copy(
        update={
            "merged_sources": [
                source.model_copy(
                    update={"record_id": _qualify_reference(source.record_id, source_store)}
                )
                for source in record.merged_sources
            ]
        }
    )


def export_bundle(engine: Engine, *, now: datetime | None = None) -> ExportBundle:
    """Bundle the store's promoted pool: a pure read, persisted nowhere.

    Export is not a lifecycle event — the source store's records, audit
    trail, and git history are untouched (``promoted_records`` only reads).
    Borrowing is what the *destination* records; the source trail keeps no
    export line because nothing moved out of it. Records arrive in the
    store's deterministic pool order, so the same store state produces the
    same bundle bytes (modulo ``exported_at``).
    """
    source_store = engine.store_name
    entries = [
        ExportBundleEntry(
            qualified_id=f"{source_store}:{record.id}",
            record=_for_export(record, source_store),
        )
        for record in engine.promoted_records()
    ]
    return ExportBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        kind=BUNDLE_KIND,
        manifest=ExportManifest(
            source_store=source_store,
            exported_at=now if now is not None else datetime.now(UTC),
            records=entries,
        ),
    )


def write_bundle(bundle: ExportBundle, out: Path | str) -> Path:
    """Write the bundle to ``out`` atomically (tmp + rename, like the store)."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


def _provenance_gaps(record: LearningRecord) -> list[str]:
    """Missing identity provenance on a record or its merged lineage."""
    gaps: list[str] = []
    for field in _IDENTITY_FIELDS:
        if not getattr(record.provenance, field).strip():
            gaps.append(f"{record.id}.provenance.{field} is empty")
    for source in record.merged_sources:
        for field in _IDENTITY_FIELDS:
            if not getattr(source.provenance, field).strip():
                gaps.append(
                    f"{record.id}.merged_sources[{source.record_id}].provenance.{field} is empty"
                )
    return gaps


def _validate_bundle(bundle: ExportBundle) -> None:
    """Semantic bundle validation: raise :class:`ImportRefused` on any defect.

    Parse-time (pydantic) already pinned the envelope; this checks the
    manifest's internal consistency — the grammar the exporter writes and
    nothing else may arrive. Every check runs before any destination state
    is consulted, so a refused bundle has touched nothing.
    """
    try:
        source_store = split_qualified_id(f"{bundle.manifest.source_store}:x")[0]
    except StoreError as exc:
        raise ImportRefused(
            f"bundle source_store {bundle.manifest.source_store!r} is not a valid "
            f"store identity: {exc}"
        ) from exc
    if source_store != bundle.manifest.source_store:
        raise ImportRefused(
            f"bundle source_store {bundle.manifest.source_store!r} is not in identity "
            "form (normalized lowercase) — export always writes the identity"
        )
    seen_ids: set[str] = set()
    for entry in bundle.manifest.records:
        try:
            entry_store, entry_record_id = split_qualified_id(entry.qualified_id)
        except StoreError as exc:
            raise ImportRefused(f"bundle entry has a malformed qualified id: {exc}") from exc
        if entry_store != source_store:
            raise ImportRefused(
                f"bundle entry {entry.qualified_id!r} does not carry the manifest's "
                f"source store ({source_store!r}) — a bundle holds one store's rows"
            )
        record = entry.record
        if entry_record_id != record.id:
            raise ImportRefused(
                f"bundle entry {entry.qualified_id!r} names record id {entry_record_id!r} "
                f"but carries {record.id!r}"
            )
        if record.status != "promoted":
            raise ImportRefused(
                f"bundle record {record.id!r} is {record.status!r}; export bundles "
                "carry promoted records only"
            )
        if record.merged_into is not None or record.retired_at is not None:
            raise ImportRefused(
                f"bundle record {record.id!r} carries lifecycle stamps from another "
                "store (merged/retired state) — exports carry live promoted rows only"
            )
        for source in record.merged_sources:
            if ":" not in source.record_id:
                raise ImportRefused(
                    f"bundle record {record.id!r} carries an unqualified merged-source "
                    f"reference {source.record_id!r} — cross-scope references must be "
                    "qualified <store-name>:<record-id> (invariant 3.4)"
                )
        gaps = _provenance_gaps(record)
        if gaps:
            raise ImportRefused(
                f"bundle record {record.id!r} has incomplete provenance — " + "; ".join(gaps)
            )
        if record.id in seen_ids:
            raise ImportRefused(
                f"bundle lists record id {record.id!r} more than once — an import is "
                "one landing per record"
            )
        seen_ids.add(record.id)


def read_bundle(path: Path | str) -> ExportBundle:
    """Read and fully validate a bundle file; refuse anything else.

    The one accepted state is a complete, internally consistent bundle in
    this reader's schema — everything else (unreadable file, foreign
    document, schema drift, malformed ids, incomplete provenance) raises
    :class:`ImportRefused` before any destination is consulted.
    """
    bundle_path = Path(path)
    try:
        raw = bundle_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImportRefused(f"bundle {bundle_path} is unreadable: {exc}") from exc
    try:
        bundle = ExportBundle.model_validate_json(raw)
    except ValidationError as exc:
        raise ImportRefused(
            f"bundle {bundle_path} is not a valid memory export (kind "
            f"{BUNDLE_KIND!r}, schema version {BUNDLE_SCHEMA_VERSION}): {exc}"
        ) from exc
    _validate_bundle(bundle)
    return bundle


def import_preview(bundle: ExportBundle, engine: Engine) -> ImportPreview:
    """The dry-run: what a confirmed import would land in ``engine``, and why
    it cannot, if it cannot.

    Checks destination state — id collisions (imported rows are new rows:
    ids are preserved, and a destination id is never overwritten or
    resurrected from a terminal state) and lineage references that name
    the destination itself (self-qualified references are refused by the
    store's write-time validation, so they are conflicts here, before any
    write). Never persists anything.
    """
    landings = [
        ImportLanding(
            qualified_id=entry.qualified_id,
            record_id=entry.record.id,
            type=entry.record.type,
            title=entry.record.title,
            confidence=entry.record.confidence,
        )
        for entry in bundle.manifest.records
    ]
    conflicts: list[str] = []
    if bundle.manifest.source_store == engine.store_name:
        conflicts.append(
            f"bundle source_store {bundle.manifest.source_store!r} is this store's own "
            "identity — an import event references another store's trail, never this one"
        )
    for entry in bundle.manifest.records:
        try:
            engine.get(entry.record.id)
        except StoreError:
            continue  # not held here — the id is free and nothing is resurrected
        conflicts.append(
            f"{entry.qualified_id} collides with an existing record in "
            f"{engine.store_name}: {entry.record.id!r} is already held (merge the "
            "duplicates by hand first, or export a bundle that excludes it) — "
            "import never overwrites or resurrects a row"
        )
    for entry in bundle.manifest.records:
        for source in entry.record.merged_sources:
            referenced_store, _ = split_qualified_id(source.record_id)
            if referenced_store == engine.store_name:
                conflicts.append(
                    f"{entry.qualified_id} carries lineage reference "
                    f"{source.record_id!r} naming the destination store "
                    f"({engine.store_name}) — references to this store's own rows "
                    "are unqualified, and an import cannot claim a local merge that "
                    "never happened"
                )
    return ImportPreview(landings=landings, conflicts=conflicts)


def apply_import(
    bundle: ExportBundle,
    engine: Engine,
    *,
    now: datetime | None = None,
) -> list[LearningRecord]:
    """Apply a fully validated import; refuse first if anything cannot land.

    All-or-nothing at the refusal boundary: :func:`import_preview` runs
    first, and any conflict raises :class:`ImportRefused` with nothing
    written. With a clean preview, each record lands through
    :meth:`Engine.import_candidate` — its own audited, committed
    transition (one commit per lifecycle transition is the store's
    convention, so a crash mid-apply leaves already-landed rows intact and
    audited, never half-written).
    """
    preview = import_preview(bundle, engine)
    if preview.conflicts:
        raise ImportRefused(
            "import refused — nothing was applied:\n"
            + "\n".join(f"  - {conflict}" for conflict in preview.conflicts)
        )
    moment = now if now is not None else datetime.now(UTC)
    return [
        engine.import_candidate(entry.record, source_ref=entry.qualified_id, now=moment)
        for entry in bundle.manifest.records
    ]
