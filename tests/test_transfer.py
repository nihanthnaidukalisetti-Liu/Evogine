"""Unit tests for the V4 cross-project import/export flow.

Covers the movement layer over the federation foundation
(docs/cross-project-memory.md, Option B): export bundles (manifest plus
verbatim payloads, merged-source lineage qualified per invariant 3.4),
bundle validation on import (schema version, qualified ids, provenance
completeness — malformed bundles refused all-or-nothing), the dry-run
review surface, the confirmed apply (candidates with destination-clock
decay stamping, audit trail referencing the source by qualified id
only), and the ``memory export|import`` CLI surface.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import make_record, replay_evidence, temp_store_root

from evogine.cli import main
from evogine.engine.records import (
    DecayConfig,
    MergedSource,
    RecordProvenance,
    decayed_confidence,
)
from evogine.engine.store import AuditEvent, Engine, StoreError
from evogine.engine.transfer import (
    BUNDLE_KIND,
    BUNDLE_SCHEMA_VERSION,
    ImportRefused,
    apply_import,
    export_bundle,
    import_preview,
    read_bundle,
    write_bundle,
)

# A fixed import moment: late enough that destination-clock anchoring and
# source-`created` anchoring give clearly different decay values.
T0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def source_root():
    root = temp_store_root("evogine-transfer-src-")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def dest_root():
    root = temp_store_root("evogine-transfer-dst-")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def source(source_root) -> Engine:
    return Engine(source_root, name="alpha")


@pytest.fixture
def dest(dest_root) -> Engine:
    return Engine(dest_root, name="beta")


def _promote(engine: Engine, record_id: str, topic: str = "429 backoff") -> None:
    """Propose, queue, and promote one record through the store's lifecycle."""
    engine.propose(make_record(record_id, topic=topic))
    engine.queue_replay(record_id, [f"run-{record_id}"])
    engine.promote(record_id, replay_evidence([f"run-{record_id}"]))


def _fingerprint(root: Path) -> tuple[int, int, str]:
    """(record files, audit lines, porcelain) — any write moves at least one.

    The porcelain component is the raw ``git status --porcelain`` output:
    a store that has only been *constructed* (never written) shows its
    birth ``review.lock`` untracked, identically before and after, so a
    dry run's honesty is still an exact before/after comparison.
    """
    root = Path(root)
    records = list((root / "records").glob("*.json"))
    audit = root / "audit.jsonl"
    lines = audit.read_text(encoding="utf-8").splitlines() if audit.exists() else []
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,  # a probe: a non-repo dir yields empty output, not a failure
    ).stdout.strip()
    return len(records), len(lines), porcelain


def _merged_twin(engine: Engine, survivor_id: str, twin_id: str, topic: str) -> None:
    """Give ``survivor_id`` absorbed lineage: merge a same-body twin into it."""
    engine.propose(make_record(twin_id, topic=topic))
    engine.merge(survivor_id, twin_id, "exact duplicate from the same rollout")


# ---------------------------------------------------------------- export


def test_export_bundle_shape(source: Engine) -> None:
    _promote(source, "lr-1")
    bundle = export_bundle(source, now=T0)
    assert bundle.kind == BUNDLE_KIND
    assert bundle.schema_version == BUNDLE_SCHEMA_VERSION
    assert bundle.manifest.source_store == "alpha"
    assert bundle.manifest.exported_at == T0
    assert len(bundle.manifest.records) == 1
    entry = bundle.manifest.records[0]
    assert entry.qualified_id == "alpha:lr-1"
    # The payload is the source record verbatim — nothing summarized.
    assert entry.record == source.get("lr-1")


def test_export_covers_the_whole_promoted_pool(source: Engine) -> None:
    _promote(source, "lr-1", topic="429 backoff")
    _promote(source, "lr-2", topic="deadlock contention")
    bundle = export_bundle(source, now=T0)
    assert [entry.qualified_id for entry in bundle.manifest.records] == [
        "alpha:lr-1",
        "alpha:lr-2",
    ]


def test_export_is_a_pure_read_of_the_source(source: Engine, source_root: Path) -> None:
    _promote(source, "lr-1")
    before = _fingerprint(source_root)
    bundle = export_bundle(source)
    write_bundle(bundle, Path(source_root).parent / "bundle.json")
    assert _fingerprint(source_root) == before


def test_export_empty_pool_is_a_valid_bundle(source: Engine, source_root: Path) -> None:
    bundle = export_bundle(source, now=T0)
    assert bundle.manifest.records == []
    path = write_bundle(bundle, Path(source_root).parent / "empty.json")
    assert read_bundle(path).manifest.records == []


def test_export_qualifies_merged_lineage(source: Engine) -> None:
    _promote(source, "lr-1")
    _merged_twin(source, "lr-1", "twin-1", topic="429 backoff")
    local_lineage = source.get("lr-1").merged_sources[0]
    bundle = export_bundle(source, now=T0)
    exported_lineage = bundle.manifest.records[0].record.merged_sources[0]
    assert exported_lineage.record_id == "alpha:twin-1"
    # Every provenance-bearing field of the absorbed twin travels verbatim
    # (the MergedSource pattern: provenance is never rewritten).
    assert exported_lineage.title == local_lineage.title
    assert (
        exported_lineage.provenance.model_dump_json() == local_lineage.provenance.model_dump_json()
    )
    assert exported_lineage.confidence == local_lineage.confidence
    assert exported_lineage.merged_at == local_lineage.merged_at


def test_export_passes_foreign_lineage_through_unchanged(source: Engine) -> None:
    """Already-qualified lineage (imported earlier from a third store) is never
    double-qualified — qualification is applied exactly once."""
    _promote(source, "lr-1")
    foreign = MergedSource(
        record_id="gamma:lr-9",
        title="A third scope's absorbed twin",
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="gamma-store",
            source="gamma rollout",
            note="lineage this store imported earlier",
        ),
        confidence=0.7,
        merged_at=T0,
    )
    promoted = source.get("lr-1").model_copy(update={"merged_sources": [foreign]})
    source._save(promoted)
    source._append_audit(
        AuditEvent(event="merge", record_id="lr-1", to_status="promoted", reason="test setup")
    )
    source._commit("merge: foreign lineage (test setup)")
    bundle = export_bundle(source, now=T0)
    assert bundle.manifest.records[0].record.merged_sources[0].record_id == "gamma:lr-9"


def test_write_bundle_is_atomic_and_round_trips(source: Engine, tmp_path: Path) -> None:
    _promote(source, "lr-1")
    bundle = export_bundle(source, now=T0)
    out = tmp_path / "nested" / "bundle.json"
    written = write_bundle(bundle, out)
    assert written == out
    assert not out.with_suffix(".json.tmp").exists()
    assert read_bundle(out) == bundle


# ------------------------------------------------------------ read/refuse


def _write_raw(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _bundle_dict(source: Engine, **overrides: object) -> dict[str, object]:
    bundle = export_bundle(source, now=T0)
    raw = json.loads(bundle.model_dump_json())
    raw.update(overrides)
    return raw


def test_read_bundle_refuses_unreadable_and_foreign_documents(
    tmp_path: Path,
) -> None:
    with pytest.raises(ImportRefused, match="unreadable"):
        read_bundle(tmp_path / "absent.json")
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(ImportRefused, match="not a valid memory export"):
        read_bundle(garbage)
    foreign = _write_raw(tmp_path / "foreign.json", {"hello": "world"})
    with pytest.raises(ImportRefused, match="not a valid memory export"):
        read_bundle(foreign)


def test_read_bundle_refuses_schema_drift(source: Engine, tmp_path: Path) -> None:
    _promote(source, "lr-1")
    wrong_version = _write_raw(tmp_path / "v99.json", _bundle_dict(source, schema_version=99))
    with pytest.raises(ImportRefused, match="not a valid memory export"):
        read_bundle(wrong_version)
    wrong_kind = _write_raw(tmp_path / "kind.json", _bundle_dict(source, kind="something-else"))
    with pytest.raises(ImportRefused, match="not a valid memory export"):
        read_bundle(wrong_kind)


def _unqualified_lineage() -> dict[str, object]:
    return {
        "record_id": "twin-1",
        "title": "unqualified lineage",
        "provenance": {
            "proposed_by": "reflector",
            "origin": "alpha",
            "source": "alpha rollout",
            "note": "",
        },
        "confidence": 0.5,
        "merged_at": "2026-10-01T00:00:00Z",
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["manifest"].update(source_store="Not A Name"), "not a valid store"),
        (lambda d: d["manifest"].update(source_store="Alpha"), "identity form"),
        (
            lambda d: d["manifest"]["records"][0].update(qualified_id="lr-1"),
            "malformed qualified id",
        ),
        (
            lambda d: d["manifest"]["records"][0].update(qualified_id="beta:lr-1"),
            "does not carry the manifest's source store",
        ),
        (
            lambda d: d["manifest"]["records"][0].update(qualified_id="alpha:other"),
            "names record id",
        ),
        (
            lambda d: d["manifest"]["records"][0]["record"].update(status="candidate"),
            "carry promoted records only",
        ),
        (
            lambda d: d["manifest"]["records"][0]["record"].update(merged_into="lr-2"),
            "lifecycle stamps",
        ),
        (
            lambda d: d["manifest"]["records"][0]["record"].update(
                retired_at="2026-10-01T00:00:00Z"
            ),
            "lifecycle stamps",
        ),
        (
            lambda d: d["manifest"]["records"][0]["record"]["merged_sources"].append(
                _unqualified_lineage()
            ),
            "unqualified merged-source reference",
        ),
        (
            lambda d: d["manifest"]["records"][0]["record"]["provenance"].update(origin=""),
            "incomplete provenance",
        ),
        (
            lambda d: d["manifest"]["records"].append(d["manifest"]["records"][0]),
            "more than once",
        ),
    ],
)
def test_read_bundle_refuses_malformed_bundles(
    source: Engine, tmp_path: Path, mutate: object, match: str
) -> None:
    _promote(source, "lr-1")
    document = _bundle_dict(source)
    mutate(document)  # type: ignore[operator]
    path = _write_raw(tmp_path / "malformed.json", document)
    with pytest.raises(ImportRefused, match=match):
        read_bundle(path)


def test_read_bundle_refuses_incomplete_lineage_provenance(source: Engine, tmp_path: Path) -> None:
    """Provenance completeness reaches into merged lineage: an absorbed twin
    whose own provenance lost its origin cannot land."""
    _promote(source, "lr-1")
    _merged_twin(source, "lr-1", "twin-1", topic="429 backoff")
    document = _bundle_dict(source)
    lineage = document["manifest"]["records"][0]["record"]["merged_sources"]
    lineage[0]["provenance"]["origin"] = " "
    path = _write_raw(tmp_path / "lineage.json", document)
    with pytest.raises(ImportRefused, match="merged_sources"):
        read_bundle(path)


# ---------------------------------------------------------------- preview


def test_preview_lists_what_would_land(source: Engine, dest: Engine) -> None:
    _promote(source, "lr-1")
    bundle = export_bundle(source, now=T0)
    preview = import_preview(bundle, dest)
    assert preview.conflicts == []
    assert len(preview.landings) == 1
    landing = preview.landings[0]
    assert landing.qualified_id == "alpha:lr-1"
    assert landing.record_id == "lr-1"
    assert landing.title == source.get("lr-1").title
    assert landing.confidence == source.get("lr-1").confidence


def test_preview_never_writes(source: Engine, dest: Engine, dest_root: Path) -> None:
    _promote(source, "lr-1")
    bundle = export_bundle(source, now=T0)
    before = _fingerprint(dest_root)
    import_preview(bundle, dest)
    assert _fingerprint(dest_root) == before


def test_preview_flags_id_collisions(source: Engine, dest: Engine) -> None:
    _promote(source, "lr-1")
    dest.propose(make_record("lr-1", topic="unrelated local knowledge"))
    preview = import_preview(export_bundle(source, now=T0), dest)
    assert preview.conflicts
    assert any("collides" in conflict for conflict in preview.conflicts)


def test_preview_flags_lineage_naming_the_destination(source: Engine, dest: Engine) -> None:
    """A bundle claiming the destination absorbed one of its rows is a
    conflict: an import cannot assert a local merge that never happened."""
    _promote(source, "lr-1")
    lineage = MergedSource(
        record_id="beta:local-row",
        title="claims a local merge",
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="alpha",
            source="alpha rollout",
            note="hostile or drifted bundle",
        ),
        confidence=0.5,
        merged_at=T0,
    )
    document = json.loads(export_bundle(source, now=T0).model_dump_json())
    document["manifest"]["records"][0]["record"]["merged_sources"] = [
        json.loads(lineage.model_dump_json())
    ]
    bundle = read_bundle(_write_raw(Path(source.root).parent / "claim.json", document))
    preview = import_preview(bundle, dest)
    assert any("naming the destination" in conflict for conflict in preview.conflicts)


def test_preview_flags_a_bundle_of_the_destination_own_identity(
    dest: Engine, dest_root: Path
) -> None:
    """A bundle whose source identity equals the destination is refused before
    the store has to — the audit trail references another store, never itself."""
    document = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "manifest": {
            "source_store": "beta",
            "exported_at": "2026-10-01T12:00:00Z",
            "records": [],
        },
    }
    bundle = read_bundle(_write_raw(Path(dest_root).parent / "self.json", document))
    preview = import_preview(bundle, dest)
    assert any("own identity" in conflict for conflict in preview.conflicts)


# ------------------------------------------------------------------ apply


def test_round_trip_lands_an_approved_candidate_with_verbatim_provenance(
    source: Engine, source_root: Path, dest: Engine
) -> None:
    """The manifest round-trip: export then confirmed import produces a
    destination candidate whose provenance is byte-identical to the source's."""
    _promote(source, "lr-1", topic="cache stampede")
    source_record = source.get("lr-1")
    bundle = export_bundle(source, now=T0)
    path = write_bundle(bundle, Path(source_root).parent / "bundle.json")
    landed = apply_import(read_bundle(path), dest, now=T0 + timedelta(hours=1))
    assert len(landed) == 1
    imported = dest.get("lr-1")
    # Provenance byte-identical to source.
    assert imported.provenance.model_dump_json() == source_record.provenance.model_dump_json()
    # Lifecycle fields are the destination's decision, not the source's.
    assert imported.status == "candidate"
    assert imported.counterfactual.replay_run_ids == []
    assert imported.counterfactual.verdict_change is None
    assert imported.promoted_at is None
    # Destination-clock decay stamping: the confidence clock starts here.
    assert imported.confidence_updated_at == T0 + timedelta(hours=1)
    # Identity and knowledge travel verbatim.
    assert imported.id == source_record.id
    assert imported.title == source_record.title
    assert imported.body == source_record.body
    assert imported.created == source_record.created
    assert imported.confidence == source_record.confidence


def test_round_trip_byte_identical_bundle_payloads(
    source: Engine, source_root: Path, tmp_path: Path, dest: Engine
) -> None:
    """The manifest payload and the landed record re-serialize to the same
    provenance bytes as the source record file."""
    _promote(source, "lr-1")
    bundle = export_bundle(source, now=T0)
    path = write_bundle(bundle, tmp_path / "bundle.json")
    on_disk = json.loads((Path(source_root) / "records" / "lr-1.json").read_text())
    in_manifest = json.loads(path.read_text())["manifest"]["records"][0]["record"]
    landed = apply_import(read_bundle(path), dest, now=T0)[0]
    assert json.loads(landed.provenance.model_dump_json()) == on_disk["provenance"]
    assert json.loads(landed.provenance.model_dump_json()) == in_manifest["provenance"]


def test_round_trip_with_merged_lineage(source: Engine, source_root: Path, dest: Engine) -> None:
    _promote(source, "lr-1")
    _merged_twin(source, "lr-1", "twin-1", topic="429 backoff")
    source_lineage = source.get("lr-1").merged_sources[0]
    bundle = export_bundle(source, now=T0)
    path = write_bundle(bundle, Path(source_root).parent / "bundle.json")
    landed = apply_import(read_bundle(path), dest, now=T0)[0]
    landed_lineage = landed.merged_sources[0]
    # The reference is qualified (invariant 3.4); every provenance-bearing
    # field is byte-identical to the source lineage.
    assert landed_lineage.record_id == f"alpha:{source_lineage.record_id}"
    assert landed_lineage.title == source_lineage.title
    assert (
        landed_lineage.provenance.model_dump_json() == source_lineage.provenance.model_dump_json()
    )
    assert landed_lineage.confidence == source_lineage.confidence
    assert landed_lineage.merged_at == source_lineage.merged_at


def test_import_audit_event_references_the_source_by_qualified_id_only(
    source: Engine, source_root: Path, dest: Engine, dest_root: Path
) -> None:
    """Cross-store audit events reference the source trail by qualified id
    only — no source sequence numbers, no invented global ordering (§3.3)."""
    _promote(source, "lr-1")
    bundle = export_bundle(source, now=T0)
    path = write_bundle(bundle, Path(source_root).parent / "bundle.json")
    apply_import(read_bundle(path), dest, now=T0)
    events = [
        json.loads(line) for line in (Path(dest_root) / "audit.jsonl").read_text().splitlines()
    ]
    assert [event["seq"] for event in events] == [1]  # this store's own clock
    import_event = events[0]
    assert import_event["event"] == "import"
    assert import_event["record_id"] == "lr-1"
    assert import_event["to_status"] == "candidate"
    assert import_event["from_status"] is None
    assert "alpha:lr-1" in import_event["reason"]
    # No source-trail sequence numbers leaked into the reason.
    assert "seq" not in import_event["reason"]
    # The landing is one commit, and the store stays clean after it.
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=dest_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert porcelain == ""


def test_imported_candidate_is_promotable_in_the_destination(source: Engine, dest: Engine) -> None:
    """The source's evidence is left in its own trail — the destination's
    lifecycle is unpoisoned: its own gate can promote the imported candidate."""
    _promote(source, "lr-1")
    apply_import(export_bundle(source, now=T0), dest, now=T0)
    promoted = dest.promote("lr-1", replay_evidence(["dest-run-1"]))
    assert promoted.status == "promoted"
    assert dest.promoted_records()[0].id == "lr-1"


def test_destination_clock_decay_governs_the_imported_survivor(
    source: Engine, dest: Engine, dest_root: Path
) -> None:
    """Destination-clock decay stamping, asserted in the audit trail: the
    destination's first decay pass anchors at the import moment, not at the
    source's belief age."""
    _promote(source, "lr-1")
    source_record = source.get("lr-1")
    apply_import(export_bundle(source, now=T0), dest, now=T0)
    decay_now = T0 + timedelta(days=30)
    decayed = dest.decay(now=decay_now)
    assert [record.id for record in decayed] == ["lr-1"]
    expected = decayed_confidence(
        source_record.confidence,
        since=T0,  # the import moment — not the source's created date
        now=decay_now,
        config=DecayConfig(),
    )
    imported = dest.get("lr-1")
    assert imported.confidence == pytest.approx(expected)
    # Source-clock anchoring would give a different (deeper-aged) number.
    source_anchor = decayed_confidence(
        source_record.confidence,
        since=source_record.confidence_updated_at or source_record.created,
        now=decay_now,
        config=DecayConfig(),
    )
    assert imported.confidence != pytest.approx(source_anchor)
    events = [
        json.loads(line) for line in (Path(dest_root) / "audit.jsonl").read_text().splitlines()
    ]
    decay_events = [event for event in events if event["event"] == "decay"]
    assert len(decay_events) == 1
    assert "half-life" in decay_events[0]["reason"]


def test_apply_import_refuses_conflicts_all_or_nothing(
    source: Engine, dest: Engine, dest_root: Path
) -> None:
    """A bundle where one record cannot land refuses whole — never partially
    applied, even when the other records are perfectly landable."""
    _promote(source, "lr-1", topic="429 backoff")
    _promote(source, "lr-2", topic="deadlock contention")
    dest.propose(make_record("lr-1", topic="unrelated local knowledge"))
    before = _fingerprint(dest_root)
    with pytest.raises(ImportRefused, match="nothing was applied"):
        apply_import(export_bundle(source, now=T0), dest, now=T0)
    assert _fingerprint(dest_root) == before


def test_apply_refuses_a_bundle_of_the_destination_own_identity(
    dest: Engine, dest_root: Path
) -> None:
    document = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "manifest": {
            "source_store": "beta",
            "exported_at": "2026-10-01T12:00:00Z",
            "records": [],
        },
    }
    bundle = read_bundle(_write_raw(Path(dest_root).parent / "self.json", document))
    with pytest.raises(ImportRefused, match="own identity"):
        apply_import(bundle, dest, now=T0)


def test_store_refuses_foreign_source_refs_and_collisions(dest: Engine) -> None:
    """The store primitive's own fail-closed contract, independent of bundles:
    it takes promoted-shape payloads only (every manifest payload is one)."""
    exported = make_record("lr-1").model_copy(update={"status": "promoted", "promoted_at": T0})
    with pytest.raises(StoreError, match="must match"):
        dest.import_candidate(exported, source_ref="not a store:lr-1")
    with pytest.raises(StoreError, match="names this store"):
        dest.import_candidate(exported, source_ref="beta:lr-1")
    with pytest.raises(StoreError, match="does not name this record"):
        dest.import_candidate(exported, source_ref="alpha:other")
    with pytest.raises(StoreError, match="is qualified"):
        dest.import_candidate(
            exported.model_copy(update={"id": "alpha:lr-1"}), source_ref="alpha:lr-1"
        )
    dest.propose(make_record("lr-1", topic="unrelated local knowledge"))
    with pytest.raises(StoreError, match="already exists"):
        dest.import_candidate(exported, source_ref="alpha:lr-1")


# ----------------------------------------------------------------- CLI


def test_cli_export_writes_the_bundle(source: Engine, source_root: Path) -> None:
    _promote(source, "lr-1")
    out = Path(source_root).parent / "cli-bundle.json"
    rc = main(
        ["memory", "export", "--store", "alpha", "--root", str(source_root), "--out", str(out)]
    )
    assert rc == 0
    bundle = read_bundle(out)
    assert bundle.manifest.source_store == "alpha"
    assert bundle.manifest.records[0].qualified_id == "alpha:lr-1"


def test_cli_import_dry_run_writes_nothing(
    source: Engine,
    source_root: Path,
    dest: Engine,
    dest_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI dry run leaves the destination store byte-identical — the
    Engine is constructed first so the fingerprint never attributes the
    store's own birth artifacts (review.lock, git init) to the import."""
    _promote(source, "lr-1")
    out = Path(source_root).parent / "cli-bundle.json"
    main(["memory", "export", "--store", "alpha", "--root", str(source_root), "--out", str(out)])
    before = _fingerprint(dest_root)
    rc = main(
        ["memory", "import", "--store", "beta", "--root", str(dest_root), "--bundle", str(out)]
    )
    assert rc == 0
    assert _fingerprint(dest_root) == before
    output = capsys.readouterr().out
    assert "dry run" in output
    assert "nothing written" in output


def test_cli_import_confirmed_lands_the_candidate(
    source: Engine,
    source_root: Path,
    dest: Engine,
    dest_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _promote(source, "lr-1")
    out = Path(source_root).parent / "cli-bundle.json"
    main(["memory", "export", "--store", "alpha", "--root", str(source_root), "--out", str(out)])
    rc = main(
        [
            "memory",
            "import",
            "--store",
            "beta",
            "--root",
            str(dest_root),
            "--bundle",
            str(out),
            "--yes",
        ]
    )
    assert rc == 0
    assert dest.get("lr-1").status == "candidate"
    assert "import complete" in capsys.readouterr().out


def test_cli_import_refuses_malformed_bundles_visibly(
    source: Engine,
    source_root: Path,
    dest: Engine,
    dest_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _promote(source, "lr-1")
    out = Path(source_root).parent / "cli-bundle.json"
    main(["memory", "export", "--store", "alpha", "--root", str(source_root), "--out", str(out)])
    broken = Path(source_root).parent / "broken.json"
    broken.write_text('{"schema_version": 1, "kind": "nope"}', encoding="utf-8")
    before = _fingerprint(dest_root)
    rc = main(
        ["memory", "import", "--store", "beta", "--root", str(dest_root), "--bundle", str(broken)]
    )
    assert rc == 1
    assert _fingerprint(dest_root) == before
    assert "error" in capsys.readouterr().err


def test_cli_refuses_unknown_store_names(
    dest_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "memory",
            "export",
            "--store",
            "Not A Name",
            "--root",
            str(dest_root),
            "--out",
            "x.json",
        ]
    )
    assert rc == 1
    assert "error" in capsys.readouterr().err
