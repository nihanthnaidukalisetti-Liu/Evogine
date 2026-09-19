"""Property tests: provenance travels verbatim across the import boundary.

The V4 import/export flow's governance properties, checked structurally
over generated stores and random lineages, against independent
restatements of the rules (docs/cross-project-memory.md §3):

- **Provenance verbatim** — the export bundle's manifest carries every
  record's full source provenance byte-for-byte, and the only
  transformation ever applied is qualifying source-local merged-source
  references (invariant 3.4); no provenance payload is ever rewritten,
  summarized, or dropped.
- **Round-trip delta** — export then confirmed import lands a record
  identical to its source except exactly the destination's own lifecycle
  decisions: candidate status, blank counterfactual, cleared
  ``promoted_at``, destination-clock ``confidence_updated_at``.
- **§3.2 promotion-only, per scope** — an imported record enters the
  destination's pool as a candidate and cannot surface in its promoted
  retrieval pool until the destination's own gate promotes it.
- **§3.3 per-store audit, no invented global ordering** — the source
  trail is untouched by export; the destination's trail grows by append
  on its own contiguous clock; import events reference the source only
  by qualified id, never by sequence or timestamp stitching.
- **§3.4 qualified ids across the boundary** — every merged-source
  reference persisted into the destination is qualified and never
  self-qualified; malformed bundles (any single-field corruption) are
  refused with the destination untouched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import make_record, replay_evidence
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.engine.records import DecayConfig, LearningRecord, decayed_confidence
from evogine.engine.store import Engine
from evogine.engine.transfer import (
    ImportRefused,
    apply_import,
    export_bundle,
    read_bundle,
    write_bundle,
)

TOPICS = ["429 backoff", "deadlock contention", "cache stampede"]

# The fixed destination stamping moment every property imports at.
IMPORT_AT = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

# The exact field delta a round trip is allowed to produce: everything
# else must be byte-identical to the source record.
ROUND_TRIP_DELTA = ("status", "counterfactual", "promoted_at", "confidence_updated_at")

# Single-field corruptions a valid bundle can suffer; each must refuse.
CORRUPTIONS = [
    lambda d: d.update(schema_version=2),
    lambda d: d.update(kind="not-a-bundle"),
    lambda d: d["manifest"].update(source_store="Gamma"),
    lambda d: d["manifest"].update(source_store="not a name"),
    lambda d: d["manifest"]["records"][0].update(qualified_id="unqualified"),
    lambda d: d["manifest"]["records"][0].update(qualified_id="elsewhere:lr-0"),
    lambda d: d["manifest"]["records"][0]["record"].update(status="dropped"),
    lambda d: d["manifest"]["records"][0]["record"]["provenance"].update(proposed_by=""),
    lambda d: d["manifest"]["records"][0]["record"].update(
        merged_sources=[
            {
                "record_id": "twin",
                "title": "unqualified",
                "provenance": {
                    "proposed_by": "r",
                    "origin": "o",
                    "source": "s",
                    "note": "",
                },
                "confidence": 0.5,
                "merged_at": "2026-10-01T00:00:00Z",
            }
        ]
    ),
]


def _promote(engine: Engine, record_id: str, topic: str = "429 backoff") -> None:
    """Propose, queue, and promote one record through the store's lifecycle."""
    engine.propose(make_record(record_id, topic=topic))
    engine.queue_replay(record_id, [f"run-{record_id}"])
    engine.promote(record_id, replay_evidence([f"run-{record_id}"]))


def _fingerprint(root: Path) -> tuple[int, int]:
    """(record file count, audit line count) — a refused write changes neither."""
    records = list((root / "records").glob("*.json"))
    audit = root / "audit.jsonl"
    lines = audit.read_text(encoding="utf-8").splitlines() if audit.exists() else []
    return len(records), len(lines)


def _fresh_federation(source_name: str, dest_name: str) -> tuple[Path, Engine, Engine]:
    """A source and a destination store, freshly git-initialized under one parent."""
    parent = Path(tempfile.mkdtemp(prefix="evogine-transfer-prop-"))
    source = Engine(parent / source_name, name=source_name)
    dest = Engine(parent / dest_name, name=dest_name)
    return parent, source, dest


def _payload(record: LearningRecord) -> dict[str, object]:
    """Canonical JSON payload — the form provenance is compared byte-for-byte."""
    return json.loads(record.model_dump_json())


def _lineage_digest(record: LearningRecord) -> list[dict[str, object]]:
    """The provenance-bearing content of a record's merged lineage."""
    return [
        {
            "title": source.title,
            "provenance": json.loads(source.provenance.model_dump_json()),
            "confidence": source.confidence,
            "merged_at": source.merged_at.isoformat(),
        }
        for source in record.merged_sources
    ]


@settings(max_examples=15, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from(TOPICS), st.integers(min_value=0, max_value=2)),
        min_size=1,
        max_size=4,
    ),
    st.sampled_from([("alpha", "beta"), ("proje.2", "other-1")]),
)
def test_export_and_import_preserve_provenance_verbatim(
    plans: list[tuple[str, int]], store_names: tuple[str, str]
) -> None:
    """Whatever the store's promoted pool looks like, the bundle's manifest
    carries it verbatim and the round trip lands it with the exact delta."""
    source_name, dest_name = store_names
    parent, source, dest = _fresh_federation(source_name, dest_name)
    try:
        # Build a promoted pool with unique bodies and real merge lineage:
        # any later plan repeating the first topic merges into lr-0.
        for index, (topic, variant) in enumerate(plans):
            record_id = f"lr-{index}"
            source.propose(make_record(record_id, topic=f"{topic} variant{variant}"))
            source.queue_replay(record_id, [f"run-{record_id}"])
            source.promote(record_id, replay_evidence([f"run-{record_id}"]))
        merged: list[tuple[str, int]] = []
        for index in range(1, len(plans)):
            # merge() demands identical bodies: only an exact plan repeat
            # (same topic AND variant) produces a mergeable twin.
            if plans[index] == plans[0] and plans[index] not in merged:
                source.merge("lr-0", f"lr-{index}", "exact duplicate across plans")
                merged.append(plans[index])
        source_records = {record.id: record for record in source.promoted_records()}
        assert source_records, "the plan must promote at least one record"

        bundle = export_bundle(source, now=IMPORT_AT)

        # --- provenance verbatim in the manifest ---------------------------
        for entry in bundle.manifest.records:
            source_record = source_records[entry.record.id]
            assert entry.qualified_id == f"{source_name}:{entry.record.id}"
            exported = _payload(entry.record)
            original = _payload(source_record)
            exported_lineage = exported.pop("merged_sources")
            original_lineage = original.pop("merged_sources")
            # Everything outside the lineage is byte-identical.
            assert exported == original
            # Inside the lineage: content verbatim, references only qualified.
            assert _lineage_digest(entry.record) == _lineage_digest(source_record)
            for exported_ref, original_ref in zip(exported_lineage, original_lineage, strict=True):
                if exported_ref["record_id"] == original_ref["record_id"]:
                    assert ":" in original_ref["record_id"]  # was already qualified
                else:
                    assert exported_ref["record_id"] == (
                        f"{source_name}:{original_ref['record_id']}"
                    )

        # --- round trip: apply and compare the exact delta -----------------
        path = write_bundle(bundle, parent / "bundle.json")
        landed = apply_import(read_bundle(path), dest, now=IMPORT_AT)
        assert len(landed) == len(bundle.manifest.records)
        for entry, landed_record in zip(bundle.manifest.records, landed, strict=True):
            imported = dest.get(entry.record.id)
            exported_payload = _payload(entry.record)
            landed_payload = _payload(imported)
            for field in ROUND_TRIP_DELTA:
                exported_payload.pop(field)
                landed_payload.pop(field)
            # References were already qualified in the bundle, so the
            # landed payload equals the bundle payload exactly.
            assert landed_payload == exported_payload
            assert imported.status == "candidate"
            assert imported.promoted_at is None
            assert imported.confidence_updated_at == IMPORT_AT
            assert imported.counterfactual.replay_run_ids == []
            assert landed_record.id == entry.record.id

        # --- §3.2: promotion-only, per scope -------------------------------
        assert dest.promoted_records() == []  # import never promotes
        assert all(record.status == "candidate" for record in dest.candidates())

        # --- §3.3: per-store trails, no invented global order ---------------
        source_audit = (source.root / "audit.jsonl").read_text().splitlines()
        dest_events = [
            json.loads(line) for line in (dest.root / "audit.jsonl").read_text().splitlines()
        ]
        # Export left the source trail exactly as the lifecycle left it:
        # three events per promotion, one per merge, nothing else.
        assert len(source_audit) == 3 * len(plans) + len(merged)
        # The destination trail is its own contiguous clock, one import
        # event per landed row, each referencing the source by qualified id.
        assert [event["seq"] for event in dest_events] == list(range(1, len(dest_events) + 1))
        assert len(dest_events) == len(landed)
        for event in dest_events:
            assert event["event"] == "import"
            assert event["reason"].startswith(f"imported from {source_name}:")

        # --- §3.4: persisted cross-scope references stay qualified ---------
        for record in dest._all_records():
            for source_ref in record.merged_sources:
                assert ":" in source_ref.record_id
                assert not source_ref.record_id.startswith(f"{dest_name}:")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


@settings(max_examples=15, deadline=None)
@given(st.sampled_from(CORRUPTIONS))
def test_any_single_field_corruption_is_refused_with_nothing_written(
    mutate: object,
) -> None:
    """Every corruption of a valid bundle refuses at read time, and the
    destination store is untouched — malformed bundles never partially apply."""
    parent, source, dest = _fresh_federation("alpha", "beta")
    try:
        _promote(source, "lr-0")
        document = json.loads(export_bundle(source, now=IMPORT_AT).model_dump_json())
        mutate(document)  # type: ignore[operator]
        before = _fingerprint(dest.root)
        bundle_path = parent / "corrupt.json"
        bundle_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        with pytest.raises(ImportRefused):
            read_bundle(bundle_path)
        assert _fingerprint(dest.root) == before
    finally:
        shutil.rmtree(parent, ignore_errors=True)


@settings(max_examples=15, deadline=None)
@given(
    st.sampled_from(TOPICS),
    st.integers(min_value=1, max_value=45),
)
def test_destination_clock_decay_anchors_at_the_import_moment(topic: str, age_days: int) -> None:
    """The destination's decay pass on an imported record computes from the
    import moment — never from the source's belief age — and the audit trail
    shows the destination's own decay event."""
    parent, source, dest = _fresh_federation("alpha", "beta")
    try:
        _promote(source, "lr-0", topic=topic)
        source_record = source.get("lr-0")
        landed = apply_import(export_bundle(source, now=IMPORT_AT), dest, now=IMPORT_AT)[0]
        decay_now = IMPORT_AT + timedelta(days=age_days)
        decayed = dest.decay(now=decay_now)
        assert [record.id for record in decayed] == ["lr-0"]
        expected = decayed_confidence(
            source_record.confidence, since=IMPORT_AT, now=decay_now, config=DecayConfig()
        )
        assert landed.confidence == source_record.confidence  # verbatim on landing
        assert dest.get("lr-0").confidence == pytest.approx(expected)
        events = [json.loads(line) for line in (dest.root / "audit.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["import", "decay"]
        assert IMPORT_AT.isoformat() in events[0]["reason"]
        assert "half-life" in events[1]["reason"]
        # The source store: untouched since its own promotion.
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source.root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert porcelain == ""
    finally:
        shutil.rmtree(parent, ignore_errors=True)
