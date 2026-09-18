"""Git-backed learning store: the engine's durable memory surface.

The :class:`Engine` owns the learning lifecycle (``ephemeral -> candidate ->
promoted`` or ``dropped``), versioned storage in git, and promotion-only
retrieval: swarm workers may retrieve only ``promoted`` records, keeping
unproven, agent-generated knowledge out of active use until the memory gate
proves it materially changes verdicts.

Persistence model: the store root is a dedicated directory that is its own
git repository — one JSON file per record under ``records/``, an append-only
JSONL audit trail at ``audit.jsonl``, and one commit per lifecycle
transition. Nesting inside an enclosing repo is safe: every git command runs
with ``cwd`` pinned to the root, so an enclosing index is never touched and
store commits can never sweep up unrelated files.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from evogine.engine.embedders import Embedder, HashingEmbedder
from evogine.engine.records import (
    DecayConfig,
    LearningRecord,
    MergedSource,
    RecordCounterfactual,
    decayed_confidence,
)
from evogine.engine.retrieval import DEFAULT_FLOOR
from evogine.engine.retrieval import retrieve as hybrid_retrieve

# Ids become filenames under ``records/``: lock them to filesystem- and
# git-safe characters so a malicious proposal can never traverse paths
# (the security review's staging-traversal lesson, applied at the boundary).
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# States that participate in the live lifecycle. ``dropped`` is a terminal
# rejection, ``merged`` a terminal absorption — neither merges, decays, nor
# retires further; their records stay readable for audit either way.
_LIVE_STATUSES = ("ephemeral", "candidate", "promoted")

class StoreError(Exception):
    """Lifecycle violation, unknown id, or malformed proposal."""


def _now() -> datetime:
    return datetime.now(UTC)


class AuditEvent(BaseModel):
    """One immutable lifecycle entry in the append-only audit trail."""

    seq: int = 0
    ts: datetime = Field(default_factory=_now)
    event: Literal[
        "propose", "queue_replay", "promote", "demote", "merge", "decay", "retire", "reinstate"
    ]
    record_id: str
    from_status: str | None = None
    to_status: str
    reason: str | None = None
    run_ids: list[str] = Field(default_factory=list)


def _git(root: Path, *args: str) -> None:
    """Run a git command inside the store repo.

    Identity and signing are pinned so commits never depend on ambient
    global git config.
    """
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evogine Engine",
            "-c",
            "user.email=engine@evogine.local",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise StoreError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")


class Engine:
    """Durable, git-backed store of :class:`LearningRecord` objects.

    Lifecycle contract (v1 design):
    ``propose`` stores a draft as ``ephemeral``; ``queue_replay`` attaches a
    pending counterfactual replay to a ``candidate``; ``promote`` marks a
    record ``promoted`` (with ``promoted_at`` stamped) only when the memory
    gate passes; ``demote`` moves a record to ``dropped`` — rejected knowledge
    stays in the store, auditable, never deleted.
    """

    def __init__(self, root: Path | str, embedder: Embedder | None = None) -> None:
        self.root = Path(root)
        self.embedder: Embedder = embedder if embedder is not None else HashingEmbedder()
        self.records_dir = self.root / "records"
        self.audit_path = self.root / "audit.jsonl"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            _git(self.root, "init", "-q")

    # ---------------------------------------------------------------- reads

    def promoted_records(self) -> list[LearningRecord]:
        """All ``promoted`` records — the only pool retrieval may draw from."""
        return [record for record in self._all_records() if record.status == "promoted"]

    def retrieve(
        self, task_context: str, k: int = 5, floor: float = DEFAULT_FLOOR
    ) -> list[LearningRecord]:
        """Return the top-``k`` promoted records relevant to ``task_context``.

        Hybrid retrieval (V2-3): promoted records are scored by embedding
        similarity and lexical overlap over their body fields, and anything
        below ``floor`` is dropped — with nothing above the floor the result
        is empty, honestly. Only ``promoted`` records are ever returned —
        warm runs must see exactly the knowledge the memory gate has proven.
        ``k <= 0`` yields an empty result (top-min(k, n) is empty).
        """
        return hybrid_retrieve(self, task_context, k=k, floor=floor, embedder=self.embedder)

    def get(self, learning_id: str) -> LearningRecord:
        """Return one record by id in whatever lifecycle state it holds.

        Audit-grade read: unlike :meth:`retrieve`, it sees ephemeral,
        candidate, and dropped records too — rejected knowledge stays
        readable, never deleted.
        """
        return self._load(learning_id)

    def candidates(self) -> list[LearningRecord]:
        """Return every record queued for replay (status ``candidate``).

        Read-only view for the promotion pass: the memory gate reviews
        exactly these and nothing else. Ordered by record id (records/
        glob order) so a pass is deterministic.
        """
        return [record for record in self._all_records() if record.status == "candidate"]

    # ------------------------------------------------------------ lifecycle

    def propose(self, draft: LearningRecord) -> LearningRecord:
        """Store ``draft`` as a new ``ephemeral`` record and return it.

        Accepts agent-proposed drafts (reflector output) and human-authored
        seeds alike; provenance travels with the record. The lifecycle is
        store-owned: drafts arrive unproven, so anything else is refused.
        """
        if draft.status != "ephemeral":
            raise StoreError(f"propose stores ephemeral records; draft {draft.id} is {draft.status}")
        counterfactual = draft.counterfactual
        if (
            counterfactual.replay_run_ids
            or counterfactual.verdict_change is not None
            or counterfactual.evidence is not None
        ):
            raise StoreError(
                f"draft {draft.id} carries replay evidence; only the lifecycle may attach it"
            )
        if draft.promoted_at is not None:
            raise StoreError(f"draft {draft.id} is pre-stamped promoted_at; only promote stamps it")
        if not _ID_PATTERN.fullmatch(draft.id):
            raise StoreError(f"record id {draft.id!r} must match {_ID_PATTERN.pattern}")
        if self._record_path(draft.id).exists():
            raise StoreError(f"record already exists: {draft.id}")
        record = draft.model_copy(update={"status": "ephemeral"})
        self._save(record)
        self._append_audit(AuditEvent(event="propose", record_id=record.id, to_status="ephemeral"))
        self._commit(f"propose: {record.id}")
        return record

    def queue_replay(self, learning_id: str, run_ids: list[str]) -> None:
        """Attach a pending counterfactual replay (``run_ids``) to a record.

        The replay compares cold and warm verdicts on the recorded failure
        cases; results feed the memory gate via :meth:`promote`.
        """
        if not run_ids:
            raise StoreError("queue_replay requires at least one replay run id")
        record = self._load(learning_id)
        if record.status != "ephemeral":
            raise StoreError(
                f"queue_replay requires an ephemeral record; {learning_id} is {record.status}"
            )
        queued = record.model_copy(
            update={
                "status": "candidate",
                "counterfactual": record.counterfactual.model_copy(
                    update={"replay_run_ids": list(run_ids)}
                ),
            }
        )
        self._save(queued)
        self._append_audit(
            AuditEvent(
                event="queue_replay",
                record_id=learning_id,
                from_status="ephemeral",
                to_status="candidate",
                run_ids=list(run_ids),
            )
        )
        self._commit(f"queue_replay: {learning_id} ({len(run_ids)} runs)")

    def promote(self, learning_id: str, replay_evidence: RecordCounterfactual) -> LearningRecord:
        """Promote ``learning_id`` to ``promoted`` given passing replay evidence.

        Called only after the memory gate accepts ``replay_evidence`` —
        a material verdict change on the queued counterfactual runs. The
        store re-checks the invariants the gate is supposed to enforce:
        promotion from ``candidate`` only, evidence naming every queued run,
        and a stamped verdict change. ``promoted_at`` is stamped here.
        """
        record = self._load(learning_id)
        if record.status != "candidate":
            raise StoreError(
                f"promote requires a candidate record; {learning_id} is {record.status}"
            )
        evidence = RecordCounterfactual.model_validate(replay_evidence)
        if not evidence.replay_run_ids:
            raise StoreError(
                f"promotion of {learning_id} requires evidence naming at least one replay run"
            )
        if evidence.verdict_change is not True:
            raise StoreError(f"promotion of {learning_id} requires a material verdict change")
        missing = set(record.counterfactual.replay_run_ids) - set(evidence.replay_run_ids)
        if missing:
            raise StoreError(f"promotion evidence omits queued replay runs: {sorted(missing)}")
        promoted = record.model_copy(
            update={
                "status": "promoted",
                "counterfactual": evidence,
                "promoted_at": _now(),
            }
        )
        self._save(promoted)
        self._append_audit(
            AuditEvent(
                event="promote",
                record_id=learning_id,
                from_status="candidate",
                to_status="promoted",
                run_ids=list(evidence.replay_run_ids),
            )
        )
        self._commit(f"promote: {learning_id}")
        return promoted

    def demote(self, learning_id: str, reason: str) -> LearningRecord:
        """Move ``learning_id`` to ``dropped`` with ``reason`` recorded.

        Dropped records remain in the store for audit; the engine never
        silently forgets what it once believed. Allowed from any live state
        (``ephemeral``/``candidate``/``promoted``): rejections and retractions
        alike stay visible in the store's history.
        """
        if not reason.strip():
            raise StoreError("demote requires a reason")
        record = self._load(learning_id)
        if record.status == "dropped":
            raise StoreError(f"record already dropped: {learning_id}")
        dropped = record.model_copy(update={"status": "dropped"})
        self._save(dropped)
        self._append_audit(
            AuditEvent(
                event="demote",
                record_id=learning_id,
                from_status=record.status,
                to_status="dropped",
                reason=reason,
            )
        )
        self._commit(f"demote: {learning_id}")
        return dropped

    # ------------------------------------------------------- v2: merge

    def find_duplicates(self) -> list[tuple[str, str]]:
        """Return pairs of live records that are exact duplicates.

        Deliberately deterministic record-level detection: two records are
        duplicates when they share a ``type`` and all four :class:`RecordBody`
        fields are identical. Title-only differences still merge — the body
        carries the knowledge, the title is presentation. Paraphrase-level
        detection belongs to the retrieval layer (V2-3), not to a
        guess-prone heuristic here. Pairs come back ordered by id.
        """
        live = [r for r in self._all_records() if r.status in _LIVE_STATUSES]
        duplicates: list[tuple[str, str]] = []
        for i, first in enumerate(live):
            for second in live[i + 1 :]:
                if first.type == second.type and first.body == second.body:
                    duplicates.append((first.id, second.id))
        return duplicates

    def merge(self, primary_id: str, duplicate_id: str, reason: str) -> LearningRecord:
        """Consolidate ``duplicate_id`` into ``primary_id`` (exact duplicates).

        Record-level consolidation: the survivor keeps its own body and
        provenance verbatim — nothing is summarized or rewritten — and the
        absorbed record's provenance is preserved in full on the survivor's
        ``merged_sources``. Corroboration can only strengthen the survivor:
        confidence becomes the max of the two, stamped into
        ``confidence_updated_at`` so a later decay pass anchors on the merge
        moment instead of double-decaying the boost. The duplicate moves to
        ``merged`` with ``merged_into`` naming the survivor; its record file
        stays in the store, readable, never deleted.
        """
        if not reason.strip():
            raise StoreError("merge requires a reason")
        if primary_id == duplicate_id:
            raise StoreError("cannot merge a record into itself")
        primary = self._load(primary_id)
        duplicate = self._load(duplicate_id)
        if primary.status not in _LIVE_STATUSES:
            raise StoreError(f"merge requires a live primary; {primary_id} is {primary.status}")
        if duplicate.status not in _LIVE_STATUSES:
            raise StoreError(
                f"merge requires a live duplicate; {duplicate_id} is {duplicate.status}"
            )
        if primary.type != duplicate.type or primary.body != duplicate.body:
            raise StoreError(
                f"merge requires duplicates (same type, identical body); "
                f"{primary_id} and {duplicate_id} differ"
            )
        merged_at = _now()
        survivor = primary.model_copy(
            update={
                "confidence": max(primary.confidence, duplicate.confidence),
                "confidence_updated_at": merged_at,
                "merged_sources": [
                    *primary.merged_sources,
                    MergedSource(
                        record_id=duplicate.id,
                        title=duplicate.title,
                        provenance=duplicate.provenance,
                        confidence=duplicate.confidence,
                        merged_at=merged_at,
                    ),
                    # A duplicate that had itself absorbed records hands
                    # their provenance to the survivor: lineage transfer,
                    # verbatim — never a summary of a summary.
                    *duplicate.merged_sources,
                ],
            }
        )
        absorbed = duplicate.model_copy(update={"status": "merged", "merged_into": primary_id})
        self._save(survivor)
        self._save(absorbed)
        self._append_audit(
            AuditEvent(
                event="merge",
                record_id=duplicate_id,
                from_status=duplicate.status,
                to_status="merged",
                reason=f"merged into {primary_id}: {reason}",
            )
        )
        self._commit(f"merge: {duplicate_id} into {primary_id}")
        return survivor

    # ------------------------------------------------------- v2: decay

    def decay(
        self,
        config: DecayConfig | None = None,
        now: datetime | None = None,
    ) -> list[LearningRecord]:
        """Apply configured confidence decay to every live record.

        Confidence decays on the configured exponential half-life curve
        with an asymptotic floor, anchored on each record's
        ``confidence_updated_at`` (or ``created`` for records that never
        carried it) so passes compose exactly and a re-run at the same
        instant is a no-op. Records flagged ``evergreen`` are exempt:
        safety-critical knowledge is never aged toward the floor, so it
        can never decay below retrieval consideration. Terminal states
        (``dropped``/``merged``) are frozen — they are history, not live
        belief. Returns the records whose confidence actually moved; a
        pass that changes nothing writes no audit events and no empty
        commit.
        """
        curve = config or DecayConfig()
        moment = now or _now()
        decayed: list[LearningRecord] = []
        for record in self._all_records():
            if record.status not in _LIVE_STATUSES or record.evergreen:
                continue
            anchor = record.confidence_updated_at or record.created
            new_confidence = decayed_confidence(
                record.confidence, since=anchor, now=moment, config=curve
            )
            if new_confidence >= record.confidence:
                continue
            updated = record.model_copy(
                update={"confidence": new_confidence, "confidence_updated_at": moment}
            )
            self._save(updated)
            decayed.append(updated)
            self._append_audit(
                AuditEvent(
                    event="decay",
                    record_id=record.id,
                    from_status=record.status,
                    to_status=record.status,
                    reason=(
                        f"confidence {record.confidence:.4f} -> {new_confidence:.4f} "
                        f"(half-life {curve.half_life_days}d, floor {curve.floor})"
                    ),
                )
            )
        if decayed:
            self._commit(f"decay: {len(decayed)} record(s) aged past the curve")
        return decayed

    # ------------------------------------------------------- v2: retire

    def retire(self, learning_id: str, reason: str) -> LearningRecord:
        """Move ``learning_id`` to the reversible ``retired`` state.

        Staleness is not rejection: a retired record leaves retrieval
        (promotion-only, and ``retired`` is not ``promoted``) but stays on
        disk, readable, never deleted. ``retired_from`` remembers the prior
        status and ``retired_at`` stamps the moment, so :meth:`reinstate`
        can restore the record exactly. Live states only — ``dropped`` is
        already a terminal rejection and ``merged`` a terminal absorption;
        neither needs a second shelf.
        """
        if not reason.strip():
            raise StoreError("retire requires a reason")
        record = self._load(learning_id)
        if record.status not in _LIVE_STATUSES:
            raise StoreError(f"retire requires a live record; {learning_id} is {record.status}")
        retired = record.model_copy(
            update={"status": "retired", "retired_at": _now(), "retired_from": record.status}
        )
        self._save(retired)
        self._append_audit(
            AuditEvent(
                event="retire",
                record_id=learning_id,
                from_status=record.status,
                to_status="retired",
                reason=reason,
            )
        )
        self._commit(f"retire: {learning_id}")
        return retired

    def reinstate(self, learning_id: str) -> LearningRecord:
        """Restore a retired record to exactly its pre-retirement state.

        The reversibility half of the retirement contract: ``retired_from``
        is restored verbatim and the retirement stamps are cleared, so a
        reinstated record equals its pre-retirement self field-for-field.
        The audit trail keeps both directions — history is never rewritten.
        """
        record = self._load(learning_id)
        if record.status != "retired":
            raise StoreError(
                f"reinstate requires a retired record; {learning_id} is {record.status}"
            )
        restored = record.model_copy(
            update={
                "status": record.retired_from,
                "retired_at": None,
                "retired_from": None,
            }
        )
        self._save(restored)
        self._append_audit(
            AuditEvent(
                event="reinstate",
                record_id=learning_id,
                from_status="retired",
                to_status=restored.status,
                reason=f"reinstated from retirement (retired_from={record.retired_from})",
            )
        )
        self._commit(f"reinstate: {learning_id}")
        return restored

    # ----------------------------------------------------------- persistence

    def _record_path(self, learning_id: str) -> Path:
        return self.records_dir / f"{learning_id}.json"

    def _load(self, learning_id: str) -> LearningRecord:
        path = self._record_path(learning_id)
        if not path.exists():
            raise StoreError(f"unknown learning id: {learning_id}")
        return LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, record: LearningRecord) -> None:
        """Atomic write (tmp + rename) so a crash never leaves a torn record."""
        path = self._record_path(record.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _append_audit(self, event: AuditEvent) -> None:
        """Append one event as a new JSONL line.

        The audit trail is only ever appended to, never rewritten; ``seq``
        is derived from the live line count so the file itself stays the
        single source of truth, even across engine restarts.
        """
        seq = 0
        if self.audit_path.exists():
            with self.audit_path.open("r", encoding="utf-8") as fh:
                seq = sum(1 for line in fh if line.strip())
        stamped = event.model_copy(update={"seq": seq + 1, "ts": _now()})
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(stamped.model_dump_json() + "\n")

    def _all_records(self) -> list[LearningRecord]:
        return [
            LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def _commit(self, message: str) -> None:
        _git(self.root, "add", "-A", ".")
        _git(self.root, "commit", "-q", "-m", message)
