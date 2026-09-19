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

import fcntl
import logging
import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
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
from evogine.ops.compaction import load_index_if_fresh
from evogine.ops.logging_setup import log_event
from evogine.ops.metrics import NO_STATUS, STORE_TRANSITIONS_TOTAL, metrics
from evogine.ops.migrations import (
    CURRENT_SCHEMA_VERSION,
    SchemaError,
    SchemaState,
    ensure_openable,
    write_schema,
)

# Ids become filenames under ``records/``: lock them to filesystem- and
# git-safe characters so a malicious proposal can never traverse paths
# (the security review's staging-traversal lesson, applied at the boundary).
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Store identity (V4 federation): a store's identity is its configured
# name after normalization (trimmed, lowercase), locked to the same
# filesystem-safe grammar lowercased. ``:`` is excluded on both sides so
# a qualified id ``<store-name>:<record-id>`` splits unambiguously on the
# first colon (docs/cross-project-memory.md, invariant 3.4).
_STORE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

#: Library-side logger: the engine only emits; formatting/level/sink belong
#: to the config module's settings via the ops logging entry point.
logger = logging.getLogger(__name__)

#: Identity of a standalone store that was never given a name. Federation
#: registries always pass an explicit name; the default keeps the
#: single-store engine (CLI, console, swarm) exactly as it was.
DEFAULT_STORE_NAME = "local"

#: The advisory lock file at the store root. The store's own git repo tracks
#: it from the first lifecycle commit, so no runtime file is ever left
#: untracked (the porcelain-clean property the audit tests pin); ``flock``
#: needs only the file to exist — its content is never read or written.
LOCK_FILENAME = "review.lock"


@contextmanager
def store_lock(root: Path | str) -> Iterator[None]:
    """Hold the store's exclusive advisory review lock for a critical window.

    ``approve_proposal``/``reject_proposal`` hold it across their whole
    pending-check -> commit window, so a second concurrent decision — from
    the CLI or any future transport — blocks, then fails its own re-check
    of pending status and refuses cleanly instead of corrupting the store.
    The documented contract stays "one writer at a time per store root";
    the lock makes the contract enforced instead of assumed.

    ``flock`` is advisory and crash-safe: the kernel drops the lock when the
    holding file description closes, so a killed worker cannot strand the
    store. Never nest: re-acquiring inside the window deadlocks, because the
    lock is held per open file description.
    """
    lock_path = Path(root) / LOCK_FILENAME
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# States that participate in the live lifecycle. ``dropped`` is a terminal
# rejection, ``merged`` a terminal absorption — neither merges, decays, nor
# retires further; their records stay readable for audit either way.
_LIVE_STATUSES = ("ephemeral", "candidate", "promoted")


class StoreError(Exception):
    """Lifecycle violation, unknown id, or malformed proposal."""


def _now() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------- v4: store identity


def normalize_store_name(name: str) -> str:
    """Normalize a configured store name to its identity: trimmed, lowercase.

    A store identity *is* its configured name after normalization (the V4
    pinned contract): ``"Alpha "`` and ``"alpha"`` are the same store, so
    references and dispatch can never drift on case or whitespace. Names
    outside the safe grammar after normalization are refused before any
    store is ever constructed on them.
    """
    normalized = name.strip().lower()
    if not _STORE_NAME_PATTERN.fullmatch(normalized):
        raise StoreError(
            f"store name {name!r} must match {_STORE_NAME_PATTERN.pattern} after normalization"
        )
    return normalized


def split_qualified_id(qualified_id: str) -> tuple[str, str]:
    """Split a qualified record id ``<store-name>:<record-id>`` into its parts.

    Splits on the first ``:`` — store names and record ids are both
    grammar-locked to exclude it, so the split is unambiguous. The store
    part is normalized (identities are lowercase); the record part must
    match the same id grammar the store enforces for its own records.
    Raises :class:`StoreError` when the id is not qualified at all or
    either half is malformed: an unparseable qualified id is never
    silently treated as a local id.
    """
    store_name, separator, record_id = qualified_id.partition(":")
    if not separator:
        raise StoreError(
            f"not a qualified id (expected <store-name>:<record-id>): {qualified_id!r}"
        )
    normalized = normalize_store_name(store_name)
    if not _ID_PATTERN.fullmatch(record_id):
        raise StoreError(f"record id in {qualified_id!r} must match {_ID_PATTERN.pattern}")
    return normalized, record_id


class AuditEvent(BaseModel):
    """One immutable lifecycle entry in the append-only audit trail."""

    seq: int = 0
    ts: datetime = Field(default_factory=_now)
    event: Literal[
        "propose",
        "queue_replay",
        "promote",
        "demote",
        "merge",
        "decay",
        "retire",
        "reinstate",
        "import",
    ]
    record_id: str
    from_status: str | None = None
    to_status: str
    reason: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    pass_id: str | None = None
    """V3 pass identity: stamped on audit events a promotion pass caused
    (its knowledge-stage demotions), ``None`` for every other event and
    for lines predating the field — the trail stays readable unchanged."""


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

    def __init__(
        self,
        root: Path | str,
        embedder: Embedder | None = None,
        *,
        name: str = DEFAULT_STORE_NAME,
    ) -> None:
        # The store's federation identity (V4): validated and normalized
        # before anything touches the filesystem — a store can never exist
        # under an ambiguous name.
        self.store_name = normalize_store_name(name)
        self.root = Path(root)
        self.embedder: Embedder = embedder if embedder is not None else HashingEmbedder()
        self.records_dir = self.root / "records"
        self.audit_path = self.root / "audit.jsonl"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        fresh_store = not (self.root / ".git").exists()
        if fresh_store:
            _git(self.root, "init", "-q")
        # Schema gate (V4 ops): a store written by NEWER code is refused
        # here — reading a schema this build does not understand is how
        # silent corruption happens; refusing is the visible alternative.
        # Legacy stores (predating the stamp) read as version 1 and keep
        # working unchanged; only fresh stores write a stamp, born current.
        try:
            ensure_openable(self.root)
        except SchemaError as exc:
            raise StoreError(str(exc)) from exc
        if fresh_store:
            write_schema(
                self.root,
                SchemaState(version=CURRENT_SCHEMA_VERSION, updated_at=_now()),
            )
        # The review lock file exists from store birth so the first lifecycle
        # commit versions it — a lock file created lazily could be left
        # untracked by a refused boundary action (no commit follows a refusal).
        self.lock_path = self.root / LOCK_FILENAME
        self.lock_path.touch()

    # ---------------------------------------------------------------- reads

    def _records_with_status(self, status: str) -> list[LearningRecord] | None:
        """Records with one status via the derived index, or ``None`` when the
        index is absent/stale (the caller falls back to the honest full
        scan). Derived state never changes results — only how many files
        must be parsed to get them.
        """
        store_index = load_index_if_fresh(self.root)
        if store_index is None:
            return None
        return [self._load(entry.id) for entry in store_index.entries if entry.status == status]

    def promoted_records(self) -> list[LearningRecord]:
        """All ``promoted`` records — the only pool retrieval may draw from."""
        promoted = self._records_with_status("promoted")
        if promoted is not None:
            return promoted
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
        candidates = self._records_with_status("candidate")
        if candidates is not None:
            return candidates
        return [record for record in self._all_records() if record.status == "candidate"]

    # ------------------------------------------------------------ lifecycle

    def propose(self, draft: LearningRecord) -> LearningRecord:
        """Store ``draft`` as a new ``ephemeral`` record and return it.

        Accepts agent-proposed drafts (reflector output) and human-authored
        seeds alike; provenance travels with the record. The lifecycle is
        store-owned: drafts arrive unproven, so anything else is refused.
        """
        if draft.status != "ephemeral":
            raise StoreError(
                f"propose stores ephemeral records; draft {draft.id} is {draft.status}"
            )
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
        if ":" in draft.id:
            raise StoreError(
                f"record id {draft.id!r} is qualified; a store's own record ids are "
                "unqualified — the <store-name>:<record-id> form is reserved for "
                "cross-scope references"
            )
        if not _ID_PATTERN.fullmatch(draft.id):
            raise StoreError(f"record id {draft.id!r} must match {_ID_PATTERN.pattern}")
        if self._record_path(draft.id).exists():
            raise StoreError(f"record already exists: {draft.id}")
        self._validate_cross_references(draft)
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

        Called only after a human approves the promotion proposal the memory
        gate rendered (V2-6) — the applied evidence is the gate's cleared
        trail: a material verdict change on the queued counterfactual runs.
        The store re-checks the invariants the gate is supposed to enforce:
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

    def demote(
        self, learning_id: str, reason: str, *, pass_id: str | None = None
    ) -> LearningRecord:
        """Move ``learning_id`` to ``dropped`` with ``reason`` recorded.

        Dropped records remain in the store for audit; the engine never
        silently forgets what it once believed. Allowed from any live state
        (``ephemeral``/``candidate``/``promoted``): rejections and retractions
        alike stay visible in the store's history. ``pass_id`` stamps the
        audit event when a promotion pass caused the demotion (V3) — the
        trail shows which pass dropped the record, not just that one did.
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
                pass_id=pass_id,
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
        store_index = load_index_if_fresh(self.root)
        if store_index is not None:
            # Index fast path: body equality is the four body fields, so
            # pairwise tuple comparison over index entries is exactly the
            # record-level comparison — with zero record files parsed.
            # Entries are id-ordered, matching the full-scan pairing order.
            indexed = [
                (
                    entry.id,
                    entry.type,
                    entry.signal,
                    entry.failure_mode,
                    entry.detection,
                    entry.fix_strategy,
                )
                for entry in store_index.entries
                if entry.status in _LIVE_STATUSES
            ]
            duplicates = [
                (first[0], second[0])
                for i, first in enumerate(indexed)
                for second in indexed[i + 1 :]
                if first[1:] == second[1:]
            ]
        else:
            live = [r for r in self._all_records() if r.status in _LIVE_STATUSES]
            duplicates = []
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

    # --------------------------------------------------- v4: import (cross-scope)

    def import_candidate(
        self,
        record: LearningRecord,
        *,
        source_ref: str,
        now: datetime | None = None,
    ) -> LearningRecord:
        """Land a cross-scope imported record as a new ``candidate`` (V4).

        The one write path that may move a row into a local pool from
        outside it — called only from the explicit, human-confirmed import
        flow (:mod:`evogine.engine.transfer`), never by retrieval-adjacent
        code, so promotion-only per scope (governance §3.2) survives: the
        record's status here is ``candidate``, and only this store's own
        gate and approval boundary can ever promote it. The source's
        replay evidence is scope-bound and is not carried in as if it were
        this store's pending evidence — the landed record's counterfactual
        is blank (``promote`` re-checks its own invariants on whatever this
        store queues), while the evidence itself stays in the source trail,
        referenced by ``source_ref``.

        Destination-clock decay stamping: ``confidence_updated_at`` is
        stamped to ``now`` (this store's clock), so the destination's first
        decay pass anchors at the import moment instead of re-aging the
        belief across the source's timeline — the destination takes the
        confidence value as of import and starts its own clock there.

        ``source_ref`` is the qualified source id ``<store>:<record-id>`` —
        the only cross-store reference the audit trail records (§3.3/§3.4):
        no source sequence numbers, no stitched global order. Refuses
        fail-closed: qualified source refs, unqualified local id free of
        collisions here, provenance-bearing payload in the exporter's
        ``promoted`` shape, and cross-references that survive this store's
        write-time validation. Refusals persist nothing.
        """
        if record.status != "promoted":
            raise StoreError(
                f"import_candidate lands exported (promoted) records; {record.id} is "
                f"{record.status}"
            )
        if record.merged_into is not None or record.retired_at is not None:
            raise StoreError(
                f"import_candidate refuses lifecycle stamps from another store: {record.id} "
                "carries merged/retired state — export bundles carry live promoted rows only"
            )
        if ":" in record.id:
            raise StoreError(
                f"record id {record.id!r} is qualified; a store's own record ids are "
                "unqualified — the <store-name>:<record-id> form is reserved for "
                "cross-scope references"
            )
        if not _ID_PATTERN.fullmatch(record.id):
            raise StoreError(f"record id {record.id!r} must match {_ID_PATTERN.pattern}")
        source_store, source_record_id = split_qualified_id(source_ref)
        if source_store == self.store_name:
            raise StoreError(
                f"source_ref {source_ref!r} names this store ({self.store_name}); an import "
                "event references another store's trail, never this one"
            )
        if source_record_id != record.id:
            raise StoreError(f"source_ref {source_ref!r} does not name this record ({record.id})")
        if self._record_path(record.id).exists():
            raise StoreError(f"record already exists: {record.id}")
        self._validate_cross_references(record)
        moment = now if now is not None else _now()
        landed = record.model_copy(
            update={
                "status": "candidate",
                "counterfactual": RecordCounterfactual(),
                "promoted_at": None,
                "confidence_updated_at": moment,
            }
        )
        self._save(landed)
        self._append_audit(
            AuditEvent(
                event="import",
                record_id=landed.id,
                to_status="candidate",
                reason=(
                    f"imported from {source_ref} — provenance verbatim; confidence "
                    f"clock re-anchored to this store's clock at {moment.isoformat()}"
                ),
            )
        )
        self._commit(f"import: {landed.id} from {source_ref}")
        return landed

    # ----------------------------------------------------------- persistence

    def _record_path(self, learning_id: str) -> Path:
        return self.records_dir / f"{learning_id}.json"

    def _load(self, learning_id: str) -> LearningRecord:
        # The read-side scope boundary (V4): a qualified id names a record
        # in another store. This store cannot serve it — it holds only its
        # own rows — and must not pretend to look for it; the federation
        # router dispatches qualified ids to their own scope.
        if ":" in learning_id:
            raise StoreError(
                f"qualified id {learning_id!r} names a record outside this store; "
                "read it from its own store (or the federation router), not this one"
            )
        path = self._record_path(learning_id)
        if not path.exists():
            raise StoreError(f"unknown learning id: {learning_id}")
        return LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _validate_cross_references(self, draft: LearningRecord) -> None:
        """Validate every cross-record reference a write would persist (V4).

        Record ids are unique only within one store
        (docs/cross-project-memory.md §1.1), so an unqualified reference is
        meaningful only as a local one: a reference to a record this store
        does not hold is cross-scope by construction and must arrive
        qualified ``<store-name>:<record-id>`` (invariant 3.4) — rejected
        here, at write time, before anything is persisted. A qualified
        reference to this store's own name is refused too: local references
        are unqualified, and self-qualification would blur exactly the
        boundary the qualifier exists to draw.
        """
        references = [source.record_id for source in draft.merged_sources]
        if draft.merged_into is not None:
            references.append(draft.merged_into)
        for reference in references:
            if ":" in reference:
                referenced_store, _ = split_qualified_id(reference)
                if referenced_store == self.store_name:
                    raise StoreError(
                        f"reference {reference!r} qualifies this store ({self.store_name}); "
                        "local references are unqualified"
                    )
            else:
                if not _ID_PATTERN.fullmatch(reference):
                    raise StoreError(f"record id {reference!r} must match {_ID_PATTERN.pattern}")
                if not self._record_path(reference).exists():
                    raise StoreError(
                        f"unqualified reference {reference!r} names no record in this "
                        "store; cross-scope references must be qualified "
                        "<store-name>:<record-id>"
                    )

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
        # The transition counter rides the audit choke point: every
        # lifecycle event is counted exactly once, with the status
        # transition it made. The same event goes to the structured log.
        metrics().increment(
            STORE_TRANSITIONS_TOTAL,
            {
                "event": event.event,
                "from_status": event.from_status or NO_STATUS,
                "to_status": event.to_status,
            },
        )
        log_event(
            logger,
            logging.INFO,
            "store.transition",
            f"{event.event} {event.record_id}: {event.from_status or NO_STATUS} -> {event.to_status}",
            event=event.event,
            record_id=event.record_id,
            from_status=event.from_status or NO_STATUS,
            to_status=event.to_status,
        )

    def _all_records(self) -> list[LearningRecord]:
        return [
            LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def _commit(self, message: str) -> None:
        _git(self.root, "add", "-A", ".")
        _git(self.root, "commit", "-q", "-m", message)
