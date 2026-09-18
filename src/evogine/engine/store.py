"""Git-backed learning store: the engine's durable memory surface.

The :class:`Engine` owns the learning lifecycle (``ephemeral -> candidate ->
promoted`` or ``dropped``), versioned storage in git, and promotion-only
retrieval: swarm workers may retrieve only ``promoted`` records, keeping
unproven, agent-generated knowledge out of active use until the memory gate
proves it materially changes verdicts.

Persistence model: the store root is a dedicated directory that is its own
git repository — one JSON file per record, one commit per lifecycle
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

from evogine.engine.records import LearningRecord, RecordCounterfactual

# Ids become filenames under ``records/``: lock them to filesystem- and
# git-safe characters so a malicious proposal can never traverse paths
# (the security review's staging-traversal lesson, applied at the boundary).
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_WORD_RE = re.compile(r"[a-z0-9]+")


class StoreError(Exception):
    """Lifecycle violation, unknown id, or malformed proposal."""


def _now() -> datetime:
    return datetime.now(UTC)


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


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 3.

    Deliberately crude: this is the v1 lexical ranker, and V2-3 replaces it
    with hybrid retrieval behind the same promotion-only filter.
    """
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) >= 3}


def _record_tokens(record: LearningRecord) -> set[str]:
    """All indexed text of a record: the title AND every body field.

    The V1 audit flagged title-token matching whose body fields never affect
    ranking; that defect is not rebuilt here.
    """
    return _tokenize(
        f"{record.type} {record.title} {record.body.signal} "
        f"{record.body.failure_mode} {record.body.detection} {record.body.fix_strategy}"
    )


def _relevance(record: LearningRecord, context_tokens: set[str]) -> float:
    """Share of the task context's tokens the record covers.

    Zero means no lexical connection at all, so retrieval returns ``[]``
    honestly instead of falling back to unrelated records.
    """
    if not context_tokens:
        return 0.0
    hits = len(_record_tokens(record) & context_tokens)
    return hits / len(context_tokens)


class Engine:
    """Durable, git-backed store of :class:`LearningRecord` objects.

    Lifecycle contract (v1 design):
    ``propose`` stores a draft as ``ephemeral``; ``queue_replay`` attaches a
    pending counterfactual replay to a ``candidate``; ``promote`` marks a
    record ``promoted`` (with ``promoted_at`` stamped) only when the memory
    gate passes; ``demote`` moves a record to ``dropped`` — rejected knowledge
    stays in the store, auditable, never deleted.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            _git(self.root, "init", "-q")

    # ---------------------------------------------------------------- reads

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Return the top-``k`` promoted records relevant to ``task_context``.

        Only ``promoted`` records are ever returned — warm runs must see
        exactly the knowledge the memory gate has proven. With no promoted
        match the result is empty: retrieval never invents relevance.
        ``k <= 0`` yields an empty result (top-min(k, n) is empty).
        """
        if k <= 0:
            return []
        context_tokens = _tokenize(task_context)
        scored = [
            (_relevance(record, context_tokens), record.id, record)
            for record in self._all_records()
            if record.status == "promoted"
        ]
        scored = [item for item in scored if item[0] > 0.0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:k]]

    def get(self, learning_id: str) -> LearningRecord:
        """Return one record by id in whatever lifecycle state it holds.

        Audit-grade read: unlike :meth:`retrieve`, it sees ephemeral,
        candidate, and dropped records too — rejected knowledge stays
        readable, never deleted.
        """
        return self._load(learning_id)

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
        self._commit(f"demote: {learning_id}")
        return dropped

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

    def _all_records(self) -> list[LearningRecord]:
        return [
            LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def _commit(self, message: str) -> None:
        _git(self.root, "add", "-A", ".")
        _git(self.root, "commit", "-q", "-m", message)
