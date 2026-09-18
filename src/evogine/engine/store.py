"""Git-backed learning store: the engine's durable memory surface.

The :class:`Engine` owns the learning lifecycle (``ephemeral -> candidate ->
promoted`` or ``dropped``), versioned storage in git, and promotion-only
retrieval: swarm workers may retrieve only ``promoted`` records, keeping
unproven, agent-generated knowledge out of active use until the memory gate
proves it materially changes verdicts.
"""

from __future__ import annotations

from evogine.engine.records import LearningRecord, RecordCounterfactual


class Engine:
    """Durable, git-backed store of :class:`LearningRecord` objects.

    Lifecycle contract (v1 design):
    ``propose`` stores a draft as ``ephemeral``; ``queue_replay`` attaches a
    pending counterfactual replay to a ``candidate``; ``promote`` marks a
    record ``promoted`` (with ``promoted_at`` stamped) only when the memory
    gate passes; ``demote`` moves a record to ``dropped`` — rejected knowledge
    stays in the store, auditable, never deleted.
    """

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Return the top-``k`` promoted records relevant to ``task_context``.

        Only ``promoted`` records are ever returned — warm runs must see
        exactly the knowledge the memory gate has proven.
        """
        raise NotImplementedError

    def propose(self, draft: LearningRecord) -> LearningRecord:
        """Store ``draft`` as a new ``ephemeral`` record and return it.

        Accepts agent-proposed drafts (reflector output) and human-authored
        seeds alike; provenance travels with the record.
        """
        raise NotImplementedError

    def queue_replay(self, learning_id: str, run_ids: list[str]) -> None:
        """Attach a pending counterfactual replay (``run_ids``) to a record.

        The replay compares cold and warm verdicts on the recorded failure
        cases; results feed the memory gate via :meth:`promote`.
        """
        raise NotImplementedError

    def promote(self, learning_id: str, replay_evidence: RecordCounterfactual) -> LearningRecord:
        """Promote ``learning_id`` to ``promoted`` given passing replay evidence.

        Called only after the memory gate accepts ``replay_evidence`` —
        a material verdict change on the queued counterfactual runs.
        """
        raise NotImplementedError

    def demote(self, learning_id: str, reason: str) -> LearningRecord:
        """Move ``learning_id`` to ``dropped`` with ``reason`` recorded.

        Dropped records remain in the store for audit; the engine never
        silently forgets what it once believed.
        """
        raise NotImplementedError
