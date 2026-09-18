"""Objective gate #2: counterfactual-replay promotion of learned knowledge.

The memory gate keeps agent-generated knowledge ephemeral until proven
useful: a learning is promoted only when counterfactual replay on the
recorded failure cases shows a material verdict change with the learning
present versus absent. Records that fail the gate are demoted to ``dropped``
but stay in the store — rejected knowledge remains auditable, never deleted.
"""

from __future__ import annotations

from evogine.engine.records import LearningRecord


class MemoryGate:
    """Objective verifier for candidate learnings (v1 stub)."""

    def review(self, record: LearningRecord, replay_run_ids: list[str]) -> bool:
        """Return ``True`` iff ``record``'s counterfactual replay is material.

        ``replay_run_ids`` name the queued replay runs to inspect. The gate
        promotes only when the replayed verdicts change with the learning
        applied; case-specific materiality thresholds apply.
        """
        raise NotImplementedError
