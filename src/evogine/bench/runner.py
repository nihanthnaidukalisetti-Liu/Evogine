"""Offline case runners for the bench and CLI surfaces.

The suite and the CLI never call live models, so the default runner is
model-free by construction: the "repair" a case run produces is composed
from the knowledge the store serves. That keeps the benchmark an honest
function of store contents and corpus alone — cold arms measure the empty
baseline, warm arms measure exactly the knowledge being evaluated.
"""

from __future__ import annotations

from evogine.bench.cases import BenchCase
from evogine.engine.records import LearningRecord


class KnowledgeRunner:
    """The spend seam's offline builder: repairs by applying knowledge.

    With no knowledge retrieved there is nothing to act on and the run
    honestly fails; with knowledge the submission is the served records' fix
    strategies. Production runners drive model clients behind the same
    :class:`evogine.bench.harness.CaseRunner` protocol.
    """

    def execute(
        self,
        case: BenchCase,
        *,
        arm: str,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        if not knowledge:
            return "no known fix for this failure"
        return " | ".join(record.body.fix_strategy for record in knowledge)
