"""Library runner for the memory gate's replay seam, backed by BenchHarness.

The memory gate sources its paired replay evidence from a
:class:`evogine.gates.replay.ReplayRunner`. :class:`HarnessReplayRunner` is
the in-process realization over :class:`evogine.bench.harness.BenchHarness`:
for the canonical run id ``replay-<learning_id>-baseline`` it replays the
corpus with the knowledge absent (the harness's cold arm — knowledge-free by
contract), and for ``replay-<learning_id>-warm`` it replays with the
candidate learning applied over an overlay retriever (the store's promotion-
only retrieval plus the candidate, as it would be served if promoted).
Verdicts pair per case instance across arms, so the gate's materiality stage
aligns them by identity.

The CLI's ``run`` subcommand wires this runner into the gate; the gate
persists the paired verdicts through its own evidence sink, so no second
file format is introduced here.
"""

from __future__ import annotations

from evogine.bench.cases import BenchCase
from evogine.bench.harness import BenchHarness, CaseRunner, Retriever
from evogine.bench.replay import REPEAT_FLOOR
from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.materiality import Verdict
from evogine.gates.replay import Arm, ReplayRun

REPLAY_RUN_PREFIX = "replay"
BASELINE_ARM: Arm = "baseline"
WARM_ARM: Arm = "warm"
_ARMS: tuple[Arm, ...] = (BASELINE_ARM, WARM_ARM)


def replay_run_ids(learning_id: str) -> list[str]:
    """The canonical run-id pair queued for one candidate learning."""
    return [
        f"{REPLAY_RUN_PREFIX}-{learning_id}-{BASELINE_ARM}",
        f"{REPLAY_RUN_PREFIX}-{learning_id}-{WARM_ARM}",
    ]


class _OverlayRetriever:
    """Store retrieval plus the candidate applied — the counterfactual warm arm.

    The engine serves promoted records only; a candidate is not promoted
    yet. The overlay injects it ahead of whatever the store already serves,
    deduplicated and still bounded by ``k``.
    """

    def __init__(self, engine: Engine, candidate: LearningRecord) -> None:
        self._engine = engine
        self._candidate = candidate

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        records = [self._candidate]
        records += [
            record
            for record in self._engine.retrieve(task_context, k=k)
            if record.id != self._candidate.id
        ]
        return records[:k]


class HarnessReplayRunner:
    """Answers the gate's ReplayRunner seam by executing BenchHarness.

    Unknown learning ids surface the store's error and malformed run ids
    raise ``KeyError`` — both are converted by the gate into fail-closed
    GATE_ERROR decisions, never crashes and never a silent pass.
    """

    def __init__(
        self,
        engine: Engine,
        runner: CaseRunner,
        *,
        corpus: list[BenchCase] | None = None,
        repeats: int = REPEAT_FLOOR,
    ) -> None:
        self._engine = engine
        self._runner = runner
        self._corpus = corpus
        self._repeats = repeats

    def run(self, run_id: str) -> ReplayRun:
        for arm in _ARMS:
            suffix = f"-{arm}"
            if not run_id.endswith(suffix):
                continue
            learning_id = run_id[: -len(suffix)].removeprefix(f"{REPLAY_RUN_PREFIX}-")
            if not learning_id or learning_id == run_id:
                break  # malformed prefix; the raise below reports it
            record = self._engine.get(learning_id)
            retriever: Retriever
            if arm == BASELINE_ARM:
                retriever = self._engine
            else:
                retriever = _OverlayRetriever(self._engine, record)
            report = BenchHarness(retriever, self._runner).run(
                corpus=self._corpus, repeats=self._repeats
            )
            verdicts = [
                Verdict(
                    instance_id=f"{outcome.case_id}-r{outcome.repeat:02d}",
                    passed=outcome.passed,
                )
                for case_report in report.cases
                for outcome in (
                    case_report.cold.outcomes if arm == BASELINE_ARM else case_report.warm.outcomes
                )
            ]
            gate_arm: Arm = "baseline" if arm == BASELINE_ARM else "warm"
            return ReplayRun(run_id=run_id, arm=gate_arm, verdicts=verdicts)
        raise KeyError(
            f"malformed replay run id (expected {REPLAY_RUN_PREFIX}-<learning_id>-"
            f"{BASELINE_ARM}|{WARM_ARM}): {run_id}"
        )
