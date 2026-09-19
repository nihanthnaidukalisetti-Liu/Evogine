"""Library runner for the memory gate's replay seam, backed by BenchHarness.

The memory gate sources its paired replay evidence from a
:class:`evogine.gates.replay.ReplayRunner`. :class:`HarnessReplayRunner` is
the in-process realization over :class:`evogine.bench.harness.BenchHarness`:
for the canonical run id ``replay-<learning_id>-baseline`` it replays the
corpus with the knowledge absent (the harness's cold arm — knowledge-free by
contract), and for ``replay-<learning_id>-warm`` it replays with the
candidate learning applied over the generalized overlay retriever
(:class:`evogine.bench.shadow.OverlayRetriever` — the store's promotion-only
pool plus the candidate, as it would be served if promoted).
Verdicts pair per case instance across arms, so the gate's materiality stage
aligns them by identity.

With a ``runs_root`` configured (the CLI wires ``<root>/runs/replay``), each
replay run also persists its own immutable run directory: per-outcome
evidence from the harness, a ``replay-run`` descriptor naming the gate arm
and learning, and the completion marker — the gate's replay evidence becomes
resumable, inspectable evidence on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from evogine.bench.cases import BenchCase
from evogine.bench.harness import BenchHarness, CaseRunner, Retriever, RunArtifacts
from evogine.bench.replay import REPEAT_FLOOR
from evogine.bench.rundirs import RunDirectory, open_run
from evogine.bench.shadow import OverlayRetriever
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
        runs_root: Path | None = None,
    ) -> None:
        self._engine = engine
        self._runner = runner
        self._corpus = corpus
        self._repeats = repeats
        self._runs_root = runs_root

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
                retriever = OverlayRetriever(
                    pool=self._engine.promoted_records(),
                    overlay=[record],
                    embedder=self._engine.embedder,
                )
            gate_arm: Arm = "baseline" if arm == BASELINE_ARM else "warm"
            run_dir: RunDirectory | None = None
            artifacts: RunArtifacts | None = None
            if self._runs_root is not None:
                # A fresh uniquely-named directory per replay run: completed runs
                # are immutable, and the gate re-reviews candidates on every pass,
                # so deterministic ids would collide across invocations.
                run_dir = open_run(self._runs_root)
                artifacts = RunArtifacts(run_dir)
            report = BenchHarness(retriever, self._runner).run(
                corpus=self._corpus, repeats=self._repeats, artifacts=artifacts
            )
            if run_dir is not None:
                # Descriptor first, completion marker last: an incomplete dir
                # without the marker is resumable, a completed one immutable.
                run_dir.write_artifact(
                    "replay-run",
                    json.dumps(
                        {
                            "replay_run_id": run_id,
                            "gate_arm": gate_arm,
                            "learning_id": learning_id,
                        }
                    ),
                )
                run_dir.mark_complete()
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
            return ReplayRun(run_id=run_id, arm=gate_arm, verdicts=verdicts)
        raise KeyError(
            f"malformed replay run id (expected {REPLAY_RUN_PREFIX}-<learning_id>-"
            f"{BASELINE_ARM}|{WARM_ARM}): {run_id}"
        )
