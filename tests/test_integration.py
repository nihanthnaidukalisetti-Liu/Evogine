"""End-to-end integration: harness → store → gates → telemetry, fully offline.

One flow drives the whole engine loop with stub models — no live calls, no
Docker: a reflector (over a scripted model client) proposes a learning; the
git-backed store queues its counterfactual replay; a harness-backed adapter
answers the gate's :class:`ReplayRunner` seam by executing
:class:`BenchHarness` (baseline arm = the corpus without the knowledge, warm
arm = the corpus with the candidate applied); the memory gate decides from
those replays, records exactly one telemetry decision, and renders an
approval proposal without promoting; a human approval applies the promotion;
and a final harness run through the store's own promotion-only retrieval
shows the loop feeding back — the warm arm now retrieves the promoted
learning and the pass rate moves, on the case the knowledge fixes and on no
other.
"""

from __future__ import annotations

import json
from pathlib import Path

from evogine.bench.cases import BenchCase
from evogine.bench.harness import REPEAT_FLOOR, BenchHarness
from evogine.engine.records import LearningRecord, RecordBody
from evogine.engine.store import Engine
from evogine.gates.approvals import ApprovalLog, ApprovalState, approve_proposal
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink
from evogine.swarm.clients import ModelRequest, ModelResponse
from evogine.swarm.roles import Reflector

# The failure the reflector diagnoses: a retry storm the learning's fix
# strategy addresses. Body fields are retrieval's scoring surface, so the
# draft's body shares the retry case's vocabulary — that is what lets the
# promoted record clear the retrieval floor for this case later.
_RETRY_DRAFT = json.dumps(
    {
        "type": "failure-pattern",
        "title": "Unjittered backoff amplifies 429 retry storms",
        "body": {
            "signal": "429 retry storm across consecutive rollouts",
            "failure_mode": "retry storm amplifies without jittered exponential backoff",
            "detection": "counterfactual replay flips the verdict on the degraded retry case",
            "fix_strategy": (
                "add jittered exponential backoff with a cap and a per-rollout retry budget"
            ),
        },
        "confidence": 0.7,
        "note": "integration stub; deterministic",
    }
)

# A second, unrelated learning whose fix strategy helps neither corpus case:
# its counterfactual replay must not flip, so the promotion pass demotes it.
_LOCALE_DRAFT = json.dumps(
    {
        "type": "playbook",
        "title": "Cache locale tables before rendering",
        "body": {
            "signal": "decimal comma drift when the locale table is re-read per row",
            "failure_mode": "per-row locale lookups interleave with rendering",
            "detection": "golden report renders differ across locales",
            "fix_strategy": "cache the locale table once before rendering",
        },
        "confidence": 0.6,
    }
)


def _integration_corpus() -> list[BenchCase]:
    """Two cases: one the retry learning fixes, one it must not flip."""
    return [
        BenchCase(
            id="retry-jitter",
            family="retries",
            task_context=(
                "429 retry storm across rollouts: backoff without jitter degrades repair"
            ),
            hidden_invariant=lambda output: "jitter" in output.lower() and "cap" in output.lower(),
        ),
        BenchCase(
            id="race-idempotency",
            family="concurrency",
            task_context="double submit creates duplicate charges without idempotent lock",
            hidden_invariant=lambda output: (
                "idempoten" in output.lower() and "lock" in output.lower()
            ),
        ),
    ]


class ScriptedReflectorClient:
    """ModelClient stub: hands back scripted drafts in order, never a live call."""

    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._drafts:
            raise AssertionError("script exhausted: more model calls than drafts")
        return ModelResponse(text=self._drafts.pop(0), model="stub", stop_reason="end_turn")


class RepairRunner:
    """Model-free stand-in for the builder: applies retrieved knowledge.

    Without knowledge the repair fails; with knowledge the submission is the
    knowledge's own fix strategy — the honest "the engine applied what it
    knows" baseline, never a canned pass.
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


class OverlayRetriever:
    """Store retrieval plus the candidate applied — the counterfactual warm arm.

    The engine retrieves promoted records only; a candidate is not promoted
    yet. The counterfactual asks what the store would serve if this learning
    WERE promoted, so the overlay injects it ahead of whatever the store
    already serves (deduplicated, still bounded by ``k``).
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


class HarnessReplayAdapter:
    """Answers the gate's ReplayRunner seam by executing BenchHarness.

    Run-id convention: ``replay-<learning_id>-baseline`` replays the corpus
    with the knowledge absent (the harness's cold arm — knowledge-free by
    contract); ``replay-<learning_id>-warm`` replays with the candidate
    applied (the harness's warm arm over an overlay retriever). Verdicts are
    paired per case instance so the gate's materiality stage can align the
    arms by identity. Any malformed run id raises — the gate converts that
    into a fail-closed GATE_ERROR decision, never a crash.
    """

    def __init__(
        self,
        engine: Engine,
        corpus: list[BenchCase],
        runner: RepairRunner,
        *,
        repeats: int = REPEAT_FLOOR,
    ) -> None:
        self._engine = engine
        self._corpus = corpus
        self._runner = runner
        self._repeats = repeats

    def run(self, run_id: str) -> ReplayRun:
        for arm in ("baseline", "warm"):
            suffix = f"-{arm}"
            if not run_id.endswith(suffix):
                continue
            learning_id = run_id[: -len(suffix)].removeprefix("replay-")
            record = self._engine.get(learning_id)  # unknown id -> StoreError
            retriever: object
            if arm == "baseline":
                retriever = self._engine
            else:
                retriever = OverlayRetriever(self._engine, record)
            report = BenchHarness(
                retriever,  # type: ignore[arg-type]  # Engine/overlay satisfy Retriever
                self._runner,
            ).run(corpus=self._corpus, repeats=self._repeats)
            verdicts = [
                Verdict(
                    instance_id=f"{outcome.case_id}-r{outcome.repeat:02d}",
                    passed=outcome.passed,
                )
                for case_report in report.cases
                for outcome in (
                    case_report.cold.outcomes if arm == "baseline" else case_report.warm.outcomes
                )
            ]
            return ReplayRun(run_id=run_id, arm=arm, verdicts=verdicts)  # type: ignore[arg-type]
        raise KeyError(f"malformed replay run id (expected -baseline/-warm): {run_id}")


def _audit_events_by_record(root: Path) -> dict[str, list[str]]:
    lines = (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    events: dict[str, list[str]] = {}
    for line in lines:
        event = json.loads(line)
        events.setdefault(event["record_id"], []).append(event["event"])
    return events


def test_full_loop_reflector_to_promoted_retrieval(tmp_path: Path) -> None:
    """The loop closes: propose → replay → decide → record → approve → serve."""
    root = tmp_path / "store"
    engine = Engine(root)

    # 1. The reflector (stub model) proposes both learnings; the store keeps
    #    them ephemeral and the drafts never touch lifecycle fields.
    client = ScriptedReflectorClient([_RETRY_DRAFT, _LOCALE_DRAFT])
    reflector = Reflector(client=client, engine=engine)
    retry = reflector.propose_learning("429 retry storm across rollouts")
    locale = reflector.propose_learning("decimal comma drift in locale rendering")
    assert retry.status == "ephemeral"
    assert locale.status == "ephemeral"

    # 2. The store queues each learning's counterfactual replay -> candidate.
    engine.queue_replay(retry.id, [f"replay-{retry.id}-baseline", f"replay-{retry.id}-warm"])
    engine.queue_replay(locale.id, [f"replay-{locale.id}-baseline", f"replay-{locale.id}-warm"])
    assert engine.get(retry.id).status == "candidate"
    assert engine.get(locale.id).status == "candidate"

    # 3. The gate, telemetry, and approvals are wired over the same store root.
    gate = MemoryGate(
        runner=HarnessReplayAdapter(engine, _integration_corpus(), RepairRunner()),
        log=DecisionLog(root),
        approvals=ApprovalLog(root),
        evidence=RunsEvidenceSink(root),
    )

    # 4. The promotion pass decides both candidates from the harness replays.
    decisions = run_promotion_pass(engine, gate)
    assert {decision.learning_id for decision in decisions} == {retry.id, locale.id}
    by_learning = {decision.learning_id: decision for decision in decisions}

    # The retry learning flips its case, stays material, regresses nothing —
    # cleared, so the verdict becomes a pending approval proposal. The gate
    # itself never promotes: the record is still a candidate.
    assert by_learning[retry.id].stage is DecisionStage.APPROVAL
    assert by_learning[retry.id].promoted is False
    assert by_learning[retry.id].reason.startswith("proposed: ")
    assert by_learning[retry.id].evidence_ref == f"runs/gate/{retry.id}.json"
    assert (root / "runs" / "gate" / f"{retry.id}.json").exists()
    assert engine.get(retry.id).status == "candidate"

    # The locale learning flips nothing — a knowledge verdict at the replay
    # stage, and the pass demotes it (dropped, auditable, never deleted).
    assert by_learning[locale.id].stage is DecisionStage.REPLAY
    assert by_learning[locale.id].reason.startswith("no_flip: ")
    assert engine.get(locale.id).status == "dropped"

    # Telemetry completeness: exactly one decision per decided proposal.
    log = DecisionLog(root)
    assert len(log.all()) == 2
    assert log.counts_by_stage()[DecisionStage.APPROVAL] == 1
    assert log.counts_by_stage()[DecisionStage.REPLAY] == 1

    # 5. Before approval the store serves nothing warm: retrieval is
    #    promotion-only, so the harness measures zero on both arms.
    corpus = _integration_corpus()
    pre = BenchHarness(engine, RepairRunner()).run(corpus=corpus, repeats=REPEAT_FLOOR)
    assert pre.summary.cold_pass_rate == 0.0
    assert pre.summary.warm_pass_rate == 0.0

    # 6. A human approves the pending proposal; the store promotes.
    approval_log = ApprovalLog(root)
    pending = approval_log.pending()
    assert [proposal.proposal_id for proposal in pending] == [f"appr-{retry.id}"]
    assert pending[0].state is ApprovalState.PENDING
    promoted = approve_proposal(
        engine,
        approval_log,
        f"appr-{retry.id}",
        decided_by="nihanth",
        reason="material flip on the retry case, no regressions",
    )
    assert promoted.status == "promoted"
    assert promoted.promoted_at is not None
    assert engine.get(retry.id).status == "promoted"

    # 7. The loop feeds back: the harness's warm arm now retrieves the
    #    promoted learning for the retry case — and for no other.
    post = BenchHarness(engine, RepairRunner()).run(corpus=corpus, repeats=REPEAT_FLOOR)
    assert post.summary.warm_pass_rate > post.summary.cold_pass_rate
    cases = {case_report.case_id: case_report for case_report in post.cases}
    assert cases["retry-jitter"].warm.pass_rate == 1.0
    assert cases["retry-jitter"].cold.pass_rate == 0.0
    assert cases["race-idempotency"].warm.pass_rate == 0.0
    for outcome in cases["retry-jitter"].warm.outcomes:
        assert outcome.knowledge_ids == [retry.id]

    # The append-only audit trail holds the whole lifecycle, in order.
    assert _audit_events_by_record(root) == {
        retry.id: ["propose", "queue_replay", "promote"],
        locale.id: ["propose", "queue_replay", "demote"],
    }


def test_malformed_replay_run_is_a_gate_error_not_a_crash(tmp_path: Path) -> None:
    """A run id the adapter cannot parse fails closed at the replay stage."""
    root = tmp_path / "store"
    engine = Engine(root)
    draft = LearningRecord.sample().model_copy(update={"id": "lr-integration-1"})
    engine.propose(draft)
    engine.queue_replay("lr-integration-1", ["not-a-conventional-run-id"])

    gate = MemoryGate(
        runner=HarnessReplayAdapter(engine, _integration_corpus(), RepairRunner()),
        log=DecisionLog(root),
        approvals=ApprovalLog(root),
        evidence=RunsEvidenceSink(root),
    )
    decisions = run_promotion_pass(engine, gate)

    assert len(decisions) == 1
    assert decisions[0].stage is DecisionStage.GATE_ERROR
    assert decisions[0].promoted is False
    # Infrastructure says nothing about the knowledge: the record stays a
    # candidate, eligible for a retry with a well-formed replay queued.
    assert engine.get("lr-integration-1").status == "candidate"


def test_retrieval_floor_keeps_the_promoted_learning_topic_bound(tmp_path: Path) -> None:
    """Retrieval serves promoted knowledge where it belongs — and nowhere else."""
    root = tmp_path / "store"
    engine = Engine(root)
    record = LearningRecord.sample().model_copy(
        update={
            "id": "lr-topic-bound",
            "status": "promoted",
            "promoted_at": None,
            # Deterministic body: the retry vocabulary known to clear the
            # retrieval floor for the retry case (and only it).
            "body": RecordBody.model_validate(json.loads(_RETRY_DRAFT)["body"]),
        }
    )
    engine.propose(record.model_copy(update={"status": "ephemeral"}))
    engine.queue_replay("lr-topic-bound", ["run-1"])
    engine.promote(
        "lr-topic-bound",
        record.counterfactual.model_copy(
            update={"replay_run_ids": ["run-1"], "verdict_change": True}
        ),
    )
    corpus = _integration_corpus()
    served: dict[str, list[str]] = {}
    for case in corpus:
        served[case.id] = [candidate.id for candidate in engine.retrieve(case.task_context)]
    assert "lr-topic-bound" in served["retry-jitter"]
    assert served["race-idempotency"] == []
