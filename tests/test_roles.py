"""Swarm role contract tests over injected seams — no live models.

The model client is a scripted stub and the engine is an in-memory fake
implementing the same interface the real store exposes. The contract under
test: the builder retrieves promoted-only and never writes, the verifier
never proposes, the reflector proposes ephemeral learnings under its own
provenance.
"""

import inspect
import json
from pathlib import Path

import pytest

from evogine.engine.records import LearningRecord, RecordCounterfactual
from evogine.gates.telemetry import DecisionStage, PromotionDecision
from evogine.swarm.clients import ModelRequest, ModelResponse
from evogine.swarm.roles import (
    Builder,
    EngineContractError,
    Reflector,
    ReflectorOutputError,
    Verifier,
)


class ScriptedClient:
    """ModelClient stub: returns one scripted reply, records every request."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text=self.text)


class FakeEngine:
    """Full engine surface with recording — write methods must stay cold."""

    def __init__(self, records: list[LearningRecord]) -> None:
        self.records = records
        self.retrieve_calls: list[tuple[str, int]] = []
        self.write_calls: list[str] = []
        self.proposed: list[LearningRecord] = []

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        self.retrieve_calls.append((task_context, k))
        return list(self.records)

    def propose(self, draft: LearningRecord) -> LearningRecord:
        self.write_calls.append("propose")
        self.proposed.append(draft)
        return draft

    def queue_replay(self, learning_id: str, run_ids: list[str]) -> None:
        self.write_calls.append("queue_replay")

    def promote(self, learning_id: str, replay_evidence: RecordCounterfactual) -> LearningRecord:
        self.write_calls.append("promote")
        raise AssertionError("FakeEngine.promote must not be called")

    def demote(self, learning_id: str, reason: str) -> LearningRecord:
        self.write_calls.append("demote")
        raise AssertionError("FakeEngine.demote must not be called")


class SpyCodeGate:
    """Records evaluate() calls; structural stand-in for the real gate."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def evaluate(self, candidate_patch: str, baseline_ref: str) -> bool:
        self.calls.append((candidate_patch, baseline_ref))
        return self.verdict


class SpyMemoryGate:
    """Records review() calls; structural stand-in for the real gate."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls: list[tuple[LearningRecord, list[str]]] = []

    def review(
        self, record: LearningRecord, replay_run_ids: list[str]
    ) -> PromotionDecision:
        self.calls.append((record, replay_run_ids))
        return PromotionDecision(
            proposal_id=record.id,
            learning_id=record.id,
            promoted=self.verdict,
            stage=DecisionStage.NEVER_WORSE if self.verdict else DecisionStage.REPLAY,
            reason="cleared: test" if self.verdict else "no_flip: test",
        )


def _promoted(record_id: str) -> LearningRecord:
    """A promoted variant of the sample record (the engine's output shape)."""
    return LearningRecord.sample().model_copy(update={"id": record_id, "status": "promoted"})


def _learning_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "failure-pattern",
        "title": "Retry storms from unjittered backoff",
        "body": {
            "signal": "rising 429s",
            "failure_mode": "unjittered retries",
            "detection": "replay flips verdicts",
            "fix_strategy": "jittered exponential backoff",
        },
        "confidence": 0.7,
        "note": "from rollout 2791",
    }
    payload.update(overrides)
    return payload


def test_builder_retrieves_promoted_knowledge_and_uses_it() -> None:
    engine = FakeEngine([_promoted("lr-1")])
    client = ScriptedClient("--- a/x.py\n+++ b/x.py\n")

    builder = Builder(client=client, engine=engine)
    patch = builder.propose_patch("fix the retry storm", k=3)

    assert engine.retrieve_calls == [("fix the retry storm", 3)]
    prompt = client.requests[0].user
    assert "[lr-1]" in prompt
    assert "Unhandled 429 backoff snowballs into retry storms" in prompt
    assert patch.diff == "--- a/x.py\n+++ b/x.py"
    assert patch.knowledge_used == ["lr-1"]


def test_builder_never_writes_to_the_engine() -> None:
    engine = FakeEngine([_promoted("lr-1")])
    builder = Builder(client=ScriptedClient("diff"), engine=engine)

    builder.propose_patch("ctx")

    assert engine.write_calls == []


def test_builder_rejects_unpromoted_knowledge_before_any_model_call() -> None:
    engine = FakeEngine([LearningRecord.sample()])  # status: ephemeral
    client = ScriptedClient("should never be returned")
    builder = Builder(client=client, engine=engine)

    with pytest.raises(EngineContractError, match="promoted"):
        builder.propose_patch("ctx")

    assert client.requests == []


def test_builder_applies_its_proposed_patch(tmp_path: Path) -> None:
    diff = "--- /dev/null\n+++ b/fix.py\n@@ -0,0 +1,2 @@\n+print(1)\n+print(2)\n"
    builder = Builder(client=ScriptedClient(diff), engine=FakeEngine([]))

    patch = builder.propose_patch("add fix.py")
    touched = builder.apply(patch, tmp_path)

    assert touched == [tmp_path / "fix.py"]
    assert (tmp_path / "fix.py").read_text() == "print(1)\nprint(2)\n"


def test_verifier_returns_the_code_gate_verdict() -> None:
    gate = SpyCodeGate(False)
    verifier = Verifier(code_gate=gate, memory_gate=SpyMemoryGate(True))
    patch = Builder(client=ScriptedClient("d"), engine=FakeEngine([])).propose_patch("ctx")

    assert verifier.verify_patch(patch, "abc123") is False
    assert gate.calls == [(patch.diff, "abc123")]


def test_verifier_reviews_learning_through_the_memory_gate() -> None:
    memory = SpyMemoryGate(True)
    verifier = Verifier(code_gate=SpyCodeGate(False), memory_gate=memory)
    record = LearningRecord.sample()

    decision = verifier.review_learning(record, ["run-1"])

    assert decision.learning_id == record.id
    assert decision.promoted is True
    assert memory.calls == [(record, ["run-1"])]


def test_verifier_has_no_proposal_surface() -> None:
    init_params = set(inspect.signature(Verifier.__init__).parameters) - {"self"}

    assert init_params == {"code_gate", "memory_gate"}
    for name in ("propose", "propose_learning", "retrieve", "complete", "apply"):
        assert not hasattr(Verifier, name), f"Verifier must not expose {name}"


def test_reflector_proposes_ephemeral_under_its_own_provenance() -> None:
    sink = FakeEngine([])
    # The model claims promoted status and builder provenance — it must not
    # control either: lifecycle and provenance are engine/reflector-owned.
    payload = _learning_payload(status="promoted", proposed_by="builder")
    reflector = Reflector(client=ScriptedClient(json.dumps(payload)), engine=sink)

    record = reflector.propose_learning("rollout 2791 failed on 429 retries")

    assert sink.write_calls == ["propose"]
    assert record is sink.proposed[0]
    assert record.status == "ephemeral"
    assert record.provenance.proposed_by == "reflector"
    assert record.title == "Retry storms from unjittered backoff"
    assert record.confidence == 0.7
    assert record.counterfactual.replay_run_ids == []
    assert record.promoted_at is None


def test_reflector_tolerates_fenced_model_json() -> None:
    sink = FakeEngine([])
    fenced = "```json\n" + json.dumps(_learning_payload()) + "\n```"
    reflector = Reflector(client=ScriptedClient(fenced), engine=sink)

    record = reflector.propose_learning("ctx")

    assert record.type == "failure-pattern"


def test_reflector_rejects_malformed_output_and_stores_nothing() -> None:
    sink = FakeEngine([])
    reflector = Reflector(client=ScriptedClient("I think the fix is to add retries"), engine=sink)

    with pytest.raises(ReflectorOutputError):
        reflector.propose_learning("ctx")

    assert sink.write_calls == []
    assert sink.proposed == []


def test_reflector_rejects_out_of_range_confidence() -> None:
    sink = FakeEngine([])
    payload = _learning_payload(confidence=1.5)
    reflector = Reflector(client=ScriptedClient(json.dumps(payload)), engine=sink)

    with pytest.raises(ReflectorOutputError):
        reflector.propose_learning("ctx")

    assert sink.write_calls == []
