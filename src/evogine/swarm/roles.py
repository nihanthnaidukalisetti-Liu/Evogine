"""The task-scaled agent swarm: builder, verifier, reflector.

Three roles with deliberately asymmetric powers, wired through seams — the
:class:`~evogine.swarm.clients.ModelClient` protocol for model access and
narrow structural views of the engine's store interface — so no role can
reach around its contract:

- :class:`Builder` — proposes candidate patches and applies them in a
  workspace. It reads knowledge only through the engine's promoted-only
  retrieval (and re-checks that contract defensively); it never writes to
  the engine.
- :class:`Verifier` — assesses proposals through the objective gates. It
  holds no model client and no engine: it cannot propose or retrieve, only
  check, and it judges the proposal itself — never the builder's claims
  about it.
- :class:`Reflector` — diagnoses failures and proposes learnings through the
  engine's ``propose`` sink. Whatever the model claims, the reflector stores
  ``ephemeral`` under reflector provenance; only the memory gate promotes.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.gates.code_gate import CodeGate
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.telemetry import PromotionDecision
from evogine.swarm.clients import ModelClient, ModelRequest
from evogine.swarm.patches import apply_unified_diff

_BUILDER_SYSTEM = (
    "You are the builder in a self-evolving engineering swarm. "
    "Propose the smallest correct unified diff patch for the task. "
    "Prefer promoted knowledge where it applies. "
    "Reply with the unified diff only — no prose, no code fences."
)

_REFLECTOR_SYSTEM = (
    "You are the reflector in a self-evolving engineering swarm. "
    "Diagnose the failure and propose exactly one learning as a single JSON "
    'object with keys: "type" ("failure-pattern" | "playbook" | "check"), '
    '"title", "body" {"signal", "failure_mode", "detection", "fix_strategy"}, '
    '"confidence" (0.0-1.0), "note". Reply with the JSON object only.'
)


class EngineContractError(RuntimeError):
    """A knowledge source violated the engine's promoted-only retrieval contract."""


class ReflectorOutputError(ValueError):
    """The reflector's model returned output that is not a valid learning draft."""


class KnowledgeSource(Protocol):
    """The engine surface the builder may use: promoted-only retrieval."""

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Return promoted records relevant to ``task_context``."""
        ...


class LearningSink(Protocol):
    """The engine surface the reflector may use: ephemeral proposals only."""

    def propose(self, draft: LearningRecord) -> LearningRecord:
        """Store ``draft`` as a new ephemeral record and return it."""
        ...


class ProposedPatch(BaseModel):
    """A builder-proposed change plus the provenance of what informed it."""

    task_context: str
    diff: str
    knowledge_used: list[str] = Field(default_factory=list)


class Builder:
    """Proposes candidate patches, informed only by promoted knowledge."""

    def __init__(self, *, client: ModelClient, engine: KnowledgeSource) -> None:
        self._client = client
        self._engine = engine

    def propose_patch(self, task_context: str, k: int = 5) -> ProposedPatch:
        """Ask the model for a patch, retrieving promoted knowledge first.

        Raises :class:`EngineContractError` when the knowledge source returns
        anything but promoted records: the builder must never act on unproven
        knowledge, even if the store misbehaves.
        """
        knowledge = _promoted_only(self._engine.retrieve(task_context, k))
        response = self._client.complete(
            ModelRequest(system=_BUILDER_SYSTEM, user=_render_builder_task(task_context, knowledge))
        )
        return ProposedPatch(
            task_context=task_context,
            diff=response.text.strip(),
            knowledge_used=[record.id for record in knowledge],
        )

    def apply(self, patch: ProposedPatch, workspace: Path) -> list[Path]:
        """Apply ``patch`` inside ``workspace``; return the touched paths."""
        return apply_unified_diff(workspace, patch.diff)


class Verifier:
    """Checks proposals through the objective gates — never proposes.

    Deliberately narrow dependencies: the verifier holds gates and nothing
    else — no model client, no engine — so it cannot propose, retrieve, or
    write a record even by accident. It assesses the proposal itself and
    never the builder's claims about it.
    """

    def __init__(self, *, code_gate: CodeGate, memory_gate: MemoryGate) -> None:
        self._code_gate = code_gate
        self._memory_gate = memory_gate

    def verify_patch(self, patch: ProposedPatch, baseline_ref: str) -> bool:
        """Return the code gate's verdict on ``patch`` against ``baseline_ref``."""
        return self._code_gate.evaluate(patch.diff, baseline_ref)

    def review_learning(
        self, record: LearningRecord, replay_run_ids: list[str]
    ) -> PromotionDecision:
        """Run the memory gate's review; return the recorded promotion decision."""
        return self._memory_gate.review(record, replay_run_ids)


class DraftLearning(BaseModel):
    """The JSON shape the reflector asks the model for.

    Deliberately narrower than :class:`LearningRecord`: lifecycle, identity,
    provenance, and replay evidence are engine-owned, never model-owned —
    the model cannot promote itself.
    """

    type: Literal["failure-pattern", "playbook", "check"]
    title: str
    body: RecordBody
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = ""


class Reflector:
    """Proposes ephemeral learnings; the memory gate decides their fate."""

    def __init__(self, *, client: ModelClient, engine: LearningSink) -> None:
        self._client = client
        self._engine = engine

    def propose_learning(self, failure_context: str) -> LearningRecord:
        """Diagnose ``failure_context`` and store the learning as ephemeral.

        The model shapes only the record's content: status is forced to
        ``ephemeral`` and provenance to the reflector, whatever the model
        output claims. Raises :class:`ReflectorOutputError` when the model
        returns anything but the requested JSON shape; nothing is stored.
        """
        response = self._client.complete(
            ModelRequest(system=_REFLECTOR_SYSTEM, user=failure_context)
        )
        draft = _parse_learning_draft(response.text)
        record = LearningRecord(
            id=_new_learning_id(),
            type=draft.type,
            title=draft.title,
            body=draft.body,
            status="ephemeral",
            confidence=draft.confidence,
            provenance=RecordProvenance(
                proposed_by="reflector",
                origin="swarm-reflection",
                source=_source_from_context(failure_context),
                note=draft.note,
            ),
            counterfactual=RecordCounterfactual(),
            created=datetime.now(UTC),
        )
        return self._engine.propose(record)


def _promoted_only(records: list[LearningRecord]) -> list[LearningRecord]:
    """Defensively enforce the engine's promoted-only retrieval contract."""
    unproven = [record.id for record in records if record.status != "promoted"]
    if unproven:
        raise EngineContractError(
            f"knowledge source returned unproven records {unproven}: "
            "builders may retrieve promoted knowledge only"
        )
    return records


def _render_builder_task(task_context: str, knowledge: list[LearningRecord]) -> str:
    if knowledge:
        rendered = "\n".join(
            f"- [{record.id}] {record.title}: {record.body.fix_strategy}"
            for record in knowledge
        )
        knowledge_block = f"Promoted knowledge (proven; prefer it where it applies):\n{rendered}"
    else:
        knowledge_block = "No promoted knowledge is available for this task yet."
    return "\n\n".join(
        [f"Task:\n{task_context}", knowledge_block, "Reply with the unified diff only."]
    )


def _parse_learning_draft(text: str) -> DraftLearning:
    """Parse the model's JSON draft, tolerating markdown code fences."""
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[1] if "\n" in payload else ""
        if payload.endswith("```"):
            payload = payload[: -3].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReflectorOutputError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReflectorOutputError("model output is not a JSON object")
    try:
        return DraftLearning.model_validate(data)
    except ValidationError as exc:
        raise ReflectorOutputError(f"learning draft does not match the contract: {exc}") from exc


def _new_learning_id() -> str:
    return f"lr-{datetime.now(UTC):%Y-%m-%d}-{uuid.uuid4().hex[:8]}"


def _source_from_context(failure_context: str) -> str:
    """Collapse the failure context into a bounded provenance string."""
    return " ".join(failure_context.split())[:80]

