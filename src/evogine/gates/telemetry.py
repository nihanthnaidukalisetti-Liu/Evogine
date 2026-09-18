"""Promotion telemetry: every proposal verdict leaves a decision record.

V1's promotion pass failed silently — nine passes decided eleven proposals
and nothing recorded why. V2-1 makes the decision record absolute: the
memory gate writes one :class:`PromotionDecision` per decided proposal,
naming the stage that stopped (or passed) it and a template-checked reason
from a fixed vocabulary, so telemetry stays aggregable instead of prose.
The completeness invariant is absolute: a decided proposal with no
decision record is a bug, and the test suite fails on it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class DecisionStage(str, Enum):
    """The gate stage a decision came from.

    REPLAY — did the warm verdict actually flip?
    MATERIALITY — did the flip qualify as material?
    NEVER_WORSE — did any evaluated case regress?
    APPROVAL — did a human approve the rendered proposal? (V2-6)
    GATE_ERROR — infrastructure, not knowledge.
    """

    REPLAY = "replay"
    MATERIALITY = "materiality"
    NEVER_WORSE = "never_worse"
    APPROVAL = "approval"
    GATE_ERROR = "gate_error"


def _now() -> datetime:
    return datetime.now(UTC)


class PromotionDecision(BaseModel):
    """Structured verdict for one proposal: which stage stopped (or passed) it.

    ``proposal_id`` and ``learning_id`` hold the same value today — the v1
    store treats one record as one proposal — but stay separate fields so
    the V2-6 approval boundary can give proposals their own identity
    without breaking this schema (realized as the approval proposal's
    ``appr-<learning_id>`` id).
    """

    proposal_id: str
    learning_id: str
    decided_at: datetime = Field(default_factory=_now)
    promoted: bool
    stage: DecisionStage
    reason: str
    replay_run_ids: list[str] = Field(default_factory=list)
    evidence_ref: str | None = None
    trajectory_ref: str | None = None
    """V2-8 anti-gaming: store-relative path to the decision's trajectory
    file (tool calls included), set when trajectory capture is wired — the
    full path to this verdict, not just the verdict."""
    pass_id: str | None = None
    """V3 pass identity: the promotion pass that produced this decision,
    ``None`` for decisions made outside a pass or predating the field —
    old decision files stay readable unchanged."""
    review_latency_seconds: float | None = None
    """V3 review latency: seconds from the record's ``created`` timestamp
    to this decision's ``decided_at``, computed at review time from those
    two existing clocks (no new instrumentation) — how long knowledge sat
    before the gate decided it. ``None`` only when unknown (legacy files)."""


class TelemetryError(ValueError):
    """A decision record violates the template-checked reason vocabulary."""


# The reason vocabulary: one label per failure (or success) shape, owned by
# exactly one stage. The gate builds reasons from these templates and
# :func:`check_reason` rejects anything else at record time — reasons cannot
# drift into free-form prose the report cannot aggregate.
_STAGE_REASON_LABELS: dict[DecisionStage, frozenset[str]] = {
    DecisionStage.REPLAY: frozenset({"no_flip"}),
    DecisionStage.MATERIALITY: frozenset({"not_material"}),
    DecisionStage.NEVER_WORSE: frozenset({"regression", "cleared"}),
    DecisionStage.APPROVAL: frozenset({"proposed", "refused_no_evidence"}),
    DecisionStage.GATE_ERROR: frozenset({"gate_error"}),
}

_REASON_PATTERNS: dict[str, re.Pattern[str]] = {
    "no_flip": re.compile(r"no_flip: \d+/\d+ replays changed verdict"),
    "not_material": re.compile(r"not_material: scored delta -?\d+\.\d{3} vs threshold \d+(?:\.\d+)?"),
    "regression": re.compile(r"regression: \d+ case\(s\) regressed(?:: .*)?"),
    "cleared": re.compile(
        r"cleared: (?:\d+ binary flips|scored delta -?\d+\.\d{3} vs threshold \d+(?:\.\d+)?)"
    ),
    # The proposed reason carries the same materiality detail as `cleared`:
    # the proposal exists because the cleared detail justified promotion.
    "proposed": re.compile(
        r"proposed: (?:\d+ binary flips|scored delta -?\d+\.\d{3} vs threshold \d+(?:\.\d+)?)"
    ),
    "refused_no_evidence": re.compile(r"refused_no_evidence: .+"),
    "gate_error": re.compile(r"gate_error: .+"),
}


def check_reason(stage: DecisionStage, reason: str) -> None:
    """Raise :class:`TelemetryError` unless ``reason`` is valid for ``stage``.

    A valid reason is ``"<label>: <evidence>"`` where the label belongs to
    the stage and the evidence matches the label's template — e.g.
    ``"no_flip: 0/3 replays changed verdict"``.
    """
    label, separator, _ = reason.partition(": ")
    if not separator:
        raise TelemetryError(f"reason must be '<label>: <evidence>', got {reason!r}")
    allowed = _STAGE_REASON_LABELS[stage]
    if label not in allowed:
        raise TelemetryError(
            f"reason label {label!r} is not valid for stage {stage.value}; "
            f"allowed: {sorted(allowed)}"
        )
    if not _REASON_PATTERNS[label].fullmatch(reason):
        raise TelemetryError(
            f"reason {reason!r} does not match the template for label {label!r}"
        )


# Same filesystem-safe id rule the store applies to record ids: decision
# and evidence files are named by proposal id, so a hostile id must never
# traverse paths (the security review's staging-traversal lesson).
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Public alias so sibling run-artifact sinks (the V2-8 trajectory capture)
#: enforce one filesystem-safe id rule instead of drifting into copies.
ID_PATTERN = _ID_PATTERN


class DecisionLog:
    """File-backed decision log: one JSON decision per proposal.

    Decisions live under ``<root>/decisions/`` inside the engine store, so
    the next store commit versions them alongside the records they refer
    to. Keyed by proposal id, the log holds the latest verdict per proposal
    — re-deciding one (a gate_error retry, say) overwrites — while full
    lifecycle history stays in the store's append-only audit trail.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.decisions_dir = self.root / "decisions"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

    def record(self, decision: PromotionDecision) -> None:
        """Persist ``decision``, refusing reasons outside the vocabulary."""
        check_reason(decision.stage, decision.reason)
        if not _ID_PATTERN.fullmatch(decision.proposal_id):
            raise TelemetryError(
                f"proposal id {decision.proposal_id!r} must match {_ID_PATTERN.pattern}"
            )
        path = self._path(decision.proposal_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(decision.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def all(self) -> list[PromotionDecision]:
        """Every recorded decision, ordered by proposal id."""
        if not self.decisions_dir.exists():
            return []
        return [
            PromotionDecision.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.decisions_dir.glob("*.json"))
        ]

    def counts_by_stage(self) -> dict[DecisionStage, int]:
        """Decision counts keyed by stage, zero-filled over every stage."""
        counts = {stage: 0 for stage in DecisionStage}
        for decision in self.all():
            counts[decision.stage] += 1
        return counts

    def _path(self, proposal_id: str) -> Path:
        return self.decisions_dir / f"{proposal_id}.json"


class RunsEvidenceSink:
    """Persists gate evidence under ``<root>/runs/gate/`` and returns the ref.

    The ref is store-root-relative — ``runs/gate/<proposal_id>.json`` — so
    decision records stay portable with the git-backed store.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write(self, proposal_id: str, payload: Mapping[str, object]) -> str:
        if not _ID_PATTERN.fullmatch(proposal_id):
            raise TelemetryError(
                f"proposal id {proposal_id!r} must match {_ID_PATTERN.pattern}"
            )
        gate_dir = self.root / "runs" / "gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        path = gate_dir / f"{proposal_id}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return f"runs/gate/{proposal_id}.json"
