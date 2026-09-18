"""Human-approval promotion boundary (V2-6): the gate proposes, a human decides.

The memory gate no longer promotes. When its knowledge stages clear, it
renders a :class:`PromotionProposal` — the record, the replay evidence, and
the decision trail that cleared — into the :class:`ApprovalLog`, and only
``evogine approvals approve|reject`` (via :func:`approve_proposal` /
:func:`reject_proposal`) applies or refuses it. The evidence requirement is
enforced here regardless of the human: a proposal without replay evidence
is refused at render time, and the gate has no promotion path at all —
no promotion without an approved proposal, structurally.

Promotion is the irreversible side of the boundary (the record becomes live
knowledge that every warm run can retrieve), so it never auto-runs;
rejection is the reversible-record side: the refused record is demoted to
``dropped`` — auditable, never deleted — with the human's identity and
reason, which also ends the cycle (a dropped record is no longer a
candidate, so refused knowledge cannot be re-proposed unchanged).
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from evogine.engine.records import LearningRecord, RecordCounterfactual
from evogine.engine.store import Engine, store_lock
from evogine.gates.telemetry import (
    # Package-internal reuse: the same filesystem-safe id rule the decision
    # log and the store apply — proposals become filenames under approvals/.
    _ID_PATTERN,
    DecisionStage,
    PromotionDecision,
    TelemetryError,
)


class ApprovalError(Exception):
    """An approval action violates the boundary contract: refused or misuse."""


# Operator identity format (V3): a nonblank printable identity of at most
# 64 characters — a handle ("nihanth"), a display name ("Nihanth Naidu
# Kalisetti"), or an email ("nihanthnaidu008@gmail.com"). Leading and
# trailing whitespace is stripped; control characters (notably newlines,
# which would corrupt one-event-per-line audit greps) and quoting/bracket
# punctuation are refused. Legacy decision files stay readable unchanged:
# validation happens here, at the boundary, only when a new identity is
# recorded — never on load.
OPERATOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._'@+-]{0,63}$")


def operator_identity(raw: str) -> str:
    """Validate an operator identity for the human-approval boundary.

    Returns the stripped identity; raises :class:`ApprovalError` when it
    is blank or does not match the accepted format. Both decision paths
    funnel ``decided_by`` through the log's ``record_*`` methods, so this
    is the one choke point every future transport inherits.
    """
    identity = raw.strip()
    if not identity:
        raise ApprovalError("operator identity must not be blank")
    if not OPERATOR_PATTERN.fullmatch(identity):
        raise ApprovalError(
            f"operator identity {identity!r} does not match the accepted format "
            "(letters, digits, spaces, . _ ' @ + - ; up to 64 characters)"
        )
    return identity


class ApprovalState(str, Enum):
    """Lifecycle of one promotion proposal.

    ``pending`` — rendered by the gate, awaiting a human decision.
    ``approved`` — a human approved it; application to the store is the
    next (immediate) step, so a crash between the two is visible here
    rather than silent.
    ``applied`` — approved and promoted into the store.
    ``rejected`` — a human refused it; the record is demoted to ``dropped``.
    """

    PENDING = "pending"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"


def _now() -> datetime:
    return datetime.now(UTC)


class PromotionProposal(BaseModel):
    """A promotion rendered for human review — never applied by itself.

    ``proposal_id`` is the proposal's own identity (``appr-<learning_id>``),
    distinct from the learning it proposes to promote — the separation the
    promotion-telemetry schema reserved for this boundary. ``decision`` is
    the gate's cleared knowledge verdict (the decision trail): there,
    ``promoted`` reads as *the gate recommends promotion*; the authoritative
    state is the proposal's own ``state``, then the store's record status.
    """

    proposal_id: str
    learning_id: str
    created_at: datetime = Field(default_factory=_now)
    decision: PromotionDecision
    evidence_ref: str | None = None
    state: ApprovalState = ApprovalState.PENDING
    decided_by: str | None = None
    decided_at: datetime | None = None
    decided_reason: str | None = None


class ApprovalLog:
    """File-backed proposal log: one JSON proposal per rendered promotion.

    Proposals live under ``<root>/approvals/`` inside the engine store, so
    the next store commit versions them alongside the record they propose
    to promote — the reviewable artifact and the knowledge it concerns are
    versioned together. Keyed by proposal id; re-rendering for the same
    learning (a gate re-review after fresh replay) overwrites the pending
    proposal with the new trail, while a decided proposal is final.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.approvals_dir = self.root / "approvals"
        self.approvals_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self, learning_id: str, decision: PromotionDecision, evidence_ref: str | None
    ) -> PromotionProposal:
        """Render a pending proposal from the gate's cleared verdict.

        The evidence requirement is absolute: ``evidence_ref`` is the
        replay evidence the proposal exists to justify — without one there
        is nothing for a human to approve, so the render refuses
        (:class:`ApprovalError`) rather than propose on faith. The trail
        must be the gate's own cleared verdict; anything else is an
        internal-invariant bug (:class:`TelemetryError`), not a refusal.
        """
        if evidence_ref is None:
            raise ApprovalError(f"no replay evidence attached for {learning_id}")
        if (
            decision.learning_id != learning_id
            or decision.stage is not DecisionStage.NEVER_WORSE
            or decision.promoted is not True
            or not decision.reason.startswith("cleared: ")
        ):
            raise TelemetryError(
                f"proposal trail for {learning_id} is not a cleared gate verdict"
            )
        proposal = PromotionProposal(
            proposal_id=f"appr-{learning_id}",
            learning_id=learning_id,
            decision=decision,
            evidence_ref=evidence_ref,
        )
        self._save(proposal)
        return proposal

    def get(self, proposal_id: str) -> PromotionProposal:
        """Return one proposal by id; unknown ids are :class:`ApprovalError`."""
        path = self._path(proposal_id)
        if not path.exists():
            raise ApprovalError(f"unknown proposal: {proposal_id}")
        return PromotionProposal.model_validate_json(path.read_text(encoding="utf-8"))

    def all(self) -> list[PromotionProposal]:
        """Every proposal in the log, ordered by proposal id."""
        if not self.approvals_dir.exists():
            return []
        return [
            PromotionProposal.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.approvals_dir.glob("*.json"))
        ]

    def pending(self) -> list[PromotionProposal]:
        """The proposals a human has yet to decide."""
        return [
            proposal for proposal in self.all() if proposal.state is ApprovalState.PENDING
        ]

    def record_approval(
        self, proposal_id: str, *, decided_by: str, reason: str | None = None
    ) -> PromotionProposal:
        """Record a human approval: ``pending -> approved``."""
        proposal = self._require_pending(proposal_id)
        decided_by = operator_identity(decided_by)
        decided = proposal.model_copy(
            update={
                "state": ApprovalState.APPROVED,
                "decided_by": decided_by,
                "decided_at": _now(),
                "decided_reason": reason,
            }
        )
        self._save(decided)
        return decided

    def record_rejection(
        self, proposal_id: str, *, decided_by: str, reason: str
    ) -> PromotionProposal:
        """Record a human rejection: ``pending -> rejected`` (reason required)."""
        proposal = self._require_pending(proposal_id)
        decided_by = operator_identity(decided_by)
        if not reason.strip():
            raise ApprovalError("rejection requires a reason")
        decided = proposal.model_copy(
            update={
                "state": ApprovalState.REJECTED,
                "decided_by": decided_by,
                "decided_at": _now(),
                "decided_reason": reason,
            }
        )
        self._save(decided)
        return decided

    def record_applied(self, proposal_id: str) -> PromotionProposal:
        """Mark an approved proposal applied: ``approved -> applied``."""
        proposal = self.get(proposal_id)
        if proposal.state is not ApprovalState.APPROVED:
            raise ApprovalError(
                f"only an approved proposal can be marked applied; "
                f"{proposal_id} is {proposal.state.value}"
            )
        applied = proposal.model_copy(update={"state": ApprovalState.APPLIED})
        self._save(applied)
        return applied

    # ----------------------------------------------------------- persistence

    def _require_pending(self, proposal_id: str) -> PromotionProposal:
        proposal = self.get(proposal_id)
        if proposal.state is not ApprovalState.PENDING:
            raise ApprovalError(
                f"proposal {proposal_id} is already {proposal.state.value}; "
                "decided proposals are final"
            )
        return proposal

    def _path(self, proposal_id: str) -> Path:
        return self.approvals_dir / f"{proposal_id}.json"

    def _save(self, proposal: PromotionProposal) -> None:
        if not _ID_PATTERN.fullmatch(proposal.proposal_id):
            raise TelemetryError(
                f"proposal id {proposal.proposal_id!r} must match {_ID_PATTERN.pattern}"
            )
        path = self._path(proposal.proposal_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(proposal.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def approve_proposal(
    engine: Engine,
    log: ApprovalLog,
    proposal_id: str,
    *,
    decided_by: str,
    reason: str | None = None,
) -> LearningRecord:
    """Apply an approved proposal: the human decision promotes the record.

    The store's advisory review lock is held across the whole window, so a
    concurrent second decision — from the CLI or any transport — blocks
    and then fails its own pending re-check below instead of racing this
    one (V3: the boundary's read-modify-write window is the one real TOCTOU).

    Order is fail-safe: the proposal is verified (pending, evidence
    attached, record still a candidate) and the human approval is recorded
    before the store is touched — every failure window leaves the
    least-privileged state, an approved-but-unapplied proposal that is
    visible in the log while the store stays untouched. The store then
    re-checks its own promotion invariants (run coverage, verdict change);
    the boundary applies the promotion, it never bypasses the store's.
    """
    with store_lock(engine.root):
        proposal = log.get(proposal_id)
        if proposal.state is not ApprovalState.PENDING:
            raise ApprovalError(
                f"proposal {proposal_id} is already {proposal.state.value}; "
                "decided proposals are final"
            )
        # Defense in depth: a proposal file that lost its evidence after render
        # (hand-edit, partial write) must not survive the human step either.
        if proposal.evidence_ref is None:
            raise ApprovalError(
                f"proposal {proposal_id} carries no replay evidence; cannot approve"
            )
        record = engine.get(proposal.learning_id)
        if record.status != "candidate":
            raise ApprovalError(
                f"proposal {proposal_id} cannot be approved: {proposal.learning_id} "
                f"is {record.status}, not candidate"
            )
        log.record_approval(proposal_id, decided_by=decided_by, reason=reason)
        promoted = engine.promote(
            proposal.learning_id,
            RecordCounterfactual(
                replay_run_ids=proposal.decision.replay_run_ids,
                verdict_change=True,
                evidence=proposal.evidence_ref,
            ),
        )
        log.record_applied(proposal_id)
        return promoted


def reject_proposal(
    engine: Engine,
    log: ApprovalLog,
    proposal_id: str,
    *,
    decided_by: str,
    reason: str,
) -> LearningRecord:
    """Refuse a proposal and demote the record: rejection is a knowledge verdict.

    The store's advisory review lock is held across the whole window, as in
    :func:`approve_proposal` — a concurrent second decision blocks and then
    fails its own pending re-check.

    The human's identity and reason are recorded on the proposal, and the
    record is demoted to ``dropped`` — the store's rejection state,
    auditable, never deleted. Demotion matches the gate's own
    knowledge-stage rejections and ends the cycle: a dropped record is no
    longer a candidate, so refused knowledge cannot be re-proposed
    unchanged. The rejection is recorded before the demotion, so a demotion
    failure (the record already dropped, say) still shows who refused and
    why.
    """
    with store_lock(engine.root):
        proposal = log.get(proposal_id)
        log.record_rejection(proposal_id, decided_by=decided_by, reason=reason)
        return engine.demote(proposal.learning_id, f"rejected by {decided_by}: {reason}")
