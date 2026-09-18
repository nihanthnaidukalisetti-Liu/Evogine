"""Objective gate #2: counterfactual-replay promotion of learned knowledge.

The memory gate keeps agent-generated knowledge ephemeral until proven
useful. For a candidate record it replays the recorded failure cases cold
(baseline, learning absent) and warm (learning applied) through the
injected :class:`ReplayRunner`, then decides through three knowledge
stages — REPLAY (did the warm verdict actually flip?), MATERIALITY (does
the flip qualify?), NEVER_WORSE (did any case regress?) — plus GATE_ERROR
for infrastructure failures, which is deliberately not a knowledge
verdict. Every verdict is written to the :class:`DecisionLog` before it
is returned: a decided proposal with no decision record is a bug (V2-1).

V2-6: the gate no longer promotes. When the knowledge stages clear, the
verdict becomes an *approval proposal* — record, replay evidence, and the
cleared decision trail, rendered into the :class:`ApprovalLog` — and only
``evogine approvals approve|reject`` applies or refuses it. The evidence
requirement is enforced by the gate regardless of the human: a cleared
review without replay evidence is refused, and the gate has no promotion
path at all, so no promotion can exist without an approved proposal.

V2-8: every decision also leaves a trajectory — the full path a review
took (replay fetches, evidence write, proposal render, verdict), captured
per review and written to the :class:`TrajectoryStore` when one is wired,
so a verdict is always reconstructible from what actually happened.

Gate errors neither promote nor demote — infrastructure says nothing about
the knowledge, so the record stays a candidate for retry. The same holds
for an evidence refusal (the refusal is about the proposal, not the
knowledge). Knowledge-stage rejections demote to ``dropped`` but stay in
the store: rejected knowledge remains auditable, never deleted.
"""

from __future__ import annotations

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.approvals import ApprovalError, ApprovalLog
from evogine.gates.materiality import (
    MaterialityError,
    align_verdicts,
    is_binary,
    is_material,
)
from evogine.gates.replay import ReplayRunner
from evogine.gates.telemetry import (
    DecisionLog,
    DecisionStage,
    PromotionDecision,
    RunsEvidenceSink,
    check_reason,
)
from evogine.gates.trajectory import TrajectoryRecorder, TrajectoryStore


class MemoryGate:
    """Objective verifier for candidate learnings, with promotion telemetry.

    ``runner`` supplies the recorded replay results; ``log`` receives one
    decision per review (the V2-1 completeness invariant lives here — a
    verdict that is not recorded never leaves the gate); ``approvals``
    receives the rendered proposals (the V2-6 boundary — a gate without it
    is misconfigured and clears nothing); ``evidence``, when given,
    persists the paired verdicts so a cleared review can attach them to
    the proposal — without it, nothing can be proposed; ``trajectories``,
    when given, persists the full review trajectory (tool calls included)
    per decision and stamps its store-relative path on the decision's
    ``trajectory_ref`` (V2-8) — the verdict plus the reconstructible path
    to it, the proposal render included.
    """

    def __init__(
        self,
        runner: ReplayRunner,
        log: DecisionLog,
        approvals: ApprovalLog,
        evidence: RunsEvidenceSink | None = None,
        scored_threshold: float = 0.15,
        trajectories: TrajectoryStore | None = None,
    ) -> None:
        self._runner = runner
        self._log = log
        self._approvals = approvals
        self._evidence = evidence
        self._scored_threshold = scored_threshold
        self._trajectories = trajectories

    def review(
        self, record: LearningRecord, replay_run_ids: list[str]
    ) -> PromotionDecision:
        """Decide ``record`` from its counterfactual replay; record the verdict.

        ``replay_run_ids`` name the queued replay runs to inspect (the run
        ids attached by ``Engine.queue_replay``). When the knowledge stages
        clear, the verdict is rendered as a pending approval proposal —
        the record itself is not promoted until
        ``evogine approvals approve`` applies it; without replay evidence
        the proposal is refused in the recorded decision instead. The
        returned :class:`PromotionDecision` is already persisted in the
        log, with its trajectory stamped when a store is wired.
        """
        trajectory = TrajectoryRecorder(gate="memory_gate", proposal_id=record.id)
        decision, evidence_payload = self._decide(record, replay_run_ids, trajectory)
        if self._evidence is not None and evidence_payload is not None:
            evidence_ref = self._evidence.write(record.id, evidence_payload)
            decision = decision.model_copy(update={"evidence_ref": evidence_ref})
            trajectory.tool_call(
                "evidence_sink.write",
                {"proposal_id": record.id},
                result=f"wrote {evidence_ref}",
            )
        if self._cleared(decision):
            decision = self._render_proposal(decision, trajectory)
        if self._trajectories is not None:
            trajectory_ref = self._trajectories.write(record.id, trajectory.freeze())
            decision = decision.model_copy(update={"trajectory_ref": trajectory_ref})
        self._log.record(decision)
        return decision

    @staticmethod
    def _cleared(decision: PromotionDecision) -> bool:
        """True when the knowledge stages cleared and only the human remains."""
        return decision.stage is DecisionStage.NEVER_WORSE and decision.promoted

    def _render_proposal(
        self, cleared: PromotionDecision, trajectory: TrajectoryRecorder
    ) -> PromotionDecision:
        """Turn a cleared verdict into a proposal — or record its refusal.

        The proposal carries the cleared decision trail and the evidence
        ref; the returned decision (what the log records) says what this
        review did: ``proposed`` — or ``refused_no_evidence``, when there
        is no evidence to justify the proposal with. A refusal is not a
        knowledge verdict: the record stays a candidate for a retry with
        evidence wired. Both outcomes are captured in the trajectory.
        """
        detail = cleared.reason.removeprefix("cleared: ")

        def approval(reason: str, evidence_ref: str | None) -> PromotionDecision:
            check_reason(DecisionStage.APPROVAL, reason)
            return cleared.model_copy(
                update={
                    "promoted": False,
                    "stage": DecisionStage.APPROVAL,
                    "reason": reason,
                    "evidence_ref": evidence_ref,
                }
            )

        try:
            proposal = self._approvals.render(
                cleared.learning_id, cleared, cleared.evidence_ref
            )
        except ApprovalError as exc:
            trajectory.tool_call(
                "approvals.render",
                {"proposal_id": cleared.learning_id},
                result=f"refused: {exc}",
            )
            return approval(f"refused_no_evidence: {exc}", None)
        trajectory.tool_call(
            "approvals.render",
            {"proposal_id": cleared.learning_id},
            result=f"rendered {proposal.proposal_id}",
        )
        return approval(f"proposed: {detail}", cleared.evidence_ref)

    def _decide(
        self,
        record: LearningRecord,
        replay_run_ids: list[str],
        trajectory: TrajectoryRecorder,
    ) -> tuple[PromotionDecision, dict[str, object] | None]:
        """Run the stage pipeline; return the decision plus evidence payload."""

        def decision(
            stage: DecisionStage, promoted: bool, reason: str
        ) -> PromotionDecision:
            # Internal consistency: gate builders must emit the vocabulary
            # the log enforces — a drifted reason fails here, not in the report.
            check_reason(stage, reason)
            trajectory.decide(stage.value, promoted=promoted, reason=reason)
            return PromotionDecision(
                proposal_id=record.id,
                learning_id=record.id,
                promoted=promoted,
                stage=stage,
                reason=reason,
                replay_run_ids=list(replay_run_ids),
            )

        if not replay_run_ids:
            return decision(
                DecisionStage.GATE_ERROR, False, "gate_error: no replay runs queued"
            ), None

        runs = []
        for run_id in replay_run_ids:
            try:
                run = self._runner.run(run_id)
            except Exception as exc:  # noqa: BLE001 — fail-closed gate boundary: any
                # infrastructure failure is a GATE_ERROR verdict, never a crash and
                # never a knowledge verdict (documented per-file exception).
                trajectory.tool_call(
                    "replay_runner.run",
                    {"run_id": run_id},
                    result=f"failed: {exc}",
                )
                return decision(
                    DecisionStage.GATE_ERROR,
                    False,
                    f"gate_error: replay run {run_id} failed: {exc}",
                ), None
            trajectory.tool_call(
                "replay_runner.run",
                {"run_id": run_id},
                result=f"{run.arm}: {len(run.verdicts)} verdict(s)",
            )
            runs.append(run)

        baseline = [v for run in runs if run.arm == "baseline" for v in run.verdicts]
        warm = [v for run in runs if run.arm == "warm" for v in run.verdicts]
        missing_arm = "baseline" if not baseline else "warm" if not warm else None
        if missing_arm is not None:
            return decision(
                DecisionStage.GATE_ERROR,
                False,
                f"gate_error: replay runs missing the {missing_arm} arm",
            ), None

        try:
            pairs = align_verdicts(baseline, warm)
            materiality = is_material(baseline, warm, self._scored_threshold)
        except MaterialityError as exc:
            # Arms that do not describe the same case set are a replay
            # infrastructure defect, not a property of the knowledge.
            return decision(DecisionStage.GATE_ERROR, False, f"gate_error: {exc}"), None

        evidence_payload: dict[str, object] = {
            "learning_id": record.id,
            "replay_run_ids": list(replay_run_ids),
            "materiality": materiality.model_dump(),
            "baseline": [v.model_dump() for v in baseline],
            "warm": [v.model_dump() for v in warm],
        }

        if is_binary(baseline):
            has_flip = any(b.passed != w.passed for b, w in pairs)
            regressions = [b.instance_id for b, w in pairs if b.passed and not w.passed]
        else:
            has_flip = any(w.score != b.score for b, w in pairs)
            # is_material has already refused arms that are not fully scored
            # (paired_median_delta raises on a missing score), so the guards
            # below are unreachable in a live gate — they keep the comparison
            # honest under the type system instead of asserting it away.
            regressions = [
                b.instance_id
                for b, w in pairs
                if b.score is not None and w.score is not None and w.score < b.score
            ]

        # Stage 1 — REPLAY: did the warm verdict actually flip?
        if not has_flip:
            count = len(pairs)
            return decision(
                DecisionStage.REPLAY,
                False,
                f"no_flip: 0/{count} replays changed verdict",
            ), evidence_payload

        # Stage 2 — MATERIALITY: does the flip qualify as material?
        if not materiality.material:
            return decision(
                DecisionStage.MATERIALITY,
                False,
                f"not_material: {materiality.detail}",
            ), evidence_payload

        # Stage 3 — NEVER_WORSE: did any evaluated case regress?
        if regressions:
            listed = ", ".join(sorted(regressions))
            return decision(
                DecisionStage.NEVER_WORSE,
                False,
                f"regression: {len(regressions)} case(s) regressed: {listed}",
            ), evidence_payload

        # All knowledge stages cleared — the decision is final here.
        return decision(
            DecisionStage.NEVER_WORSE, True, f"cleared: {materiality.detail}"
        ), evidence_payload


def run_promotion_pass(engine: Engine, gate: MemoryGate) -> list[PromotionDecision]:
    """Run the gate over every candidate in the store and apply the verdict.

    The gate never promotes (V2-6): a cleared candidate comes back as a
    rendered proposal awaiting ``evogine approvals approve|reject`` — the
    pass leaves it a candidate, untouched by the store. Knowledge-stage
    rejections are demoted to ``dropped`` (auditable, never deleted);
    gate errors and evidence refusals leave the record a candidate —
    infrastructure says nothing about the knowledge, so it stays eligible
    for a retry. Every decision leaves a trajectory when the gate has a
    store wired (V2-8). Returns one decision per decided proposal, in
    store order. Every decision is already persisted in the gate's log: a
    decided proposal without a decision record is a bug (V2-1 completeness
    invariant).
    """
    decisions: list[PromotionDecision] = []
    for record in engine.candidates():
        decision = gate.review(record, record.counterfactual.replay_run_ids)
        # Only knowledge-stage rejections move the store. GATE_ERROR is
        # infrastructure (retryable); APPROVAL is the human boundary —
        # proposed or refused, the decision belongs to a person, not a pass.
        if (
            not decision.promoted
            and decision.stage is not DecisionStage.GATE_ERROR
            and decision.stage is not DecisionStage.APPROVAL
        ):
            engine.demote(record.id, decision.reason)
        decisions.append(decision)
    return decisions
