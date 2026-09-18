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

Gate errors neither promote nor demote — infrastructure says nothing about
the knowledge, so the record stays a candidate for retry. Knowledge-stage
rejections demote to ``dropped`` but stay in the store: rejected knowledge
remains auditable, never deleted.
"""

from __future__ import annotations

from evogine.engine.records import LearningRecord
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


class MemoryGate:
    """Objective verifier for candidate learnings, with promotion telemetry.

    ``runner`` supplies the recorded replay results; ``log`` receives one
    decision per review (the V2-1 completeness invariant lives here — a
    verdict that is not recorded never leaves the gate); ``evidence``, when
    given, persists the paired verdicts and the decision's ``evidence_ref``
    points into ``runs/``.
    """

    def __init__(
        self,
        runner: ReplayRunner,
        log: DecisionLog,
        evidence: RunsEvidenceSink | None = None,
        scored_threshold: float = 0.15,
    ) -> None:
        self._runner = runner
        self._log = log
        self._evidence = evidence
        self._scored_threshold = scored_threshold

    def review(
        self, record: LearningRecord, replay_run_ids: list[str]
    ) -> PromotionDecision:
        """Decide ``record`` from its counterfactual replay; record the verdict.

        ``replay_run_ids`` name the queued replay runs to inspect (the run
        ids attached by ``Engine.queue_replay``). The returned
        :class:`PromotionDecision` is already persisted in the log.
        """
        decision, evidence_payload = self._decide(record, replay_run_ids)
        if self._evidence is not None and evidence_payload is not None:
            evidence_ref = self._evidence.write(record.id, evidence_payload)
            decision = decision.model_copy(update={"evidence_ref": evidence_ref})
        self._log.record(decision)
        return decision

    def _decide(
        self, record: LearningRecord, replay_run_ids: list[str]
    ) -> tuple[PromotionDecision, dict[str, object] | None]:
        """Run the stage pipeline; return the decision plus evidence payload."""

        def decision(
            stage: DecisionStage, promoted: bool, reason: str
        ) -> PromotionDecision:
            # Internal consistency: gate builders must emit the vocabulary
            # the log enforces — a drifted reason fails here, not in the report.
            check_reason(stage, reason)
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
                runs.append(self._runner.run(run_id))
            except Exception as exc:  # noqa: BLE001 — fail-closed gate boundary: any
                # infrastructure failure is a GATE_ERROR verdict, never a crash and
                # never a knowledge verdict (documented per-file exception).
                return decision(
                    DecisionStage.GATE_ERROR,
                    False,
                    f"gate_error: replay run {run_id} failed: {exc}",
                ), None

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
            regressions = [b.instance_id for b, w in pairs if w.score < b.score]

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
