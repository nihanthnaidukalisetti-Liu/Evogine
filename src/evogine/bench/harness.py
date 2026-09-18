"""Warm-versus-cold benchmark harness with variance-aware reporting.

The harness runs each benchmark case twice — cold (no engine knowledge) and
warm (promoted knowledge retrieved for the task context) — under
decay-pressure conditions such as concurrency, retries, time zones, locale,
overflow, partial failure, and integration drift. It reports variance-aware
outcomes and preserves rollout provenance, so the claim "the engine helps"
is always backed by comparable, reproducible evidence. Hidden host-side
invariants and oracle-leak protection live in the case definitions
(:mod:`evogine.bench.cases`), not here.

Replay spend is queued before execution: the harness first builds the full
replay schedule (case × arm × repeat) so the run is a checked plan rather
than a side effect. When a plan declares replay budgets, exceeding schedules
are refused before the first replay is spent — never reconciled after.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from evogine.bench.budgets import BenchPlan, queue_replays
from evogine.bench.cases import BenchCase, default_corpus
from evogine.bench.probes import ContaminationError, recall_probe
from evogine.bench.protocol import (
    ClaimTag,
    EvalProtocol,
    ProtocolCheck,
    ProtocolValidationError,
    pre_run_validate,
)
from evogine.bench.replay import REPEAT_FLOOR, Arm, build_replay_requests
from evogine.bench.rundirs import RunDirectory
from evogine.bench.stats import (
    HeadlineClaim,
    PairedCaseStats,
    headline_claims,
    pair_verdicts,
)
from evogine.engine.records import LearningRecord


class CaseOutcome(BaseModel):
    """One graded execution of one case in one arm, with rollout provenance."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    arm: Arm
    repeat: int
    passed: bool
    output: str
    knowledge_ids: list[str]


class RunArtifacts:
    """CaseOutcome persistence over a run directory (P2 resume).

    Outcomes are keyed by their full replay identity (case, arm, repeat),
    so a resumed run reuses exactly the outcomes already on disk and
    re-executes only what is genuinely missing — a rerun neither duplicates
    spend nor clobbers recorded evidence. The harness stays pure when no
    ``RunArtifacts`` is passed.
    """

    def __init__(self, run: RunDirectory) -> None:
        self._run = run

    @property
    def run(self) -> RunDirectory:
        return self._run

    def load(self, case_id: str, arm: Arm, repeat: int) -> CaseOutcome | None:
        raw = self._run.read_artifact(self._key(case_id, arm, repeat))
        if raw is None:
            return None
        return CaseOutcome.model_validate_json(raw)

    def save(self, outcome: CaseOutcome) -> None:
        self._run.write_artifact(
            self._key(outcome.case_id, outcome.arm, outcome.repeat),
            outcome.model_dump_json(),
        )

    @staticmethod
    def _key(case_id: str, arm: Arm, repeat: int) -> str:
        return f"outcome-{case_id}-{arm}-{repeat:04d}"


class Retriever(Protocol):
    """What the harness needs from the engine: promotion-only retrieval.

    :class:`evogine.engine.store.Engine` satisfies this structurally;
    tests use offline stubs.
    """

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Return promoted records relevant to ``task_context``."""
        ...


class CaseRunner(Protocol):
    """The spend seam: produces a submission for a case in one arm/repeat.

    Production runners drive model clients; unit tests use deterministic
    stubs — the suite never calls live models.
    """

    def execute(
        self,
        case: BenchCase,
        *,
        arm: Arm,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        """Return the submission output for one replay."""
        ...


class ArmSummary(BaseModel):
    """Variance-aware outcome summary for one arm of one case."""

    model_config = ConfigDict(extra="forbid")

    arm: Arm
    outcomes: list[CaseOutcome]
    passes: int
    pass_rate: float
    variance: float
    """Population variance of the binary verdicts across repeats (p·(1−p))."""


class CaseReport(BaseModel):
    """Warm-vs-cold comparison for one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    family: str
    proposal_id: str
    cold: ArmSummary
    warm: ArmSummary
    delta_pass_rate: float
    """``warm.pass_rate - cold.pass_rate``; negative means the case regressed."""
    claim_tag: ClaimTag | None = None
    """Evidence-surface tag (V2-7): ``fresh-holdout`` for the locked held-out
    family, the protocol's benchmark tag otherwise; ``None`` when the run had
    no protocol."""


class RunSummary(BaseModel):
    """Run-level aggregates across all cases."""

    model_config = ConfigDict(extra="forbid")

    repeats: int
    replays: int
    cold_pass_rate: float
    warm_pass_rate: float
    mean_delta_pass_rate: float
    regressions: list[str]
    improvements: list[str]


class BenchReport(BaseModel):
    """The variance-aware report for one bench run."""

    model_config = ConfigDict(extra="forbid")

    repeats: int
    cases: list[CaseReport]
    summary: RunSummary
    claims: list[HeadlineClaim] = Field(default_factory=list)
    """V2-7: every headline number with its clustered 95% bootstrap interval
    and claim tag. Empty unless the run was pre-registered under a protocol."""


class BenchHarness:
    """Warm-vs-cold decay-pressure benchmark runner."""

    def __init__(self, retriever: Retriever, runner: CaseRunner) -> None:
        self._retriever = retriever
        self._runner = runner

    def run(
        self,
        case_ids: Sequence[str] | None = None,
        *,
        repeats: int | None = None,
        corpus: Sequence[BenchCase] | None = None,
        plan: BenchPlan | None = None,
        protocol: EvalProtocol | None = None,
        artifacts: RunArtifacts | None = None,
    ) -> BenchReport:
        """Run ``case_ids`` (default: the full 10-case corpus) cold and warm.

        The full replay schedule is queued before execution so spend is a
        checked plan; a repeat count below the 4-repeat floor is refused.
        When ``plan`` declares replay budgets, an exceeding schedule is
        refused at queue time — before the first replay is spent.

        When ``protocol`` is given (V2-7), the run is pre-registered: the
        protocol is validated against the corpus before anything is spent,
        and explicit arguments that contradict it are refused rather than
        silently overriding it.

        When ``artifacts`` is given (P2), every outcome is recorded into the
        run directory as it is spent, and outcomes already recorded there
        are reused instead of re-executed — rerunning with the same run
        directory resumes it, redoing only missing work, and never
        clobbers recorded evidence. Completion is the caller's call
        (``run.mark_complete()``); the harness never freezes a run itself.
        """
        pool = list(corpus) if corpus is not None else default_corpus()
        if protocol is not None:
            if plan is not None:
                raise ValueError(
                    "plan conflicts: the protocol already pins the harness config — "
                    "pass protocol or plan, not both"
                )
            check = _validate_protocol_pre_run(protocol, pool, case_ids, repeats)
            _refuse_contaminated_holdout(self._retriever, check, pool)
            cases = list(check.cases)
            effective_repeats = protocol.repeats
        else:
            cases, effective_repeats = _resolve_inputs(case_ids, repeats, corpus, plan)
            requests = build_replay_requests(cases, effective_repeats)
            if plan is not None:
                queue_replays(requests, plan.budgets)  # V2-4: refuse before spend
        claim_tags = (
            {case.id: _claim_tag_for(case, protocol) for case in cases}
            if protocol is not None
            else {}
        )
        case_reports: list[CaseReport] = []

        def outcome_for(
            case: BenchCase,
            arm: Arm,
            repeat: int,
            knowledge: list[LearningRecord],
        ) -> CaseOutcome:
            """Reuse a recorded outcome when resuming; spend and record otherwise."""
            if artifacts is not None:
                recorded = artifacts.load(case.id, arm, repeat)
                if recorded is not None:
                    return recorded
            outcome = self._execute(case, arm, repeat, knowledge)
            if artifacts is not None:
                artifacts.save(outcome)
            return outcome

        for case in cases:
            cold_knowledge: list[LearningRecord] = []
            cold_outcomes = [
                outcome_for(case, "cold", repeat, cold_knowledge)
                for repeat in range(effective_repeats)
            ]
            warm_knowledge = list(self._retriever.retrieve(case.task_context))
            warm_outcomes = [
                outcome_for(case, "warm", repeat, warm_knowledge)
                for repeat in range(effective_repeats)
            ]
            case_reports.append(
                _case_report(case, cold_outcomes, warm_outcomes, claim_tags.get(case.id))
            )
        claims = (
            headline_claims(_paired_stats(case_reports), protocol) if protocol is not None else []
        )
        return _build_report(case_reports, effective_repeats, claims)

    def _execute(
        self,
        case: BenchCase,
        arm: Arm,
        repeat: int,
        knowledge: list[LearningRecord],
    ) -> CaseOutcome:
        """Spend one replay: run the case, then grade it with the hidden oracle."""
        output = self._runner.execute(case, arm=arm, knowledge=knowledge, repeat=repeat)
        return CaseOutcome(
            case_id=case.id,
            arm=arm,
            repeat=repeat,
            passed=case.grade(output),
            output=output,
            knowledge_ids=[record.id for record in knowledge],
        )


def _validate_protocol_pre_run(
    protocol: EvalProtocol,
    pool: Sequence[BenchCase],
    case_ids: Sequence[str] | None,
    repeats: int | None,
) -> ProtocolCheck:
    """Run the V2-7 pre-registration gate: nothing is spent until it passes.

    The protocol is the first-class source of run config (mirroring the
    V2-4 plan semantics): explicit arguments that contradict it are refused
    rather than silently overriding it.
    """
    check = pre_run_validate(protocol, pool)
    if case_ids is not None and set(case_ids) != {case.id for case in check.cases}:
        raise ProtocolValidationError(
            ["case_ids conflict: the explicit argument contradicts the protocol"]
        )
    if repeats is not None and repeats != protocol.repeats:
        raise ProtocolValidationError(
            ["repeats conflict: the explicit argument contradicts the protocol"]
        )
    return check


def _refuse_contaminated_holdout(
    retriever: Retriever, check: ProtocolCheck, pool: Sequence[BenchCase]
) -> None:
    """V2-7 recall gate: a contaminated holdout is not fresh — refuse pre-spend.

    The protocol's holdout is only fresh while the store holds nothing
    derived from it. Quoting held-out-unique content means the engine has
    already seen the family, so any fresh-holdout number the run would
    produce is void; spend nothing.
    """
    if not check.heldout_cases:
        return
    probe = recall_probe(retriever, list(check.heldout_cases), corpus=pool)
    if probe.contaminated:
        finding = probe.findings[0]
        raise ContaminationError(
            f"held-out family {check.protocol.held_out_family!r} is contaminated: "
            f"record {finding.record_id!r} quotes held-out tokens "
            f"{finding.matched_tokens} — refusing to run against a holdout "
            "the engine has already seen (nothing was spent)"
        )


def _resolve_inputs(
    case_ids: Sequence[str] | None,
    repeats: int | None,
    corpus: Sequence[BenchCase] | None,
    plan: BenchPlan | None,
) -> tuple[list[BenchCase], int]:
    """Resolve the effective case selection and repeat count for a run.

    A plan is the first-class source of run config (V2-4): explicit
    arguments that contradict a declared plan are refused rather than
    silently overriding it.
    """
    cases = list(corpus) if corpus is not None else default_corpus()
    by_id = {case.id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case ids in corpus")

    plan_ids = plan.case_ids if plan is not None else None
    if case_ids is not None and plan_ids is not None and set(case_ids) != set(plan_ids):
        raise ValueError("case_ids conflict: the explicit argument contradicts the plan")
    selected_ids = case_ids if case_ids is not None else plan_ids

    if selected_ids is not None:
        if not selected_ids:
            raise ValueError("case selection is empty — refusing to run zero cases")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f"case selection contains duplicates: {selected_ids}")
        unknown = sorted(set(selected_ids) - set(by_id))
        if unknown:
            raise ValueError(f"unknown case ids {unknown}; corpus has {sorted(by_id)}")
        cases = [by_id[case_id] for case_id in selected_ids]

    plan_repeats = plan.repeats if plan is not None else None
    if repeats is not None and plan_repeats is not None and repeats != plan_repeats:
        raise ValueError("repeats conflict: the explicit argument contradicts the plan")
    effective_repeats = repeats if repeats is not None else plan_repeats
    if effective_repeats is None:
        effective_repeats = REPEAT_FLOOR
    if effective_repeats < REPEAT_FLOOR:
        raise ValueError(f"repeats={effective_repeats} is below the {REPEAT_FLOOR}-repeat floor")
    return cases, effective_repeats


def _arm_summary(arm: Arm, outcomes: list[CaseOutcome]) -> ArmSummary:
    passes = sum(outcome.passed for outcome in outcomes)
    total = len(outcomes)
    return ArmSummary(
        arm=arm,
        outcomes=outcomes,
        passes=passes,
        pass_rate=passes / total,
        variance=_binary_variance([outcome.passed for outcome in outcomes]),
    )


def _claim_tag_for(case: BenchCase, protocol: EvalProtocol) -> ClaimTag:
    """Held-out cases report as fresh-holdout; benchmark families carry the protocol tag."""
    if protocol.held_out_family is not None and case.family == protocol.held_out_family:
        return ClaimTag.FRESH_HOLDOUT
    return protocol.benchmark_claim_tag


def _paired_stats(case_reports: list[CaseReport]) -> list[PairedCaseStats]:
    """Pair both arms' verdicts per repeat index — identical case instances."""
    return [
        pair_verdicts(
            report.case_id,
            report.family,
            [o.passed for o in sorted(report.cold.outcomes, key=lambda o: o.repeat)],
            [o.passed for o in sorted(report.warm.outcomes, key=lambda o: o.repeat)],
        )
        for report in case_reports
    ]


def _case_report(
    case: BenchCase,
    cold_outcomes: list[CaseOutcome],
    warm_outcomes: list[CaseOutcome],
    claim_tag: ClaimTag | None = None,
) -> CaseReport:
    cold = _arm_summary("cold", cold_outcomes)
    warm = _arm_summary("warm", warm_outcomes)
    return CaseReport(
        case_id=case.id,
        family=case.family,
        proposal_id=case.proposal(),
        cold=cold,
        warm=warm,
        delta_pass_rate=warm.pass_rate - cold.pass_rate,
        claim_tag=claim_tag,
    )


def _build_report(
    case_reports: list[CaseReport],
    repeats: int,
    claims: list[HeadlineClaim] | None = None,
) -> BenchReport:
    """Assemble the run-level summary from the per-case reports."""
    summary = RunSummary(
        repeats=repeats,
        replays=sum(
            len(report.cold.outcomes) + len(report.warm.outcomes) for report in case_reports
        ),
        cold_pass_rate=_mean([report.cold.pass_rate for report in case_reports]),
        warm_pass_rate=_mean([report.warm.pass_rate for report in case_reports]),
        mean_delta_pass_rate=_mean([report.delta_pass_rate for report in case_reports]),
        regressions=sorted(report.case_id for report in case_reports if report.delta_pass_rate < 0),
        improvements=sorted(
            report.case_id for report in case_reports if report.delta_pass_rate > 0
        ),
    )
    return BenchReport(repeats=repeats, cases=case_reports, summary=summary, claims=claims or [])


def _binary_variance(verdicts: Sequence[bool]) -> float:
    """Return the population variance of binary verdicts: p·(1−p)."""
    total = len(verdicts)
    if total == 0:
        raise ValueError("variance of an empty verdict set is undefined")
    p = sum(verdicts) / total
    return p * (1.0 - p)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


__all__ = [
    "REPEAT_FLOOR",
    "Arm",
    "ArmSummary",
    "BenchHarness",
    "BenchReport",
    "CaseOutcome",
    "CaseReport",
    "CaseRunner",
    "Retriever",
    "RunSummary",
]
