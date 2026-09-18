"""Objective gate #1: sandboxed behavioral verification of code changes.

The code gate decides whether a builder-proposed change may land, and it
never trusts the proposing agent's own claims. Its checks, per the v1 design:
a sandboxed pytest run inside the locked-down execution sandbox; differential
equivalence, ensuring the patch is behavior-preserving where it claims to be;
and a never-worse check, ensuring the patched candidate is not worse than the
baseline on any evaluated case. Patches claiming to be pure refactors must
additionally prove it in bytecode. A patch that fails any check is rejected
outright — the gate is objective, not advisory — and the gate fails CLOSED:
any crash or infrastructure failure is a rejection, never an acceptance.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from evogine.gates.equivalence import (
    CaseResult,
    CaseVerdict,
    bytecode_signature,
    differential_flips,
    never_worse_regression,
)
from evogine.gates.wallclock import crosscheck_too_fast_pass
from evogine.sandbox.docker_runner import SandboxResult
from evogine.sandbox.sanitize import sanitize_output
from evogine.sandbox.staging import (
    PatchRejected,
    StagingError,
    apply_patch,
    create_staging_root,
    extract_baseline,
    validate_relative_path,
)


class GateStage(str, Enum):
    """Where a verdict was decided — the check that stopped (or accepted) it."""

    STAGING = "staging"
    PYTEST = "pytest"
    DIFFERENTIAL = "differential"
    BYTECODE = "bytecode"
    NEVER_WORSE = "never_worse"
    ACCEPTED = "accepted"
    GATE_ERROR = "gate_error"


class EquivalenceClaim(str, Enum):
    """What the patch claims about its own behavior change.

    ``REFACTOR`` claims behavior preservation: the gate then requires both a
    zero-flip differential run AND bytecode-identical claimed files. Claim it
    only for comment/formatting-level edits — anything that changes bytecode
    (docstrings, annotations, code) must be claimed as ``NONE`` instead, where
    the never-worse rule governs.
    """

    NONE = "none"
    REFACTOR = "refactor"


class EvalCase(BaseModel):
    """One behavioral case: a command run inside the sandbox per tree."""

    case_id: str
    command: str
    env: dict[str, str] = Field(default_factory=dict)
    # P2 wall-clock cross-check: a PASS must not arrive faster than the case
    # could genuinely execute. ``None`` disables the check for the case; the
    # pinned suite always carries a floor (see DEFAULT_SUITE).
    min_pass_seconds: float | None = Field(default=None, gt=0)


class GateCheckResult(BaseModel):
    """One check's typed outcome for the audit trail."""

    stage: GateStage
    passed: bool
    detail: str


class GateVerdict(BaseModel):
    """Typed, auditable gate decision; ``stage`` names the deciding check."""

    allowed: bool
    stage: GateStage
    reason: str
    checks: list[GateCheckResult] = Field(default_factory=list)
    baseline_outcomes: list[CaseResult] = Field(default_factory=list)
    candidate_outcomes: list[CaseResult] = Field(default_factory=list)


class CaseEvaluator(Protocol):
    """Runs one case against one staged tree (``tree``: 'baseline'|'candidate')."""

    def run_case(self, tree: str, staging: Path, case: EvalCase) -> CaseResult: ...


class SandboxRun(Protocol):
    """What a case evaluator needs from the sandbox: one sanitized command run.

    :class:`DockerSandbox` satisfies this structurally; offline tests stub it
    with canned :class:`SandboxResult` values.
    """

    def run(
        self,
        command: str,
        *,
        staging_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> SandboxResult: ...


class DockerCaseEvaluator:
    """CaseEvaluator that executes cases inside the Docker sandbox.

    Applies the verifier's wall-clock cross-check (P2): a PASS the sandbox
    reports faster than the case's ``min_pass_seconds`` floor is a verifier
    failure and fails closed — the case is recorded as failed with the
    cross-check reason, never as a pass.
    """

    def __init__(self, sandbox: SandboxRun) -> None:
        self._sandbox = sandbox

    def run_case(self, tree: str, staging: Path, case: EvalCase) -> CaseResult:
        result = self._sandbox.run(case.command, staging_dir=staging, env=case.env)
        duration_ms = float(result.duration_ms)
        if result.exit_code == 0 and not result.timed_out:
            if case.min_pass_seconds is not None:
                too_fast = crosscheck_too_fast_pass(
                    verdict=CaseVerdict.PASS,
                    duration_ms=duration_ms,
                    min_pass_seconds=case.min_pass_seconds,
                )
                if too_fast is not None:
                    return CaseResult(
                        case_id=case.case_id,
                        verdict=CaseVerdict.FAIL,
                        detail=too_fast,
                        duration_ms=duration_ms,
                    )
            return CaseResult(
                case_id=case.case_id,
                verdict=CaseVerdict.PASS,
                detail="exit 0",
                duration_ms=duration_ms,
            )
        detail = f"exit {result.exit_code}" + (" (timed out)" if result.timed_out else "")
        tail = result.stderr or result.stdout
        if tail:
            detail += f"; output tail: {tail[-200:]}"
        return CaseResult(
            case_id=case.case_id,
            verdict=CaseVerdict.FAIL,
            detail=detail,
            duration_ms=duration_ms,
        )


#: The pinned suite, run on the candidate tree inside the sandbox. Its
#: 1s pass floor is deliberate (P2 wall-clock cross-check): interpreter
#: startup, dependency imports, and collection alone cost more than a
#: second, so a sub-second "suite green" is fabricated or hollow (zero
#: tests collected) — and the gate fails it closed.
DEFAULT_SUITE = EvalCase(
    case_id="suite",
    command="python -m pytest -q -p no:cacheprovider",
    env={"PYTHONPATH": "src"},
    min_pass_seconds=1.0,
)


class CodeGate:
    """Objective verifier for candidate code changes.

    ``repo_root`` is the git checkout the gate archives baselines from (it
    defaults to the current directory); ``cases`` are the evaluated cases the
    differential and never-worse checks run on every evaluation.
    """

    def __init__(
        self,
        evaluator: CaseEvaluator,
        *,
        repo_root: Path | None = None,
        cases: Sequence[EvalCase] = (),
        suite: EvalCase | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._repo_root = repo_root or Path.cwd()
        self._cases = list(cases)
        self._suite = suite or DEFAULT_SUITE

    def evaluate(self, candidate_patch: str, baseline_ref: str) -> bool:
        """Scaffold v1 contract: True iff the patch passes every check.

        Never raises — a gate crash rejects the patch (fail-closed).
        """
        return self.evaluate_detailed(candidate_patch, baseline_ref).allowed

    def evaluate_detailed(
        self,
        candidate_patch: str,
        baseline_ref: str,
        *,
        claim: EquivalenceClaim = EquivalenceClaim.NONE,
        claimed_files: Sequence[str] = (),
        cases: Sequence[EvalCase] | None = None,
    ) -> GateVerdict:
        """Full typed verdict. Any internal failure yields a GATE_ERROR rejection."""
        chosen = list(cases if cases is not None else self._cases)
        try:
            return self._evaluate(candidate_patch, baseline_ref, claim, claimed_files, chosen)
        except Exception as exc:
            reason = f"gate crashed, failing closed: {type(exc).__name__}: " + sanitize_output(
                str(exc), max_chars=200
            )
            return GateVerdict(
                allowed=False,
                stage=GateStage.GATE_ERROR,
                reason=reason,
                checks=[GateCheckResult(stage=GateStage.GATE_ERROR, passed=False, detail=reason)],
            )

    def _evaluate(
        self,
        candidate_patch: str,
        baseline_ref: str,
        claim: EquivalenceClaim,
        claimed_files: Sequence[str],
        cases: list[EvalCase],
    ) -> GateVerdict:
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("duplicate case_id in evaluated cases")
        root = create_staging_root()
        try:
            baseline_tree = root / "baseline"
            candidate_tree = root / "candidate"
            try:
                extract_baseline(self._repo_root, baseline_ref, baseline_tree)
                extract_baseline(self._repo_root, baseline_ref, candidate_tree)
                apply_patch(candidate_patch, candidate_tree)
            except PatchRejected as exc:
                return self._reject(GateStage.STAGING, f"patch rejected before staging: {exc}", [])
            except StagingError as exc:
                return self._gate_error(exc)

            # Check 1: the pinned suite must be green on the patched tree.
            suite_result = self._evaluator.run_case("candidate", candidate_tree, self._suite)
            if suite_result.verdict is not CaseVerdict.PASS:
                return self._reject(
                    GateStage.PYTEST,
                    f"suite failed on candidate: {suite_result.detail}",
                    [],
                    candidate_outcomes={self._suite.case_id: suite_result},
                )
            checks = [
                GateCheckResult(
                    stage=GateStage.PYTEST,
                    passed=True,
                    detail=f"suite green ({self._suite.case_id})",
                )
            ]

            baseline_outcomes = {
                case.case_id: self._evaluator.run_case("baseline", baseline_tree, case)
                for case in cases
            }
            candidate_outcomes = {
                case.case_id: self._evaluator.run_case("candidate", candidate_tree, case)
                for case in cases
            }

            # Check 2: differential equivalence — behavior-preserving where
            # the patch claims to be.
            flips = differential_flips(baseline_outcomes, candidate_outcomes)
            if claim is EquivalenceClaim.REFACTOR:
                if flips:
                    flip_ids = [flip.case_id for flip in flips]
                    return self._reject(
                        GateStage.DIFFERENTIAL,
                        f"refactor claim contradicted: behavior changed on {flip_ids}",
                        checks,
                        baseline_outcomes,
                        candidate_outcomes,
                    )
                checks.append(
                    GateCheckResult(
                        stage=GateStage.DIFFERENTIAL,
                        passed=True,
                        detail="claimed behavior-preserving: all evaluated cases identical",
                    )
                )
                # Check 3: bytecode verification of the claimed refactor.
                bc_passed, bc_detail = self._verify_bytecode(
                    baseline_tree, candidate_tree, claimed_files
                )
                checks.append(
                    GateCheckResult(stage=GateStage.BYTECODE, passed=bc_passed, detail=bc_detail)
                )
                if not bc_passed:
                    return self._reject(
                        GateStage.BYTECODE,
                        f"bytecode verification failed: {bc_detail}",
                        checks,
                        baseline_outcomes,
                        candidate_outcomes,
                    )
            else:
                changed = (
                    f"{len(flips)} case(s) changed behavior"
                    if flips
                    else "no case changed behavior"
                )
                checks.append(
                    GateCheckResult(
                        stage=GateStage.DIFFERENTIAL,
                        passed=True,
                        detail=f"no equivalence claim ({changed}; never-worse governs)",
                    )
                )

            # Check 4: never-worse across the evaluated cases.
            regression = never_worse_regression(baseline_outcomes, candidate_outcomes)
            if regression is not None:
                return self._reject(
                    GateStage.NEVER_WORSE,
                    f"case '{regression.case_id}' regressed: baseline passed, candidate failed",
                    checks,
                    baseline_outcomes,
                    candidate_outcomes,
                )
            checks.append(
                GateCheckResult(
                    stage=GateStage.NEVER_WORSE,
                    passed=True,
                    detail="no evaluated case regressed"
                    if cases
                    else "no evaluated cases configured",
                )
            )
            return GateVerdict(
                allowed=True,
                stage=GateStage.ACCEPTED,
                reason="all checks passed",
                checks=checks,
                baseline_outcomes=list(baseline_outcomes.values()),
                candidate_outcomes=list(candidate_outcomes.values()),
            )
        finally:
            # Staging cleanup is best-effort: it holds no verdict evidence
            # (results are already typed into the verdict) and a leftover 0700
            # temp dir cannot affect later evaluations.
            shutil.rmtree(root, ignore_errors=True)

    def _verify_bytecode(
        self, baseline_tree: Path, candidate_tree: Path, claimed_files: Sequence[str]
    ) -> tuple[bool, str]:
        """Every claimed file must compile to identical bytecode on both trees."""
        if not claimed_files:
            return False, "refactor claim named no files — nothing verified, rejected (fail-closed)"
        for relative in claimed_files:
            validate_relative_path(relative)
            baseline_file = baseline_tree / relative
            candidate_file = candidate_tree / relative
            if not baseline_file.is_file() or not candidate_file.is_file():
                return False, f"claimed file missing on a tree: {relative}"
            if bytecode_signature(baseline_file.read_text()) != bytecode_signature(
                candidate_file.read_text()
            ):
                return False, f"bytecode changed in claimed file: {relative}"
        return True, f"{len(claimed_files)} claimed file(s) bytecode-identical"

    @staticmethod
    def _reject(
        stage: GateStage,
        reason: str,
        checks: list[GateCheckResult],
        baseline_outcomes: Mapping[str, CaseResult] | None = None,
        candidate_outcomes: Mapping[str, CaseResult] | None = None,
    ) -> GateVerdict:
        return GateVerdict(
            allowed=False,
            stage=stage,
            reason=reason,
            checks=[
                *checks,
                GateCheckResult(stage=stage, passed=False, detail=reason),
            ],
            baseline_outcomes=(list(baseline_outcomes.values()) if baseline_outcomes else []),
            candidate_outcomes=(list(candidate_outcomes.values()) if candidate_outcomes else []),
        )

    @staticmethod
    def _gate_error(exc: Exception) -> GateVerdict:
        reason = f"gate infrastructure failed: {sanitize_output(str(exc), max_chars=200)}"
        return GateVerdict(
            allowed=False,
            stage=GateStage.GATE_ERROR,
            reason=reason,
            checks=[GateCheckResult(stage=GateStage.GATE_ERROR, passed=False, detail=reason)],
        )
