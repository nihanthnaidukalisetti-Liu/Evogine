"""Offline unit tests for the code gate.

Covers the acceptance criteria: differential-equivalence verdicts, the
never-worse rule, bytecode verification of claimed refactors, and fail-closed
behavior on evaluator crashes, bad refs, and coverage gaps. The evaluator is
a deterministic stub — no Docker, no live models — while staging is real
(git archive + git apply against this checkout at HEAD), so the tests also
exercise the traversal-safe staging path end to end.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from evogine.gates.code_gate import (
    CaseResult,
    CaseVerdict,
    CodeGate,
    EquivalenceClaim,
    EvalCase,
    GateStage,
)
from evogine.gates.equivalence import differential_flips
from evogine.sandbox.staging import PatchRejected, validate_patch_paths

REPO_ROOT = Path(__file__).resolve().parents[1]

CASE_A = EvalCase(case_id="case-a", command="true")
CASE_B = EvalCase(case_id="case-b", command="true")

#: Generated against the v1 scaffold's src/evogine/__init__.py; comment-only.
COMMENT_ONLY_PATCH = """diff --git a/src/evogine/__init__.py b/src/evogine/__init__.py
index 2ef1563..858599d 100644
--- a/src/evogine/__init__.py
+++ b/src/evogine/__init__.py
@@ -10,3 +10,4 @@ __version__ = "0.1.0"
 from evogine.engine.records import LearningRecord

 __all__ = ["LearningRecord", "__version__"]
+# gate: comment-only refactor probe
"""

#: Changes a docstring constant — bytecode differs, behavior does not.
DOCSTRING_PATCH = '''diff --git a/src/evogine/gates/__init__.py b/src/evogine/gates/__init__.py
index 416d5a8..e33f969 100644
--- a/src/evogine/gates/__init__.py
+++ b/src/evogine/gates/__init__.py
@@ -1 +1 @@
-"""Objective gates: deterministic, independent verdicts on what the swarm proposes."""
+"""Objective gates: deterministic, independent verdicts on what the swarm proposes (probed)."""
'''

ALL_PASS = {"case-a": True, "case-b": True}


class StubEvaluator:
    """Deterministic per-tree outcomes; records every (tree, case) it ran."""

    def __init__(
        self,
        baseline: Mapping[str, bool],
        candidate: Mapping[str, bool],
        *,
        suite_passes: bool = True,
    ) -> None:
        self._baseline = baseline
        self._candidate = candidate
        self._suite_passes = suite_passes
        self.calls: list[tuple[str, str]] = []

    def run_case(self, tree: str, staging: Path, case: EvalCase) -> CaseResult:
        self.calls.append((tree, case.case_id))
        if case.case_id == "suite":
            verdict = CaseVerdict.PASS if self._suite_passes else CaseVerdict.FAIL
            return CaseResult(case_id=case.case_id, verdict=verdict, detail="stub suite")
        outcomes = self._baseline if tree == "baseline" else self._candidate
        passed = outcomes[case.case_id]
        return CaseResult(
            case_id=case.case_id,
            verdict=CaseVerdict.PASS if passed else CaseVerdict.FAIL,
            detail="stub",
        )


def make_gate(evaluator: StubEvaluator) -> CodeGate:
    return CodeGate(evaluator, repo_root=REPO_ROOT, cases=[CASE_A, CASE_B])


def test_accepts_patch_when_all_checks_pass() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed(COMMENT_ONLY_PATCH, "HEAD")
    assert verdict.allowed is True
    assert verdict.stage is GateStage.ACCEPTED
    assert [check.stage for check in verdict.checks] == [
        GateStage.PYTEST,
        GateStage.DIFFERENTIAL,
        GateStage.NEVER_WORSE,
    ]
    assert verdict.baseline_outcomes
    assert verdict.candidate_outcomes


def test_suite_failure_rejects_before_cases_run() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS, suite_passes=False)
    verdict = make_gate(evaluator).evaluate_detailed(COMMENT_ONLY_PATCH, "HEAD")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.PYTEST
    assert evaluator.calls == [("candidate", "suite")]


def test_never_worse_rejects_single_regression() -> None:
    evaluator = StubEvaluator(
        baseline={"case-a": True, "case-b": True},
        candidate={"case-a": False, "case-b": True},
    )
    verdict = make_gate(evaluator).evaluate_detailed(COMMENT_ONLY_PATCH, "HEAD")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.NEVER_WORSE
    assert "case-a" in verdict.reason


def test_improvement_is_not_a_regression() -> None:
    evaluator = StubEvaluator(
        baseline={"case-a": False, "case-b": True},
        candidate={"case-a": True, "case-b": True},
    )
    verdict = make_gate(evaluator).evaluate_detailed(COMMENT_ONLY_PATCH, "HEAD")
    assert verdict.allowed is True
    assert verdict.stage is GateStage.ACCEPTED


def test_refactor_claim_rejected_on_any_flip() -> None:
    evaluator = StubEvaluator(
        baseline={"case-a": True, "case-b": True},
        candidate={"case-a": False, "case-b": True},
    )
    verdict = make_gate(evaluator).evaluate_detailed(
        COMMENT_ONLY_PATCH,
        "HEAD",
        claim=EquivalenceClaim.REFACTOR,
        claimed_files=["src/evogine/__init__.py"],
    )
    assert verdict.allowed is False
    assert verdict.stage is GateStage.DIFFERENTIAL
    assert all(check.stage is not GateStage.BYTECODE for check in verdict.checks)


def test_refactor_claim_accepted_for_comment_only_change() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed(
        COMMENT_ONLY_PATCH,
        "HEAD",
        claim=EquivalenceClaim.REFACTOR,
        claimed_files=["src/evogine/__init__.py"],
    )
    assert verdict.allowed is True, verdict.reason
    bytecode_check = next(c for c in verdict.checks if c.stage is GateStage.BYTECODE)
    assert bytecode_check.passed is True
    assert "bytecode-identical" in bytecode_check.detail


def test_refactor_claim_rejected_when_bytecode_changes() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed(
        DOCSTRING_PATCH,
        "HEAD",
        claim=EquivalenceClaim.REFACTOR,
        claimed_files=["src/evogine/gates/__init__.py"],
    )
    assert verdict.allowed is False
    assert verdict.stage is GateStage.BYTECODE
    assert "bytecode changed" in verdict.reason


def test_refactor_claim_without_files_fails_closed() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed(
        COMMENT_ONLY_PATCH, "HEAD", claim=EquivalenceClaim.REFACTOR
    )
    assert verdict.allowed is False
    assert verdict.stage is GateStage.BYTECODE


def test_fail_closed_when_evaluator_crashes() -> None:
    class ExplodingEvaluator(StubEvaluator):
        def run_case(self, tree: str, staging: Path, case: EvalCase) -> CaseResult:
            raise RuntimeError("evaluator exploded")

    gate = CodeGate(ExplodingEvaluator(ALL_PASS, ALL_PASS), repo_root=REPO_ROOT, cases=[CASE_A])
    verdict = gate.evaluate_detailed(COMMENT_ONLY_PATCH, "HEAD")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.GATE_ERROR
    assert "evaluator exploded" in verdict.reason
    assert gate.evaluate(COMMENT_ONLY_PATCH, "HEAD") is False


def test_fail_closed_on_traversal_ref() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed(COMMENT_ONLY_PATCH, "../../elsewhere")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.GATE_ERROR


def test_fail_closed_on_unapplicable_patch() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed("not a patch at all", "HEAD")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.STAGING


def test_duplicate_case_ids_fail_closed() -> None:
    evaluator = StubEvaluator(ALL_PASS, ALL_PASS)
    verdict = make_gate(evaluator).evaluate_detailed(
        COMMENT_ONLY_PATCH,
        "HEAD",
        cases=[EvalCase(case_id="case-a", command="true"), CASE_A],
    )
    assert verdict.allowed is False
    assert verdict.stage is GateStage.GATE_ERROR


def test_coverage_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="different case sets"):
        differential_flips(
            {"case-a": CaseResult(case_id="case-a", verdict=CaseVerdict.PASS)},
            {
                "case-a": CaseResult(case_id="case-a", verdict=CaseVerdict.PASS),
                "case-b": CaseResult(case_id="case-b", verdict=CaseVerdict.FAIL),
            },
        )


def test_patch_rejection_survives_as_typed_error() -> None:
    with pytest.raises(PatchRejected):
        validate_patch_paths("diff --git a/../x b/../x\n")
