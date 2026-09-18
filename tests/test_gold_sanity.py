"""Gold-sanity tests: a generated test judges nothing until it survives gold.

The V2-8 acceptance: a rigged test fixture — one that waves everything
through — is rejected. Both rig shapes are covered: the always-pass test
(passes the known-broken reference, so it discriminates nothing) and the
test that fails the known-good solution. Honest tests that discriminate
gold from broken are admitted; evaluator crashes fail closed. All offline;
the evaluator is a deterministic stub — no Docker, no live models.
"""

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from evogine.gates.code_gate import CaseResult, CaseVerdict, EvalCase
from evogine.gates.gold_sanity import (
    GeneratedTest,
    GoldSanityGate,
    GoldStage,
)


class StubEvaluator:
    """Deterministic per-tree outcomes; records every (tree, case) it ran.

    ``crash_on`` names trees where the evaluator raises, exercising the
    fail-closed path without Docker.
    """

    def __init__(
        self,
        outcomes: Mapping[str, Mapping[str, bool]],
        *,
        crash_on: frozenset[str] = frozenset(),
    ) -> None:
        self._outcomes = {k: dict(v) for k, v in outcomes.items()}
        self._crash_on = crash_on
        self.calls: list[tuple[str, str]] = []

    def run_case(self, tree: str, staging: Path, case: EvalCase) -> CaseResult:
        self.calls.append((tree, case.case_id))
        if tree in self._crash_on:
            raise RuntimeError(f"evaluator unavailable for tree {tree}")
        passed = self._outcomes[tree][case.case_id]
        return CaseResult(
            case_id=case.case_id,
            verdict=CaseVerdict.PASS if passed else CaseVerdict.FAIL,
            detail="stub",
        )


def generated_test(test_id: str = "gen-check-1") -> GeneratedTest:
    return GeneratedTest(
        test_id=test_id,
        command="python -m pytest tests/test_generated.py -q",
        proposed_by="reflector",
    )


def make_gate(outcomes: Mapping[str, Mapping[str, bool]], **kw: object) -> GoldSanityGate:
    evaluator = StubEvaluator(outcomes, **kw)  # type: ignore[arg-type]
    return GoldSanityGate(
        evaluator,  # type: ignore[arg-type]
        gold_tree=Path("/staging/gold"),
        broken_tree=Path("/staging/broken"),
    )


def test_honest_generated_test_is_admitted():
    """Passes the known-good solution, fails the known-broken one — admitted."""
    test = generated_test()
    gate = make_gate(
        {
            "gold": {f"gold-sanity:{test.test_id}": True},
            "broken": {f"gold-sanity:{test.test_id}": False},
        }
    )
    verdict = gate.admit(test)

    assert verdict.admitted
    assert verdict.stage is GoldStage.ADMITTED
    assert verdict.gold_result is not None and verdict.broken_result is not None
    assert verdict.gold_result.verdict is CaseVerdict.PASS
    assert verdict.broken_result.verdict is CaseVerdict.FAIL


def test_rigged_always_pass_fixture_rejected():
    """The V2-8 acceptance: a rigged fixture that passes both trees is refused.

    An always-pass test cannot judge anything — it would wave every defect
    through, including the one it was generated to catch.
    """
    test = generated_test("rigged-pass")
    case_id = f"gold-sanity:{test.test_id}"
    gate = make_gate({"gold": {case_id: True}, "broken": {case_id: True}})
    verdict = gate.admit(test)

    assert not verdict.admitted
    assert verdict.stage is GoldStage.NO_DISCRIMINATION
    assert "discriminates nothing" in verdict.reason
    assert verdict.broken_result is not None
    assert verdict.broken_result.verdict is CaseVerdict.PASS


def test_rigged_fails_gold_fixture_rejected_and_broken_run_not_spent():
    """A test that fails the known-good solution is refused — gold runs first."""
    test = generated_test("rigged-fail")
    case_id = f"gold-sanity:{test.test_id}"
    evaluator = StubEvaluator({"gold": {case_id: False}, "broken": {case_id: True}})
    gate = GoldSanityGate(
        evaluator, gold_tree=Path("/staging/gold"), broken_tree=Path("/staging/broken")
    )
    verdict = gate.admit(test)

    assert not verdict.admitted
    assert verdict.stage is GoldStage.FAILS_GOLD
    assert "fails the known-good solution" in verdict.reason
    assert verdict.broken_result is None
    # Literal gold-first: the known-broken run is never spent on a test
    # that cannot even pass the correct behavior.
    assert [tree for tree, _ in evaluator.calls] == ["gold"]


def test_evaluator_crash_fails_closed():
    """An evaluator crash is a refusal, never an admission."""
    test = generated_test("crashing")
    case_id = f"gold-sanity:{test.test_id}"
    gate = make_gate(
        {"gold": {case_id: True}, "broken": {case_id: False}},
        crash_on=frozenset({"gold"}),
    )
    verdict = gate.admit(test)
    assert not verdict.admitted
    assert verdict.stage is GoldStage.GATE_ERROR
    assert "fail-closed" in verdict.reason


def test_known_broken_run_crash_fails_closed():
    """Same fail-closed rule when the crash hits the second run."""
    test = generated_test("crashing-late")
    case_id = f"gold-sanity:{test.test_id}"
    gate = make_gate(
        {"gold": {case_id: True}, "broken": {case_id: False}},
        crash_on=frozenset({"broken"}),
    )
    verdict = gate.admit(test)
    assert not verdict.admitted
    assert verdict.stage is GoldStage.GATE_ERROR


def test_evaluated_case_names_the_generated_test():
    """The case id carries the test id, so evidence is traceable to its judge."""
    test = generated_test("traceable-check")
    case_id = f"gold-sanity:{test.test_id}"
    evaluator = StubEvaluator({"gold": {case_id: True}, "broken": {case_id: False}})
    gate = GoldSanityGate(
        evaluator, gold_tree=Path("/staging/gold"), broken_tree=Path("/staging/broken")
    )
    verdict = gate.admit(test)

    assert verdict.admitted
    assert all(ran_case_id == case_id for _, ran_case_id in evaluator.calls)


def test_generated_test_refuses_extra_fields():
    """Generated tests are typed and closed: unknown fields are refused."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GeneratedTest.model_validate(
            {
                "test_id": "x",
                "command": "true",
                "proposed_by": "reflector",
                "sneaky": "field",
            }
        )
