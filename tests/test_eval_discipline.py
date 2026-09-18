"""V2-7 integration: protocol-driven runs emit tagged, intervalled claims.

These are the CI-emission tests on the bench report: a run under a
pre-registered protocol must carry every headline number with a clustered
bootstrap interval and a claim tag, deterministically, while a run without
a protocol keeps the legacy point-estimate shape.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from evogine.bench.budgets import BenchPlan, ReplayBudgets
from evogine.bench.cases import BenchCase
from evogine.bench.harness import BenchHarness
from evogine.bench.protocol import (
    AnalysisPlan,
    ClaimTag,
    EvalProtocol,
    ProtocolValidationError,
    fingerprint_corpus,
)
from evogine.engine.records import LearningRecord


class ScriptedRunner:
    """Deterministic runner stub: (case_id, arm) -> per-repeat outputs."""

    def __init__(self, outputs: dict[tuple[str, str], list[str]] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, str]] = []

    def execute(
        self,
        case: BenchCase,
        *,
        arm: str,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        self.calls.append((case.id, arm))
        sequence = self.outputs.get((case.id, arm), ["ok"])
        return sequence[repeat % len(sequence)]


class StubRetriever:
    """Promotion-only retrieval stub, offline by construction."""

    def __init__(self, records: dict[str, list[LearningRecord]] | None = None) -> None:
        self.records = records or {}

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        return list(self.records.get(task_context, []))[:k]


def _case(case_id: str, family: str, invariant: Callable[[str], bool]) -> BenchCase:
    return BenchCase(
        id=case_id,
        family=family,
        task_context=f"context-for-{case_id}",
        hidden_invariant=invariant,
    )


def _discipline_cases() -> list[BenchCase]:
    return [
        _case("alpha", "fam-a", lambda output: "ok" in output),
        _case("beta", "fam-a", lambda output: "perfect" in output),
        _case("gamma", "holdout-fam", lambda output: "fix" in output.lower()),
    ]


def _protocol(corpus: list[BenchCase], **overrides: object) -> EvalProtocol:
    fields: dict[str, object] = {
        "name": "warm-vs-cold discipline",
        "corpus_fingerprint": fingerprint_corpus(corpus),
        "prompt_pack": "pack-2026-09-18",
        "repeats": 4,
        "budgets": ReplayBudgets(
            max_per_proposal=999, max_per_case_family=999, max_per_run=999
        ),
        "analysis": AnalysisPlan(bootstrap_iterations=50, seed=42),
        "held_out_family": "holdout-fam",
    }
    fields.update(overrides)
    return EvalProtocol.model_validate(fields)


def _harness(
    outputs: dict[tuple[str, str], list[str]] | None = None,
) -> tuple[BenchHarness, ScriptedRunner]:
    runner = ScriptedRunner(outputs)
    return BenchHarness(StubRetriever(), runner), runner


def test_run_under_protocol_emits_tagged_intervals() -> None:
    cases = _discipline_cases()
    harness, _ = _harness(
        {
            ("beta", "cold"): ["no"] * 4,
            ("beta", "warm"): ["perfect"] * 4,
            ("gamma", "warm"): ["fix"] * 4,
        }
    )

    report = harness.run(corpus=cases, protocol=_protocol(cases))

    assert report.claims, "a protocol run must emit claims"
    assert all(claim.claim_tag in ClaimTag for claim in report.claims)

    run_delta = next(
        c for c in report.claims if c.metric == "delta_pass_rate" and c.scope == "run"
    )
    # Run-level numbers cover benchmark families only: beta's +1.0 over two
    # fam-a cases averages to +0.5, and gamma (holdout) is excluded.
    assert run_delta.value == pytest.approx(0.5)
    assert run_delta.claim_tag is ClaimTag.CONTAMINABLE_BENCHMARK

    holdout_warm = next(
        c
        for c in report.claims
        if c.metric == "warm_pass_rate" and c.scope == "family:holdout-fam"
    )
    assert holdout_warm.value == pytest.approx(1.0)
    assert holdout_warm.claim_tag is ClaimTag.FRESH_HOLDOUT

    margins = {
        (claim.scope, claim.never_worse_holds)
        for claim in report.claims
        if claim.metric == "never_worse_margin"
    }
    assert ("run", True) in margins
    assert ("family:fam-a", True) in margins
    assert ("family:holdout-fam", True) in margins


def test_per_case_reports_carry_their_evidence_tag() -> None:
    cases = _discipline_cases()
    harness, _ = _harness()

    report = harness.run(corpus=cases, protocol=_protocol(cases))

    tags = {report.case_id: report.claim_tag for report in report.cases}
    assert tags == {
        "alpha": ClaimTag.CONTAMINABLE_BENCHMARK,
        "beta": ClaimTag.CONTAMINABLE_BENCHMARK,
        "gamma": ClaimTag.FRESH_HOLDOUT,
    }


def test_claim_emission_is_deterministic_run_to_run() -> None:
    cases = _discipline_cases()
    harness, _ = _harness()
    protocol = _protocol(cases)

    first = harness.run(corpus=cases, protocol=protocol)
    second = harness.run(corpus=cases, protocol=protocol)

    assert [claim.model_dump() for claim in first.claims] == [
        claim.model_dump() for claim in second.claims
    ]


def test_run_without_protocol_keeps_the_point_estimate_shape() -> None:
    cases = _discipline_cases()
    harness, _ = _harness()

    report = harness.run([case.id for case in cases], corpus=cases)

    assert report.claims == []
    assert all(case_report.claim_tag is None for case_report in report.cases)


def test_protocol_and_plan_are_mutually_exclusive() -> None:
    cases = _discipline_cases()
    harness, _ = _harness()
    plan = BenchPlan(
        name="plan",
        repeats=4,
        budgets=ReplayBudgets(
            max_per_proposal=999, max_per_case_family=999, max_per_run=999
        ),
    )

    with pytest.raises(ValueError, match="pass protocol or plan, not both"):
        harness.run(corpus=cases, plan=plan, protocol=_protocol(cases))


def test_contradicting_run_arguments_are_refused() -> None:
    cases = _discipline_cases()
    harness, _ = _harness()

    with pytest.raises(ProtocolValidationError, match="repeats conflict"):
        harness.run(corpus=cases, repeats=6, protocol=_protocol(cases))

    with pytest.raises(ProtocolValidationError, match="case_ids conflict"):
        harness.run(["alpha"], corpus=cases, protocol=_protocol(cases))


def test_budget_refusal_under_protocol_spends_nothing() -> None:
    cases = _discipline_cases()
    harness, runner = _harness()
    protocol = _protocol(
        cases,
        budgets=ReplayBudgets(max_per_proposal=999, max_per_case_family=999, max_per_run=4),
    )

    with pytest.raises(ProtocolValidationError, match="budgets refuse"):
        harness.run(corpus=cases, protocol=protocol)

    assert runner.calls == []
