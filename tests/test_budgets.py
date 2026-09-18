"""V2-4 budget tests: queue-time refusal before spend, and plan-config parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evogine.bench.budgets import (
    BenchPlan,
    BudgetExceededError,
    BudgetScope,
    ReplayBudgets,
    load_plan,
    queue_replays,
)
from evogine.bench.cases import BenchCase
from evogine.bench.harness import BenchHarness
from evogine.bench.replay import build_replay_requests


class CountingRunner:
    """Records every execution; a refusal must leave it untouched."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, case: BenchCase, **_: object) -> str:
        self.calls.append(case.id)
        return "ok"


class StubRetriever:
    def retrieve(self, task_context: str, k: int = 5) -> list[object]:
        return []


def _budget_cases() -> list[BenchCase]:
    """Three stub cases: alpha+beta in fam-a, gamma in fam-b.

    A full run at 4 repeats spends 24 replays: 16 in fam-a, 8 in fam-b, and
    8 per proposal (alpha, beta, gamma).
    """
    return [
        BenchCase(
            id="alpha",
            family="fam-a",
            task_context="ctx-alpha",
            hidden_invariant=lambda output: True,
        ),
        BenchCase(
            id="beta",
            family="fam-a",
            task_context="ctx-beta",
            hidden_invariant=lambda output: True,
        ),
        BenchCase(
            id="gamma",
            family="fam-b",
            task_context="ctx-gamma",
            hidden_invariant=lambda output: True,
        ),
    ]


def _budgets(
    per_proposal: int = 99, per_family: int = 99, per_run: int = 99
) -> ReplayBudgets:
    return ReplayBudgets(
        max_per_proposal=per_proposal,
        max_per_case_family=per_family,
        max_per_run=per_run,
    )


def _plan(
    budgets: ReplayBudgets | None = None,
    repeats: int = 4,
    case_ids: list[str] | None = None,
) -> BenchPlan:
    return BenchPlan(
        name="test-plan",
        repeats=repeats,
        budgets=budgets or _budgets(),
        case_ids=case_ids,
    )


# ---------------------------------------------------------------------------
# Queue-time refusal: the run refuses to exceed, before any spend.
# ---------------------------------------------------------------------------


def test_refusal_per_run_happens_before_any_spend() -> None:
    runner = CountingRunner()
    plan = _plan(_budgets(per_run=23))

    with pytest.raises(BudgetExceededError) as excinfo:
        BenchHarness(StubRetriever(), runner).run(corpus=_budget_cases(), plan=plan)  # type: ignore[arg-type]

    assert excinfo.value.scope is BudgetScope.RUN
    assert excinfo.value.required == 24
    assert excinfo.value.allowed == 23
    assert runner.calls == []  # refused before spend — never reconciled after


def test_refusal_per_case_family_happens_before_any_spend() -> None:
    runner = CountingRunner()
    plan = _plan(_budgets(per_family=15))

    with pytest.raises(BudgetExceededError) as excinfo:
        BenchHarness(StubRetriever(), runner).run(corpus=_budget_cases(), plan=plan)  # type: ignore[arg-type]

    assert excinfo.value.scope is BudgetScope.CASE_FAMILY
    assert excinfo.value.key == "fam-a"
    assert excinfo.value.required == 16
    assert excinfo.value.allowed == 15
    assert runner.calls == []


def test_refusal_per_proposal_happens_before_any_spend() -> None:
    runner = CountingRunner()
    plan = _plan(_budgets(per_proposal=7))

    with pytest.raises(BudgetExceededError) as excinfo:
        BenchHarness(StubRetriever(), runner).run(corpus=_budget_cases(), plan=plan)  # type: ignore[arg-type]

    assert excinfo.value.scope is BudgetScope.PROPOSAL
    assert excinfo.value.key == "alpha"  # first proposal over budget, sorted
    assert excinfo.value.required == 8
    assert excinfo.value.allowed == 7
    assert runner.calls == []


def test_shared_proposal_id_aggregates_across_cases() -> None:
    cases = _budget_cases()
    shared = [case.model_copy(update={"proposal_id": "shared-1"}) for case in cases[:2]]
    runner = CountingRunner()
    plan = _plan(_budgets(per_proposal=15))

    with pytest.raises(BudgetExceededError) as excinfo:
        BenchHarness(StubRetriever(), runner).run(  # type: ignore[arg-type]
            corpus=[*shared, cases[2]], plan=plan
        )

    assert excinfo.value.scope is BudgetScope.PROPOSAL
    assert excinfo.value.key == "shared-1"
    assert excinfo.value.required == 16
    assert runner.calls == []


def test_run_scope_is_checked_first_when_all_budgets_exceeded() -> None:
    runner = CountingRunner()
    plan = _plan(_budgets(per_proposal=1, per_family=1, per_run=1))

    with pytest.raises(BudgetExceededError) as excinfo:
        BenchHarness(StubRetriever(), runner).run(corpus=_budget_cases(), plan=plan)  # type: ignore[arg-type]

    assert excinfo.value.scope is BudgetScope.RUN
    assert runner.calls == []


def test_exact_agreement_with_budgets_spends_the_full_run() -> None:
    runner = CountingRunner()
    plan = _plan(_budgets(per_proposal=8, per_family=16, per_run=24))

    report = BenchHarness(StubRetriever(), runner).run(  # type: ignore[arg-type]
        corpus=_budget_cases(), plan=plan
    )

    assert report.summary.replays == 24
    assert len(runner.calls) == 24


def test_queue_replays_returns_an_inspectable_schedule() -> None:
    requests = build_replay_requests(_budget_cases(), 4)

    schedule = queue_replays(requests, _budgets())

    assert schedule.total() == 24
    assert schedule.by_family() == {"fam-a": 16, "fam-b": 8}
    assert schedule.by_proposal() == {"alpha": 8, "beta": 8, "gamma": 8}


# ---------------------------------------------------------------------------
# Plan config parsing.
# ---------------------------------------------------------------------------

VALID_YAML = """\
name: v2-bench
repeats: 4
budgets:
  max_per_proposal: 8
  max_per_case_family: 16
  max_per_run: 24
"""


def _plan_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_plan_parses_declared_budgets(tmp_path: Path) -> None:
    plan = load_plan(_plan_file(tmp_path, VALID_YAML))

    assert plan.name == "v2-bench"
    assert plan.repeats == 4
    assert plan.budgets.max_per_proposal == 8
    assert plan.budgets.max_per_case_family == 16
    assert plan.budgets.max_per_run == 24
    assert plan.case_ids is None


def test_load_plan_accepts_case_subset(tmp_path: Path) -> None:
    plan = load_plan(_plan_file(tmp_path, VALID_YAML + "case_ids:\n  - alpha\n"))

    assert plan.case_ids == ["alpha"]


def test_load_plan_requires_budgets(tmp_path: Path) -> None:
    text = "name: v2-bench\nrepeats: 4\n"

    with pytest.raises(ValidationError, match="budgets"):
        load_plan(_plan_file(tmp_path, text))


def test_load_plan_rejects_repeats_below_floor(tmp_path: Path) -> None:
    text = VALID_YAML.replace("repeats: 4", "repeats: 3")

    with pytest.raises(ValidationError, match="repeats"):
        load_plan(_plan_file(tmp_path, text))


def test_load_plan_rejects_non_positive_budgets(tmp_path: Path) -> None:
    text = VALID_YAML.replace("max_per_run: 24", "max_per_run: 0")

    with pytest.raises(ValidationError, match="max_per_run"):
        load_plan(_plan_file(tmp_path, text))


def test_load_plan_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    text = VALID_YAML + "prompt_override: sneaky\n"

    with pytest.raises(ValidationError, match="prompt_override"):
        load_plan(_plan_file(tmp_path, text))


def test_load_plan_rejects_unknown_budget_fields(tmp_path: Path) -> None:
    text = VALID_YAML.replace(
        "  max_per_run: 24", "  max_per_run: 24\n  max_per_epoch: 5"
    )

    with pytest.raises(ValidationError, match="max_per_epoch"):
        load_plan(_plan_file(tmp_path, text))


def test_load_plan_rejects_non_mapping_documents(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_plan(_plan_file(tmp_path, "- just\n- a\n- list\n"))


def test_load_plan_rejects_malformed_yaml(tmp_path: Path) -> None:
    with pytest.raises(yaml.YAMLError):
        load_plan(_plan_file(tmp_path, "budgets: [oops"))


def test_loaded_plan_enforces_budgets_inside_harness_run(tmp_path: Path) -> None:
    """The full path: plan file -> load_plan -> harness refusal before spend."""
    text = VALID_YAML.replace("max_per_run: 24", "max_per_run: 10")
    runner = CountingRunner()

    plan = load_plan(_plan_file(tmp_path, text))
    with pytest.raises(BudgetExceededError):
        BenchHarness(StubRetriever(), runner).run(corpus=_budget_cases(), plan=plan)  # type: ignore[arg-type]

    assert runner.calls == []


def test_loaded_plan_case_subset_runs_within_its_budgets(tmp_path: Path) -> None:
    text = VALID_YAML + "case_ids:\n  - alpha\n"
    runner = CountingRunner()

    plan = load_plan(_plan_file(tmp_path, text))
    report = BenchHarness(StubRetriever(), runner).run(  # type: ignore[arg-type]
        corpus=_budget_cases(), plan=plan
    )

    assert [case.case_id for case in report.cases] == ["alpha"]
    assert report.summary.replays == 8
    assert len(runner.calls) == 8


def test_plan_conflicting_with_explicit_arguments_is_refused(tmp_path: Path) -> None:
    plan = load_plan(
        _plan_file(tmp_path, VALID_YAML.replace("repeats: 4", "repeats: 5"))
    )

    with pytest.raises(ValueError, match="repeats conflict"):
        BenchHarness(StubRetriever(), CountingRunner()).run(  # type: ignore[arg-type]
            corpus=_budget_cases(), plan=plan, repeats=4
        )

    plan_with_ids = load_plan(
        _plan_file(tmp_path, VALID_YAML + "case_ids:\n  - alpha\n")
    )
    with pytest.raises(ValueError, match="case_ids conflict"):
        BenchHarness(StubRetriever(), CountingRunner()).run(  # type: ignore[arg-type]
            ["beta"], corpus=_budget_cases(), plan=plan_with_ids
        )


def test_plan_and_explicit_arguments_in_agreement_run(tmp_path: Path) -> None:
    # At 5 repeats the full-corpus schedule spends 30 replays (10 per
    # proposal, 20 in fam-a), so all three budgets must be sized for it.
    plan = load_plan(
        _plan_file(
            tmp_path,
            VALID_YAML.replace("repeats: 4", "repeats: 5")
            .replace("max_per_proposal: 8", "max_per_proposal: 15")
            .replace("max_per_case_family: 16", "max_per_case_family: 25")
            .replace("max_per_run: 24", "max_per_run: 40"),
        )
    )

    report = BenchHarness(StubRetriever(), CountingRunner()).run(  # type: ignore[arg-type]
        corpus=_budget_cases(), plan=plan, repeats=5
    )

    assert report.repeats == 5
    assert report.summary.replays == 30
