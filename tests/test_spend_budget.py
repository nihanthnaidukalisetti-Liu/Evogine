"""Dollar-ceiling tests: fail-closed config, queue-time projection refusal, reconciliation.

The USD ceiling is the spend-side twin of the replay-count budgets: a bad
configuration is a construction error (never a run), an over-ceiling
projection is refused before the first replay is spent, and post-run
reconciliation prices what the run actually recorded — never hiding calls
that carried no usage.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from evogine.bench.budgets import (
    DollarBudget,
    ReplayBudgets,
    SpendCeilingExceededError,
    SpendExpectation,
    projected_run_cost_usd,
    queue_replays,
    queue_spend,
    reconcile_spend,
)
from evogine.bench.cases import BenchCase
from evogine.bench.replay import ReplaySchedule, build_replay_requests


class StubCall:
    """Minimal structural TokenRecord: one call's recorded token usage."""

    def __init__(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _expectation(**overrides: object) -> SpendExpectation:
    """$3.00/MTok in, $15.00/MTok out, 1M in + 100k out per replay → $4.50."""
    fields: dict[str, object] = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "input_price_per_mtok_usd": Decimal("3.00"),
        "output_price_per_mtok_usd": Decimal("15.00"),
    }
    fields.update(overrides)
    return SpendExpectation(**fields)  # type: ignore[arg-type]


def _budget(**overrides: object) -> DollarBudget:
    fields: dict[str, object] = {"expectation": _expectation()}
    fields.update(overrides)
    return DollarBudget(**fields)  # type: ignore[arg-type]


def _schedule(total: int) -> ReplaySchedule:
    """A schedule of exactly ``total`` replays (one stub case, two arms)."""
    stub = BenchCase(
        id="alpha", family="fam-a", task_context="ctx", hidden_invariant=lambda o: True
    )
    return ReplaySchedule(requests=build_replay_requests([stub], total // 2))


# ---------------------------------------------------------------------------
# Fail-closed construction: bad money config is a validation error.
# ---------------------------------------------------------------------------


def test_default_ceiling_is_25_dollars_per_run() -> None:
    assert _budget().max_run_cost_usd == Decimal("25.00")


def test_expectation_is_required() -> None:
    with pytest.raises(ValidationError):
        DollarBudget()  # type: ignore[call-arg]


@pytest.mark.parametrize("ceiling", [Decimal(0), Decimal("-1.00")])
def test_non_positive_ceiling_is_refused(ceiling: Decimal) -> None:
    with pytest.raises(ValidationError):
        _budget(max_run_cost_usd=ceiling)


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity")])
def test_non_finite_ceiling_is_refused(value: Decimal) -> None:
    with pytest.raises(ValidationError, match="finite"):
        _budget(max_run_cost_usd=value)


def test_non_finite_price_is_refused() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _expectation(input_price_per_mtok_usd=Decimal("NaN"))


def test_negative_expectation_values_are_refused() -> None:
    with pytest.raises(ValidationError):
        _expectation(input_tokens=-1)
    with pytest.raises(ValidationError):
        _expectation(output_price_per_mtok_usd=Decimal("-0.01"))


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        _expectation(currency="EUR")
    with pytest.raises(ValidationError):
        _budget(soft_limit=True)


# ---------------------------------------------------------------------------
# Projection: linear in the replay count, from the configured expectation.
# ---------------------------------------------------------------------------


def test_replay_cost_is_expected_tokens_priced_by_the_table() -> None:
    assert _expectation().replay_cost_usd() == Decimal("4.50")


def test_projection_scales_with_the_replay_count() -> None:
    assert projected_run_cost_usd(_budget(), 10) == Decimal("45.0000")
    assert projected_run_cost_usd(_budget(), 0) == 0


def test_negative_replay_count_is_caller_error() -> None:
    with pytest.raises(ValueError, match="at least 0"):
        projected_run_cost_usd(_budget(), -1)


# ---------------------------------------------------------------------------
# Queue-time refusal: an over-ceiling projection spends nothing.
# ---------------------------------------------------------------------------


def test_projection_within_the_ceiling_returns_it() -> None:
    """10 × $2.00 = $20.00 fits the default $25.00 ceiling."""
    cheap = _budget(
        expectation=_expectation(
            input_tokens=1_000_000,
            output_tokens=0,
            input_price_per_mtok_usd=Decimal("2.00"),
        )
    )

    assert queue_spend(_schedule(10), cheap) == Decimal("20.00")


def test_projection_exactly_at_the_ceiling_is_allowed() -> None:
    """$2.50/replay × 10 replays = exactly the $25.00 ceiling — the boundary holds."""
    budget = _budget(
        expectation=_expectation(
            input_tokens=1_000_000,
            output_tokens=0,
            input_price_per_mtok_usd=Decimal("2.50"),
        )
    )

    assert queue_spend(_schedule(10), budget) == Decimal("25.00")


def test_projection_over_the_ceiling_is_refused_before_spend() -> None:
    with pytest.raises(SpendCeilingExceededError) as excinfo:
        queue_spend(_schedule(12), _budget())  # 12 × $4.50 = $54.00 > $25.00

    assert excinfo.value.projected == Decimal("54.00")
    assert excinfo.value.allowed == Decimal("25.00")
    assert excinfo.value.replays == 12
    assert "before spend" in str(excinfo.value)


def test_spend_ceiling_composes_with_the_count_budgets() -> None:
    """The full queue path: count budgets pass, the dollar ceiling still refuses."""
    cases = [
        BenchCase(
            id=case_id,
            family="fam-a",
            task_context=f"ctx-{case_id}",
            hidden_invariant=lambda o: True,
        )
        for case_id in ("alpha", "beta", "gamma")
    ]
    requests = build_replay_requests(cases, 4)  # 24 replays

    schedule = queue_replays(
        requests,
        ReplayBudgets(max_per_proposal=99, max_per_case_family=99, max_per_run=99),
    )

    with pytest.raises(SpendCeilingExceededError):
        queue_spend(schedule, _budget())  # 24 × $4.50 = $108.00 > $25.00


# ---------------------------------------------------------------------------
# Reconciliation: post-run evidence from persisted token records.
# ---------------------------------------------------------------------------


def test_reconciliation_prices_recorded_tokens() -> None:
    records = [
        StubCall(2_000_000, 200_000),  # $6.00 + $3.00
        StubCall(1_000_000, 0),  # $3.00
    ]

    report = reconcile_spend(_budget(), records)

    assert report.recorded_calls == 2
    assert report.calls_with_usage == 2
    assert report.calls_without_usage == 0
    assert report.actual_cost_usd == Decimal("12.00")
    assert report.within_ceiling is True
    assert report.overage_usd == 0


def test_reconciliation_never_hides_calls_without_usage() -> None:
    records = [
        StubCall(None, None),  # a failed call: recorded, unbilled, counted
        StubCall(1_000_000, 100_000),  # $4.50
    ]

    report = reconcile_spend(_budget(), records)

    assert report.recorded_calls == 2
    assert report.calls_with_usage == 1
    assert report.calls_without_usage == 1
    assert report.actual_cost_usd == Decimal("4.50")


def test_reconciliation_partial_usage_is_unbilled() -> None:
    """Half-reported usage cannot be priced honestly — it counts, not bills."""
    report = reconcile_spend(_budget(), [StubCall(1_000_000, None)])

    assert report.calls_without_usage == 1
    assert report.actual_cost_usd == 0


def test_reconciliation_reports_the_overage_over_the_ceiling() -> None:
    records = [StubCall(1_000_000, 100_000)] * 10  # 10 × $4.50 = $45.00

    report = reconcile_spend(_budget(), records)

    assert report.within_ceiling is False
    assert report.overage_usd == Decimal("20.00")


def test_reconciliation_of_an_empty_run_is_zero() -> None:
    report = reconcile_spend(_budget(), [])

    assert report.recorded_calls == 0
    assert report.actual_cost_usd == 0
    assert report.within_ceiling is True
