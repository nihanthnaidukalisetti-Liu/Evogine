"""V2-4: replay budgets as first-class plan config, enforced before spend.

Plans declare replay budgets — max replays per proposal, per case family,
and per run — and the queue refuses a schedule that would exceed any of
them *before* the first replay is spent. A run never discovers an exceeded
budget on the bill after the fact: queue-time refusal is all-or-nothing,
and nothing is reconciled afterwards.

The V3 live bench (W0) adds the spend-side twin: a USD dollar ceiling
(:class:`DollarBudget`, default $25.00/run) enforced the same way — the
declared :class:`SpendExpectation` projects the run's cost at queue time
(:func:`queue_spend`) and refuses an over-ceiling projection before the
first replay is spent; after the run, :func:`reconcile_spend` prices the
persisted per-call token records against the same table as evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.bench.replay import REPEAT_FLOOR, ReplayRequest, ReplaySchedule

_PER_MILLION = Decimal(1_000_000)
"""Divisor converting token counts against a per-million-token price table."""


class ReplayBudgets(BaseModel):
    """Declared replay budget caps.

    All three knobs are required: an optional knob would reintroduce the
    ambient-budget failure mode V2-4 exists to fix.
    """

    model_config = ConfigDict(extra="forbid")

    max_per_proposal: int = Field(ge=1)
    max_per_case_family: int = Field(ge=1)
    max_per_run: int = Field(ge=1)


class BenchPlan(BaseModel):
    """First-class bench plan config: repeats, budgets, optional case subset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    repeats: int = Field(ge=REPEAT_FLOOR)
    budgets: ReplayBudgets
    case_ids: list[str] | None = None
    """Omitted means the full corpus; a subset must name existing case ids."""


class BudgetScope(str, Enum):
    """Which declared budget a schedule exceeded."""

    RUN = "run"
    CASE_FAMILY = "case_family"
    PROPOSAL = "proposal"


class BudgetExceededError(Exception):
    """Queue-time refusal: the schedule exceeds a declared budget before spend."""

    def __init__(self, scope: BudgetScope, key: str, required: int, allowed: int) -> None:
        self.scope = scope
        self.key = key
        self.required = required
        self.allowed = allowed
        super().__init__(
            f"replay budget exceeded at queue time (scope={scope.value}, key={key!r}): "
            f"schedule requires {required} replays, budget allows {allowed} — "
            "refusing before spend"
        )


def queue_replays(
    requests: list[ReplayRequest], budgets: ReplayBudgets
) -> ReplaySchedule:
    """Enforce the declared budgets on the full schedule BEFORE any replay is spent.

    Check order is per-run, then per-family, then per-proposal. Refusal is
    all-or-nothing: an exceeding schedule spends nothing and is never
    reconciled after the fact.
    """
    schedule = ReplaySchedule(requests=requests)
    total = schedule.total()
    if total > budgets.max_per_run:
        raise BudgetExceededError(BudgetScope.RUN, "*", total, budgets.max_per_run)
    for family, count in sorted(schedule.by_family().items()):
        if count > budgets.max_per_case_family:
            raise BudgetExceededError(
                BudgetScope.CASE_FAMILY, family, count, budgets.max_per_case_family
            )
    for proposal, count in sorted(schedule.by_proposal().items()):
        if count > budgets.max_per_proposal:
            raise BudgetExceededError(
                BudgetScope.PROPOSAL, proposal, count, budgets.max_per_proposal
            )
    return schedule


def load_plan(path: str | Path) -> BenchPlan:
    """Parse and validate a bench plan from a YAML file.

    Unknown fields are refused (config drift), the repeat count must meet
    the 4-repeat floor, and budgets must be present and positive.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BenchPlan.model_validate(data)


# ---------------------------------------------------------------------------
# V3 W0: the USD dollar ceiling — the spend-side twin of the replay counts.
# ---------------------------------------------------------------------------


class TokenRecord(Protocol):
    """What spend reconciliation reads: one call's recorded token usage.

    Structural by design — any outcome record carrying token counts
    satisfies it (the live runner's per-call records do).
    """

    input_tokens: int | None
    output_tokens: int | None


class SpendExpectation(BaseModel):
    """Configured per-replay cost expectation: the queue-time projection inputs.

    Expected token counts priced by a configured per-million-token table.
    All four knobs are required — an absent knob would silently project zero
    and disarm the ceiling, the same ambient-budget failure mode V2-4 exists
    to fix. Validation fails closed: non-finite money is a construction
    error, never a run.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    input_price_per_mtok_usd: Decimal = Field(ge=0)
    output_price_per_mtok_usd: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def _finite_prices(self) -> SpendExpectation:
        if not self.input_price_per_mtok_usd.is_finite():
            raise ValueError("input_price_per_mtok_usd must be a finite dollar amount")
        if not self.output_price_per_mtok_usd.is_finite():
            raise ValueError("output_price_per_mtok_usd must be a finite dollar amount")
        return self

    def replay_cost_usd(self) -> Decimal:
        """Return the expected USD cost of one replay."""
        input_cost = Decimal(self.input_tokens) * self.input_price_per_mtok_usd
        output_cost = Decimal(self.output_tokens) * self.output_price_per_mtok_usd
        return (input_cost + output_cost) / _PER_MILLION


class DollarBudget(BaseModel):
    """The run's USD spend ceiling beside :class:`ReplayBudgets`' count caps.

    ``max_run_cost_usd`` defaults to $25.00 per run. The declared expectation
    projects the run's spend at queue time, before the first replay is
    spent; after the run, reconciliation prices the persisted token records
    against the same table. Validation fails closed: a non-finite or
    non-positive ceiling is a construction error, never a run.
    """

    model_config = ConfigDict(extra="forbid")

    expectation: SpendExpectation
    max_run_cost_usd: Decimal = Field(default=Decimal("25.00"), gt=0)

    @model_validator(mode="after")
    def _finite_ceiling(self) -> DollarBudget:
        if not self.max_run_cost_usd.is_finite():
            raise ValueError("max_run_cost_usd must be a finite dollar amount")
        return self


class SpendCeilingExceededError(Exception):
    """Queue-time refusal: the projected spend exceeds the declared ceiling."""

    def __init__(self, projected: Decimal, allowed: Decimal, replays: int) -> None:
        self.projected = projected
        self.allowed = allowed
        self.replays = replays
        super().__init__(
            f"dollar ceiling exceeded at queue time: {replays} replays project "
            f"${projected} against a ${allowed} ceiling — refusing before spend"
        )


def projected_run_cost_usd(budget: DollarBudget, replays: int) -> Decimal:
    """Project the run's USD spend from the declared expectation.

    Linear in the replay count; zero replays project zero. A negative count
    is caller error and raises.
    """
    if replays < 0:
        raise ValueError(f"replays must be at least 0, got {replays}")
    return budget.expectation.replay_cost_usd() * replays


def queue_spend(schedule: ReplaySchedule, budget: DollarBudget) -> Decimal:
    """Enforce the dollar ceiling on the queued schedule BEFORE any spend.

    The spend-side twin of :func:`queue_replays`: the count budgets cap how
    many replays a schedule may spend, this caps what they may cost.
    Refusal is all-or-nothing — an exceeding projection spends nothing and
    is never reconciled after the fact. Returns the projection when it fits.
    """
    projected = projected_run_cost_usd(budget, schedule.total())
    if projected > budget.max_run_cost_usd:
        raise SpendCeilingExceededError(
            projected, budget.max_run_cost_usd, schedule.total()
        )
    return projected


class SpendReconciliation(BaseModel):
    """Post-run spend evidence: persisted token records priced and totaled."""

    model_config = ConfigDict(extra="forbid")

    recorded_calls: int
    calls_with_usage: int
    """Calls whose persisted record carries both token counts."""
    calls_without_usage: int
    """Failed calls (or providers that reported no usage): recorded evidence,
    unbilled here, never hidden."""
    actual_cost_usd: Decimal
    ceiling_usd: Decimal
    within_ceiling: bool
    overage_usd: Decimal
    """``max(0, actual − ceiling)``; zero when the run finished within it."""


def reconcile_spend(
    budget: DollarBudget, records: Sequence[TokenRecord]
) -> SpendReconciliation:
    """Reconcile post-run: price persisted token records against the declared table.

    Reads only what the run actually recorded. A call without token counts —
    a failure, or a provider that reported nothing — contributes zero
    dollars but is counted as ``calls_without_usage`` so the report never
    silently under-counts spend. This is evidence, not enforcement: the
    ceiling was enforced at queue time; the actuals here are what they are.
    """
    actual = Decimal(0)
    with_usage = 0
    without_usage = 0
    for record in records:
        if record.input_tokens is None or record.output_tokens is None:
            without_usage += 1
            continue
        with_usage += 1
        actual += (
            Decimal(record.input_tokens) * budget.expectation.input_price_per_mtok_usd
            + Decimal(record.output_tokens)
            * budget.expectation.output_price_per_mtok_usd
        ) / _PER_MILLION
    overage = actual - budget.max_run_cost_usd
    return SpendReconciliation(
        recorded_calls=len(records),
        calls_with_usage=with_usage,
        calls_without_usage=without_usage,
        actual_cost_usd=actual,
        ceiling_usd=budget.max_run_cost_usd,
        within_ceiling=overage <= 0,
        overage_usd=max(Decimal(0), overage),
    )
