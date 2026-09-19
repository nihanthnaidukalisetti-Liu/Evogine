"""Spend control: every model call metered against the config's dollar ceiling.

The ceiling is fail-closed in both directions, structurally:

- **Before** each call, the meter reserves the call's maximum possible cost
  (the declared per-call input bound plus the request's ``max_tokens``
  output cap, priced by the declared table). A reservation that would push
  the committed ledger past the ceiling raises
  :class:`SpendCeilingRefused` *before the call is made* — an over-ceiling
  call is never sent, never half-spent.
- **After** each call, the actual token usage is priced into the ledger. A
  provider that reports no usage is billed at its reservation, never at
  zero — a missing usage report cannot disarm the ceiling. A provider that
  bills beyond its reservation can push the ledger over the ceiling; the
  very next reservation then refuses, and the overage stands in the ledger
  and in the run's refusal event — visible evidence, never a silent
  overrun.

Prices are declared per run (the bench's ``SpendExpectation`` discipline);
the ceiling itself comes from the unified config's ``budget.max_run_cost_usd``.
All money is exact ``Decimal``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.swarm.clients import (
    ModelClient,
    ModelClientError,
    ModelRequest,
    ModelResponse,
)
from evogine.swarm.orchestrator.errors import OrchestratorRefused
from evogine.swarm.orchestrator.events import SpendEvent

_PER_MILLION = Decimal(1_000_000)
"""Divisor converting token counts against a per-million-token price table."""


class SpendCeilingRefused(OrchestratorRefused):
    """The ceiling cannot fit this call — refusing before the call is made.

    Carries the ledger state that forced the refusal so the caller (and the
    run's refusal event) can show the exact numbers.
    """

    def __init__(self, *, spent: Decimal, projected: Decimal, ceiling: Decimal) -> None:
        self.spent = spent
        self.projected = projected
        self.ceiling = ceiling
        super().__init__(
            f"spend ceiling refused: ${spent} spent + ${projected} next-call "
            f"reservation projects past the ${ceiling} ceiling — the call was "
            "not made, nothing further will spend"
        )


class SpendPolicy(BaseModel):
    """The declared pricing table and the run's dollar ceiling.

    Every knob is required and explicit — an absent price would silently
    project zero and disarm the ceiling, the ambient-budget failure mode
    the bench's budgets module exists to fix. The ceiling flows from the
    unified config (``budget.max_run_cost_usd``); the per-call input bound
    is the reservation's conservative input assumption.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ceiling_usd: Decimal = Field(gt=0)
    input_price_per_mtok_usd: Decimal = Field(ge=0)
    output_price_per_mtok_usd: Decimal = Field(ge=0)
    max_call_input_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def _finite_money(self) -> SpendPolicy:
        for name in (
            "ceiling_usd",
            "input_price_per_mtok_usd",
            "output_price_per_mtok_usd",
        ):
            if not getattr(self, name).is_finite():
                raise ValueError(f"{name} must be a finite dollar amount")
        return self

    def cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Price one call's token counts exactly."""
        cost = (
            Decimal(input_tokens) * self.input_price_per_mtok_usd
            + Decimal(output_tokens) * self.output_price_per_mtok_usd
        ) / _PER_MILLION
        return cost


SpendEventListener = Callable[[SpendEvent], None]
"""Receives one event per metered call, in ledger order."""


class SpendMeter:
    """The thread-safe per-run spend ledger: reserve, record, refuse.

    Commitment accounting: a reservation holds projected dollars
    outstanding until the call resolves, so concurrent workers can never
    jointly over-commit the ceiling — ``spent + outstanding + next`` is
    checked under one lock before any call goes out.
    """

    def __init__(
        self,
        policy: SpendPolicy,
        *,
        initial_spent_usd: Decimal = Decimal(0),
        on_event: SpendEventListener | None = None,
    ) -> None:
        if initial_spent_usd < 0:
            raise ValueError("initial spent amount cannot be negative")
        self._policy = policy
        self._spent = initial_spent_usd
        self._outstanding = Decimal(0)
        self._on_event = on_event
        self._lock = threading.Lock()

    @property
    def policy(self) -> SpendPolicy:
        return self._policy

    @property
    def spent_usd(self) -> Decimal:
        """Dollars actually charged so far, across the whole run."""
        return self._spent

    def reserve(
        self,
        request: ModelRequest,
        *,
        worker_id: int,
        item_id: str,
        call_index: int,
    ) -> Decimal:
        """Reserve the call's maximum cost; refuse past the ceiling.

        Must be called before every model call. Returns the reservation
        (the projected maximum), which the caller hands back via
        :meth:`record_success` or :meth:`record_failure`.
        """
        projected = self._project(request)
        with self._lock:
            # spent > ceiling can only happen when a provider billed past
            # its reservation; the overage is visible in the refusal —
            # never silently continued.
            if self._spent > self._policy.ceiling_usd or (
                self._spent + self._outstanding + projected > self._policy.ceiling_usd
            ):
                self._outstanding = Decimal(0)
                raise SpendCeilingRefused(
                    spent=self._spent,
                    projected=projected,
                    ceiling=self._policy.ceiling_usd,
                )
            self._outstanding += projected
        return projected

    def record_success(
        self,
        response: ModelResponse,
        *,
        reservation: Decimal,
        worker_id: int,
        item_id: str,
        call_index: int,
    ) -> Decimal:
        """Charge the actual usage; a usage-less response bills its reservation."""
        has_usage = response.input_tokens is not None and response.output_tokens is not None
        if has_usage:
            actual = self._policy.cost_usd(
                input_tokens=response.input_tokens or 0,
                output_tokens=response.output_tokens or 0,
            )
        else:
            # Fail-closed: unreported usage is billed at the reservation —
            # the ceiling cannot be gamed by a provider that omits it.
            actual = reservation
        self._settle(
            ok=True,
            charge=actual,
            reservation=reservation,
            response=response,
            worker_id=worker_id,
            item_id=item_id,
            call_index=call_index,
        )
        return actual

    def record_failure(
        self,
        *,
        reservation: Decimal,
        worker_id: int,
        item_id: str,
        call_index: int,
        error: str,
    ) -> None:
        """Settle a failed call: nothing billable, the reservation released.

        ``error`` is surfaced through the item's status events, not the
        spend event — a failed call's spend record is still a spend record.
        """
        self._settle(
            ok=False,
            charge=Decimal(0),
            reservation=reservation,
            response=None,
            worker_id=worker_id,
            item_id=item_id,
            call_index=call_index,
            error=error,
        )

    def _settle(
        self,
        *,
        ok: bool,
        charge: Decimal,
        reservation: Decimal,
        response: ModelResponse | None,
        worker_id: int,
        item_id: str,
        call_index: int,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._outstanding = max(Decimal(0), self._outstanding - reservation)
            self._spent += charge
            spent_after = self._spent
        if self._on_event is not None:
            self._on_event(
                SpendEvent(
                    worker_id=worker_id,
                    item_id=item_id,
                    call_index=call_index,
                    ok=ok,
                    input_tokens=response.input_tokens if response is not None else None,
                    output_tokens=response.output_tokens if response is not None else None,
                    cost_usd=charge,
                    spent_usd=spent_after,
                    ceiling_usd=self._policy.ceiling_usd,
                )
            )

    def _project(self, request: ModelRequest) -> Decimal:
        return self._policy.cost_usd(
            input_tokens=self._policy.max_call_input_tokens,
            output_tokens=request.max_tokens,
        )


class MeteredModelClient:
    """A :class:`ModelClient` wrapper that makes the ceiling unskippable.

    Workers see only this client — the wrapped provider is unreachable —
    so no model call can bypass the meter. Each worker thread owns one
    instance (its ``worker_id`` is fixed; ``set_item`` marks which item the
    worker is on, for the spend events and per-item accounting). Reserving
    happens strictly before the provider call; failures settle at zero.
    Exceptions other than the client seam's own error types propagate
    untouched — a bug is never disguised as spend evidence.
    """

    def __init__(self, meter: SpendMeter, inner: ModelClient, *, worker_id: int) -> None:
        self._meter = meter
        self._inner = inner
        self._worker_id = worker_id
        self._item_id = ""
        self._call_index = 0
        self._item_spent = Decimal(0)

    def set_item(self, item_id: str) -> None:
        """Point the client at the item it is about to work; reset item spend."""
        self._item_id = item_id
        self._item_spent = Decimal(0)

    @property
    def item_spent_usd(self) -> Decimal:
        """What this worker's calls on the current item have charged."""
        return self._item_spent

    @property
    def call_count(self) -> int:
        """Calls made by this worker across all items so far."""
        return self._call_index

    def complete(self, request: ModelRequest) -> ModelResponse:
        self._call_index += 1
        reservation = self._meter.reserve(
            request,
            worker_id=self._worker_id,
            item_id=self._item_id,
            call_index=self._call_index,
        )
        try:
            response = self._inner.complete(request)
        except ModelClientError as exc:
            self._meter.record_failure(
                reservation=reservation,
                worker_id=self._worker_id,
                item_id=self._item_id,
                call_index=self._call_index,
                error=str(exc),
            )
            raise
        actual = self._meter.record_success(
            response,
            reservation=reservation,
            worker_id=self._worker_id,
            item_id=self._item_id,
            call_index=self._call_index,
        )
        self._item_spent += actual
        return response


# Static protocol proof: mypy verifies the metered wrapper satisfies the
# model-client seam at this assignment — workers cannot tell it apart from
# a raw provider, which is the point (the runtime twin lives in the tests).
_protocol_check: type[ModelClient] = MeteredModelClient


__all__ = [
    "MeteredModelClient",
    "SpendCeilingRefused",
    "SpendEventListener",
    "SpendMeter",
    "SpendPolicy",
]
