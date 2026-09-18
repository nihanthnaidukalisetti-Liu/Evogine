"""Forced-failover units: second-provider routing behind the client seam.

Stub providers (never live models) verify the routing contract: first
success wins, every seam error fails over, the last error propagates when
the chain is exhausted, and retry policies compose per provider.
"""

from __future__ import annotations

import pytest

from evogine.swarm.clients import (
    FailoverModelClient,
    ModelClientError,
    ModelPermanentError,
    ModelRequest,
    ModelResponse,
    ModelTransientError,
    RetryingModelClient,
)

REQUEST = ModelRequest(system="s", user="u")


class StubProvider:
    """ModelClient double: canned outcome, per-call counter, fixed latency."""

    def __init__(
        self,
        name: str,
        *,
        response: ModelResponse | None = None,
        error: ModelClientError | None = None,
    ) -> None:
        self.name = name
        self._response = response or ModelResponse(text=f"from-{name}")
        self._error = error
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._response


def test_first_healthy_provider_serves() -> None:
    primary = StubProvider("primary")
    secondary = StubProvider("secondary")
    client = FailoverModelClient([primary, secondary])
    assert client.complete(REQUEST).text == "from-primary"
    assert primary.calls == 1 and secondary.calls == 0


def test_transient_primary_failure_forces_failover() -> None:
    primary = StubProvider("primary", error=ModelTransientError("model server error: 503"))
    secondary = StubProvider("secondary")
    client = FailoverModelClient([primary, secondary])
    assert client.complete(REQUEST).text == "from-secondary"
    assert primary.calls == 1 and secondary.calls == 1


def test_permanent_primary_failure_also_fails_over() -> None:
    """A 4xx is one provider's judgment — the router tries the next provider."""
    primary = StubProvider("primary", error=ModelPermanentError("model rejected"))
    secondary = StubProvider("secondary")
    client = FailoverModelClient([primary, secondary])
    assert client.complete(REQUEST).text == "from-secondary"


def test_exhausted_chain_raises_the_last_provider_error() -> None:
    primary = StubProvider("primary", error=ModelTransientError("primary down"))
    secondary = StubProvider("secondary", error=ModelPermanentError("secondary down"))
    client = FailoverModelClient([primary, secondary])
    with pytest.raises(ModelPermanentError, match="secondary down"):
        client.complete(REQUEST)
    assert primary.calls == 1 and secondary.calls == 1


def test_empty_chain_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        FailoverModelClient([])


def test_retry_policy_composes_per_provider() -> None:
    """Each child exhausts its own retries before the router moves on."""

    class FlakyThenFine:
        """Transient error twice, then success — per construction."""

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls < 3:
                raise ModelTransientError("flaky")
            return ModelResponse(text="recovered")

    def never_sleep(_seconds: float) -> None:
        return None

    primary = FlakyThenFine()
    secondary = StubProvider("secondary")
    client = FailoverModelClient(
        [
            RetryingModelClient(primary, sleep=never_sleep, jitter=_NoJitter()),
            RetryingModelClient(secondary, sleep=never_sleep, jitter=_NoJitter()),
        ]
    )
    assert client.complete(REQUEST).text == "recovered"
    assert primary.calls == 3 and secondary.calls == 0


class _NoJitter:
    """Deterministic jitter: always take the full computed backoff."""

    def uniform(self, a: float, b: float) -> float:
        return b


def test_router_makes_no_extra_calls_on_success() -> None:
    """A mid-chain provider serving stops the walk — later ones stay cold."""
    first = StubProvider("first", error=ModelTransientError("down"))
    second = StubProvider("second")
    third = StubProvider("third")
    client = FailoverModelClient([first, second, third])
    assert client.complete(REQUEST).text == "from-second"
    assert third.calls == 0
