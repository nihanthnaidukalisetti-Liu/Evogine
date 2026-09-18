"""Model client seam: the swarm's single doorway to language models.

Roles depend only on the :class:`ModelClient` protocol — never on a concrete
SDK — so providers stay swappable and multi-provider routing layers on top
without touching role code. Resilience (timeouts, bounded retries, jittered
exponential backoff, Retry-After) is owned in this module rather than in
role code or provider SDKs: providers make exactly one attempt, and
:class:`FailoverModelClient` owns provider routing.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Protocol

import anthropic
from anthropic.types import TextBlock
from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    """One single-turn completion request sent to a model."""

    system: str
    user: str
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    """What a model returned, normalized across providers."""

    text: str
    stop_reason: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelClientError(Exception):
    """Base class for every failure raised through the model client seam."""


class ModelTransientError(ModelClientError):
    """A retryable model failure: timeouts, connection errors, 429, 5xx."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelPermanentError(ModelClientError):
    """A non-retryable model failure: the request itself will not succeed."""


class ModelClient(Protocol):
    """The seam roles code against: one single-turn model completion.

    Implementations make a single attempt — no internal retries — so retry
    policy stays uniform and testable in exactly one place. Failures surface
    as :class:`ModelTransientError` (worth retrying) or
    :class:`ModelPermanentError` (never retry); anything else means a broken
    provider implementation and propagates untouched.
    """

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the model's completion for ``request``."""
        ...


class Jitter(Protocol):
    """Source of backoff jitter; :class:`random.Random` satisfies it."""

    def uniform(self, a: float, b: float) -> float:
        """Return a random float in ``[a, b]``."""
        ...


class RetryPolicy(BaseModel):
    """Bounded, jittered exponential backoff for transient model failures."""

    max_attempts: int = Field(default=4, ge=1, description="1 means never retry")
    initial_backoff_s: float = Field(default=0.5, gt=0)
    max_backoff_s: float = Field(default=8.0, gt=0)
    multiplier: float = Field(default=2.0, ge=1.0)


class RetryingModelClient:
    """A :class:`ModelClient` wrapping one single-attempt provider.

    Retries only :class:`ModelTransientError`; permanent failures and
    anything unexpected surface immediately. Backoff grows exponentially
    from ``initial_backoff_s`` up to ``max_backoff_s`` and is fully jittered
    (uniform in ``[0, computed]``) — unjittered retries amplify rate-limit
    pressure into storms. A server ``Retry-After`` hint in whole seconds
    raises the pre-jitter floor, capped by ``max_backoff_s``.
    """

    def __init__(
        self,
        provider: ModelClient,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Jitter | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy or RetryPolicy()
        self._sleep = sleep
        self._jitter = jitter or random.Random()

    def complete(self, request: ModelRequest) -> ModelResponse:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._provider.complete(request)
            except ModelTransientError as exc:
                if attempts >= self._policy.max_attempts:
                    raise
                self._sleep(self._backoff_delay(attempts, exc.retry_after))

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        exponential = self._policy.initial_backoff_s * self._policy.multiplier ** (attempt - 1)
        delay = min(exponential, self._policy.max_backoff_s)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self._policy.max_backoff_s))
        return self._jitter.uniform(0.0, delay)


class FailoverModelClient:
    """Route one completion across providers in priority order.

    The P2 second-provider routing behind the client seam: roles keep coding
    against :class:`ModelClient`, never knowing which provider served them.
    The router tries each provider in order and the first success wins.

    Failure semantics — every :class:`ModelClientError` fails over, transient
    and permanent alike. A permanent error is one provider's judgment about
    its own account, quota, or request handling; the router is an
    availability layer and must not treat one provider's rejection as a
    global truth about the request. When every provider fails, the last
    error propagates with its origin intact.

    No retries live here — compose each child with
    :class:`RetryingModelClient` when it should exhaust its own bounded
    retries before the next provider is tried. A raw child gets exactly one
    attempt, matching the seam's single-attempt provider contract.
    """

    def __init__(self, providers: list[ModelClient]) -> None:
        if not providers:
            raise ValueError("failover chain must contain at least one provider")
        self._providers = providers

    def complete(self, request: ModelRequest) -> ModelResponse:
        errors: list[ModelClientError] = []
        for provider in self._providers:
            try:
                return provider.complete(request)
            except ModelClientError as exc:
                errors.append(exc)
        # Unreachable for a non-empty chain (validated in __init__), but the
        # last error — not an artifact — is what callers must see.
        raise errors[-1]


class AnthropicClient:
    """Concrete provider: the Anthropic-compatible Messages API.

    One concrete provider behind the P2 seam. SDK-level retries are disabled
    (``max_retries=0``) — :class:`RetryingModelClient` owns resilience — and
    SDK failures map onto the seam's error types so callers never see
    provider exceptions: timeouts, connection errors, 429s, and 5xx are
    transient; other 4xx (bad request, auth, not found, unprocessable) are
    permanent.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._model = model
        self._sdk = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            message = self._sdk.messages.create(
                model=self._model,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                max_tokens=request.max_tokens,
            )
        except anthropic.APITimeoutError as exc:  # before APIConnectionError: subclass
            raise ModelTransientError(f"model request timed out: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ModelTransientError(f"model connection failed: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ModelTransientError(
                "model rate limited", retry_after=_retry_after_seconds(exc)
            ) from exc
        except anthropic.InternalServerError as exc:
            raise ModelTransientError(f"model server error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ModelPermanentError(f"model rejected the request: {exc}") from exc
        return ModelResponse(
            text="".join(block.text for block in message.content if isinstance(block, TextBlock)),
            stop_reason=message.stop_reason,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


def _retry_after_seconds(exc: anthropic.RateLimitError) -> float | None:
    """Whole-second ``Retry-After`` header, if present and parseable.

    HTTP-date form is ignored: bounded jittered backoff covers it.
    """
    response = exc.response
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
