"""Model client seam: the swarm's single doorway to language models.

Roles depend only on the :class:`ModelClient` protocol — never on a concrete
SDK — so providers stay swappable and P2 can layer multi-provider routing on
top without touching role code. Resilience (timeouts, bounded retries,
jittered exponential backoff, Retry-After) is owned in this module rather
than in role code or provider SDKs: providers make exactly one attempt.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    """One single-turn completion request sent to a model."""

    system: str
    user: str
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


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
