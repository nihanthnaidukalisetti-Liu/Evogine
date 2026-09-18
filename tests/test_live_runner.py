"""LiveCaseRunner tests: fail-closed mapping, per-call accounting, protocol compliance.

Fully offline by contract — the fake clients below ARE the model boundary:
scripted responses and errors, no network, no keys, no live model. The
pins: a client exception after its own retries/failover becomes a recorded
failing outcome and the run continues; every call persists model identity
and token counts; the runner satisfies the harness CaseRunner protocol.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from conftest import make_record

from evogine.bench.budgets import DollarBudget, SpendExpectation
from evogine.bench.cases import BenchCase
from evogine.bench.harness import CaseRunner
from evogine.bench.live import FAILURE_OUTPUT, LiveCallRecord, LiveCaseRunner
from evogine.bench.prompts import DEFAULT_PROMPT_PACK
from evogine.bench.rundirs import open_run
from evogine.swarm.clients import (
    ModelClientError,
    ModelPermanentError,
    ModelRequest,
    ModelResponse,
    ModelTransientError,
    RetryingModelClient,
    RetryPolicy,
)


class ScriptedClient:
    """Fake ModelClient: a script of responses and errors, calls recorded."""

    def __init__(self, script: list[ModelResponse | ModelClientError]) -> None:
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        step = self._script.pop(0)
        if isinstance(step, ModelClientError):
            raise step
        return step


class AlwaysFailingProvider:
    """Fake provider raising the same transient error on every attempt."""

    def __init__(self) -> None:
        self.attempts = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.attempts += 1
        raise ModelTransientError("provider down")


class _NoJitter:
    """Deterministic jitter: always take the full computed backoff."""

    def uniform(self, a: float, b: float) -> float:
        return b


def _response(
    text: str = "the fix",
    model: str | None = "fake-model",
    input_tokens: int | None = 100,
    output_tokens: int | None = 50,
) -> ModelResponse:
    return ModelResponse(
        text=text, model=model, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _case() -> BenchCase:
    return BenchCase(
        id="alpha",
        family="fam-a",
        task_context="fix the 429 storm",
        hidden_invariant=lambda o: True,
    )


def _runner(client: object, **kwargs: object) -> LiveCaseRunner:
    return LiveCaseRunner(
        client, DEFAULT_PROMPT_PACK, **kwargs  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Protocol compliance.
# ---------------------------------------------------------------------------


def test_live_runner_satisfies_the_case_runner_protocol() -> None:
    runner: CaseRunner = _runner(ScriptedClient([_response()]))

    assert isinstance(runner, CaseRunner)


# ---------------------------------------------------------------------------
# Fail-closed mapping: a terminal client failure is evidence, not a crash.
# ---------------------------------------------------------------------------


def test_client_failure_becomes_a_recorded_failing_outcome() -> None:
    client = ScriptedClient([ModelTransientError("provider down"), _response()])
    runner = _runner(client)

    first = runner.execute(_case(), arm="cold", knowledge=[], repeat=1)
    second = runner.execute(_case(), arm="cold", knowledge=[], repeat=2)  # run continues

    assert first == FAILURE_OUTPUT
    assert second == "the fix"
    assert [call.ok for call in runner.calls] == [False, True]
    failed = runner.calls[0]
    assert failed.error == "provider down"
    assert failed.model is None
    assert failed.input_tokens is None
    assert failed.output_tokens is None


def test_permanent_client_failure_is_also_fail_closed() -> None:
    runner = _runner(ScriptedClient([ModelPermanentError("quota exceeded")]))

    output = runner.execute(_case(), arm="warm", knowledge=[], repeat=1)

    assert output == FAILURE_OUTPUT
    assert runner.calls[0].ok is False
    assert runner.calls[0].error == "quota exceeded"


def test_failure_is_recorded_after_the_client_retries_are_exhausted() -> None:
    """The provider fails 3 times (the client's bounded retries), THEN the
    runner records — the fail-closed mapping owns post-retry failure only."""
    provider = AlwaysFailingProvider()
    retrying = RetryingModelClient(
        provider,
        RetryPolicy(max_attempts=3, initial_backoff_s=0.001),
        sleep=lambda _seconds: None,
        jitter=_NoJitter(),
    )
    runner = LiveCaseRunner(retrying, DEFAULT_PROMPT_PACK)

    output = runner.execute(_case(), arm="warm", knowledge=[], repeat=1)

    assert output == FAILURE_OUTPUT
    assert provider.attempts == 3
    assert runner.calls[0].ok is False


def test_unexpected_exceptions_propagate_untouched() -> None:
    """A broken provider implementation is a bug, never benchmark evidence."""

    class BrokenProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError("bug in the provider implementation")

    runner = _runner(BrokenProvider())

    with pytest.raises(RuntimeError):
        runner.execute(_case(), arm="cold", knowledge=[], repeat=1)
    assert runner.calls == []


def test_failure_output_cannot_be_mistaken_for_a_real_answer() -> None:
    assert FAILURE_OUTPUT.strip() != ""
    assert "failed" in FAILURE_OUTPUT


# ---------------------------------------------------------------------------
# Per-call accounting: model identity and token counts on every record.
# ---------------------------------------------------------------------------


def test_every_call_persists_model_identity_and_tokens() -> None:
    client = ScriptedClient(
        [
            _response(model="model-a", input_tokens=1000, output_tokens=200),
            _response(model="model-b", input_tokens=3000, output_tokens=400),
        ]
    )
    runner = _runner(client)

    runner.execute(_case(), arm="cold", knowledge=[], repeat=1)
    runner.execute(_case(), arm="warm", knowledge=[], repeat=2)

    first, second = runner.calls
    assert (first.model, first.input_tokens, first.output_tokens) == (
        "model-a",
        1000,
        200,
    )
    assert (second.model, second.input_tokens, second.output_tokens) == (
        "model-b",
        3000,
        400,
    )
    assert first.call_index == 1
    assert second.call_index == 2
    assert (first.case_id, str(first.arm), first.repeat) == ("alpha", "cold", 1)
    assert (second.case_id, str(second.arm), second.repeat) == ("alpha", "warm", 2)


def test_each_call_is_bound_to_the_prompt_pack() -> None:
    client = ScriptedClient([_response()])
    runner = _runner(client)
    case = _case()

    runner.execute(case, arm="cold", knowledge=[], repeat=1)

    assert client.requests[0].system == DEFAULT_PROMPT_PACK.system
    assert client.requests[0].user == DEFAULT_PROMPT_PACK.render_user(
        task_context=case.task_context, knowledge=[]
    )
    assert runner.calls[0].prompt_pack_fingerprint == DEFAULT_PROMPT_PACK.fingerprint


def test_knowledge_reaches_the_model_prompt() -> None:
    client = ScriptedClient([_response()])
    runner = _runner(client)
    records = [
        make_record("rec-1"),
        make_record("rec-2", topic="timezone drift"),
    ]

    runner.execute(_case(), arm="warm", knowledge=records, repeat=3)

    user = client.requests[0].user
    assert "jittered exponential backoff" in user  # rec-1's fix strategy
    assert "timezone drift" in user  # rec-2's topic-shaped body


# ---------------------------------------------------------------------------
# Spend reconciliation over the recorded calls (deliverable 2 tie-in).
# ---------------------------------------------------------------------------


def test_reconcile_prices_the_recorded_calls() -> None:
    client = ScriptedClient(
        [_response(input_tokens=1_000_000, output_tokens=0), ModelTransientError("down")]
    )
    runner = _runner(client)
    budget = DollarBudget(
        expectation=SpendExpectation(
            input_tokens=0,
            output_tokens=0,
            input_price_per_mtok_usd=Decimal("3.00"),
            output_price_per_mtok_usd=Decimal("15.00"),
        )
    )

    runner.execute(_case(), arm="cold", knowledge=[], repeat=1)
    runner.execute(_case(), arm="cold", knowledge=[], repeat=2)

    report = runner.reconcile(budget)
    assert report.recorded_calls == 2
    assert report.calls_with_usage == 1
    assert report.calls_without_usage == 1
    assert report.actual_cost_usd == Decimal("3.00")
    assert report.within_ceiling is True


# ---------------------------------------------------------------------------
# Persistence: records survive the process when a run directory is given.
# ---------------------------------------------------------------------------


def test_records_persist_into_the_run_directory(tmp_path: Path) -> None:
    run = open_run(tmp_path)
    runner = _runner(ScriptedClient([_response(), ModelPermanentError("no")]), run=run)

    runner.execute(_case(), arm="cold", knowledge=[], repeat=1)
    runner.execute(_case(), arm="cold", knowledge=[], repeat=2)

    first_key = "call-alpha-cold-0001-0001"
    raw = run.read_artifact(first_key)
    assert raw is not None
    persisted = LiveCallRecord.model_validate_json(raw)
    assert persisted.model == "fake-model"
    assert persisted.ok is True
    failed_key = "call-alpha-cold-0002-0002"
    failed_raw = run.read_artifact(failed_key)
    assert failed_raw is not None
    assert LiveCallRecord.model_validate_json(failed_raw).ok is False


def test_runner_without_a_run_directory_persists_nothing_to_disk(
    tmp_path: Path,
) -> None:
    runner = _runner(ScriptedClient([_response()]))

    runner.execute(_case(), arm="cold", knowledge=[], repeat=1)

    assert runner.calls[0].model == "fake-model"
    assert list(tmp_path.iterdir()) == []
