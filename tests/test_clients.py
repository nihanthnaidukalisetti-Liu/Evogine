"""Client resilience tests against a loopback stub of the Messages API.

The stub is a real HTTP server on 127.0.0.1, so timeout and retry behavior
is exercised at the socket level through the real ``anthropic`` SDK client —
no live model is ever contacted. Retry tests stay deterministic: sleep is
recorded, never awaited, and jitter is pinned to its upper bound.
"""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from evogine.swarm.clients import (
    AnthropicClient,
    ModelPermanentError,
    ModelRequest,
    ModelTransientError,
    RetryingModelClient,
    RetryPolicy,
)


def _message_body(text: str = "patch ok") -> dict[str, object]:
    return {
        "id": "msg_stub_1",
        "type": "message",
        "role": "assistant",
        "model": "stub-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


def _error_body(type_: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": type_, "message": message}}


class _StubServer(ThreadingHTTPServer):
    """Gate-container-tolerant stub server.

    The CI gate container runs suites under tight process limits; a larger
    listen backlog keeps loaded runs from refusing connections that arrive
    while previous connection threads are still being reaped.
    """

    daemon_threads = True
    request_queue_size = 64


class StubAnthropicAPI:
    """Scriptable loopback stub of the Anthropic Messages API.

    The last scripted response repeats, so exhaustion tests need one entry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._script: list[tuple[int, dict[str, object], float, dict[str, str]]] = []
        self._requests: list[dict[str, object]] = []
        self._server = _StubServer(("127.0.0.1", 0), _handler_for(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def enqueue(
        self,
        status: int,
        payload: dict[str, object] | None = None,
        *,
        delay_s: float = 0.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            self._script.append((status, payload or _message_body(), delay_s, headers or {}))

    def requests(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._requests)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _next_response(
        self, body: dict[str, object]
    ) -> tuple[int, dict[str, object], float, dict[str, str]]:
        with self._lock:
            self._requests.append(body)
            if len(self._script) > 1:
                return self._script.pop(0)
            return self._script[0]


def _handler_for(api: StubAnthropicAPI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            status, payload, delay, extra = api._next_response(body)
            if delay:
                time.sleep(delay)
            data = json.dumps(payload).encode()
            try:
                self.send_response(status)
                for key, value in extra.items():
                    self.send_header(key, value)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The client timed out and hung up while this handler was
                # still working (e.g. the scripted delay outlived the client's
                # timeout under a loaded runner). Abandon the response quietly
                # instead of surfacing a BrokenPipe traceback; the test asserts
                # on the client side.
                return

        def log_message(self, format: str, *args: object) -> None:
            return  # keep test output clean

    return Handler


class MaxJitter:
    """Jitter stand-in pinned to the upper bound, for exact delay asserts."""

    def uniform(self, a: float, b: float) -> float:
        return b


class SleepRecorder:
    """Records requested sleeps; nothing actually waits."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture
def stub_api() -> Iterator[StubAnthropicAPI]:
    api = StubAnthropicAPI()
    try:
        yield api
    finally:
        api.close()


def _provider(api: StubAnthropicAPI, *, timeout_s: float = 2.0) -> AnthropicClient:
    return AnthropicClient(
        model="stub-model", api_key="test-key", base_url=api.url, timeout_s=timeout_s
    )


def _retrying(
    api: StubAnthropicAPI, policy: RetryPolicy, sleep: SleepRecorder, *, timeout_s: float = 2.0
) -> RetryingModelClient:
    return RetryingModelClient(
        _provider(api, timeout_s=timeout_s), policy, sleep=sleep, jitter=MaxJitter()
    )


def test_retries_transient_5xx_then_succeeds(stub_api: StubAnthropicAPI) -> None:
    stub_api.enqueue(500, _error_body("api_error", "boom"))
    stub_api.enqueue(500, _error_body("api_error", "boom"))
    stub_api.enqueue(200)
    sleep = SleepRecorder()

    client = _retrying(
        stub_api, RetryPolicy(max_attempts=3, initial_backoff_s=0.01, max_backoff_s=1.0), sleep
    )
    response = client.complete(ModelRequest(system="s", user="u"))

    assert response.text == "patch ok"
    assert response.model == "stub-model"
    assert len(stub_api.requests()) == 3
    assert sleep.delays == [0.01, 0.02]  # exponential, jitter pinned high


def test_honors_retry_after_over_exponential_backoff(stub_api: StubAnthropicAPI) -> None:
    stub_api.enqueue(
        429, _error_body("rate_limit_error", "slow down"), headers={"retry-after": "2"}
    )
    stub_api.enqueue(200)
    sleep = SleepRecorder()

    client = _retrying(
        stub_api, RetryPolicy(max_attempts=2, initial_backoff_s=0.01, max_backoff_s=8.0), sleep
    )
    response = client.complete(ModelRequest(system="s", user="u"))

    assert response.text == "patch ok"
    assert sleep.delays == [2.0]  # Retry-After floor beats 0.01s exponential


def test_does_not_retry_permanent_4xx(stub_api: StubAnthropicAPI) -> None:
    stub_api.enqueue(400, _error_body("invalid_request_error", "nope"))
    sleep = SleepRecorder()

    client = _retrying(stub_api, RetryPolicy(max_attempts=4), sleep)
    with pytest.raises(ModelPermanentError):
        client.complete(ModelRequest(system="s", user="u"))

    assert len(stub_api.requests()) == 1
    assert sleep.delays == []


def test_raises_after_transient_exhaustion(stub_api: StubAnthropicAPI) -> None:
    stub_api.enqueue(500, _error_body("api_error", "boom"))
    sleep = SleepRecorder()

    client = _retrying(
        stub_api,
        RetryPolicy(max_attempts=3, initial_backoff_s=0.01, max_backoff_s=0.02),
        sleep,
    )
    with pytest.raises(ModelTransientError):
        client.complete(ModelRequest(system="s", user="u"))

    assert len(stub_api.requests()) == 3
    assert sleep.delays == [0.01, 0.02]


def test_timeout_is_transient_and_retried(stub_api: StubAnthropicAPI) -> None:
    stub_api.enqueue(200, delay_s=1.0)  # far beyond the 0.1s client timeout
    sleep = SleepRecorder()

    client = _retrying(
        stub_api,
        RetryPolicy(max_attempts=2, initial_backoff_s=0.01, max_backoff_s=0.02),
        sleep,
        timeout_s=0.1,
    )
    with pytest.raises(ModelTransientError, match="timed out"):
        client.complete(ModelRequest(system="s", user="u"))

    assert len(stub_api.requests()) == 2  # both attempts reached the stub


def test_rate_limit_maps_to_transient_with_retry_after(stub_api: StubAnthropicAPI) -> None:
    stub_api.enqueue(
        429, _error_body("rate_limit_error", "slow down"), headers={"retry-after": "7"}
    )

    with pytest.raises(ModelTransientError) as raised:
        _provider(stub_api).complete(ModelRequest(system="s", user="u"))

    assert raised.value.retry_after == 7.0


def test_retry_policy_rejects_zero_attempts() -> None:
    with pytest.raises(ValidationError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
