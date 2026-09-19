"""Tests for ``evogine init`` and ``evogine doctor``: happy paths, refusal paths.

All offline: the Docker verdict is injected (core CI has no daemon by
design — the docker-gated job runs the real probe), the provider key
rides the environment as a fake value that must never be echoed, and the
sync probe hits a loopback HTTP server — never an external network.
"""

from __future__ import annotations

import http.server
import socket
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from evogine.cli import (
    CheckResult,
    CheckState,
    _check_docker,
    _disk_verdict,
    main,
)
from evogine.config import DEFAULT_CONFIG_FILENAME, load_config
from evogine.engine.store import Engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(check: str, detail: str = "ok") -> CheckResult:
    return CheckResult(check=check, state=CheckState.OK, detail=detail)


def _fake_docker_module(*, ping_raises: bool) -> types.ModuleType:
    """A stand-in for the optional docker-py module (the import is lazy)."""

    class _FakeClient:
        def ping(self) -> None:
            if ping_raises:
                raise RuntimeError("cannot connect to the Docker daemon")

    module = types.ModuleType("docker")
    module.from_env = lambda: _FakeClient()  # type: ignore[attr-defined]
    return module


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Silence per-request logging — the test report is the evidence."""


@pytest.fixture
def sync_server_port() -> int:
    """A loopback HTTP server answering 200; yields its port, then shuts down."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def closed_port() -> int:
    """A loopback port that (almost certainly) refuses connections."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _config_with_endpoint(learning_root: str, port: int) -> str:
    return f"""\
default_profile: default
profiles:
  default:
    store:
      learning_root: {learning_root}
    provider:
      model: test-model
    sync:
      endpoints:
        - name: local
          url: http://127.0.0.1:{port}
"""


# ---------------------------------------------------------------------------
# evogine init
# ---------------------------------------------------------------------------


def test_init_creates_the_store_and_the_starter_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh init: store with git history, starter config, key fix printed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main(["init"]) == 0

    store = tmp_path / ".evogine" / "store"
    assert (store / ".git").is_dir()
    assert (store / "records").is_dir()
    config_path = tmp_path / DEFAULT_CONFIG_FILENAME
    assert config_path.is_file()

    out = capsys.readouterr().out
    assert "created at .evogine/store (0 record(s))" in out  # the root as given
    assert f"wrote {config_path}" in out
    assert "ANTHROPIC_API_KEY is not set" in out
    assert "export ANTHROPIC_API_KEY=<your provider key>" in out
    assert "next: evogine doctor" in out

    # The written config loads back and points at the store init created.
    config = load_config(config_path)
    assert config.profile().store.learning_root == Path(".evogine/store")


def test_init_reports_a_present_key_without_echoing_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walkthrough names the env var and never prints any key value."""
    monkeypatch.chdir(tmp_path)
    fake_key = "sk-ant-fake-secret-value-123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)

    assert main(["init"]) == 0

    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY is set in the environment" in out
    assert fake_key not in out


def test_init_refuses_to_overwrite_an_existing_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing config refuses before anything is created — all-or-nothing."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / DEFAULT_CONFIG_FILENAME
    config_path.write_text("operator: hand-written\n", encoding="utf-8")

    assert main(["init"]) == 1

    err = capsys.readouterr().err
    assert "already exists" in err
    assert "refusing to overwrite" in err
    assert config_path.read_text(encoding="utf-8") == "operator: hand-written\n"
    assert not (tmp_path / ".evogine" / "store").exists()  # nothing was created


def test_init_on_an_existing_store_is_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    record_factory: Any,
) -> None:
    """A pre-existing store reports itself, with its record count — no reset."""
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "store"
    engine = Engine(store)
    engine.propose(record_factory("rec-onboard-1"))

    assert main(["init", "--root", str(store)]) == 0

    out = capsys.readouterr().out
    assert f"already present at {store}" in out
    assert "1 record(s)" in out
    # The store is untouched (still one record) and the config was written.
    assert len(list(Engine(store).records_dir.glob("*.json"))) == 1
    assert (tmp_path / DEFAULT_CONFIG_FILENAME).is_file()


def test_init_failure_path_is_a_visible_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store root under a file (not a directory) refuses with the reason."""
    monkeypatch.chdir(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    assert main(["init", "--root", str(blocker / "store")]) == 1

    assert "could not create the learning store" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# evogine doctor
# ---------------------------------------------------------------------------


def _patch_environment_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    git: CheckResult | None = None,
    docker: CheckResult | None = None,
    disk: CheckResult | None = None,
) -> None:
    """Pin the environment-dependent checks so verdicts are deterministic."""
    if git is not None:
        monkeypatch.setattr("evogine.cli._check_git", lambda: git)
    if docker is not None:
        monkeypatch.setattr("evogine.cli._check_docker", lambda: docker)
    if disk is not None:
        monkeypatch.setattr("evogine.cli._check_disk", lambda _root: disk)


def test_doctor_all_ok_reports_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every check ok: ready, exit 0, one line per check plus the summary."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-secret-value-123")
    store = tmp_path / "store"
    Engine(store)
    config_path = tmp_path / DEFAULT_CONFIG_FILENAME
    config_path.write_text(
        f"""\
default_profile: default
profiles:
  default:
    store:
      learning_root: {store}
    provider:
      model: test-model
""",
        encoding="utf-8",
    )
    _patch_environment_checks(
        monkeypatch,
        git=_ok("git", "git version 9.9.9"),
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    for name in ("config", "git", "docker", "provider", "disk", "sync"):
        assert f"  {name}" in out
    assert str(config_path) in out
    assert "(profile: default)" in out
    assert "git version 9.9.9" in out
    assert "sk-ant-fake-secret-value-123" not in out  # the value is never echoed
    assert "doctor: ready" in out
    assert "not ready" not in out


def test_doctor_without_a_config_warns_and_still_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No config: a visible warning with the fix, environment checks still run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_environment_checks(
        monkeypatch,
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "no config file at" in out
    assert "fix: run 'evogine init'" in out
    assert "doctor: ready" in out  # a warning is not a failure


def test_doctor_an_invalid_config_fails_the_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config that refuses to load is a failure state with the fix."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / DEFAULT_CONFIG_FILENAME).write_text(
        "default_profile: [unclosed", encoding="utf-8"
    )
    _patch_environment_checks(
        monkeypatch,
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "not valid YAML" in out
    assert "fix: correct the config file by hand" in out
    assert "doctor: not ready" in out


def test_doctor_reports_a_failed_check_with_its_fix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing check names the defect, prints the fix, and fails the run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-secret-value-123")
    git_fail = CheckResult(
        check="git",
        state=CheckState.FAIL,
        detail="the git executable was not found",
        fix="install git (https://git-scm.com)",
    )
    _patch_environment_checks(
        monkeypatch,
        git=git_fail,
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "the git executable was not found" in out
    assert "fix: install git (https://git-scm.com)" in out
    assert "doctor: not ready" in out
    assert "1 failure(s)" in out


def test_doctor_warns_when_the_provider_key_is_absent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent key warns with the exact fix; warnings never fail the run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_environment_checks(
        monkeypatch,
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY is not set" in out
    assert "export ANTHROPIC_API_KEY=<your provider key>" in out
    assert "doctor: ready" in out


def test_doctor_probes_a_reachable_sync_endpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    sync_server_port: int,
) -> None:
    """A configured endpoint that answers reports ok."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / DEFAULT_CONFIG_FILENAME).write_text(
        _config_with_endpoint(str(tmp_path / "store"), sync_server_port),
        encoding="utf-8",
    )
    _patch_environment_checks(
        monkeypatch,
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "endpoint 'local' answered" in out
    assert f"http://127.0.0.1:{sync_server_port}" in out
    assert "doctor: ready" in out


def test_doctor_fails_when_a_sync_endpoint_is_unreachable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    closed_port: int,
) -> None:
    """An unreachable endpoint is a failure state with the fix, exit 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / DEFAULT_CONFIG_FILENAME).write_text(
        _config_with_endpoint(str(tmp_path / "store"), closed_port),
        encoding="utf-8",
    )
    _patch_environment_checks(
        monkeypatch,
        docker=_ok("docker", "daemon reachable"),
        disk=_ok("disk", "100.0 GiB free"),
    )

    assert main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "endpoint 'local' unreachable" in out
    assert "fix: start the sync daemon" in out
    assert "doctor: not ready" in out


# ---------------------------------------------------------------------------
# Unit checks: docker probe states and disk thresholds
# ---------------------------------------------------------------------------


def test_check_docker_ok_when_a_daemon_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "docker", _fake_docker_module(ping_raises=False))
    result = _check_docker()
    assert result.state is CheckState.OK
    assert "daemon reachable" in result.detail


def test_check_docker_fails_when_the_client_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "docker", None)  # None in sys.modules: import halts
    result = _check_docker()
    assert result.state is CheckState.FAIL
    assert "not installed" in result.detail
    assert result.fix is not None and "evogine[docker]" in result.fix


def test_check_docker_fails_when_the_daemon_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "docker", _fake_docker_module(ping_raises=True))
    result = _check_docker()
    assert result.state is CheckState.FAIL
    assert "no usable Docker daemon" in result.detail
    assert result.fix is not None and "start the Docker daemon" in result.fix


@pytest.mark.parametrize(
    ("free_bytes", "expected"),
    [
        (0, CheckState.FAIL),
        (1024**3 - 1, CheckState.FAIL),  # one byte under the fail floor
        (1024**3, CheckState.WARN),  # exactly at 1 GiB: warn, not fail
        (5 * 1024**3 - 1, CheckState.WARN),  # one byte under the warn floor
        (5 * 1024**3, CheckState.OK),  # exactly at 5 GiB: ok, not warn
        (100 * 1024**3, CheckState.OK),
    ],
)
def test_disk_verdict_thresholds(free_bytes: int, expected: CheckState) -> None:
    state, _detail, _fix = _disk_verdict(free_bytes, Path("/somewhere"))
    assert state is expected
