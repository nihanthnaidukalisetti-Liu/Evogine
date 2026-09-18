"""Offline hardening units: env allowlist, output sanitization, runner behavior.


No Docker daemon is needed — the runner is exercised through a fake client,
so these tests run in the plain offline suite and verify the contract at the
docker-py call boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evogine.sandbox.config import SandboxConfig
from evogine.sandbox.docker_runner import (
    DockerSandbox,
    SandboxError,
    SandboxUnavailable,
    build_container_env,
)
from evogine.sandbox.sanitize import sanitize_output

# --- env allowlist (security review H1) ---


def test_container_env_never_inherits_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOGINE_PROBE_SECRET", "probe-secret-value-9f2c")
    monkeypatch.setenv("EVOGINE_ALLOWED_VAR", "yes")
    env = build_container_env({"EVOGINE_ALLOWED_VAR": "yes"})
    assert env["EVOGINE_ALLOWED_VAR"] == "yes"
    assert "EVOGINE_PROBE_SECRET" not in env
    assert "probe-secret-value-9f2c" not in repr(env)


def test_container_env_includes_runner_defaults() -> None:
    env = build_container_env({})
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["HOME"] == "/tmp"


def test_container_env_rejects_bad_keys() -> None:
    for key in ("lowercase", "WITH-DASH", "9STARTS-DIGIT", "PATH=INJECTION"):
        with pytest.raises(SandboxError):
            build_container_env({key: "1"})


# --- output sanitization ---


def test_sanitize_redacts_secret_shaped_tokens() -> None:
    out = sanitize_output("key sk-abc123defgh456 end")
    assert "sk-abc123defgh456" not in out
    assert "<redacted>" in out


def test_sanitize_redacts_labeled_api_keys() -> None:
    out = sanitize_output('config: api_key = "supersecretvalue" ok')
    assert "supersecretvalue" not in out


def test_sanitize_redacts_staging_paths() -> None:
    out = sanitize_output(
        "error at /tmp/evogine-gate-x/baseline/mod.py",
        redactions={"/tmp/evogine-gate-x": "<staging>"},
    )
    assert "/tmp/evogine-gate-x" not in out
    assert "<staging>/baseline/mod.py" in out


def test_sanitize_caps_length() -> None:
    out = sanitize_output("x" * 30_000, max_chars=1_000)
    assert len(out) <= 1_100
    assert "truncated" in out


def test_sanitize_strips_ansi_escapes() -> None:
    assert "\x1b[" not in sanitize_output("\x1b[31mred\x1b[0m plain")


# --- runner behavior through a fake client ---


class FakeContainer:
    def __init__(
        self,
        *,
        statuses: list[str] | None = None,
        exit_code: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self._statuses = statuses if statuses is not None else ["exited"]
        self.attrs: dict = {"State": {"ExitCode": exit_code}}
        self._stdout = stdout
        self._stderr = stderr
        self.status = "created"
        self.kill_count = 0
        self.remove_count = 0
        # Content of any seccomp profile passed at container creation — the
        # seccomp tests snapshot it here, because the runner removes the
        # profile file once the daemon has consumed it.
        self.seccomp_profile_read: str | None = None

    def reload(self) -> None:
        if self._statuses:
            self.status = self._statuses.pop(0)

    def kill(self) -> None:
        self.kill_count += 1
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.remove_count += 1

    def logs(self, stdout: bool = False, stderr: bool = False) -> bytes:
        return self._stdout if stdout else self._stderr


class FakeImages:
    def __init__(self) -> None:
        self.built: list[str] = []
        self.get_fails = True
        self.build_fails = False

    def get(self, tag: str) -> None:
        if not self.get_fails:
            return
        raise RuntimeError("no such image")

    def build(self, *, fileobj: object, tag: str, rm: bool) -> None:
        if self.build_fails:
            raise RuntimeError("build exploded")
        self.built.append(tag)
        self.last_fileobj = fileobj


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self._container = container
        self.run_fails = False
        self.last_kwargs: dict[str, object] = {}

    def run(self, *args: object, **kwargs: object) -> FakeContainer:
        self.last_kwargs = kwargs
        security_opt = kwargs.get("security_opt")
        if isinstance(security_opt, list):
            for opt in security_opt:
                if isinstance(opt, str) and opt.startswith("seccomp="):
                    # Engine 25+ takes the profile JSON inline — capture it directly.
                    self._container.seccomp_profile_read = opt[len("seccomp=") :]
        if self.run_fails:
            raise RuntimeError("cannot start container")
        return self._container


class FakeClient:
    def __init__(self, container: FakeContainer | None = None) -> None:
        self.container = container or FakeContainer()
        self.images = FakeImages()
        self.containers = FakeContainers(self.container)


def test_run_returns_sanitized_result(tmp_path: Path) -> None:
    staging = tmp_path / "evogine-gate-secret"
    staging.mkdir()
    container = FakeContainer(stdout=f"leak {staging} done\n".encode())
    sandbox = DockerSandbox(client=FakeClient(container))
    result = sandbox.run("true", staging_dir=staging)
    assert result.exit_code == 0
    assert str(staging) not in result.stdout
    assert "<staging>" in result.stdout
    assert container.remove_count == 1


def test_run_kills_containers_past_the_deadline() -> None:
    container = FakeContainer(statuses=["running"], exit_code=137)
    sandbox = DockerSandbox(config=SandboxConfig(timeout_seconds=0.4), client=FakeClient(container))
    result = sandbox.run("sleep 1000")
    assert result.timed_out is True
    assert result.exit_code == 124
    assert container.kill_count == 1
    assert container.remove_count == 1


def test_run_wraps_container_start_failures() -> None:
    client = FakeClient()
    client.containers.run_fails = True
    sandbox = DockerSandbox(client=client)
    with pytest.raises(SandboxError, match="container start failed"):
        sandbox.run("true")


def test_ensure_image_builds_once_then_reuses() -> None:
    client = FakeClient()
    sandbox = DockerSandbox(client=client)
    tag = sandbox.ensure_image()
    assert sandbox.ensure_image() == tag
    assert client.images.built == [tag]


def test_ensure_image_builds_from_bytesio_context() -> None:
    """The build fileobj must be an ``io.BytesIO`` of the Dockerfile.

    docker-py's ``mkbuildcontext`` packs a ``BytesIO`` into the build
    context, but anything else goes through ``t.gettarinfo`` which reads
    ``fileobj.name`` — raw bytes explode there (caught by the CI
    docker-gated job, the first run that ever executed this path).
    """
    import io
    import tarfile

    client = FakeClient()
    sandbox = DockerSandbox(client=client)
    sandbox.ensure_image()

    fileobj = client.images.last_fileobj
    assert isinstance(fileobj, io.BytesIO)
    assert fileobj.getvalue() == sandbox._config.dockerfile.encode("utf-8")

    # Round-trip through the real docker-py context builder (client-side
    # only, no daemon): the tar must contain the Dockerfile verbatim.
    mkbuildcontext = pytest.importorskip("docker.utils.build").mkbuildcontext
    context = mkbuildcontext(io.BytesIO(fileobj.getvalue()))
    with tarfile.open(fileobj=context) as tar:
        packed = tar.extractfile(tar.getmember("Dockerfile"))
        assert packed is not None and packed.read() == fileobj.getvalue()


def test_image_tag_is_a_digest_of_the_full_recipe() -> None:
    """Any provisioning change must rehash the tag.

    The tag hashes the rendered dockerfile, not just its inputs — a recipe
    change without a dep-list change (e.g. a new apt package) would otherwise
    match the cached tag and silently reuse an image built from a different
    recipe (caught by the CI docker-gated job).
    """
    import hashlib

    client = FakeClient()
    sandbox = DockerSandbox(client=client)
    digest = hashlib.sha256(sandbox._config.dockerfile.encode()).hexdigest()[:12]
    assert sandbox._config.image_tag == f"evogine-gate:py312-{digest}"


def test_ensure_image_build_failure_raises_sandbox_error() -> None:
    client = FakeClient()
    client.images.build_fails = True
    sandbox = DockerSandbox(client=client)
    with pytest.raises(SandboxError, match="gate image build failed"):
        sandbox.ensure_image()


@pytest.mark.skipif(
    DockerSandbox.available(),
    reason="a working daemon is present — SandboxUnavailable cannot be probed",
)
def test_run_without_daemon_raises_unavailable() -> None:
    sandbox = DockerSandbox()
    with pytest.raises(SandboxUnavailable):
        sandbox.run("true")
