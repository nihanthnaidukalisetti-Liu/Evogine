"""The Docker execution sandbox and its typed, sanitized results.

Execution posture (security review C1/H1, carried into V2):

- ``network_mode="none"``: no interface beyond loopback — outbound connections
  blackhole. The network is touched only once, at pinned image build.
- Read-only root filesystem; writable ``/tmp`` is a size-capped tmpfs.
- No capabilities (``cap_drop=ALL``), ``no-new-privileges``, PID limit,
  memory cap, CPU quota, non-root user, and a host-side wall-clock kill.
- The container environment is BUILT from runner defaults plus an explicit
  allowlist — host ``os.environ`` is never forwarded (review H1).
- Container output leaves the sandbox only through :func:`sanitize_output`.

docker-py is imported lazily so the offline suite never needs it installed.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from evogine.sandbox.config import SandboxConfig
from evogine.sandbox.sanitize import sanitize_output

if TYPE_CHECKING:
    from docker import DockerClient

class SandboxError(RuntimeError):
    """The sandbox could not run the workload (infrastructure failure)."""


class SandboxUnavailable(SandboxError):
    """No usable Docker daemon (or docker-py) on this host."""


#: Environment every container gets. Fixed, explicit, never host-derived.
RUNNER_ENV: dict[str, str] = {
    "HOME": "/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}

_ENV_KEY = re.compile(r"[A-Z][A-Z0-9_]*")


def build_container_env(allowlist: Mapping[str, str]) -> dict[str, str]:
    """Runner defaults plus explicit allowlisted keys — and nothing else, ever.

    This is the H1 guarantee: even a poisoned ``os.environ`` on the host cannot
    reach the container, because nothing here reads ``os.environ``.
    """
    env = dict(RUNNER_ENV)
    for key, value in allowlist.items():
        if not _ENV_KEY.fullmatch(key):
            raise SandboxError(f"env key rejected: {key[:50]!r}")
        env[key] = value
    return env


class SandboxResult(BaseModel):
    """Sanitized outcome of one sandboxed command run."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    truncated: bool


def _make_readable(root: Path) -> None:
    """Containers read as uid 65534; staged trees must be world-readable."""
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            continue
        path.chmod(0o755 if path.is_dir() else 0o644)


class DockerSandbox:
    """Runs commands in a locked-down container; fails closed by design."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        client: DockerClient | None = None,
    ) -> None:
        self._config = config or SandboxConfig()
        self._client = client
        self._image: str | None = None

    @property
    def config(self) -> SandboxConfig:
        return self._config

    @staticmethod
    def available() -> bool:
        """True iff docker-py imports AND a daemon answers — probe, never raise."""
        try:
            import docker

            docker.from_env().ping()
        except Exception:
            return False
        return True

    def _get_client(self) -> DockerClient:
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
            except Exception as exc:
                raise SandboxUnavailable(f"docker unavailable: {exc}") from exc
        return self._client

    def ensure_image(self) -> str:
        """Build (or reuse) the pinned gate image. The ONLY networked step.

        A missing image is normal (first run) and built from the pinned
        Dockerfile; a failed build is an infrastructure error, not a retry loop.
        """
        if self._image is not None:
            return self._image
        client = self._get_client()
        tag = self._config.image_tag
        try:
            client.images.get(tag)
        except Exception:  # image absent → build it once
            try:
                client.images.build(
                    fileobj=self._config.dockerfile.encode(), tag=tag, rm=True
                )
            except Exception as exc:
                raise SandboxError(
                    f"gate image build failed: {sanitize_output(str(exc), max_chars=300)}"
                ) from exc
        self._image = tag
        return tag

    def run(
        self,
        command: str,
        *,
        staging_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> SandboxResult:
        """Run ``command`` in the sandbox and return its sanitized result.

        ``staging_dir`` is mounted read-only at ``/workspace`` — the container
        never writes what it verifies. ``env`` keys are allowlist-only. A wall
        clock past ``timeout_seconds`` kills the container (exit code 124).
        """
        cfg = self._config
        client = self._get_client()
        image = self.ensure_image()
        redactions: dict[str, str] = {}
        volumes: dict[str, object] = {}
        wd = workdir or cfg.container_workspace
        if staging_dir is not None:
            _make_readable(staging_dir)
            volumes[str(staging_dir)] = {"bind": cfg.container_workspace, "mode": "ro"}
            redactions[str(staging_dir)] = "<staging>"

        started = time.monotonic()
        try:
            container = client.containers.run(
                image,
                command,
                detach=True,
                user=cfg.container_user,
                working_dir=wd,
                environment=build_container_env(env or {}),
                mem_limit=f"{cfg.memory_mb}m",
                nano_cpus=int(cfg.cpus * 1_000_000_000),
                pids_limit=cfg.pids_limit,
                network_mode="none",
                read_only=True,
                tmpfs={"/tmp": f"size={cfg.tmpfs_size},mode=1777"},
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                volumes=volumes,
            )
        except Exception as exc:
            raise SandboxError(
                f"container start failed: {sanitize_output(str(exc), max_chars=300)}"
            ) from exc

        timed_out = False
        exit_code = 1
        try:
            deadline = started + cfg.timeout_seconds
            while True:
                container.reload()
                if container.status in {"exited", "dead"}:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    container.kill()
                    container.reload()
                    break
                time.sleep(0.1)
            exit_code = (
                124
                if timed_out
                else int(container.attrs.get("State", {}).get("ExitCode", 1))
            )
            raw_stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", "replace"
            )
            raw_stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", "replace"
            )
        finally:
            # Best-effort cleanup, deliberately: the run's output is already
            # captured, and a leaked container is visible in `docker ps` but
            # can never influence a verdict. Discarding output over a cleanup
            # failure would be worse than leaking a dead container.
            with contextlib.suppress(Exception):
                container.remove(force=True)

        truncated = (
            len(raw_stdout) > cfg.output_max_chars
            or len(raw_stderr) > cfg.output_max_chars
        )
        return SandboxResult(
            exit_code=exit_code,
            stdout=sanitize_output(
                raw_stdout, max_chars=cfg.output_max_chars, redactions=redactions
            ),
            stderr=sanitize_output(
                raw_stderr, max_chars=cfg.output_max_chars, redactions=redactions
            ),
            timed_out=timed_out,
            duration_ms=int((time.monotonic() - started) * 1000),
            truncated=truncated,
        )
