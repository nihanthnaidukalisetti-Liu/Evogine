"""Sandboxed execution for the objective gates.

Untrusted candidate code runs in a locked-down Docker container: no network
beyond loopback, pinned dependencies baked into a controlled image, capped
CPU/memory/PIDs, a host-side wall-clock kill, and output that is redacted and
size-capped before it can influence a verdict. Staging is path-traversal-safe
and container environments are built from an explicit allowlist — host
``os.environ`` is never forwarded (security review H1). The gate consuming
this module fails closed: sandbox failures are rejections, never acceptances.
"""

from evogine.sandbox.config import SandboxConfig
from evogine.sandbox.docker_runner import (
    DockerSandbox,
    SandboxError,
    SandboxResult,
    SandboxUnavailable,
    build_container_env,
)

__all__ = [
    "DockerSandbox",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "SandboxUnavailable",
    "build_container_env",
]
