"""Sandbox configuration contract: pinned dependencies, capped resources.

Everything the sandbox is allowed to do is pinned here, frozen. ``pinned_deps``
are installed at image-build time — the only moment the sandbox touches the
network. Execution never does: containers run with no interface beyond
loopback, so outbound connections blackhole.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict


class SandboxConfig(BaseModel):
    """Immutable knobs for the execution sandbox."""

    model_config = ConfigDict(frozen=True)

    python_image: str = "python:3.12-slim"
    # Exact pins: the evaluated tree may import pydantic/yaml and pytest must
    # exist inside the container. Matches the versions this repo's CI validates.
    pinned_deps: tuple[str, ...] = (
        "pydantic==2.13.5",
        "pyyaml==6.0.3",
        "pytest==9.1.1",
    )
    memory_mb: int = 512
    cpus: float = 1.0
    pids_limit: int = 64
    timeout_seconds: float = 120.0
    output_max_chars: int = 20_000
    tmpfs_size: str = "64m"
    # Run as nobody:nogroup — the container is never root.
    container_user: str = "65534:65534"
    container_workspace: str = "/workspace"

    @property
    def image_tag(self) -> str:
        """Content-addressed tag: a config change rebuilds, a match reuses."""
        payload = self.python_image + "|" + " ".join(self.pinned_deps)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"evogine-gate:py312-{digest}"

    @property
    def dockerfile(self) -> str:
        """The minimal, pinned provisioning recipe (built once, then cached)."""
        deps = " ".join(self.pinned_deps)
        return f"FROM {self.python_image}\nRUN pip install --no-cache-dir {deps}\n"
