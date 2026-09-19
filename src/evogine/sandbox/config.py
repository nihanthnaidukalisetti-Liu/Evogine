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
    # Exact pins: the evaluated tree may import pydantic/yaml, the suite uses
    # pytest + hypothesis, so all must exist inside the container. Matches the
    # versions this repo's CI validates.
    pinned_deps: tuple[str, ...] = (
        # anthropic: the swarm test modules import evogine.swarm.clients at
        # collection — without it the gate image cannot run the candidate
        # suite and every clean-patch acceptance is rejected on collection.
        "anthropic==1.7.0",
        "pydantic==2.13.5",
        "pyyaml==6.0.3",
        "pytest==9.1.1",
        "hypothesis==6.168.0",
    )
    memory_mb: int = 512
    cpus: float = 1.0
    pids_limit: int = 64
    timeout_seconds: float = 120.0
    # CI-diagnostic: widened from 20000 so the candidate suite's FULL output
    # (dots, sections, summary line) survives sanitization and reaches the
    # verdict reason — sanitize_output keeps the FIRST N chars, and the gate
    # takes the LAST 20KB of that window (revert after diagnosis).
    output_max_chars: int = 500_000
    tmpfs_size: str = "64m"
    # Run as nobody:nogroup — the container is never root.
    container_user: str = "65534:65534"
    container_workspace: str = "/workspace"
    # P2: seccomp is default-on and has no off switch. ``None`` selects the
    # vendored deny-by-default profile (see sandbox/seccomp.py); a non-None
    # value swaps in a custom profile — which must itself deny by default —
    # but can never disable syscall filtering.
    seccomp_profile_json: str | None = None

    @property
    def image_tag(self) -> str:
        """Content-addressed tag over the FULL rendered recipe.

        Hashing the rendered dockerfile (not just its inputs) means any
        provisioning change — a new pin, a new apt package — rehashes the tag
        and forces a rebuild; a matching tag can never silently reuse an image
        built from a different recipe.
        """
        digest = hashlib.sha256(self.dockerfile.encode()).hexdigest()[:12]
        return f"evogine-gate:py312-{digest}"

    @property
    def dockerfile(self) -> str:
        """The minimal, pinned provisioning recipe (built once, then cached).

        The pytest stage runs the full candidate suite inside this image, and
        the suite exercises the git-backed learning store directly — git is a
        hard runtime requirement of the candidate suite itself, so the image
        installs it via apt before the pip layer.
        """
        deps = " ".join(self.pinned_deps)
        return (
            f"FROM {self.python_image}\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends git"
            " && rm -rf /var/lib/apt/lists/*\n"
            f"RUN pip install --no-cache-dir {deps}\n"
        )
