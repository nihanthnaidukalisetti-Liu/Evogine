"""Seccomp syscall filtering: default-on for the execution sandbox.

P2 hardening. The sandbox always runs under a seccomp profile — there is no
off switch. The bundled default profile is vendored verbatim from moby
(``moby/moby`` tag ``v27.0.3``, ``profiles/seccomp/default.json``, Apache-2.0)
with no keys added or removed: it is the exact profile the Docker daemon
applies by default, pinned in-tree so the guarantee becomes explicit,
versioned, and independent of how the host daemon was configured — v1 left
seccomp to the daemon (effectively opt-in), so a daemon running with a
weakened or disabled profile silently downgraded the sandbox.

The profile is deny-by-default: anything not expressly permitted gets
``SCMP_ACT_ERRNO``, and privileged syscalls (``mount``, ``unshare``, ``setns``)
carry allow rules conditioned on capabilities the sandbox never grants
(``cap_drop=ALL``). The runner hands the daemon the profile as a file at
container creation; the file is written fresh per run and removed after.
The file-path handoff assumes the daemon shares this host's filesystem —
the single-host daemon posture this sandbox already documents.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Deny-by-default is the contract: a profile that defaults to allow (or to
#: killing the process, which is not fail-closed — it crashes the workload)
#: is rejected before it can weaken the sandbox.
_REQUIRED_DEFAULT_ACTION = "SCMP_ACT_ERRNO"


class SeccompError(RuntimeError):
    """A seccomp profile is missing, malformed, or too permissive to apply."""


def _load_vendored_profile() -> str:
    path = Path(__file__).parent / "seccomp_default.json"
    return path.read_text(encoding="utf-8")


def validate_profile_json(profile_json: str) -> dict[str, object]:
    """Return the parsed profile, rejecting anything that is not deny-by-default.

    The same validation covers the vendored default (a corrupted in-tree file
    must fail at run time, not silently weaken the sandbox) and caller-supplied
    overrides: a profile whose default action permits unlisted syscalls, or
    with no syscall rules at all, is an attempt to switch the hardening off.
    """
    try:
        profile = json.loads(profile_json)
    except json.JSONDecodeError as exc:
        raise SeccompError(f"seccomp profile is not valid JSON: {exc}") from exc
    if not isinstance(profile, dict):
        raise SeccompError("seccomp profile must be a JSON object")
    if profile.get("defaultAction") != _REQUIRED_DEFAULT_ACTION:
        raise SeccompError(
            "seccomp profile must deny by default "
            f"(defaultAction {_REQUIRED_DEFAULT_ACTION!r}); "
            f"got {profile.get('defaultAction')!r} — refusing to weaken the sandbox"
        )
    rules = profile.get("syscalls")
    if not isinstance(rules, list) or not rules:
        raise SeccompError("seccomp profile must carry a non-empty syscalls rule list")
    return profile


def load_profile_json(override: str | None) -> str:
    """Resolve the profile to run under: the bundled default, or a validated override.

    ``None`` selects the vendored default — seccomp is default-on and callers
    may swap profiles, never disable them.
    """
    if override is None:
        return _load_vendored_profile()
    validate_profile_json(override)
    return override


def build_security_opts(profile_path: str) -> list[str]:
    """The ``security_opt`` list the container runs under: no-new-privileges + seccomp."""
    return ["no-new-privileges:true", f"seccomp={profile_path}"]


@contextmanager
def profile_file(profile_json: str) -> Iterator[Path]:
    """Materialize the profile as a 0600 file the daemon reads at creation.

    The daemon consumes the path during container setup, so the file only
    needs to live until the create call returns; removal is best-effort — a
    leftover temp file cannot affect any verdict, while failing the run over
    cleanup would be worse than leaking it.
    """
    validate_profile_json(profile_json)
    fd, name = tempfile.mkstemp(prefix="evogine-seccomp-", suffix=".json")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(profile_json.encode("utf-8"))
        Path(name).chmod(0o600)
        yield Path(name)
    finally:
        with_cleanup_suppressed(Path(name))


def with_cleanup_suppressed(path: Path) -> None:
    """Best-effort unlink; failures are unobservable to the verdict either way."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


__all__ = [
    "SeccompError",
    "build_security_opts",
    "load_profile_json",
    "profile_file",
    "validate_profile_json",
]
