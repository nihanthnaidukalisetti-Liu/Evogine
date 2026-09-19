"""Daemon settings and the binding policy — self-contained by design.

The daemon's knobs live here and nowhere else (the V4 plan assigns the
unified config file to another task; this module must not grow a shared
config import). Everything the ``evogine-syncd`` entry point needs to
refuse a bad launch before any socket exists is defined in this module.

The binding policy is contractual, from the Blueprint's sync-region
invariant: the daemon binds loopback by default; an explicit flag may
expose *read* endpoints on the LAN; store-mutating and approval endpoints
stay loopback-only regardless of any flag. There is no auth stack by
scope decision (single-operator deployment), so the bind address and the
peer address are the trust anchors — which is why both are validated
here, structurally, rather than left to caller discipline.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import cast

from pydantic import BaseModel

#: The loopback bind the daemon uses unless told otherwise.
DEFAULT_HOST = "127.0.0.1"

#: Default listen port. The operator console owns 8765; the daemon takes
#: the adjacent port so both can run on one machine untouched.
DEFAULT_PORT = 8766


class SyncError(Exception):
    """A sync-daemon contract violation: launch refusal or read-path refusal.

    ``http_status`` carries the visible refusal state a route (or the
    exception handler in :mod:`evogine.sync.app`) should answer with;
    launch-time refusals simply use the exception's message on stderr.
    """

    def __init__(self, message: str, *, http_status: int = 500) -> None:
        super().__init__(message)
        self.http_status = http_status


def resolve_bind_host(host: str, *, expose_lan: bool) -> str:
    """Return the concrete address to bind, enforcing the binding policy.

    With ``expose_lan`` false (the default) every address ``host`` resolves
    to must be loopback — a name resolving to a mix of loopback and public
    addresses is refused too, because the bind would take the public one.
    With ``expose_lan`` true any address is accepted: that flag is the
    explicit operator decision to expose *read* endpoints beyond the
    machine, and it cannot be made accidentally.

    The returned value is the resolved address itself (``localhost`` binds
    ``127.0.0.1``, not a name), so what is validated is exactly what
    uvicorn hands to ``socket.bind``.
    """
    try:
        infos = socket.getaddrinfo(host.strip(), None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SyncError(f"cannot resolve bind host {host!r}: {exc}") from exc
    if not infos:
        raise SyncError(f"bind host {host!r} resolves to no addresses")
    addresses = {cast(str, sockaddr[0]) for *_, sockaddr in infos}
    if not expose_lan:
        exposed = sorted(
            address for address in addresses if not ipaddress.ip_address(address).is_loopback
        )
        if exposed:
            raise SyncError(
                f"refusing non-loopback bind: {host!r} resolves to {', '.join(exposed)}; "
                "the daemon binds loopback only by default — pass --expose-lan to expose "
                "READ endpoints beyond this machine (store-mutating and approval "
                "endpoints stay loopback-only regardless)"
            )
    # Deterministic pick: IPv4 sorts before IPv6 lexically, and every
    # resolved address is loopback by the check above (when the policy gate
    # is active).
    return min(addresses)


def is_loopback_peer(peer: str | None) -> bool:
    """Whether the request's peer address is this machine (loopback).

    The gate for every store-mutating endpoint. Fail-closed by construction:
    a missing peer (no client address in the ASGI scope) or a non-IP peer
    (a proxy-supplied name) is treated as remote — the mutating surface
    never trusts what it cannot resolve to loopback.
    """
    if not peer:
        return False
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


class SyncSettings(BaseModel):
    """The daemon's own settings: where the stores live and how it binds.

    Validation is explicit (:meth:`validated`, :meth:`resolved_bind_host`)
    rather than a pydantic validator, so construction stays cheap and the
    launch refusals are the entry point's first output, not a wrapped
    ValidationError.
    """

    stores_root: Path
    """Directory whose direct children are the hosted learning stores."""

    host: str = DEFAULT_HOST
    """Bind host. Loopback by default; anything else needs ``expose_lan``."""

    port: int = DEFAULT_PORT
    """Listen port, 1-65535."""

    expose_lan: bool = False
    """The explicit flag that allows a non-loopback bind for READ endpoints.

    Store-mutating endpoints stay loopback-only regardless: the gate is the
    per-request peer check in :mod:`evogine.sync.app`, not the bind address.
    """

    def validated(self) -> SyncSettings:
        """Refuse a launch the daemon cannot honor, before any socket exists.

        A stores root that does not exist means the daemon was pointed at
        the wrong place (it hosts existing stores; it does not create a
        farm of empty ones), and an out-of-range port is a typo that
        deserves a visible refusal rather than uvicorn's own error.
        """
        root = self.stores_root
        if not root.is_dir():
            raise SyncError(f"stores root does not exist: {root}")
        if not 0 < self.port <= 65535:
            raise SyncError(f"port must be in 1-65535, got {self.port}")
        return self

    def resolved_bind_host(self) -> str:
        """The binding policy applied to this settings instance."""
        return resolve_bind_host(self.host, expose_lan=self.expose_lan)
