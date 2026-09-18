"""Launch-time refusals for the operator console: root, identity, bind host.

The console has no auth stack by design (the V3 spec's deployment decision:
loopback only, machine possession is the trust anchor), so its safety is
enforced here, before any socket exists — a store root that is not a store,
an empty operator identity, or a bind host that leaves the loopback
interface is refused at startup, never at request time.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import cast


class ConsoleError(Exception):
    """A console launch or view request violates the console contract.

    Raised for launch-time refusals (bad root, empty identity, non-loopback
    bind) and for read-path refusals (unknown proposal, malformed audit
    line) — everywhere the console must fail visibly rather than render a
    plausible lie.
    """


# What makes a directory an Evogine store root. A root that carries none of
# these is a launch mistake (the console was pointed at the wrong place),
# not an empty store — refusing it beats rendering five honest-looking
# empty pages for the wrong directory.
_STORE_MARKERS = ("records", "audit.jsonl", "decisions")


def validate_root(root: str | Path) -> Path:
    """Return the store root as a resolved path, refusing one that is not a store.

    Read-only hygiene: the console never creates what it was pointed at —
    the engine's own constructors create directories freely, so the console
    validates existence itself before touching them.
    """
    store_root = Path(root).resolve()
    if not store_root.is_dir():
        raise ConsoleError(f"store root does not exist: {store_root}")
    if not any((store_root / marker).exists() for marker in _STORE_MARKERS):
        raise ConsoleError(
            f"{store_root} does not look like an Evogine store root "
            f"(none of {', '.join(_STORE_MARKERS)} exists)"
        )
    return store_root


def validate_operator(operator: str) -> str:
    """Return the operator identity, refusing an empty one.

    Identity is supplied once at launch and recorded verbatim on every
    action the console may later take (and shown on every page), so a blank
    identity is refused up front rather than laundered into a write path.
    """
    identity = operator.strip()
    if not identity:
        raise ConsoleError(
            "operator identity is required: --operator 'Name <email>'"
        )
    return identity


def resolve_bind_host(host: str) -> str:
    """Return the concrete loopback address to bind; refuse anything else.

    Every address ``host`` resolves to must be loopback — a name resolving
    to a mix of loopback and public addresses is refused too, because the
    bind would take the public one. The returned value is the resolved
    address itself (``localhost`` binds ``127.0.0.1``, not a name), so
    what is validated is exactly what uvicorn hands to ``socket.bind``.
    """
    try:
        infos = socket.getaddrinfo(host.strip(), None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ConsoleError(f"cannot resolve bind host {host!r}: {exc}") from exc
    if not infos:
        raise ConsoleError(f"bind host {host!r} resolves to no addresses")
    addresses = {cast(str, sockaddr[0]) for *_, sockaddr in infos}
    exposed = sorted(
        address for address in addresses if not ipaddress.ip_address(address).is_loopback
    )
    if exposed:
        raise ConsoleError(
            f"refusing non-loopback bind: {host!r} resolves to "
            f"{', '.join(exposed)}; the console binds 127.0.0.1 only "
            "(loopback-only by design, no auth stack)"
        )
    # Deterministic pick: IPv4 sorts before IPv6 lexically, and every resolved
    # address is loopback by the check above.
    return min(addresses)
