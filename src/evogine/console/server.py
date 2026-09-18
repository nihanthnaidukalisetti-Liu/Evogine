"""The console launcher: validate everything, then bind loopback and serve.

``evogine console --root <root> --operator "Name <email>"`` (wired in
:mod:`evogine.cli`) resolves here; the CLI's import of this module is lazy
so the core install never loads FastAPI or uvicorn — they exist only behind
the optional ``[console]`` extra.

Refusals happen before any socket exists: an unknown store root, an empty
operator identity, or a bind host that leaves the loopback interface never
reaches :func:`uvicorn.run`.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from evogine.console.app import create_app
from evogine.console.launch import (
    ConsoleError,
    resolve_bind_host,
    validate_operator,
    validate_root,
)

DEFAULT_PORT = 8765


def serve(
    root: str,
    operator: str,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
) -> int:
    """Validate the launch and run the console until the operator interrupts it.

    Returns 0 after a clean shutdown; refusals raise :class:`ConsoleError`
    before the server starts, so the process exit is the operator's first
    feedback, not a half-open socket.
    """
    store_root = validate_root(root)
    identity = validate_operator(operator)
    if not 0 < port <= 65535:
        raise ConsoleError(f"port must be in 1-65535, got {port}")
    bind_host = resolve_bind_host(host)
    app = create_app(store_root, identity)
    uvicorn.run(app, host=bind_host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    """The ``python -m evogine.console.server`` entry point."""
    parser = argparse.ArgumentParser(
        prog="evogine console",
        description="Localhost operator console over one store's telemetry (read-only).",
    )
    parser.add_argument(
        "--root", required=True, help="engine store root to read (required; git-safety)"
    )
    parser.add_argument(
        "--operator",
        required=True,
        help='operator identity recorded on actions, e.g. "Nihanth <nihanth@example.eg>"',
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host; loopback only, anything else is refused (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default: {DEFAULT_PORT}")
    args = parser.parse_args(argv)
    try:
        return serve(args.root, args.operator, host=args.host, port=args.port)
    except ConsoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation entry point
    raise SystemExit(main())
