"""The console launcher: validate everything, then bind loopback and serve.

``evogine console --root <root> --operator "Nihanth nihanth@example.eg"`` (wired in
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
from pathlib import Path

import uvicorn

from evogine.console.app import create_app
from evogine.console.launch import (
    ConsoleError,
    load_sync_endpoints,
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
    swarm_runs_root: str | None = None,
    imports_dir: str | None = None,
    sync_config: str | None = None,
) -> int:
    """Validate the launch and run the console until the operator interrupts it.

    Returns 0 after a clean shutdown; refusals raise :class:`ConsoleError`
    before the server starts, so the process exit is the operator's first
    feedback, not a half-open socket. An explicit ``swarm_runs_root`` or
    ``imports_dir`` must exist — a typo'd override would otherwise render a
    confident empty panel for the wrong directory. ``sync_config`` is a
    config file whose default profile's sync endpoints the stores-and-sync
    selector reads as read-only remote sources; a config that refuses to
    load refuses the launch.
    """
    store_root = validate_root(root)
    identity = validate_operator(operator)
    if not 0 < port <= 65535:
        raise ConsoleError(f"port must be in 1-65535, got {port}")
    if swarm_runs_root is not None and not Path(swarm_runs_root).is_dir():
        raise ConsoleError(f"swarm runs root does not exist: {swarm_runs_root}")
    if imports_dir is not None and not Path(imports_dir).is_dir():
        raise ConsoleError(f"import bundles directory does not exist: {imports_dir}")
    sync_endpoints = load_sync_endpoints(sync_config)
    bind_host = resolve_bind_host(host)
    app = create_app(
        store_root,
        identity,
        swarm_runs_root=swarm_runs_root,
        imports_dir=imports_dir,
        sync_endpoints=sync_endpoints,
    )
    uvicorn.run(app, host=bind_host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    """The ``python -m evogine.console.server`` entry point."""
    parser = argparse.ArgumentParser(
        prog="evogine console",
        description="Localhost operator console over one store's telemetry.",
    )
    parser.add_argument(
        "--root", required=True, help="engine store root to read (required; git-safety)"
    )
    parser.add_argument(
        "--operator",
        required=True,
        help='operator identity recorded on decisions, e.g. "Nihanth nihanth@example.eg"',
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host; loopback only, anything else is refused (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default: {DEFAULT_PORT}")
    parser.add_argument(
        "--swarm-runs-root",
        default=None,
        help="swarm run-directory root the board reads (default: <root>/runs/swarm)",
    )
    parser.add_argument(
        "--imports-dir",
        default=None,
        help="directory the knowledge-injection panel reads export bundles from "
        "(default: <root>/imports)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config file whose default profile's sync endpoints the stores-and-sync "
        "selector reads as read-only remote sources (default: no remotes)",
    )
    args = parser.parse_args(argv)
    try:
        return serve(
            args.root,
            args.operator,
            host=args.host,
            port=args.port,
            swarm_runs_root=args.swarm_runs_root,
            imports_dir=args.imports_dir,
            sync_config=args.config,
        )
    except ConsoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation entry point
    raise SystemExit(main())
