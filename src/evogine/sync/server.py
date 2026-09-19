"""The ``evogine-syncd`` entry point: bind policy first, socket second.

The daemon's launch contract, in order:

1. Validate settings (stores root exists, port sane) — refuse on stderr
   with exit 1 before any socket exists.
2. Resolve the bind host under the binding policy — loopback by default;
   a non-loopback address is refused unless ``--expose-lan`` was passed,
   and even then only READ endpoints become reachable beyond the machine
   (the per-request peer gate in :mod:`evogine.sync.app` keeps store
   mutations loopback-only regardless of any flag).
3. Serve the app with uvicorn.

The console's own ``launch`` validation remains untouched; this module
mirrors its *shape* (validated settings → uvicorn) without sharing a
config surface, per the task's boundary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from evogine.sync.app import create_app
from evogine.sync.settings import DEFAULT_HOST, DEFAULT_PORT, SyncError, SyncSettings


def serve(settings: SyncSettings) -> int:
    """Validate, resolve the bind, and run the daemon. Blocks until stop."""
    settings = settings.validated()
    bind_host = settings.resolved_bind_host()
    uvicorn.run(
        create_app(settings),
        host=bind_host,
        port=settings.port,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the ``evogine-syncd`` console script."""
    parser = argparse.ArgumentParser(
        prog="evogine-syncd",
        description=(
            "Evogine sync daemon: git-native store transport plus a read/query "
            "API over hosted learning stores."
        ),
    )
    parser.add_argument(
        "--stores-root",
        required=True,
        help="directory of hosted learning stores (each store root is its own git repository)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"bind host; loopback by default (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"listen port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--expose-lan",
        action="store_true",
        help=(
            "expose READ endpoints beyond loopback; store-mutating endpoints stay "
            "loopback-only regardless of this flag"
        ),
    )
    args = parser.parse_args(argv)
    try:
        return serve(
            SyncSettings(
                stores_root=Path(args.stores_root),
                host=args.host,
                port=args.port,
                expose_lan=args.expose_lan,
            )
        )
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
