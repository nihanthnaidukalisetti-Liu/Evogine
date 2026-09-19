"""Localhost operator console over existing store telemetry (read-only).

The console is transport, not logic: pages render typed views built from
the store's own file-backed logs (``decisions/``, ``approvals/``,
``runs/gate/``, ``runs/trajectory/``, ``audit.jsonl``) through the same
read APIs the CLI uses. It binds ``127.0.0.1`` only and ships behind the
optional ``[console]`` extra, so the core install gains no web
dependencies. There is no write route in the console foundation PR —
approvals stay with the CLI until the write-path change lands.
"""

from evogine.console.launch import ConsoleError

__all__ = ["ConsoleError"]
