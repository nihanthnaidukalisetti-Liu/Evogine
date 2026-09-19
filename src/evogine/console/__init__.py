"""Localhost operator console over existing store telemetry.

The console is transport, not logic: pages render typed views built from
the store's own file-backed logs (``decisions/``, ``approvals/``,
``runs/gate/``, ``runs/trajectory/``, ``audit.jsonl``) through the same
read APIs the CLI uses, and the two write routes decide one approval
proposal each through the boundary functions under the store's review
lock. It binds ``127.0.0.1`` only and ships behind the optional
``[console]`` extra, so the core install gains no web dependencies.
"""

from evogine.console.launch import ConsoleError

__all__ = ["ConsoleError"]
