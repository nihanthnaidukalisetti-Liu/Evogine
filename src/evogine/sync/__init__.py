"""Server-side sync: the daemon that makes one store visible to every machine.

The V4 sync pillar's decision (Blueprint region `sync`): a small FastAPI
service becomes the shared spine — it hosts store repositories, serves a
read/query API for telemetry and approvals, and accepts pushes of local
lifecycle commits. Nothing here is a second store: the store format and its
invariants are unchanged, and transport is git-native because the store
already *is* a git repository (one JSON file per record, append-only JSONL
audit, one commit per transition — merge-safe since V2).

Surfaces, in their own modules:

- :mod:`evogine.sync.settings` — daemon settings and the binding policy
  (loopback by default; an explicit flag exposes reads on the LAN;
  store-mutating endpoints loopback-only regardless of any flag).
- :mod:`evogine.sync.stores` — the registry of hosted stores.
- :mod:`evogine.sync.gitbackend` — the git smart-HTTP transport
  (``git http-backend`` CGI handoff), with per-store audit order preserved
  by git itself: no invented global sequence, per the governance record's
  invariant 3.3 (``docs/cross-project-memory.md``).
- :mod:`evogine.sync.queries` — pure read models over the same typed files
  the console reads: records, decisions, approvals, funnel aggregates, run
  evidence, shadow outputs, spend.
- :mod:`evogine.sync.app` — the FastAPI factory that wires transport and
  query routes behind the binding guards.
- :mod:`evogine.sync.server` — the ``evogine-syncd`` entry point.
- :mod:`evogine.sync.merge` — the append-only audit merge driver: the
  documented resolution for the one surface two machines can both append
  to. The low-conflict merge surface is one JSON file per record (disjoint
  files merge clean) plus this driver for ``audit.jsonl``.
"""

from evogine.sync.settings import DEFAULT_HOST, DEFAULT_PORT, SyncError, SyncSettings

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "SyncError", "SyncSettings"]
