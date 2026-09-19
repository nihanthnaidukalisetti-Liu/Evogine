"""The registry of learning stores the daemon hosts.

A hosted store is an *existing* Evogine store root — a directory that is
its own git repository (the engine's persistence model), discovered by
scanning the stores root's direct children. The daemon hosts; it never
creates or mutates the shape of a store.

Store names are the federation's public identity contract
(:func:`evogine.engine.store.normalize_store_name`): trimmed, lowercase,
locked to a filesystem-safe grammar that excludes path separators and
colons — the same validation the registry-backed routing uses, so the
daemon can never disagree with the rest of the platform about what a
store is called, and a hostile URL name can never traverse paths.

Two refusals live here so both transport and query routes share one
vocabulary: a name that is not a hosted store (404 — the URL named
something the daemon does not host) and a hosted directory that is not a
git repository (409 — the transport needs the store's repo; serving reads
alone is not enough for a fetch/push surface).
"""

from __future__ import annotations

from pathlib import Path

from evogine.engine.store import StoreError, normalize_store_name
from evogine.sync.settings import SyncError

#: What makes a directory an Evogine store root — the same marker rule the
#: console's launch validation applies, so daemon and console agree on
#: store shape.
_STORE_MARKERS = ("records", "audit.jsonl", "decisions")


class StoreRegistry:
    """Discovers the hosted stores under one stores root, read-per-request.

    Discovery is a live scan per request (a handful of stat calls for a
    deployment's store count): a store added or removed on disk is
    reflected without a daemon restart, and there is no cached registry to
    drift from the filesystem. A child directory whose name is not a
    conforming store identity (or that carries no store marker) is simply
    not a hosted store.
    """

    def __init__(self, stores_root: Path | str) -> None:
        self.stores_root = Path(stores_root)

    def names(self) -> list[str]:
        """Hosted store identities (normalized names), sorted."""
        if not self.stores_root.is_dir():
            return []
        found: list[str] = []
        for entry in self.stores_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                normalize_store_name(entry.name)
            except StoreError:
                continue  # not a conforming store identity, not hosted
            if self._is_store(entry):
                found.append(entry.name)
        return sorted(found)

    def identity(self, name: str) -> str:
        """Normalize and validate a requested store name, or refuse.

        Grammar violations (including anything traversal-shaped) are a
        visible 400 before the name ever touches a path.
        """
        try:
            return normalize_store_name(name)
        except StoreError as exc:
            raise SyncError(str(exc), http_status=400) from exc

    def resolve(self, name: str) -> Path:
        """Return the store root for ``name``, refusing anything else."""
        normalized = self.identity(name)
        root = self.stores_root / normalized
        if not (root.is_dir() and self._is_store(root)):
            hosted = ", ".join(self.names()) or "none"
            raise SyncError(
                f"no hosted store {name!r} under {self.stores_root} (hosted: {hosted})",
                http_status=404,
            )
        return root

    def resolve_git(self, name: str) -> Path:
        """Like :meth:`resolve`, but additionally requires a git repository.

        The transport serves the store's repository — a store-shaped
        directory without ``.git`` is hosted for reads only, and a fetch
        or push against it refuses visibly instead of failing inside the
        git backend.
        """
        root = self.resolve(name)
        if not (root / ".git").exists():
            raise SyncError(
                f"store {name!r} is not a git repository; the transport hosts "
                "store repositories (every Evogine store root is one)",
                http_status=409,
            )
        return root

    def _is_store(self, path: Path) -> bool:
        return any((path / marker).exists() for marker in _STORE_MARKERS)
