"""Federated memory: per-project stores behind one scope-dispatched seam.

The V4 federation (docs/cross-project-memory.md, Option B): every project
keeps its own :class:`~evogine.engine.store.Engine` — its own root, its own
promotion-only pool, its own append-only audit trail — and this module adds
the two protocol objects that live *above* the store, never inside it:

- :class:`FederationRegistry` — the named set of per-project stores. Store
  paths are injected here, at registration, by the operator's setup code;
  the registry is the abstraction they are injected through. A shared
  config module is deliberately not part of this wave — a later task owns
  configuration, and this registry accepts whatever paths it is handed.
- :class:`FederationRouter` — the retrieval seam. Dispatch is by scope:
  every read names exactly one store and is served from that store alone,
  so retrieval stays per-store and no cross-store row can enter a local
  pool. The explicit, human-reviewed cross-store import is a separate
  later task; nothing here moves records between stores.

The four governance invariants (docs/cross-project-memory.md §3) hold
structurally, not by caller discipline: scoping is enforced where the pool
is assembled — a store serves only its own rows (§3.1); retrieval draws
only the addressed scope's own promoted pool (§3.2); each trail stays its
store's own append-only sequence, and nothing here stitches a global order
(§3.3); references across a scope boundary travel qualified
``<store-name>:<record-id>`` (§3.4) — the qualified-id grammar and the
write-time enforcement live with the store
(:mod:`evogine.engine.store`).
"""

from __future__ import annotations

from pathlib import Path

from evogine.engine.embedders import Embedder
from evogine.engine.records import LearningRecord
from evogine.engine.retrieval import DEFAULT_FLOOR
from evogine.engine.store import (
    Engine,
    StoreError,
    normalize_store_name,
    split_qualified_id,
)


class FederationError(Exception):
    """Federation routing violation: unknown scope, duplicate registration."""


class FederationRegistry:
    """The named set of per-project stores: scope name -> store root.

    Registration is the injection point: the caller hands over each
    store's root path, and the registry builds the :class:`Engine` on it —
    where those paths come from (a file, a profile, operator flags) is the
    configuration task's concern, deliberately not this module's. Names
    are store identities, normalized once here so dispatch can never
    disagree with a store about who it is. Duplicate names are refused —
    fail-closed, because silently re-pointing a scope at a different root
    would move every future read to a different pool under the same name.
    """

    def __init__(self) -> None:
        self._stores: dict[str, Engine] = {}

    def register(
        self,
        name: str,
        root: Path | str,
        *,
        embedder: Embedder | None = None,
    ) -> Engine:
        """Register the store at the injected ``root`` under ``name``.

        Returns the store's engine. The name is normalized to its
        identity form; a name already taken — by any spelling — is a
        :class:`FederationError`.
        """
        store_name = normalize_store_name(name)
        if store_name in self._stores:
            raise FederationError(f"store already registered: {store_name}")
        engine = Engine(root, embedder, name=store_name)
        self._stores[store_name] = engine
        return engine

    def store(self, name: str) -> Engine:
        """The store behind ``name`` — fail-closed for unknown scopes.

        An unknown scope is a routing error, not an empty pool: an empty
        retrieval result says "nothing relevant here", while a mistyped
        scope says "this store does not exist" — two different facts, and
        only the second is honest to report as an error.
        """
        store_name = normalize_store_name(name)
        try:
            return self._stores[store_name]
        except KeyError:
            registered = ", ".join(sorted(self._stores)) or "none"
            raise FederationError(
                f"unknown federation scope: {store_name!r} (registered: {registered})"
            ) from None

    @property
    def names(self) -> list[str]:
        """Registered scope names, sorted — deterministic for audit and tests."""
        return sorted(self._stores)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` (normalized) is a registered scope.

        Membership is a question, not an action: an invalid name is simply
        not registered, never an error.
        """
        if not isinstance(name, str):
            return False
        try:
            store_name = normalize_store_name(name)
        except StoreError:
            return False
        return store_name in self._stores


class FederationRouter:
    """The retrieval seam over a federation: dispatch by scope, per-store.

    Every read names exactly one scope and is served by that scope's own
    store. The router never pools rows across stores — no method returns
    a union of pools — so a warm run in one scope structurally cannot act
    on another scope's knowledge (invariants §3.1/§3.2). Reads are the
    whole surface: writes stay with each store's own lifecycle, and
    cross-store movement belongs to the explicit import flow (a separate
    later task). Until that flow exists, the boundary is absolute — and
    even after, only an import may move a row into a local pool.
    """

    def __init__(self, registry: FederationRegistry) -> None:
        self._registry = registry

    def retrieve(
        self,
        scope: str,
        task_context: str,
        k: int = 5,
        floor: float = DEFAULT_FLOOR,
    ) -> list[LearningRecord]:
        """Top-``k`` promoted records from ``scope``'s own pool.

        The addressed store's own hybrid retrieval, promotion-only by its
        own construction: what the gate proved in this scope, and nothing
        from any other scope's pool.
        """
        return self._registry.store(scope).retrieve(task_context, k=k, floor=floor)

    def get(self, scope: str, record_id: str) -> LearningRecord:
        """One record by id from ``scope``'s store, in any lifecycle state."""
        return self._registry.store(scope).get(record_id)

    def resolve(self, qualified_id: str) -> LearningRecord:
        """One record by qualified id ``<store-name>:<record-id>``.

        The qualifier is the dispatch: the record is read from exactly the
        store its qualifier names — never from any caller-chosen pool.
        Unqualified ids are refused (ambiguous across a federation: which
        store's ``lr-1``?), and a qualifier naming no registered store
        fails closed rather than falling back to some other scope.
        """
        store_name, record_id = split_qualified_id(qualified_id)
        if store_name not in self._registry:
            raise FederationError(
                f"qualified id {qualified_id!r} names an unregistered store: {store_name!r}"
            )
        return self._registry.store(store_name).get(record_id)
