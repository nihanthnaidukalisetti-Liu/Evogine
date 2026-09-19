"""Unit tests for the V4 federation foundation.

Covers the three pieces the memory-federation wave adds on top of the
single-store engine (docs/cross-project-memory.md, Option B):

- the qualified-id grammar (``<store-name>:<record-id>``) and store-name
  normalization, with write-time enforcement at the store path: a store
  rejects a qualified id as its own record id, rejects unqualified
  references to records it does not hold, and rejects self-qualified
  references (governance invariant 3.4);
- :class:`~evogine.engine.federation.FederationRegistry` — per-project
  store paths injected at registration, names normalized to store
  identities, duplicate identities refused;
- :class:`~evogine.engine.federation.FederationRouter` — scope-dispatched
  reads whose surface is reads only: nothing here can pool rows across
  stores.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import make_record, replay_evidence, temp_store_root

from evogine.engine.federation import (
    FederationError,
    FederationRegistry,
    FederationRouter,
)
from evogine.engine.records import LearningRecord, MergedSource
from evogine.engine.store import (
    DEFAULT_STORE_NAME,
    Engine,
    StoreError,
    normalize_store_name,
    split_qualified_id,
)


@pytest.fixture
def store_root():
    """A git-initialized store root, cleaned up after the test."""
    root = temp_store_root("evogine-federation-")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fingerprint(root: Path | str) -> tuple[int, int]:
    """(record file count, audit line count) — a refused write changes neither."""
    root = Path(root)
    records = list((root / "records").glob("*.json"))
    audit = root / "audit.jsonl"
    lines = audit.read_text(encoding="utf-8").splitlines() if audit.exists() else []
    return len(records), len(lines)


def _draft_with_reference(record_id: str, reference: str) -> LearningRecord:
    """A draft carrying ``reference`` as the merged-source of an absorbed twin."""
    draft = make_record(record_id)
    absorbed = make_record("absorbed")
    return draft.model_copy(
        update={
            "merged_sources": [
                MergedSource(
                    record_id=reference,
                    title=absorbed.title,
                    provenance=absorbed.provenance,
                    confidence=absorbed.confidence,
                    merged_at=absorbed.created,
                )
            ]
        }
    )


def _promote(engine: Engine, record_id: str, topic: str = "429 backoff") -> None:
    """Propose, queue, and promote one record through the store's own lifecycle."""
    engine.propose(make_record(record_id, topic=topic))
    engine.queue_replay(record_id, [f"run-{record_id}"])
    engine.promote(record_id, replay_evidence([f"run-{record_id}"]))


# ------------------------------------------------------- store identity


def test_normalize_store_name_trims_and_lowercases() -> None:
    assert normalize_store_name("  Alpha ") == "alpha"
    assert normalize_store_name("PROJ.2") == "proj.2"


@pytest.mark.parametrize("bad", ["", "   ", "a b", "a:b", "_lead", "-lead", ":x", "x:"])
def test_normalize_store_name_rejects_invalid(bad: str) -> None:
    with pytest.raises(StoreError, match="must match"):
        normalize_store_name(bad)


def test_engine_default_identity_is_local(store_root) -> None:
    assert Engine(store_root).store_name == DEFAULT_STORE_NAME


def test_engine_identity_is_normalized(store_root) -> None:
    assert Engine(store_root, name="  Beta ").store_name == "beta"


@pytest.mark.parametrize("bad", ["", "not a name"])
def test_engine_refuses_invalid_identity_before_touching_disk(store_root, bad: str) -> None:
    target = Path(store_root) / "never-created"
    with pytest.raises(StoreError, match="must match"):
        Engine(target, name=bad)
    assert not target.exists()


# ------------------------------------------------- qualified-id grammar


def test_split_qualified_id_splits_and_normalizes_store_part() -> None:
    assert split_qualified_id("beta:lr-1") == ("beta", "lr-1")
    assert split_qualified_id("Beta :lr-1") == ("beta", "lr-1")


@pytest.mark.parametrize("bad", ["lr-1", "", "beta:", ":lr-1", "beta:lr:1", "beta:bad id"])
def test_split_qualified_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(StoreError):
        split_qualified_id(bad)


# ------------------------------------- write-time validation at the store


def test_propose_rejects_qualified_record_id(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(make_record("lr-local-1"))
    before = _fingerprint(store_root)
    with pytest.raises(StoreError, match="is qualified"):
        engine.propose(make_record("alpha:lr-1"))
    assert _fingerprint(store_root) == before


def test_propose_rejects_unqualified_cross_scope_reference(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(make_record("lr-local-1"))
    before = _fingerprint(store_root)
    with pytest.raises(StoreError, match="unqualified reference"):
        engine.propose(_draft_with_reference("lr-new", "ghost"))
    assert _fingerprint(store_root) == before


def test_propose_allows_unqualified_reference_to_local_record(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(make_record("lr-local-1"))
    stored = engine.propose(_draft_with_reference("lr-new", "lr-local-1"))
    assert stored.merged_sources[0].record_id == "lr-local-1"


def test_propose_allows_unqualified_merged_into_local_record(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(make_record("lr-local-1"))
    draft = make_record("lr-new").model_copy(update={"merged_into": "lr-local-1"})
    assert engine.propose(draft).merged_into == "lr-local-1"


def test_propose_rejects_self_qualified_reference(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    with pytest.raises(StoreError, match="qualifies this store"):
        engine.propose(_draft_with_reference("lr-new", "alpha:lr-9"))
    # Any spelling of this store's own identity is still self-qualification.
    with pytest.raises(StoreError, match="qualifies this store"):
        engine.propose(_draft_with_reference("lr-new-2", "ALPHA:lr-9"))


def test_propose_keeps_well_formed_foreign_reference_verbatim(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(_draft_with_reference("lr-new", "beta:lr-9"))
    # Provenance travels verbatim — normalization decides identity, it never
    # rewrites a reference.
    assert engine.get("lr-new").merged_sources[0].record_id == "beta:lr-9"


@pytest.mark.parametrize("bad", ["beta:", ":lr-1", "beta:lr:1"])
def test_propose_rejects_malformed_qualified_reference(store_root, bad: str) -> None:
    engine = Engine(store_root, name="alpha")
    with pytest.raises(StoreError):
        engine.propose(_draft_with_reference("lr-new", bad))


def test_default_identity_participates_in_reference_checks(store_root) -> None:
    engine = Engine(store_root)  # identity: the default store name
    engine.propose(make_record("lr-local-1"))
    with pytest.raises(StoreError, match="qualifies this store"):
        engine.propose(_draft_with_reference("lr-new", f"{DEFAULT_STORE_NAME}:lr-local-1"))


def test_store_read_refuses_qualified_id(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(make_record("lr-local-1"))
    with pytest.raises(StoreError, match="outside this store"):
        engine.get("beta:lr-local-1")


def test_store_lifecycle_refuses_qualified_id(store_root) -> None:
    engine = Engine(store_root, name="alpha")
    engine.propose(make_record("lr-local-1"))
    with pytest.raises(StoreError, match="outside this store"):
        engine.demote("beta:lr-local-1", "cross-scope drop attempt")


# ------------------------------------------------------------------ registry


def test_registry_injects_store_path_and_normalizes_identity(tmp_path) -> None:
    registry = FederationRegistry()
    root = tmp_path / "stores" / "alpha"
    engine = registry.register("Alpha", root)
    assert engine.root == root  # the injected path, verbatim
    assert engine.store_name == "alpha"  # normalized identity
    assert registry.names == ["alpha"]
    assert "alpha" in registry
    assert "ALPHA" in registry  # membership asks the identity, not the spelling
    assert "beta" not in registry


def test_registry_refuses_duplicate_identity(tmp_path) -> None:
    registry = FederationRegistry()
    registry.register("alpha", tmp_path / "a")
    with pytest.raises(FederationError, match="already registered"):
        registry.register("ALPHA", tmp_path / "b")  # same identity, any spelling


def test_registry_engines_are_returned_by_identity(tmp_path) -> None:
    registry = FederationRegistry()
    engine = registry.register("alpha", tmp_path / "a")
    assert registry.store("alpha") is engine
    assert registry.store("ALPHA") is engine  # dispatch normalizes too


def test_registry_store_fails_closed_for_unknown_scope(tmp_path) -> None:
    registry = FederationRegistry()
    registry.register("alpha", tmp_path / "a")
    with pytest.raises(FederationError, match="unknown federation scope"):
        registry.store("ghost")


def test_registry_membership_is_a_question_not_an_error() -> None:
    registry = FederationRegistry()
    assert "" not in registry
    assert "not a name!" not in registry
    assert 42 not in registry


# --------------------------------------------------------------------- router


def _federation(tmp_path, names: tuple[str, ...]):
    registry = FederationRegistry()
    engines = {name: registry.register(name, tmp_path / name) for name in names}
    return registry, FederationRouter(registry), engines


def test_router_retrieve_dispatches_to_the_named_scope(tmp_path) -> None:
    _, router, engines = _federation(tmp_path, ("alpha", "beta"))
    _promote(engines["alpha"], "lr-a1", topic="429 backoff")
    _promote(engines["beta"], "lr-b1", topic="deadlock contention")
    assert [record.id for record in router.retrieve("alpha", "429 backoff")] == ["lr-a1"]
    assert [record.id for record in router.retrieve("beta", "deadlock contention")] == ["lr-b1"]


def test_router_retrieve_agrees_with_the_engine_seam(tmp_path) -> None:
    _, router, engines = _federation(tmp_path, ("alpha", "beta"))
    _promote(engines["alpha"], "lr-a1", topic="429 backoff")
    context = "429 backoff"
    assert router.retrieve("alpha", context) == engines["alpha"].retrieve(context)


def test_router_retrieve_unknown_scope_fails_closed(tmp_path) -> None:
    _, router, _ = _federation(tmp_path, ("alpha",))
    with pytest.raises(FederationError, match="unknown federation scope"):
        router.retrieve("ghost", "429 backoff")


def test_router_get_and_resolve_dispatch_by_identity(tmp_path) -> None:
    _, router, engines = _federation(tmp_path, ("alpha", "beta"))
    engines["alpha"].propose(make_record("lr-a1"))
    assert router.get("alpha", "lr-a1").id == "lr-a1"
    assert router.resolve("alpha:lr-a1").id == "lr-a1"  # the qualifier is the dispatch


def test_router_resolve_refuses_unqualified_ids(tmp_path) -> None:
    _, router, engines = _federation(tmp_path, ("alpha",))
    engines["alpha"].propose(make_record("lr-a1"))
    with pytest.raises(StoreError, match="not a qualified id"):
        router.resolve("lr-a1")  # ambiguous across a federation


def test_router_resolve_fails_closed_for_unregistered_stores(tmp_path) -> None:
    _, router, _ = _federation(tmp_path, ("alpha",))
    with pytest.raises(FederationError, match="unregistered"):
        router.resolve("zeta:lr-1")


def test_router_surface_is_reads_only_and_scope_addressed() -> None:
    # Structural pin on the seam itself: every public router operation is a
    # read addressed to exactly one scope. A method that pooled rows across
    # stores (or wrote across stores) would have to appear here first — and
    # updating this assert must be a deliberate governance decision, not a
    # drive-by (the explicit-import flow is a separate, later task).
    public = [name for name in vars(FederationRouter) if not name.startswith("_")]
    assert public == ["retrieve", "get", "resolve"]
