"""Schema versioning: forward-only migrations, downgrade refusal, engine gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evogine.engine.store import Engine, StoreError
from evogine.ops.migrations import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_FILENAME,
    Migration,
    SchemaError,
    SchemaState,
    apply_migrations,
    ensure_openable,
    read_schema,
    read_schema_version,
    registered_migrations,
    write_schema,
)


def _git_head(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _git_commit_count(root: Path) -> int:
    proc = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def test_fresh_store_is_born_versioned(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    assert read_schema_version(engine.root) == CURRENT_SCHEMA_VERSION
    state = read_schema(engine.root)
    assert state is not None
    assert state.version == CURRENT_SCHEMA_VERSION
    assert state.migrated_from is None  # born current, not migrated


def test_store_without_stamp_reads_as_legacy(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    (engine.root / SCHEMA_FILENAME).unlink()
    assert read_schema(engine.root) is None
    assert read_schema_version(engine.root) == LEGACY_SCHEMA_VERSION
    # Legacy stores keep working unchanged: the engine opens them as ever.
    reopened = Engine(engine.root)
    with pytest.raises(StoreError, match="unknown learning id"):
        reopened.get("missing")


def test_ensure_openable_refuses_newer_store(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    write_schema(engine.root, SchemaState(version=CURRENT_SCHEMA_VERSION + 7))
    with pytest.raises(SchemaError, match="forward-only"):
        ensure_openable(engine.root)


def test_engine_refuses_newer_store_as_store_error(tmp_path: Path) -> None:
    root = tmp_path / "store"
    Engine(root)
    write_schema(root, SchemaState(version=99))
    with pytest.raises(StoreError, match="newer than this build supports"):
        Engine(root)


def test_engine_rejects_malformed_schema_stamp(tmp_path: Path) -> None:
    root = tmp_path / "store"
    Engine(root)
    (root / SCHEMA_FILENAME).write_text("not json at all", encoding="utf-8")
    with pytest.raises(StoreError, match="malformed"):
        Engine(root)


def test_apply_migrations_stamps_legacy_store(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    (engine.root / SCHEMA_FILENAME).unlink()

    applied = apply_migrations(engine.root)
    assert len(applied) == 1
    assert applied[0].from_version == LEGACY_SCHEMA_VERSION
    assert applied[0].to_version == CURRENT_SCHEMA_VERSION
    assert read_schema_version(engine.root) == CURRENT_SCHEMA_VERSION
    state = read_schema(engine.root)
    assert state is not None
    assert state.migrated_from == LEGACY_SCHEMA_VERSION
    # The migration committed the stamp into the store's git history.
    assert _git_commit_count(engine.root) >= 1


def test_apply_migrations_refuses_downgrade(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    with pytest.raises(SchemaError, match="refusing downgrade"):
        apply_migrations(engine.root, target=LEGACY_SCHEMA_VERSION)
    # The refusal is not a silent no-op wearing success: version unchanged.
    assert read_schema_version(engine.root) == CURRENT_SCHEMA_VERSION


def test_apply_migrations_refuses_target_beyond_build(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    with pytest.raises(SchemaError, match="beyond this build"):
        apply_migrations(engine.root, target=CURRENT_SCHEMA_VERSION + 1)


def test_apply_migrations_refuses_registry_gap(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    (engine.root / SCHEMA_FILENAME).unlink()
    with pytest.raises(SchemaError, match="registry gap"):
        apply_migrations(engine.root, migrations={})


def test_apply_migrations_walks_synthetic_registry_in_order(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    (engine.root / SCHEMA_FILENAME).unlink()

    def step_two(root: Path) -> None:
        write_schema(root, SchemaState(version=3))

    synthetic = {
        1: Migration(
            1, 2, "synthetic step one", lambda root: write_schema(root, SchemaState(version=2))
        ),
        2: Migration(2, 3, "synthetic step two", step_two),
    }
    applied = apply_migrations(engine.root, target=3, migrations=synthetic)
    assert [step.name for step in applied] == ["synthetic step one", "synthetic step two"]
    assert read_schema_version(engine.root) == 3


def test_apply_migrations_is_an_honest_noop_when_current(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "store")
    assert apply_migrations(engine.root) == []


def test_shipped_registry_has_the_versioning_step() -> None:
    steps = registered_migrations()
    assert [step.from_version for step in steps] == sorted(step.from_version for step in steps)
    assert steps[-1].to_version == CURRENT_SCHEMA_VERSION


def test_migrated_store_stays_fully_functional(tmp_path: Path, record_factory) -> None:
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    (engine.root / SCHEMA_FILENAME).unlink()
    apply_migrations(engine.root)
    reopened = Engine(engine.root)
    assert reopened.get("lr-1").status == "ephemeral"
    assert _git_head(engine.root)  # history intact
