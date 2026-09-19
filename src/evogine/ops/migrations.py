"""Store schema versioning: forward-only migrations, visible downgrade refusal.

A store records its schema version in ``schema.json`` at its root — the
same directory that is its own git repository, so the version stamp is
versioned content like everything else the store holds. Stores that
predate the field are *legacy*: they read as version 1 and keep working
unchanged (back-compat is absolute — versioning never strands an existing
store), and :func:`apply_migrations` upgrades them explicitly.

The contract, all enforced here:

- **Forward-only.** Migrations run strictly upward, one registered step at
  a time, each step committing its change to the store's git history. An
  explicit request to go below the store's current version refuses —
  loudly, with the versions named. A downgrade is never a silent no-op.
- **Older code refuses newer stores.** :func:`ensure_openable` (called by
  ``Engine.__init__``) refuses a store whose schema version exceeds the
  running build's :data:`CURRENT_SCHEMA_VERSION` — reading a schema you
  do not understand is how silent corruption happens; refusing is the
  visible alternative.
- **Registry gaps refuse.** Applying migrations walks the registry step
  by step; a missing step (from version V with no registered migration)
  is a bug and raises rather than skipping ahead.

The registry ships with one real step — version 1 → 2, "adopt schema
versioning" — which stamps ``schema.json`` into the store's history.
Future format changes register their own steps; the machinery (sequential
application, audited commits, refusal states) is exercised by tests that
run synthetic registries end to end.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel

SCHEMA_FILENAME = "schema.json"
"""The store's schema-version stamp, versioned by the store's own git repo."""

LEGACY_SCHEMA_VERSION = 1
"""Stores that predate the field: no ``schema.json`` reads as version 1."""

CURRENT_SCHEMA_VERSION = 2
"""The newest store schema this build understands (and writes fresh)."""


class SchemaError(Exception):
    """A schema-version refusal: unreadable stamp, gap, or a downgrade.

    Every refusal names the store's version and the build's — a schema
    boundary refusal is always specific about which side is which.
    """


class SchemaState(BaseModel):
    """The content of a store's ``schema.json``."""

    version: int
    migrated_from: int | None = None
    """Set only when this stamp was written by a migration step."""
    updated_at: datetime | None = None


def schema_path(root: Path | str) -> Path:
    """The store's schema stamp path."""
    return Path(root) / SCHEMA_FILENAME


def read_schema(root: Path | str) -> SchemaState | None:
    """The store's schema stamp, or ``None`` for a legacy (unversioned) store."""
    path = schema_path(root)
    if not path.exists():
        return None
    try:
        state = SchemaState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SchemaError(
            f"store schema stamp {path} is malformed and cannot be trusted: {exc}"
        ) from exc
    if state.version < LEGACY_SCHEMA_VERSION:
        raise SchemaError(
            f"store schema stamp {path} records version {state.version}, below the "
            f"known floor ({LEGACY_SCHEMA_VERSION})"
        )
    return state


def read_schema_version(root: Path | str) -> int:
    """The store's schema version; legacy stores read as version 1."""
    state = read_schema(root)
    return state.version if state is not None else LEGACY_SCHEMA_VERSION


def write_schema(root: Path | str, state: SchemaState) -> None:
    """Write the schema stamp atomically (tmp + rename, like a record save)."""
    path = schema_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git(root: Path, *args: str) -> None:
    """Run git inside the store repo, pinned exactly like the engine's own
    commits (``Engine``'s ``_git`` is the convention; this module cannot
    import it — the engine imports this module — so the pin is mirrored
    here and a drift fails the migration tests loudly)."""
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evogine Engine",
            "-c",
            "user.email=engine@evogine.local",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SchemaError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")


class Migration(NamedTuple):
    """One forward step in the registry: ``from_version`` -> ``to_version``."""

    from_version: int
    to_version: int
    name: str
    apply: Callable[[Path], None]


def _adopt_schema_versioning(root: Path) -> None:
    """Version 1 -> 2: stamp ``schema.json`` into the store's history.

    The data layout does not change — every v1 record validates as v2 —
    but the *governance* changes: from this commit on, the store carries
    an explicit schema version, so every later format change can register
    a real migration instead of guessing which stores have seen what.
    """
    write_schema(
        root,
        SchemaState(version=2, migrated_from=LEGACY_SCHEMA_VERSION, updated_at=datetime.now(UTC)),
    )
    _git(root, "add", "-A", ".")
    _git(root, "commit", "-q", "-m", "ops: adopt schema versioning (1 -> 2)")


_MIGRATIONS: dict[int, Migration] = {
    LEGACY_SCHEMA_VERSION: Migration(
        from_version=LEGACY_SCHEMA_VERSION,
        to_version=2,
        name="adopt schema versioning",
        apply=_adopt_schema_versioning,
    ),
}


def registered_migrations() -> tuple[Migration, ...]:
    """The registry, ordered by from_version (diagnostics and tests)."""
    return tuple(_MIGRATIONS[key] for key in sorted(_MIGRATIONS))


def ensure_openable(root: Path | str) -> None:
    """Refuse a store this build cannot honestly read.

    The engine's open-time gate: a store whose schema version exceeds
    :data:`CURRENT_SCHEMA_VERSION` was written by newer code, and this
    build's read paths make no promises about it. The refusal is the
    visible, forward-only answer — never a lenient parse of a schema the
    build does not know.
    """
    version = read_schema_version(root)
    if version > CURRENT_SCHEMA_VERSION:
        raise SchemaError(
            f"store schema version {version} is newer than this build supports "
            f"({CURRENT_SCHEMA_VERSION}); the store format is forward-only — "
            "upgrade evogine to open this store"
        )


def apply_migrations(
    root: Path | str,
    *,
    target: int | None = None,
    migrations: dict[int, Migration] | None = None,
) -> list[Migration]:
    """Apply forward migrations until ``target`` (default: this build's newest).

    ``migrations`` replaces the registry wholesale (tests run synthetic
    registries through the same machinery). Refusals, all visible:

    - the store's version exceeds :data:`CURRENT_SCHEMA_VERSION` — the
      same forward-only refusal :func:`ensure_openable` raises;
    - an explicit ``target`` below the store's current version — a
      downgrade attempt, named as such and refused;
    - an explicit ``target`` above this build's newest — unknown ground;
    - a registry gap — no registered step for a version on the way.

    Returns the applied steps in order (empty when the store is already
    at ``target`` — an honest no-op, reported as such, never as a
    downgrade).
    """
    registry = migrations if migrations is not None else _MIGRATIONS
    root_path = Path(root)
    current = read_schema_version(root_path)
    limit = target if target is not None else CURRENT_SCHEMA_VERSION
    if current > CURRENT_SCHEMA_VERSION:
        raise SchemaError(
            f"store schema version {current} is newer than this build supports "
            f"({CURRENT_SCHEMA_VERSION}); the store format is forward-only — "
            "upgrade evogine to migrate this store"
        )
    # Unknown ground: beyond this build's newest OR beyond the registry's
    # reach (a synthetic registry may legitimately go further; the shipped
    # one never does).
    registry_max = max(
        (step.to_version for step in registry.values()), default=CURRENT_SCHEMA_VERSION
    )
    if target is not None and target > max(CURRENT_SCHEMA_VERSION, registry_max):
        raise SchemaError(
            f"target schema version {target} is beyond this build's newest "
            f"({max(CURRENT_SCHEMA_VERSION, registry_max)})"
        )
    if target is not None and target < current:
        raise SchemaError(
            f"refusing downgrade: store is at schema version {current}; "
            f"{target} is backward — the format is forward-only"
        )
    applied: list[Migration] = []
    version = current
    while version < limit:
        step = registry.get(version)
        if step is None:
            raise SchemaError(
                f"no registered migration from schema version {version} "
                f"(registry gap while reaching {limit})"
            )
        step.apply(root_path)
        applied.append(step)
        version = step.to_version
    return applied


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "SCHEMA_FILENAME",
    "Migration",
    "SchemaError",
    "SchemaState",
    "apply_migrations",
    "ensure_openable",
    "read_schema",
    "read_schema_version",
    "registered_migrations",
    "schema_path",
    "write_schema",
]
