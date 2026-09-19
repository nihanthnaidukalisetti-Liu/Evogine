"""Checkpointed runs: immutable per-item resume markers, crash-safe by design.

A run's progress lives in one marker per completed work item, written into
the run directory the V3 resume-marker way (:mod:`evogine.bench.rundirs`):
atomic writes, and a completed marker is never rewritten — a marker file's
existence *is* the fact that the item finished, so a resumed run skips it
without re-executing a single model call (no duplicated spend). The run's
``complete.marker`` still lands last, via the same machinery.

Marker file names are content-addressed (a SHA-256 prefix of the item id)
because module ids are dotted names like ``evogine.swarm.roles`` — not safe
path segments. The payload carries the real item id, so a hash collision
with a different item's marker is caught at write time and refused rather
than silently overwriting anyone's completion.
"""

from __future__ import annotations

import hashlib
import threading
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.bench.rundirs import RunDirectory
from evogine.swarm.orchestrator.errors import CheckpointError

MARKER_PREFIX = "checkpoint-"
"""Marker artifact name prefix; the suffix is the item id's hash prefix."""

_HASH_WIDTH = 16


class ItemResult(BaseModel):
    """The durable outcome of one work item — exactly what a marker holds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    modules: list[str] = Field(min_length=1)
    outcome: Literal["verified", "unverified", "learning_proposed", "failed"]
    worker_id: int = Field(ge=0)
    calls: int = Field(default=0, ge=0)
    spend_usd: Decimal = Field(default=Decimal(0), ge=0)
    learning_id: str | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _learning_id_matches_outcome(self) -> ItemResult:
        if self.outcome == "learning_proposed" and not self.learning_id:
            raise ValueError("outcome learning_proposed requires learning_id")
        if self.outcome != "learning_proposed" and self.learning_id is not None:
            raise ValueError("learning_id is only set on learning_proposed items")
        return self


def marker_name(item_id: str) -> str:
    """The segment-safe marker artifact name for ``item_id``."""
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:_HASH_WIDTH]
    return f"{MARKER_PREFIX}{digest}"


class CheckpointStore:
    """Per-item resume markers in one run directory; write-once, never mutated."""

    def __init__(self, run: RunDirectory) -> None:
        self._run = run
        self._lock = threading.Lock()

    def completed(self) -> dict[str, ItemResult]:
        """Every completed item's result, keyed by item id — the resume map."""
        done: dict[str, ItemResult] = {}
        for path in sorted(self._run.directory.glob(f"artifacts/{MARKER_PREFIX}*.json")):
            result = ItemResult.model_validate_json(path.read_text(encoding="utf-8"))
            if result.item_id in done:
                raise CheckpointError(
                    f"two checkpoint markers claim item {result.item_id!r} in "
                    f"run {self._run.run_id!r} — the marker set is ambiguous"
                )
            done[result.item_id] = result
        return done

    def is_completed(self, item_id: str) -> bool:
        """Whether a marker for ``item_id`` exists in this run."""
        return self._run.read_artifact(marker_name(item_id)) is not None

    def mark_completed(self, result: ItemResult) -> None:
        """Write the item's marker exactly once; a second write is refused.

        Markers are never mutated: the item is already done, and rewriting
        a completion would make the resume record a matter of who wrote
        last. The write itself is atomic (temp file + rename), so a crash
        leaves either the old marker or none — never a half marker.
        """
        name = marker_name(result.item_id)
        with self._lock:
            existing = self._run.read_artifact(name)
            if existing is not None:
                prior = ItemResult.model_validate_json(existing)
                if prior.item_id != result.item_id:
                    raise CheckpointError(
                        f"checkpoint marker hash collision in run "
                        f"{self._run.run_id!r}: {result.item_id!r} maps to a "
                        f"marker already held by {prior.item_id!r}"
                    )
                raise CheckpointError(
                    f"item {result.item_id!r} already has a checkpoint marker in "
                    f"run {self._run.run_id!r} — markers are never mutated"
                )
            self._run.write_artifact(name, result.model_dump_json() + "\n")


__all__ = ["MARKER_PREFIX", "CheckpointStore", "ItemResult", "marker_name"]
