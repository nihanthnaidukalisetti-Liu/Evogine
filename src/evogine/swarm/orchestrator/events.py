"""Run events: the orchestrator's JSONL feed, and the dashboard's contract.

Every run writes one ``events.jsonl`` at its run-directory root — one JSON
object per line, append-only. Each line is a self-describing typed event
(exact ``Decimal`` money, tz-aware UTC timestamps); the mission-control
dashboard consumes this file and nothing else about the run's interior.
The schema is the contract this module owns:

Envelope — every event carries:

- ``type``   — discriminator: ``spawn | status | spend | complete | refusal``
- ``run_id`` — the run directory id the event belongs to
- ``seq``    — 1-based, strictly increasing within one run's file
- ``ts``     — RFC 3339 UTC timestamp (always tz-aware; naive refuses)

Payloads:

- ``spawn``    — one per worker thread, emitted by the parent before the
  thread starts: ``worker_id`` (0-based), ``pool_size``.
- ``status``   — per-item transitions: ``worker_id`` (``None`` for
  run-level statements), ``item_id``, ``status`` (``started | completed |
  failed | skipped``), ``detail``. ``skipped`` marks a resumed item whose
  checkpoint marker already completed it — it is never re-executed, so it
  never re-spends. ``failed`` marks an item whose model calls failed after
  the client's bounded retries were exhausted; the item is still
  checkpointed under that outcome.
- ``spend``    — one per model call, success or failure: ``worker_id``,
  ``item_id``, ``call_index`` (1-based within the worker), ``ok``,
  ``input_tokens`` / ``output_tokens`` (``None`` when the provider
  reported no usage or the call failed), ``cost_usd`` (what the ledger
  charged this call), ``spent_usd`` (ledger total after it),
  ``ceiling_usd``. Money is exact ``Decimal``, serialized as a string —
  never a float.
- ``complete`` — run-level, written once just before ``complete.marker``
  freezes the run: ``items_total``, ``items_completed``, ``resumed_items``,
  ``spent_usd``, ``ceiling_usd``.
- ``refusal``  — run-level, fail-closed: ``reason``
  (``spend_ceiling | missing_api_key``), ``detail``, ``spent_usd``,
  ``projected_usd`` (the reservation that would have exceeded the ceiling;
  ``None`` for key refusals), ``ceiling_usd``.

Crash semantics: a kill mid-append can tear the final line. Readers skip
an unparsable *trailing* line (checkpoint markers, not the event feed, are
the resume authority) and refuse a corrupt line in the middle — that is
real corruption and is never hidden. Before appending after a torn tail,
the writer seals it with a newline, so old garbage can never merge with a
new event line.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from evogine.bench.rundirs import RunDirectory
from evogine.swarm.orchestrator.errors import EventLogError

EVENTS_FILENAME = "events.jsonl"
"""The event feed's file name inside a run directory."""


class _EventEnvelope(BaseModel):
    """Fields every run event carries; the event log assigns run_id and seq."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = ""
    seq: int = Field(default=0, ge=0)
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("ts")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware UTC")
        return value


class SpawnEvent(_EventEnvelope):
    """One worker thread, spawned by the parent within the configured bounds."""

    type: Literal["spawn"] = "spawn"
    worker_id: int = Field(ge=0)
    pool_size: int = Field(ge=1)


class StatusEvent(_EventEnvelope):
    """One work item's transition, or a run-level statement (worker_id None)."""

    type: Literal["status"] = "status"
    worker_id: int | None = None
    item_id: str
    status: Literal["started", "completed", "failed", "skipped"]
    detail: str = ""


class SpendEvent(_EventEnvelope):
    """One metered model call — the per-call ledger the ceiling reads."""

    type: Literal["spend"] = "spend"
    worker_id: int = Field(ge=0)
    item_id: str
    call_index: int = Field(ge=1)
    ok: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal = Field(ge=0)
    spent_usd: Decimal = Field(ge=0)
    ceiling_usd: Decimal = Field(gt=0)


class CompleteEvent(_EventEnvelope):
    """The run finished every planned item; written just before the freeze."""

    type: Literal["complete"] = "complete"
    items_total: int = Field(ge=0)
    items_completed: int = Field(ge=0)
    resumed_items: int = Field(ge=0)
    spent_usd: Decimal = Field(ge=0)
    ceiling_usd: Decimal = Field(gt=0)


class RefusalEvent(_EventEnvelope):
    """The run refused — fail-closed, with the numbers that forced it."""

    type: Literal["refusal"] = "refusal"
    reason: Literal["spend_ceiling", "missing_api_key"]
    detail: str
    spent_usd: Decimal = Field(ge=0)
    projected_usd: Decimal | None = None
    ceiling_usd: Decimal = Field(gt=0)


RunEvent = Annotated[
    SpawnEvent | StatusEvent | SpendEvent | CompleteEvent | RefusalEvent,
    Field(discriminator="type"),
]
"""The discriminated union every JSONL line parses back into."""

_RUN_EVENT_ADAPTER: TypeAdapter[RunEvent] = TypeAdapter(RunEvent)


class EventLog:
    """Append-only JSONL event writer for one run directory.

    Thread-safe: worker threads emit concurrently and each event's ``seq``
    is assigned under the log's lock, so sequence numbers are dense and
    strictly increasing in file order. Appends are single ``write`` calls
    of one line — a crash can tear the last line at worst, which readers
    tolerate (see :func:`read_events`).
    """

    def __init__(self, run: RunDirectory) -> None:
        self._run = run
        self._path = run.directory / EVENTS_FILENAME
        self._lock = threading.Lock()
        self._seq = len(read_events(run))
        self._seal_torn_tail()

    def append(self, event: RunEvent) -> RunEvent:
        """Stamp ``event`` with run_id/seq, append its line, return it stamped."""
        with self._lock:
            self._seq += 1
            stamped = event.model_copy(update={"run_id": self._run.run_id, "seq": self._seq})
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(stamped.model_dump_json() + "\n")
            return stamped

    def _seal_torn_tail(self) -> None:
        """Cap a torn trailing line with a newline before any new append.

        A kill mid-write leaves a line without its newline; appending after
        it would merge old garbage into the next event. Sealing strands the
        partial line (readers skip it) and keeps every future line whole.
        """
        if not self._path.exists():
            return
        size = self._path.stat().st_size
        if size == 0:
            return
        with self._path.open("rb") as probe:
            probe.seek(size - 1)
            torn = probe.read(1) != b"\n"
        if torn:
            with self._path.open("a", encoding="utf-8") as out:
                out.write("\n")


def read_events(run: RunDirectory) -> list[RunEvent]:
    """Read a run's event feed back, in file order.

    A line that fails to parse is tolerated only when the log demonstrably
    outlived it: either the file itself ends mid-line (a kill torn the
    write — the fragment is the last line and carries no newline), or a
    later line parsed successfully (a resumed writer sealed the torn tail
    and re-anchored the feed after it). Anything else raises
    :class:`EventLogError`: a complete write that no later event follows is
    corruption, and silently dropping it would falsify the record the
    dashboard reads. Skipped debris stays visible as a ``seq`` jump.
    """
    path = run.directory / EVENTS_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = raw.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    torn_tail = bool(lines) and not raw.endswith("\n")

    parsed: list[RunEvent | None] = []
    failures: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        try:
            parsed.append(_RUN_EVENT_ADAPTER.validate_json(line))
        except ValueError as exc:
            parsed.append(None)
            failures.append((index, str(exc)))

    # Walk backward: a failure is survivable debris iff a later line parsed
    # (the writer came back) or it is the torn, newline-less final line.
    survived_after = False
    for index in range(len(lines) - 1, -1, -1):
        if parsed[index] is not None:
            survived_after = True
            continue
        _index, message = next(f for f in failures if f[0] == index)
        if survived_after or (index == len(lines) - 1 and torn_tail):
            continue
        raise EventLogError(f"event line {index + 1} of {path} is corrupt: {message}")
    return [event for event in parsed if event is not None]


__all__ = [
    "EVENTS_FILENAME",
    "CompleteEvent",
    "EventLog",
    "RefusalEvent",
    "RunEvent",
    "SpawnEvent",
    "SpendEvent",
    "StatusEvent",
    "read_events",
]
