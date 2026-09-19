"""In-process metrics: counters over store transitions and gate outcomes.

The metrics surface is deliberately small and typed: a process-wide
:class:`MetricsRegistry` of monotonically increasing counters, a pydantic
read API (:meth:`MetricsRegistry.snapshot` / :func:`render_metrics`) the
mission-control dashboard consumes in-process, and nothing else. There are
no exporters, no sockets, and no dependencies — the registry is plain
stdlib plus pydantic, so the core install gains nothing.

Every counter name is pinned to the ``evogine_*_total`` grammar and every
label to a small vocabulary checked at increment time — metric drift fails
loudly at the call site instead of quietly reshaping the dashboard's
aggregates (the same machine-checked-vocabulary discipline as the gate's
reason templates). Counters are code constants, so a :class:`MetricError`
is always a bug: raise it, never absorb it.

The three counters the operations wave wires today:

- ``evogine_store_transitions_total`` — one per audit event the store
  appends (every lifecycle transition), labeled by event and the status
  transition it made.
- ``evogine_gate_decisions_total`` — one per verdict the memory gate
  records, labeled by stage and promoted.
- ``evogine_funnel_stage_total`` — one per funnel-stage event: a decision
  or boundary action passing candidates / replay-cleared /
  materiality-cleared / approved / promoted / gate-errors.
"""

from __future__ import annotations

import re
import threading

from pydantic import BaseModel, ConfigDict, Field

#: Counter names: the ``evogine_`` prefix, snake-case words, ``_total``
#: suffix — the Prometheus-style convention, enforced so dashboards can
#: rely on the shape.
_METRIC_NAME_PATTERN = re.compile(r"^evogine_[a-z0-9_]+_total$")

#: Label keys: lowercase snake-case identifiers.
_LABEL_KEY_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")

#: Label values: short, printable, single-line. Record ids, store names,
#: stage names, and status words all fit; prose does not (a prose label
#: means someone passed a reason string where an enum belonged).
_LABEL_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/'@+:.-]{0,63}$")

#: One audit event per store lifecycle transition (every ``Engine``
#: ``_append_audit``), labeled ``event`` / ``from_status`` / ``to_status``.
STORE_TRANSITIONS_TOTAL = "evogine_store_transitions_total"

#: One per recorded gate verdict (``MemoryGate.review``), labeled
#: ``stage`` / ``promoted``.
GATE_DECISIONS_TOTAL = "evogine_gate_decisions_total"

#: One per durable funnel-stage event (``DecisionLog.record``), labeled
#: ``stage`` — the vocabulary the proposal funnel aggregates. Human-boundary
#: outcomes (approved→promoted, rejected→dropped) additionally surface as
#: store transitions and in the ApprovalLog, never double-counted here.
FUNNEL_STAGE_TOTAL = "evogine_funnel_stage_total"

#: The label value used when a transition has no prior status (propose).
NO_STATUS = "none"


class MetricError(ValueError):
    """A counter name or label is outside the machine-checked vocabulary."""


class MetricSeries(BaseModel):
    """One counter: its name, its label set, and its in-process value."""

    model_config = ConfigDict(frozen=True)

    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    value: int = Field(ge=0)

    @property
    def key(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        """The registry key: name plus canonically-ordered labels."""
        return (self.name, tuple(sorted(self.labels.items())))


class MetricsSnapshot(BaseModel):
    """A point-in-time read of every counter, deterministically ordered.

    This is the read API the dashboard consumes: snapshot the registry,
    ship the model (it renders to JSON cleanly), or render it as text with
    :func:`render_metrics`. Snapshot order is stable — name, then label
    pairs in sorted order — so two snapshots of the same counter set diff
    cleanly.
    """

    counters: list[MetricSeries] = Field(default_factory=list)

    def counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """One counter's value; absent counters read 0 (honest zero)."""
        wanted = (name, tuple(sorted((labels or {}).items())))
        for series in self.counters:
            if series.key == wanted:
                return series.value
        return 0


class MetricsRegistry:
    """Thread-safe in-process counter registry.

    Counters only ever increase (:meth:`increment` refuses non-positive
    amounts — a counter that can go down is a gauge, and this surface has
    no gauges). The registry is process-local by design: metrics answer
    "what did this running engine do", not "what happened across machines"
    — cross-machine aggregation is the sync pillar's job, not a file
    format invented here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}

    def increment(self, name: str, labels: dict[str, str] | None = None, *, by: int = 1) -> None:
        """Add ``by`` to the counter ``name`` with ``labels``.

        Raises :class:`MetricError` when the name or any label/key is
        outside the vocabulary, or when ``by`` is not positive.
        """
        if by <= 0:
            raise MetricError(f"counters only increase; refusing increment by={by}")
        if not _METRIC_NAME_PATTERN.fullmatch(name):
            raise MetricError(
                f"metric name {name!r} must match {_METRIC_NAME_PATTERN.pattern} "
                "(evogine_ prefix, snake-case words, _total suffix)"
            )
        checked: list[tuple[str, str]] = []
        for key, value in (labels or {}).items():
            if not _LABEL_KEY_PATTERN.fullmatch(key):
                raise MetricError(f"label key {key!r} must match {_LABEL_KEY_PATTERN.pattern}")
            if not _LABEL_VALUE_PATTERN.fullmatch(value):
                raise MetricError(f"label value {value!r} for {key!r} is not a safe label")
            checked.append((key, value))
        metric_key = (name, tuple(sorted(checked)))
        with self._lock:
            self._counters[metric_key] = self._counters.get(metric_key, 0) + by

    def counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """One counter's current value; absent counters read 0."""
        metric_key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            return self._counters.get(metric_key, 0)

    def snapshot(self) -> MetricsSnapshot:
        """Every counter, deterministically ordered (name, then labels)."""
        with self._lock:
            series = [
                MetricSeries(name=name, labels=dict(labels), value=value)
                for (name, labels), value in sorted(self._counters.items())
            ]
        return MetricsSnapshot(counters=series)


_REGISTRY = MetricsRegistry()


def metrics() -> MetricsRegistry:
    """The process-wide registry the engine and gates increment."""
    return _REGISTRY


def reset_metrics() -> None:
    """Zero every counter in the process-wide registry.

    For tests: each test that asserts on counters starts from a clean
    registry. Production code never calls this — counters are monotonic.
    """
    with _REGISTRY._lock:  # test-only reset of the registry's twin surface
        _REGISTRY._counters.clear()


def render_metrics(snapshot: MetricsSnapshot) -> str:
    """Render a snapshot as stable text: ``name{label="value",...} value``.

    The text form is for humans and CLI debugging; the model itself is the
    machine surface. Counter names with no labels render bare.
    """
    lines = []
    for series in snapshot.counters:
        rendered_labels = ",".join(
            f'{key}="{value}"' for key, value in sorted(series.labels.items())
        )
        rendered = f"{series.name}{{{rendered_labels}}}" if rendered_labels else series.name
        lines.append(f"{rendered} {series.value}")
    return "\n".join(lines)


__all__ = [
    "FUNNEL_STAGE_TOTAL",
    "GATE_DECISIONS_TOTAL",
    "NO_STATUS",
    "STORE_TRANSITIONS_TOTAL",
    "MetricError",
    "MetricSeries",
    "MetricsRegistry",
    "MetricsSnapshot",
    "metrics",
    "render_metrics",
    "reset_metrics",
]
