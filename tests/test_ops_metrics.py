"""Metrics registry: vocabulary enforcement, monotonicity, deterministic reads."""

from __future__ import annotations

import threading

import pytest

from evogine.ops.metrics import (
    FUNNEL_STAGE_TOTAL,
    GATE_DECISIONS_TOTAL,
    NO_STATUS,
    STORE_TRANSITIONS_TOTAL,
    MetricError,
    MetricsRegistry,
    metrics,
    render_metrics,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    reset_metrics()


def test_counter_starts_at_zero_and_increments() -> None:
    registry = MetricsRegistry()
    assert registry.counter(STORE_TRANSITIONS_TOTAL) == 0
    registry.increment(STORE_TRANSITIONS_TOTAL, {"event": "propose"})
    assert registry.counter(STORE_TRANSITIONS_TOTAL, {"event": "propose"}) == 1
    registry.increment(STORE_TRANSITIONS_TOTAL, {"event": "propose"}, by=4)
    assert registry.counter(STORE_TRANSITIONS_TOTAL, {"event": "propose"}) == 5


def test_labels_distinguish_series() -> None:
    registry = MetricsRegistry()
    registry.increment(GATE_DECISIONS_TOTAL, {"stage": "replay", "promoted": "true"})
    registry.increment(GATE_DECISIONS_TOTAL, {"stage": "replay", "promoted": "false"})
    snapshot = registry.snapshot()
    assert snapshot.counter(GATE_DECISIONS_TOTAL, {"stage": "replay", "promoted": "true"}) == 1
    assert snapshot.counter(GATE_DECISIONS_TOTAL, {"stage": "replay", "promoted": "false"}) == 1
    assert snapshot.counter(GATE_DECISIONS_TOTAL, {"stage": "replay"}) == 0


def test_metric_name_grammar_is_enforced() -> None:
    registry = MetricsRegistry()
    for bad_name in ("store_transitions_total", "evogine_total", "evogine_X_total", "counter"):
        with pytest.raises(MetricError, match="metric name"):
            registry.increment(bad_name)


def test_label_keys_and_values_are_enforced() -> None:
    registry = MetricsRegistry()
    with pytest.raises(MetricError, match="label key"):
        registry.increment(STORE_TRANSITIONS_TOTAL, {"Event": "propose"})
    with pytest.raises(MetricError, match="label value"):
        registry.increment(STORE_TRANSITIONS_TOTAL, {"event": "has\nnewline"})
    with pytest.raises(MetricError, match="label value"):
        registry.increment(STORE_TRANSITIONS_TOTAL, {"event": "x" * 65})


def test_counters_refuse_to_decrease() -> None:
    registry = MetricsRegistry()
    with pytest.raises(MetricError, match="only increase"):
        registry.increment(STORE_TRANSITIONS_TOTAL, by=0)
    with pytest.raises(MetricError, match="only increase"):
        registry.increment(STORE_TRANSITIONS_TOTAL, by=-1)


def test_snapshot_ordering_is_deterministic() -> None:
    registry = MetricsRegistry()
    registry.increment(FUNNEL_STAGE_TOTAL, {"stage": "promoted"})
    registry.increment(STORE_TRANSITIONS_TOTAL, {"event": "promote"})
    registry.increment(FUNNEL_STAGE_TOTAL, {"stage": "approved"})
    names = [series.name for series in registry.snapshot().counters]
    assert names == sorted(names)
    again = registry.snapshot()
    assert again == registry.snapshot()


def test_render_metrics_is_stable_text() -> None:
    registry = MetricsRegistry()
    registry.increment(STORE_TRANSITIONS_TOTAL, {"event": "propose", "to_status": "ephemeral"})
    rendered = render_metrics(registry.snapshot())
    assert rendered == ('evogine_store_transitions_total{event="propose",to_status="ephemeral"} 1')


def test_no_status_label_value() -> None:
    assert NO_STATUS == "none"


def test_process_wide_registry_is_the_same_object() -> None:
    assert metrics() is metrics()


def test_parallel_increments_are_thread_safe() -> None:
    registry = MetricsRegistry()
    threads = [
        threading.Thread(
            target=lambda: [registry.increment(STORE_TRANSITIONS_TOTAL) for _ in range(100)]
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert registry.counter(STORE_TRANSITIONS_TOTAL) == 800
