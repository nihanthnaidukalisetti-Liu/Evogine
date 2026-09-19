"""Orchestrator unit tests: planning, pool bounds, events, checkpoints, spend.

All offline and deterministic; the properties run at a modest example
budget. The end-to-end runs (fake clients, crash and resume, ceiling
refusal, sink-only) live in ``test_orchestrator_run.py``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from orchestrator_helpers import make_dag, make_plan
from pydantic import ValidationError

from evogine.bench.rundirs import RunDirectory, open_run
from evogine.swarm.clients import ModelRequest, ModelResponse
from evogine.swarm.orchestrator import (
    CheckpointError,
    CheckpointStore,
    EventLog,
    EventLogError,
    ItemResult,
    RefusalEvent,
    Scheduler,
    SpawnEvent,
    SpendEvent,
    SpendMeter,
    SpendPolicy,
    WorkItem,
    WorkPlan,
    compute_pool_size,
    item_id_for,
    plan_work,
    read_events,
)
from evogine.swarm.orchestrator.checkpoint import marker_name
from evogine.swarm.orchestrator.planning import TIGHT_FRACTION
from evogine.swarm.orchestrator.spend import SpendCeilingRefused

EXAMPLES = 25


# --- planning: partition, order, dependencies, tiers ------------------------


def test_plan_partitions_every_module_exactly_once() -> None:
    dag = make_dag(
        {
            "evogine.core": {
                "kind": "code",
                "tested_by": ["tests.test_core"],
                "criticality": 0.9,
            },
            "evogine.cli": {
                "kind": "code",
                "imports": ["evogine.core"],
                "criticality": 0.4,
            },
            "tests.test_core": {"kind": "test", "criticality": 0.2},
            "tests.orphan": {"kind": "test", "criticality": 0.1},
        }
    )
    plan = plan_work(dag)

    members = [member for item in plan.items for member in item.modules]
    assert sorted(members) == sorted(node.module for node in dag.modules)
    assert len(members) == len(set(members)), "no module may own two items"
    assert {item.module for item in plan.items if item.kind == "code"} == {
        "evogine.core",
        "evogine.cli",
    }
    core_item = next(item for item in plan.items if item.module == "evogine.core")
    assert core_item.modules == ["evogine.core", "tests.test_core"]
    orphan = next(item for item in plan.items if "tests.orphan" in item.modules)
    assert orphan.kind == "test"


def test_plan_orders_by_criticality_then_id() -> None:
    dag = make_dag(
        {
            "a": {"criticality": 0.5},
            "b": {"criticality": 0.9},
            "c": {"criticality": 0.9},
            "d": {"criticality": 0.1},
        }
    )
    plan = plan_work(dag)
    keys = [(-item.criticality, item.item_id) for item in plan.items]
    assert keys == sorted(keys)
    assert plan.items[0].module in {"b", "c"}


def test_plan_dependencies_follow_import_ownership() -> None:
    dag = make_dag(
        {
            "core": {"criticality": 0.8},
            "api": {"imports": ["core", "missing_module"], "criticality": 0.6},
            "tests.test_api": {"kind": "test", "criticality": 0.1},
        }
    )
    plan = plan_work(dag)

    api = next(item for item in plan.items if item.module == "api")
    core = next(item for item in plan.items if item.module == "core")
    assert api.dependencies == [core.item_id]
    assert core.dependencies == []
    for item in plan.items:
        assert item.item_id not in item.dependencies


def test_plan_test_modules_attach_to_first_listing_host() -> None:
    dag = make_dag(
        {
            "alpha": {"tested_by": ["tests.shared"], "criticality": 0.7},
            "beta": {"tested_by": ["tests.shared"], "criticality": 0.7},
            "tests.shared": {"kind": "test"},
        }
    )
    plan = plan_work(dag)
    owners = [item.module for item in plan.items if "tests.shared" in item.modules]
    assert owners == ["alpha"], "shared test attaches to exactly one host"


def test_tight_tier_is_top_quarter() -> None:
    criticalities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    dag = make_dag({f"m{i}": {"criticality": score} for i, score in enumerate(criticalities)})
    plan = plan_work(dag)
    assert plan.tight_count == 2  # ceil(8 * 0.25) items
    tight_modules = {item.module for item in plan.items if item.review_tier == "tight"}
    assert tight_modules == {"m0", "m1"}


def test_tight_tier_includes_ties_at_the_cut() -> None:
    dag = make_dag({f"m{i}": {"criticality": 0.5} for i in range(4)})
    plan = plan_work(dag)
    # ceil(4 * 0.25) = 1 rank, but every score ties at the cut -> all included
    assert plan.tight_count == 4


def test_plan_is_deterministic() -> None:
    dag = make_dag(
        {
            "evogine.z": {"criticality": 0.3, "imports": ["evogine.a"]},
            "evogine.a": {"criticality": 0.9},
            "t": {"kind": "test", "criticality": None},
        }
    )
    first = plan_work(dag).model_dump_json()
    second = plan_work(dag).model_dump_json()
    assert first == second


@given(module_id=st.text(min_size=1, max_size=40))
def test_item_ids_are_segment_safe_and_injective(module_id: str) -> None:
    from evogine.swarm.orchestrator.planning import _SAFE_SEGMENT

    first = item_id_for(module_id)
    assert first == item_id_for(module_id)
    assert _SAFE_SEGMENT.fullmatch(first)
    # Injectivity for unsafe names: the digest suffix must react to edits.
    if module_id != first:
        perturbed = module_id + "x"
        if not _SAFE_SEGMENT.fullmatch(perturbed):
            assert item_id_for(perturbed) != first


def test_item_id_distinguishes_lookalike_names() -> None:
    assert item_id_for("a.b") != item_id_for("a_b")


# --- pool sizing: the bounded-spawn property --------------------------------


@given(
    total=st.integers(min_value=0, max_value=400),
    min_workers=st.integers(min_value=1, max_value=8),
    max_workers=st.integers(min_value=1, max_value=32),
    tight_share=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=EXAMPLES)
def test_pool_size_never_leaves_bounds(
    total: int, min_workers: int, max_workers: int, tight_share: float
) -> None:
    if max_workers < min_workers:
        return
    size = compute_pool_size(
        total,
        min_workers=min_workers,
        max_workers=max_workers,
        tight_share=tight_share,
    )
    if total <= 0:
        assert size == 0
    else:
        assert min_workers <= size <= max_workers


@given(
    sizes=st.lists(st.integers(min_value=1, max_value=300), min_size=2, max_size=30),
    min_workers=st.integers(min_value=1, max_value=4),
    max_workers=st.integers(min_value=4, max_value=16),
)
@settings(max_examples=EXAMPLES)
def test_pool_size_grows_monotonically_with_the_plan(
    sizes: list[int], min_workers: int, max_workers: int
) -> None:
    previous = compute_pool_size(
        0, min_workers=min_workers, max_workers=max_workers, tight_share=0.0
    )
    for total in sorted(sizes):
        size = compute_pool_size(
            total, min_workers=min_workers, max_workers=max_workers, tight_share=0.0
        )
        assert size >= previous
        previous = size


def test_pool_size_scales_with_size_and_criticality() -> None:
    small = compute_pool_size(4, min_workers=1, max_workers=8, tight_share=0.0)
    large = compute_pool_size(40, min_workers=1, max_workers=8, tight_share=0.0)
    assert large > small  # scale with codebase size
    pressured = compute_pool_size(4, min_workers=1, max_workers=8, tight_share=0.9)
    assert pressured > small  # criticality concentration pushes parallelism
    assert pressured <= 8


# --- scheduler: real threads, real bounds -----------------------------------


def _verified_result(item: WorkItem, worker_id: int) -> ItemResult:
    return ItemResult(
        item_id=item.item_id,
        modules=list(item.modules),
        outcome="verified",
        worker_id=worker_id,
    )


def _scheduler(
    plan: WorkPlan,
    root: Path,
    *,
    min_workers: int,
    max_workers: int,
    work_fn: Callable[[WorkItem, int], ItemResult],
) -> Scheduler:
    """A Scheduler wired to a real run directory and event log."""
    run = open_run(root)
    return Scheduler(
        plan,
        run,
        EventLog(run),
        min_workers=min_workers,
        max_workers=max_workers,
        work_fn=work_fn,
    )


def test_scheduler_spawns_within_bounds_and_finishes_all_items(tmp_path: Path) -> None:
    plan = make_plan(17)
    scheduler = _scheduler(
        plan,
        tmp_path,
        min_workers=2,
        max_workers=5,
        work_fn=_verified_result,
    )
    results = scheduler.run(frozenset())
    assert 2 <= scheduler.spawned <= 5
    assert set(results) == {item.item_id for item in plan.items}
    assert all(r.outcome == "verified" for r in results.values())


def test_scheduler_respects_maximum_under_many_items(tmp_path: Path) -> None:
    plan = make_plan(60)
    scheduler = _scheduler(
        plan,
        tmp_path,
        min_workers=3,
        max_workers=3,
        work_fn=_verified_result,
    )
    scheduler.run(frozenset())
    assert scheduler.spawned == 3


def test_scheduler_runs_dependencies_before_dependents(tmp_path: Path) -> None:
    order: list[str] = []
    lock = threading.Lock()

    def work(item: WorkItem, worker_id: int) -> ItemResult:
        with lock:
            order.append(item.item_id)
        return _verified_result(item, worker_id)

    plan = make_plan(
        2,
        criticalities=[0.9, 0.8],
        dependencies={"m0": ["m1"]},  # m0 depends on m1
    )
    scheduler = _scheduler(
        plan,
        tmp_path,
        min_workers=2,
        max_workers=4,
        work_fn=work,
    )
    scheduler.run(frozenset())
    assert order.index("m1") < order.index("m0")


def test_scheduler_breaks_import_cycles_liveness_first(tmp_path: Path) -> None:
    plan = WorkPlan(
        items=[
            WorkItem(
                item_id=f"c{index}",
                module=f"c{index}",
                kind="code",
                modules=[f"c{index}"],
                criticality=0.5,
                review_tier="standard",
                dependencies=[f"c{(index + 1) % 3}"],
                loc=10,
            )
            for index in range(3)
        ]
    )
    scheduler = _scheduler(
        plan,
        tmp_path,
        min_workers=1,
        max_workers=2,
        work_fn=_verified_result,
    )
    results = scheduler.run(frozenset())
    assert set(results) == {"c0", "c1", "c2"}  # liveness: the cycle completed


def test_scheduler_records_item_failure_without_stopping_the_pool(
    tmp_path: Path,
) -> None:
    plan = make_plan(3)

    def flaky(item: WorkItem, worker_id: int) -> ItemResult:
        if item.item_id == "m1":
            raise RuntimeError("boom")
        return _verified_result(item, worker_id)

    scheduler = _scheduler(
        plan,
        tmp_path,
        min_workers=1,
        max_workers=2,
        work_fn=flaky,
    )
    results = scheduler.run(frozenset())
    assert results["m1"].outcome == "failed"
    assert "boom" in results["m1"].detail
    assert results["m0"].outcome == "verified"
    assert results["m2"].outcome == "verified"


def test_scheduler_run_with_empty_plan_spawns_nothing(tmp_path: Path) -> None:
    scheduler = _scheduler(
        WorkPlan(items=[]),
        tmp_path,
        min_workers=1,
        max_workers=2,
        work_fn=_verified_result,
    )
    assert scheduler.run(frozenset()) == {}
    assert scheduler.spawned == 0


# --- events: schema round-trips and the JSONL feed --------------------------

_TS = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)


def _event_samples(
    run_id: str = "r",
) -> list[SpawnEvent | SpendEvent | RefusalEvent]:
    return [
        SpawnEvent(run_id=run_id, seq=1, ts=_TS, worker_id=0, pool_size=2),
        SpendEvent(
            run_id=run_id,
            seq=2,
            ts=_TS,
            worker_id=0,
            item_id="m0",
            call_index=1,
            ok=True,
            input_tokens=1_000,
            output_tokens=500,
            cost_usd=Decimal("0.0015"),
            spent_usd=Decimal("0.0015"),
            ceiling_usd=Decimal("10.00"),
        ),
        RefusalEvent(
            run_id=run_id,
            seq=3,
            ts=_TS,
            reason="spend_ceiling",
            detail="refused",
            spent_usd=Decimal("9.99"),
            projected_usd=Decimal("0.50"),
            ceiling_usd=Decimal("10.00"),
        ),
    ]


def test_event_round_trips_are_lossless(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    log = EventLog(run)
    samples = _event_samples(run_id=run.run_id)
    for event in samples:
        log.append(event)
    events = read_events(run)
    assert [e.model_dump() for e in events] == [e.model_dump() for e in samples]
    assert events[1].cost_usd == Decimal("0.0015")
    assert events[1].ts.tzinfo is not None


def test_event_log_assigns_dense_sequence_numbers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    log = EventLog(run)
    event = _event_samples()[0]

    def append_ten() -> None:
        for _ in range(10):
            log.append(event)

    threads = [threading.Thread(target=append_ten) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = read_events(run)
    assert [e.seq for e in events] == list(range(1, 51))
    assert all(e.run_id == run.run_id for e in events)


def test_event_log_survives_a_torn_trailing_line(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    log = EventLog(run)
    log.append(_event_samples()[0])
    log.append(_event_samples()[1])

    # Simulate a kill mid-append: half of the next line is on disk.
    path = run.directory / "events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"spend","run_id":')

    assert len(read_events(run)) == 2  # the torn tail is skipped, not fatal

    # A resumed writer seals the tail and new events still parse.
    resumed_log = EventLog(RunDirectory(tmp_path, run.run_id))
    resumed_log.append(_event_samples()[2])
    events = read_events(run)
    assert len(events) == 3
    assert events[2].reason == "spend_ceiling"  # type: ignore[attr-defined]


def test_event_log_refuses_midfile_corruption(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    log = EventLog(run)
    log.append(_event_samples()[0])

    path = run.directory / "events.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json at all}\n", encoding="utf-8")

    with pytest.raises(EventLogError):
        read_events(run)


def test_event_log_refuses_naive_timestamps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime

    run = open_run(tmp_path)
    log = EventLog(run)
    with pytest.raises(ValueError, match="timezone-aware"):
        log.append(
            SpawnEvent(
                run_id="r",
                seq=0,
                ts=datetime(2026, 9, 19),  # noqa: DTZ001 — naive ts is the refusal subject
                worker_id=0,
                pool_size=1,
            )
        )


def test_event_schema_money_is_string_json_and_unknown_types_refused(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    from evogine.swarm.orchestrator.events import _RUN_EVENT_ADAPTER

    run = open_run(tmp_path)
    log = EventLog(run)
    stamped = log.append(_event_samples()[1])
    raw = json.loads(stamped.model_dump_json())
    assert isinstance(raw["cost_usd"], str), "money must serialize as a string"

    with pytest.raises(ValueError):
        _RUN_EVENT_ADAPTER.validate_json('{"type": "unknown"}')


# --- checkpoints: immutable resume markers ----------------------------------


def _result(item_id: str, **overrides: object) -> ItemResult:
    fields: dict[str, object] = {
        "item_id": item_id,
        "modules": [item_id],
        "outcome": "verified",
        "worker_id": 0,
        "spend_usd": Decimal("0.01"),
    }
    fields.update(overrides)
    return ItemResult(**fields)  # type: ignore[arg-type]


def test_checkpoint_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    store = CheckpointStore(run)
    store.mark_completed(_result("m0"))
    store.mark_completed(_result("m1", outcome="learning_proposed", learning_id="lr-1"))

    done = CheckpointStore(run).completed()
    assert set(done) == {"m0", "m1"}
    assert done["m1"].learning_id == "lr-1"
    assert done["m0"].spend_usd == Decimal("0.01")


def test_checkpoint_markers_are_never_mutated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    store = CheckpointStore(run)
    store.mark_completed(_result("m0", spend_usd=Decimal("0.10")))

    original = run.read_artifact(marker_name("m0"))
    with pytest.raises(CheckpointError, match="never mutated"):
        store.mark_completed(_result("m0", spend_usd=Decimal("9.99")))
    assert run.read_artifact(marker_name("m0")) == original


def test_checkpoint_refuses_hash_collisions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = open_run(tmp_path)
    run.write_artifact(marker_name("victim"), _result("other").model_dump_json() + "\n")
    store = CheckpointStore(run)
    with pytest.raises(CheckpointError, match="collision"):
        store.mark_completed(_result("victim"))


def test_learning_proposed_requires_learning_id() -> None:
    with pytest.raises(ValueError, match="learning_id"):
        _result("m0", outcome="learning_proposed")


# --- spend meter: the fail-closed ceiling -----------------------------------


def _policy(**overrides: object) -> SpendPolicy:
    fields: dict[str, object] = {
        "ceiling_usd": Decimal("1.00"),
        "input_price_per_mtok_usd": Decimal("1.00"),
        "output_price_per_mtok_usd": Decimal("1.00"),
        "max_call_input_tokens": 1_000_000,
    }
    fields.update(overrides)
    return SpendPolicy(**fields)  # type: ignore[arg-type]


def _request(max_tokens: int = 1_024) -> ModelRequest:
    return ModelRequest(system="s", user="u", max_tokens=max_tokens)


def _response(input_tokens: int | None, output_tokens: int | None) -> ModelResponse:
    return ModelResponse(text="ok", input_tokens=input_tokens, output_tokens=output_tokens)


def test_meter_bills_usage_and_refuses_past_the_ceiling() -> None:
    meter = SpendMeter(_policy(ceiling_usd=Decimal("2.50")))
    request = _request()
    # usage: 1M in + 0 out at $1/MTok = $1.00 per successful call
    usage = (1_000_000, 0)

    first = meter.reserve(request, worker_id=0, item_id="m0", call_index=1)
    actual = meter.record_success(
        _response(*usage), reservation=first, worker_id=0, item_id="m0", call_index=1
    )
    assert actual == Decimal("1.00")

    second = meter.reserve(request, worker_id=0, item_id="m1", call_index=1)
    meter.record_success(
        _response(*usage), reservation=second, worker_id=0, item_id="m1", call_index=1
    )
    assert meter.spent_usd == Decimal("2.00")

    with pytest.raises(SpendCeilingRefused):
        meter.reserve(request, worker_id=0, item_id="m2", call_index=1)
    assert meter.spent_usd == Decimal("2.00")


def test_meter_bills_reservation_when_usage_is_unreported() -> None:
    meter = SpendMeter(_policy(ceiling_usd=Decimal("1.00"), max_call_input_tokens=1_000))
    request = _request()
    reservation = meter.reserve(request, worker_id=0, item_id="m0", call_index=1)
    actual = meter.record_success(
        _response(None, None),
        reservation=reservation,
        worker_id=0,
        item_id="m0",
        call_index=1,
    )
    assert actual == reservation
    assert meter.spent_usd > 0


def test_meter_releases_reservation_on_failure() -> None:
    meter = SpendMeter(_policy(max_call_input_tokens=1_000))
    request = _request()
    reservation = meter.reserve(request, worker_id=0, item_id="m0", call_index=1)
    meter.record_failure(
        reservation=reservation, worker_id=0, item_id="m0", call_index=1, error="boom"
    )
    assert meter.spent_usd == 0
    # Ceiling still fits the next call: the reservation was released.
    meter.reserve(request, worker_id=0, item_id="m0", call_index=2)


def test_meter_events_carry_exact_money() -> None:
    seen: list[SpendEvent] = []
    meter = SpendMeter(_policy(max_call_input_tokens=1_000), on_event=seen.append)
    request = _request()
    reservation = meter.reserve(request, worker_id=0, item_id="m0", call_index=1)
    meter.record_success(
        _response(1_000_000, 0),
        reservation=reservation,
        worker_id=0,
        item_id="m0",
        call_index=1,
    )
    assert len(seen) == 1  # one settlement per settled call — reservations are internal
    settle = seen[0]
    assert settle.ok is True
    assert settle.cost_usd == Decimal("1.00")
    assert settle.spent_usd == Decimal("1.00")
    assert settle.ceiling_usd == Decimal("1.00")


def test_concurrent_reservations_never_exceed_the_ceiling() -> None:
    meter = SpendMeter(_policy(ceiling_usd=Decimal("0.01"), max_call_input_tokens=3_000))
    request = _request(max_tokens=1_024)
    # Each reservation ~ (3000 + 1024)/1e6 ~ $0.004024 -> at most 2 fit.
    refusals: list[Exception] = []
    lock = threading.Lock()

    def hammer(worker_id: int) -> None:
        for index in range(1, 11):
            try:
                reservation = meter.reserve(
                    request,
                    worker_id=worker_id,
                    item_id=f"m{worker_id}",
                    call_index=index,
                )
            except SpendCeilingRefused as exc:
                with lock:
                    refusals.append(exc)
                continue
            meter.record_success(
                _response(1_000, 500),
                reservation=reservation,
                worker_id=worker_id,
                item_id=f"m{worker_id}",
                call_index=index,
            )

    threads = [threading.Thread(target=hammer, args=(worker_id,)) for worker_id in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert refusals, "the ceiling must actually refuse once exhausted"
    assert meter.spent_usd <= Decimal("0.01")


def test_spend_policy_refuses_non_finite_and_negative_money() -> None:
    with pytest.raises(ValueError):
        _policy(ceiling_usd=Decimal("Infinity"))
    with pytest.raises(ValueError):
        _policy(ceiling_usd=Decimal("-1.00"))


def test_meter_refuses_negative_initial_spend() -> None:
    with pytest.raises(ValueError, match="negative"):
        SpendMeter(_policy(), initial_spent_usd=Decimal("-0.01"))


# --- plan model guards ------------------------------------------------------


def test_work_plan_is_frozen() -> None:
    plan = make_plan(2)
    with pytest.raises(ValidationError):
        plan.items = []  # type: ignore[misc]


def test_plan_rejects_bad_tight_fraction() -> None:
    dag = make_dag({"a": {"criticality": 0.5}})
    with pytest.raises(ValueError, match="tight_fraction"):
        plan_work(dag, tight_fraction=0.0)


def test_tight_fraction_constant_is_a_quarter() -> None:
    assert TIGHT_FRACTION == 0.25
