"""Swarm-board and learning-loop tests: fixture runs, exact money, honest states.

The board pages consume the orchestrator's typed event feed, so these tests
build real run directories through the orchestrator's own public write APIs
(:class:`EventLog` appends, :class:`CheckpointStore` markers,
``mark_complete``) and then exercise the views and pages through the FastAPI
TestClient — fully offline. The spend-vs-ceiling acceptance is exact
``Decimal`` values rendered verbatim; the honesty acceptance is that missing
feeds render an explicit empty state, refused runs render as first-class
states, and a corrupt feed refuses the page visibly instead of fabricating
a board.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from evogine.bench.rundirs import RunDirectory, open_run
from evogine.engine.store import Engine
from evogine.gates.telemetry import DecisionLog, DecisionStage, PromotionDecision
from evogine.swarm.orchestrator import (
    CheckpointStore,
    CompleteEvent,
    EventLog,
    ItemResult,
    RefusalEvent,
    SpawnEvent,
    SpendEvent,
    StatusEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from evogine.engine.records import LearningRecord

# The console is optional transport behind the [console] extra: the core CI
# jobs never install it. The sibling console suites skip via module-level
# ``importorskip`` with a per-file E402 entry in pyproject; this suite adds
# no pyproject entry, so the same skip happens through an indented import
# guard instead — identical semantics, zero configuration change.
try:
    from fastapi.testclient import TestClient

    from evogine.console.app import create_app
    from evogine.console.swarm import (
        build_learning_loop,
        build_swarm_board,
        resolve_swarm_runs_root,
    )
except ImportError:  # pragma: no cover - core CI without the [console] extra
    pytest.skip("the [console] extra is not installed", allow_module_level=True)

OPERATOR = "nihanth"

CEILING = Decimal("25.00")


def _spend(
    worker_id: int, item_id: str, call_index: int, cost: str, spent: str
) -> SpendEvent:
    """One metered call at the fixture's ceiling; money is a string Decimal."""
    return SpendEvent(
        worker_id=worker_id,
        item_id=item_id,
        call_index=call_index,
        ok=True,
        input_tokens=1_000,
        output_tokens=100,
        cost_usd=Decimal(cost),
        spent_usd=Decimal(spent),
        ceiling_usd=CEILING,
    )


def _complete_run(runs_root: Path, run_id: str) -> RunDirectory:
    """A frozen two-worker run: spawn/status/spend/complete events + markers."""
    run = RunDirectory(runs_root, run_id)
    run.directory.mkdir(parents=True)
    log = EventLog(run)
    log.append(SpawnEvent(worker_id=0, pool_size=2))
    log.append(SpawnEvent(worker_id=1, pool_size=2))
    log.append(StatusEvent(worker_id=0, item_id="m1", status="started"))
    log.append(_spend(0, "m1", 1, "0.0134", "0.0134"))
    log.append(_spend(0, "m1", 2, "0.0120", "0.0254"))
    log.append(StatusEvent(worker_id=0, item_id="m1", status="completed"))
    log.append(StatusEvent(worker_id=1, item_id="m2", status="started"))
    log.append(SpendEvent(
        worker_id=1,
        item_id="m2",
        call_index=1,
        ok=True,
        input_tokens=800,
        output_tokens=120,
        cost_usd=Decimal("0.0041"),
        spent_usd=Decimal("0.0295"),
        ceiling_usd=CEILING,
    ))
    log.append(StatusEvent(worker_id=1, item_id="m2", status="completed"))
    log.append(
        CompleteEvent(
            items_total=2,
            items_completed=2,
            resumed_items=0,
            spent_usd=Decimal("0.0295"),
            ceiling_usd=CEILING,
        )
    )
    store = CheckpointStore(run)
    store.mark_completed(
        ItemResult(
            item_id="m1",
            modules=["evogine.swarm.roles"],
            outcome="verified",
            worker_id=0,
            calls=2,
            spend_usd=Decimal("0.0254"),
        )
    )
    store.mark_completed(
        ItemResult(
            item_id="m2",
            modules=["evogine.swarm.patches"],
            outcome="learning_proposed",
            worker_id=1,
            calls=1,
            spend_usd=Decimal("0.0041"),
            learning_id="rec-20260919T030000-abcd1234",
        )
    )
    run.mark_complete()
    return run


def _ceiling_refused_run(runs_root: Path, run_id: str) -> RunDirectory:
    """A run the spend ceiling refused mid-flight: the refusal is the state."""
    run = RunDirectory(runs_root, run_id)
    run.directory.mkdir(parents=True)
    log = EventLog(run)
    log.append(SpawnEvent(worker_id=0, pool_size=1))
    log.append(StatusEvent(worker_id=0, item_id="m1", status="started"))
    log.append(SpendEvent(
        worker_id=0,
        item_id="m1",
        call_index=1,
        ok=True,
        input_tokens=9_000,
        output_tokens=900,
        cost_usd=Decimal("0.98"),
        spent_usd=Decimal("0.98"),
        ceiling_usd=Decimal("1.00"),
    ))
    log.append(StatusEvent(worker_id=0, item_id="m1", status="completed"))
    log.append(StatusEvent(worker_id=0, item_id="m2", status="started"))
    log.append(
        RefusalEvent(
            reason="spend_ceiling",
            detail="spend ceiling refused: $0.98 spent + $0.05 next-call "
            "reservation projects past the $1.00 ceiling",
            spent_usd=Decimal("0.98"),
            projected_usd=Decimal("0.05"),
            ceiling_usd=Decimal("1.00"),
        )
    )
    CheckpointStore(run).mark_completed(
        ItemResult(
            item_id="m1",
            modules=["evogine.swarm.roles"],
            outcome="verified",
            worker_id=0,
            calls=1,
            spend_usd=Decimal("0.98"),
        )
    )
    return run


def _key_refused_run(runs_root: Path, run_id: str) -> RunDirectory:
    """A run refused before any call: no spend, no projection."""
    run = RunDirectory(runs_root, run_id)
    run.directory.mkdir(parents=True)
    log = EventLog(run)
    log.append(
        RefusalEvent(
            reason="missing_api_key",
            detail="environment variable 'ANTHROPIC_API_KEY' is not set — "
            "refusing to run without provider credentials",
            spent_usd=Decimal(0),
            projected_usd=None,
            ceiling_usd=CEILING,
        )
    )
    return run


def _resumed_run(runs_root: Path, run_id: str) -> RunDirectory:
    """A crashed-then-resumed run: one skipped item, one resumed-item count."""
    run = RunDirectory(runs_root, run_id)
    run.directory.mkdir(parents=True)
    first = EventLog(run)
    first.append(SpawnEvent(worker_id=0, pool_size=1))
    first.append(StatusEvent(worker_id=0, item_id="m1", status="started"))
    first.append(_spend(0, "m1", 1, "0.0134", "0.0134"))
    first.append(StatusEvent(worker_id=0, item_id="m1", status="completed"))
    CheckpointStore(run).mark_completed(
        ItemResult(
            item_id="m1",
            modules=["evogine.swarm.roles"],
            outcome="verified",
            worker_id=0,
            calls=1,
            spend_usd=Decimal("0.0134"),
        )
    )
    # The process died here; the resumed writer re-anchors the same feed.
    second = EventLog(run)
    second.append(
        StatusEvent(
            worker_id=None,
            item_id="m1",
            status="skipped",
            detail="resumed from checkpoint marker; not re-executed",
        )
    )
    second.append(StatusEvent(worker_id=0, item_id="m2", status="started"))
    second.append(_spend(0, "m2", 1, "0.0041", "0.0175"))
    second.append(StatusEvent(worker_id=0, item_id="m2", status="completed"))
    second.append(
        CompleteEvent(
            items_total=2,
            items_completed=2,
            resumed_items=1,
            spent_usd=Decimal("0.0175"),
            ceiling_usd=CEILING,
        )
    )
    CheckpointStore(run).mark_completed(
        ItemResult(
            item_id="m2",
            modules=["evogine.swarm.patches"],
            outcome="verified",
            worker_id=0,
            calls=1,
            spend_usd=Decimal("0.0041"),
        )
    )
    run.mark_complete()
    return run


def _corrupt_run(runs_root: Path, run_id: str) -> RunDirectory:
    """A run whose feed is corrupt mid-file: a valid line, then garbage.

    The garbage line has no later parsed line and the file ends with its
    newline, so :func:`read_events` refuses — this is corruption, not a
    torn tail, and the board must never render past it.
    """
    run = RunDirectory(runs_root, run_id)
    run.directory.mkdir(parents=True)
    feed = run.directory / "events.jsonl"
    good = SpawnEvent(worker_id=0, pool_size=1).model_dump_json()
    feed.write_text(f"{good}\n{{corrupt\n", encoding="utf-8")
    return run


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def store(root: Path) -> Engine:
    return Engine(root)


@pytest.fixture
def client(root: Path, runs: Path) -> TestClient:
    """The console over the fixture store, reading the fixture runs root."""
    return TestClient(create_app(root, OPERATOR, swarm_runs_root=str(runs)))


@pytest.fixture
def runs(root: Path) -> Path:
    """The derived swarm runs root with the fixture run directories."""
    runs_root = resolve_swarm_runs_root(root)
    runs_root.mkdir(parents=True)
    _complete_run(runs_root, "20260919T030000-abcd1234")
    _ceiling_refused_run(runs_root, "20260919T030100-abcd1234")
    _key_refused_run(runs_root, "20260919T030200-abcd1234")
    _resumed_run(runs_root, "20260919T030300-abcd1234")
    _corrupt_run(runs_root, "20260919T030400-abcd1234")
    open_run(runs_root)  # a run killed before its first event append
    return runs_root


def test_resolve_runs_root_derives_default_and_override(root: Path) -> None:
    assert resolve_swarm_runs_root(root) == root / "runs" / "swarm"
    assert resolve_swarm_runs_root(root, "/tmp/elsewhere") == Path("/tmp/elsewhere")


def test_board_rows_render_exact_state(root: Path, runs: Path) -> None:
    """The complete run: exact Decimal money, worker sums, markers, counts."""
    view = build_swarm_board(root, runs)
    rows = {row.run_id: row for row in view.runs}
    assert len(rows) == 6  # five fixtures + the run killed before its first event

    complete = rows["20260919T030000-abcd1234"]
    assert complete.complete is True
    assert complete.spent_usd == Decimal("0.0295")
    assert complete.ceiling_usd == Decimal("25.00")
    # 0.0295 / 25.00 = 0.118% — exact Decimal math, quantized for display.
    assert complete.spend_percent_display == "0.1%"
    assert complete.spend_percent_clamped == 0
    assert complete.items_total == 2 and complete.items_completed == 2
    assert complete.checkpoint_markers == 2
    assert complete.item_counts == {"completed": 2}
    assert [(w.calls, w.spent_usd) for w in complete.workers] == [
        (2, Decimal("0.0254")),
        (1, Decimal("0.0041")),
    ]
    assert complete.refusal is None


def test_board_renders_refusals_as_states(root: Path, runs: Path) -> None:
    """Refused runs are first-class rows with exact numbers — never errors."""
    view = build_swarm_board(root, runs)
    rows = {row.run_id: row for row in view.runs}

    ceiling = rows["20260919T030100-abcd1234"]
    assert ceiling.refusal is not None
    assert ceiling.refusal.reason == "spend_ceiling"
    assert ceiling.refusal.spent_usd == Decimal("0.98")
    assert ceiling.refusal.projected_usd == Decimal("0.05")
    assert ceiling.refusal.ceiling_usd == Decimal("1.00")
    assert ceiling.spent_usd == Decimal("0.98")
    assert ceiling.ceiling_usd == Decimal("1.00")
    assert ceiling.spend_percent_display == "98.0%"
    assert ceiling.spend_percent_clamped == 98
    assert ceiling.complete is False

    missing_key = rows["20260919T030200-abcd1234"]
    assert missing_key.refusal is not None
    assert missing_key.refusal.reason == "missing_api_key"
    assert missing_key.refusal.projected_usd is None
    assert missing_key.spent_usd == Decimal(0)


def test_board_renders_resume_state(root: Path, runs: Path) -> None:
    """The resumed run shows its checkpoint/resume state, not a fresh board."""
    view = build_swarm_board(root, runs)
    rows = {row.run_id: row for row in view.runs}
    resumed = rows["20260919T030300-abcd1234"]
    assert resumed.complete is True
    assert resumed.resumed_items == 1
    assert resumed.item_counts == {"completed": 1, "skipped": 1}
    assert resumed.checkpoint_markers == 2


def test_board_renders_missing_feed_as_empty_state(root: Path, runs: Path) -> None:
    """A run directory with no feed is an explicit empty state, never zeros."""
    view = build_swarm_board(root, runs)
    empty = [row for row in view.runs if row.feed_state == "empty"]
    assert len(empty) == 1
    assert empty[0].events == 0
    assert empty[0].feed_note is not None
    assert "events.jsonl" in empty[0].feed_note
    assert empty[0].spent_usd is None
    assert empty[0].complete is False


def test_corrupt_feed_renders_as_row_refusal(root: Path, runs: Path) -> None:
    """A corrupt feed is a visible corrupt row — and the board still renders."""
    view = build_swarm_board(root, runs)
    rows = {row.run_id: row for row in view.runs}
    corrupt = rows["20260919T030400-abcd1234"]
    assert corrupt.feed_state == "corrupt"
    assert corrupt.feed_note is not None
    assert "events.jsonl" in corrupt.feed_note  # the defect names its file
    # One poisoned run must not hide the others.
    assert rows["20260919T030000-abcd1234"].feed_state == "ok"
    assert len(rows) == 6


def test_board_without_runs_root_is_explicit(root: Path) -> None:
    """No runs directory at all: runs_present is False — nothing fabricated."""
    view = build_swarm_board(root)
    assert view.runs_present is False
    assert view.runs == []


def test_swarm_page_renders_board_state(client: TestClient, runs: Path) -> None:
    """The /swarm page renders the fixture runs with exact money in HTML."""
    response = client.get("/swarm")
    assert response.status_code == 200
    text = response.text
    assert "20260919T030000-abcd1234" in text
    assert "$0.0295 of $25.00" in text  # exact Decimal rendering
    assert "spend_ceiling" in text  # refusal as a visible state
    assert "no events yet" in text  # missing feed as an honest empty state


def test_swarm_page_marks_corrupt_feed_visibly(root: Path, runs: Path) -> None:
    """The /swarm page marks the corrupt feed and keeps rendering the rest."""
    client = TestClient(create_app(root, OPERATOR, swarm_runs_root=str(runs)))
    response = client.get("/swarm")
    assert response.status_code == 200  # visible refusal state, not a blind 500
    assert "corrupt feed" in response.text
    assert "20260919T030400-abcd1234" in response.text
    assert "$0.0295 of $25.00" in response.text  # healthy runs still render


def test_loop_page_renders_funnel_health_promotions(
    client: TestClient,
    store: Engine,
    root: Path,
    record_factory: Callable[..., LearningRecord],
) -> None:
    """The /loop page shows the funnel engine's stages, health, and telemetry."""
    record = record_factory("rec-loop-1")
    stored = store.propose(record)
    store.queue_replay(stored.id, ["replay-1"])
    DecisionLog(root).record(
        PromotionDecision(
            proposal_id=stored.id,
            learning_id=stored.id,
            promoted=True,
            stage=DecisionStage.APPROVAL,
            reason="proposed: 3 binary flips",
            pass_id="pass-20260919T030303-cccccc01",
            review_latency_seconds=42.5,
            decided_at=datetime(2026, 9, 19, 3, 1, tzinfo=UTC),
        )
    )
    DecisionLog(root).record(
        PromotionDecision(
            proposal_id="rec-loop-2",
            learning_id="rec-loop-2",
            promoted=False,
            stage=DecisionStage.REPLAY,
            reason="no_flip: 0/3 replays changed verdict",
            decided_at=datetime(2026, 9, 19, 3, 0, tzinfo=UTC),
        )
    )

    response = client.get("/loop")
    assert response.status_code == 200
    text = response.text
    assert "promoted records" in text  # store health panel
    assert "pass-20260919T030303-cccccc01" in text  # pass identity
    assert "42.5s" in text  # review latency
    assert text.index("rec-loop-1") < text.index("rec-loop-2")  # newest first


def test_loop_view_empty_store_is_honest(root: Path) -> None:
    """An existing-but-empty store renders zero counts and an explicit
    promotion absence (the funnel engine's own empty-store semantics)."""
    root.mkdir()
    view = build_learning_loop(root)
    assert [row.count for row in view.funnel.stage_rows] == [0, 0, 0, 0, 0, 0, 0, 0]
    assert {row.label: row.value for row in view.health} == {
        "promoted records": 0,
        "candidate records": 0,
        "duplicate pairs": 0,
        "audit events": 0,
        "pending approvals": 0,
    }
    assert view.promotions == []


def test_loop_page_promotion_absence_is_explicit(root: Path) -> None:
    """The /loop page says there is no promotion history, never a blank table."""
    root.mkdir()
    client = TestClient(create_app(root, OPERATOR))
    response = client.get("/loop")
    assert response.status_code == 200
    assert "No promotion decisions recorded" in response.text


def test_swarm_and_loop_links_in_nav(client: TestClient) -> None:
    """Both pages are reachable from every page's nav."""
    for path in ("/swarm", "/loop"):
        text = client.get(path).text
        assert '<a href="/swarm">Swarm</a>' in text
        assert '<a href="/loop">Loop</a>' in text
