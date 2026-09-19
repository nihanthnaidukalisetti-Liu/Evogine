"""End-to-end tests for the ``evogine swarm run`` CLI (offline).

The CLI chains existing pieces — decomposer, orchestrator, LearningSink,
promotion pass — so these tests exercise the real chain over fixture
repos: decompose → run → proposals land in the sink → the funnel
telemetry reports them; resume (including a killed subprocess) never
duplicates spend; every listed refusal fails closed and stays visible;
and the orchestrator's pool bounds hold through the CLI path for any
module count. Fake model clients follow the repo's fake-client idiom; the
one subprocess test talks to a localhost stub of the Anthropic Messages
API so the killed-process story is a real kill of a real interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from conftest import temp_store_root
from hypothesis import given, settings
from hypothesis import strategies as st
from orchestrator_helpers import BUILDER_REPLY, FakeModelClient

from evogine import cli as cli_module
from evogine.bench.rundirs import RunDirectory, open_run
from evogine.cli import SwarmSpendReport, _latest_incomplete_run, _swarm_spend_report, main
from evogine.gates.funnel import build_funnel_report
from evogine.swarm.decomposition import decompose
from evogine.swarm.orchestrator import (
    CheckpointStore,
    CompleteEvent,
    EventLog,
    ItemResult,
    SpawnEvent,
    SpendEvent,
    StatusEvent,
    plan_work,
    read_events,
)

USAGE = (1_000, 500)  # per fake call: $0.0015 exactly at $1.00/MTok
CALL_COST = Decimal("0.0015")
# The reservation for such a call is (1000 max_input + 1024 max_tokens)/1e6
# = $0.002024, so with prices at $1.00/MTok a $0.004 ceiling fits exactly
# two calls and refuses the third (same arithmetic as the orchestrator tests).
TIGHT_CEILING = "0.004"
PRICES = ("--input-price-per-mtok-usd", "1.00", "--output-price-per-mtok-usd", "1.00")
KNOBS = ("--max-call-input-tokens", "100000")


def _write_config(
    root: Path,
    store_root: Path,
    *,
    min_workers: int = 1,
    max_workers: int = 2,
    ceiling: str = "25.00",
) -> Path:
    """The unified config for a swarm run: one named profile, exact ceiling."""
    path = root / "evogine.yaml"
    path.write_text(
        "default_profile: default\n"
        "profiles:\n"
        "  default:\n"
        "    store:\n"
        f"      learning_root: {store_root}\n"
        "    provider:\n"
        "      model: test-model\n"
        "    workers:\n"
        f"      min_workers: {min_workers}\n"
        f"      max_workers: {max_workers}\n"
        "    budget:\n"
        f"      max_run_cost_usd: '{ceiling}'\n",
        encoding="utf-8",
    )
    return path


def _fixture_repo(root: Path, count: int) -> Path:
    """A small independent Python tree — one module per planned work item."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / "src").mkdir(exist_ok=True)
        (root / "src" / f"m{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
    return root


def _swarm_args(repo: Path, config_path: Path, *extra: str) -> list[str]:
    return [
        "swarm",
        "run",
        "--repo",
        str(repo),
        "--config",
        str(config_path),
        *PRICES,
        *KNOBS,
        *extra,
    ]


def _run_dir(store_root: Path) -> RunDirectory:
    """The one run directory under a fresh store's default runs root."""
    runs_root = store_root / "runs" / "swarm"
    run_dirs = [child for child in runs_root.iterdir() if child.is_dir()]
    assert len(run_dirs) == 1, f"expected one run dir, found {len(run_dirs)}"
    return RunDirectory(runs_root, run_dirs[0].name)


class _SingleThreadServer(HTTPServer):
    """One thread total — the repo's stub-server idiom for CI thread limits."""

    request_queue_size = 64


class _FakeMessagesServer:
    """A localhost stub of the Anthropic Messages API for subprocess tests.

    Every POST gets the same successful builder-shaped reply, so a real
    ``AnthropicClient`` inside a real interpreter drives the whole swarm
    chain offline; ``delay_s`` paces calls so a mid-run kill is catchable.
    """

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", "0")))
                if outer._delay_s:
                    time.sleep(outer._delay_s)
                body = json.dumps(
                    {
                        "id": "msg_stub_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "stub-model",
                        "content": [{"type": "text", "text": BUILDER_REPLY}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 3},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self._server = _SingleThreadServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def test_swarm_run_end_to_end_lands_learnings_and_funnel_reports_them(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """One failing item → one learning → queued → decided → funnel-reported."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 3)
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store)
    # The builder prompt names the item's module, so ``fail_on=["m2"]`` fails
    # exactly one item's builder (independent modules never cross-reference);
    # the reflector's failure context quotes the item id, and the helper's
    # fail_on only trips builder calls — the learning survives to be proposed.
    client = FakeModelClient(fail_on=["m2"])
    monkeypatch.setattr(cli_module, "_build_swarm_client", lambda **kwargs: client)

    code = main(_swarm_args(repo, config_path, "--repeats", "4"))

    assert code == 0
    out = capsys.readouterr().out
    assert "swarm run " in out and ": completed" in out
    # The failed item's learning landed through the sink (reflectors mark
    # swarm origin), and the CLI queued it for replay — the pass consumed
    # it: an empty promoted pool means no verdict flip, so the replay stage
    # declines and the record's lifecycle ends honestly at `dropped`.
    record_files = sorted((store / "records").glob("*.json"))
    assert len(record_files) == 1
    record = json.loads(record_files[0].read_text())
    assert record["provenance"]["origin"] == "swarm-reflection"
    assert record["status"] == "dropped"
    # The funnel telemetry reports the swarm-run decision in the exact
    # machine-checked vocabulary.
    report = build_funnel_report(store)
    assert report.funnel.candidates == 1
    assert report.funnel.promoted == 0
    assert report.funnel.gate_errors == 0
    assert report.reasons_by_stage["replay"]["no_flip"] == 1
    # And the funnel CLI renders the same rows.
    assert main(["decisions", "funnel", "--root", str(store)]) == 0
    assert "no_flip" in capsys.readouterr().out


def test_swarm_run_refuses_missing_repo(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store)

    code = main(_swarm_args(tmp_path / "no-such-repo", config_path))

    assert code == 1
    assert "not a directory" in capsys.readouterr().err
    assert not store.exists()  # fail-closed: nothing written anywhere


def test_swarm_run_refuses_non_python_tree_and_records_it(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = tmp_path / "repo"
    repo.mkdir()  # empty directory — nothing to decompose
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store)

    code = main(_swarm_args(repo, config_path))

    assert code == 1
    assert "decomposition refused" in capsys.readouterr().err
    # The refusal is visible in the store's telemetry, never swallowed.
    report = build_funnel_report(store)
    assert report.funnel.gate_errors == 1
    assert report.reasons_by_stage["gate_error"]["gate_error"] == 1


def test_swarm_run_refuses_config_without_budget_ceiling(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 1)
    config_path = tmp_path / "evogine.yaml"
    config_path.write_text(
        "default_profile: default\n"
        "profiles:\n"
        "  default:\n"
        "    store:\n"
        f"      learning_root: {tmp_path / 'store'}\n"
        "    provider:\n"
        "      model: test-model\n"
        "    budget:\n"
        "      max_run_cost_usd: '0'\n",  # no usable ceiling
        encoding="utf-8",
    )

    code = main(_swarm_args(repo, config_path))

    assert code == 1
    assert "max_run_cost_usd" in capsys.readouterr().err


def test_swarm_run_refuses_a_missing_config_file(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 1)

    code = main(_swarm_args(repo, tmp_path / "absent.yaml"))

    assert code == 1
    assert capsys.readouterr().err  # the refusal names the config problem


def test_swarm_run_refuses_unreachable_store(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 1)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory\n", encoding="utf-8")
    config_path = _write_config(tmp_path, blocker / "store")

    code = main(_swarm_args(repo, config_path))

    assert code == 1
    assert "unreachable" in capsys.readouterr().err


def test_swarm_run_requires_the_full_spend_expectation(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 1)
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store)

    code = main(
        [
            "swarm",
            "run",
            "--repo",
            str(repo),
            "--config",
            str(config_path),
            *PRICES,  # --max-call-input-tokens deliberately absent
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "--max-call-input-tokens" in err
    assert not store.exists()  # refused before any store or run work


def test_swarm_run_spend_ceiling_refusal_is_visible_in_funnel(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 3)
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store, ceiling=TIGHT_CEILING)

    code = main(_swarm_args(repo, config_path))

    assert code == 1
    # Run-level refusals print the run summary (stdout) — the feed shows the
    # typed refusal event below; the detail names the spend ceiling.
    assert "ceiling" in capsys.readouterr().out
    run = _run_dir(store)
    assert not run.is_complete()  # a refused run stays resumable
    refusal = [event for event in read_events(run) if event.type == "refusal"]
    assert len(refusal) == 1
    assert refusal[0].reason == "spend_ceiling"
    # The run-level refusal flows into the existing funnel telemetry.
    report = build_funnel_report(store)
    assert report.funnel.gate_errors == 1


def test_swarm_run_resume_of_completed_run_is_a_visible_no_op(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 2)
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store)
    monkeypatch.setattr(cli_module, "_build_swarm_client", lambda **kwargs: FakeModelClient())

    assert main(_swarm_args(repo, config_path)) == 0
    out = capsys.readouterr().out
    run_id = out.split("swarm run ", 1)[1].split(":", 1)[0]
    run = RunDirectory(store / "runs" / "swarm", run_id)
    before = read_events(run)

    code = main(_swarm_args(repo, config_path, "--resume", run_id))

    assert code == 0
    again = capsys.readouterr().out
    assert "already complete" in again
    assert "prior result" in again
    assert "2/2" in again  # the prior result's item count
    assert read_events(run) == before  # immutable: not one new event


def test_swarm_run_auto_resumes_newest_incomplete_run_without_duplicate_spend(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 3)
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store)
    # All builders succeed offline; the prior attempt's one item is replayed
    # from its marker, never re-executed.
    monkeypatch.setattr(cli_module, "_build_swarm_client", lambda **kwargs: FakeModelClient())
    # A prior attempt got one item done before dying: a checkpoint marker
    # (with its spend) and a spend event, but no completion marker.
    plan = plan_work(decompose(repo))
    first = plan.items[0]
    runs_root = store / "runs" / "swarm"
    runs_root.mkdir(parents=True)
    run = open_run(runs_root)
    CheckpointStore(run).mark_completed(
        ItemResult(
            item_id=first.item_id,
            modules=list(first.modules),
            outcome="unverified",
            worker_id=0,
            calls=1,
            spend_usd=CALL_COST,
        )
    )
    EventLog(run).append(
        SpendEvent(
            item_id=first.item_id,
            worker_id=0,
            call_index=1,
            ok=True,
            input_tokens=USAGE[0],
            output_tokens=USAGE[1],
            cost_usd=CALL_COST,
            spent_usd=CALL_COST,
            ceiling_usd=Decimal("25.00"),
        )
    )

    code = main(_swarm_args(repo, config_path))

    assert code == 0
    out = capsys.readouterr().out
    assert "resuming incomplete run" in out
    resumed = _run_dir(store)
    assert resumed.run_id == run.run_id
    events = read_events(resumed)
    assert any(event.type == "complete" for event in events)
    # The marked item was skipped, not re-executed: exactly the one seeded
    # spend event exists for it — the resume never re-spends a marked item —
    # and the ledger resumes from the marker's spend.
    spends_for_first = [
        event for event in events if event.type == "spend" and event.item_id == first.item_id
    ]
    assert len(spends_for_first) == 1
    final_ledger = [event for event in events if event.type == "spend"][-1]
    assert final_ledger.spent_usd == CALL_COST * 3  # marker + two fresh items


def _wait_for_markers(runs_root: Path, count: int, *, timeout_s: float) -> str:
    """Poll for a run directory holding ``count`` checkpoint markers; return its id."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if runs_root.is_dir():
            for child in sorted(runs_root.iterdir()):
                if (
                    child.is_dir()
                    and len(list((child / "artifacts").glob("checkpoint-*"))) >= count
                ):
                    return child.name
        time.sleep(0.02)
    raise AssertionError(f"no run reached {count} markers within {timeout_s}s")


def test_swarm_run_killed_subprocess_resumes_without_duplicate_spend(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    repo = _fixture_repo(tmp_path / "repo", 6)
    store = tmp_path / "store"
    config_path = _write_config(tmp_path, store, max_workers=2)
    server = _FakeMessagesServer(delay_s=0.5)  # pace calls so the kill lands mid-run
    try:
        env = {
            **os.environ,
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_BASE_URL": server.url,
        }
        cmd = [sys.executable, "-c", "from evogine.cli import main; raise SystemExit(main())"]
        args = _swarm_args(repo, config_path)

        runs_root = store / "runs" / "swarm"
        proc = subprocess.Popen(cmd + args, env=env, stdout=subprocess.DEVNULL)
        try:
            killed_run_id = _wait_for_markers(runs_root, 2, timeout_s=60)
        finally:
            proc.kill()  # kill semantics: no cleanup, no flush, no marker
            proc.wait(timeout=30)

        assert not RunDirectory(runs_root, killed_run_id).is_complete()
        resumed = subprocess.run(
            cmd + args, env=env, capture_output=True, text=True, timeout=300, check=False
        )
        assert resumed.returncode == 0, resumed.stderr

        run = RunDirectory(runs_root, killed_run_id)
        events = read_events(run)
        assert any(event.type == "complete" for event in events)  # one contiguous feed
        skipped = {
            event.item_id
            for event in events
            if isinstance(event, StatusEvent) and event.status == "skipped"
        }
        assert len(skipped) >= 2  # everything marked before the kill
        # Skipped items never spend again — resume replays markers, not work.
        # (Their pre-kill spend events from the first attempt legitimately
        # remain in the feed; the check is stream-wise: after the resume's
        # first skip, no metered call may name a marked item.)
        seen_skip = False
        for event in events:
            if isinstance(event, StatusEvent) and event.status == "skipped":
                seen_skip = True
            elif seen_skip and isinstance(event, SpendEvent):
                assert event.item_id not in skipped
        # The second invocation auto-resumed the interrupted run.
        assert "resuming incomplete run" in resumed.stdout
    finally:
        server.close()


def test_swarm_spend_report_reconciles_from_the_persisted_feed(tmp_path: Path) -> None:
    run = open_run(tmp_path)
    log = EventLog(run)
    log.append(SpawnEvent(worker_id=0, pool_size=1))
    log.append(
        SpendEvent(
            item_id="m0",
            worker_id=0,
            call_index=1,
            ok=True,
            input_tokens=USAGE[0],
            output_tokens=USAGE[1],
            cost_usd=CALL_COST,
            spent_usd=CALL_COST,
            ceiling_usd=Decimal("25.00"),
        )
    )
    log.append(
        SpendEvent(
            item_id="m1",
            worker_id=0,
            call_index=1,
            ok=False,
            cost_usd=Decimal(0),
            spent_usd=CALL_COST,  # a failed call settles at zero cost
            ceiling_usd=Decimal("25.00"),
        )
    )

    report = _swarm_spend_report(run, ceiling_usd=Decimal("25.00"))

    assert isinstance(report, SwarmSpendReport)
    assert report.calls == 2
    assert report.calls_ok == 1
    assert report.calls_failed == 1
    assert report.calls_without_usage == 0
    assert report.input_tokens == USAGE[0]
    assert report.output_tokens == USAGE[1]
    assert report.cost_usd == CALL_COST
    assert report.spent_usd == CALL_COST


def test_latest_incomplete_run_picks_the_newest_incomplete(tmp_path: Path) -> None:
    assert _latest_incomplete_run(tmp_path / "absent") is None
    (tmp_path / "definitely-not-a-run").mkdir()
    assert _latest_incomplete_run(tmp_path) is None  # foreign directories ignored

    older = open_run(tmp_path, "20260919T010101-aaaaaa01")
    CheckpointStore(older).mark_completed(
        ItemResult(item_id="m0", modules=["m0"], outcome="unverified", worker_id=0)
    )  # a marker is not a completion — still resumable
    newer = open_run(tmp_path, "20260919T020202-bbbbbbbb02")
    EventLog(newer).append(
        CompleteEvent(
            items_total=0,
            items_completed=0,
            resumed_items=0,
            spent_usd=Decimal(0),
            ceiling_usd=Decimal("25.00"),
        )
    )
    newer.mark_complete()

    assert _latest_incomplete_run(tmp_path) == "20260919T010101-aaaaaa01"


def _isolated_patch(client: FakeModelClient) -> Callable[[], None]:
    """Point the CLI's provider seam at ``client``; returns a restore callable."""
    original = cli_module._build_swarm_client
    cli_module._build_swarm_client = lambda **kwargs: client
    return lambda: setattr(cli_module, "_build_swarm_client", original)


# The CLI run is process-heavy per example (git commits in the store), so
# the suite pins a modest example budget instead of the profile's.
@given(module_count=st.integers(min_value=1, max_value=14))
@settings(max_examples=12, deadline=None)
def test_pool_bounds_hold_through_the_cli_for_any_module_count(module_count: int) -> None:
    """Property: any module count, the CLI's pool stays inside [min, max]."""
    key = "ANTHROPIC_API_KEY"
    original_key = os.environ.get(key)
    os.environ[key] = "test-key"
    store_root = temp_store_root("swarm-cli-prop-")
    repo_root = Path(tempfile.mkdtemp(prefix="swarm-cli-repo-"))
    restore = _isolated_patch(FakeModelClient())
    try:
        _fixture_repo(repo_root, module_count)
        config_path = _write_config(repo_root, store_root, max_workers=3)
        code = main(_swarm_args(repo_root, config_path))
        assert code == 0

        run = _run_dir(store_root)
        events = read_events(run)
        spawns = [event for event in events if isinstance(event, SpawnEvent)]
        assert spawns  # the pool always spawns within its bounds
        pool_sizes = {event.pool_size for event in spawns}
        assert len(pool_sizes) == 1  # one pool decision per run
        pool_size = pool_sizes.pop()
        assert 1 <= pool_size <= 3  # the config's [min_workers, max_workers]
        assert len(spawns) == pool_size  # one spawn event per worker, once
        for event in events:
            if isinstance(event, SpendEvent):
                assert event.worker_id < pool_size
        completed = [
            event
            for event in events
            if isinstance(event, StatusEvent) and event.status == "completed"
        ]
        assert len(completed) == module_count  # every item settles exactly once
    finally:
        restore()
        if original_key is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_key
        shutil.rmtree(repo_root, ignore_errors=True)
        shutil.rmtree(store_root, ignore_errors=True)
