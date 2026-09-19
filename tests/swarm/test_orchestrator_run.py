"""End-to-end orchestrator runs: resume, refusal, bounded spawning, sink-only.

Offline — every run uses scripted fake clients and a recording sink. The
crash test runs a real subprocess that dies mid-run (``os._exit``) so the
resume path is exercised against actual on-disk checkpoint state, not a
rehearsal of it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from orchestrator_helpers import (
    FakeModelClient,
    FakeSink,
    make_plan,
    make_settings,
)

from evogine.bench.rundirs import RunDirectory, open_run
from evogine.config import load_config
from evogine.swarm.orchestrator import (
    CheckpointStore,
    RunOutcome,
    RunSettings,
    SwarmOrchestrator,
    read_events,
)

SRC = Path(__file__).resolve().parents[2] / "src"

# usage per fake call: 1k in + 500 out at $1.00/MTok = $0.0015 exactly.
# The reservation for such a call is (1000 max_input + 1024 max_tokens)/1e6
# = $0.002024, so with prices at $1.00/MTok a ceiling of $0.004 fits
# exactly two calls and refuses the third.
CALL_COST = Decimal("0.0015")
TIGHT_CEILING = "0.004"


def _orchestrator(
    client: FakeModelClient,
    sink: FakeSink,
    settings: RunSettings | None = None,
) -> SwarmOrchestrator:
    return SwarmOrchestrator(
        settings=settings or make_settings(),
        client=client,
        sink=sink,
        env={"EVOGINE_TEST_API_KEY": "test-key"},
    )


def test_happy_run_completes_with_checkpoints_and_events(tmp_path: Path) -> None:
    plan = make_plan(4)
    client = FakeModelClient()
    sink = FakeSink()

    outcome = _orchestrator(client, sink).run(plan, tmp_path)

    assert outcome.outcome == "completed"
    assert outcome.items_succeeded == 4
    assert outcome.spent_usd == CALL_COST * 4
    assert outcome.workers_spawned >= 1

    # The run is complete and immutable — reopen via the plain handle.
    run = RunDirectory(tmp_path, outcome.run_id)
    markers = sorted(path.name for path in (run.directory / "artifacts").glob("checkpoint-*"))
    assert len(markers) == 4
    events = read_events(run)
    assert events[0].type == "spawn"
    assert events[-1].type == "complete"
    assert all(event.run_id == outcome.run_id for event in events)


def test_run_without_verification_records_honest_unverified(tmp_path: Path) -> None:
    outcome = _orchestrator(FakeModelClient(), FakeSink()).run(make_plan(2), tmp_path)
    assert all(item.outcome == "unverified" for item in outcome.items)


def test_missing_api_key_refuses_before_any_call(tmp_path: Path) -> None:
    plan = make_plan(3)
    client = FakeModelClient()
    sink = FakeSink()
    orchestrator = SwarmOrchestrator(
        settings=make_settings(),
        client=client,
        sink=sink,
        env={},  # the configured key variable is absent
    )

    outcome = orchestrator.run(plan, tmp_path)

    assert outcome.outcome == "refused"
    assert "EVOGINE_TEST_API_KEY" in outcome.detail
    assert client.requests == []  # fail-closed: no model call is attempted
    assert sink.calls == 0
    assert outcome.spent_usd == 0
    events = read_events(open_run(tmp_path, outcome.run_id))
    assert any(event.type == "refusal" for event in events)


def test_missing_api_key_refusal_is_resumable(tmp_path: Path) -> None:
    plan = make_plan(1)
    orchestrator = SwarmOrchestrator(
        settings=make_settings(), client=FakeModelClient(), sink=FakeSink(), env={}
    )
    refused = orchestrator.run(plan, tmp_path)

    resumed = _orchestrator(FakeModelClient(), FakeSink()).run(
        plan, tmp_path, resume_run_id=refused.run_id
    )
    assert resumed.outcome == "completed"


def test_spend_ceiling_refuses_fail_closed_mid_run(tmp_path: Path) -> None:
    tight = make_settings(min_workers=1, max_workers=1, ceiling_usd=TIGHT_CEILING)
    plan = make_plan(3)
    client = FakeModelClient()
    sink = FakeSink()

    outcome = _orchestrator(client, sink, tight).run(plan, tmp_path)

    assert outcome.outcome == "refused"
    assert outcome.items_succeeded == 2
    assert outcome.items_failed == 1  # the refused item is a visible failure
    assert outcome.spent_usd == CALL_COST * 2
    assert len(client.requests) == 2  # the refused call never reached the client

    run = open_run(tmp_path, outcome.run_id)
    assert not run.is_complete()  # a refusal stays resumable
    refusals = [event for event in read_events(run) if event.type == "refusal"]
    assert len(refusals) == 1
    assert refusals[0].reason == "spend_ceiling"
    assert refusals[0].spent_usd == CALL_COST * 2
    assert refusals[0].ceiling_usd == Decimal("0.004")


def test_refused_run_resumes_under_a_larger_ceiling_without_duplicate_spend(
    tmp_path: Path,
) -> None:
    plan = make_plan(3)
    tight = make_settings(min_workers=1, max_workers=1, ceiling_usd=TIGHT_CEILING)
    refused = _orchestrator(FakeModelClient(), FakeSink(), tight).run(plan, tmp_path)

    roomy = make_settings(min_workers=1, max_workers=1, ceiling_usd="10.00")
    client = FakeModelClient()
    resumed = _orchestrator(client, FakeSink(), roomy).run(
        plan, tmp_path, resume_run_id=refused.run_id
    )

    assert resumed.outcome == "completed"
    assert resumed.resumed_items == 2
    assert resumed.items_succeeded == 3  # 2 resumed + 1 newly executed
    # No duplicated spend: only the one unfinished item called the model.
    assert len(client.requests) == 1
    assert resumed.spent_usd == refused.spent_usd + CALL_COST


_CRASH_SCRIPT = """\
import os
import sys
from decimal import Decimal
from pathlib import Path

from evogine.bench.rundirs import open_run
from evogine.swarm.orchestrator import SwarmOrchestrator, WorkItem, WorkPlan
from evogine.swarm.orchestrator.runner import RunSettings
from evogine.swarm.orchestrator.spend import SpendPolicy


class CrashClient:
    \"\"\"Completes two calls, then kills the process mid-run.\"\"\"

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if len(self.requests) >= 3:
            os._exit(67)  # kill semantics: no cleanup, no marker, no flush
        return type("R", (), {"text": "diff", "input_tokens": 1000,
                              "output_tokens": 500})()


class Sink:
    def __init__(self) -> None:
        self.proposed = []

    def propose(self, draft):
        self.proposed.append(draft)
        return draft


settings = RunSettings(
    min_workers=1, max_workers=1, api_key_env="K",
    spend=SpendPolicy(
        ceiling_usd=Decimal("10.00"),
        input_price_per_mtok_usd=Decimal("1.00"),
        output_price_per_mtok_usd=Decimal("1.00"),
        max_call_input_tokens=1_000_000,
    ),
)
plan = WorkPlan(items=[
    WorkItem(item_id=f"m{i}", module=f"evogine.m{i}", kind="code",
             modules=[f"evogine.m{i}"], criticality=0.5, review_tier="standard",
             dependencies=[], loc=10)
    for i in range(4)
])
root = Path(sys.argv[2])
run = open_run(root)  # fresh run directory for the crashed attempt
print(run.run_id, flush=True)  # the parent reads this before we die
orchestrator = SwarmOrchestrator(settings=settings, client=CrashClient(),
                                 sink=Sink(), env={"K": "k"})
orchestrator.run(plan, root, resume_run_id=run.run_id)
print("UNREACHED", flush=True)  # os._exit must fire first
"""


def test_killed_run_resumes_from_last_checkpoint_without_duplicate_spend(
    tmp_path: Path,
) -> None:
    plan = make_plan(4)

    crashed = subprocess.run(
        [sys.executable, "-c", _CRASH_SCRIPT, "unused", str(tmp_path)],
        env={
            **os.environ,
            "PYTHONPATH": f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # the exit code is the test subject
    )
    assert crashed.returncode == 67, crashed.stderr
    assert "UNREACHED" not in crashed.stdout
    run_id = crashed.stdout.splitlines()[0]

    run = open_run(tmp_path, run_id)
    assert not run.is_complete()  # killed mid-run, never frozen

    # The killed run completed exactly two items before dying.
    done = CheckpointStore(run).completed()
    assert set(done) == {"m0", "m1"}
    assert sum(done[item].spend_usd for item in done) == CALL_COST * 2

    # Resume: items 0-1 are skipped untouched; only 2-3 execute.
    client = FakeModelClient()
    resumed = _orchestrator(client, FakeSink()).run(plan, tmp_path, resume_run_id=run_id)

    assert resumed.outcome == "completed"
    assert resumed.resumed_items == 2
    assert resumed.items_succeeded == 4  # 2 resumed + 2 newly executed
    assert len(client.requests) == 2
    prompted = " ".join(request.user for request in client.requests)
    assert "evogine.m0" not in prompted and "evogine.m1" not in prompted
    assert "evogine.m2" in prompted and "evogine.m3" in prompted

    # One contiguous event stream across both processes.
    all_events = read_events(run)
    assert [event.seq for event in all_events] == list(range(1, len(all_events) + 1))
    assert any(event.type == "complete" for event in all_events)
    assert any(event.type == "status" and event.status == "skipped" for event in all_events)
    # Money never double counts: spend resumes from the checkpoint sum.
    assert resumed.spent_usd == CALL_COST * 4


def test_events_are_valid_jsonl_with_string_money(tmp_path: Path) -> None:
    outcome = _orchestrator(FakeModelClient(), FakeSink()).run(make_plan(2), tmp_path)
    run = RunDirectory(tmp_path, outcome.run_id)
    lines = (run.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()

    parsed = [json.loads(line) for line in lines]
    assert parsed
    for event in parsed:
        for key, value in event.items():
            if key.endswith("_usd"):
                assert isinstance(value, str)


def test_settings_from_config_reads_bounds_and_ceiling(tmp_path: Path) -> None:
    config_path = tmp_path / "evogine.yaml"
    config_path.write_text(
        "default_profile: default\n"
        "profiles:\n"
        "  default:\n"
        "    store:\n"
        "      learning_root: .evogine/store\n"
        "    provider:\n"
        "      model: test-model\n"
        "    workers:\n"
        "      min_workers: 2\n"
        "      max_workers: 6\n"
        "    budget:\n"
        "      max_run_cost_usd: '12.50'\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    settings = RunSettings.from_config(
        config,
        input_price_per_mtok_usd=Decimal("1.00"),
        output_price_per_mtok_usd=Decimal("1.00"),
        max_call_input_tokens=100_000,
    )
    assert settings.min_workers == 2
    assert settings.max_workers == 6
    assert settings.spend.ceiling_usd == Decimal("12.50")
    assert settings.api_key_env == "ANTHROPIC_API_KEY"


def test_learning_proposals_flow_only_through_the_sink(tmp_path: Path) -> None:
    # One item's builder calls fail permanently (bounded retries exhausted);
    # the reflector then proposes a learning through the sink — the only
    # store path — and the item records learning_proposed, not failed.
    client = FakeModelClient(fail_on=["evogine.m1"])
    sink = FakeSink()
    outcome = _orchestrator(client, sink).run(make_plan(3), tmp_path)

    assert outcome.outcome == "completed"  # an item failure is not a run failure
    assert outcome.items_succeeded == 2
    assert outcome.items_failed == 0
    assert sink.calls == 1
    proposed = next(item for item in outcome.items if item.outcome == "learning_proposed")
    assert proposed.item_id == "m1"
    assert proposed.learning_id is not None
    # Only the reflector call is billable: the failed builder call settles
    # at zero (record_failure) but stays visible as an ok=False spend event.
    assert proposed.calls == 2
    assert proposed.spend_usd == CALL_COST


def test_orchestrator_never_imports_the_store_directly() -> None:
    """Source-level guard: the orchestrator path writes nothing but the sink."""
    package = SRC / "evogine" / "swarm" / "orchestrator"
    forbidden = (
        "evogine.engine.store",
        "from evogine.engine import store",
        "LearningStore",
        ".queue_replay(",
        ".promote(",
        ".demote(",
        "StoreError",
    )
    violations = [
        (path.name, needle)
        for path in sorted(package.glob("*.py"))
        for needle in forbidden
        if needle in path.read_text(encoding="utf-8")
    ]
    assert violations == [], f"direct store usage found: {violations}"
    # The LearningRecord TYPE is allowed (annotations); the store module is not.
    assert "evogine.engine.records" in "".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )


def test_no_orchestrator_module_uses_sqlite_or_network() -> None:
    package = SRC / "evogine" / "swarm" / "orchestrator"
    forbidden = ("import socket", "urllib", "requests", "sqlite3", "httpx")
    violations = [
        (path.name, needle)
        for path in sorted(package.glob("*.py"))
        for needle in forbidden
        if needle in path.read_text(encoding="utf-8")
    ]
    assert violations == []


def test_outcome_carries_event_contract_fields(tmp_path: Path) -> None:
    outcome = _orchestrator(FakeModelClient(), FakeSink()).run(make_plan(2), tmp_path)
    assert isinstance(outcome, RunOutcome)
    assert outcome.items_total == 2
    assert outcome.ceiling_usd == Decimal("10.00")
    assert set(outcome.items[0].model_dump()) >= {
        "item_id",
        "modules",
        "outcome",
        "worker_id",
        "spend_usd",
    }
