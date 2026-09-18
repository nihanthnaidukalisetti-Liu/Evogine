"""Console route, refusal, and launch tests against a fixture store.

The console is a read-only transport over telemetry the gate and CLI
already write, so these tests build a real store (propose, queue replay,
review through the memory gate, approve through the approval log) and then
exercise the pages through the FastAPI TestClient — fully offline. The
launch tests assert the loopback-only contract and the startup refusals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from evogine.cli import main as cli_main
from evogine.console import views
from evogine.console.app import create_app
from evogine.console.launch import (
    ConsoleError,
    resolve_bind_host,
    validate_operator,
    validate_root,
)
from evogine.console.server import main as console_main
from evogine.console.server import serve
from evogine.engine.store import Engine
from evogine.gates.approvals import ApprovalLog, approve_proposal
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink
from evogine.gates.trajectory import TrajectoryStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from evogine.engine.records import LearningRecord

OPERATOR = "Nihanth <nihanth@localhost>"


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=ok) for i, ok in enumerate(outcomes)
        ],
    )


def scored_run(run_id: str, arm: str, scores: list[float]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=None, score=score)
            for i, score in enumerate(scores)
        ],
    )


class StubRunner:
    """Deterministic replay runner: run id -> scripted ReplayRun."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise KeyError(f"unknown replay run: {run_id}")
        return self._runs[run_id]


def scripted_runner() -> StubRunner:
    """Five reviews, one per gate outcome the decision log distinguishes.

    prop-1 flips every case (cleared, then approved by the fixture); prop-2
    flips nothing (replay refusal); prop-3's scored delta stays under the
    threshold (materiality refusal); prop-4 regresses a case (never-worse
    refusal); prop-5 clears but is left pending for the approvals queue.
    """
    return StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
            "warm-1": binary_run("warm-1", "warm", [True, True, True]),
            "cold-2": binary_run("cold-2", "baseline", [False, False, False]),
            "warm-2": binary_run("warm-2", "warm", [False, False, False]),
            "cold-3": scored_run("cold-3", "baseline", [0.4, 0.4, 0.4]),
            "warm-3": scored_run("warm-3", "warm", [0.5, 0.4, 0.4]),
            "cold-4": binary_run("cold-4", "baseline", [True, False, False]),
            "warm-4": binary_run("warm-4", "warm", [False, True, True]),
            "cold-5": binary_run("cold-5", "baseline", [False, False, False]),
            "warm-5": binary_run("warm-5", "warm", [True, True, True]),
        }
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def store(root: Path) -> Engine:
    return Engine(root)


def review(store: Engine, root: Path, record_id: str, run_ids: list[str]) -> None:
    """Run one memory-gate review the way ``evogine run`` does."""
    MemoryGate(
        runner=scripted_runner(),
        log=DecisionLog(root),
        approvals=ApprovalLog(root),
        evidence=RunsEvidenceSink(root),
        trajectories=TrajectoryStore(root),
    ).review(store.get(record_id), run_ids)


@pytest.fixture
def console_store(
    root: Path, store: Engine, record_factory: Callable[[str], LearningRecord]
) -> Path:
    """A store with the full telemetry set: five decisions across every gate
    stage, paired evidence, sanitized trajectories, a pending proposal, an
    approved promotion, and audit history."""
    for name in ("prop-1", "prop-2", "prop-3", "prop-4", "prop-5"):
        topic = "merge conflicts" if name == "prop-2" else "flaky retry loop"
        store.queue_replay(store.propose(record_factory(name, topic=topic)).id, [
            f"cold-{name[-1]}",
            f"warm-{name[-1]}",
        ])
        review(store, root, name, [f"cold-{name[-1]}", f"warm-{name[-1]}"])

    approve_proposal(
        store, ApprovalLog(root), "appr-prop-1", decided_by="nihanth", reason="convincing"
    )
    return root


@pytest.fixture
def client(console_store: Path) -> TestClient:
    return TestClient(create_app(console_store, OPERATOR), raise_server_exceptions=False)


def write_decision_file(console_store: Path, decision_file: str, payload: str) -> None:
    """Drop a hand-crafted decision file into the store, bypassing the log.

    This is the adversarial case the read path must survive: a file on disk
    that no gate wrote, with content no ``check_reason`` ever approved.
    """
    (console_store / "decisions" / decision_file).write_text(payload, encoding="utf-8")


def hand_decision(console_store: Path, **update: object) -> str:
    """A valid gate decision, mutated and written directly to disk.

    The learning id defaults to the new proposal id so the hand-written
    decision points at files only a real record would have.
    """
    proposal_id = str(update.get("proposal_id", "prop-1"))
    base = DecisionLog(console_store).all()[0].model_copy(
        update={
            "decided_at": datetime.now(UTC),
            "evidence_ref": None,
            "trajectory_ref": None,
            "learning_id": proposal_id,
            **update,
        }
    )
    write_decision_file(console_store, f"{base.proposal_id}.json", base.model_dump_json())
    return str(base.proposal_id)


# ------------------------------------------------------------------- overview


def test_overview_renders_counts_and_launch_context(
    client: TestClient, console_store: Path
) -> None:
    response = client.get("/")
    assert response.status_code == 200
    # The operator identity is shown on every page — escaped, like all data.
    assert "Nihanth &lt;nihanth@localhost&gt;" in response.text
    assert str(console_store) in response.text  # the store root is shown
    assert "pending" in response.text


def test_overview_matches_the_cli_report(
    console_store: Path, client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The page and ``evogine decisions report`` agree on the same numbers."""
    exit_code = cli_main(["decisions", "report", "--root", str(console_store)])
    assert exit_code == 0
    out = capsys.readouterr().out

    cli_total = int(
        next(line for line in out.splitlines() if "total" in line).split()[2]
    )
    cli_promoted = int(
        next(line for line in out.splitlines() if line.startswith("promoted")).split()[1]
    )
    cli_by_stage: dict[str, int] = {}
    for line in out.splitlines():
        stripped = line.strip()
        if (
            stripped
            and ": " in stripped
            and not stripped.startswith("promotion")
            and not stripped.startswith("promoted")
        ):
            stage, count = stripped.rsplit(": ", 1)
            cli_by_stage[stage] = int(count)

    counts = views.build_overview(console_store)
    page = client.get("/")

    assert counts.total_decisions == cli_total
    assert counts.promoted == cli_promoted
    assert counts.by_stage == cli_by_stage
    for stage, count in cli_by_stage.items():
        assert stage in page.text
        assert str(count) in page.text


# ------------------------------------------------------------------- decisions


def test_decision_list_shows_machine_checked_reasons(client: TestClient) -> None:
    response = client.get("/decisions")
    assert response.status_code == 200
    for proposal_id in ("prop-1", "prop-2", "prop-3", "prop-4", "prop-5"):
        assert proposal_id in response.text
    # Reasons come from the gate's machine-checked vocabulary, not prose —
    # one fixture decision per stage the vocabulary distinguishes.
    assert "proposed:" in response.text
    assert "no_flip:" in response.text
    assert "not_material:" in response.text
    assert "regression:" in response.text
    # Every row links its drill-down.
    assert "/decisions/prop-1" in response.text


def test_decision_detail_shows_paired_evidence_and_trajectory(
    client: TestClient,
) -> None:
    response = client.get("/decisions/prop-1")
    assert response.status_code == 200
    assert "case-0" in response.text and "case-2" in response.text
    assert "fail" in response.text and "pass" in response.text
    # The approved record's lifecycle status is shown.
    assert "promoted" in response.text
    # The sanitized trajectory capture is rendered, tool calls included.
    assert "evidence_sink.write" in response.text


def test_unknown_proposal_is_a_refusal_page(client: TestClient) -> None:
    response = client.get("/decisions/appr-nobody")
    assert response.status_code == 404
    assert "refuses" in response.text
    assert "appr-nobody" in response.text


def test_unsafe_proposal_id_is_refused_before_any_disk_read(
    console_store: Path,
) -> None:
    """A traversal-shaped id is caught by the id rule, never resolved on disk.

    (An HTTP-level probe cannot exercise this: the test client normalizes
    dot-segments out of the URL before the route ever sees them, so the
    guard is pinned at the view boundary it lives on.)
    """
    with pytest.raises(ConsoleError, match="not a filesystem-safe id"):
        views.build_decision_detail(console_store, "../etc/passwd")


def test_gate_error_decision_shows_honest_absence(
    console_store: Path, client: TestClient
) -> None:
    """A gate_error decision has no evidence or trajectory; the page says so."""
    proposal_id = hand_decision(
        console_store,
        proposal_id="prop-err",
        stage=DecisionStage.GATE_ERROR,
        reason="gate_error: replay runner exploded",
    )
    response = client.get(f"/decisions/{proposal_id}")
    assert response.status_code == 200
    assert "no evidence file" in response.text
    assert "no trajectory recorded" in response.text


def test_scored_evidence_renders_scores(client: TestClient) -> None:
    response = client.get("/decisions/prop-3")
    assert response.status_code == 200
    assert "score 0.400" in response.text
    assert "score 0.500" in response.text
    assert "Material: <strong>no</strong>" in response.text


def test_mismatched_arms_show_pairing_error(
    console_store: Path, client: TestClient
) -> None:
    """Arms covering different case sets are shown as refused, not half-rendered."""
    proposal_id = hand_decision(
        console_store, proposal_id="prop-mismatch", learning_id="prop-mismatch"
    )
    (console_store / "runs" / "gate" / "prop-mismatch.json").write_text(
        '{"learning_id": "prop-mismatch", "replay_run_ids": ["r-1"],'
        ' "materiality": {"material": true, "detail": "1 binary flips"},'
        ' "baseline": [{"instance_id": "case-1", "passed": true},'
        ' {"instance_id": "case-2", "passed": false}],'
        ' "warm": [{"instance_id": "case-1", "passed": true},'
        ' {"instance_id": "case-3", "passed": false}]}',
        encoding="utf-8",
    )
    response = client.get(f"/decisions/{proposal_id}")
    assert response.status_code == 200
    assert "cannot be paired" in response.text


def test_verdict_without_outcome_is_refused(
    console_store: Path, client: TestClient
) -> None:
    proposal_id = hand_decision(
        console_store, proposal_id="prop-blank", learning_id="prop-blank"
    )
    (console_store / "runs" / "gate" / "prop-blank.json").write_text(
        '{"learning_id": "prop-blank", "replay_run_ids": ["r-1"],'
        ' "materiality": {"material": true, "detail": "1 binary flips"},'
        ' "baseline": [{"instance_id": "case-9"}],'
        ' "warm": [{"instance_id": "case-9", "passed": true}]}',
        encoding="utf-8",
    )
    response = client.get(f"/decisions/{proposal_id}")
    assert response.status_code == 404
    # The Verdict model itself refuses outcome-less verdicts at parse time,
    # so the page shows the malformed-evidence refusal carrying that message.
    assert "is malformed" in response.text
    assert "exactly one of passed/score" in response.text


def test_record_absent_note_is_shown(console_store: Path, client: TestClient) -> None:
    proposal_id = hand_decision(
        console_store, proposal_id="prop-ghost", learning_id="prop-ghost"
    )
    response = client.get(f"/decisions/{proposal_id}")
    assert response.status_code == 200
    assert "no record prop-ghost in this store" in response.text


def test_malformed_decision_file_refuses_the_list(
    console_store: Path, client: TestClient
) -> None:
    write_decision_file(console_store, "prop-junk.json", "{not json at all")
    response = client.get("/decisions")
    assert response.status_code == 500
    assert "malformed" in response.text


# ------------------------------------------------------------------- approvals


def test_approvals_queue_lists_pending_first_then_decided(
    client: TestClient, console_store: Path
) -> None:
    response = client.get("/approvals")
    assert response.status_code == 200
    assert "nihanth" in response.text  # decided_by is shown
    assert "convincing" in response.text  # decided_reason is shown
    assert "cleared:" in response.text  # the proposal's machine-checked trail
    assert "read-only" in response.text  # the page points writers at the CLI
    # The drill-down links by record id — the key the decision log uses.
    assert "/decisions/prop-1" in response.text
    assert "/decisions/prop-5" in response.text

    rows = views.list_approvals(console_store)
    assert [row.proposal_id for row in rows] == ["appr-prop-5", "appr-prop-1"]
    assert rows[0].state == "pending"
    assert rows[1].state == "applied"


# ----------------------------------------------------------------------- audit


def test_audit_timeline_renders_store_history(
    client: TestClient, console_store: Path
) -> None:
    response = client.get("/audit")
    assert response.status_code == 200
    timeline = views.build_timeline(console_store)
    assert timeline, "the fixture store has real audit history"
    seqs = [row.seq for row in timeline]
    assert seqs == sorted(seqs, reverse=True)  # newest first
    assert any(row.to_status == "promoted" for row in timeline)
    assert "cold-1" in response.text  # run ids render


def test_audit_page_is_honestly_empty(root: Path) -> None:
    empty = TestClient(create_app(root, OPERATOR))
    response = empty.get("/audit")
    assert response.status_code == 200
    assert "empty" in response.text


def test_malformed_audit_line_is_refused_not_skipped(
    console_store: Path, client: TestClient
) -> None:
    with open(console_store / "audit.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"seq": 999, "event": }\n')  # malformed JSON
    response = client.get("/audit")
    assert response.status_code == 500
    assert "malformed" in response.text


# ------------------------------------------------------------------ escaping


def test_store_data_is_escaped_in_pages(console_store: Path) -> None:
    """A hand-crafted decision file is store data: escaped, never HTML."""
    hand_decision(
        console_store,
        proposal_id="prop-evil",
        reason="proposed: <script>alert(1)</script>",
    )
    page = TestClient(create_app(console_store, OPERATOR)).get("/decisions")
    assert "&lt;script&gt;" in page.text
    assert "<script>alert(1)" not in page.text


# --------------------------------------------------------------------- launch


def test_non_loopback_bind_is_refused_at_startup(console_store: Path) -> None:
    with pytest.raises(ConsoleError, match="loopback"):
        serve(str(console_store), OPERATOR, host="0.0.0.0")


def test_public_hostname_is_refused(console_store: Path) -> None:
    with pytest.raises(ConsoleError, match="refusing non-loopback bind"):
        serve(str(console_store), OPERATOR, host="example.com")


def test_console_main_refuses_non_loopback_host(
    console_store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = console_main(
        ["--root", str(console_store), "--operator", OPERATOR, "--host", "0.0.0.0"]
    )
    assert exit_code == 1
    assert "loopback" in capsys.readouterr().err


def test_localhost_resolves_to_loopback() -> None:
    host = resolve_bind_host("localhost")
    assert host in {"127.0.0.1", "::1"}


def test_unresolvable_host_is_refused() -> None:
    with pytest.raises(ConsoleError, match="cannot resolve"):
        resolve_bind_host("no-such-host.invalid")


def test_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConsoleError, match="does not exist"):
        validate_root(tmp_path / "nope")


def test_directory_without_store_markers_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-store"
    empty.mkdir()
    with pytest.raises(ConsoleError, match="does not look like an Evogine store root"):
        validate_root(empty)


def test_empty_operator_identity_is_refused() -> None:
    with pytest.raises(ConsoleError, match="operator identity is required"):
        validate_operator("   ")
