"""Console write-path tests: approve/reject through the boundary, over HTTP.

The write routes are transport over ``approve_proposal``/``reject_proposal``
— the boundary owns the lock, the pending re-check, the evidence
requirement, and the rejection reason — so these tests exercise the routes
against a real fixture store and assert what the transport itself adds:
the operator identity on the promote commit, the visible refusal states
that leave the store untouched, the machine-checked reason gate, and the
loopback-only bind. Fully offline: fixture store, scripted replay runs,
loopback sockets only.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

import pytest

# The write path lives behind the optional [console] extra: core CI jobs
# never install it, so this module skips as a whole there (one visible
# skip, the read-path suite's precedent) and runs for real in the
# dedicated console job. The imports after the skip call are
# module-level-late by design; E402 is ignored for this file in pyproject.
pytest.importorskip("fastapi")

import httpx
import uvicorn
from fastapi.testclient import TestClient

from evogine.console.app import create_app
from evogine.console.launch import ConsoleError
from evogine.console.server import serve
from evogine.console.write import is_safe_proposal_id, parse_reason_field
from evogine.engine.store import LOCK_FILENAME, AuditEvent, Engine
from evogine.gates.approvals import ApprovalLog, ApprovalState, approve_proposal
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink

OPERATOR = "nihanth"
PROPOSAL_ID = "appr-prop-9"
LEARNING_ID = "prop-9"
# The promotion pass that cleared the fixture record mints its own id; the
# fixture pins one so the pass-identity assertions are exact.
PASS_ID = "pass-20260919T120000-cafe01"
REASON = "warm arm is unconvincing on the degraded case"


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=ok) for i, ok in enumerate(outcomes)
        ],
    )


CLEARED_RUNNER_STUB = {
    "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
    "warm-1": binary_run("warm-1", "warm", [True, True, True]),
}


class StubRunner:
    """Deterministic replay runner: run id -> scripted ReplayRun."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise KeyError(f"unknown replay run: {run_id}")
        return self._runs[run_id]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def store(root: Path) -> Engine:
    return Engine(root)


@pytest.fixture
def pending(root: Path, store: Engine, record_factory) -> Path:
    """A store with one pending proposal cleared by a pass (pass_id stamped)."""
    record = store.propose(record_factory(LEARNING_ID))
    store.queue_replay(record.id, ["cold-1", "warm-1"])
    MemoryGate(
        runner=StubRunner(CLEARED_RUNNER_STUB),
        log=DecisionLog(root),
        approvals=ApprovalLog(root),
        evidence=RunsEvidenceSink(root),
    ).review(store.get(record.id), ["cold-1", "warm-1"], pass_id=PASS_ID)
    return root


@pytest.fixture
def decided(pending: Path, store: Engine) -> Path:
    """The pending store, but its proposal already approved (applied)."""
    approve_proposal(
        store, ApprovalLog(pending), PROPOSAL_ID, decided_by="nihanth", reason="convincing"
    )
    return pending


@pytest.fixture
def client(pending: Path) -> TestClient:
    return TestClient(create_app(pending, OPERATOR), raise_server_exceptions=False)


def git_out(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def audit_events(root: Path) -> list[AuditEvent]:
    return [
        AuditEvent.model_validate_json(line)
        for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def store_snapshot(root: Path) -> tuple[bytes, str]:
    """Audit bytes plus HEAD sha — the pair that must not move on a refusal."""
    audit = (root / "audit.jsonl").read_bytes()
    return audit, git_out(root, "rev-parse", "HEAD")


class LiveConsole:
    """A real uvicorn console on an ephemeral loopback port (offline)."""

    def __init__(self, root: Path) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(root, OPERATOR), host="127.0.0.1", port=0, log_level="warning"
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("console server did not start")
            time.sleep(0.05)
        assert self.server.servers, "uvicorn exposed no socket"
        host, port = self.server.servers[0].sockets[0].getsockname()[:2]
        assert host in {"127.0.0.1", "::1"}  # the bind itself is loopback
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30)


# ------------------------------------------------------------------ end-to-end


def test_end_to_end_approve_records_operator_identity_and_pass(
    pending: Path, client: TestClient
) -> None:
    """Approve through the console: promoted record, identity on the commit.

    The promote commit must carry the operator identity (the console's
    launched operator is the ``decided_by`` the boundary records, and the
    store's promotion commit versions the decided proposal), and the pass
    identity must survive the transport (the clearing pass's ``pass_id``
    stays on the approval trail that attributes the promotion to it).
    """
    response = client.post(f"/approvals/{PROPOSAL_ID}/approve", data={}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/approvals"  # PRG: refresh is safe

    log = ApprovalLog(pending)
    decided = log.get(PROPOSAL_ID)
    assert decided.state is ApprovalState.APPLIED
    assert decided.decided_by == OPERATOR

    # The promote commit versions the decided proposal — the operator
    # identity is in the store's history at the moment of promotion.
    log_lines = git_out(pending, "log", "--format=%H %s").splitlines()
    promote_sha = next(
        line.split()[0] for line in log_lines if line.endswith(f"promote: {LEARNING_ID}")
    )
    committed = json.loads(
        git_out(pending, "show", f"{promote_sha}:approvals/{PROPOSAL_ID}.json")
    )
    assert committed["decided_by"] == OPERATOR
    # The boundary marks the proposal applied only after the promote commit,
    # so the versioned snapshot carries the approved state at promote time.
    assert committed["state"] == "approved"

    # The pass identity survives the transport: the approval-stage decision
    # line and the proposal's own trail both carry the clearing pass's id.
    approval_line = next(
        decision
        for decision in DecisionLog(pending).all()
        if decision.stage is DecisionStage.APPROVAL
    )
    assert approval_line.pass_id == PASS_ID
    assert decided.decision.pass_id == PASS_ID

    # The promote audit line names the record and the evidence runs.
    promote_events = [
        event
        for event in audit_events(pending)
        if event.event == "promote" and event.record_id == LEARNING_ID
    ]
    assert len(promote_events) == 1
    assert promote_events[0].from_status == "candidate"
    assert promote_events[0].to_status == "promoted"
    assert promote_events[0].run_ids == ["cold-1", "warm-1"]


def test_approve_note_is_recorded(pending: Path, client: TestClient) -> None:
    """The form's note lands as ``decided_reason``."""
    approved = client.post(
        f"/approvals/{PROPOSAL_ID}/approve", data={"reason": "lgm"}, follow_redirects=False
    )
    assert approved.status_code == 303
    assert ApprovalLog(pending).get(PROPOSAL_ID).decided_reason == "lgm"


def test_approve_without_a_note_records_no_reason(
    pending: Path, client: TestClient
) -> None:
    approved = client.post(f"/approvals/{PROPOSAL_ID}/approve", data={}, follow_redirects=False)
    assert approved.status_code == 303
    assert ApprovalLog(pending).get(PROPOSAL_ID).decided_reason is None


# ----------------------------------------------------------------- concurrency


def test_concurrent_approves_through_the_http_layer_produce_one_promotion(
    pending: Path,
) -> None:
    """Two simultaneous HTTP approves: one wins, one refuses visibly.

    Both requests block on the store's review lock (the boundary holds it
    across the decision window); after release the second one fails its own
    pending re-check and renders the refusal — exactly one promotion, one
    promote commit, one audit event.
    """
    with LiveConsole(pending) as base:
        lock_path = pending / LOCK_FILENAME
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # hold the review window
            results: dict[int, httpx.Response] = {}

            def post(index: int) -> None:
                results[index] = httpx.post(
                    f"{base}/approvals/{PROPOSAL_ID}/approve",
                    data={},
                    timeout=30,
                )

            threads = [threading.Thread(target=post, args=(index,)) for index in (1, 2)]
            for thread in threads:
                thread.start()
            time.sleep(0.5)  # both requests have arrived; both block on the lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # let them race
        for thread in threads:
            thread.join(timeout=30)

    assert sorted(response.status_code for response in results.values()) == [303, 409]
    loser = next(response for response in results.values() if response.status_code == 409)
    assert "already" in loser.text  # the refusal names the decided proposal

    # Exactly one promotion, in the audit trail and in the store history.
    promote_events = [
        event
        for event in audit_events(pending)
        if event.event == "promote" and event.record_id == LEARNING_ID
    ]
    assert len(promote_events) == 1
    promote_commits = [
        line
        for line in git_out(pending, "log", "--format=%s").splitlines()
        if line == f"promote: {LEARNING_ID}"
    ]
    assert len(promote_commits) == 1
    assert ApprovalLog(pending).get(PROPOSAL_ID).state is ApprovalState.APPLIED


# -------------------------------------------------------------- refusal states


def test_approve_on_a_decided_proposal_refuses_and_leaves_the_store_untouched(
    decided: Path, store: Engine
) -> None:
    """A stale proposal gets a visible refusal; the store does not move."""
    before = store_snapshot(decided)
    client = TestClient(create_app(decided, OPERATOR), raise_server_exceptions=False)
    response = client.post(f"/approvals/{PROPOSAL_ID}/approve", data={})
    assert response.status_code == 409
    assert "refuses" in response.text  # the visible refusal page
    assert "already applied" in response.text
    assert store_snapshot(decided) == before
    assert store.get(LEARNING_ID).status == "promoted"  # untouched, not re-promoted


def test_reject_on_a_decided_proposal_refuses_and_leaves_the_store_untouched(
    decided: Path, store: Engine
) -> None:
    before = store_snapshot(decided)
    client = TestClient(create_app(decided, OPERATOR), raise_server_exceptions=False)
    response = client.post(f"/approvals/{PROPOSAL_ID}/reject", data={"reason": REASON})
    assert response.status_code == 409
    assert "refuses" in response.text
    assert store_snapshot(decided) == before
    assert store.get(LEARNING_ID).status == "promoted"  # a decided proposal is final


def test_reject_requires_a_nonblank_reason(
    pending: Path, store: Engine, client: TestClient
) -> None:
    """The machine-checked reason gate refuses before the boundary is called."""
    before = store_snapshot(pending)
    for form_body in ({}, {"reason": ""}, {"reason": "   "}):
        response = client.post(f"/approvals/{PROPOSAL_ID}/reject", data=form_body)
        assert response.status_code == 400
        assert "refuses" in response.text
        assert "nonblank reason" in response.text
    assert store_snapshot(pending) == before
    assert store.get(LEARNING_ID).status == "candidate"
    assert ApprovalLog(pending).get(PROPOSAL_ID).state is ApprovalState.PENDING


def test_reject_records_identity_and_reason_and_drops_the_record(
    pending: Path, store: Engine, client: TestClient
) -> None:
    """A rejection is the boundary's knowledge verdict, with identity attached."""
    response = client.post(
        f"/approvals/{PROPOSAL_ID}/reject", data={"reason": REASON}, follow_redirects=False
    )
    assert response.status_code == 303

    decided = ApprovalLog(pending).get(PROPOSAL_ID)
    assert decided.state is ApprovalState.REJECTED
    assert decided.decided_by == OPERATOR
    assert decided.decided_reason == REASON
    assert store.get(LEARNING_ID).status == "dropped"

    demote_events = [
        event
        for event in audit_events(pending)
        if event.event == "demote" and event.record_id == LEARNING_ID
    ]
    assert len(demote_events) == 1
    assert demote_events[0].to_status == "dropped"
    assert demote_events[0].reason == f"rejected by {OPERATOR}: {REASON}"


def test_unknown_proposal_is_a_refusal(client: TestClient) -> None:
    approve = client.post("/approvals/appr-nobody/approve", data={})
    reject = client.post("/approvals/appr-nobody/reject", data={"reason": REASON})
    assert approve.status_code == 404
    assert reject.status_code == 404
    assert "appr-nobody" in approve.text
    assert "refuses" in approve.text


def test_malformed_proposal_file_is_refused(pending: Path, client: TestClient) -> None:
    (pending / "approvals" / "appr-junk.json").write_text("{not json", encoding="utf-8")
    response = client.post("/approvals/appr-junk/approve", data={})
    assert response.status_code == 500
    assert "malformed" in response.text


def test_unsafe_proposal_id_is_refused_before_any_disk_use(client: TestClient) -> None:
    """A pattern-violating id is a visible refusal, never a path use.

    (Traversal-shaped ids cannot reach the route over HTTP — the test
    client normalizes dot-segments — so the guard is pinned at its function
    boundary, the read path's precedent, and the route branch is exercised
    with an HTTP-safe but pattern-violating id.)
    """
    assert is_safe_proposal_id(PROPOSAL_ID)
    assert not is_safe_proposal_id("../etc/passwd")
    assert not is_safe_proposal_id("appr bad")
    assert not is_safe_proposal_id("")
    response = client.post("/approvals/appr+bad/approve", data={})
    assert response.status_code == 400
    assert "filesystem-safe" in response.text


# -------------------------------------------------------------------- identity


def test_boundary_incompatible_launch_identity_is_refused_at_launch(
    pending: Path,
) -> None:
    """A launched identity the boundary will never record fails at startup.

    The documented ``Name <email>`` format is exactly what the boundary's
    operator rule refuses — better a launch refusal naming the fix than a
    surprise 409 on the operator's first decision.
    """
    with pytest.raises(RuntimeError, match="relaunch"):
        create_app(pending, "Nihanth <nihanth@localhost>")


# ------------------------------------------------------------------------ bind


def test_write_routes_are_loopback_only(decided: Path) -> None:
    """The loopback-only contract covers the write path: no non-loopback bind.

    The refusal happens before any socket exists, even though the write
    routes are registered on the app being launched.
    """
    with pytest.raises(ConsoleError, match="loopback"):
        serve(str(decided), OPERATOR, host="0.0.0.0")


def test_post_to_the_queue_page_is_method_not_allowed(client: TestClient) -> None:
    assert client.post("/approvals").status_code == 405


# ---------------------------------------------------------- form-body parsing


def test_parse_reason_field_reads_the_reason_among_fields() -> None:
    body = urllib.parse.urlencode({"other": "x", "reason": "warm arm regressed"}).encode()
    assert parse_reason_field(body) == "warm arm regressed"


def test_parse_reason_field_defaults_to_empty() -> None:
    assert parse_reason_field(b"") == ""
    assert parse_reason_field(b"other=x") == ""
