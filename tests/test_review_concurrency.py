"""Concurrency tests for the human-approval boundary (V3).

The boundary's pending check is read-modify-write; a second concurrent
writer — the CLI and a console transport against one store, say — is the
first real race this store has met. The advisory lock file at the store
root makes the documented "one writer at a time per store root" contract
enforced: these tests run the boundary functions in two independent
workers (each with its own :class:`Engine` and :class:`ApprovalLog` —
separate forked processes, like a CLI and a console) and hold the outcome
to exactly one success and one clean refusal. Fully offline; no sleeps —
the lock serializes, the loser's pending re-check decides the outcome.
"""

import json
import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from evogine.engine.store import LOCK_FILENAME, Engine, store_lock
from evogine.gates.approvals import (
    ApprovalError,
    ApprovalLog,
    ApprovalState,
    approve_proposal,
    reject_proposal,
)
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, RunsEvidenceSink

PROPOSAL_ID = "appr-prop-9"
LEARNING_ID = "prop-9"


class StubRunner:
    """Deterministic replay runner: run id -> scripted ReplayRun."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise KeyError(f"unknown replay run: {run_id}")
        return self._runs[run_id]


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[Verdict(instance_id=f"case-{i}", passed=ok) for i, ok in enumerate(outcomes)],
    )


CLEARED_RUNNER = StubRunner(
    {
        "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
        "warm-1": binary_run("warm-1", "warm", [True, True, True]),
    }
)


def _audit_events(root: Path) -> list[dict[str, object]]:
    lines = (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def pending(root: Path, record_factory) -> str:
    """One reviewed record whose cleared knowledge is a pending proposal."""
    store = Engine(root)
    approvals = ApprovalLog(root)
    record = store.propose(record_factory(LEARNING_ID))
    store.queue_replay(record.id, ["cold-1", "warm-1"])
    gate = MemoryGate(CLEARED_RUNNER, DecisionLog(root), approvals, evidence=RunsEvidenceSink(root))
    gate.review(store.get(record.id), ["cold-1", "warm-1"])
    return PROPOSAL_ID


Decider = Callable[[Engine, ApprovalLog], str]


def _decide_child(root: Path, decide: Decider, out_w: int) -> None:
    """One decision worker, running as its own process (see ``_run_decider``)."""
    try:
        outcome = decide(Engine(root), ApprovalLog(root))
    except ApprovalError as exc:
        outcome = f"refused: {exc}"
    except OSError as exc:  # environment failure — surfaced, never swallowed
        outcome = f"error: {exc!r}"
    os.write(out_w, f"{outcome}\n".encode())


def _run_decider(root: Path, decide: Decider) -> str:
    """Run one decision worker as a forked child process; return its outcome.

    A forked process, not a thread: a real transport forks git from its
    own main thread, and the sandboxed candidate container refuses
    fork+exec from secondary threads outright (EAGAIN, observed in the
    Docker-gated run) — threads cannot perform a real approval there.
    The child reports its outcome on a pipe; the parent reaps it and
    reads to EOF, so nothing a worker raises is swallowed. Two deciders
    run as two sequential children: the container also refuses the
    concurrent-fork version, and the race semantics hold regardless of
    interleaving — the loser's pending re-check (this design) and the
    mutual exclusion itself (the one-holder test) pin both halves.
    """
    out_r, out_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        _decide_child(root, decide, out_w)
        os._exit(0)  # never unwind the parent's pytest state

    os.close(out_w)  # the parent's copy; EOF arrives when the child exits
    try:
        status = os.waitpid(pid, 0)[1]
        assert os.waitstatus_to_exitcode(status) == 0, f"a worker exited {status}"
        chunks: list[bytes] = []
        while chunk := os.read(out_r, 4096):
            chunks.append(chunk)
    finally:
        os.close(out_r)
    lines = b"".join(chunks).decode().splitlines()
    assert len(lines) == 1, f"a worker reported {len(lines)} outcomes"
    return lines[0]


# ------------------------------------------------------------------ the lock


def test_lock_file_lives_at_the_store_root_and_is_committed(
    root: Path, record_factory
) -> None:
    """Store-root state from birth: the store's own git repo tracks the file."""
    engine = Engine(root)
    assert engine.lock_path == root / LOCK_FILENAME
    assert engine.lock_path.exists()

    # The first lifecycle commit versions the lock file — it is store state,
    # not transient runtime residue (a lazily created lock would linger
    # untracked after a refused boundary action, which runs no commit).
    engine.propose(record_factory(LEARNING_ID))
    tracked = subprocess.run(
        ["git", "ls-files", "--", LOCK_FILENAME],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert tracked == LOCK_FILENAME


def test_store_lock_admits_one_holder_at_a_time(root: Path) -> None:
    """A second holder blocks until the first releases — no concurrent window."""
    root.mkdir(parents=True)  # bare root: acquisition itself creates the lock file
    entered = threading.Event()
    released = threading.Event()

    def second_holder() -> None:
        with store_lock(root):
            entered.set()
        released.set()

    with store_lock(root):
        thread = threading.Thread(target=second_holder)
        thread.start()
        assert not entered.wait(timeout=0.5), "second holder entered while locked"
    assert released.wait(timeout=5), "second holder never acquired after release"
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_store_lock_creates_the_file_for_older_stores(root: Path) -> None:
    """A store predating the lock still coordinates: acquisition creates the file."""
    root.mkdir(parents=True)  # a bare pre-V3 root without Engine's init
    with store_lock(root):
        assert (root / LOCK_FILENAME).exists()


# --------------------------------------------------- concurrent human decisions


def _git_state(root: Path) -> tuple[list[str], list[str]]:
    """Store git history (oldest first) and uncommitted residue paths."""
    subjects = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    subjects.reverse()
    residue = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return subjects, residue


def test_two_workers_approving_one_proposal_yield_one_success_one_refusal(
    root: Path, pending: str
) -> None:
    """Two independent workers approve the same proposal: exactly one wins.

    Each worker owns its Engine and ApprovalLog — independent transports,
    like the CLI and a console against one store. The loser blocks on the
    store's review lock, then fails its own re-check of pending status: a
    clean refusal, never a corrupt store.
    """
    def approve_it(engine: Engine, log: ApprovalLog) -> str:
        approve_proposal(engine, log, pending, decided_by="worker")
        return "promoted"

    outcomes = [
        _run_decider(root, approve_it),
        _run_decider(root, approve_it),
    ]

    # Exactly one success, one clean refusal — the loser failed its own
    # pending re-check ("already ..."), it did not crash or corrupt.
    successes = [outcome for outcome in outcomes if not outcome.startswith("refused")]
    refusals = [outcome for outcome in outcomes if outcome.startswith("refused")]
    assert len(successes) == 1, outcomes
    assert len(refusals) == 1, outcomes
    assert "already" in refusals[0], refusals

    # Store consistent: the record is promoted exactly once, retrieval sees it.
    store = Engine(root)
    assert store.get(LEARNING_ID).status == "promoted"
    assert [record.id for record in store.promoted_records()] == [LEARNING_ID]
    decided = ApprovalLog(root).get(pending)
    assert decided.state is ApprovalState.APPLIED

    # Audit trail sane: every transition happened exactly once, in order,
    # with the lock keeping the append-only sequence gap-free.
    events = _audit_events(root)
    assert [event["event"] for event in events] == [
        "propose",
        "queue_replay",
        "promote",
    ]
    assert [event["seq"] for event in events] == [1, 2, 3]

    # Git history consistent: exactly one promote commit (a double-apply
    # would show two), and the only uncommitted residue is the applied
    # marker — record_applied deliberately runs after the promote commit,
    # so it waits for the next lifecycle action to be swept.
    subjects, residue = _git_state(root)
    assert subjects == [
        "propose: prop-9",
        "queue_replay: prop-9 (2 runs)",
        "promote: prop-9",
    ]
    assert residue == [" M approvals/appr-prop-9.json"]


def test_concurrent_approve_and_reject_yield_exactly_one_verdict(root: Path, pending: str) -> None:
    """Approve and reject racing on one proposal: one verdict, one refusal."""
    def approve_it(engine: Engine, log: ApprovalLog) -> str:
        approve_proposal(engine, log, pending, decided_by="approver")
        return "promoted"

    def reject_it(engine: Engine, log: ApprovalLog) -> str:
        reject_proposal(engine, log, pending, decided_by="rejecter", reason="no")
        return "dropped"

    outcomes = [_run_decider(root, approve_it), _run_decider(root, reject_it)]

    successes = [outcome for outcome in outcomes if not outcome.startswith("refused")]
    refusals = [outcome for outcome in outcomes if outcome.startswith("refused")]
    assert len(successes) == 1, outcomes
    assert len(refusals) == 1, outcomes
    assert "already" in refusals[0], refusals

    # The store sits in exactly the terminal state the winner chose.
    store = Engine(root)
    final_status = store.get(LEARNING_ID).status
    assert final_status == successes[0]
    events = _audit_events(root)
    expected_event = "promote" if final_status == "promoted" else "demote"
    assert [event["event"] for event in events] == [
        "propose",
        "queue_replay",
        expected_event,
    ]
    assert [event["seq"] for event in events] == [1, 2, 3]

    # Git history: exactly one terminal transition commit, matching the
    # winner's verdict. Residue follows the fail-safe order — an approver
    # win leaves the applied marker uncommitted (record_applied runs after
    # the promote commit); a rejecter win leaves nothing (demote is last).
    subjects, residue = _git_state(root)
    terminal = "promote: prop-9" if final_status == "promoted" else "demote: prop-9"
    assert subjects == [
        "propose: prop-9",
        "queue_replay: prop-9 (2 runs)",
        terminal,
    ]
    expected_residue = (
        [" M approvals/appr-prop-9.json"] if final_status == "promoted" else []
    )
    assert residue == expected_residue


def test_refused_boundary_action_leaves_the_store_clean(root: Path, pending: str) -> None:
    """A refusal inside the lock window adds nothing to the store.

    The refusal runs no commit, so the honest invariant is that git status
    is unchanged by it — the lock file itself is versioned from store birth,
    never residue.
    """
    store = Engine(root)
    log = ApprovalLog(root)
    decided = log.get(pending).model_copy(update={"state": ApprovalState.APPLIED})
    (root / "approvals" / f"{pending}.json").write_text(decided.model_dump_json())

    def porcelain() -> str:
        return subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    before = porcelain()
    with pytest.raises(ApprovalError, match="already applied"):
        approve_proposal(store, log, pending, decided_by="nihanth")

    assert store.get(LEARNING_ID).status == "candidate"
    assert porcelain() == before
