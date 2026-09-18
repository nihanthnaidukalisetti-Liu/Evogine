"""Pass-identity and review-latency telemetry tests (V3), fully offline.

Every promotion pass is attributable: the pass mints a filesystem-safe
``pass_id``, stamps it on the decisions it produces and on the audit
events its demotions cause, and leaves a manifest at
``passes/<pass_id>.json`` — what ran, what it decided, and how long each
review took. Review latency is computed from the two clocks that already
exist (the record's ``created`` and the decision's ``decided_at``). The
operator identity is validated once, at the approval boundary's
``record_*`` choke point; legacy stored values stay readable because the
load path never validates.
"""

import json
import re
from pathlib import Path

import pytest

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.approvals import (
    ApprovalError,
    ApprovalLog,
    ApprovalState,
    operator_identity,
)
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import (
    DecisionLog,
    DecisionStage,
    RunsEvidenceSink,
)


class ScriptedRunner:
    """Deterministic replay runner; unknown run ids raise (gate_error path)."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise RuntimeError(f"replay executor unavailable for {run_id}")
        return self._runs[run_id]


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=o) for i, o in enumerate(outcomes)
        ],
    )


# prop-a: full binary flip — the gate clears it and renders a pending
# proposal (approval stage). prop-b: no flip — the pass demotes it.
RUNNER = ScriptedRunner(
    {
        "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
        "warm-1": binary_run("warm-1", "warm", [True, True, True]),
        "prop-b-cold": binary_run("prop-b-cold", "baseline", [False, True, False]),
        "prop-b-warm": binary_run("prop-b-warm", "warm", [False, True, False]),
    }
)


def _audit_events(root: Path) -> list[dict[str, object]]:
    lines = (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture
def store(tmp_path: Path) -> Engine:
    return Engine(tmp_path / "store")


def _queue(store: Engine, record: LearningRecord, runs: list[str]) -> LearningRecord:
    store.queue_replay(record.id, runs)
    return store.get(record.id)


def _gate(store: Engine) -> MemoryGate:
    return MemoryGate(
        RUNNER,
        DecisionLog(store.root),
        ApprovalLog(store.root),
        evidence=RunsEvidenceSink(store.root),
    )


@pytest.fixture
def two_candidates(store: Engine, record_factory) -> list[LearningRecord]:
    """prop-a (clears, pending proposal) and prop-b (stopped at REPLAY)."""
    record_a = store.propose(record_factory("prop-a"))
    queued_a = _queue(store, record_a, ["cold-1", "warm-1"])
    record_b = store.propose(record_factory("prop-b"))
    queued_b = _queue(store, record_b, ["prop-b-cold", "prop-b-warm"])
    return [queued_a, queued_b]


# ------------------------------------------------------------- pass identity


def test_promotion_pass_stamps_one_pass_id_on_every_decision(
    store: Engine, two_candidates
) -> None:
    decisions = run_promotion_pass(store, _gate(store))

    assert len(decisions) == 2
    pass_ids = {d.pass_id for d in decisions}
    assert len(pass_ids) == 1
    assert None not in pass_ids


def test_promotion_pass_writes_a_manifest_at_the_store_root(
    store: Engine, two_candidates
) -> None:
    decisions = run_promotion_pass(store, _gate(store))
    pass_id = decisions[0].pass_id
    assert pass_id is not None

    manifest_path = store.root / "passes" / f"{pass_id}.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["pass_id"] == pass_id
    assert manifest["candidates"] == ["prop-a", "prop-b"]
    assert manifest["started_at"] <= manifest["finished_at"]
    by_proposal = {entry["proposal_id"]: entry for entry in manifest["decisions"]}
    assert set(by_proposal) == {"prop-a", "prop-b"}
    for entry in manifest["decisions"]:
        assert entry["review_latency_seconds"] is not None
    assert by_proposal["prop-a"]["stage"] == DecisionStage.APPROVAL.value
    assert by_proposal["prop-a"]["promoted"] is False
    assert by_proposal["prop-b"]["stage"] == DecisionStage.REPLAY.value
    assert manifest["counts_by_stage"] == {
        DecisionStage.APPROVAL.value: 1,
        DecisionStage.REPLAY.value: 1,
    }


def test_promotion_pass_audit_events_carry_the_pass_id(
    store: Engine, two_candidates
) -> None:
    """Demotion audit lines name the pass that caused them; nothing else moves."""
    decisions = run_promotion_pass(store, _gate(store))
    pass_id = decisions[0].pass_id
    assert pass_id is not None

    demotes = [e for e in _audit_events(store.root) if e["event"] == "demote"]
    assert [e["record_id"] for e in demotes] == ["prop-b"]
    for event in demotes:
        assert event["pass_id"] == pass_id


def test_each_pass_mints_its_own_identity_and_manifest(
    store: Engine, two_candidates, record_factory
) -> None:
    first = run_promotion_pass(store, _gate(store))
    first_id = first[0].pass_id
    assert first_id is not None

    record_c = store.propose(record_factory("prop-c"))
    _queue(store, record_c, ["cold-1", "warm-1"])
    second = run_promotion_pass(store, _gate(store))

    second_ids = {d.pass_id for d in second}
    assert second_ids == {second[0].pass_id}
    assert second[0].pass_id != first_id
    assert (store.root / "passes" / f"{second[0].pass_id}.json").exists()
    # The first manifest is untouched: pass artifacts are immutable history.
    assert (store.root / "passes" / f"{first_id}.json").exists()


# ------------------------------------------------------------- review latency


def test_promotion_pass_stamps_review_latency_from_existing_clocks(
    store: Engine, two_candidates
) -> None:
    decisions = run_promotion_pass(store, _gate(store))

    for d in decisions:
        assert d.review_latency_seconds is not None
        assert d.review_latency_seconds >= 0.0


def test_manifest_latency_matches_the_returned_decisions(
    store: Engine, two_candidates
) -> None:
    decisions = run_promotion_pass(store, _gate(store))

    manifest = json.loads(
        (store.root / "passes" / f"{decisions[0].pass_id}.json").read_text(
            encoding="utf-8"
        )
    )
    latency = {
        entry["proposal_id"]: entry["review_latency_seconds"]
        for entry in manifest["decisions"]
    }
    assert latency == {
        d.proposal_id: d.review_latency_seconds for d in decisions
    }


def test_review_outside_a_pass_leaves_latency_stamped_and_pass_id_none(
    store: Engine, record_factory
) -> None:
    """A direct gate review is still latency-attributed but pass-unattributed."""
    record = store.propose(record_factory("prop-a"))
    queued = _queue(store, record, ["cold-1", "warm-1"])

    decision = _gate(store).review(queued, ["cold-1", "warm-1"])

    assert decision.pass_id is None
    assert decision.review_latency_seconds is not None


# -------------------------------------------------------- operator identities


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("nihanth", "nihanth"),
        ("  nihanth  ", "nihanth"),
        ("Nihanth Naidu Kalisetti", "Nihanth Naidu Kalisetti"),
        ("nihanthnaidu008@gmail.com", "nihanthnaidu008@gmail.com"),
        ("worker-1", "worker-1"),
        ("o'connor", "o'connor"),
    ],
)
def test_operator_identity_accepts_handles_names_and_emails(
    raw: str, expected: str
) -> None:
    assert operator_identity(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "bad\nnewline",
        "bad\ttab",
        "a" * 65,
        "<script>",
        'quote"inject',
        "semicolon;",
    ],
)
def test_operator_identity_refuses_blank_and_unprintable_identities(raw: str) -> None:
    with pytest.raises(ApprovalError):
        operator_identity(raw)


def test_record_approval_stores_the_stripped_identity(
    store: Engine, record_factory
) -> None:
    record = store.propose(record_factory("prop-a"))
    queued = _queue(store, record, ["cold-1", "warm-1"])
    log = ApprovalLog(store.root)
    _gate(store).review(queued, ["cold-1", "warm-1"])

    log.record_approval("appr-prop-a", decided_by="  nihanth  ")

    assert log.get("appr-prop-a").decided_by == "nihanth"


def test_record_approval_refuses_an_invalid_identity_and_stays_pending(
    store: Engine, record_factory
) -> None:
    """A refused identity leaves the least-privileged state: still pending."""
    record = store.propose(record_factory("prop-a"))
    queued = _queue(store, record, ["cold-1", "warm-1"])
    log = ApprovalLog(store.root)
    _gate(store).review(queued, ["cold-1", "warm-1"])

    with pytest.raises(ApprovalError):
        log.record_approval("appr-prop-a", decided_by="bad\nidentity")

    assert log.get("appr-prop-a").state is ApprovalState.PENDING


def test_legacy_decision_files_stay_readable(store: Engine, record_factory) -> None:
    """The load path never validates: an old free-form identity round-trips."""
    record = store.propose(record_factory("prop-a"))
    queued = _queue(store, record, ["cold-1", "warm-1"])
    log = ApprovalLog(store.root)
    _gate(store).review(queued, ["cold-1", "warm-1"])

    # Simulate a pre-V3 file on disk: a legacy free-form operator string.
    path = log.approvals_dir / "appr-prop-a.json"
    proposal = json.loads(path.read_text(encoding="utf-8"))
    proposal["decided_by"] = "legacy free-form operator <old>"
    path.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")

    loaded = log.get("appr-prop-a")
    assert loaded.decided_by == "legacy free-form operator <old>"


def test_pass_identity_error_is_a_telemetry_error(
    store: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drifted generator must never reach a path: the pass refuses itself."""
    from evogine.gates import memory_gate as gate_module

    monkeypatch.setattr(gate_module, "ID_PATTERN", re.compile(r"^NEVER-MATCHES$"))

    with pytest.raises(gate_module.TelemetryError):
        run_promotion_pass(store, _gate(store))
