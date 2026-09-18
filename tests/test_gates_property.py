"""Property tests: the approval state machine and gate pipelines (V3 seams).

Two seams V3 growth will press, fuzzed offline:

- **Approval boundary** — proposals move ``pending -> approved -> applied``
  (or ``pending -> rejected``), and the boundary's contract is that no
  interleaving of approvals, rejections, direct demotions, and gate
  re-reviews can produce a promotion without a pending, evidenced
  proposal over a candidate record. Interrupted sequences (a store
  refusal between recording the approval and applying it) must leave the
  least-privileged visible state: an approved-but-unapplied proposal,
  visible in the log, over an untouched candidate store.
- **Gate pipelines** — a replay-infrastructure failure classifies as
  ``GATE_ERROR`` and never as a knowledge verdict (no proposal is
  rendered, the record stays a candidate, and the decision is still
  recorded); protocol validation reports ALL violations in one pass, not
  just the first.

The approval walk drives the public boundary API only; the fuzzed
interleavings pay a git commit per store transition, so the walk pins its
example budget explicitly. The in-memory pipeline properties ride the
Hypothesis profile (dev 30 / ci 200).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from conftest import make_record
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.bench.budgets import ReplayBudgets
from evogine.bench.cases import BenchCase, default_corpus
from evogine.bench.protocol import (
    EvalProtocol,
    ProtocolValidationError,
    fingerprint_corpus,
    pre_run_validate,
)
from evogine.engine.store import Engine, StoreError
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
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink

TOPIC = "429 backoff"


class StubRunner:
    """Deterministic replay runner: run id -> scripted ReplayRun."""

    def __init__(self, runs: dict[str, ReplayRun]) -> None:
        self._runs = runs

    def run(self, run_id: str) -> ReplayRun:
        if run_id not in self._runs:
            raise KeyError(f"unknown replay run: {run_id}")
        return self._runs[run_id]


class ExplodingRunner:
    """A replay executor that dies: infrastructure, not knowledge."""

    def __init__(self, message: str) -> None:
        self._message = message

    def run(self, run_id: str) -> ReplayRun:
        raise RuntimeError(self._message)


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=ok) for i, ok in enumerate(outcomes)
        ],
    )


def cleared_runner() -> StubRunner:
    """Baseline fails everything; warm passes everything — a cleared verdict."""
    return StubRunner(
        {
            "cold-1": binary_run("cold-1", "baseline", [False, False, False]),
            "warm-1": binary_run("warm-1", "warm", [True, True, True]),
        }
    )


def _render_pending(
    root: Path, store: Engine, approvals: ApprovalLog, record_id: str
) -> MemoryGate:
    """A candidate with queued replays, reviewed into a pending proposal."""
    store.propose(make_record(record_id, topic=TOPIC))
    store.queue_replay(record_id, ["cold-1", "warm-1"])
    gate = MemoryGate(
        cleared_runner(), DecisionLog(root), approvals, evidence=RunsEvidenceSink(root)
    )
    gate.review(store.get(record_id), ["cold-1", "warm-1"])
    return gate


# ----------------------------------------------------- the approval boundary


@settings(max_examples=20, deadline=None)  # git commits per op: budget pinned
@given(ops=st.lists(st.sampled_from(["approve", "reject", "demote", "review"]), max_size=6))
def test_no_interleaving_approves_without_pending_evidence_and_candidate(
    ops: list[str],
) -> None:
    """Fuzzed interleavings can never manufacture an un-proven promotion.

    The walk tracks what the boundary actually granted; after every single
    operation, retrieval may only surface records the walk saw approved,
    and a record on disk in ``promoted`` implies a successful approve call
    happened somewhere in the history.
    """
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-fuzz-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-walk"
        _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"

        promoted_now: set[str] = set()
        ever_approved: set[str] = set()

        for op in ops:
            try:
                if op == "approve":
                    approve_proposal(
                        store, approvals, proposal_id, decided_by="nihanth", reason="ok"
                    )
                    promoted_now.add(record_id)
                    ever_approved.add(record_id)
                elif op == "reject":
                    reject_proposal(
                        store, approvals, proposal_id, decided_by="nihanth", reason="no"
                    )
                    promoted_now.discard(record_id)
                elif op == "demote":
                    store.demote(record_id, "direct drop in the walk")
                    promoted_now.discard(record_id)
                elif op == "review":
                    gate = MemoryGate(
                        cleared_runner(),
                        DecisionLog(root),
                        approvals,
                        evidence=RunsEvidenceSink(root),
                    )
                    gate.review(store.get(record_id), ["cold-1", "warm-1"])
            except (ApprovalError, StoreError):
                # Refused transitions raise and leave the visible state alone;
                # the invariants below are re-checked after refusals too.
                pass

            record = store.get(record_id)
            for hit in store.retrieve(TOPIC, k=3):
                assert hit.id in promoted_now, (
                    f"{op}: retrieved {hit.id} which the boundary never approved"
                )
            assert record.status != "promoted" or record_id in ever_approved, (
                f"{op}: {record_id} is promoted with no successful approve call"
            )

        record = store.get(record_id)
        hits = {hit.id for hit in store.retrieve(TOPIC, k=3)}
        assert hits == ({record_id} if record.status == "promoted" else set())
    finally:
        shutil.rmtree(root)


def test_approve_refuses_a_proposal_that_lost_its_evidence() -> None:
    """Defense in depth: no evidence in the proposal file, no approval."""
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-evidence-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-noev"
        _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"

        path = root / "approvals" / f"{proposal_id}.json"
        stripped = json.loads(path.read_text(encoding="utf-8"))
        stripped["evidence_ref"] = None  # a hand-edit / partial write
        path.write_text(json.dumps(stripped), encoding="utf-8")

        with pytest.raises(ApprovalError, match="carries no replay evidence"):
            approve_proposal(store, approvals, proposal_id, decided_by="nihanth")

        assert approvals.get(proposal_id).state is ApprovalState.PENDING
        assert store.get(record_id).status == "candidate"
    finally:
        shutil.rmtree(root)


def test_approve_refuses_when_the_record_lost_candidate() -> None:
    """No approval without a candidate: a direct demotion blocks the apply."""
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-candidate-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-demoted"
        _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"
        store.demote(record_id, "dropped before the human decided")

        with pytest.raises(ApprovalError, match="not candidate"):
            approve_proposal(store, approvals, proposal_id, decided_by="nihanth")

        assert approvals.get(proposal_id).state is ApprovalState.PENDING
        assert store.get(record_id).status == "dropped"
        assert store.retrieve(TOPIC, k=3) == []
    finally:
        shutil.rmtree(root)


def test_interrupted_apply_leaves_the_least_privileged_visible_state() -> None:
    """A store refusal after the approval is recorded: approved, unapplied.

    The crash window between ``record_approval`` and ``record_applied`` is
    the boundary's documented failure mode — the visible state must be the
    least-privileged one: the approval is visible in the log (never silent),
    while the store stays untouched at ``candidate``.
    """
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-crash-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-crash"
        _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"

        # Corrupt the trail's run ids so the store's own promotion invariant
        # (evidence naming every queued run) refuses mid-apply.
        path = root / "approvals" / f"{proposal_id}.json"
        corrupted = json.loads(path.read_text(encoding="utf-8"))
        corrupted["decision"]["replay_run_ids"] = []
        path.write_text(json.dumps(corrupted), encoding="utf-8")

        with pytest.raises(StoreError, match="requires evidence naming"):
            approve_proposal(store, approvals, proposal_id, decided_by="nihanth")

        proposal = approvals.get(proposal_id)
        assert proposal.state is ApprovalState.APPROVED
        assert proposal.decided_by == "nihanth"
        assert store.get(record_id).status == "candidate"
        assert store.retrieve(TOPIC, k=3) == []
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("decide_first", ["approve", "reject"])
def test_decided_proposals_are_final_for_both_decisions(decide_first: str) -> None:
    """After either decision, both approve and reject are refused."""
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-final-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-final"
        _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"

        if decide_first == "approve":
            approve_proposal(store, approvals, proposal_id, decided_by="nihanth")
        else:
            reject_proposal(
                store, approvals, proposal_id, decided_by="nihanth", reason="no"
            )

        decided_state = approvals.get(proposal_id).state
        with pytest.raises(ApprovalError, match="decided proposals are final"):
            approve_proposal(store, approvals, proposal_id, decided_by="nihanth")
        with pytest.raises(ApprovalError, match="decided proposals are final"):
            reject_proposal(
                store, approvals, proposal_id, decided_by="nihanth", reason="no"
            )
        assert approvals.get(proposal_id).state is decided_state
    finally:
        shutil.rmtree(root)


def test_rejection_requires_a_reason() -> None:
    """A blank reason is refused before anything is recorded."""
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-reason-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-reason"
        _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"

        with pytest.raises(ApprovalError, match="rejection requires a reason"):
            reject_proposal(
                store, approvals, proposal_id, decided_by="nihanth", reason="  "
            )

        assert approvals.get(proposal_id).state is ApprovalState.PENDING
        assert store.get(record_id).status == "candidate"
    finally:
        shutil.rmtree(root)


def test_rereview_of_promoted_knowledge_cannot_repromote() -> None:
    """Re-reviewing promoted knowledge never grants a second promotion.

    A fresh review of promoted knowledge can render again (the store's
    candidate re-check is the last line of defense), but approving the
    re-rendered proposal is refused — the record is not a candidate — so
    the boundary never promotes the same knowledge twice.
    """
    root = Path(tempfile.mkdtemp(prefix="evogine-approval-rereview-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-rerev"
        gate = _render_pending(root, store, approvals, record_id)
        proposal_id = f"appr-{record_id}"
        approve_proposal(store, approvals, proposal_id, decided_by="nihanth")

        gate.review(store.get(record_id), ["cold-1", "warm-1"])  # re-review

        with pytest.raises(ApprovalError, match="not candidate"):
            approve_proposal(store, approvals, proposal_id, decided_by="nihanth")
        assert store.get(record_id).status == "promoted"
        assert [hit.id for hit in store.retrieve(TOPIC, k=3)] == [record_id]
    finally:
        shutil.rmtree(root)


# ------------------------------------------------------------- gate pipelines


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(
    mode=st.sampled_from(
        ["crash", "unknown_run", "missing_warm", "instance_mismatch", "no_runs"]
    ),
    message=st.sampled_from(
        ["replay executor crashed", "OOM at step 3", "segfault in the harness"]
    ),
)
def test_replay_infrastructure_failures_are_gate_errors_never_knowledge_verdicts(
    mode: str, message: str
) -> None:
    """Any replay-infra failure classifies GATE_ERROR with zero side effects.

    Whatever breaks — a crashing executor, an unknown run id, a missing
    warm arm, arms covering different case instances, an empty schedule —
    the verdict is never a knowledge stage, no proposal is rendered, the
    record stays a candidate, and the decision is still recorded (the V2-1
    completeness invariant holds for errors too).
    """
    root = Path(tempfile.mkdtemp(prefix="evogine-gate-error-"))
    try:
        store = Engine(root)
        approvals = ApprovalLog(root)
        record_id = "prop-gate-error"
        store.propose(make_record(record_id, topic=TOPIC))
        store.queue_replay(record_id, ["cold-1", "warm-1"])

        if mode == "crash":
            runner: StubRunner | ExplodingRunner = ExplodingRunner(message)
            run_ids = ["cold-1", "warm-1"]
        elif mode == "unknown_run":
            runner = StubRunner(
                {"cold-1": binary_run("cold-1", "baseline", [False, False, False])}
            )
            run_ids = ["cold-1", "missing-warm"]
        elif mode == "missing_warm":
            runner = StubRunner(
                {"cold-1": binary_run("cold-1", "baseline", [False, False, False])}
            )
            run_ids = ["cold-1"]
        elif mode == "instance_mismatch":
            runner = StubRunner(
                {
                    "cold-1": ReplayRun(
                        run_id="cold-1",
                        arm="baseline",
                        verdicts=[Verdict(instance_id="case-a", passed=False)],
                    ),
                    "warm-1": ReplayRun(
                        run_id="warm-1",
                        arm="warm",
                        verdicts=[Verdict(instance_id="case-b", passed=True)],
                    ),
                }
            )
            run_ids = ["cold-1", "warm-1"]
        else:  # no_runs
            runner = StubRunner({})
            run_ids = []

        gate = MemoryGate(
            runner,
            DecisionLog(root),
            approvals,
            evidence=RunsEvidenceSink(root),
        )
        decision = gate.review(store.get(record_id), run_ids)

        assert decision.promoted is False
        assert decision.stage is DecisionStage.GATE_ERROR
        assert decision.reason.startswith("gate_error:")
        # An infrastructure failure renders nothing to approve...
        assert approvals.all() == []
        # ...is recorded exactly once (V2-1 completeness, errors included)...
        recorded = DecisionLog(root).all()
        assert len(recorded) == 1
        assert recorded[0].stage is DecisionStage.GATE_ERROR
        # ...and says nothing about the knowledge: the record stays a
        # candidate for a retry with fixed infrastructure.
        assert store.get(record_id).status == "candidate"
        assert store.retrieve(TOPIC, k=3) == []
    finally:
        shutil.rmtree(root)


# --------------------------------------------- protocol validation: one pass


def _budgets(per_run: int) -> ReplayBudgets:
    return ReplayBudgets(
        max_per_proposal=999, max_per_case_family=999, max_per_run=per_run
    )


def _protocol(corpus: list[BenchCase], **overrides: object) -> EvalProtocol:
    fields: dict[str, object] = {
        "name": "property-fuzz protocol",
        "corpus_fingerprint": fingerprint_corpus(corpus),
        "prompt_pack": "pack-2026-09-18",
        "repeats": 4,
        "budgets": _budgets(999),
    }
    fields.update(overrides)
    return EvalProtocol.model_validate(fields)


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(
    fingerprint_broken=st.booleans(),
    duplicate_corpus=st.booleans(),
    selection=st.sampled_from(["none", "valid", "duplicate", "unknown"]),
    heldout_broken=st.booleans(),
    budget_broken=st.booleans(),
)
def test_protocol_validation_reports_every_violation_in_one_pass(
    fingerprint_broken: bool,
    duplicate_corpus: bool,
    selection: str,
    heldout_broken: bool,
    budget_broken: bool,
) -> None:
    """A rejecting protocol names EVERYTHING wrong with it — not the first.

    The expected violation set is computed from the same injections the
    test makes (mirroring the validator's resolution order), then every
    expected marker must appear in the raised error's violation list; with
    nothing broken, validation must pass.
    """
    corpus = list(default_corpus())
    if duplicate_corpus:
        corpus.append(corpus[0].model_copy())  # a second case with one id

    case_ids: list[str] | None = None
    if selection == "valid":
        case_ids = [corpus[0].id]
    elif selection == "duplicate":
        case_ids = [corpus[0].id, corpus[0].id]
    elif selection == "unknown":
        case_ids = ["no-such-case"]

    protocol = _protocol(
        corpus,
        corpus_fingerprint=(
            "sha256:deadbeef" if fingerprint_broken else fingerprint_corpus(corpus)
        ),
        case_ids=case_ids,
        held_out_family="no-such-family" if heldout_broken else None,
        budgets=_budgets(1 if budget_broken else 999),
    )

    expected: list[str] = []
    resolvable = True
    if duplicate_corpus:
        expected.append("duplicate case ids in corpus")
        resolvable = False
    if fingerprint_broken:
        expected.append("corpus fingerprint mismatch")
        resolvable = False
    if selection == "duplicate":
        expected.append("duplicate case ids in selection")
        resolvable = False
    if selection == "unknown":
        expected.append("unknown case ids")
        resolvable = False
    if heldout_broken:
        expected.append("does not exist in the corpus")
    if resolvable and budget_broken:
        expected.append("declared budgets refuse")

    if not expected:
        check = pre_run_validate(protocol, corpus)
        assert check.total_replays == len(check.cases) * 2 * 4
        return

    with pytest.raises(ProtocolValidationError) as excinfo:
        pre_run_validate(protocol, corpus)
    violations = excinfo.value.violations
    for marker in expected:
        assert any(marker in violation for violation in violations), (
            f"missing violation {marker!r} in a one-pass report: {violations}"
        )
    assert len(violations) >= len(expected)
