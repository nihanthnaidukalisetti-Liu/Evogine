"""Separation construction tests: the V2-8 anti-gaming invariants.

Hidden benchmark oracles and gate-visible tests are different registries
by construction, and the promotion proxy is never the evaluation
verifier. The fixtures assert these rules against the REAL corpus — the
10-case decay-pressure set with its hidden invariants — and against the
locked judge identities from the V2 spec, so a regression that collapses
either separation fails loudly instead of silently.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from evogine.bench.cases.corpus import default_corpus
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.gates.approvals import ApprovalLog
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.replay import ReplayRun
from evogine.gates.separation import (
    SeparationError,
    assert_proxy_is_not_verifier,
    assert_test_registries_disjoint,
)
from evogine.gates.telemetry import DecisionLog, DecisionStage, RunsEvidenceSink
from evogine.gates.trajectory import TrajectoryStore

# Locked decision from the V2 spec, pinned here as the production fixture:
# two different judges with two different names. The promotion proxy is
# the V2-1 replay materiality never-worse check; the evaluation verifier
# is the V2-7 scored benchmark protocol graded by hidden host invariants.
PROMOTION_PROXY = "replay never-worse materiality check"
EVALUATION_VERIFIER = "pre-registered benchmark protocol with hidden invariants"

# Gate-visible id namespaces (construction, not convention):
GATE_SANITY_PREFIX = "gold-sanity:"


def test_real_corpus_and_gate_registries_are_disjoint():
    """The V2-8 acceptance: production fixture asserts registry separation."""
    hidden_ids = [case.id for case in default_corpus()]
    # The ids a gate can see or admit: gold-sanity admission case ids
    # (namespaced) and replay run ids. None may collide with a bench id.
    gate_visible_ids = [
        f"{GATE_SANITY_PREFIX}reflector-check-1",
        f"{GATE_SANITY_PREFIX}reflector-check-2",
        "run-cold",
        "run-warm",
    ]
    assert_test_registries_disjoint(gate_visible_ids, hidden_ids)


def test_gold_sanity_namespacing_keeps_corpus_ids_out():
    """Admission case ids are namespaced away from the benchmark corpus."""
    hidden_ids = [case.id for case in default_corpus()]
    # The corpus itself is not namespaced into the gate's namespace, so a
    # gold-sanity admission case can never shadow a benchmark case id.
    assert all(not case_id.startswith(GATE_SANITY_PREFIX) for case_id in hidden_ids)


def test_collapsing_registries_is_a_construction_error():
    """An id shared by both registries is refused, naming the overlap."""
    with pytest.raises(SeparationError, match="overlapping ids.*leak-probe"):
        assert_test_registries_disjoint(
            ["run-cold", "leak-probe"], ["leak-probe", "tz-offset"]
        )


def test_fixture_fixes_proxy_and_verifier_apart():
    """The V2-8 acceptance: the proxy/verifier identities are asserted apart."""
    # The fixture's own premise — a tautological equality would make the
    # invariant test meaningless.
    assert PROMOTION_PROXY != EVALUATION_VERIFIER
    assert_proxy_is_not_verifier(PROMOTION_PROXY, EVALUATION_VERIFIER)


def test_proxy_equal_to_verifier_is_a_construction_error():
    """A promotion proxy that equals the evaluation verifier is refused."""
    with pytest.raises(SeparationError, match="must never equal"):
        assert_proxy_is_not_verifier("one-judge", "one-judge")


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=o) for i, o in enumerate(outcomes)
        ],
    )


def separation_probe() -> LearningRecord:
    return LearningRecord(
        id="separation-probe",
        type="failure-pattern",
        title="Separation probe",
        body=RecordBody(signal="s", failure_mode="f", detection="d", fix_strategy="x"),
        status="candidate",
        confidence=0.5,
        provenance=RecordProvenance(
            proposed_by="reflector", origin="test", source="unit-test", note=""
        ),
        counterfactual=RecordCounterfactual(replay_run_ids=["run-cold", "run-warm"]),
        created=datetime(2026, 9, 18, tzinfo=UTC),
    )


def test_real_gate_decisions_carry_proxy_not_verifier(tmp_path: Path):
    """End-to-end: a real promotion decision cites the proxy judge, never the
    evaluation verifier — the two stay apart in shipped artifacts, not just
    in the fixture constants."""

    class ScriptedRunner:
        def run(self, run_id: str) -> ReplayRun:
            # Cold arm fails everything, warm arm passes everything — the
            # material flip the never-worse proxy needs to clear the record.
            cold = run_id == "run-cold"
            return binary_run(
                run_id,
                "baseline" if cold else "warm",
                [False, False] if cold else [True, True],
            )

    class OneProbeStore:
        """Minimal store double over one candidate learning record."""

        def __init__(self) -> None:
            self.record = separation_probe()
            self.promoted: list[str] = []
            self.demoted: list[tuple[str, str]] = []

        def candidates(self) -> list[LearningRecord]:
            return [self.record]

        def promote(self, record_id: str, counterfactual: object) -> None:
            self.promoted.append(record_id)

        def demote(self, record_id: str, reason: str) -> None:
            self.demoted.append((record_id, reason))

    store = OneProbeStore()
    gate = MemoryGate(
        ScriptedRunner(),  # type: ignore[arg-type]
        DecisionLog(tmp_path),
        approvals=ApprovalLog(tmp_path),
        evidence=RunsEvidenceSink(tmp_path),
        trajectories=TrajectoryStore(tmp_path),
    )
    decisions = run_promotion_pass(store, gate)  # type: ignore[arg-type]

    # V2-6: the cleared verdict renders a pending proposal — the pass
    # promotes nothing and demotes nothing (the human boundary holds).
    assert len(decisions) == 1 and store.promoted == [] and store.demoted == []
    decision = decisions[0]
    # The decision cleared the proxy (replay materiality) — and the
    # verifier identity appears nowhere in the shipped decision payload:
    # the two judges stay apart in artifacts, not just fixture constants.
    assert decision.stage is DecisionStage.APPROVAL
    assert EVALUATION_VERIFIER not in decision.model_dump_json()
