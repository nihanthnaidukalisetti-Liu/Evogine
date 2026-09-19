"""Query API over real store fixtures — offline.

Fixtures build an actual store through the engine's own paths (propose →
replay → gate review → human approval), plus shadow and spend artifacts
through the benchmark run directories, then read them back through the
daemon's query endpoints. The endpoint shapes are the console's own view
builders, so these tests pin the shared contract, not a private dialect.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

# The daemon is optional transport behind the [server] extra: the core CI
# jobs never install it, so this module skips as a whole there (the console
# precedent) and runs for real in the dedicated sync CI job. The imports
# after the skip call are module-level-late by design.
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evogine.bench.budgets import SpendReconciliation
from evogine.bench.harness import RunSummary
from evogine.bench.probes import ProbeReport
from evogine.bench.rundirs import RunDirectory
from evogine.bench.shadow import ShadowReport
from evogine.engine.store import Engine
from evogine.gates.approvals import ApprovalLog, approve_proposal
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import DecisionLog, RunsEvidenceSink
from evogine.sync.app import create_app
from evogine.sync.settings import SyncSettings


def binary_run(run_id: str, arm: str, outcomes: list[bool]) -> ReplayRun:
    return ReplayRun(
        run_id=run_id,
        arm=arm,  # type: ignore[arg-type]  # test fixture pins the literal
        verdicts=[
            Verdict(instance_id=f"case-{i}", passed=ok) for i, ok in enumerate(outcomes)
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


RUNNER = StubRunner(
    {
        "cold-a": binary_run("cold-a", "baseline", [False, False, False]),
        "warm-a": binary_run("warm-a", "warm", [True, True, True]),
        "cold-b": binary_run("cold-b", "baseline", [False, False]),
        "warm-b": binary_run("warm-b", "warm", [True, True]),
    }
)


def shadow_report() -> ShadowReport:
    return ShadowReport(
        candidate_id="cand-1",
        k=3,
        floor=0.7,
        overlay_record_ids=["prop-1"],
        contamination=ProbeReport(
            probe="recall", probed_cases=0, findings=[], contaminated=False
        ),
        replays=0,
        projected_cost_usd=Decimal("0.02"),
        ceiling_usd=Decimal("25.00"),
        summary=RunSummary(
            repeats=1,
            replays=0,
            cold_pass_rate=0.0,
            warm_pass_rate=0.0,
            mean_delta_pass_rate=0.0,
            regressions=[],
            improvements=[],
        ),
        cases=[],
        run_ref="runs/shadow/cand-1",
    )


def spend_report() -> SpendReconciliation:
    return SpendReconciliation(
        recorded_calls=3,
        calls_with_usage=2,
        calls_without_usage=1,
        actual_cost_usd=Decimal("0.12"),
        ceiling_usd=Decimal("25.00"),
        within_ceiling=True,
        overage_usd=Decimal("0.00"),
    )


def seeded_store(stores_root: Path, make_draft) -> Path:  # type: ignore[no-untyped-def]
    """A real store with one applied proposal and one still pending."""
    root = stores_root / "alpha"
    store = Engine(root)
    approvals = ApprovalLog(root)
    log = DecisionLog(root)

    for record_id, cold, warm in (("prop-1", "a", "a"), ("prop-2", "b", "b")):
        store.propose(make_draft(record_id))
        store.queue_replay(record_id, [f"cold-{cold}", f"warm-{warm}"])
        MemoryGate(RUNNER, log, approvals, evidence=RunsEvidenceSink(root)).review(
            store.get(record_id), [f"cold-{cold}", f"warm-{warm}"]
        )

    approve_proposal(
        store,
        approvals,
        "appr-prop-1",
        decided_by="nihanth",
        reason="replays convince me",
    )
    return root


@pytest.fixture()
def client(tmp_path: Path, record_factory):  # type: ignore[no-untyped-def]
    stores_root = tmp_path / "stores"
    root = seeded_store(stores_root, record_factory)
    RunDirectory(root / "runs" / "shadow", "cand-1").write_artifact(
        "report", shadow_report().model_dump_json()
    )
    RunDirectory(root / "runs" / "bench", "bench-1").write_artifact(
        "spend-report", spend_report().model_dump_json()
    )
    app = create_app(SyncSettings(stores_root=stores_root))
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc, root


def test_health_reports_hosted_stores_and_bind_policy(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["stores"] == ["alpha"]
    assert body["bind_policy"] == {
        "host": "127.0.0.1",
        "expose_lan": False,
        "mutations": "loopback-only regardless of any flag",
    }


def test_stores_list_endpoint(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores")
    assert res.status_code == 200
    assert res.json() == {"stores": ["alpha"]}


def test_overview_aggregates(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/overview")
    assert res.status_code == 200
    body = res.json()
    assert body["total_decisions"] == 2
    assert body["pending_approvals"] == 1
    # ``promoted`` counts gate decisions that themselves promoted knowledge —
    # the pre-V2-6 path. In the human-approval flow the gate only renders a
    # proposal; the promotion itself is an approvals-log fact.
    assert body["promoted"] == 0
    # Every cleared review records its final stage as APPROVAL (the gate
    # rendered a proposal; the promotion is the human's next step).
    assert body["by_stage"]["approval"] == 2


def test_records_list_and_status_filter(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    records = tc.get("/api/stores/alpha/records").json()
    ids = {r["id"] for r in records}
    assert {"prop-1", "prop-2"} <= ids
    promoted = tc.get("/api/stores/alpha/records", params={"status": "promoted"})
    assert [r["id"] for r in promoted.json()] == ["prop-1"]
    refused = tc.get("/api/stores/alpha/records", params={"status": "bogus"})
    assert refused.status_code == 400
    assert "bogus" in refused.json()["error"]


def test_record_detail_and_refusals(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/records/prop-1")
    assert res.status_code == 200
    assert res.json()["id"] == "prop-1"
    assert res.json()["status"] == "promoted"
    unknown = tc.get("/api/stores/alpha/records/lr-does-not-exist")
    assert unknown.status_code == 404
    assert "unknown record" in unknown.json()["error"]
    malformed = tc.get("/api/stores/alpha/records/not a record id")
    assert malformed.status_code == 400


def test_decisions_endpoint_exposes_stage_and_evidence(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/decisions")
    assert res.status_code == 200
    decisions = res.json()
    assert len(decisions) == 2
    # The decision log keys rows by learning id; the "appr-" prefix belongs
    # to the approvals log's proposals.
    assert {d["proposal_id"] for d in decisions} == {"prop-1", "prop-2"}
    assert {d["learning_id"] for d in decisions} == {"prop-1", "prop-2"}
    # A cleared review's recorded decision is the proposal render: final
    # stage APPROVAL, promoted False — the human applies the promotion.
    assert all(d["stage"] == "approval" for d in decisions)
    evidence_refs = {d["evidence_ref"] for d in decisions}
    assert "runs/gate/prop-1.json" in evidence_refs


def test_approvals_endpoint_lists_pending_first(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/approvals")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 2
    assert rows[0]["state"] == "pending"
    assert rows[0]["proposal_id"] == "appr-prop-2"
    applied = [r for r in rows if r["state"] == "applied"]
    assert [r["proposal_id"] for r in applied] == ["appr-prop-1"]
    assert applied[0]["decided_by"] == "nihanth"


def test_funnel_endpoint(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/funnel")
    assert res.status_code == 200
    body = res.json()
    assert "counts" in body
    assert "stage_rows" in body


def test_evidence_endpoint_lists_gate_payloads(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/evidence")
    assert res.status_code == 200
    by_learning = {row["learning_id"]: row for row in res.json()}
    assert set(by_learning) == {"prop-1", "prop-2"}
    row = by_learning["prop-1"]
    assert row["replay_run_ids"] == ["cold-a", "warm-a"]
    assert len(row["baseline"]) == len(row["warm"]) == 3


def test_shadow_endpoint(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/alpha/shadow")
    assert res.status_code == 200
    outputs = res.json()
    assert [o["candidate_id"] for o in outputs] == ["cand-1"]
    assert outputs[0]["report"]["candidate_id"] == "cand-1"


def test_spend_endpoint_and_honest_absence(  # type: ignore[no-untyped-def]
    client, tmp_path, record_factory
) -> None:
    tc, _ = client
    res = tc.get("/api/stores/alpha/spend")
    assert res.status_code == 200
    body = res.json()
    assert body["recorded_runs"] == 1
    assert body["rows"][0]["run_id"] == "bench-1"
    assert body["rows"][0]["reconciliation"]["recorded_calls"] == 3
    # pydantic v2 serializes Decimal money as JSON strings — exact cents.
    assert Decimal(body["actual_cost_usd"]) == Decimal("0.12")
    assert body["within_ceiling"] is True

    # a second hosted store with no spend evidence at all: honest absence
    empty_root = tmp_path / "stores" / "beta"
    empty_store = Engine(empty_root)
    empty_store.propose(record_factory("lr-seed-1"))
    fresh = tc.get("/api/stores/beta/spend")
    assert fresh.status_code == 200
    assert fresh.json()["recorded_runs"] == 0
    assert fresh.json()["rows"] == []
    assert fresh.json()["actual_cost_usd"] is None


def test_malformed_record_is_a_visible_refusal_naming_the_file(client) -> None:  # type: ignore[no-untyped-def]
    tc, root = client
    (root / "records" / "junk-1.json").write_text("{oops", encoding="utf-8")
    res = tc.get("/api/stores/alpha/records")
    assert res.status_code == 500
    assert "junk-1.json" in res.json()["error"]


def test_unknown_store_is_a_visible_refusal(client) -> None:  # type: ignore[no-untyped-def]
    tc, _ = client
    res = tc.get("/api/stores/nope/records")
    assert res.status_code == 404
    assert "nope" in res.json()["error"]


# --- Visible refusal states for unreadable or misshapen store content ---


def test_non_object_record_file_refuses_visibly(client) -> None:  # type: ignore[no-untyped-def]
    """A record file that parses as JSON but is not an object refuses loudly."""
    tc, root = client
    (root / "records" / "lr-notobj.json").write_text("[]", encoding="utf-8")
    res = tc.get("/api/stores/alpha/records")
    assert res.status_code == 500
    assert "not a JSON object" in res.json()["error"]


def test_schema_violating_record_refuses_visibly(client) -> None:  # type: ignore[no-untyped-def]
    tc, root = client
    (root / "records" / "lr-bad.json").write_text("{}", encoding="utf-8")
    res = tc.get("/api/stores/alpha/records")
    assert res.status_code == 500
    assert "does not match the store schema" in res.json()["error"]


def test_schema_violating_evidence_refuses_visibly(client) -> None:  # type: ignore[no-untyped-def]
    tc, root = client
    (root / "runs" / "gate" / "broken.json").write_text("{}", encoding="utf-8")
    res = tc.get("/api/stores/alpha/evidence")
    assert res.status_code == 500
    assert "does not match the evidence schema" in res.json()["error"]


def test_schema_violating_shadow_report_refuses_visibly(client) -> None:  # type: ignore[no-untyped-def]
    tc, root = client
    shadow = root / "runs" / "shadow" / "shadow-broken" / "artifacts"
    shadow.mkdir(parents=True)
    (shadow / "report.json").write_text("{}", encoding="utf-8")
    res = tc.get("/api/stores/alpha/shadow")
    assert res.status_code == 500
    assert "does not match the shadow schema" in res.json()["error"]


def test_schema_violating_spend_report_refuses_visibly(client) -> None:  # type: ignore[no-untyped-def]
    tc, root = client
    spend = root / "runs" / "bench" / "bench-broken" / "artifacts"
    spend.mkdir(parents=True)
    (spend / "spend-report.json").write_text("{}", encoding="utf-8")
    res = tc.get("/api/stores/alpha/spend")
    assert res.status_code == 500
    assert "does not match the spend schema" in res.json()["error"]


def test_store_without_records_dir_reads_empty(tmp_path: Path, record_factory) -> None:  # type: ignore[no-untyped-def]
    """A store-shaped root with only the audit trail reads records as empty."""
    stores_root = tmp_path / "stores"
    sparse = stores_root / "sparse"
    sparse.mkdir(parents=True)
    (sparse / "audit.jsonl").write_text("", encoding="utf-8")
    app = create_app(SyncSettings(stores_root=stores_root))
    tc = TestClient(app, raise_server_exceptions=False)
    res = tc.get("/api/stores/sparse/records")
    assert res.status_code == 200
    assert res.json() == []
