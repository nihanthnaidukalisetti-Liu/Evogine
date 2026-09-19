"""Funnel-panel tests: CLI parity, honest absence, visible refusals.

The acceptance for the console funnel panel: the page shows the same numbers
``evogine decisions funnel`` prints — the panel computes through
:func:`evogine.gates.funnel.build_funnel_report` (the CLI's own engine), so
the parity test compares the view and the page against the CLI's JSON report
on one real store, with hand-computed literals alongside (the arithmetic is
commented in the fixture). The empty-state tests prove absence renders
honestly (zeros as zeros, "no passes recorded", "none found"), and the
refusal tests prove store corruption shows the error page instead of a blind
500 or a half-empty table. All offline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

# The console is optional transport behind the [console] extra: the core CI
# jobs never install it, so this module skips as a whole there (the
# test_console.py precedent) and runs for real in the dedicated console job.
# The imports after the skip call are module-level-late by design; E402 is
# ignored for this file in pyproject.
pytest.importorskip("fastapi")

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from evogine.cli import main as cli_main
from evogine.console.app import create_app
from evogine.console.funnel import _fmt, build_funnel_view
from evogine.console.launch import ConsoleError
from evogine.engine.store import Engine
from evogine.gates.approvals import ApprovalLog, approve_proposal
from evogine.gates.materiality import Verdict
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.replay import ReplayRun
from evogine.gates.telemetry import (
    DecisionLog,
    PassManifest,
    RunsEvidenceSink,
)
from evogine.gates.trajectory import TrajectoryStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from evogine.engine.records import LearningRecord

OPERATOR = "nihanth@localhost"

#: The orphan pass id on the legacy decision: no manifest exists for it, so
#: it is the crash-window gap the report must show, never absorb.
LEGACY_PASS_ID = "pass-20260919T030303-cccccc03"


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
    """Five reviews, one per gate outcome the decision log distinguishes."""
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


@pytest.fixture
def funnel_store(
    root: Path, store: Engine, record_factory: Callable[[str], LearningRecord]
) -> Path:
    """One real promotion pass over five candidates, then a human approval.

    Built through the production flow — propose, queue replay, one
    ``run_promotion_pass`` (which stamps pass identity and writes the
    manifest), one ``approve_proposal`` — so every file the funnel reads is
    one the gate actually writes. Hand computation (this table IS the
    arithmetic; every expected literal below derives from it):

        proposal | stage       | label          | cleared replay? materiality?
        ---------+-------------+----------------+-----------------------------
        prop-1   | approval    | proposed       | yes             yes
        prop-2   | replay      | no_flip        | no              no
        prop-3   | materiality | not_material   | yes             no
        prop-4   | never_worse | regression     | yes             yes
        prop-5   | approval    | proposed       | yes             yes

        + prop-legacy (copy of prop-1's decision, orphan pass id).

        candidates          = 6   (five pass decisions + one legacy)
        replay_cleared      = 5   (prop-1, prop-3, prop-4, prop-5, prop-legacy —
                                   a materiality-stage decision cleared replay,
                                   then failed materiality)
        materiality_cleared = 4   (same five, minus prop-3)
        approved            = 1   (prop-1 applied by the fixture)
        promoted            = 1   (prop-1's record)
        gate_errors         = 0
        pending / rejected  = 1 (prop-5) / 0
        decisions_without_manifest = 1 (the legacy orphan pass id)
    """
    gate = MemoryGate(
        runner=scripted_runner(),
        log=DecisionLog(root),
        approvals=ApprovalLog(root),
        evidence=RunsEvidenceSink(root),
        trajectories=TrajectoryStore(root),
    )
    for name in ("prop-1", "prop-2", "prop-3", "prop-4", "prop-5"):
        topic = "merge conflicts" if name == "prop-2" else "flaky retry loop"
        store.queue_replay(
            store.propose(record_factory(name, topic=topic)).id,
            [f"cold-{name[-1]}", f"warm-{name[-1]}"],
        )
    run_promotion_pass(store, gate)
    approve_proposal(
        store, ApprovalLog(root), "appr-prop-1", decided_by="nihanth", reason="convincing"
    )

    # A decision stamped with a pass id that has no manifest: the pass's
    # last decision landed before the manifest write. Copied from a real
    # decision so its reason stays inside the machine-checked vocabulary.
    real = DecisionLog(root).all()[0]
    legacy = real.model_copy(
        update={
            "proposal_id": "prop-legacy",
            "learning_id": "prop-legacy",
            "pass_id": LEGACY_PASS_ID,
        }
    )
    (root / "decisions" / "prop-legacy.json").write_text(
        legacy.model_dump_json(), encoding="utf-8"
    )

    # Live token evidence under runs/: one ok call (model-a, 10 in / 5 out),
    # one failed call (no model, no tokens — spent nothing billable), and
    # one file that matches the artifact naming but is unreadable.
    run_dir = root / "runs" / "live-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "call-a-ok.json").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "arm": "warm",
                "repeat": 1,
                "call_index": 1,
                "prompt_pack_fingerprint": "fp-1",
                "ok": True,
                "model": "model-a",
                "input_tokens": 10,
                "output_tokens": 5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "call-b-failed.json").write_text(
        json.dumps(
            {
                "case_id": "case-b",
                "arm": "warm",
                "repeat": 1,
                "call_index": 2,
                "prompt_pack_fingerprint": "fp-1",
                "ok": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "call-c-broken.json").write_text("{not json", encoding="utf-8")
    return root


@pytest.fixture
def client(funnel_store: Path) -> TestClient:
    return TestClient(create_app(funnel_store, OPERATOR), raise_server_exceptions=False)


# --------------------------------------------------------------------- parity


def test_panel_matches_the_cli_report(
    funnel_store: Path, client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The view, the page, and ``evogine decisions funnel --format json`` agree."""
    exit_code = cli_main(
        ["decisions", "funnel", "--root", str(funnel_store), "--format", "json"]
    )
    assert exit_code == 0
    cli_report = json.loads(capsys.readouterr().out)

    view = build_funnel_view(funnel_store)

    # Field-for-field parity with the CLI report: same engine, same store.
    assert view.counts.model_dump() == cli_report["funnel"]
    assert {(row.stage, row.label): row.count for row in view.reasons} == {
        (stage, label): count
        for stage, labels in cli_report["reasons_by_stage"].items()
        for label, count in labels.items()
    }
    assert [row.pass_id for row in view.passes] == [
        cli_pass["pass_id"] for cli_pass in cli_report["passes"]
    ]
    for view_pass, cli_pass in zip(view.passes, cli_report["passes"], strict=True):
        assert view_pass.candidates == cli_pass["candidates"]
        assert view_pass.decisions == cli_pass["decisions"]
        assert view_pass.replay_cleared == cli_pass["replay_cleared"]
        assert view_pass.materiality_cleared == cli_pass["materiality_cleared"]
        assert view_pass.approved == cli_pass["approved"]
        assert view_pass.promoted == cli_pass["promoted"]
        assert view_pass.gate_errors == cli_pass["gate_errors"]
    assert view.decisions_without_manifest == cli_report["decisions_without_manifest"]
    assert (view.live_tokens.model_dump() if view.live_tokens else None) == cli_report[
        "live_tokens"
    ]

    # Hand-computed literals (the arithmetic lives in the fixture docstring).
    counts = view.counts
    assert counts.candidates == 6
    assert counts.replay_cleared == 5
    assert counts.materiality_cleared == 4
    assert counts.approved == 1
    assert counts.promoted == 1
    assert counts.gate_errors == 0
    assert counts.proposals_pending == 1
    assert counts.proposals_rejected == 0
    assert view.decisions_without_manifest == 1


def test_funnel_page_renders_stages_reasons_and_passes(
    funnel_store: Path, client: TestClient
) -> None:
    """The page carries the nav link, every stage row, and the zero-filled
    vocabulary; absence lines (crash window, unreadable records) render."""
    response = client.get("/funnel")
    assert response.status_code == 200
    assert 'href="/funnel"' in response.text  # discoverable from every page
    for stage in (
        "candidates",
        "replay-cleared",
        "materiality-cleared",
        "approved",
        "promoted",
        "gate errors",
        "proposals pending",
        "proposals rejected",
    ):
        assert stage in response.text
    for label in (
        "no_flip",
        "not_material",
        "regression",
        "cleared",
        "proposed",
        "refused_no_evidence",
        "gate_error",
    ):
        assert label in response.text  # zero rows stay visible
    assert "<strong>6</strong>" in response.text  # the candidates cell
    assert "decisions without a finished pass manifest: 1" in response.text
    assert "1 ok, 1 failed" in response.text  # live evidence: honest counts
    assert "10 in / 5 out" in response.text
    assert "model-a" in response.text
    assert "unreadable call records (shown, not hidden): 1" in response.text
    assert "pass-" in response.text  # the pass id minted by the real pass


# --------------------------------------------------------------- honest absence


def test_funnel_page_is_honestly_empty(root: Path) -> None:
    """An empty store renders zero counts, no passes, and no live evidence —
    nothing fabricated into activity."""
    root.mkdir()
    empty = TestClient(create_app(root, OPERATOR), raise_server_exceptions=False)
    response = empty.get("/funnel")
    assert response.status_code == 200
    assert "<strong>0</strong>" in response.text
    assert "No passes recorded" in response.text
    assert "none found" in response.text


def test_timestamp_formatter_renders_absence_as_none() -> None:
    assert _fmt(None) is None
    formatted = _fmt(datetime(2026, 9, 19, 3, 4, 5, tzinfo=UTC))
    assert formatted == "2026-09-19T03:04:05+00:00"


def test_hostile_pass_id_is_escaped(funnel_store: Path) -> None:
    """A hand-crafted manifest's pass id is store data: escaped, never HTML."""
    manifest = PassManifest(
        pass_id="<script>alert(1)</script>",
        started_at=datetime(2026, 9, 19, 3, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 19, 3, 5, 0, tzinfo=UTC),
        candidates=[],
        decisions=[],
    )
    (funnel_store / "passes" / "pass-evil.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    page = TestClient(create_app(funnel_store, OPERATOR)).get("/funnel")
    assert "&lt;script&gt;" in page.text
    assert "<script>alert(1)" not in page.text


# ------------------------------------------------------------------ refusals


def test_malformed_decision_file_refuses_the_panel(
    funnel_store: Path, client: TestClient
) -> None:
    (funnel_store / "decisions" / "prop-junk.json").write_text(
        "{not json at all", encoding="utf-8"
    )
    response = client.get("/funnel")
    assert response.status_code == 500
    assert "validation error" in response.text  # the defect message is shown


def test_missing_root_refuses_as_console_error(tmp_path: Path) -> None:
    """The funnel engine's TelemetryError surfaces in the console's own
    refusal vocabulary, so every panel error path is one shape."""
    with pytest.raises(ConsoleError, match="does not exist"):
        build_funnel_view(tmp_path / "nope")
