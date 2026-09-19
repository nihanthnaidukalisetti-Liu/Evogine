"""Tests for the ``evogine run | bench | decisions | approvals`` CLI surface.

The run tests exercise the promotion pass end-to-end through the CLI over
the default corpus with the offline KnowledgeRunner — no live models. The
learning under test anchors its signal to the real retry case's task
context, so retrieval scoring is self-adjusting to corpus wording rather
than tuned to a snapshot of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evogine.bench.cases.base import BenchCase
from evogine.bench.cases.corpus import default_corpus
from evogine.bench.gate_replay import replay_run_ids
from evogine.bench.harness import CaseOutcome, RunArtifacts
from evogine.bench.rundirs import open_run
from evogine.cli import main
from evogine.engine.records import LearningRecord, RecordBody
from evogine.engine.store import Engine
from evogine.gates.telemetry import DecisionLog, DecisionStage


def _retry_case() -> BenchCase:
    return next(case for case in default_corpus() if case.id == "retry-jitter-budget")


def _retry_learning_draft() -> str:
    """A learning whose signal echoes the real retry case's own context."""
    case = _retry_case()
    return json.dumps(
        {
            "type": "failure-pattern",
            "title": "Retry policy for the rate-limited client",
            "body": {
                "signal": case.task_context,
                "failure_mode": ("immediate tight retries amplify a small stall into an outage"),
                "detection": "the retry policy ignores attempt bounds",
                "fix_strategy": ("use jittered exponential backoff with a hard cap on attempts"),
            },
            "confidence": 0.8,
        }
    )


def _propose_candidate(
    root: Path, draft: str, *, run_ids: list[str] | None = None
) -> LearningRecord:
    """Propose the draft's learning and queue replay runs for it."""
    parsed = json.loads(draft)
    record = LearningRecord.sample().model_copy(
        update={
            "title": parsed["title"],
            "body": RecordBody.model_validate(parsed["body"]),
        }
    )
    engine = Engine(str(root))
    engine.propose(record)
    engine.queue_replay(
        record.id,
        run_ids if run_ids is not None else replay_run_ids(record.id),
    )
    return record


def test_run_cli_drives_promotion_pass_to_an_approval_proposal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`evogine run` decides the candidate and renders the approval proposal."""
    root = tmp_path / "store"
    record = _propose_candidate(root, _retry_learning_draft())

    assert main(["run", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert "promotion pass: 1 candidate(s) decided" in out
    decision = DecisionLog(str(root)).all()[0]
    assert decision.learning_id == record.id
    assert decision.stage is DecisionStage.APPROVAL
    assert decision.promoted is False
    # The gate never promotes: the record stays a candidate for a human.
    assert Engine(str(root)).get(record.id).status == "candidate"
    assert "await human approval" in out


def test_run_cli_exits_nonzero_on_gate_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed queued run id fails closed with exit code 2."""
    root = tmp_path / "store"
    record = _propose_candidate(root, _retry_learning_draft(), run_ids=["garbage-run"])

    assert main(["run", "--root", str(root)]) == 2

    decision = DecisionLog(str(root)).all()[0]
    assert decision.stage is DecisionStage.GATE_ERROR
    assert Engine(str(root)).get(record.id).status == "candidate"
    assert "gate-error" in capsys.readouterr().err


def test_run_cli_on_an_empty_store_is_a_clean_noop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No candidates means nothing decided and a zero exit."""
    root = tmp_path / "store"

    assert main(["run", "--root", str(root)]) == 0
    assert "0 candidate(s) decided" in capsys.readouterr().out
    assert DecisionLog(str(root)).all() == []


def test_bench_cli_reports_table_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`evogine bench` reports warm-vs-cold rates over the default corpus."""
    root = tmp_path / "store"

    assert main(["bench", "--root", str(root)]) == 0
    table = capsys.readouterr().out
    assert "cold pass rate: 0.000" in table  # empty store: honest zero baseline
    assert "warm pass rate: 0.000" in table
    assert "retry-jitter-budget" in table

    assert main(["bench", "--root", str(root), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cold_pass_rate"] == 0.0
    assert payload["warm_pass_rate"] == 0.0
    assert {case["id"] for case in payload["cases"]} == {case.id for case in default_corpus()}


def test_bench_cli_refuses_repeats_below_the_replay_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repeats count under the 4-repeat floor exits non-zero with the reason."""
    root = tmp_path / "store"

    assert main(["bench", "--root", str(root), "--repeats", "2"]) == 1
    assert "below the 4-repeat floor" in capsys.readouterr().err


def test_run_cli_persists_per_replay_outcome_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`evogine run` leaves per-replay run dirs: outcomes, descriptor, marker."""
    root = tmp_path / "store"
    _propose_candidate(root, _retry_learning_draft())

    assert main(["run", "--root", str(root)]) == 0
    capsys.readouterr()

    replay_dirs = sorted((root / "runs" / "replay").iterdir())
    assert len(replay_dirs) == 2  # one baseline + one warm replay for the candidate
    for replay_dir in replay_dirs:
        assert (replay_dir / "complete.marker").exists()
        artifacts = replay_dir / "artifacts"
        # 10 cases x 2 arms x 8 repeats per harness run
        assert len(list(artifacts.glob("outcome-*.json"))) == 160
        descriptor = json.loads((artifacts / "replay-run.json").read_text(encoding="utf-8"))
        assert descriptor["replay_run_id"].startswith("replay-")
        assert descriptor["gate_arm"] in ("baseline", "warm")


def test_bench_cli_persists_a_resumable_run_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`evogine bench` persists a complete run dir; completed runs are immutable."""
    root = tmp_path / "store"

    assert main(["bench", "--root", str(root), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    assert (run_dir / "complete.marker").exists()
    assert len(list((run_dir / "artifacts").glob("outcome-*.json"))) == 160

    assert main(["bench", "--root", str(root), "--run-id", payload["run_id"]]) == 1
    assert "immutable" in capsys.readouterr().err


def test_bench_cli_resume_reuses_persisted_outcomes(tmp_path: Path) -> None:
    """An interrupted bench run resumes: the persisted outcome is reused, not clobbered."""
    root = tmp_path / "store"
    run = open_run(root / "runs" / "bench", run_id="bench-resume")
    RunArtifacts(run).save(
        CaseOutcome(
            case_id="retry-jitter-budget",
            arm="cold",
            repeat=0,
            passed=False,
            output="pre-existing",
            knowledge_ids=[],
        )
    )

    assert main(["bench", "--root", str(root), "--run-id", "bench-resume"]) == 0
    assert (run.directory / "complete.marker").exists()
    outcome = json.loads(
        (run.directory / "artifacts" / "outcome-retry-jitter-budget-cold-0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert outcome["output"] == "pre-existing"
