"""Shadow-pass tests: profiling without mutation, refusal before spend.

The V3 W2 non-negotiable is pinned from every direction here:

- Engine state — record files, audit trail, and everything warm retrieval
  can see — is byte-identical across a shadow pass, with results confined
  to ``runs/shadow/<candidate_id>/`` (even when that lives inside the
  store root, the production layout).
- The contamination recall probe and the dollar-ceiling projection both
  refuse before anything is evaluated or written: no run directory, no
  replay executed, $0 spent, non-zero exit.
- Run-dir conventions hold: resume by candidate id reuses recorded
  outcomes; a completed run is immutable evidence.

Offline by contract — the runner is the deterministic KnowledgeRunner and
the CLI never touches a model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from evogine.bench.budgets import DollarBudget, SpendExpectation
from evogine.bench.cases import BenchCase
from evogine.bench.harness import REPEAT_FLOOR, CaseOutcome, RunArtifacts
from evogine.bench.rundirs import RunCompleteError, open_run
from evogine.bench.runner import KnowledgeRunner
from evogine.bench.shadow import (
    RUNS_SUBDIR,
    OverlayRetriever,
    ShadowBudgetRefused,
    ShadowCandidate,
    ShadowContaminationRefused,
    ShadowReport,
    load_shadow_candidate,
    run_shadow_pass,
)
from evogine.cli import main as cli_main
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.store import Engine

HOLDOUT_FAMILY = "fresh-family"
HELDOUT_TOKEN = "xylophonequorum"
"""Content token unique to the held-out case's context: a record quoting it
can only have gotten it from the holdout."""


def _case(case_id: str, family: str, context: str) -> BenchCase:
    """A bench case whose oracle passes on any submission naming the fix."""
    return BenchCase(
        id=case_id,
        family=family,
        task_context=context,
        hidden_invariant=lambda output: "jittered" in output,
    )


def shadow_corpus() -> list[BenchCase]:
    """Three benchmark families plus one held-out case with a unique token."""
    return [
        _case(
            "bench-retry-1",
            "retries",
            "the client hammers the api with tight retries until the rate limiter trips",
        ),
        _case(
            "bench-retry-2",
            "retries",
            "retry storms amplify a small stall into a full outage without backoff",
        ),
        _case(
            "bench-lock-1",
            "concurrency",
            "two writers race on one row and the lost update corrupts the ledger",
        ),
        _case(
            "bench-fresh-1",
            HOLDOUT_FAMILY,
            f"the {HELDOUT_TOKEN} subsystem drops writes when the wal segment rotates",
        ),
    ]


def shadow_holdout() -> list[BenchCase]:
    return [case for case in shadow_corpus() if case.family == HOLDOUT_FAMILY]


def live_record(
    record_id: str,
    *,
    status: str = "promoted",
    topic: str = "jittered backoff",
    leak: bool = False,
) -> LearningRecord:
    """A fully-populated record in any lifecycle state; body varies by topic.

    ``leak=True`` echoes the held-out case's context verbatim in the signal —
    a record that could only exist if something saw the holdout.
    """
    signal = (
        shadow_holdout()[0].task_context
        if leak
        else f"rising {topic} counts across consecutive rollouts"
    )
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=f"Unhandled {topic} escalates across rollouts",
        body=RecordBody(
            signal=signal,
            failure_mode=f"{topic} amplifies without a bounded retry budget",
            detection="counterfactual replay flips the verifier verdict on the degraded case",
            fix_strategy="apply jittered exponential backoff with a per-rollout retry budget",
        ),
        status=status,
        confidence=0.6,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="test-fixture",
            source="unit-test",
            note="deterministic fixture; never touches a live model",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 18, tzinfo=UTC),
    )


def free_budget() -> DollarBudget:
    """A zero-priced expectation: offline replays spend no billable tokens."""
    return DollarBudget(
        expectation=SpendExpectation(
            input_tokens=0,
            output_tokens=0,
            input_price_per_mtok_usd=Decimal(0),
            output_price_per_mtok_usd=Decimal(0),
        ),
        max_run_cost_usd=Decimal("25.00"),
    )


def priced_budget(ceiling: str) -> DollarBudget:
    """A budget with real prices, for exercising the queue-time projection."""
    return DollarBudget(
        expectation=SpendExpectation(
            input_tokens=1000,
            output_tokens=1000,
            input_price_per_mtok_usd=Decimal(1),
            output_price_per_mtok_usd=Decimal(1),
        ),
        max_run_cost_usd=Decimal(ceiling),
    )


class CountingRunner:
    """Delegates to :class:`KnowledgeRunner` while counting executions."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self,
        case: BenchCase,
        *,
        arm: str,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        self.calls += 1
        return KnowledgeRunner().execute(case, arm=arm, knowledge=knowledge, repeat=repeat)


def store_with_lifecycle_states(root: Path) -> Engine:
    """A store holding one record in every lifecycle state, via real calls.

    promoted (live), decayed (promoted, confidence aged past a half-life),
    candidate (replay queued), ephemeral (draft), rejected (ephemeral →
    dropped), demoted (promoted → dropped), merged (absorbed duplicate),
    retired (reversible shelf). Everything enters as an ``ephemeral``
    draft — the lifecycle is store-owned.
    """
    engine = Engine(str(root))

    def promote_record(record_id: str) -> None:
        engine.propose(live_record(record_id, status="ephemeral"))
        engine.queue_replay(record_id, [f"run-{record_id}"])
        engine.promote(
            record_id,
            RecordCounterfactual(
                replay_run_ids=[f"run-{record_id}"],
                verdict_change=True,
                evidence="runs/replay/pair.json",
            ),
        )

    promote_record("lr-live")
    promote_record("lr-decayed")
    engine.decay(now=datetime(2026, 12, 31, tzinfo=UTC))  # created 2026-09-18: >1 half-life
    candidate = live_record("lr-candidate", status="ephemeral")
    engine.propose(candidate)
    engine.queue_replay(candidate.id, ["run-candidate"])
    engine.propose(live_record("lr-ephemeral", status="ephemeral"))
    rejected = live_record("lr-rejected", status="ephemeral")
    engine.propose(rejected)
    engine.demote(rejected.id, "rejected: failed the gate")
    promote_record("lr-demoted")
    engine.demote("lr-demoted", "demoted on re-review")
    keep = live_record("lr-merge-keep", status="ephemeral")
    gone = live_record("lr-merge-gone", status="ephemeral")
    engine.propose(keep)
    engine.propose(gone)
    engine.merge(keep.id, gone.id, "duplicate")
    retired = live_record("lr-retired", status="ephemeral")
    engine.propose(retired)
    engine.retire(retired.id, "stale")
    return engine


def engine_snapshot(root: Path) -> dict[str, bytes]:
    """Byte snapshot of engine state, the promoted pool, and warm retrieval.

    Every file under the store root except ``.git/`` and ``runs/`` (the
    store's own git internals, and the tree a shadow pass owns), plus the
    serialized promoted pool and warm retrieval results for every corpus
    case — the exact surfaces the non-negotiable protects.
    """
    engine = Engine(str(root))
    snapshot: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if relative.parts[0] in {".git", "runs"}:
            continue
        snapshot[relative.as_posix()] = path.read_bytes()
    snapshot["::promoted"] = "\n".join(
        record.model_dump_json() for record in engine.promoted_records()
    ).encode()
    for case in shadow_corpus():
        snapshot[f"::retrieve::{case.id}"] = "\n".join(
            record.model_dump_json() for record in engine.retrieve(case.task_context)
        ).encode()
    return snapshot


def assert_results_confined(root: Path, candidate_id: str) -> None:
    """Every file under ``runs/`` belongs to this candidate's shadow run."""
    runs = root / "runs"
    assert runs.exists(), "a completed shadow pass must create its runs tree"
    for path in runs.rglob("*"):
        relative = path.relative_to(runs).as_posix()
        if relative == RUNS_SUBDIR:  # the shadow root itself, not a result
            continue
        assert relative.startswith(f"{RUNS_SUBDIR}/{candidate_id}"), (
            f"shadow results leaked outside the run dir: {relative}"
        )


def shadow_candidate(**overrides: object) -> ShadowCandidate:
    defaults: dict[str, object] = {
        "candidate_id": "cand-jitter",
        "records": [live_record("lr-shadow-jitter")],
        "budget": free_budget(),
    }
    defaults.update(overrides)
    return ShadowCandidate(**defaults)  # type: ignore[arg-type]


def run_pass(
    engine: Engine,
    candidate: ShadowCandidate,
    *,
    runs_root: Path,
    runner: CountingRunner | None = None,
) -> ShadowReport:
    return run_shadow_pass(
        engine,
        candidate,
        runner or CountingRunner(),
        runs_root=runs_root,
        heldout_cases=shadow_holdout(),
        corpus=shadow_corpus(),
        repeats=REPEAT_FLOOR,
    )


# --------------------------------------------------------------------- the pass


def test_shadow_pass_is_byte_non_mutating_and_confined(tmp_path: Path) -> None:
    """The acceptance core: engine state byte-identical across a full pass."""
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR  # production layout: inside the store
    before = engine_snapshot(store_root)

    report = run_pass(engine, shadow_candidate(), runs_root=runs_root)

    assert engine_snapshot(store_root) == before
    assert report.candidate_id == "cand-jitter"
    assert report.overlay_record_ids == ["lr-shadow-jitter"]
    assert not report.contamination.contaminated
    # 4 cases × 2 arms × 4 repeats
    assert report.summary.replays == 2 * 4 * REPEAT_FLOOR
    assert report.run_ref == f"runs/{RUNS_SUBDIR}/cand-jitter"
    assert_results_confined(store_root, "cand-jitter")
    run_dir = runs_root / "cand-jitter"
    assert (run_dir / "complete.marker").exists()
    persisted = ShadowReport.model_validate_json(
        (run_dir / "artifacts" / "report.json").read_text(encoding="utf-8")
    )
    assert persisted.candidate_id == report.candidate_id


def test_shadow_run_resume_reuses_recorded_outcomes(tmp_path: Path) -> None:
    """Resume by candidate id: a crashed run redoes only the missing replays."""
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR
    crashed = open_run(runs_root, run_id="cand-jitter")
    RunArtifacts(crashed).save(
        CaseOutcome(
            case_id="bench-retry-1",
            arm="cold",
            repeat=0,
            passed=True,
            output="pre-seeded by the crashed process",
            knowledge_ids=[],
        )
    )

    runner = CountingRunner()
    report = run_pass(engine, shadow_candidate(), runs_root=runs_root, runner=runner)

    assert report.summary.replays == 2 * 4 * REPEAT_FLOOR
    assert runner.calls == 2 * 4 * REPEAT_FLOOR - 1
    resumed = crashed.read_artifact("outcome-bench-retry-1-cold-0000")
    assert resumed is not None and "pre-seeded" in resumed
    assert (crashed.directory / "complete.marker").exists()


def test_completed_shadow_run_is_immutable(tmp_path: Path) -> None:
    """A completed shadow run is evidence: reopening it is refused, untouched."""
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR
    run_pass(engine, shadow_candidate(), runs_root=runs_root)
    before = engine_snapshot(store_root)

    with pytest.raises(RunCompleteError, match="complete and immutable"):
        run_pass(engine, shadow_candidate(), runs_root=runs_root)

    assert engine_snapshot(store_root) == before


def test_warm_retrieval_cannot_see_shadow_results(tmp_path: Path) -> None:
    """runs/shadow/ inside the store root is invisible to a fresh warm engine.

    The governance record's warning, pinned: a second engine reading the
    same root after a shadow pass retrieves exactly what the first did —
    the shadow run's files are not records and cannot enter the pool.
    """
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR
    warm_before = {case.id: engine.retrieve(case.task_context) for case in shadow_corpus()}
    pool_before = engine.promoted_records()

    run_pass(engine, shadow_candidate(), runs_root=runs_root)

    fresh = Engine(str(store_root))
    assert [record.id for record in fresh.promoted_records()] == [
        record.id for record in pool_before
    ]
    for case in shadow_corpus():
        assert [r.id for r in fresh.retrieve(case.task_context)] == [
            r.id for r in warm_before[case.id]
        ]


# ------------------------------------------------------------------- refusals


def test_contamination_refusal_spends_nothing(tmp_path: Path) -> None:
    """A pool quoting the holdout is refused before any evaluation or write."""
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    leaked = live_record("lr-leaked", status="ephemeral", leak=True)
    engine.propose(leaked)
    engine.queue_replay(leaked.id, ["run-leak"])
    engine.promote(
        leaked.id,
        RecordCounterfactual(
            replay_run_ids=["run-leak"], verdict_change=True, evidence="runs/replay/pair.json"
        ),
    )
    before = engine_snapshot(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR
    runner = CountingRunner()

    with pytest.raises(ShadowContaminationRefused) as refused:
        run_pass(engine, shadow_candidate(), runs_root=runs_root, runner=runner)

    assert refused.value.report.contaminated
    assert refused.value.report.findings[0].record_id == "lr-leaked"
    assert refused.value.report.findings[0].case_id == "bench-fresh-1"
    assert runner.calls == 0
    assert not runs_root.exists(), "a refused pass must not create its run directory"
    assert engine_snapshot(store_root) == before


def test_contamination_refusal_survives_narrow_k(tmp_path: Path) -> None:
    """A leak cannot hide behind a narrow retrieval width (k=1)."""
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR
    # The lead overlay record fills k=1 for every context, so the leak would
    # never be served at this width — and is still refused: shadow evidence
    # must survive a later k change.
    candidate = shadow_candidate(
        records=[live_record("lr-lead"), live_record("lr-leaked", leak=True)],
        k=1,
    )

    with pytest.raises(ShadowContaminationRefused):
        run_pass(engine, candidate, runs_root=runs_root)

    assert not runs_root.exists()


def test_budget_refusal_spends_nothing(tmp_path: Path) -> None:
    """An over-ceiling projection is refused before any evaluation or write."""
    store_root = tmp_path / "store"
    engine = store_with_lifecycle_states(store_root)
    before = engine_snapshot(store_root)
    runs_root = store_root / "runs" / RUNS_SUBDIR
    runner = CountingRunner()
    # 32 replays × $0.002 projects $0.064 — well over the $0.01 ceiling.
    candidate = shadow_candidate(budget=priced_budget("0.01"))

    with pytest.raises(ShadowBudgetRefused) as refused:
        run_pass(engine, candidate, runs_root=runs_root, runner=runner)

    assert refused.value.replays == 2 * 4 * REPEAT_FLOOR
    assert refused.value.projected > refused.value.allowed
    assert runner.calls == 0
    assert not runs_root.exists()
    assert engine_snapshot(store_root) == before


# ------------------------------------------------------- overlay retrieval units


def test_overlay_retrieval_semantics() -> None:
    """Overlay leads and dedupes; pool fills under the configured floor and k."""
    # lr-stale shares tokens with the probe context, so it clears the pool
    # floor; lr-live does not and must be filtered out.
    pool = [live_record("lr-live"), live_record("lr-stale", topic="retries rate limiter")]
    overlay = [live_record("lr-new"), live_record("lr-new")]  # duplicate id collapses
    retriever = OverlayRetriever(pool=pool, overlay=overlay, k=2)
    context = shadow_corpus()[0].task_context

    served = retriever.retrieve(context)

    assert [record.id for record in served] == ["lr-new", "lr-stale"]
    assert retriever.overlay_record_ids == ["lr-new"]
    # The overlay is the counterfactual being profiled: served regardless of floor.
    floored = OverlayRetriever(pool=pool, overlay=[live_record("lr-new")], floor=1.0, k=5)
    assert [record.id for record in floored.retrieve(context)] == ["lr-new"]
    # A width of zero is honestly empty.
    assert OverlayRetriever(pool=pool, overlay=[], k=0).retrieve(context) == []


def test_shadow_candidate_refuses_duplicate_overlay_ids() -> None:
    with pytest.raises(ValueError, match="duplicate overlay record ids"):
        shadow_candidate(
            records=[live_record("lr-new"), live_record("lr-new")],
        )


# --------------------------------------------------------------- config loading


def write_candidate_file(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_shadow_candidate_roundtrip(tmp_path: Path) -> None:
    """The declaration file survives the YAML round-trip field for field."""
    record = live_record("lr-dropped", status="dropped")  # resurrection candidate
    payload = {
        "candidate_id": "cand-file",
        "records": [record.model_dump(mode="json")],
        "k": 3,
        "floor": 0.2,
        "budget": {
            "expectation": {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_price_per_mtok_usd": "0.003",
                "output_price_per_mtok_usd": "0.015",
            },
            "max_run_cost_usd": "10.00",
        },
        "heldout_family": HOLDOUT_FAMILY,
    }
    loaded = load_shadow_candidate(write_candidate_file(tmp_path, payload))

    assert loaded.candidate_id == "cand-file"
    assert loaded.k == 3
    assert loaded.floor == 0.2
    assert loaded.records == [record]
    assert loaded.budget.max_run_cost_usd == Decimal("10.00")
    assert loaded.heldout_family == HOLDOUT_FAMILY


def test_load_shadow_candidate_refuses_config_drift(tmp_path: Path) -> None:
    payload: dict[str, object] = {
        "candidate_id": "cand-file",
        "budget": {
            "expectation": {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_price_per_mtok_usd": "0",
                "output_price_per_mtok_usd": "0",
            },
            "max_run_cost_usd": "25.00",
        },
    }
    with pytest.raises(ValueError, match="knob"):
        load_shadow_candidate(write_candidate_file(tmp_path, {**payload, "knob": 1}))
    with pytest.raises(ValueError):
        load_shadow_candidate(write_candidate_file(tmp_path, {"candidate_id": "cand-file"}))


# ------------------------------------------------------------------- CLI surface


def test_cli_shadow_profiles_without_touching_the_store(tmp_path: Path) -> None:
    """Happy path: exit 0, summary printed, store untouched, results confined."""
    store_root = tmp_path / "store"
    store_with_lifecycle_states(store_root)
    before = engine_snapshot(store_root)
    candidate = write_candidate_file(
        tmp_path,
        {
            "candidate_id": "cand-cli",
            "records": [live_record("lr-dropped", status="dropped").model_dump(mode="json")],
            "budget": {
                "expectation": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_price_per_mtok_usd": "0",
                    "output_price_per_mtok_usd": "0",
                },
                "max_run_cost_usd": "25.00",
            },
        },
    )

    rc = cli_main(["shadow", "--root", str(store_root), "--candidate", str(candidate)])

    assert rc == 0
    assert engine_snapshot(store_root) == before
    assert_results_confined(store_root, "cand-cli")
    assert (store_root / "runs" / RUNS_SUBDIR / "cand-cli" / "complete.marker").exists()


def test_cli_shadow_refusal_exits_non_zero_with_zero_spent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_root = tmp_path / "store"
    store_with_lifecycle_states(store_root)
    candidate = write_candidate_file(
        tmp_path,
        {
            "candidate_id": "cand-broke",
            "budget": {
                "expectation": {
                    "input_tokens": 1000,
                    "output_tokens": 1000,
                    "input_price_per_mtok_usd": "1",
                    "output_price_per_mtok_usd": "1",
                },
                "max_run_cost_usd": "0.01",
            },
        },
    )

    rc = cli_main(["shadow", "--root", str(store_root), "--candidate", str(candidate)])

    assert rc == 1
    assert "$0 spent" in capsys.readouterr().err
    assert not (store_root / "runs" / RUNS_SUBDIR).exists()


def test_cli_shadow_unknown_holdout_family_is_refused(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store_with_lifecycle_states(store_root)
    candidate = write_candidate_file(
        tmp_path,
        {
            "candidate_id": "cand-drift",
            "budget": {
                "expectation": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_price_per_mtok_usd": "0",
                    "output_price_per_mtok_usd": "0",
                },
                "max_run_cost_usd": "25.00",
            },
            "heldout_family": "no-such-family",
        },
    )

    assert cli_main(["shadow", "--root", str(store_root), "--candidate", str(candidate)]) == 1
    assert not (store_root / "runs" / RUNS_SUBDIR).exists()


def test_cli_shadow_completed_rerun_is_refused(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store_with_lifecycle_states(store_root)
    candidate = write_candidate_file(
        tmp_path,
        {
            "candidate_id": "cand-once",
            "budget": {
                "expectation": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_price_per_mtok_usd": "0",
                    "output_price_per_mtok_usd": "0",
                },
                "max_run_cost_usd": "25.00",
            },
        },
    )
    first = cli_main(["shadow", "--root", str(store_root), "--candidate", str(candidate)])
    assert first == 0

    second = cli_main(["shadow", "--root", str(store_root), "--candidate", str(candidate)])

    assert second == 1  # the completed run is immutable evidence


# ------------------------------------------------------------ protocol runtime twin


def test_real_store_satisfies_the_readonly_seam(tmp_path: Path) -> None:
    """Runtime twin of shadow.py's static protocol check: Engine is a ShadowStore."""
    engine = store_with_lifecycle_states(tmp_path / "store")
    from evogine.bench.shadow import ShadowStore

    assert isinstance(engine, ShadowStore)
    # And the report artifact is a self-contained JSON document:
    runs_root = tmp_path / "store" / "runs" / RUNS_SUBDIR
    report = run_pass(engine, shadow_candidate(), runs_root=runs_root)
    assert isinstance(json.loads((runs_root / "cand-jitter" / "artifacts" / "report.json").read_text())["candidate_id"], str)
    assert report.contamination.probe == "recall-holdout"
