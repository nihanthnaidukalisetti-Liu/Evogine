"""The 10k-record benchmark: what compaction buys, measured, not argued.

Generates a deterministic synthetic store (``seed`` pins every record),
measures duplicate detection and retrieval latency over the honest full
scan (no index), compacts, and measures again over the fresh derived
index. The run fails loudly if the two phases disagree on outcomes —
duplicate-pair counts and retrieval results must be identical, or the
timings describe different programs and are worthless.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.store import Engine
from evogine.ops.compaction import compact_store


class BenchmarkError(RuntimeError):
    """Raised when the two benchmark phases disagree on store outcomes."""


class PhaseTiming(BaseModel):
    """One measurement phase: per-operation latency plus its outcomes."""

    model_config = ConfigDict(frozen=True)

    duplicates_ms: float
    """Per-call latency of ``find_duplicates`` over the whole store."""
    retrieval_ms: float
    """Per-query latency of ``retrieve`` averaged over the probe set."""
    duplicate_pairs: int
    retrieval_top_ids: list[list[str]]
    """Retrieval outcome per probe — compared between phases."""


class StoreBenchmark(BaseModel):
    """Full benchmark result: outcomes, latencies, and store sizes."""

    model_config = ConfigDict(frozen=True)

    record_count: int
    promoted_count: int
    candidate_count: int
    repeats: int
    before: PhaseTiming
    after: PhaseTiming
    store_bytes_before: int
    store_bytes_after: int

    def render(self) -> str:
        """Human-readable summary for the CLI and the PR description."""

        def _kib(size: int) -> str:
            return f"{size / 1024:.1f} KiB"

        return (
            f"store benchmark: {self.record_count} records "
            f"(promoted={self.promoted_count}, candidates={self.candidate_count}, "
            f"repeats={self.repeats})\n"
            f"duplicate detection: {self.before.duplicates_ms} ms -> "
            f"{self.after.duplicates_ms} ms "
            f"({self.after.duplicate_pairs} exact pairs)\n"
            f"retrieval per query: {self.before.retrieval_ms} ms -> "
            f"{self.after.retrieval_ms} ms ({len(_QUERIES)} probes, k=5)\n"
            f"store size: {_kib(self.store_bytes_before)} -> "
            f"{_kib(self.store_bytes_after)} (derived index included)"
        )


#: Body vocabulary — rotated per record so bodies stay mostly unique.
_SIGNALS = [
    "rising 429 counts across consecutive rollouts",
    "OOM in the verifier under large diffs",
    "flaky retry storm during warm replays",
    "stale lock contention between workers",
    "drifting token budgets across arms",
    "silently dropped tool results in transcripts",
]
_FAILURE_MODES = [
    "amplifies without jittered backoff",
    "truncates context without headroom",
    "poisons the promotion pool with near-duplicates",
    "deadlocks under concurrent review windows",
    "leaks budget into failed arms",
    "hides verifier verdict flips",
]
_DETECTIONS = [
    "counterfactual replay flips the verifier verdict",
    "budget telemetry flags a cold-arm regression",
    "duplicate detector reports exact body pairs",
    "lock audit shows overlapping critical sections",
    "spend report diverges from the arm ledger",
    "trajectory log shows an unreviewed approval",
]
_FIX_STRATEGIES = [
    "add jittered exponential backoff with a per-rollout retry budget",
    "cap context growth and fail closed on overflow",
    "deduplicate the promotion pool before gating",
    "serialize review windows behind the advisory lock",
    "charge budgets per arm with a hard ceiling",
    "surface tool results as first-class evidence",
]
#: Fixed retrieval probes drawn from the same vocabulary.
_QUERIES = [
    "429 backoff escalates across rollouts",
    "verifier OOM on large diffs",
    "duplicate promotion pool entries",
    "lock contention between review windows",
    "budget drift across replay arms",
]

_DUPLICATE_EVERY = 10
"""Every Nth record repeats the previous body — an exact duplicate pair."""
_PROMOTE_EVERY = 7
"""Every Nth candidate is promoted, sizing the retrieval pool."""


def _synthetic_record(record_id: str, body_index: int) -> LearningRecord:
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=f"Synthetic pattern {record_id}",
        body=RecordBody(
            signal=_SIGNALS[body_index],
            failure_mode=_FAILURE_MODES[body_index],
            detection=_DETECTIONS[body_index],
            fix_strategy=_FIX_STRATEGIES[body_index],
        ),
        status="ephemeral",
        confidence=0.6,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="synthetic-benchmark",
            source="unit-test",
            note="deterministic synthetic record; never touches a live model",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 19, tzinfo=UTC),
        promoted_at=None,
    )


def generate_synthetic_store(root: Path, record_count: int, seed: int = 0) -> Engine:
    """Build a deterministic store with ``record_count`` records.

    Bodies rotate through a fixed vocabulary (``seed`` shuffles the
    starting offset), every tenth record exactly duplicates its
    predecessor's body, and every seventh candidate is promoted — a
    realistic lifecycle mix with duplicate pairs for the detector to
    find. Pure function of ``(record_count, seed)``: same inputs, same
    store content.
    """
    rng = random.Random(seed)
    start_offset = rng.randrange(len(_SIGNALS))
    engine = Engine(root)
    previous_body_index: int | None = None
    for i in range(record_count):
        if previous_body_index is not None and i % _DUPLICATE_EVERY == 1:
            body_index = previous_body_index  # exact duplicate pair
        else:
            body_index = (start_offset + i) % len(_SIGNALS)
        record = _synthetic_record(f"lr-{i:05d}", body_index)
        engine.propose(record)
        engine.queue_replay(record.id, [f"run-{i}-a", f"run-{i}-b"])
        if i % _PROMOTE_EVERY == 0:
            engine.promote(
                record.id,
                RecordCounterfactual(
                    replay_run_ids=[f"run-{i}-a", f"run-{i}-b"], verdict_change=True
                ),
            )
        previous_body_index = body_index
    return engine


def _timed(function: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Return ``(ms_per_call, last_result)`` over ``repeats`` calls."""
    start = time.perf_counter()
    result: Any = None
    for _ in range(repeats):
        result = function()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return elapsed_ms, result


def _measure(engine: Engine, queries: list[str], repeats: int) -> PhaseTiming:
    duplicates_ms, duplicates = _timed(engine.find_duplicates, repeats)

    def run_retrieval() -> list[list[str]]:
        return [[r.id for r in engine.retrieve(query, k=5)] for query in queries]

    retrieval_ms, retrieval = _timed(run_retrieval, repeats)
    return PhaseTiming(
        duplicates_ms=round(duplicates_ms, 3),
        retrieval_ms=round(retrieval_ms / len(queries), 3),
        duplicate_pairs=len(duplicates),
        retrieval_top_ids=retrieval,
    )


def _store_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run_benchmark(root: Path, record_count: int, seed: int = 0, repeats: int = 3) -> StoreBenchmark:
    """Generate, measure the full scan, compact, measure the index path.

    Raises ``BenchmarkError`` if the two phases produce different
    outcomes — compaction must never change what the store answers,
    only how fast it answers.
    """
    queries = list(_QUERIES)
    engine = generate_synthetic_store(root, record_count, seed=seed)
    promoted = len(engine.promoted_records())
    candidates = len(engine.candidates())

    before = _measure(engine, queries, repeats)
    bytes_before = _store_bytes(engine.root)

    compact_store(engine.root)

    after = _measure(engine, queries, repeats)
    bytes_after = _store_bytes(engine.root)

    if (
        before.duplicate_pairs != after.duplicate_pairs
        or before.retrieval_top_ids != after.retrieval_top_ids
    ):
        raise BenchmarkError(
            "compaction changed store outcomes — timings are void: "
            f"duplicates {before.duplicate_pairs}!={after.duplicate_pairs} "
            "or retrieval diverged"
        )
    return StoreBenchmark(
        record_count=record_count,
        promoted_count=promoted,
        candidate_count=candidates,
        repeats=repeats,
        before=before,
        after=after,
        store_bytes_before=bytes_before,
        store_bytes_after=bytes_after,
    )


__all__ = [
    "BenchmarkError",
    "StoreBenchmark",
    "generate_synthetic_store",
    "run_benchmark",
]
