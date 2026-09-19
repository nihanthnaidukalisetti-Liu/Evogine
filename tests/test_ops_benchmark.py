"""The synthetic-store generator and the compaction benchmark."""

from __future__ import annotations

import shutil

from conftest import temp_store_root

from evogine.ops.benchmark import generate_synthetic_store, run_benchmark


def test_synthetic_generation_is_deterministic() -> None:
    first = temp_store_root("evogine-bench-a-")
    second = temp_store_root("evogine-bench-b-")
    try:
        engine_a = generate_synthetic_store(first, 30, seed=7)
        engine_b = generate_synthetic_store(second, 30, seed=7)

        # Same seed, same store outcomes: duplicate pairs are identical.
        assert engine_a.find_duplicates() == engine_b.find_duplicates()
        assert len(engine_a.find_duplicates()) > 0  # duplicates are planted
        assert [r.id for r in engine_a.promoted_records()] == [
            r.id for r in engine_b.promoted_records()
        ]
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


def test_synthetic_store_shape() -> None:
    root = temp_store_root("evogine-bench-shape-")
    try:
        engine = generate_synthetic_store(root, 21, seed=0)
        promoted = engine.promoted_records()
        candidates = engine.candidates()
        # Every seventh candidate is promoted (i % 7 == 0 → i = 0, 7, 14).
        assert len(promoted) == 3
        assert len(promoted) + len(candidates) == 21
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_benchmark_phases_agree() -> None:
    root = temp_store_root("evogine-bench-run-")
    try:
        benchmark = run_benchmark(root, 21, seed=0, repeats=1)

        assert benchmark.record_count == 21
        # Compaction changed the speed, never the answers.
        assert benchmark.before.duplicate_pairs == benchmark.after.duplicate_pairs
        assert benchmark.before.retrieval_top_ids == benchmark.after.retrieval_top_ids
        assert benchmark.after.duplicates_ms > 0
        assert benchmark.after.retrieval_ms > 0
        # The derived index adds bytes; it never removes store content.
        assert benchmark.store_bytes_after >= benchmark.store_bytes_before
        summary = benchmark.render()
        assert "21 records" in summary
        assert "->" in summary
    finally:
        shutil.rmtree(root, ignore_errors=True)
