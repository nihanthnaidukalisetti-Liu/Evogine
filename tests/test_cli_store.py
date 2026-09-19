"""The ``evogine store`` maintenance surface: backup, restore, compact."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import make_record

from evogine.cli import main
from evogine.engine.store import Engine


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _build_store(tmp_path: Path) -> Engine:
    engine = Engine(tmp_path / "store")
    for i in range(3):
        engine.propose(make_record(f"lr-{i}"))
        engine.queue_replay(f"lr-{i}", [f"run-{i}-a"])
    return engine


def test_store_backup_and_restore_cli_round_trip(tmp_path: Path) -> None:
    engine = _build_store(tmp_path)
    archive = tmp_path / "store.tar"
    dest = tmp_path / "restored"

    assert main(["store", "backup", "--root", str(engine.root), "--to", str(archive)]) == 0
    assert main(["store", "restore", "--from", str(archive), "--to", str(dest)]) == 0

    assert _tree_bytes(dest) == _tree_bytes(engine.root)
    assert _git(dest, "git", "rev-parse", "HEAD") == _git(engine.root, "git", "rev-parse", "HEAD")


def test_store_compact_cli_writes_the_index(tmp_path: Path) -> None:
    engine = _build_store(tmp_path)

    assert main(["store", "compact", "--root", str(engine.root)]) == 0

    assert (engine.root / "index.jsonl").exists()


def test_store_backup_cli_refuses_destination_inside_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _build_store(tmp_path)
    inside = engine.root / "store.tar"

    assert main(["store", "backup", "--root", str(engine.root), "--to", str(inside)]) == 1
    assert "outside the store" in capsys.readouterr().err


def test_store_benchmark_cli_runs_small(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store_dir = tmp_path / "bench-store"

    assert (
        main(
            [
                "store",
                "benchmark",
                "--records",
                "20",
                "--repeats",
                "1",
                "--dir",
                str(store_dir),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "store benchmark: 20 records" in out
    assert "->" in out  # before -> after latencies are the point
