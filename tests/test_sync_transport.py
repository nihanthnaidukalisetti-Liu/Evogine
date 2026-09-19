"""Transport round-trip over a real uvicorn server and real git.

The daemon serves existing store repositories; this module proves the
whole loop offline: init a store, clone it over smart HTTP elsewhere,
advance the clone, push back, pull from a third copy, and verify every
store file is byte-identical across all three. It also proves the
per-store advisory lock serializes pushes against local lifecycle
writes. Offline: loopback only, no external network.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from evogine.engine.store import Engine, store_lock
from evogine.sync.app import create_app
from evogine.sync.settings import SyncSettings

fastapi = pytest.importorskip("fastapi")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def daemon(tmp_path: Path, record_factory):  # type: ignore[no-untyped-def]
    """A committed hosted store served by a real uvicorn server, loopback."""
    stores_root = tmp_path / "stores"
    store = Engine(stores_root / "alpha")
    store.propose(record_factory("lr-seed-1"))

    app = create_app(SyncSettings(stores_root=stores_root))
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:  # pragma: no cover - startup failure is a test bug
        raise RuntimeError("uvicorn did not start")
    yield port, stores_root / "alpha"
    server.should_exit = True
    thread.join(timeout=10)


def _run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=True,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )


def _tree(root: Path) -> dict[str, bytes]:
    """Every store file, byte-for-byte, ignoring git's own plumbing."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def test_clone_push_pull_round_trip_is_byte_identical(daemon, tmp_path, record_factory) -> None:  # type: ignore[no-untyped-def]
    port, hosted = daemon
    url = f"http://127.0.0.1:{port}/git/alpha"

    clone_a = tmp_path / "clone-a"
    _run_git("clone", url, str(clone_a))
    assert _tree(clone_a) == _tree(hosted)

    # A remote syncer advances ITS copy through the engine's own paths.
    remote = Engine(clone_a)
    remote.propose(record_factory("lr-remote-1"))
    remote.propose(record_factory("lr-remote-2"))
    _run_git("push", cwd=clone_a)

    # The hosted store's working tree followed its HEAD (updateInstead).
    assert (hosted / "records" / "lr-remote-1.json").is_file()
    assert (hosted / "records" / "lr-remote-2.json").is_file()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=hosted, capture_output=True, text=True, check=False
    )
    assert status.stdout.strip() == "", status.stdout

    # A third copy, cloned BEFORE the push, catches up by pulling.
    clone_b = tmp_path / "clone-b"
    _run_git("clone", url, str(clone_b))
    remote_b = Engine(clone_b)
    remote_b.propose(record_factory("lr-remote-3"))
    _run_git("push", cwd=clone_b)
    _run_git("pull", "--ff-only", cwd=clone_a)

    # All three trees are the same store, byte for byte.
    assert _tree(clone_a) == _tree(hosted)
    assert _tree(clone_a) == _tree(clone_b)
    assert "records/lr-remote-3.json" in _tree(clone_a)


def test_push_serializes_with_local_lifecycle_writes(daemon, tmp_path, record_factory) -> None:  # type: ignore[no-untyped-def]
    port, hosted = daemon
    url = f"http://127.0.0.1:{port}/git/alpha"

    clone = tmp_path / "clone-c"
    _run_git("clone", url, str(clone))
    remote = Engine(clone)
    remote.propose(record_factory("lr-locked-1"))

    # Hold the store's advisory lock (as a local lifecycle write would):
    # the push's receive-pack must block until the lock is released, then
    # proceed — serialization, not corruption.
    lock_held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with store_lock(hosted):
            lock_held.set()
            release.wait(timeout=10)

    lock_thread = threading.Thread(target=holder, daemon=True)
    lock_thread.start()
    assert lock_held.wait(timeout=10)

    push_done = threading.Event()

    def pusher() -> None:
        _run_git("push", cwd=clone)
        push_done.set()

    push_thread = threading.Thread(target=pusher, daemon=True)
    push_thread.start()
    time.sleep(1.0)
    assert not push_done.is_set(), "push ran while the advisory lock was held"

    release.set()
    assert push_done.wait(timeout=30)
    lock_thread.join(timeout=10)

    assert (hosted / "records" / "lr-locked-1.json").is_file()
