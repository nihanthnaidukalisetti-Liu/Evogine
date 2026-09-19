"""Binding-policy tests: loopback by default, LAN reads by explicit flag.

The binding policy is contractual, so these tests pin its VISIBLE refusal
states: a non-loopback peer pushing to a store gets a 403 body that names
the policy (regardless of any flag), reads are gated by the bind address
only, and a non-loopback bind without ``--expose-lan`` refuses before any
socket is opened. Peer addresses are simulated through TestClient's
``client`` argument; no socket is bound here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The daemon is optional transport behind the [server] extra: the core CI
# jobs never install it, so this module skips as a whole there (the console
# precedent) and runs for real in the dedicated sync CI job. uvicorn is
# guarded too: the serve() import pulls in evogine.sync.server, which
# imports uvicorn at module level. The imports after the skip calls are
# module-level-late by design.
fastapi = pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient

from evogine.engine.store import Engine
from evogine.sync.app import create_app
from evogine.sync.server import serve
from evogine.sync.settings import SyncError, SyncSettings


@pytest.fixture()
def hosted(tmp_path: Path, record_factory) -> Path:  # type: ignore[no-untyped-def]
    """One committed hosted store under the daemon's stores root."""
    root = tmp_path / "stores" / "alpha"
    store = Engine(root)
    store.propose(record_factory("lr-bind-1"))
    # A store-shaped directory that is NOT a git repository: reads only.
    readonly = tmp_path / "stores" / "readonly"
    (readonly / "records").mkdir(parents=True)
    (readonly / "records" / "lr-seed.json").write_text("{}", encoding="utf-8")
    return tmp_path / "stores"


def test_mutating_transport_refused_for_non_loopback_peer(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted, expose_lan=True))
    # Default TestClient peer is "testclient" — not a loopback address, so
    # the refusal must fire EVEN THOUGH this app was explicitly exposed.
    tc = TestClient(app, raise_server_exceptions=False)
    res = tc.post("/git/alpha/git-receive-pack", content=b"0000")
    assert res.status_code == 403
    body = res.json()
    assert "loopback-only" in body["error"]
    assert "regardless" in body["error"]
    assert "testclient" in body["error"]


def test_read_transport_allowed_beyond_loopback_when_exposed(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted, expose_lan=True))
    tc = TestClient(
        app,
        client=("203.0.113.9", 5555),  # a LAN peer
        raise_server_exceptions=False,
    )
    res = tc.get("/git/alpha/info/refs", params={"service": "git-upload-pack"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-git-upload-pack")
    assert b"upload-pack" in res.content


def test_query_reads_allowed_beyond_loopback_when_exposed(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted, expose_lan=True))
    tc = TestClient(app, client=("203.0.113.9", 5555))
    res = tc.get("/api/stores")
    assert res.status_code == 200
    assert "alpha" in res.json()["stores"]
    health = tc.get("/health").json()
    assert health["bind_policy"]["expose_lan"] is True
    assert health["bind_policy"]["mutations"] == "loopback-only regardless of any flag"


def test_mutating_transport_allowed_for_loopback_peer(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted))
    tc = TestClient(app, client=("127.0.0.1", 5555), raise_server_exceptions=False)
    res = tc.post("/git/alpha/git-receive-pack", content=b"0000")
    assert res.status_code != 403
    assert "loopback-only" not in res.text


def test_unknown_store_transport_refusal(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted))
    tc = TestClient(app, client=("127.0.0.1", 5555), raise_server_exceptions=False)
    res = tc.get("/git/nope/info/refs", params={"service": "git-upload-pack"})
    assert res.status_code == 404
    assert "nope" in res.json()["error"]


def test_non_git_store_refuses_transport_visibly(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted))
    tc = TestClient(app, client=("127.0.0.1", 5555), raise_server_exceptions=False)
    res = tc.get("/git/readonly/info/refs", params={"service": "git-upload-pack"})
    assert res.status_code == 409
    assert "not a git repository" in res.json()["error"]


def test_lan_bind_without_flag_refuses_before_any_socket(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SyncError, match="--expose-lan"):
        serve(SyncSettings(stores_root=tmp_path, host="0.0.0.0"))


def test_unresolvable_bind_refuses(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SyncError, match="cannot resolve"):
        serve(SyncSettings(stores_root=tmp_path, host="no-such-host.invalid"))


def test_grammar_violating_store_name_refuses_with_400(hosted) -> None:  # type: ignore[no-untyped-def]
    app = create_app(SyncSettings(stores_root=hosted))
    tc = TestClient(app, raise_server_exceptions=False)
    # Leading hyphen fails the identity grammar even after normalization.
    res = tc.get("/api/stores/-bad/records")
    assert res.status_code == 400
    assert "must match" in res.json()["error"]


def test_discovery_skips_files_and_non_conforming_dirs(hosted) -> None:  # type: ignore[no-untyped-def]
    (hosted / "stray.txt").write_text("not a store", encoding="utf-8")
    bad_grammar = hosted / "-nope"  # invalid even after normalization
    (bad_grammar / "records").mkdir(parents=True)
    app = create_app(SyncSettings(stores_root=hosted))
    tc = TestClient(app)
    stores = tc.get("/api/stores").json()["stores"]
    # The non-git 'readonly' dir IS hosted (store-shaped); transport refuses
    # it separately — discovery lists, transport validates.
    assert set(stores) == {"alpha", "readonly"}


def test_missing_stores_root_discovery_is_empty(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from evogine.sync.stores import StoreRegistry

    assert StoreRegistry(tmp_path / "does-not-exist").names() == []
