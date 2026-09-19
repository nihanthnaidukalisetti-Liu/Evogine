"""Sync daemon settings and binding-policy refusals — offline, core-safe.

This module imports nothing from FastAPI or uvicorn: the settings layer is
core (stdlib + pydantic) so the CI core job exercises it even without the
``server`` extra installed. The binding policy is contractual, so these
tests pin its refusal messages as visible states, not just exception types.
"""

from evogine.sync.settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    SyncError,
    SyncSettings,
    is_loopback_peer,
    resolve_bind_host,
)


def test_default_settings_are_loopback() -> None:
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8766
    assert SyncSettings(stores_root=".").host == DEFAULT_HOST
    assert SyncSettings(stores_root=".").expose_lan is False


def test_localhost_resolves_to_concrete_loopback() -> None:
    assert resolve_bind_host("localhost", expose_lan=False) in {"127.0.0.1", "::1"}
    assert resolve_bind_host("127.0.0.1", expose_lan=False) == "127.0.0.1"


def test_non_loopback_bind_refused_without_flag() -> None:
    for host in ("0.0.0.0", "10.0.0.1"):
        try:
            resolve_bind_host(host, expose_lan=False)
        except SyncError as exc:
            assert "--expose-lan" in str(exc), host
        else:  # pragma: no cover - the refusals above must fire
            raise AssertionError(f"non-loopback bind {host!r} was not refused")


def test_non_loopback_bind_allowed_with_flag() -> None:
    assert resolve_bind_host("0.0.0.0", expose_lan=True) == "0.0.0.0"
    assert resolve_bind_host("10.0.0.1", expose_lan=True) == "10.0.0.1"


def test_unresolvable_host_refused() -> None:
    try:
        resolve_bind_host("no-such-host.invalid", expose_lan=False)
    except SyncError as exc:
        assert "cannot resolve" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unresolvable host was not refused")


def test_settings_validation_refuses_missing_root_and_bad_port(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with_root = SyncSettings(stores_root=tmp_path)
    assert with_root.validated() is with_root
    try:
        SyncSettings(stores_root=tmp_path / "absent").validated()
    except SyncError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing stores root was not refused")
    for port in (0, -1, 65536):
        try:
            SyncSettings(stores_root=tmp_path, port=port).validated()
        except SyncError as exc:
            assert "port" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"port {port} was not refused")


def test_is_loopback_peer_fails_closed() -> None:
    assert is_loopback_peer("127.0.0.1") is True
    assert is_loopback_peer("127.0.0.2") is True
    assert is_loopback_peer("::1") is True
    assert is_loopback_peer(None) is False
    assert is_loopback_peer("") is False
    assert is_loopback_peer("testclient") is False
    assert is_loopback_peer("203.0.113.9") is False
    assert is_loopback_peer("::ffff:203.0.113.9") is False
