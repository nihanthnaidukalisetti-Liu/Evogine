"""The ``evogine-syncd`` entry point: validate first, bind second, serve last.

The launch contract refuses on stderr with exit 1 before any socket
exists — these tests pin that behavior for the three refusal shapes a
real operator hits first: a LAN bind without ``--expose-lan``, an
unresolvable host, and a missing ``--stores-root``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The daemon is optional transport behind the [server] extra: the core CI
# jobs never install it, so this module skips as a whole there (the console
# precedent) and runs for real in the dedicated sync CI job. The sync server
# entry imports uvicorn at module level, so both extras' packages must be
# guarded. The import after the skip calls is module-level-late by design.
fastapi = pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from evogine.sync.server import main as syncd_main


def test_cli_refuses_lan_bind_without_flag(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = syncd_main(["--stores-root", str(tmp_path), "--host", "0.0.0.0"])
    assert rc == 1
    assert "--expose-lan" in capsys.readouterr().err


def test_cli_refuses_unresolvable_host(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = syncd_main(["--stores-root", str(tmp_path), "--host", "no-such-host.invalid"])
    assert rc == 1
    assert "cannot resolve" in capsys.readouterr().err


def test_cli_requires_stores_root(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit):
        syncd_main([])
    assert "--stores-root" in capsys.readouterr().err
