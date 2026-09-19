"""The ``evogine-syncd`` entry point: validate first, bind second, serve last.

The launch contract refuses on stderr with exit 1 before any socket
exists — these tests pin that behavior for the three refusal shapes a
real operator hits first: a LAN bind without ``--expose-lan``, an
unresolvable host, and a missing ``--stores-root``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evogine.sync.server import main as syncd_main

fastapi = pytest.importorskip("fastapi")


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
