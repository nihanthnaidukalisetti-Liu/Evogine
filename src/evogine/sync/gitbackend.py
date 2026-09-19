"""Git smart-HTTP transport: the ``git http-backend`` CGI handoff.

The store already is a git repository, so store transport is fetch/push of
that repository — served here by the standard ``git http-backend``, driven
as a CGI subprocess per request (the same handoff every smart-HTTP server
performs underneath its own plumbing). One subprocess per request keeps
git's own parsing, object transfer, and ref updates in git's hands: the
daemon adds the binding policy and the lock discipline, never a second
implementation of git.

The store-mutating half of the protocol (``receive-pack`` — a push) is
gated by the route in :mod:`evogine.sync.app` before this module runs:
loopback peers only, regardless of any flag. This module's contract is the
git-side configuration that makes hosted *working-tree* stores receive
pushes safely:

- ``http.receivepack=true`` — smart-HTTP push is disabled by default for
  git hosting; the daemon enables it and owns the peer gate instead.
- ``receive.denyCurrentBranch=updateInstead`` — a hosted store is a
  working tree (the query API serves its files), so a push must update
  HEAD's worktree, not just the ref. Git refuses the update when the
  worktree is not clean; every real store is porcelain-clean by the
  engine's own invariant (one lifecycle commit sweeps ``git add -A .``),
  so the refusal fires only on corruption or on a zero-commit store whose
  lock file was touched but never committed — the one bootstrap edge,
  documented rather than papered over: a store with no lifecycle history
  supports fetch only.

Per-store audit order is preserved by git itself: a push transfers the
sender's commits verbatim, in order, onto the hosted store's history. No
global ordering is invented across stores (governance invariant 3.3) —
each trail's internal order is the only order that exists, and the git
history is the reconciliation record.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from evogine.sync.settings import SyncError

#: Config pinned on every http-backend invocation (see module docstring).
#: ``-c`` settings travel to the ``receive-pack`` child through
#: ``GIT_CONFIG_PARAMETERS``, so both halves of a push honor them.
_BACKEND_CONFIG = ("http.receivepack=true", "receive.denyCurrentBranch=updateInstead")


@dataclass(frozen=True)
class CgiResponse:
    """The git backend's CGI response, ready to become an HTTP response."""

    status: int
    headers: list[tuple[str, str]]
    body: bytes


def receive_pack_requested(git_path: str, query_string: str) -> bool:
    """Whether this request participates in the push (receive-pack) handshake.

    Both halves of the handshake count: the ``info/refs`` advertisement for
    ``git-receive-pack`` (harmless alone, but a LAN peer has no legitimate
    reason to start a push it cannot finish) and the ``git-receive-pack``
    POST that mutates the store.
    """
    if git_path.rstrip("/").endswith("git-receive-pack"):
        return True
    return "service=git-receive-pack" in query_string


def run_git_http_backend(
    stores_root: Path,
    store_name: str,
    git_path: str,
    *,
    method: str,
    query_string: str,
    content_type: str | None,
    body: bytes,
    remote_addr: str | None,
) -> CgiResponse:
    """Run one ``git http-backend`` CGI handoff and parse its response.

    ``git_path`` is the git protocol path under the store (``/info/refs``,
    ``/git-upload-pack``, ``/git-receive-pack``); the CGI env presents the
    stores root as ``GIT_PROJECT_ROOT`` and the store as the first path
    segment, exactly as a directory-per-repo host would. Blocking (a
    subprocess) — callers run it off the event loop.
    """
    env = dict(
        os.environ,
        GIT_PROJECT_ROOT=str(stores_root),
        GIT_HTTP_EXPORT_ALL="1",
        GATEWAY_INTERFACE="CGI/1.1",
        REQUEST_METHOD=method,
        PATH_INFO=f"/{store_name}{git_path}",
        QUERY_STRING=query_string,
        SERVER_PROTOCOL="HTTP/1.1",
        SERVER_NAME="evogine-syncd",
        SERVER_PORT="0",
        REMOTE_ADDR=remote_addr or "127.0.0.1",
        CONTENT_LENGTH=str(len(body)),
    )
    if content_type:
        env["CONTENT_TYPE"] = content_type
    backend_args: list[str] = ["git"]
    for option in _BACKEND_CONFIG:
        backend_args += ["-c", option]
    backend_args.append("http-backend")
    proc = subprocess.run(
        backend_args,
        env=env,
        input=body,
        capture_output=True,
        check=False,
    )
    head, separator, payload = proc.stdout.partition(b"\r\n\r\n")
    if not separator:
        # A backend that died before emitting headers (a missing git, a
        # corrupted repo) must answer visibly, not as a blind empty 200.
        raise SyncError(
            f"git http-backend produced no response for {method} "
            f"/{store_name}{git_path}: {proc.stderr.decode(errors='replace').strip()}",
            http_status=502,
        )
    status = 200
    headers: list[tuple[str, str]] = []
    for line in head.split(b"\r\n"):
        name, _, value = line.decode(errors="replace").partition(":")
        name = name.strip()
        if name.lower() == "status":
            status = int(value.strip().split()[0])
        elif name.lower() == "transfer-encoding":
            continue  # framing belongs to the HTTP layer, not the CGI body
        elif name:
            headers.append((name, value.strip()))
    return CgiResponse(status=status, headers=headers, body=payload)
