"""The sync client: one machine's view of the remotes its config declares.

The daemon modules (``evogine.sync.app`` and friends) are the stable public
API this module consumes and never modifies: the daemon hosts store
repositories over git smart HTTP and serves a read/query API. This module
is the client half — resolve a configured endpoint into a remote, read its
query API, publish local store commits over the transport, fetch and merge
from it using the documented low-conflict merge surface, and roll spend up
across every configured machine with exact ``Decimal`` accounting.

Refusal discipline — fail closed, never fail open:

- An unknown or ambiguous ``--remote``, an endpoint with no store name, and
  a store name outside the platform's identity grammar refuse before any
  network happens (misconfiguration is a visible refusal, never a guess).
- An unreachable daemon, an HTTP refusal, and a response that is not the
  documented JSON shape refuse with the remote named — a query result is
  never fabricated, and a refused machine is never silently summed away.
- A push aimed at a non-loopback daemon URL refuses client-side: the
  daemon refuses store mutations from non-loopback peers regardless of any
  flag, so the attempt can only end in that refusal — saying so first is
  the same contract, reported nearer the mistake.
- A non-fast-forward push refuses (pull first); a pull onto a dirty store
  refuses; a conflicting merge aborts and refuses — the client, like the
  daemon, never rewrites knowledge.

Only the standard library lives here: the core install (no ``[server]``
extra) can resolve remotes, read query APIs, and move git history.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shlex
import socket
import subprocess
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evogine.bench.budgets import SpendReconciliation
from evogine.config import SyncEndpoint
from evogine.engine.store import StoreError, normalize_store_name, store_lock

#: Read timeout for the query API. Git transport runs in its own subprocess
#: with its own (generous) timeout — this bounds only the HTTP reads.
DEFAULT_TIMEOUT_SECONDS = 5.0

_GIT_TIMEOUT_SECONDS = 300.0
"""Bound for git subprocesses; loopback transports finish in well under it."""

_LOCK_FILENAME = "review.lock"
"""The engine's advisory lock file — runtime plumbing, never store content."""


class SyncClientError(Exception):
    """A client-side sync refusal: misconfiguration, unreachable remote,
    transport refusal, or a merge the client will not perform.

    Every message names the remote (or the local store) and the cause —
    the visible refusal state the brief requires. A refused operation has
    written nothing.
    """


class ResolvedRemote(BaseModel):
    """One configured endpoint, validated: a daemon base URL and its store."""

    model_config = ConfigDict(frozen=True)

    name: str
    """The endpoint's config name, as declared (attribution in reports)."""

    store: str
    """The hosted store name, normalized to the platform's identity."""

    base_url: str
    """The daemon URL, trailing slash stripped."""

    def git_url(self) -> str:
        """The git smart-HTTP URL of the hosted store."""
        return f"{self.base_url}/git/{self.store}"


def resolve_endpoint(endpoint: SyncEndpoint) -> ResolvedRemote:
    """Validate one configured endpoint into a sync remote, refusing gaps.

    An endpoint without a store name is misconfigured for sync purposes:
    the daemon hosts stores *by name* and the client never guesses one —
    the refusal says what to add. The store name runs through the same
    :func:`normalize_store_name` the daemon's registry uses, so client and
    daemon cannot disagree about what a store is called.
    """
    if endpoint.store is None or not endpoint.store.strip():
        raise SyncClientError(
            f"sync endpoint {endpoint.name!r} declares no store — add "
            f"'store: <name>' under the endpoint in the config (the daemon hosts "
            "stores by name; the client never guesses one)"
        )
    try:
        store = normalize_store_name(endpoint.store)
    except StoreError as exc:
        raise SyncClientError(
            f"sync endpoint {endpoint.name!r} has an unusable store name: {exc}"
        ) from exc
    return ResolvedRemote(name=endpoint.name, store=store, base_url=str(endpoint.url).rstrip("/"))


def resolve_remote(endpoints: Sequence[SyncEndpoint], remote_name: str | None) -> ResolvedRemote:
    """Resolve the one remote a push or pull operates on, refusing ambiguity.

    ``None`` means "the only configured endpoint" — with several configured,
    the choice is the operator's and refusing names them; with none, the
    refusal points at the config. An unknown name is config drift and lists
    what *is* configured.
    """
    if remote_name is None:
        if not endpoints:
            raise SyncClientError(
                "no sync endpoints configured — add one under "
                "profiles.<name>.sync.endpoints in the config"
            )
        if len(endpoints) > 1:
            names = ", ".join(endpoint.name for endpoint in endpoints)
            raise SyncClientError(
                f"--remote is required when more than one endpoint is configured ({names})"
            )
        return resolve_endpoint(endpoints[0])
    for endpoint in endpoints:
        if endpoint.name == remote_name:
            return resolve_endpoint(endpoint)
    known = ", ".join(endpoint.name for endpoint in endpoints) or "none configured"
    raise SyncClientError(f"unknown sync remote {remote_name!r} (configured: {known})")


# ---------------------------------------------------------------------------
# The query API (read-only): overview and spend from one remote.
# ---------------------------------------------------------------------------


class RemoteOverview(BaseModel):
    """One hosted store's overview as the query API served it."""

    total_decisions: int
    promoted: int
    pending_approvals: int
    by_stage: dict[str, int] = Field(default_factory=dict)


class RemoteSpendRow(BaseModel):
    """One bench run's spend on one remote machine (the daemon's row)."""

    run_id: str
    reconciliation: SpendReconciliation


class RemoteSpend(BaseModel):
    """One machine's spend report, attributed.

    The aggregate fields stay ``None`` when that machine recorded no spend —
    absence of evidence is carried as absence, never dressed up as zero.
    """

    remote: str
    store: str
    recorded_runs: int = 0
    rows: list[RemoteSpendRow] = Field(default_factory=list)
    actual_cost_usd: Decimal | None = None
    overage_usd: Decimal | None = None
    within_ceiling: bool | None = None


def _get_json(remote: ResolvedRemote, path: str, *, timeout: float) -> dict[str, object]:
    """GET one query-API path and return its JSON object, refusing loudly.

    A daemon that answers with an error body, does not answer, or answers
    with something other than the documented JSON shape produces a refusal
    naming the remote — never a fabricated empty result.
    """
    url = f"{remote.base_url}{path}"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = ""
        try:
            payload = json.loads(exc.read())
            if isinstance(payload, dict) and "error" in payload:
                detail = f": {payload['error']}"
        except (OSError, ValueError):
            detail = ""
        raise SyncClientError(
            f"remote {remote.name!r} refused the query at {url} (HTTP {exc.code}){detail}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SyncClientError(f"remote {remote.name!r} unreachable at {url}: {exc}") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SyncClientError(
            f"remote {remote.name!r} answered at {url} but not with JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SyncClientError(
            f"remote {remote.name!r} answered at {url} with a non-object JSON payload"
        )
    return payload


def fetch_overview(
    remote: ResolvedRemote, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> RemoteOverview:
    """One hosted store's overview from the daemon's query API."""
    payload = _get_json(remote, f"/api/stores/{remote.store}/overview", timeout=timeout)
    try:
        return RemoteOverview.model_validate(payload)
    except ValidationError as exc:
        raise SyncClientError(
            f"remote {remote.name!r} returned an overview that does not match the "
            f"overview schema: {exc}"
        ) from exc


def fetch_spend(remote: ResolvedRemote, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> RemoteSpend:
    """One hosted store's spend report from the daemon's query API."""
    payload = _get_json(remote, f"/api/stores/{remote.store}/spend", timeout=timeout)
    try:
        return RemoteSpend.model_validate({"remote": remote.name, "store": remote.store, **payload})
    except ValidationError as exc:
        raise SyncClientError(
            f"remote {remote.name!r} returned a spend report that does not match the "
            f"spend schema: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Cross-machine spend aggregation — exact Decimal accounting.
# ---------------------------------------------------------------------------


class MachineSpend(BaseModel):
    """One machine's attribution row in the merged spend view."""

    remote: str
    store: str
    recorded_runs: int
    actual_cost_usd: Decimal | None = None
    overage_usd: Decimal | None = None
    within_ceiling: bool | None = None


class MergedSpend(BaseModel):
    """Spend rolled up across machines, with per-machine attribution kept.

    Totals are ``None`` when no reporting machine recorded spend — a merged
    view that has nothing to sum says so rather than fabricating a zero.
    """

    machines: list[MachineSpend]
    recorded_runs: int = 0
    total_actual_cost_usd: Decimal | None = None
    total_overage_usd: Decimal | None = None
    within_ceiling: bool | None = None


def merge_spend(spend: Sequence[RemoteSpend]) -> MergedSpend:
    """Roll spend up across machines: exact ``Decimal`` sums, absence kept.

    A machine that recorded no spend contributes its attribution row and
    nothing to the totals; when no machine recorded spend the totals are
    ``None``. ``within_ceiling`` is False if any reporting machine breached
    its ceiling, True only when every reporting machine finished within.
    """
    actuals = [report.actual_cost_usd for report in spend if report.actual_cost_usd is not None]
    overages = [report.overage_usd for report in spend if report.overage_usd is not None]
    ceilings = [report.within_ceiling for report in spend if report.within_ceiling is not None]
    return MergedSpend(
        machines=[
            MachineSpend(
                remote=report.remote,
                store=report.store,
                recorded_runs=report.recorded_runs,
                actual_cost_usd=report.actual_cost_usd,
                overage_usd=report.overage_usd,
                within_ceiling=report.within_ceiling,
            )
            for report in spend
        ],
        recorded_runs=sum(report.recorded_runs for report in spend),
        total_actual_cost_usd=sum(actuals, Decimal(0)) if actuals else None,
        total_overage_usd=sum(overages, Decimal(0)) if overages else None,
        within_ceiling=all(ceilings) if ceilings else None,
    )


class RemoteStatus(BaseModel):
    """One remote's status: its query results, or the refusal that answered."""

    label: str
    remote: ResolvedRemote | None = None
    overview: RemoteOverview | None = None
    spend: RemoteSpend | None = None
    refusal: str | None = None

    @property
    def ok(self) -> bool:
        return self.refusal is None


def fleet_status(
    endpoints: Sequence[SyncEndpoint], *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> list[RemoteStatus]:
    """Status every configured endpoint; each result is ok or an explicit refusal.

    A misconfigured or unreachable endpoint never aborts the fleet scan —
    it becomes that machine's refusal row, so the report shows every
    machine's state and hides none.
    """
    statuses: list[RemoteStatus] = []
    for endpoint in endpoints:
        try:
            remote = resolve_endpoint(endpoint)
            overview = fetch_overview(remote, timeout=timeout)
            spend = fetch_spend(remote, timeout=timeout)
        except SyncClientError as exc:
            statuses.append(RemoteStatus(label=endpoint.name, refusal=str(exc)))
            continue
        statuses.append(
            RemoteStatus(label=endpoint.name, remote=remote, overview=overview, spend=spend)
        )
    return statuses


# ---------------------------------------------------------------------------
# The git transport (client half): push and pull over smart HTTP.
# ---------------------------------------------------------------------------


class PushResult(BaseModel):
    """What one push published — and what it deliberately did not."""

    head: str
    """The local store commit the push transferred."""
    ref: str
    """The hosted branch the commits landed on."""
    url: str
    """The daemon git URL used."""
    uncommitted: int
    """Local changes that exist only on this machine (nothing published)."""


class PullResult(BaseModel):
    """What one pull brought into the local store."""

    head: str
    """The local store's HEAD after the merge."""
    remote: str
    store: str


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in the store repo; prompts disabled so a refusal never hangs."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise SyncClientError(
            f"git {' '.join(args[:2])} timed out after {_GIT_TIMEOUT_SECONDS:.0f}s"
        ) from exc


def _git_checked(root: Path, *args: str, refusal: str) -> str:
    """Run git and return stdout, turning a non-zero exit into the refusal."""
    proc = _run_git(root, *args)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise SyncClientError(f"{refusal}: {detail}")
    return proc.stdout


def _refuse_off_loopback(remote: ResolvedRemote, *, operation: str) -> None:
    """Refuse a store-mutating operation aimed off loopback, client-side.

    The daemon's binding policy refuses store mutations from non-loopback
    peers regardless of any flag, so a push at a LAN URL can only end in
    that refusal — the client says so first, with the same policy reason.
    A host that resolves to any non-loopback address is refused; a host
    that resolves to nothing at all is refused too (nothing to push to).
    """
    host = urlparse(remote.base_url).hostname
    if not host:  # HttpUrl guarantees a host; this is belt-and-suspenders
        raise SyncClientError(f"remote {remote.name!r} URL {remote.base_url!r} names no host")
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SyncClientError(
            f"cannot resolve remote {remote.name!r} host {host!r}: {exc}"
        ) from exc
    addresses = {str(info[4][0]) for info in infos}
    exposed = sorted(
        address for address in addresses if not ipaddress.ip_address(address).is_loopback
    )
    if exposed:
        raise SyncClientError(
            f"refusing {operation} to remote {remote.name!r} at {remote.base_url}: its host "
            f"resolves off loopback ({', '.join(exposed)}), and store mutation is "
            "loopback-only regardless of the daemon's --expose-lan — push from the "
            "machine itself (reads may still be exposed for other machines)"
        )
    if not addresses:
        raise SyncClientError(f"remote {remote.name!r} host {host!r} resolves to no addresses")


def _store_root_or_refuse(root: Path) -> None:
    """Refuse a root that is not a git-backed store — sync moves store repos."""
    if not (root / ".git").is_dir():
        raise SyncClientError(
            f"{root} is not a git-backed store root (no .git) — sync moves store "
            "repositories; create one with 'evogine init' or pass --root"
        )


def _dirty_files(root: Path) -> list[str]:
    """The store's uncommitted changes, except the advisory lock file itself.

    ``review.lock`` is the per-store advisory lock — runtime plumbing the
    engine and daemon create by design, never store content, so its
    presence never blocks a sync. Any other local change does.
    """
    proc = _run_git(root, "status", "--porcelain")
    if proc.returncode != 0:
        raise SyncClientError(f"could not read the store's git status: {proc.stderr.strip()}")
    return [
        line
        for line in proc.stdout.splitlines()
        if line.strip() and not line.endswith(f" {_LOCK_FILENAME}")
    ]


def _hosted_head_branch(root: Path, remote: ResolvedRemote) -> str:
    """The hosted store's branch, from the transport's own advertisement."""
    proc = _run_git(root, "ls-remote", "--symref", remote.git_url(), "HEAD")
    if proc.returncode != 0:
        raise SyncClientError(
            f"remote {remote.name!r} not reachable for git transport at "
            f"{remote.git_url()}: {proc.stderr.strip()}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("ref:"):
            ref = line.split("\t")[0].removeprefix("ref:").strip()
            return ref.rsplit("/", 1)[-1]
    raise SyncClientError(
        f"the hosted store {remote.store!r} on {remote.name!r} has no lifecycle "
        "history yet — a zero-commit hosted store supports fetch only; make one "
        "transition on the hosting machine first"
    )


def push_store(root: Path | str, remote: ResolvedRemote) -> PushResult:
    """Publish the local store's committed history to the daemon's store.

    The transport is the daemon's git smart-HTTP surface; the daemon takes
    its own per-store advisory lock across the receive, so a push never
    interleaves with a local lifecycle transition on the hosting machine.
    Writes nothing on refusal.
    """
    root = Path(root)
    _refuse_off_loopback(remote, operation="push")
    _store_root_or_refuse(root)
    if _run_git(root, "rev-parse", "--verify", "HEAD").returncode != 0:
        raise SyncClientError(
            f"the store at {root} has no commits yet — a zero-history store supports "
            "fetch only; to bootstrap a new machine, clone the daemon's URL instead"
        )
    branch = _hosted_head_branch(root, remote)
    proc = _run_git(root, "push", remote.git_url(), f"HEAD:refs/heads/{branch}")
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "fetch first" in stderr or "non-fast-forward" in stderr:
            raise SyncClientError(
                f"push to remote {remote.name!r} refused (non-fast-forward): the hosted "
                "store has commits this store lacks — run 'evogine sync pull' first, "
                "then push"
            )
        raise SyncClientError(f"push to remote {remote.name!r} failed: {stderr}")
    head = _git_checked(
        root, "rev-parse", "HEAD", refusal="could not read the local store's HEAD"
    ).strip()
    uncommitted = len(_dirty_files(root))
    return PushResult(
        head=head, ref=f"refs/heads/{branch}", url=remote.git_url(), uncommitted=uncommitted
    )


def _register_audit_merge_driver(root: Path) -> None:
    """Register the append-only audit union driver in this repo, idempotently.

    The daemon's documented merge surface: one JSON file per record merges
    clean with stock git (disjoint lifecycles touch disjoint files), and
    ``audit.jsonl`` — the one file two machines can both append to — merges
    through the ``evogine-audit`` union driver (:mod:`evogine.sync.merge`).
    Registration is repo-local plumbing (``.git/config`` and
    ``.git/info/attributes``): never committed, so it never dirties the
    store or alters what the daemon serves. The driver runs this same
    interpreter, so ``evogine`` is importable exactly as it is here.
    """
    driver = f"{shlex.quote(sys.executable)} -m evogine.sync.merge %O %A %B"
    for key, value in (
        ("merge.evogine-audit.name", "Evogine audit union (append-only)"),
        ("merge.evogine-audit.driver", driver),
    ):
        _git_checked(
            root, "config", key, value, refusal="could not register the audit merge driver"
        )
    attributes = root / ".git" / "info" / "attributes"
    line = "audit.jsonl merge=evogine-audit"
    existing = attributes.read_text(encoding="utf-8") if attributes.is_file() else ""
    if line not in existing.splitlines():
        attributes.parent.mkdir(parents=True, exist_ok=True)
        with attributes.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def pull_store(root: Path | str, remote: ResolvedRemote) -> PullResult:
    """Fetch the hosted store's history and merge it into the local store.

    The merge uses the daemon's documented low-conflict surface (disjoint
    record files; the audit union driver for ``audit.jsonl``), and the
    engine's per-store advisory lock is held across the merge so a
    concurrent local lifecycle transition cannot interleave. A merge that
    cannot complete cleanly is aborted before refusing: the client, like
    the daemon, never rewrites knowledge — the operator reconciles by hand.
    """
    root = Path(root)
    _store_root_or_refuse(root)
    dirty = _dirty_files(root)
    if dirty:
        raise SyncClientError(
            f"the store at {root} has uncommitted changes ({len(dirty)} file(s)) — "
            "commit them (the store's lifecycle commands do) and re-run the pull; "
            "merging onto a dirty store is refused"
        )
    if _run_git(root, "rev-parse", "--verify", "HEAD").returncode != 0:
        raise SyncClientError(
            f"the store at {root} has no commits yet — bootstrap a new machine by "
            f"cloning the daemon's URL ({remote.git_url()}) instead"
        )
    _register_audit_merge_driver(root)
    _git_checked(
        root,
        "fetch",
        remote.git_url(),
        "HEAD",
        refusal=f"fetch from remote {remote.name!r} failed",
    )
    # The per-store advisory lock, held across the mutating window: a local
    # lifecycle transition cannot interleave with the merge (the daemon
    # holds the same lock class across its receive — one writer at a time
    # per store root, enforced both ends).
    with store_lock(root):
        proc = _run_git(
            root,
            "merge",
            "-m",
            f"sync: merge from remote '{remote.name}' (store '{remote.store}')",
            "FETCH_HEAD",
        )
    if proc.returncode != 0:
        _run_git(root, "merge", "--abort")
        raise SyncClientError(
            f"pull from remote {remote.name!r} hit a conflict git cannot merge "
            "cleanly — the merge was aborted and nothing changed; reconcile the "
            f"store by hand (the client never rewrites knowledge): {proc.stderr.strip()}"
        )
    head = _git_checked(
        root, "rev-parse", "HEAD", refusal="could not read the local store's HEAD"
    ).strip()
    return PullResult(head=head, remote=remote.name, store=remote.store)
