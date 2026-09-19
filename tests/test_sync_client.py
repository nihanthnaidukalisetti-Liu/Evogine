"""Offline tests for the sync client: query reads, guarded transport, roll-up.

The daemon runs in-process behind a real uvicorn server on an ephemeral
loopback port — the same fixture shape ``test_sync_transport.py`` uses —
and the client module (``evogine.sync.client``) runs against it: status
reads over fixtures, transport round-trips (push then pull through the
client, not bare git), the merge surface under the per-store advisory
lock, cross-machine spend aggregation with exact ``Decimal`` sums, and
every refusal state the brief requires: unreachable, misconfigured,
policy-refusing (non-loopback push), non-fast-forward, dirty store,
conflicting merge. No network beyond loopback.
"""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

# The client drives a real uvicorn server in this module, so it skips as a
# whole on core jobs that never install the [server] extra (the daemon
# tests' precedent) and runs for real in the dedicated sync CI job.
fastapi = pytest.importorskip("fastapi")
# importorskip returns the imported module, so this binding IS uvicorn.
uvicorn = pytest.importorskip("uvicorn")

from evogine.bench.budgets import SpendReconciliation
from evogine.cli import CheckState, _check_sync_endpoint, main
from evogine.config import SyncEndpoint, starter_config_yaml
from evogine.engine.store import Engine
from evogine.gates.telemetry import DecisionLog, DecisionStage, PromotionDecision
from evogine.sync.app import create_app
from evogine.sync.client import (
    RemoteSpend,
    ResolvedRemote,
    SyncClientError,
    _get_json,
    fetch_overview,
    fetch_spend,
    fleet_status,
    merge_spend,
    pull_store,
    push_store,
    resolve_endpoint,
    resolve_remote,
)
from evogine.sync.settings import SyncSettings

_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def daemon_factory(record_factory):  # type: ignore[no-untyped-def]
    """Serve seeded stores behind real uvicorn servers; stop them at teardown."""
    running: list[tuple[object, threading.Thread]] = []

    def _factory(stores_root: Path, store: str, seeds: int = 0) -> tuple[str, Path]:
        root = stores_root / store
        engine = Engine(root)
        for index in range(seeds):
            engine.propose(record_factory(f"lr-seed-{index}"))
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
        running.append((server, thread))
        return f"http://127.0.0.1:{port}", root

    yield _factory
    for server, thread in running:
        server.should_exit = True
        thread.join(timeout=10)


def _git_commit_all(root: Path, message: str) -> None:
    """Commit everything in the store tree with a pinned test identity."""
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True, env=_GIT_ENV)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=evogine-test",
            "-c",
            "user.email=evogine-test@example.invalid",
            "commit",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        env=_GIT_ENV,
    )


def _clone(url: str, dest: Path) -> Path:
    subprocess.run(["git", "clone", url, str(dest)], check=True, capture_output=True, env=_GIT_ENV)
    return dest


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    ).stdout.strip()


def _tree(root: Path) -> dict[str, bytes]:
    """Every store file, byte-for-byte, ignoring git's own plumbing."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def _seed_spend(store_root: Path, run_id: str, actual: str, ceiling: str = "1.00") -> None:
    """Write one bench spend report into the store's runs tree and commit it.

    The daemon's spend query reads ``runs/bench/<run>/artifacts/spend-report.json``
    from the hosted store's working tree, so this is the real row shape the
    query API serves — committed so the transport carries it too.
    """
    artifacts = store_root / "runs" / "bench" / run_id / "artifacts"
    artifacts.mkdir(parents=True)
    actual_usd, ceiling_usd = Decimal(actual), Decimal(ceiling)
    (artifacts / "spend-report.json").write_text(
        SpendReconciliation(
            recorded_calls=3,
            calls_with_usage=3,
            calls_without_usage=0,
            actual_cost_usd=actual_usd,
            ceiling_usd=ceiling_usd,
            within_ceiling=actual_usd <= ceiling_usd,
            overage_usd=max(Decimal(0), actual_usd - ceiling_usd),
        ).model_dump_json(),
        encoding="utf-8",
    )
    _git_commit_all(store_root, f"seed spend {run_id}")


def _endpoint(name: str, url: str, store: str | None = None) -> SyncEndpoint:
    return SyncEndpoint(name=name, url=url, store=store)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Endpoint/remote resolution — misconfiguration refuses before any network.
# ---------------------------------------------------------------------------


def test_resolve_endpoint_requires_store_and_normalizes() -> None:
    remote = resolve_endpoint(_endpoint("laptop", "http://127.0.0.1:8766/", " Alpha "))
    assert remote.store == "alpha"
    assert remote.base_url == "http://127.0.0.1:8766"
    assert remote.git_url() == "http://127.0.0.1:8766/git/alpha"

    with pytest.raises(SyncClientError, match="declares no store"):
        resolve_endpoint(_endpoint("laptop", "http://127.0.0.1:8766", None))

    with pytest.raises(SyncClientError, match="unusable store name"):
        resolve_endpoint(_endpoint("laptop", "http://127.0.0.1:8766", "Not A Store!"))


def test_resolve_remote_selection_rules() -> None:
    laptop = _endpoint("laptop", "http://127.0.0.1:8766", "alpha")
    desktop = _endpoint("desktop", "http://127.0.0.1:8767", "alpha")

    assert resolve_remote([laptop], None) == resolve_endpoint(laptop)
    assert resolve_remote([laptop, desktop], "desktop").name == "desktop"

    with pytest.raises(SyncClientError, match="--remote is required"):
        resolve_remote([laptop, desktop], None)
    with pytest.raises(SyncClientError, match="unknown sync remote"):
        resolve_remote([laptop, desktop], "tablet")
    with pytest.raises(SyncClientError, match="no sync endpoints configured"):
        resolve_remote([], None)


# ---------------------------------------------------------------------------
# Status reads over the query API, and the refusal rows around them.
# ---------------------------------------------------------------------------


def test_status_reads_overview_and_spend_from_query_api(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_url, hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=2)
    DecisionLog(hosted).record(
        PromotionDecision(
            proposal_id="lr-seed-0",
            learning_id="lr-seed-0",
            promoted=False,
            stage=DecisionStage.REPLAY,
            reason="no_flip: 0/3 replays changed verdict",
        )
    )
    _git_commit_all(hosted, "seed decision")
    _seed_spend(hosted, "run-1", "0.0826")

    remote = resolve_endpoint(_endpoint("laptop", base_url, "alpha"))
    overview = fetch_overview(remote)
    assert overview.total_decisions == 1
    assert overview.by_stage["replay"] == 1
    assert overview.promoted == 0
    assert overview.pending_approvals == 0

    spend = fetch_spend(remote)
    assert spend.remote == "laptop"
    assert spend.recorded_runs == 1
    assert spend.actual_cost_usd == Decimal("0.0826")
    assert spend.rows[0].run_id == "run-1"
    assert spend.rows[0].reconciliation.recorded_calls == 3


def test_status_refuses_unreachable_and_misconfigured_but_reports_the_rest(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    dead_port = _free_port()  # nothing listens here
    endpoints = [
        _endpoint("laptop", base_url, "alpha"),
        _endpoint("desktop", f"http://127.0.0.1:{dead_port}", "alpha"),
        _endpoint("tablet", "http://127.0.0.1:8799", None),  # misconfigured
    ]
    statuses = fleet_status(endpoints)
    assert [status.label for status in statuses] == ["laptop", "desktop", "tablet"]
    assert statuses[0].ok
    assert statuses[0].spend is not None
    assert statuses[0].overview is not None
    assert not statuses[1].ok
    assert statuses[1].refusal is not None
    assert "unreachable" in statuses[1].refusal
    assert "desktop" in statuses[1].refusal
    assert not statuses[2].ok
    assert statuses[2].refusal is not None
    assert "declares no store" in statuses[2].refusal

    # The merged view sums only reporting machines — refusals are visible
    # rows, never silently-zero contributions.
    merged = merge_spend(
        [status.spend for status in statuses if status.ok and status.spend is not None]
    )
    assert [machine.remote for machine in merged.machines] == ["laptop"]


def test_query_refuses_http_errors_and_bad_payload_shapes(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)

    # A store the daemon does not host: HTTP 404 with the daemon's error body.
    ghost = ResolvedRemote(name="ghost", store="gamma", base_url=base_url)
    with pytest.raises(SyncClientError, match="HTTP 404"):
        fetch_overview(ghost)

    # A daemon that answers but not with the documented shapes: one raw
    # server, path-addressed to trigger each refusal branch.
    class _OddHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            # Real path branches: store alpha's overview is non-JSON, store
            # beta's overview is an object of the wrong shape, alpha's spend
            # row shape is unusable, and a direct /array fetch returns an
            # array so the non-object refusal is tested on its own.
            if "/overview" in self.path and "stores/alpha" in self.path:
                body, ctype = b"not json at all", "text/plain"
            elif "/overview" in self.path:  # stores/beta
                body, ctype = json.dumps({"total_decisions": 0}).encode(), "application/json"
            elif "/spend" in self.path:
                body, ctype = (
                    json.dumps(
                        {"rows": [{"run_id": "x", "reconciliation": {"bogus": 1}}]}
                    ).encode(),
                    "application/json",
                )
            elif "/array" in self.path:
                body, ctype = b"[1, 2]", "application/json"
            else:
                body, ctype = b"{}", "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return None  # keep test output clean

    port = _free_port()
    raw = http.server.HTTPServer(("127.0.0.1", port), _OddHandler)
    thread = threading.Thread(target=raw.serve_forever, daemon=True)
    thread.start()
    try:
        odd = ResolvedRemote(name="odd", store="alpha", base_url=f"http://127.0.0.1:{port}")
        with pytest.raises(SyncClientError, match="not with JSON"):
            fetch_overview(odd)  # the alpha overview path serves non-JSON
        odd_beta = ResolvedRemote(name="odd", store="beta", base_url=f"http://127.0.0.1:{port}")
        with pytest.raises(SyncClientError, match="does not match the overview schema"):
            fetch_overview(odd_beta)  # the beta overview path serves the wrong shape
        with pytest.raises(SyncClientError, match="non-object JSON"):
            _get_json(odd, "/api/stores/alpha/array", timeout=5.0)
        with pytest.raises(SyncClientError, match="spend schema"):
            fetch_spend(odd)  # the spend path serves an unusable shape
    finally:
        raw.shutdown()
        raw.server_close()


# ---------------------------------------------------------------------------
# Transport: push and pull through the client, over real git.
# ---------------------------------------------------------------------------


def test_push_pull_round_trip_over_transport(
    daemon_factory,
    tmp_path,
    record_factory,  # type: ignore[no-untyped-def]
) -> None:
    base_url, hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    remote = resolve_endpoint(_endpoint("laptop", base_url, "alpha"))

    clone = _clone(remote.git_url(), tmp_path / "clone-a")
    Engine(clone).propose(record_factory("lr-remote-1"))
    pushed = push_store(clone, remote)
    assert pushed.head == _head(clone)
    # The published ref is the local branch the store sits on (the client
    # discovers the hosted HEAD branch rather than assuming a name).
    local_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    ).stdout.strip()
    assert pushed.ref == f"refs/heads/{local_branch}"
    assert (hosted / "records" / "lr-remote-1.json").is_file()

    # A second machine advances its own copy, pushes, and the first catches
    # up by pulling through the client — one file per record merges clean,
    # and the audit union driver keeps the append-only trail merged.
    clone_b = _clone(remote.git_url(), tmp_path / "clone-b")
    Engine(clone_b).propose(record_factory("lr-remote-2"))
    push_store(clone_b, remote)
    pulled = pull_store(clone, remote)
    assert pulled.remote == "laptop"
    assert pulled.store == "alpha"
    assert _tree(clone) == _tree(hosted)
    assert _tree(clone) == _tree(clone_b)
    assert (clone / "records" / "lr-remote-2.json").is_file()
    audit = [
        line for line in (clone / "audit.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(audit) == 3  # seed + remote-1 + remote-2, union-merged

    # The merge driver registers repo-locally and git's own parser sees it
    # (the pull above already ran it — the audit trail is union-merged).
    registered_driver = subprocess.run(
        ["git", "config", "--get", "merge.evogine-audit.driver"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    ).stdout.strip()
    assert "evogine.sync.merge" in registered_driver
    assert _head(clone) == _head(hosted)


def test_pull_refuses_dirty_and_zero_commit_stores(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    remote = resolve_endpoint(_endpoint("laptop", base_url, "alpha"))

    clone = _clone(remote.git_url(), tmp_path / "clone-dirty")
    (clone / "stray.txt").write_text("uncommitted local change", encoding="utf-8")
    with pytest.raises(SyncClientError, match="uncommitted changes"):
        pull_store(clone, remote)

    # A root that is not a git-backed store at all: sync moves store
    # repositories, so it refuses rather than guessing.
    with pytest.raises(SyncClientError, match="not a git-backed store"):
        pull_store(tmp_path / "not-a-store", remote)

    # The advisory lock file itself is runtime plumbing, never store
    # content: its presence alone never blocks a pull.
    empty_repo = tmp_path / "empty-repo"
    subprocess.run(
        ["git", "init", "-q", str(empty_repo)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=_GIT_ENV,
    )
    (empty_repo / "review.lock").write_text("", encoding="utf-8")
    with pytest.raises(SyncClientError, match="no commits yet"):
        pull_store(empty_repo, remote)


def test_push_refuses_non_loopback_targets(
    daemon_factory,
    tmp_path,
    record_factory,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    clone = _clone(f"{base_url}/git/alpha", tmp_path / "clone-policy")
    Engine(clone).propose(record_factory("lr-policy-1"))

    # A private address (resolves, but not loopback): the daemon would
    # refuse the mutation regardless of any flag — the client refuses first,
    # with the policy reason, and nothing leaves the machine.
    lan = resolve_endpoint(_endpoint("nas", "http://10.0.0.5:8766", "alpha"))
    with pytest.raises(SyncClientError, match="loopback"):
        push_store(clone, lan)

    # A hostname that resolves to nothing: nothing to push to.
    unresolvable = resolve_endpoint(_endpoint("ghost", "http://no-such-host.invalid:8766", "alpha"))
    with pytest.raises(SyncClientError, match="no-such-host"):
        push_store(clone, unresolvable)


def test_push_refuses_non_fast_forward(
    daemon_factory,
    tmp_path,
    record_factory,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    remote = resolve_endpoint(_endpoint("laptop", base_url, "alpha"))

    clone_a = _clone(remote.git_url(), tmp_path / "clone-ff-a")
    clone_b = _clone(remote.git_url(), tmp_path / "clone-ff-b")
    Engine(clone_a).propose(record_factory("lr-ff-a"))
    Engine(clone_b).propose(record_factory("lr-ff-b"))

    push_store(clone_a, remote)  # the hosted store advances
    with pytest.raises(SyncClientError, match="non-fast-forward"):
        push_store(clone_b, remote)  # clone_b is now behind it


def test_push_to_zero_commit_hosted_store_refuses(
    daemon_factory,
    tmp_path,
    record_factory,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "fresh", seeds=0)
    remote = resolve_endpoint(_endpoint("laptop", base_url, "fresh"))

    # A local store WITH history pushing at a hosted store with none: the
    # daemon documents zero-commit hosted stores as fetch-only, and the
    # transport's advertisement is empty — the refusal says so.
    local_root = tmp_path / "local-full"
    Engine(local_root).propose(record_factory("lr-local-1"))
    with pytest.raises(SyncClientError, match="no lifecycle history"):
        push_store(local_root, remote)

    # And a clone OF the zero-commit store has no commits of its own.
    clone = _clone(remote.git_url(), tmp_path / "clone-empty")
    with pytest.raises(SyncClientError, match="no commits yet"):
        push_store(clone, remote)


def test_pull_refuses_conflict_and_leaves_the_store_clean(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    remote = resolve_endpoint(_endpoint("laptop", base_url, "alpha"))

    clone_a = _clone(remote.git_url(), tmp_path / "clone-c-a")
    clone_b = _clone(remote.git_url(), tmp_path / "clone-c-b")
    seed_rel = Path("records") / "lr-seed-0.json"

    # Both machines rewrite the SAME record file with different content:
    # one JSON file per record merges clean only when lifecycles touch
    # disjoint files — same-file divergence is a true conflict.
    (clone_a / seed_rel).write_text('{"version": "A"}', encoding="utf-8")
    (clone_b / seed_rel).write_text('{"version": "B"}', encoding="utf-8")
    _git_commit_all(clone_a, "rewrite A")
    _git_commit_all(clone_b, "rewrite B")

    push_store(clone_a, remote)
    head_before = _head(clone_b)
    with pytest.raises(SyncClientError, match="conflict"):
        pull_store(clone_b, remote)

    # The abort restored everything: same HEAD, clean status, no MERGE_HEAD.
    assert _head(clone_b) == head_before
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=clone_b,
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )
    assert status.stdout.strip() == ""
    assert not (clone_b / ".git" / "MERGE_HEAD").exists()


# ---------------------------------------------------------------------------
# Cross-machine spend aggregation — exact Decimal accounting.
# ---------------------------------------------------------------------------


def test_merge_spend_is_exact_decimal_with_honest_absence() -> None:
    laptop = RemoteSpend(
        remote="laptop",
        store="alpha",
        recorded_runs=1,
        actual_cost_usd=Decimal("0.0001"),
        overage_usd=Decimal(0),
        within_ceiling=True,
    )
    desktop = RemoteSpend(
        remote="desktop",
        store="alpha",
        recorded_runs=1,
        actual_cost_usd=Decimal("0.0002"),
        overage_usd=Decimal(0),
        within_ceiling=True,
    )
    merged = merge_spend([laptop, desktop])
    assert merged.recorded_runs == 2
    assert merged.total_actual_cost_usd == Decimal("0.0003")
    assert str(merged.total_actual_cost_usd) == "0.0003"  # exact base-10, no float drift
    assert merged.within_ceiling is True

    # A machine with no recorded spend keeps its attribution row and adds
    # nothing — absence is not zero.
    idle = RemoteSpend(remote="idle", store="alpha")
    merged = merge_spend([laptop, idle])
    assert merged.machines[1].actual_cost_usd is None
    assert merged.recorded_runs == 1
    assert merged.total_actual_cost_usd == Decimal("0.0001")

    # Any reporting machine breaching its ceiling fails the fleet view.
    breached = RemoteSpend(
        remote="breached",
        store="alpha",
        recorded_runs=1,
        actual_cost_usd=Decimal("2.00"),
        overage_usd=Decimal("1.00"),
        within_ceiling=False,
    )
    merged = merge_spend([laptop, breached])
    assert merged.within_ceiling is False
    assert merged.total_overage_usd == Decimal("1.00")

    # Nothing reported anywhere: the totals say so instead of showing $0.
    empty = merge_spend([])
    assert empty.machines == []
    assert empty.total_actual_cost_usd is None
    assert empty.within_ceiling is None


def test_spend_aggregates_exactly_across_two_remotes(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_a, hosted_a = daemon_factory(tmp_path / "stores-a", "alpha", seeds=1)
    base_b, hosted_b = daemon_factory(tmp_path / "stores-b", "alpha", seeds=1)
    _seed_spend(hosted_a, "run-1", "0.0826")
    _seed_spend(hosted_b, "run-1", "0.0174")

    statuses = fleet_status(
        [
            _endpoint("laptop", base_a, "alpha"),
            _endpoint("desktop", base_b, "alpha"),
        ]
    )
    assert all(status.ok for status in statuses)
    merged = merge_spend([status.spend for status in statuses if status.spend is not None])
    assert merged.recorded_runs == 2
    assert merged.total_actual_cost_usd == Decimal("0.1000")  # exact; float drift would fail
    assert merged.total_overage_usd == Decimal("0.0000")
    assert merged.within_ceiling is True
    assert [machine.remote for machine in merged.machines] == ["laptop", "desktop"]
    assert merged.machines[0].actual_cost_usd == Decimal("0.0826")
    assert merged.machines[1].actual_cost_usd == Decimal("0.0174")


# ---------------------------------------------------------------------------
# The CLI surface: exit codes and the merged view's visible refusals.
# ---------------------------------------------------------------------------


def _write_config(path: Path, learning_root: Path, endpoints: list[dict[str, str]]) -> None:
    """A real profile config: the starter, with the sync endpoints replaced."""
    config_data = yaml.safe_load(starter_config_yaml(str(learning_root)))
    assert isinstance(config_data, dict)  # narrowing for mypy
    config_data["profiles"]["default"]["sync"]["endpoints"] = endpoints
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")


def test_cli_status_push_and_pull(
    daemon_factory,
    tmp_path,
    record_factory,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    base_url, hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    _seed_spend(tmp_path / "stores" / "alpha", "run-1", "0.0826")
    clone = _clone(f"{base_url}/git/alpha", tmp_path / "clone")
    Engine(clone).propose(record_factory("lr-cli-1"))

    config = tmp_path / "evogine.yaml"
    _write_config(config, clone, [{"name": "laptop", "url": base_url, "store": "alpha"}])

    assert main(["sync", "push", "--config", str(config)]) == 0
    assert (hosted / "records" / "lr-cli-1.json").is_file()

    assert main(["sync", "status", "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "laptop (store 'alpha')" in output
    assert "merged spend (1 of 1 machine(s) reporting)" in output
    assert "$0.0826" in output

    assert main(["sync", "pull", "--remote", "laptop", "--config", str(config)]) == 0
    assert "local HEAD is now" in capsys.readouterr().out


def test_cli_status_shows_refusals_and_exits_nonzero(
    daemon_factory,
    tmp_path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    clone = _clone(f"{base_url}/git/alpha", tmp_path / "clone")
    dead_port = _free_port()
    config = tmp_path / "evogine.yaml"
    _write_config(
        config,
        clone,
        [
            {"name": "laptop", "url": base_url, "store": "alpha"},
            {"name": "desktop", "url": f"http://127.0.0.1:{dead_port}", "store": "alpha"},
        ],
    )

    assert main(["sync", "status", "--config", str(config)]) == 1
    output = capsys.readouterr().out
    assert "desktop: REFUSED" in output
    assert "1 of 2 machine(s) reporting" in output
    assert "refusals are never summed away" in output


def test_cli_refuses_unknown_remote_and_empty_endpoints(
    daemon_factory,
    tmp_path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    clone = _clone(f"{base_url}/git/alpha", tmp_path / "clone")

    config = tmp_path / "evogine.yaml"
    _write_config(config, clone, [{"name": "laptop", "url": base_url, "store": "alpha"}])
    assert main(["sync", "push", "--remote", "tablet", "--config", str(config)]) == 1
    assert "unknown sync remote" in capsys.readouterr().err

    bare = tmp_path / "bare.yaml"
    _write_config(bare, clone, [])
    assert main(["sync", "status", "--config", str(bare)]) == 1
    assert "no sync endpoints configured" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Doctor keeps probing the URL, and now flags a missing store name.
# ---------------------------------------------------------------------------


def test_doctor_warns_when_endpoint_declares_no_store(
    daemon_factory,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    base_url, _hosted = daemon_factory(tmp_path / "stores", "alpha", seeds=1)
    result = _check_sync_endpoint(_endpoint("laptop", base_url, None))
    assert result.state is CheckState.WARN
    assert "declares no store" in result.detail

    result = _check_sync_endpoint(_endpoint("laptop", base_url, "alpha"))
    assert result.state is CheckState.OK
    assert "store 'alpha'" in result.detail
