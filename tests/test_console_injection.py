"""Console tests for the knowledge-injection panel and the stores-and-sync selector.

The injection suite drives the real transfer flow end to end: fixture bundles
produced by an actual ``export_bundle`` of a seeded source store, rendered
through the panel's builders and routes — summaries before any action, the
dry run that writes nothing, the one guarded confirm, and every refusal
state (invalid manifest, wrong schema version, duplicate qualified ids,
destination conflicts, self-store import, unapproved injection) as a visible
panel with the store untouched. The selector suite runs the real sync client
against a real uvicorn daemon (the ``test_sync_client`` fixture shape) on a
loopback port: per-source health rows, honest refusal rows for unreachable
remotes, and a selector that only ever changes what pages read. No network
beyond loopback.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

# The console renders through FastAPI/Jinja2 and the selector drives a real
# uvicorn daemon, so this module skips as a whole on core jobs that never
# install the [console] extra (the daemon tests' importorskip precedent).
pytest.importorskip("fastapi")
pytest.importorskip("jinja2")
# importorskip returns the imported module, so this binding IS uvicorn.
uvicorn = pytest.importorskip("uvicorn")

from conftest import make_record, replay_evidence, temp_store_root
from fastapi.testclient import TestClient

from evogine.config import SyncEndpoint, starter_config_yaml
from evogine.console.app import create_app
from evogine.console.injection import IMPORTS_DIRNAME, resolve_imports_dir
from evogine.console.launch import ConsoleError, load_sync_endpoints
from evogine.engine.store import Engine
from evogine.engine.transfer import export_bundle, read_bundle, write_bundle
from evogine.gates.telemetry import DecisionLog, DecisionStage, PromotionDecision

OPERATOR = "nihanth"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def store_root():
    """The console's destination store: an initialized repo, cleaned at teardown."""
    root = temp_store_root("evogine-console-injection-")
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def imports_dir(store_root):
    """The bundles directory the panel reads, created under the store root."""
    path = store_root / IMPORTS_DIRNAME
    path.mkdir()
    return path


def _client(store_root: Path, **kwargs: object) -> TestClient:
    """A console bound to the destination store with the given launch options."""
    app = create_app(store_root, OPERATOR, **kwargs)
    return TestClient(app, follow_redirects=False)


def _promote(engine: Engine, record_id: str, topic: str = "429 backoff") -> str:
    """One deterministic record through the full approve path; returns its proposal id."""
    proposal_id = engine.propose(make_record(record_id, topic=topic))
    engine.queue_replay(record_id, [f"run-{record_id}"])
    engine.promote(record_id, replay_evidence([f"run-{record_id}"]))
    return proposal_id


@pytest.fixture()
def bundle_factory():
    """Export a reviewed bundle from a seeded source store into a bundles dir.

    The source store lives outside the destination root and is removed after
    the export — the bundle file is the only artifact, exactly like a real
    cross-project handoff.
    """
    source_roots: list[Path] = []

    def _factory(
        imports: Path,
        *,
        name: str = "alpha-export",
        records: int = 2,
        store_name: str = "alpha",
    ) -> Path:
        source_root = temp_store_root("evogine-console-source-")
        source_roots.append(source_root)
        engine = Engine(source_root, name=store_name)
        for index in range(records):
            _promote(engine, f"lr-{store_name}-{index}", topic=f"topic {index}")
        bundle = export_bundle(engine)
        return write_bundle(bundle, imports / f"{name}.json")

    yield _factory
    for source_root in source_roots:
        shutil.rmtree(source_root, ignore_errors=True)


def _store_fingerprint(store_root: Path) -> tuple[int, int, str]:
    """(records on disk, audit lines, git porcelain) — the untouched-store triple."""
    records_dir = store_root / "records"
    record_files = list(records_dir.rglob("*.json")) if records_dir.is_dir() else []
    audit = store_root / "audit.jsonl"
    audit_lines = audit.read_text(encoding="utf-8").splitlines() if audit.exists() else []
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=store_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return len(record_files), len(audit_lines), porcelain


def _seed_decision(store_root: Path, index: int) -> None:
    """One approved promotion decision, for overview counts."""
    DecisionLog(store_root).record(
        PromotionDecision(
            proposal_id=f"pr-seed-{index}",
            learning_id=f"lr-remote-{index}",
            promoted=True,
            stage=DecisionStage.APPROVAL,
            verdict="approved",
            reviewer=OPERATOR,
            reason="proposed: 3 binary flips",
        )
    )


# --- the bundles panel: discovery and summaries ------------------------------


def test_resolve_imports_dir_default_and_override(store_root: Path) -> None:
    """The bundles dir is the derived default under the store root, or the override."""
    assert resolve_imports_dir(store_root) == store_root / IMPORTS_DIRNAME
    override = Path("/tmp/some-bundles")
    assert resolve_imports_dir(store_root, override) == override


def test_list_renders_bundle_summaries(store_root: Path, imports_dir: Path, bundle_factory) -> None:
    """A fixture bundle renders its manifest summary before any action."""
    bundle_factory(imports_dir, records=2)
    client = _client(store_root)
    response = client.get("/injection")
    assert response.status_code == 200
    page = response.text
    assert "alpha-export.json" in page
    assert "alpha" in page  # source store identity
    assert "Schema version" in page and "v1" in page
    assert "alpha:lr-alpha-0" in page and "alpha:lr-alpha-1" in page  # qualified ids
    assert "Review dry run" in page  # the only next step is review
    assert 'method="post"' not in page  # nothing is confirmable from the list


def test_list_empty_when_directory_missing(store_root: Path) -> None:
    """No bundles directory at all renders the explicit empty state."""
    client = _client(store_root)
    response = client.get("/injection")
    assert response.status_code == 200
    assert "No import bundles directory" in response.text


def test_list_empty_directory_renders_empty_state(store_root: Path, imports_dir: Path) -> None:
    """An existing but empty directory renders the explicit empty state."""
    client = _client(store_root)
    response = client.get("/injection")
    assert response.status_code == 200
    assert "No export bundles" in response.text


def test_list_renders_invalid_manifest_refusal_panel(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """An invalid manifest renders its refusal panel; healthy bundles still render."""
    bundle_factory(imports_dir)
    (imports_dir / "garbage.json").write_text("not a bundle at all", encoding="utf-8")
    client = _client(store_root)
    response = client.get("/injection")
    assert response.status_code == 200
    page = response.text
    assert "garbage.json" in page and "import refused" in page
    assert "alpha-export.json" in page and "alpha:lr-alpha-0" in page


def test_list_renders_wrong_schema_refusal_panel(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """A bundle declaring a wrong schema version refuses with the expected version named."""
    healthy = read_bundle(bundle_factory(imports_dir))
    wrong_schema = {
        "schema_version": 99,
        "kind": "evogine-memory-export",
        "manifest": healthy.manifest.model_dump(mode="json"),
    }
    (imports_dir / "future.json").write_text(json.dumps(wrong_schema), encoding="utf-8")
    client = _client(store_root)
    response = client.get("/injection")
    assert response.status_code == 200
    page = response.text
    assert "future.json" in page and "import refused" in page
    assert "schema version 1" in page


def test_list_renders_duplicate_id_refusal_panel(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """Duplicate qualified ids inside one bundle refuse before any destination read."""
    path = bundle_factory(imports_dir, records=2)
    bundle = read_bundle(path)
    entries = bundle.manifest.model_dump(mode="json")["records"]
    duplicated = bundle.model_dump(mode="json")
    duplicated["manifest"]["records"] = [entries[0], entries[0]]
    path.write_text(json.dumps(duplicated), encoding="utf-8")
    client = _client(store_root)
    response = client.get("/injection")
    assert response.status_code == 200
    assert "more than once" in response.text and "import refused" in response.text


# --- the bundle detail: dry run, conflicts, guarded confirm ------------------


def test_detail_renders_summary_and_dry_run_form(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """A clean bundle renders the summary, the dry run, and the guarded confirm."""
    bundle_factory(imports_dir, records=2)
    client = _client(store_root)
    response = client.get("/injection/bundles/alpha-export.json")
    assert response.status_code == 200
    page = response.text
    assert "Dry run" in page and "Nothing is written yet" in page
    assert "alpha:lr-alpha-0" in page and "alpha:lr-alpha-1" in page
    assert 'action="/injection/apply"' in page
    assert 'name="confirmed" value="yes"' in page


def test_detail_renders_conflict_panel_after_landing(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """After landing, the same bundle's page refuses with the conflict named."""
    bundle_factory(imports_dir, records=1)
    client = _client(store_root)
    applied = client.post(
        "/injection/apply", data={"bundle": "alpha-export.json", "confirmed": "yes"}
    )
    assert applied.status_code == 303
    response = client.get("/injection/bundles/alpha-export.json")
    assert response.status_code == 200
    page = response.text
    assert "import refused" in page and "collides" in page
    assert 'method="post"' not in page  # a refused import offers no action


def test_detail_renders_self_store_conflict_panel(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """A bundle from a store with the destination's own identity refuses by name."""
    bundle_factory(imports_dir, records=1, store_name="local")
    client = _client(store_root)
    response = client.get("/injection/bundles/alpha-export.json")
    assert response.status_code == 200
    assert "own identity" in response.text  # the reason names the store-identity rule
    assert 'method="post"' not in response.text  # refuse, then offer nothing


def test_detail_rejects_unsafe_bundle_name(store_root: Path) -> None:
    """A bundle name outside the filesystem-safe grammar refuses, naming the rule."""
    client = _client(store_root)
    response = client.get("/injection/bundles/bundle.txt")
    assert response.status_code == 400
    assert "filesystem-safe" in response.text


# --- the guarded write: confirmation, refusals, landing ----------------------


def test_apply_without_confirmation_refuses(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """A confirm POST that skipped the dry run is an unapproved injection — refused."""
    bundle_factory(imports_dir, records=1)
    before = _store_fingerprint(store_root)
    client = _client(store_root)
    response = client.post("/injection/apply", data={"bundle": "alpha-export.json"})
    assert response.status_code == 403
    assert "unapproved injection" in response.text
    assert _store_fingerprint(store_root) == before


def test_apply_invalid_bundle_refuses(store_root: Path, imports_dir: Path) -> None:
    """A POST for an unreadable bundle refuses with the transfer module's reason."""
    (imports_dir / "garbage.json").write_text("not a bundle", encoding="utf-8")
    before = _store_fingerprint(store_root)
    client = _client(store_root)
    response = client.post(
        "/injection/apply", data={"bundle": "garbage.json", "confirmed": "yes"}
    )
    assert response.status_code == 422
    assert "nothing was applied" in response.text
    assert _store_fingerprint(store_root) == before


def test_apply_conflicting_bundle_refuses(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """A confirmed POST that would collide with existing ids refuses before writing."""
    bundle_factory(imports_dir, records=1)
    client = _client(store_root)
    first = client.post(
        "/injection/apply", data={"bundle": "alpha-export.json", "confirmed": "yes"}
    )
    assert first.status_code == 303
    before = _store_fingerprint(store_root)
    second = client.post(
        "/injection/apply", data={"bundle": "alpha-export.json", "confirmed": "yes"}
    )
    assert second.status_code == 409
    assert "nothing was applied" in second.text
    assert _store_fingerprint(store_root) == before


def test_apply_rejects_unsafe_bundle_name(store_root: Path) -> None:
    """A POST naming a path-shaped bundle refuses before touching anything."""
    client = _client(store_root)
    response = client.post(
        "/injection/apply", data={"bundle": "../escape.json", "confirmed": "yes"}
    )
    assert response.status_code == 400
    assert "filesystem-safe" in response.text


def test_apply_confirmed_lands_candidates_and_audits(
    store_root: Path, imports_dir: Path, bundle_factory
) -> None:
    """The confirmed import lands candidates with provenance, audit, and the banner.

    Landed records are candidates — never promoted knowledge: they face this
    store's gate and the human approval queue, exactly like swarm output.
    """
    bundle_factory(imports_dir, records=2)
    before = _store_fingerprint(store_root)
    client = _client(store_root)
    applied = client.post(
        "/injection/apply", data={"bundle": "alpha-export.json", "confirmed": "yes"}
    )
    assert applied.status_code == 303
    assert applied.headers["location"] == "/injection/bundles/alpha-export.json?applied=2"

    engine = Engine(store_root)
    for record_id in ("lr-alpha-0", "lr-alpha-1"):
        record = engine.get(record_id)
        assert record.status == "candidate"
        # Bundle payloads travel verbatim — the source record's own provenance,
        # not a rewrite. The import lineage is the audit event and manifest.
        assert record.provenance.origin == "test-fixture"
    after = _store_fingerprint(store_root)
    assert after[1] == before[1] + 2  # one audit event per landed record
    assert after[2] == ""  # the landing committed cleanly — no dirty store

    followed = client.get("/injection/bundles/alpha-export.json?applied=2")
    assert followed.status_code == 200
    assert "import applied" in followed.text and "candidates" in followed.text


# --- stores and sync: per-source health and the selector ---------------------


@pytest.fixture()
def daemon_factory():
    """Serve seeded stores behind real uvicorn servers; stop them at teardown."""
    from evogine.sync.app import create_app as create_sync_app
    from evogine.sync.settings import SyncSettings

    running: list[tuple[object, threading.Thread]] = []

    def _factory(stores_root: Path, store: str, decisions: int = 2) -> tuple[str, Path]:
        root = stores_root / store
        Engine(root)
        for index in range(decisions):
            _seed_decision(root, index)
        app = create_sync_app(SyncSettings(stores_root=stores_root))
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


def _endpoint(name: str, url: str, store: str) -> SyncEndpoint:
    return SyncEndpoint(name=name, url=url, store=store)


def test_local_row_counts_and_audit_position(store_root: Path) -> None:
    """The local row reads the store's own builders: overview counts, last audit event."""
    _promote(Engine(store_root), "lr-1")
    _promote(Engine(store_root), "lr-2")
    _seed_decision(store_root, 0)
    client = _client(store_root)
    response = client.get("/sources")
    assert response.status_code == 200
    page = response.text
    assert "reachable" in page and "local" in page
    assert "1 decision(s)" in page  # the overview builder's count, not a new accounting
    assert "seq" in page  # the audit position renders from the trail


def test_unreachable_remote_renders_refusal_row(store_root: Path) -> None:
    """A remote that will not answer renders its refusal — never a blank row."""
    dead = _endpoint("ghost", f"http://127.0.0.1:{_free_port()}", "ghost-store")
    client = _client(store_root, sync_endpoints=[dead])
    response = client.get("/sources")
    assert response.status_code == 200
    page = response.text
    assert "ghost" in page and "refused" in page
    assert "no overview served" in page


def test_remote_row_renders_query_status(
    store_root: Path, tmp_path, daemon_factory
) -> None:
    """A live remote renders the query API's counts, spend, and the audit note."""
    url, _ = daemon_factory(tmp_path / "fleet", "beta", decisions=2)
    remote = _endpoint("fleet-b", url, "beta")
    client = _client(store_root, sync_endpoints=[remote])
    response = client.get("/sources")
    assert response.status_code == 200
    page = response.text
    assert "fleet-b" in page and "reachable" in page
    assert "2 decision(s)" in page
    assert "no spend recorded (0 run(s))" in page
    assert "never stitched" in page  # the audit absence is stated, not blank


def test_sources_page_selector_links(store_root: Path, tmp_path, daemon_factory) -> None:
    """The selector renders one link per source, the current one marked."""
    url, _ = daemon_factory(tmp_path / "fleet", "beta", decisions=0)
    remote = _endpoint("fleet-b", url, "beta")
    client = _client(store_root, sync_endpoints=[remote])
    page = client.get("/sources").text
    # Local is selected by default: marked, not a link; the remote is a link.
    assert "source-selected" in page
    assert "/sources?source=fleet-b" in page
    # Selecting the remote flips both: fleet-b marked, local a link.
    remote_page = client.get("/sources?source=fleet-b").text
    assert "/sources?source=local" in remote_page
    assert "source-selected" in remote_page


def test_overview_remote_source_renders_query_view(
    store_root: Path, tmp_path, daemon_factory
) -> None:
    """The overview page targets the selected remote through the query API."""
    url, _ = daemon_factory(tmp_path / "fleet", "beta", decisions=2)
    remote = _endpoint("fleet-b", url, "beta")
    client = _client(store_root, sync_endpoints=[remote])
    response = client.get("/?source=fleet-b")
    assert response.status_code == 200
    page = response.text
    assert "remote source" in page and "fleet-b" in page
    assert "2 promotion decision(s)" in page
    assert "read-only" in page  # the page says what the selector does not change


def test_overview_unknown_source_refuses(store_root: Path) -> None:
    """An unknown source name refuses with what is configured — no silent local read."""
    client = _client(store_root)
    response = client.get("/?source=ghost")
    assert response.status_code == 400
    assert "unknown store context" in response.text
    assert "no sync remotes are configured" in response.text


def test_overview_unreachable_source_refuses(store_root: Path) -> None:
    """A configured-but-dead source refuses the overview with the client's reason."""
    dead = _endpoint("ghost", f"http://127.0.0.1:{_free_port()}", "ghost-store")
    client = _client(store_root, sync_endpoints=[dead])
    response = client.get("/?source=ghost")
    assert response.status_code == 503
    assert "ghost" in response.text


def test_selector_absent_without_remotes(store_root: Path) -> None:
    """With no remotes configured, the overview page renders no selector strip."""
    client = _client(store_root)
    response = client.get("/")
    assert response.status_code == 200
    assert '<div class="sources-bar">' not in response.text  # the strip, not its CSS


def test_injection_and_sources_in_nav(store_root: Path) -> None:
    """Both new surfaces are reachable from every page's nav."""
    client = _client(store_root)
    page = client.get("/").text
    assert 'href="/injection"' in page
    assert 'href="/sources"' in page


# --- launch-time config: endpoints and refusal before any socket -------------


def test_load_sync_endpoints_none_means_empty() -> None:
    """No --config means no remotes: the selector shows the local store alone."""
    assert load_sync_endpoints(None) == ()


def test_load_sync_endpoints_reads_default_profile(tmp_path) -> None:
    """The selector reads the config's default profile's endpoints."""
    config = yaml.safe_load(starter_config_yaml(str(tmp_path)))
    config["profiles"]["default"]["sync"]["endpoints"] = [
        {"name": "fleet-b", "url": "http://127.0.0.1:8766", "store": "beta"}
    ]
    path = tmp_path / "evogine.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    endpoints = load_sync_endpoints(str(path))
    assert len(endpoints) == 1
    assert endpoints[0].name == "fleet-b"
    assert endpoints[0].store == "beta"


def test_load_sync_endpoints_refuses_broken_config(tmp_path) -> None:
    """A config that refuses to load refuses the launch — never a silent empty list."""
    path = tmp_path / "broken.yaml"
    path.write_text("profiles: [not, a, mapping", encoding="utf-8")
    with pytest.raises(ConsoleError, match="unusable"):
        load_sync_endpoints(str(path))


def test_serve_refuses_missing_imports_dir(store_root: Path) -> None:
    """A typo'd --imports-dir refuses the launch before any socket exists."""
    Engine(store_root)  # serve validates the store root first
    from evogine.console.server import serve

    with pytest.raises(ConsoleError, match="import bundles directory does not exist"):
        serve(
            str(store_root),
            OPERATOR,
            imports_dir=str(store_root / "no-such-dir"),
        )
