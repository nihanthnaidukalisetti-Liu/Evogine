"""Console entry point for the ``evogine`` command.

Subcommands grow with the engine. ``decisions report`` renders promotion
telemetry — decision counts by gate stage — from a store root's decision
log, so the 0/11-style silent failures of V1 cannot happen again: every
decided proposal's verdict is aggregated here. ``decisions funnel`` (V3)
aggregates the same records into the proposal-quality funnel — candidates
→ replay-cleared → materiality-cleared → approved → promoted, per pass and
cumulatively, with the refusal-reason breakdown over the machine-checked
vocabulary and live-run token evidence when recorded — from files that
already exist (decisions, pass manifests, approvals, records).
``approvals`` is the
human-approval promotion boundary (V2-6): the memory gate renders
promotion proposals into the store's approval log, and
``approvals list|approve|reject`` is how a person reviews and decides
them — approve applies the promotion, reject refuses it and demotes the
record. The gate never promotes on its own.

``run`` drives a full promotion pass over a store root: it wires the memory
gate to the bench-backed replay runner (HarnessReplayRunner) with the
offline KnowledgeRunner, decides every candidate learning, records one
telemetry decision per proposal, and renders approval proposals — exiting
non-zero when any gate-error decision was made. ``bench`` runs the
warm-vs-cold harness over the default corpus and reports pass rates; it is
informational and always exits zero when the run completes.

``bench --runner live`` is the V3 W0 spend seam: it drives the swarm model
client for real, so its gates are checked before anything else — a
pre-registered protocol (``--protocol``) and a configured provider key are
required, and pre-run validation fails closed: a projected spend over the
dollar ceiling, a repeat count below the floor, or a contaminated holdout
(the retrieval-only recall probe runs first, before any client exists)
each refuse the run before the first dollar is spent. ``shadow`` (V3 W2)
profiles a candidate retrieval configuration over that corpus without
touching lifecycle state: the store is handed over through a read-only
seam, results land only under ``runs/shadow/<candidate_id>/``, and a
contaminated or over-ceiling run is refused before anything is evaluated —
exiting non-zero with $0 spent. Both operate on an explicit ``--root``:
they create or read a store, and silently defaulting to the working
directory could benchmark or mutate the wrong store (git-safety).

``init`` (V4 productization) is the first-run wizard: it creates the
learning store, writes the starter ``evogine.yaml`` — refusing to
overwrite an existing config — and reports the provider key's state,
read from the environment by name and never echoed or stored.
``doctor`` renders environment readiness — config, git, Docker,
provider key, disk, sync endpoints — with every state explicit (ok /
warn / fail, each with its fix where one exists), exiting non-zero only
when a check fails.

``memory export`` (V4 federation) writes an export bundle of a store's
promoted pool — a manifest with per-record qualified ids plus the verbatim
record payloads, the ``MergedSource`` provenance pattern extended across
stores. ``memory import`` is the explicit, human-reviewed landing: without
``--yes`` it is a dry-run summary of what would land and nothing is
written; with ``--yes`` the reviewed records enter the destination as new
candidates — provenance verbatim, confidence clock re-anchored to the
destination (destination-clock decay stamping), source evidence left in
its own trail, the landing recorded in the audit trail referencing the
source by qualified id only. Both commands inject store paths through the
federation registry (``--store`` names the scope, ``--root`` supplies the
path); malformed or conflicting bundles are refused all-or-nothing.

``sync status|push|pull`` (V4) is the client half of the git-native sync
daemon: the config's sync-endpoints section becomes live — a named remote
is a daemon URL plus the hosted store it serves. ``status`` queries every
configured remote and rolls spend into one merged view with per-machine
attribution (exact ``Decimal`` accounting); ``push`` publishes local store
commits over the daemon transport (loopback-only, non-fast-forward
refused); ``pull`` fetches and merges using the documented low-conflict
merge surface under the engine's per-store advisory lock. Every command
fails closed: an unreachable, misconfigured, or policy-refusing remote is
rendered as the refusal it is — never a silent skip, never a fail-open
write.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from enum import Enum
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

from evogine import __version__
from evogine.bench.budgets import (
    DollarBudget,
    SpendCeilingExceededError,
    SpendReconciliation,
    queue_spend,
    reconcile_spend,
)
from evogine.bench.cases import BenchCase, default_corpus
from evogine.bench.gate_replay import HarnessReplayRunner
from evogine.bench.harness import (
    REPEAT_FLOOR,
    BenchHarness,
    CaseRunner,
    RunArtifacts,
    refuse_contaminated_holdout,
)
from evogine.bench.live import LiveCaseRunner, read_call_records
from evogine.bench.probes import ContaminationError
from evogine.bench.prompts import DEFAULT_PROMPT_PACK
from evogine.bench.protocol import (
    EvalProtocol,
    ProtocolValidationError,
    load_protocol,
    pre_run_validate,
)
from evogine.bench.replay import ReplaySchedule, build_replay_requests
from evogine.bench.rundirs import RunCompleteError, open_run
from evogine.bench.runner import KnowledgeRunner
from evogine.bench.shadow import ShadowRefusalError, load_shadow_candidate, run_shadow_pass
from evogine.config import (
    ConfigError,
    SyncEndpoint,
    default_config_path,
    load_config,
    starter_config_yaml,
)
from evogine.console.launch import ConsoleError
from evogine.engine.federation import FederationError, FederationRegistry
from evogine.engine.store import Engine, StoreError
from evogine.engine.transfer import (
    ImportRefused,
    apply_import,
    export_bundle,
    import_preview,
    read_bundle,
    write_bundle,
)
from evogine.gates.approvals import (
    ApprovalError,
    ApprovalLog,
    ApprovalState,
    approve_proposal,
    reject_proposal,
)
from evogine.gates.funnel import FunnelReport, build_funnel_report
from evogine.gates.memory_gate import MemoryGate, run_promotion_pass
from evogine.gates.telemetry import (
    DecisionLog,
    DecisionStage,
    RunsEvidenceSink,
    TelemetryError,
)
from evogine.ops.backup import BackupError, create_backup, restore_backup
from evogine.ops.benchmark import BenchmarkError, run_benchmark
from evogine.ops.compaction import CompactionError, compact_store

# --- V4 swarm: repository decomposer (appended in its own section to ease rebases) ---
from evogine.swarm.clients import AnthropicClient, ModelClient, RetryingModelClient
from evogine.swarm.decomposition import DecompositionError, decompose, render_summary_table

# --- V4 sync: client half of the daemon (appended in its own section to ease rebases) ---
from evogine.sync.client import (
    MergedSpend,
    PullResult,
    PushResult,
    RemoteStatus,
    SyncClientError,
    fleet_status,
    merge_spend,
    pull_store,
    push_store,
    resolve_remote,
)

DEFAULT_BENCH_REPEATS = 8
"""The offline bench's repeat count when neither a flag nor a protocol pins one."""


def main(argv: list[str] | None = None) -> int:
    """Run the ``evogine`` CLI and return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="evogine",
        description="Self-evolving agentic engine.",
    )
    parser.add_argument("--version", action="version", version=f"evogine {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="promotion pass: decide candidate learnings (replay, materiality, approval)",
    )
    run_parser.add_argument("--root", required=True, help="engine store root to operate on")
    run_parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help=f"repetitions per replay arm (floor: {REPEAT_FLOOR}; default: 8)",
    )

    bench_parser = subparsers.add_parser(
        "bench", help="warm-vs-cold benchmark over the default corpus"
    )
    bench_parser.add_argument("--root", required=True, help="engine store root to benchmark")
    bench_parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=f"repetitions per arm (floor: {REPEAT_FLOOR}; default: {DEFAULT_BENCH_REPEATS}; "
        "a protocol pins its own)",
    )
    bench_parser.add_argument("--format", choices=["table", "json"], default="table", dest="format")
    bench_parser.add_argument(
        "--runner",
        choices=["offline", "live"],
        default="offline",
        dest="runner",
        help="case runner: offline composes submissions from store knowledge (default); "
        "live drives the swarm model client and is protocol-gated",
    )
    bench_parser.add_argument(
        "--protocol",
        default=None,
        help="pre-registered protocol YAML pinning the run (required for --runner live)",
    )
    bench_parser.add_argument(
        "--model", default=None, help="provider model id (required for --runner live)"
    )
    bench_parser.add_argument(
        "--max-cost-usd",
        default=None,
        dest="max_cost_usd",
        help="live run spend ceiling in USD (default: 25.00)",
    )
    bench_parser.add_argument(
        "--expected-input-tokens",
        type=int,
        default=None,
        dest="expected_input_tokens",
        help="expected input tokens per replay (live; required so the ceiling cannot project zero)",
    )
    bench_parser.add_argument(
        "--expected-output-tokens",
        type=int,
        default=None,
        dest="expected_output_tokens",
        help="expected output tokens per replay (live; required so the ceiling cannot project zero)",
    )
    bench_parser.add_argument(
        "--input-price-per-mtok-usd",
        default=None,
        dest="input_price_per_mtok_usd",
        help="input price per million tokens in USD (live; required)",
    )
    bench_parser.add_argument(
        "--output-price-per-mtok-usd",
        default=None,
        dest="output_price_per_mtok_usd",
        help="output price per million tokens in USD (live; required)",
    )

    bench_parser.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="resume a specific run directory instead of starting a fresh one",
    )

    shadow_parser = subparsers.add_parser(
        "shadow",
        help="shadow-pass profiling of a candidate config — never mutates the store (V3 W2)",
    )
    shadow_parser.add_argument(
        "--root", required=True, help="engine store root to profile against (read-only)"
    )
    shadow_parser.add_argument(
        "--candidate", required=True, help="path to the shadow candidate declaration (YAML)"
    )
    shadow_parser.add_argument(
        "--runs-root",
        default=None,
        help="shadow runs root (default: <root>/runs/shadow)",
    )
    shadow_parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help=f"repetitions per arm (floor: {REPEAT_FLOOR}; default: 8)",
    )

    decisions_parser = subparsers.add_parser("decisions", help="promotion telemetry")
    decisions_subparsers = decisions_parser.add_subparsers(dest="decisions_command", required=True)
    report_parser = decisions_subparsers.add_parser(
        "report", help="counts of promotion decisions by stage"
    )
    report_parser.add_argument(
        "--root",
        default=".",
        help="engine store root holding the decisions/ log (default: .)",
    )
    funnel_parser = decisions_subparsers.add_parser(
        "funnel",
        help="proposal-quality funnel per pass, with refusal reasons and live tokens",
    )
    funnel_parser.add_argument(
        "--root",
        default=".",
        help="engine store root holding the decisions/ log (default: .)",
    )
    funnel_parser.add_argument(
        "--format", choices=["table", "json"], default="table", dest="format"
    )

    approvals_parser = subparsers.add_parser(
        "approvals", help="human-approval promotion boundary (V2-6)"
    )
    approvals_subparsers = approvals_parser.add_subparsers(dest="approvals_command", required=True)
    approvals_list_parser = approvals_subparsers.add_parser(
        "list", help="promotion proposals rendered by the memory gate"
    )
    approvals_list_parser.add_argument(
        "--root",
        default=".",
        help="engine store root holding the approvals/ log (default: .)",
    )
    approvals_approve_parser = approvals_subparsers.add_parser(
        "approve", help="apply a pending proposal (promotes the record)"
    )
    approvals_approve_parser.add_argument("proposal_id", help="id of the proposal (appr-...)")
    approvals_approve_parser.add_argument(
        "--by", required=True, help="who approved; recorded on the proposal"
    )
    approvals_approve_parser.add_argument(
        "--reason", default=None, help="optional note recorded with the approval"
    )
    approvals_approve_parser.add_argument(
        "--root", default=".", help="engine store root (default: .)"
    )
    approvals_reject_parser = approvals_subparsers.add_parser(
        "reject", help="refuse a pending proposal (demotes the record)"
    )
    approvals_reject_parser.add_argument("proposal_id", help="id of the proposal (appr-...)")
    approvals_reject_parser.add_argument(
        "--by", required=True, help="who rejected; recorded on the proposal"
    )
    approvals_reject_parser.add_argument(
        "--reason", required=True, help="why the proposal is refused"
    )
    approvals_reject_parser.add_argument(
        "--root", default=".", help="engine store root (default: .)"
    )

    console_parser = subparsers.add_parser(
        "console",
        help="localhost operator console over existing telemetry (read-only; [console] extra)",
    )
    console_parser.add_argument("--root", required=True, help="engine store root to read")
    console_parser.add_argument(
        "--operator",
        required=True,
        help='operator identity recorded on console actions, e.g. "Nihanth nihanth@example.eg"',
    )
    console_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host; loopback only, anything else is refused (default: 127.0.0.1)",
    )
    console_parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")

    # --- onboarding & config (V4 productization) ---------------------------
    # Appended as its own section so parallel CLI registrations merge cleanly.
    init_parser = subparsers.add_parser(
        "init",
        help="first-run setup: create the learning store, write the starter config, check the provider key",
    )
    init_parser.add_argument(
        "--root",
        default=".evogine/store",
        help="learning store root to create (default: .evogine/store); recorded in the written config",
    )
    init_parser.add_argument(
        "--config",
        default=None,
        dest="config_path",
        help="config file to write (default: ./evogine.yaml); an existing one is never overwritten",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="environment readiness: config, git, Docker, provider key, disk, sync endpoints",
    )
    doctor_parser.add_argument(
        "--config",
        default=None,
        dest="config_path",
        help="config file to read (default: ./evogine.yaml); a missing config is a warning, not an error",
    )

    # --- V4 swarm: repository decomposer (appended in its own section so a
    # parallel task's parser work rebases without touching this block) ---
    decompose_parser = subparsers.add_parser(
        "decompose",
        help=(
            "repository decomposer: module graph + criticality scoring "
            "(pure analysis; no store writes)"
        ),
    )
    decompose_parser.add_argument(
        "--repo", required=True, help="path to the Python repository to decompose"
    )
    decompose_parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="write the full DAG as JSON to this path, or - for stdout",
    )

    # --- cross-project memory import/export (V4 federation) -----------------
    # Appended as its own section so parallel CLI registrations merge cleanly.
    # Store paths are injected through the federation registry at invocation
    # time (--root is handed to register()); no shared config module is read.
    memory_parser = subparsers.add_parser(
        "memory",
        help="cross-project memory import/export over per-project stores (V4 federation)",
    )
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_export_parser = memory_subparsers.add_parser(
        "export",
        help="write an export bundle of this store's promoted pool (manifest + verbatim payloads)",
    )
    memory_export_parser.add_argument(
        "--store", required=True, help="store identity to export (registry name, e.g. alpha)"
    )
    memory_export_parser.add_argument(
        "--root",
        required=True,
        help="engine store root backing --store (injected via the registry)",
    )
    memory_export_parser.add_argument("--out", required=True, help="bundle destination path")
    memory_import_parser = memory_subparsers.add_parser(
        "import",
        help="import a bundle as reviewed candidates — dry-run by default; --yes applies",
    )
    memory_import_parser.add_argument(
        "--store", required=True, help="destination store identity (registry name, e.g. beta)"
    )
    memory_import_parser.add_argument(
        "--root",
        required=True,
        help="engine store root backing --store (injected via the registry)",
    )
    memory_import_parser.add_argument(
        "--bundle", required=True, help="path to the export bundle to review"
    )
    memory_import_parser.add_argument(
        "--yes",
        action="store_true",
        dest="confirmed",
        help="confirm the reviewed import; without it nothing is ever written",
    )

    # --- store maintenance: backup/restore/compaction/benchmark (ops hardening) ---
    store_parser = subparsers.add_parser(
        "store", help="store maintenance: backup, restore, compaction, benchmark"
    )
    store_subparsers = store_parser.add_subparsers(dest="store_command", required=True)

    store_backup_parser = store_subparsers.add_parser(
        "backup", help="snapshot a store (including git history) into a tar archive"
    )
    store_backup_parser.add_argument("--root", required=True, help="engine store root to back up")
    store_backup_parser.add_argument(
        "--to",
        required=True,
        help="archive path to write (must live outside the store)",
    )

    store_restore_parser = store_subparsers.add_parser(
        "restore", help="restore a store backup into an empty directory"
    )
    store_restore_parser.add_argument(
        "--from", required=True, dest="from_archive", help="backup archive to restore"
    )
    store_restore_parser.add_argument(
        "--to",
        required=True,
        help="destination directory (must be empty or absent)",
    )

    store_compact_parser = store_subparsers.add_parser(
        "compact",
        help="rebuild the derived index; preserves the audit trail and history",
    )
    store_compact_parser.add_argument("--root", required=True, help="engine store root to compact")

    store_benchmark_parser = store_subparsers.add_parser(
        "benchmark",
        help="measure duplicate detection and retrieval before/after compaction "
        "over a deterministic synthetic store",
    )
    store_benchmark_parser.add_argument(
        "--records", type=int, default=10_000, help="synthetic record count (default: 10000)"
    )
    store_benchmark_parser.add_argument(
        "--seed", type=int, default=0, help="deterministic generation seed (default: 0)"
    )
    store_benchmark_parser.add_argument(
        "--repeats", type=int, default=3, help="measurement repetitions per phase (default: 3)"
    )
    store_benchmark_parser.add_argument(
        "--dir",
        default=None,
        dest="store_dir",
        help="directory for the synthetic store (default: a fresh temp dir)",
    )

    # --- V4 sync client: cross-machine push/pull and spend roll-up ----------
    # Appended as its own section so parallel CLI registrations merge cleanly.
    # This is where the config's sync-endpoints section becomes live: a named
    # remote is a daemon URL plus the hosted store it serves, and every
    # command fails closed — an unreachable, misconfigured, or
    # policy-refusing remote renders as the refusal it is, never a silent skip.
    sync_parser = subparsers.add_parser(
        "sync",
        help="cross-machine sync over configured sync endpoints (status, push, pull)",
    )
    sync_subparsers = sync_parser.add_subparsers(dest="sync_command", required=True)
    sync_status_parser = sync_subparsers.add_parser(
        "status",
        help=(
            "query every configured remote's store and roll its spend into one "
            "merged, per-machine-attributed view"
        ),
    )
    sync_status_parser.add_argument(
        "--remote",
        default=None,
        help="limit the status to one configured remote (default: all of them)",
    )
    sync_status_parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="config file to read (default: ./evogine.yaml)",
    )
    for sync_verb, sync_verb_help in (
        ("push", "publish local store commits to a configured remote"),
        ("pull", "fetch and merge a remote's hosted store into the local store"),
    ):
        sync_transfer_parser = sync_subparsers.add_parser(sync_verb, help=sync_verb_help)
        sync_transfer_parser.add_argument(
            "--remote",
            default=None,
            help=(
                "configured remote to operate on (default: the only endpoint; "
                "required when several are configured)"
            ),
        )
        sync_transfer_parser.add_argument(
            "--root",
            default=None,
            help="local store root (default: the profile's learning_root)",
        )
        sync_transfer_parser.add_argument(
            "--config",
            dest="config_path",
            default=None,
            help="config file to read (default: ./evogine.yaml)",
        )

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args.root, repeats=args.repeats)
    if args.command == "bench":
        return _bench(
            args.root,
            repeats=args.repeats,
            as_json=args.format == "json",
            runner_name=args.runner,
            protocol_path=args.protocol,
            run_id=args.run_id,
            model=args.model,
            max_cost_usd=args.max_cost_usd,
            expected_input_tokens=args.expected_input_tokens,
            expected_output_tokens=args.expected_output_tokens,
            input_price_per_mtok_usd=args.input_price_per_mtok_usd,
            output_price_per_mtok_usd=args.output_price_per_mtok_usd,
        )
    if args.command == "shadow":
        return _shadow(
            args.root,
            candidate_path=args.candidate,
            runs_root=args.runs_root,
            repeats=args.repeats,
        )
    if args.command == "decisions" and args.decisions_command == "report":
        return _decisions_report(args.root)
    if args.command == "decisions" and args.decisions_command == "funnel":
        return _decisions_funnel(args.root, as_json=args.format == "json")
    if args.command == "approvals" and args.approvals_command == "list":
        return _approvals_list(args.root)
    if args.command == "approvals" and args.approvals_command == "approve":
        return _approvals_decide(
            args.root, args.proposal_id, decided_by=args.by, reason=args.reason, reject=False
        )
    if args.command == "approvals" and args.approvals_command == "reject":
        return _approvals_decide(
            args.root, args.proposal_id, decided_by=args.by, reason=args.reason, reject=True
        )
    if args.command == "console":
        return _console(args.root, args.operator, host=args.host, port=args.port)
    if args.command == "init":
        return _init(args.root, config_path=args.config_path)
    if args.command == "doctor":
        return _doctor(config_path=args.config_path)
    if args.command == "decompose":
        return _decompose(args.repo, json_out=args.json_out)
    if args.command == "memory" and args.memory_command == "export":
        return _memory_export(args.store, args.root, out=args.out)
    if args.command == "memory" and args.memory_command == "import":
        return _memory_import(
            args.store, args.root, bundle_path=args.bundle, confirmed=args.confirmed
        )
    if args.command == "store" and args.store_command == "backup":
        return _store_backup(args.root, args.to)
    if args.command == "store" and args.store_command == "restore":
        return _store_restore(args.from_archive, args.to)
    if args.command == "store" and args.store_command == "compact":
        return _store_compact(args.root)
    if args.command == "store" and args.store_command == "benchmark":
        return _store_benchmark(
            args.records,
            seed=args.seed,
            repeats=args.repeats,
            store_dir=args.store_dir,
        )
    if args.command == "sync" and args.sync_command == "status":
        return _sync_status(args.remote, config_path=args.config_path)
    if args.command == "sync" and args.sync_command in ("push", "pull"):
        return _sync_transfer(
            args.sync_command, args.remote, args.root, config_path=args.config_path
        )
    return 0  # unreachable: subcommands are required


def _run(root: str, *, repeats: int) -> int:
    """Drive a full promotion pass; exit 2 on any gate-error decision."""
    engine = Engine(root)
    gate = MemoryGate(
        runner=HarnessReplayRunner(
            engine,
            KnowledgeRunner(),
            corpus=None,
            repeats=repeats,
            runs_root=Path(root) / "runs" / "replay",
        ),
        log=DecisionLog(root),
        approvals=ApprovalLog(root),
        evidence=RunsEvidenceSink(root),
    )
    try:
        decisions = run_promotion_pass(engine, gate)
    except (StoreError, ValueError) as exc:
        # ValueError: a repeats count below the harness's replay floor.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"promotion pass: {len(decisions)} candidate(s) decided")
    for decision in decisions:
        print(f"  {decision.learning_id} [{decision.stage.value}] {decision.reason}")
        if decision.evidence_ref:
            print(f"    evidence: {decision.evidence_ref}")
    gate_errors = [decision for decision in decisions if decision.stage is DecisionStage.GATE_ERROR]
    if gate_errors:
        print(
            f"error: {len(gate_errors)} gate-error decision(s) — the replay "
            "infrastructure failed; records remain candidates",
            file=sys.stderr,
        )
        return 2
    pending = ApprovalLog(root).pending()
    if pending:
        print(
            f"next: {len(pending)} proposal(s) await human approval "
            f"(evogine approvals list --root {root})"
        )
    return 0


def _refuse(reason: str) -> int:
    """Print a fail-closed refusal to stderr; every refusal spends nothing."""
    print(f"error: {reason}", file=sys.stderr)
    return 1


def _resolve_provider_key() -> str | None:
    """The configured provider key, or None when absent.

    Read before any model call: the live path refuses to spend with no key
    configured — a refusal is cheaper than a failed call.
    """
    value = os.environ.get("ANTHROPIC_API_KEY", "")
    return value if value.strip() else None


def _build_live_client(*, api_key: str, model: str) -> ModelClient:
    """The live spend seam: bounded retries around one Anthropic-compatible provider."""
    return RetryingModelClient(AnthropicClient(model=model, api_key=api_key))


def _live_spend_plan(
    *,
    model: str | None,
    expected_input_tokens: int | None,
    expected_output_tokens: int | None,
    input_price_per_mtok_usd: str | None,
    output_price_per_mtok_usd: str | None,
    max_cost_usd: str | None,
) -> DollarBudget:
    """Build the declared dollar budget for a live run, refusing gaps.

    The spend expectation must be declared in full: an absent token or price
    knob would project zero and silently disarm the dollar ceiling, so the
    live path refuses rather than guessing. Pydantic coerces the declared
    values and refuses negatives and unparseable prices.
    """
    required = {
        "--model": model,
        "--expected-input-tokens": expected_input_tokens,
        "--expected-output-tokens": expected_output_tokens,
        "--input-price-per-mtok-usd": input_price_per_mtok_usd,
        "--output-price-per-mtok-usd": output_price_per_mtok_usd,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"--runner live requires {', '.join(missing)}: the spend expectation must be "
            "declared in full — an absent knob would project zero and disarm the dollar ceiling"
        )
    fields: dict[str, object] = {
        "expectation": {
            "input_tokens": expected_input_tokens,
            "output_tokens": expected_output_tokens,
            "input_price_per_mtok_usd": input_price_per_mtok_usd,
            "output_price_per_mtok_usd": output_price_per_mtok_usd,
        },
    }
    if max_cost_usd is not None:
        fields["max_run_cost_usd"] = max_cost_usd
    return DollarBudget.model_validate(fields)


def _bench(
    root: str,
    *,
    repeats: int | None,
    as_json: bool,
    runner_name: str,
    protocol_path: str | None,
    run_id: str | None,
    model: str | None,
    max_cost_usd: str | None,
    expected_input_tokens: int | None,
    expected_output_tokens: int | None,
    input_price_per_mtok_usd: str | None,
    output_price_per_mtok_usd: str | None,
) -> int:
    """Run the warm-vs-cold harness; ``live`` is protocol-gated and fail-closed."""
    if runner_name == "live" and protocol_path is None:
        return _refuse(
            "--runner live requires --protocol: the live path does not exist without a "
            "pre-registered protocol (nothing was spent)"
        )
    if repeats is not None and repeats < REPEAT_FLOOR:
        return _refuse(f"repeats={repeats} is below the {REPEAT_FLOOR}-repeat floor")

    protocol: EvalProtocol | None = None
    if protocol_path is not None:
        try:
            protocol = load_protocol(protocol_path)
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            return _refuse(f"protocol {protocol_path} refused: {exc}")

    engine = Engine(root)
    budget: DollarBudget | None = None
    if runner_name == "live":
        api_key = _resolve_provider_key()
        if api_key is None:
            return _refuse(
                "--runner live requires a configured provider key (ANTHROPIC_API_KEY): "
                "refusing before any model call (nothing was spent)"
            )
        assert protocol is not None  # the protocol-less live path refused above
        try:
            budget = _live_spend_plan(
                model=model,
                expected_input_tokens=expected_input_tokens,
                expected_output_tokens=expected_output_tokens,
                input_price_per_mtok_usd=input_price_per_mtok_usd,
                output_price_per_mtok_usd=output_price_per_mtok_usd,
                max_cost_usd=max_cost_usd,
            )
        except ValueError as exc:  # missing knobs, negatives, unparseable prices
            return _refuse(str(exc))
        if protocol.prompt_pack != DEFAULT_PROMPT_PACK.fingerprint:
            return _refuse(
                f"protocol pins prompt pack {protocol.prompt_pack!r} but this build's pack is "
                f"{DEFAULT_PROMPT_PACK.fingerprint!r} — live spend only follows the pinned text"
            )
        corpus = default_corpus()
        try:
            check = pre_run_validate(protocol, corpus)
            # The contamination probe runs FIRST: it is retrieval-only (free), and a
            # holdout the store already quotes voids the run before any money moves.
            refuse_contaminated_holdout(engine, check, corpus)
            schedule = ReplaySchedule(
                requests=build_replay_requests(list(check.cases), protocol.repeats)
            )
            queue_spend(schedule, budget)
        except (ProtocolValidationError, ContaminationError, SpendCeilingExceededError) as exc:
            return _refuse(str(exc))
        assert model is not None  # refused above when absent
        live_client: ModelClient | None = _build_live_client(api_key=api_key, model=model)
    else:
        live_client = None

    # All refusals are done: open the run directory and persist evidence.
    try:
        run_dir = open_run(Path(root) / "runs" / "bench", run_id=run_id)
    except ValueError as exc:  # includes RunCompleteError: completed runs are immutable
        return _refuse(str(exc))
    harness_runner: CaseRunner
    if live_client is not None:
        harness_runner = LiveCaseRunner(live_client, DEFAULT_PROMPT_PACK, run=run_dir)
    else:
        harness_runner = KnowledgeRunner()
    harness_repeats = (
        repeats
        if protocol is not None
        else (repeats if repeats is not None else DEFAULT_BENCH_REPEATS)
    )
    try:
        report = BenchHarness(engine, harness_runner).run(
            repeats=harness_repeats,
            protocol=protocol,
            artifacts=RunArtifacts(run_dir),
        )
        reconciliation: SpendReconciliation | None = None
        if live_client is not None and budget is not None:
            # Reconciled from the persisted per-call records — not the
            # runner's memory — so a resumed run's report also accounts
            # for every call loaded back from disk.
            reconciliation = reconcile_spend(budget, read_call_records(run_dir))
            run_dir.write_artifact("spend-report", reconciliation.model_dump_json(indent=2))
        run_dir.mark_complete()
    except (StoreError, ValueError, ProtocolValidationError, ContaminationError) as exc:
        return _refuse(str(exc))
    summary = report.summary
    if as_json:
        payload = {
            "run_id": run_dir.run_id,
            "run_dir": str(run_dir.directory),
            "repeats": summary.repeats,
            "replays": summary.replays,
            "cold_pass_rate": summary.cold_pass_rate,
            "warm_pass_rate": summary.warm_pass_rate,
            "mean_delta_pass_rate": summary.mean_delta_pass_rate,
            "cases": [
                {
                    "id": case_report.case_id,
                    "cold_pass_rate": case_report.cold.pass_rate,
                    "warm_pass_rate": case_report.warm.pass_rate,
                    "delta_pass_rate": case_report.delta_pass_rate,
                }
                for case_report in report.cases
            ],
        }
        if reconciliation is not None:
            payload["spend"] = reconciliation.model_dump(mode="json")
        print(json.dumps(payload, indent=2))
        return 0
    runner_label = "live" if live_client is not None else "offline"
    print(
        f"bench ({runner_label} runner): {len(report.cases)} cases, "
        f"{summary.replays} replays at {summary.repeats} repeats"
    )
    print(f"  run dir: {run_dir.directory}")
    if reconciliation is not None:
        print(
            f"  spend: ${reconciliation.actual_cost_usd} of "
            f"${reconciliation.ceiling_usd} ceiling over "
            f"{reconciliation.recorded_calls} calls "
            f"({reconciliation.calls_without_usage} without usage)"
        )
    print(f"  cold pass rate: {summary.cold_pass_rate:.3f}")
    print(f"  warm pass rate: {summary.warm_pass_rate:.3f}")
    print(f"  mean delta:     {summary.mean_delta_pass_rate:+.3f}")
    for case_report in report.cases:
        print(
            f"  {case_report.case_id}: cold {case_report.cold.pass_rate:.3f} "
            f"-> warm {case_report.warm.pass_rate:.3f} "
            f"({case_report.delta_pass_rate:+.3f})"
        )
    return 0


def _shadow(root: str, *, candidate_path: str, runs_root: str | None, repeats: int) -> int:
    """Profile a shadow candidate; the store is read-only, the run dir the only sink."""
    try:
        candidate = load_shadow_candidate(candidate_path)
        report = run_shadow_pass(
            Engine(root),  # handed over as the read-only ShadowStore seam only
            candidate,
            KnowledgeRunner(),
            runs_root=(
                Path(runs_root) if runs_root is not None else Path(root) / "runs" / "shadow"
            ),
            heldout_cases=_resolve_holdout_family(candidate.heldout_family),
            repeats=repeats,
        )
    except ShadowRefusalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RunCompleteError, StoreError, ValueError) as exc:
        # RunCompleteError: a completed shadow run is immutable evidence —
        # rerunning it means declaring a new candidate id.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = report.summary
    print(
        f"shadow {report.candidate_id}: {summary.replays} replays at {summary.repeats} repeats"
        f" (projected ${report.projected_cost_usd} of ${report.ceiling_usd})"
    )
    print(f"  cold pass rate: {summary.cold_pass_rate:.3f}")
    print(f"  warm pass rate: {summary.warm_pass_rate:.3f}")
    print(f"  mean delta:     {summary.mean_delta_pass_rate:+.3f}")
    print(f"  heldout recall probe: clean ({report.contamination.probed_cases} case(s) probed)")
    print(f"  run: {report.run_ref} (complete — immutable)")
    return 0


def _resolve_holdout_family(family: str | None) -> list[BenchCase]:
    """Resolve the declared held-out family against the default corpus.

    An undeclared family means no holdout — the recall probe runs and
    reports zero cases. A declared family that matches nothing is config
    drift: refuse rather than silently probe nothing.
    """
    if family is None:
        return []
    cases = [case for case in default_corpus() if case.family == family]
    if not cases:
        raise ValueError(f"heldout family {family!r} matches no case in the corpus")
    return cases


def _decisions_report(root: str) -> int:
    """Render decision counts by stage; every stage is always listed."""
    log = DecisionLog(root)
    decisions = log.all()
    counts = log.counts_by_stage()
    print(f"promotion decisions: {len(decisions)} total")
    for stage in DecisionStage:
        print(f"  {stage.value}: {counts[stage]}")
    promoted = sum(1 for decision in decisions if decision.promoted)
    print(f"promoted: {promoted}")
    return 0


def _decisions_funnel(root: str, *, as_json: bool) -> int:
    """Render the proposal-quality funnel; corrupt telemetry exits non-zero."""
    try:
        report = build_funnel_report(root)
    except (TelemetryError, ValidationError) as exc:
        # Store-owned telemetry (decisions, manifests, approvals) is versioned
        # evidence: a malformed file is a store defect, reported — never
        # silently skipped, which would understate the funnel.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(report.model_dump_json(indent=2))
        return 0
    print(_render_funnel_table(report, root))
    return 0


def _render_funnel_table(report: FunnelReport, root: str) -> str:
    """Human-readable funnel: cumulative stages, reasons, passes, live tokens."""
    lines: list[str] = []
    counts = report.funnel
    lines.append(f"proposal funnel: {root}")
    lines.append(
        f"  candidates:          {counts.candidates}  decided proposals (latest verdict each)"
    )
    lines.append(f"  replay-cleared:      {counts.replay_cleared}  replay flipped a verdict")
    lines.append(
        f"  materiality-cleared: {counts.materiality_cleared}  flip met the materiality threshold"
    )
    lines.append(
        f"  approved:            {counts.approved}  human approved (or applied) the proposal"
    )
    lines.append(f"  promoted:            {counts.promoted}  live in the promoted pool")
    lines.append(
        f"  gate errors:         {counts.gate_errors}  infrastructure — undecided, retryable"
    )
    lines.append(f"  proposals pending:   {counts.proposals_pending}  awaiting a human decision")
    lines.append(
        f"  proposals rejected:  {counts.proposals_rejected}  refused at the approval boundary"
    )
    if report.decisions_without_manifest:
        lines.append(
            f"  decisions without a finished pass manifest: "
            f"{report.decisions_without_manifest} (crash window before the manifest write)"
        )
    lines.append("")
    lines.append("decision reasons by stage (machine-checked vocabulary, zero-filled):")
    for stage_value, labels in report.reasons_by_stage.items():
        lines.append(f"  {stage_value}:")
        for label, count in labels.items():
            lines.append(f"    {label}: {count}")
    lines.append("")
    if report.passes:
        lines.append(f"passes: {len(report.passes)}")
        for entry in report.passes:
            lines.append(f"  {entry.pass_id}:")
            lines.append(f"    candidates: {entry.candidates}, decisions: {entry.decisions}")
            lines.append(
                f"    replay-cleared: {entry.replay_cleared}, "
                f"materiality-cleared: {entry.materiality_cleared}"
            )
            lines.append(
                f"    approved: {entry.approved}, promoted: {entry.promoted}, "
                f"gate errors: {entry.gate_errors}"
            )
    else:
        lines.append("passes: none recorded (no pass manifests under passes/)")
    lines.append("")
    live = report.live_tokens
    if live is None:
        lines.append(
            "live run token evidence: none found (no per-call token records under <root>/runs)"
        )
    else:
        lines.append(
            f"live run token evidence: {live.call_records} call record(s), "
            f"{live.ok} ok, {live.failed} failed"
        )
        lines.append(f"  tokens: {live.input_tokens} in / {live.output_tokens} out")
        for model, summary in sorted(live.by_model.items()):
            lines.append(
                f"  {model}: {summary.calls} call(s), "
                f"{summary.input_tokens} in / {summary.output_tokens} out"
            )
        if live.unreadable_records:
            lines.append(
                f"  unreadable call records (shown, not hidden): {live.unreadable_records}"
            )
    return "\n".join(lines)


def _console(root: str, operator: str, *, host: str, port: int) -> int:
    """Serve the operator console; web dependencies live behind the [console] extra.

    The import is lazy so the core install never loads FastAPI or uvicorn:
    ``evogine console`` without the extra installed refuses with a visible
    instruction rather than crashing on an ImportError.
    """
    try:
        from evogine.console.server import serve
    except ModuleNotFoundError as exc:  # fastapi/uvicorn absent
        print(
            f"error: the console needs the optional web dependencies: "
            f"pip install 'evogine[console]' ({exc})",
            file=sys.stderr,
        )
        return 1
    try:
        return serve(root, operator, host=host, port=port)
    except ConsoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _approvals_list(root: str) -> int:
    """Render the proposal log: every proposal, with pending ones marked."""
    log = ApprovalLog(root)
    proposals = log.all()
    pending = [proposal for proposal in proposals if proposal.state is ApprovalState.PENDING]
    print(f"approval proposals: {len(proposals)} total, {len(pending)} pending")
    for proposal in proposals:
        print(f"  {proposal.proposal_id} [{proposal.state.value}]")
        print(f"    learning: {proposal.learning_id}")
        trail = proposal.decision.reason
        print(f"    trail: {trail} (evidence: {proposal.evidence_ref or 'none'})")
        if proposal.decided_by is not None:
            decided_reason = f": {proposal.decided_reason}" if proposal.decided_reason else ""
            print(f"    decided by {proposal.decided_by}{decided_reason}")
    return 0


def _approvals_decide(
    root: str,
    proposal_id: str,
    *,
    decided_by: str,
    reason: str | None,
    reject: bool,
) -> int:
    """Apply or refuse one proposal; a refused action exits non-zero."""
    engine = Engine(root)
    log = ApprovalLog(root)
    action = "rejected" if reject else "approved"
    try:
        if reject:
            record = reject_proposal(
                engine, log, proposal_id, decided_by=decided_by, reason=reason or ""
            )
        else:
            record = approve_proposal(
                engine, log, proposal_id, decided_by=decided_by, reason=reason
            )
    except (ApprovalError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{proposal_id} {action}: record {record.id} is {record.status}")
    return 0


# ---------------------------------------------------------------------------
# Onboarding & config (V4 productization): init, doctor, and their checks.
# Every state is explicit — ok / warn / fail plus the fix — the house style
# the gates set, inherited not reinvented.
# ---------------------------------------------------------------------------


class CheckState(str, Enum):
    """A doctor check's verdict: ok, warn (usable, with a catch), or fail."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    """One doctor check's outcome: the state, what was found, and the fix."""

    model_config = ConfigDict(frozen=True)

    check: str
    state: CheckState
    detail: str
    fix: str | None = None


_GIB = 1024**3
_WARN_DISK_GIB = 5.0
_FAIL_DISK_GIB = 1.0
_SYNC_PROBE_TIMEOUT_SECONDS = 2.0


def _check_git() -> CheckResult:
    """git is the store's substrate: a missing executable fails, hard."""
    fix = (
        "install git (https://git-scm.com) — the learning store is a git "
        "repository; nothing runs without it"
    )
    try:
        proc = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
    except OSError as exc:  # FileNotFoundError: no git on PATH
        return CheckResult(
            check="git",
            state=CheckState.FAIL,
            detail=f"the git executable was not found: {exc}",
            fix=fix,
        )
    if proc.returncode != 0:
        return CheckResult(
            check="git",
            state=CheckState.FAIL,
            detail=f"git --version exited {proc.returncode}: {proc.stderr.strip()}",
            fix=fix,
        )
    return CheckResult(check="git", state=CheckState.OK, detail=proc.stdout.strip())


def _check_docker() -> CheckResult:
    """Probe Docker the way the code gate does: never raises, every state named.

    The docker-py import is lazy, so the client being absent (the core
    install) and the daemon being down are distinct failures with distinct
    fixes — neither is silently collapsed into "docker missing".
    """
    try:
        import docker  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001 — any import failure means "not installed"
        return CheckResult(
            check="docker",
            state=CheckState.FAIL,
            detail=f"the docker python client is not installed ({exc})",
            fix=(
                "pip install 'evogine[docker]' — the code gate runs candidate "
                "patches inside a Docker sandbox"
            ),
        )
    try:
        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001 — a probe, never a raise (DockerSandbox.available's contract)
        return CheckResult(
            check="docker",
            state=CheckState.FAIL,
            detail=f"no usable Docker daemon: {exc}",
            fix="start the Docker daemon — the code gate refuses (fails closed) without it",
        )
    return CheckResult(check="docker", state=CheckState.OK, detail="daemon reachable")


def _check_provider_key(api_key_env: str | None) -> CheckResult:
    """Key presence by name, from the environment; the value is never shown."""
    env_name = api_key_env if api_key_env is not None else "ANTHROPIC_API_KEY"
    if os.environ.get(env_name, "").strip():
        return CheckResult(
            check="provider",
            state=CheckState.OK,
            detail=(
                f"{env_name} is set in the environment (the value is never "
                "printed, logged, or written to disk)"
            ),
        )
    return CheckResult(
        check="provider",
        state=CheckState.WARN,
        detail=(
            f"{env_name} is not set — offline work proceeds, but live runs "
            "refuse before spending until a key is configured"
        ),
        fix=(
            f"export {env_name}=<your provider key> — read from the environment "
            "at run time, never stored in the config"
        ),
    )


def _disk_verdict(free_bytes: int, path: Path) -> tuple[CheckState, str, str | None]:
    """Pure disk thresholds: under 1 GiB free fails, under 5 GiB warns."""
    free_gib = free_bytes / _GIB
    readable = f"{free_gib:.1f} GiB free on {path}"
    if free_gib < _FAIL_DISK_GIB:
        return (
            CheckState.FAIL,
            f"only {readable}",
            "free disk space — the store, run dirs, and gate images need room to grow",
        )
    if free_gib < _WARN_DISK_GIB:
        return (
            CheckState.WARN,
            f"only {readable}",
            "free disk space before a long run — run dirs and gate images accumulate",
        )
    return CheckState.OK, readable, None


def _nearest_existing(path: Path) -> Path:
    """The deepest ancestor of ``path`` that exists (the path itself counts).

    A configured store root may not exist yet; its filesystem is still
    measurable at the nearest existing ancestor — and the report says which
    path was measured.
    """
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe


def _check_disk(root: Path) -> CheckResult:
    try:
        usage = shutil.disk_usage(_nearest_existing(root))
    except OSError as exc:
        return CheckResult(
            check="disk",
            state=CheckState.FAIL,
            detail=f"could not measure free space near {root}: {exc}",
        )
    state, detail, fix = _disk_verdict(usage.free, root)
    return CheckResult(check="disk", state=state, detail=detail, fix=fix)


def _check_sync_endpoint(endpoint: SyncEndpoint) -> CheckResult:
    """Probe one configured sync endpoint; unreachable fails with the fix.

    The probe targets the daemon's ``/health`` route — the documented
    liveness surface — so a correctly configured base URL cannot fail just
    because the origin itself answers 404.
    """
    url = str(endpoint.url)
    probe_url = url.rstrip("/") + "/health"
    try:
        with urlopen(probe_url, timeout=_SYNC_PROBE_TIMEOUT_SECONDS) as response:
            status = response.status
    except HTTPError as exc:
        return CheckResult(
            check="sync",
            state=CheckState.FAIL,
            detail=f"endpoint {endpoint.name!r} answered HTTP {exc.code} at {probe_url}",
            fix=(
                "the daemon is answering but this path is wrong — correct the "
                "endpoint URL in the config"
            ),
        )
    except OSError as exc:  # URLError, ConnectionRefusedError, timeouts — all OSError
        return CheckResult(
            check="sync",
            state=CheckState.FAIL,
            detail=f"endpoint {endpoint.name!r} unreachable at {probe_url}: {exc}",
            fix="start the sync daemon, or correct the endpoint URL in the config",
        )
    if endpoint.store is None or not endpoint.store.strip():
        return CheckResult(
            check="sync",
            state=CheckState.WARN,
            detail=(
                f"endpoint {endpoint.name!r} answered (HTTP {status}) at {probe_url} but "
                "declares no store — 'evogine sync' commands refuse an endpoint "
                "without one"
            ),
            fix="add 'store: <hosted-store-name>' under the endpoint in the config",
        )
    return CheckResult(
        check="sync",
        state=CheckState.OK,
        detail=(
            f"endpoint {endpoint.name!r} answered (HTTP {status}) at {probe_url} "
            f"(store '{endpoint.store}')"
        ),
    )


def _init(root: str, *, config_path: str | None) -> int:
    """First-run setup: store, starter config, provider key walkthrough.

    Refusal-first ordering: an existing config file refuses before anything
    is created — init never overwrites. A pre-existing store is fine (init
    is idempotent there) and is reported as such, with its record count.
    """
    cfg_path = Path(config_path) if config_path is not None else default_config_path()
    if cfg_path.exists():
        return _refuse(
            f"config {cfg_path} already exists — refusing to overwrite it "
            "(remove it by hand or pass --config to write elsewhere); nothing was created"
        )
    store_root = Path(root)
    store_existed = (store_root / ".git").exists()
    try:
        engine = Engine(store_root)
    except (OSError, StoreError) as exc:
        return _refuse(f"could not create the learning store at {store_root}: {exc}")
    record_count = len(list(engine.records_dir.glob("*.json")))
    try:
        cfg_path.write_text(starter_config_yaml(str(store_root)), encoding="utf-8")
    except OSError as exc:
        # Partial init is reported, never hidden: the store exists and is
        # usable; only the config write failed.
        return _refuse(
            f"the learning store is ready at {store_root}, but writing the config "
            f"failed: {exc} — create {cfg_path} by hand, then run 'evogine doctor'"
        )
    print("evogine init")
    state = "already present" if store_existed else "created"
    print(f"  store:  {state} at {store_root} ({record_count} record(s))")
    print(f"  config: wrote {cfg_path} (default profile: default)")
    # Load back what was just written: the starter config is itself subject
    # to the loader's contract, and a failure here is a bug that says so
    # loudly instead of printing a walkthrough from the wrong schema.
    try:
        profile = load_config(cfg_path).profile()
    except ConfigError as exc:
        return _refuse(
            f"the config just written does not load ({exc}) — this is a bug, please report it"
        )
    key_env = profile.provider.api_key_env
    if os.environ.get(key_env, "").strip():
        print(
            f"  provider key: {key_env} is set in the environment "
            "(the value is never printed or stored)"
        )
    else:
        print(f"  provider key: {key_env} is not set — live runs refuse until it is configured")
        print(f"    fix: export {key_env}=<your provider key>")
        print(
            "         the key is read from the environment at run time; "
            "it is never written to the config or any file"
        )
    print("next: evogine doctor   # verify the environment end-to-end")
    return 0


def _doctor(*, config_path: str | None) -> int:
    """Environment readiness: every state explicit; exit 1 only on a failure.

    A missing config warns (environment defaults apply) rather than failing:
    offline work proceeds without one. Warnings never fail the run; a fail
    means the environment cannot do its job, and the fix is printed.
    """
    print("evogine doctor — environment readiness")
    results: list[CheckResult] = []
    learning_root: Path | None = None
    api_key_env: str | None = None
    endpoints: list[SyncEndpoint] = []
    cfg_path = Path(config_path) if config_path is not None else default_config_path()
    if cfg_path.is_file():
        try:
            config = load_config(cfg_path)
            profile = config.profile()
            learning_root = profile.store.learning_root
            api_key_env = profile.provider.api_key_env
            endpoints = list(profile.sync.endpoints)
            results.append(
                CheckResult(
                    check="config",
                    state=CheckState.OK,
                    detail=f"{cfg_path} (profile: {config.default_profile})",
                )
            )
        except ConfigError as exc:
            results.append(
                CheckResult(
                    check="config",
                    state=CheckState.FAIL,
                    detail=str(exc),
                    fix="correct the config file by hand, or remove it and re-run 'evogine init'",
                )
            )
    else:
        results.append(
            CheckResult(
                check="config",
                state=CheckState.WARN,
                detail=f"no config file at {cfg_path} — environment defaults apply",
                fix="run 'evogine init' to create the store and write one",
            )
        )
    results.append(_check_git())
    results.append(_check_docker())
    results.append(_check_provider_key(api_key_env))
    results.append(_check_disk(learning_root if learning_root is not None else Path.cwd()))
    if endpoints:
        results.extend(_check_sync_endpoint(endpoint) for endpoint in endpoints)
    else:
        results.append(
            CheckResult(
                check="sync",
                state=CheckState.OK,
                detail="no sync endpoints configured (nothing to probe)",
            )
        )
    for result in results:
        print(f"  {result.check:<8} {result.state.value:<5} {result.detail}")
        if result.fix is not None:
            print(f"           fix: {result.fix}")
    ok = sum(1 for result in results if result.state is CheckState.OK)
    warns = sum(1 for result in results if result.state is CheckState.WARN)
    fails = sum(1 for result in results if result.state is CheckState.FAIL)
    if fails:
        print(f"doctor: not ready — {fails} failure(s), {warns} warning(s), {ok} ok")
        return 1
    print(f"doctor: ready ({ok} ok, {warns} warning(s))")
    return 0


# --- V4 swarm: repository decomposer (appended in its own section to ease rebases) ---


def _decompose(repo: str, *, json_out: str | None) -> int:
    """Decompose a Python repository: summary table, optional JSON DAG dump.

    Pure analysis over the working tree — the store is never touched; the
    only file this writes is the ``--json`` output the operator asked for
    (``-`` prints the JSON to stdout instead).
    """
    repo_path = Path(repo)
    if not repo_path.is_dir():
        return _refuse(f"--repo path is not a directory: {repo}")
    try:
        dag = decompose(repo_path)
    except DecompositionError as exc:
        return _refuse(str(exc))
    print(render_summary_table(dag))
    if json_out is None:
        return 0
    payload = dag.model_dump_json(indent=2)
    if json_out == "-":
        print(payload)
        return 0
    out_path = Path(json_out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        return _refuse(f"cannot write --json output to {json_out}: {exc}")
    return 0


# --- cross-project memory import/export (V4 federation) ---------------------


def _registered_store(store: str, root: str) -> Engine:
    """Build the engine for ``store`` with its path injected via the registry.

    The federation registry is the only path-injection abstraction: the CLI
    hands the operator's ``--root`` to ``register()`` and works through the
    engine it returns — no shared config module is read or written.
    """
    return FederationRegistry().register(store, Path(root))


def _memory_export(store: str, root: str, *, out: str) -> int:
    """Write the export bundle for ``store``'s promoted pool; a pure read."""
    try:
        engine = _registered_store(store, root)
        bundle = export_bundle(engine)
        written = write_bundle(bundle, Path(out))
    except (FederationError, StoreError) as exc:
        return _refuse(str(exc))
    print(
        f"exported {len(bundle.manifest.records)} promoted record(s) "
        f"from {bundle.manifest.source_store} — provenance verbatim, lineage qualified"
    )
    print(f"  bundle: {written}")
    return 0


def _memory_import(store: str, root: str, *, bundle_path: str, confirmed: bool) -> int:
    """Review an export bundle against ``store``; land it only when confirmed.

    Without ``--yes`` this is the dry-run the human reviews: what would
    land, nothing written. A refused or conflicting bundle exits non-zero
    with the refusal state on stderr — nothing is ever partially applied.
    """
    try:
        engine = _registered_store(store, root)
    except (FederationError, StoreError) as exc:
        return _refuse(str(exc))
    try:
        bundle = read_bundle(Path(bundle_path))
        preview = import_preview(bundle, engine)
        if not confirmed:
            print(
                f"dry run — {len(preview.landings)} record(s) would land in "
                f"{engine.store_name} as new candidates (nothing written):"
            )
            for landing in preview.landings:
                print(
                    f"  {landing.qualified_id} -> {landing.record_id} "
                    f'[{landing.type}] "{landing.title}" confidence {landing.confidence:.2f}'
                )
            print(
                "  on apply: provenance verbatim, confidence clock re-anchored "
                "to this store, source evidence left in its own trail"
            )
            print("re-run with --yes to apply the reviewed import")
            return 0
        landed = apply_import(bundle, engine)
    except ImportRefused as exc:
        return _refuse(str(exc))
    except StoreError as exc:
        # Outside the refusal vocabulary: a contract violation mid-apply
        # (e.g. a concurrent writer broke the one-writer-per-root rule).
        return _refuse(f"import failed after landing began — {exc}")
    for record in landed:
        print(f"imported {record.id} as candidate in {engine.store_name}")
    print(
        f"import complete: {len(landed)} record(s) landed, provenance verbatim, "
        "recorded in the audit trail"
    )
    return 0


# ---------------------------------------------------------------------------
# Store maintenance (V4 ops hardening): backup, restore, compaction, and the
# compaction benchmark. Every refusal is an explicit error, never a shrug.


def _store_backup(root: str, dest: str) -> int:
    try:
        result = create_backup(Path(root), Path(dest))
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"backup: {result.archive} ({result.file_count} files, {result.archive_bytes} bytes)")
    return 0


def _store_restore(archive: str, dest: str) -> int:
    try:
        restored = restore_backup(Path(archive), Path(dest))
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"restored: {restored}")
    return 0


def _store_compact(root: str) -> int:
    try:
        result = compact_store(Path(root))
    except CompactionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"compacted: {result.record_count} records indexed into "
        f"{result.index_path} ({result.index_bytes} bytes)"
    )
    return 0


def _store_benchmark(records: int, *, seed: int, repeats: int, store_dir: str | None) -> int:
    store_root = (
        Path(store_dir) if store_dir else Path(tempfile.mkdtemp(prefix="evogine-benchmark-"))
    )
    try:
        benchmark = run_benchmark(store_root, records, seed=seed, repeats=repeats)
    except (BenchmarkError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(benchmark.render())
    return 0


# --- V4 sync: client handlers (status roll-up, guarded push/pull) ----------


def _sync_status(remote_name: str | None, *, config_path: str | None) -> int:
    """Query the configured remotes and render one merged, attributed view.

    Fail-closed visibility: a remote that cannot be resolved or reached is
    rendered as the refusal it is — never skipped — and the merged view says
    how many machines reported. Exits non-zero when any remote refused, so
    an automation that greps exit codes notices a fleet that is not green.
    """
    cfg_path = Path(config_path) if config_path is not None else default_config_path()
    try:
        endpoints = list(load_config(cfg_path).profile().sync.endpoints)
    except ConfigError as exc:
        return _refuse(str(exc))
    if remote_name is not None:
        endpoints = [endpoint for endpoint in endpoints if endpoint.name == remote_name]
        if not endpoints:
            return _refuse(f"unknown sync remote {remote_name!r}")
    if not endpoints:
        return _refuse(
            "no sync endpoints configured — add one under "
            "profiles.<name>.sync.endpoints in the config"
        )
    statuses = fleet_status(endpoints)
    reporting = [status.spend for status in statuses if status.ok and status.spend is not None]
    merged = merge_spend(reporting)
    print(_render_sync_status(statuses, merged))
    if any(not status.ok for status in statuses):
        return 1
    return 0


def _spend_line(
    label: str,
    recorded_runs: int,
    actual_cost_usd: Decimal | None,
    overage_usd: Decimal | None,
    within_ceiling: bool | None,
) -> str:
    """One spend row; absence of spend is stated, never rendered as $0.0000."""
    if recorded_runs == 0 or actual_cost_usd is None:
        return f"{label}: {recorded_runs} run(s), no spend recorded"
    line = f"{label}: {recorded_runs} run(s), actual ${actual_cost_usd:.4f}"
    if overage_usd is not None:
        line += f", overage ${overage_usd:.4f}"
    if within_ceiling is not None:
        line += ", within ceiling" if within_ceiling else ", OVER CEILING"
    return line


def _render_sync_status(statuses: list[RemoteStatus], merged: MergedSpend) -> str:
    """Render per-remote states plus the merged, attributed spend view."""
    lines: list[str] = []
    for status in statuses:
        if not status.ok:
            lines.append(f"- {status.label}: REFUSED — {status.refusal}")
            continue
        remote = status.remote
        overview = status.overview
        if remote is None or overview is None or status.spend is None:
            lines.append(
                f"- {status.label}: REFUSED — incomplete status from the remote"
            )
            continue
        lines.append(
            f"- {status.label} (store '{remote.store}'): "
            f"{overview.total_decisions} decision(s), {overview.promoted} promoted, "
            f"{overview.pending_approvals} pending approval(s)"
        )
        spend = status.spend
        lines.append(
            "  spend: "
            + _spend_line(
                "this machine",
                spend.recorded_runs,
                spend.actual_cost_usd,
                spend.overage_usd,
                spend.within_ceiling,
            )
        )
    refused = any(not status.ok for status in statuses)
    lines.append("")
    if not merged.machines:
        lines.append(
            "merged spend: no remote reported — nothing is summed (absence is not zero)"
        )
        return "\n".join(lines)
    qualifier = " — refusals are never summed away" if refused else ""
    lines.append(
        f"merged spend ({len(merged.machines)} of {len(statuses)} machine(s) "
        f"reporting{qualifier}):"
    )
    for machine in merged.machines:
        lines.append(
            "  "
            + _spend_line(
                machine.remote,
                machine.recorded_runs,
                machine.actual_cost_usd,
                machine.overage_usd,
                machine.within_ceiling,
            )
        )
    if merged.total_actual_cost_usd is None:
        lines.append("  total: no spend recorded on any reporting machine")
    else:
        total = f"  total: {merged.recorded_runs} run(s), actual ${merged.total_actual_cost_usd:.4f}"
        if merged.total_overage_usd is not None:
            total += f", overage ${merged.total_overage_usd:.4f}"
        if merged.within_ceiling is not None:
            total += ", within ceiling" if merged.within_ceiling else ", OVER CEILING"
        lines.append(total)
    return "\n".join(lines)


def _sync_transfer(
    operation: str, remote_name: str | None, root: str | None, *, config_path: str | None
) -> int:
    """Push or pull one store over a configured remote's transport.

    Both directions fail closed: an unknown or ambiguous ``--remote``, an
    endpoint without a store name, an unreachable daemon, a dirty or
    zero-history local store, a non-fast-forward push, and a conflicting
    merge each refuse with the remote named — nothing is written.
    """
    cfg_path = Path(config_path) if config_path is not None else default_config_path()
    try:
        profile = load_config(cfg_path).profile()
        remote = resolve_remote(profile.sync.endpoints, remote_name)
        store_root = (
            Path(root) if root is not None else Path(profile.store.learning_root)
        )
        result: PushResult | PullResult
        if operation == "push":
            result = push_store(store_root, remote)
        else:
            result = pull_store(store_root, remote)
    except ConfigError as exc:
        return _refuse(str(exc))
    except SyncClientError as exc:
        return _refuse(str(exc))
    if isinstance(result, PushResult):
        print(
            f"sync push: transferred {result.head[:12]} to {result.url} ({result.ref})"
        )
        print(f"  store root: {store_root}")
        if result.uncommitted:
            print(
                f"note: {result.uncommitted} uncommitted change(s) exist only on this "
                "machine — nothing is published until the store's lifecycle commands "
                "commit them"
            )
        return 0
    print(
        f"sync pull: merged the hosted store '{result.store}' from remote "
        f"'{result.remote}' — local HEAD is now {result.head[:12]}"
    )
    print(f"  store root: {store_root}")
    return 0
