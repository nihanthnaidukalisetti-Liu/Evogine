# Operator runbook

Operating a deployed store: backup and restore, schema migrations,
compaction, the sync daemon, doctor checks, and logs. Every command and
observed refusal here was executed against the real CLI and daemon from
this repository's `main`; where a capability does not exist on main yet,
the section says so plainly instead of guessing.

Contents:

1. [What is on disk](#what-is-on-disk)
2. [Backup and restore](#backup-and-restore)
3. [Schema migrations](#schema-migrations)
4. [Compaction](#compaction)
5. [Sync daemon operation](#sync-daemon-operation)
6. [Doctor checks](#doctor-checks)
7. [Structured logs and metrics](#structured-logs-and-metrics)

## What is on disk

A store root is a git repository. Everything below is inside the root
(default `.evogine/store`; each path is committed by the engine unless
noted):

| path | what it is |
|---|---|
| `records/*.json` | one file per learning record; ids are store-local filenames |
| `audit.jsonl` | the append-only audit trail; `seq` is derived from the live line count |
| `decisions/*.json` | one promotion-telemetry decision per decided proposal |
| `approvals/*.json` | the approval log — proposals and their human decisions |
| `passes/<pass_id>.json` | one manifest per promotion pass (written only if the pass finished) |
| `runs/replay/`, `runs/gate/` | replay evidence: per-run directories and paired verdicts |
| `runs/bench/`, `runs/shadow/` | benchmark and shadow-pass run directories (immutable, resumable) |
| `review.lock` | the per-store advisory lock serializing lifecycle writes (a lock file, not a store file) |

## Backup and restore

The store is a git repository, and its full committed state is the
backup: records, audit trail, decisions, approvals, pass manifests, and
replay evidence are all tracked files.

Backup — clone the store:

```bash
git clone /path/to/.evogine/store /backups/store-$(date +%F)
```

A verified roundtrip: after cloning, the restored copy is a fully
working store — `evogine decisions report --root /backups/store-restored`
returns the complete decision history, and `evogine approvals list`
shows every proposal with its decided state.

Restore — clone back to the store path:

```bash
git clone /backups/store-2026-09-19 /srv/project/.evogine/store
evogine doctor && evogine decisions report --root /srv/project/.evogine/store
```

Two operational notes, both observed in practice:

- Some telemetry files are swept into git history lazily — a proposal
  file's final decided state can sit as an unstaged modification between
  lifecycle commits. Immediately after a busy session, check
  `git -C <root> status --porcelain` and commit any residue; a backup
  taken right then captures the committed state, and the residue rides
  the next lifecycle commit.
- Run directories under `runs/bench/` and `runs/shadow/` accumulate with
  use (the doctor's disk check warns below 5 GiB free). They are
  evidence, not state — but they are committed, so a backup includes
  them by design.

The sync daemon (next sections) is also a live replication path: every
hosted store is itself a git repository served over git's smart-HTTP
protocol, so `git fetch` against the daemon is an incremental backup
mechanism for remote machines.

## Schema migrations

**No migration tooling exists on main today.** There is no `evogine
migrate` command, and this runbook will not invent one.

What exists instead is a deliberate compatibility property: the record
contract evolves by adding fields with defaults, so older record JSON
round-trips through newer readers unchanged (the V2 lifecycle fields —
`merged_sources`, `retired_at`, and friends — all defaulted this way).
Records are versioned pydantic contracts; a contract violation is a
refusal naming the defect, never a silent default.

Store-growth tooling — audited compaction, migration commands, and a
`store` maintenance surface — is planned as a follow-up ops-hardening
change. When it lands on main, this section documents its real commands
and numbers; until then, nothing here pretends they exist.

## Compaction

**Audited compaction is not on main today** — no CLI command, and no
published benchmark numbers for it. The runbook will not quote numbers
that were not measured.

What is true today:

- The store grows one JSON file per record plus git history; one
  lifecycle transition is one commit.
- The store's own merge/duplicate machinery (`find_duplicates()`,
  `merge()` in the store API) is knowledge-level deduplication, not
  history compaction — it merges duplicate records into survivors with
  provenance, leaving the audit trail intact.
- Git's own housekeeping applies to the store repo as to any git repo;
  the engine commits with git's defaults and takes no gc action of its
  own.

When compaction lands with benchmark numbers, this section will carry
the measured figures and the command that produced them.

## Sync daemon operation

The daemon (`evogine-syncd`, installed with the `[server]` extra) hosts
learning-store repositories over git smart-HTTP and serves a read/query
API. Launch:

```bash
evogine-syncd --stores-root /srv/evogine/stores --port 8766
```

`--stores-root` is a directory whose direct children are store roots —
the daemon hosts existing stores; it does not create a farm of empty
ones, and it refuses a stores root that does not exist before any socket
is opened. The default port is 8766: the operator console owns 8765, and
the daemon takes the adjacent port so both run on one machine untouched.

### Binding policy — loopback first

The daemon binds `127.0.0.1` by default, and a non-loopback bind is
refused before any socket exists:

```console
$ evogine-syncd --stores-root /srv/evogine/stores --host 0.0.0.0
error: refusing non-loopback bind: '0.0.0.0' resolves to 0.0.0.0; the daemon binds loopback only by default — pass --expose-lan to expose READ endpoints beyond this machine (store-mutating and approval endpoints stay loopback-only regardless)
```

`--expose-lan` is the explicit operator decision to expose **read**
endpoints beyond the machine (clone/fetch and the query API):

```bash
evogine-syncd --stores-root /srv/evogine/stores --host 0.0.0.0 --port 8766 --expose-lan
```

There is no auth stack — by scope decision, single-operator deployment —
so the bind address and the per-request peer address are the trust
anchors, and both are enforced structurally:

- **Read endpoints** (git clone/fetch, the query API) are gated by the
  bind address: loopback unless `--expose-lan` was passed.
- **Store-mutating endpoints** are gated per request by the peer
  address, regardless of any flag: a non-loopback peer is refused with
  HTTP 403 and a JSON body naming the reason. Pushing to a store,
  therefore, only ever works from the machine itself — even on a LAN-exposed daemon. A missing or non-IP peer address is treated as
  remote (fail-closed): the mutating surface never trusts what it cannot
  resolve to loopback.
- Store lifecycle writes run under the engine's per-store advisory lock,
  held across the push, so a push and a local lifecycle transition can
  never interleave within one store. Reads and fetches take no lock.

`GET /health` reports the policy as the daemon understood it — a
verified payload:

```json
{"status":"ok","stores":["alpha"],"bind_policy":{"host":"0.0.0.0","expose_lan":true,"mutations":"loopback-only regardless of any flag"}}
```

### Read/query API

All verified against a live daemon:

| endpoint | returns |
|---|---|
| `GET /health` | status, hosted store names, the bind policy |
| `GET /api/stores` | hosted store names |
| `GET /api/stores/{name}/overview` | decision counts by stage, promoted and pending totals |
| `GET /api/stores/{name}/records?status=` | records, optionally filtered by status |
| `GET /api/stores/{name}/records/{record_id}` | one record |
| `GET /api/stores/{name}/decisions` | the promotion-decision log |
| `GET /api/stores/{name}/approvals` | the approval log |
| `GET /api/stores/{name}/funnel` | the proposal-funnel report |
| `GET /api/stores/{name}/evidence` | runs evidence |
| `GET /api/stores/{name}/shadow` | shadow-pass outputs |
| `GET /api/stores/{name}/spend` | spend records |

Every refusal is a JSON body naming the reason — never a silent 500.

### Git transport

Clone, fetch, and push go through `/git/{store}`:

```bash
git clone http://127.0.0.1:8766/git/alpha
```

A verified session: clone the store, propose a record through the
engine in the clone (one commit), `git push` — the commit lands in the
hosted store and appears through the query API immediately.

One push refusal to know: git's receive-pack refuses to update the
hosted store's checked-out branch while its working tree is dirty, with
a visible `! [remote rejected] … (Working directory has unstaged
changes)`. The fix is the sweep note from [Backup and
restore](#backup-and-restore): commit the pending telemetry residue
host-side, then push again.

### Merging two machines' histories

Two machines appending to the same store between syncs is the one
surface git cannot merge by default (`audit.jsonl` changes on both
sides). The documented merge surface is the append-only audit merge
driver, `evogine.sync.merge`: with the common base as the anchor, each
side must be base lines plus appended lines — append-only is enforced,
and a side that rewrote its prefix makes the driver exit non-zero so
git reports a conflict instead of silently adopting the rewrite. The
merge is the base, then our appends, then their appends, duplicate
appended lines kept once.

Register it in any store that will pull:

```bash
git config merge.evogine-audit.name "Evogine audit union (append-only)"
git config merge.evogine-audit.driver "python3 -m evogine.sync.merge %O %A %B"
printf 'audit.jsonl merge=evogine-audit\n' >> .gitattributes
```

A same-record lifecycle collision (two machines transitioning the same
learning id) is deliberately not the driver's to resolve: the per-record
JSON conflicts visibly in stock git, and the operator reconciles — the
daemon never rewrites knowledge. The git history is the reconciliation
record; no global ordering across stores is invented.

## Doctor checks

`evogine doctor [--config PATH]` renders environment readiness with
every state explicit. Warnings never fail the run; it exits non-zero
only when a check fails. The checks, verified against a live
environment (where the docker fail and provider warn are the honest
states of a machine without a Docker daemon or a key):

| check | verifies | non-ok states |
|---|---|---|
| `config` | loads the config file, resolves the default profile | missing file → `warn` (environment defaults apply, fix: `evogine init`); invalid → `fail` |
| `git` | the git executable runs | missing/broken → `fail` (the store is a git repo; nothing runs without it) |
| `docker` | the docker client imports and the daemon answers a ping | client absent → `fail` (fix: `pip install 'evogine[docker]'`); daemon down → `fail` (fix: start it — the code gate fails closed) |
| `provider` | the env var named by `api_key_env` is set | unset → `warn` (offline proceeds; live runs refuse before spending) — the value is never printed, logged, or written |
| `disk` | free space at the store root (or its nearest existing ancestor, which the report names) | under 5 GiB → `warn`; under 1 GiB → `fail` |
| `sync` | each configured sync endpoint answers within 2 s | unreachable → `fail` (fix: start the daemon or correct the URL); wrong path answering → `fail`; none configured → `ok` ("nothing to probe") |

Sync endpoints are declared in `evogine.yaml` under
`profiles.<name>.sync.endpoints` (each a `name` and an `url` — the
config refuses duplicates and non-http schemes at load time). Run doctor
after any config or infrastructure change; a cron-friendly exit code
makes it scriptable as a readiness probe.

## Structured logs and metrics

**Structured log sinks and a metrics endpoint are not on main today** —
they land with the ops-hardening change, and this section will document
the real config knobs and output formats then. No invented field names.

What exists today, honestly:

- The config's `logging.level` knob (`debug`/`info`/`warning`/`error`)
  is the logging surface that exists; the sync daemon serves with
  uvicorn's `info`-level logs.
- The audit trail (`audit.jsonl`) **is** the structured event log of
  record: append-only JSON lines, one per lifecycle event, with
  sequence, timestamp, event, record id, status transition, reason,
  replay run ids, and pass id — readable with `jq` and tailable like
  any JSONL log.
- Promotion telemetry (`decisions/`, `passes/`) and run evidence
  (`runs/`) are the metrics source of record; the query API above
  exposes them as JSON for programmatic monitoring, and
  `GET /api/stores/{name}/overview` is the cheapest live health probe
  of a store's pipeline.
- `evogine doctor` is the machine-checkable readiness surface (exit
  code, per-check states).
