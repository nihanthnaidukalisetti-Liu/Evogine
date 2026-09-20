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
6. [Sync client commands](#sync-client-commands)
7. [Doctor checks](#doctor-checks)
8. [Structured logs and metrics](#structured-logs-and-metrics)

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
| `schema.json` | the store's schema-version stamp (forward-only migrations; see below) |
| `index.jsonl` | the derived retrieval index (rebuilt by `evogine store compact`; see below) |
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

The store also ships its own archive path (`evogine store backup`/
`restore`), a complete tar snapshot including the full `.git` history,
with a fail-closed restore. The git-clone path above and the tar archive
are equivalent snapshots of committed state — clone is the zero-flags
habit, the archive is the one-file artifact:

```console
$ evogine store backup --root .evogine/store --to /backups/store-2026-09-20.evogine-backup.tar
backup: /backups/store-2026-09-20.evogine-backup.tar (86 files, 266240 bytes)
$ evogine store restore --from /backups/store-2026-09-20.evogine-backup.tar --to /srv/project/.evogine/store
restored: /srv/project/.evogine/store
```

Restore refuses to overwrite a non-empty destination and verifies store
post-conditions (git history, schema stamp, records) before reporting
success:

```console
$ evogine store restore --from /backups/store-2026-09-20.evogine-backup.tar --to /srv/project/.evogine/store
error: refusing to restore into a non-empty directory: /srv/project/.evogine/store
(exit=1 — point --to at an empty or absent directory)
```

The sync daemon (next sections) is also a live replication path: every
hosted store is itself a git repository served over git's smart-HTTP
protocol, so `git fetch` against the daemon is an incremental backup
mechanism for remote machines.

## Schema migrations

Store schemas are versioned and migrated forward-only: the store stamps
its schema version in `schema.json` at its root (versioned content like
everything else the store holds), and the machinery in
`ops/migrations.py` applies registered migrations one step at a time,
each step committing its change to the store's git history. It is wired
into the engine's store open path — there is no migration step for an
operator to run.

The enforced contract:

- **Forward-only.** An explicit request to go below the store's current
  version refuses loudly, with both versions named; a downgrade is never
  a silent no-op.
- **Back-compat is absolute.** Stores that predate the stamp read as
  version 1 and keep working unchanged; opening a store upgrades it
  explicitly.
- **Older code refuses newer stores.** Opening a store whose schema
  version exceeds the running build's refuses — reading a schema you do
  not understand is how silent corruption happens; refusing is the
  visible alternative.
- **Registry gaps refuse.** A version with no registered migration step
  is a bug and raises rather than skipping ahead.

The registry ships with one real step — version 1 → 2, "adopt schema
versioning", which stamps `schema.json` into the store's history. Future
format changes register their own steps against this machinery.

What does **not** exist is a `evogine migrate` CLI command — migrations
are applied automatically at store open, not driven by hand. Record-level
evolution keeps the older compatibility property: the record contract
changes by adding fields with defaults, so older record JSON round-trips
through newer readers unchanged (the V2 lifecycle fields —
`merged_sources`, `retired_at`, and friends — all defaulted this way).
Records are versioned pydantic contracts; a contract violation is a
refusal naming the defect, never a silent default.

## Compaction

`evogine store compact` rebuilds the store's derived retrieval index and
garbage-collects the store's git repo. The audit trail and record files
are untouched: compaction adds one derived file (`index.jsonl`) and
repacks git's object store — HEAD and the commit graph's meaning are
preserved, and the win is fewer parsed files on hot reads and a smaller
`.git`:

```console
$ evogine store compact --root .evogine/store
compacted: 4 records indexed into .evogine/store/index.jsonl (3479 bytes)
```

The store's own merge/duplicate machinery (`find_duplicates()`,
`merge()` in the store API) remains knowledge-level deduplication, not
history compaction — it merges duplicate records into survivors with
provenance, leaving the audit trail intact.

Before committing to compaction at scale, `evogine store benchmark`
measures duplicate detection and retrieval on a deterministic synthetic
store, before versus after the index rebuild. The default is 10,000
records; scale it down for a quick run (output prints on completion):

```console
$ evogine store benchmark --records 500 --dir /tmp/evogine-bench-docs
store benchmark: 500 records (promoted=72, candidates=428, repeats=3)
duplicate detection: 189.549 ms -> 38.947 ms (21418 exact pairs)
retrieval per query: 24.235 ms -> 20.429 ms (5 probes, k=5)
store size: 18775.9 KiB -> 19116.9 KiB (derived index included)
```

The measured trade: duplicate detection and retrieval get several times
faster on the rebuilt index; the store on disk grows by the index's own
size. Run the benchmark at your own record scale and hardware before
treating any one run's numbers as typical — these are one measurement,
not a claim.

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

## Sync client commands

The daemon's client half ships as `evogine sync` — three commands over
the endpoints configured in `evogine.yaml` under
`profiles.<name>.sync.endpoints` (each endpoint a `name`, an `url`, and
the hosted `store` it names — see [Doctor checks](#doctor-checks)). All
outputs below were executed against a live daemon (the examples' daemon
runs on port 8791; the documented default is 8766 — the console owns
8765, so the daemon takes the adjacent port).

`sync status` queries every configured remote's store and rolls the
fleet's spend into one merged, per-machine-attributed view:

```console
$ evogine sync status
- local (store 'alpha'): 0 decision(s), 0 promoted, 0 pending approval(s)
  spend: this machine: 0 run(s), no spend recorded

merged spend (1 of 1 machine(s) reporting):
  local: 0 run(s), no spend recorded
  total: no spend recorded on any reporting machine
(exit=0)
```

`sync push` publishes local store commits to a configured remote — a
normal git push under the hosted store's advisory lock:

```console
$ evogine sync push
sync push: transferred c58de599f2a8 to http://127.0.0.1:8791/git/alpha (refs/heads/master)
  store root: store
(exit=0)
```

A push whose hosted store has commits the local store lacks refuses —
pull first, then push:

```console
$ evogine sync push
error: push to remote 'local' refused (non-fast-forward): the hosted store has commits this store lacks — run 'evogine sync pull' first, then push
(exit=1)
```

`sync pull` fetches and merges the remote's hosted store into the local
store — the append-only audit merge driver (registered above) is what
makes two machines' audit trails merge instead of conflict:

```console
$ evogine sync pull
sync pull: merged the hosted store 'alpha' from remote 'local' — local HEAD is now f3cbbabf579c
  store root: store
(exit=0)
```

A pull git cannot merge cleanly aborts and refuses — nothing changes,
and the client never rewrites knowledge:

```console
$ evogine sync pull
error: pull from remote 'local' hit a conflict git cannot merge cleanly — the merge was aborted and nothing changed; reconcile the store by hand (the client never rewrites knowledge): fatal: refusing to merge unrelated histories
(exit=1)
```

Two config refusals, both verified: with no endpoints configured, the
command refuses rather than reporting an empty fleet as if it were a
good one:

```console
$ evogine sync status
error: no sync endpoints configured — add one under profiles.<name>.sync.endpoints in the config
(exit=1)
```

and an endpoint without a `store` name refuses — the client never
guesses which hosted store it serves (see
[Doctor checks](#doctor-checks) for the exact refusal).

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
`profiles.<name>.sync.endpoints` — each endpoint carries a `name`, an
`url`, and the `store` name of the hosted store it serves. The config
refuses duplicate names and non-http schemes at load time; `store` is
optional at load (pre-sync configs load unchanged, and `doctor` still
probes the URL) but a sync command against an endpoint without one
refuses visibly — the client never guesses a store name:

```console
$ evogine sync status
- local: REFUSED — sync endpoint 'local' declares no store — add 'store: <name>' under the endpoint in the config (the daemon hosts stores by name; the client never guesses one)

merged spend: no remote reported — nothing is summed (absence is not zero)
(exit=1)
```

A verified probe line with the store named:

```console
$ evogine doctor
  sync     ok    endpoint 'local' answered (HTTP 200) at http://127.0.0.1:8791/health (store 'alpha')
```

Run doctor
after any config or infrastructure change; a cron-friendly exit code
makes it scriptable as a readiness probe.

## Structured logs and metrics

Every process configures logging exactly once, through
`ops/logging_setup.py` at the CLI's entry point — the config's
`logging` section is the surface:

- `logging.level` — `debug`/`info`/`warning`/`error` (default `info`).
- `logging.format` — `text` (default; human-oriented single lines with
  structured fields appended as `key=value`) or `json` (one JSON object
  per line: `ts`, `level`, `logger`, `message`, plus the event's
  structured fields — for log shippers and the dashboard's log view).

Logs go to stderr; stdout is reserved for command output. There is no
third sink and no per-module handler — library code only emits typed
fields, and format/level stay the config file's decision.

The metrics surface is deliberately small and in-process:
`ops/metrics.py` keeps a process-wide registry of counters
(`evogine_store_transitions_total`, `evogine_gate_decisions_total`,
`evogine_funnel_stage_total`, each name and label pinned to a checked
vocabulary) read in-process by the console and dashboard. There are **no
exporters and no sockets** — no network metrics endpoint exists, by
design, so a core install gains nothing it does not use.

What already existed and remains the record:

- The audit trail (`audit.jsonl`) **is** the structured event log of
  record: append-only JSON lines, one per lifecycle event, with
  sequence, timestamp, event, record id, status transition, reason,
  replay run ids, and pass id — readable with `jq` and tailable like
  any JSONL log.
- Promotion telemetry (`decisions/`, `passes/`) and run evidence
  (`runs/`) are the metrics source of record; the sync daemon's query
  API above exposes them as JSON for programmatic monitoring, and
  `GET /api/stores/{name}/overview` is the cheapest live health probe
  of a store's pipeline.
- `evogine doctor` is the machine-checkable readiness surface (exit
  code, per-check states).
