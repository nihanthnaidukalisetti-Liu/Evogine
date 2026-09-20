# Architecture

What the system is made of, the contracts between its parts, and the
vocabulary it uses to refuse things. This describes `main` as it stands;
the last section states the operational limits that remain, plainly, so
nothing above over-claims.

## The system in one paragraph

A git-backed knowledge engine for self-evolving software work. A
task-scaled builder / verifier / reflector swarm proposes patches and
learnings; two objective gates decide what lands and what is remembered —
a sandboxed **code gate** (sandboxed pytest + differential equivalence +
never-worse check) and a **memory gate** (counterfactual-replay
promotion behind a human approval boundary). A decay-pressure benchmark
measures whether accumulated knowledge actually improves outcomes. Every
state change is an append-only audit event and a git commit; every
refusal is explicit and named.

## Module map

```
src/evogine/
├── cli.py             # the `evogine` entry point: run, bench, shadow, decisions,
│                      # approvals, console, init, doctor, decompose, memory,
│                      # store, sync, swarm
├── config.py          # unified config: one YAML, named profiles, Decimal money
├── engine/
│   ├── records.py     # LearningRecord: the versioned knowledge atom
│   ├── store.py       # Engine: git-backed lifecycle, promotion-only retrieval,
│   │                  # qualified ids, per-store advisory lock, import_candidate
│   ├── retrieval.py   # hybrid scoring (cosine | lexical), honest empty state
│   ├── embedders.py   # HashingEmbedder default; the Embedder protocol
│   ├── federation.py  # FederationRegistry / FederationRouter: scope dispatch
│   │                  # ABOVE per-project stores — never inside one
│   └── transfer.py    # export/import bundles: manifest + verbatim payloads,
│                      # all-or-nothing reviewed landing
├── gates/
│   ├── code_gate.py   # gate 1: sandboxed pytest + differential equivalence
│   │                  # + never-worse; fails closed on its own crash
│   ├── equivalence.py # pure gate logic: bytecode signatures, differential
│   │                  # flips, never-worse
│   ├── memory_gate.py # gate 2: counterfactual-replay promotion (V2-6: never
│   │                  # promotes — renders approval proposals)
│   ├── materiality.py # does the replay flip qualify?
│   ├── approvals.py   # the approval log and the approve/reject application
│   ├── funnel.py      # proposal-quality funnel from existing telemetry files
│   ├── telemetry.py   # DecisionLog, pass manifests, RunsEvidenceSink; the
│   │                  # machine-checked refusal vocabulary
│   ├── replay.py      # the ReplayRunner seam and run-id grammar
│   ├── trajectory.py  # per-decision review trajectories (V2-8)
│   ├── separation.py, gold_sanity.py, wallclock.py   # anti-gaming (V2-8)
├── sandbox/
│   ├── config.py, docker_runner.py, staging.py, sanitize.py, seccomp.py
│                      # Docker-gated execution: no network, pinned deps, capped
│                      # resources, path-traversal-safe staging, output
│                      # redaction, seccomp default-on, fail-closed
├── swarm/
│   ├── roles.py       # builder / verifier / reflector with asymmetric powers;
│   │                  # KnowledgeSource and LearningSink protocols
│   ├── clients.py     # ModelClient protocol; AnthropicClient + retrying wrapper
│   ├── patches.py     # unified-diff application
│   ├── decomposition/ # repository decomposer: module graph, test/import
│   │                  # mapping, evidence-based criticality scoring
│   └── orchestrator/  # the swarm run driver: bounded worker pool, planning,
│                      # checkpoints, typed events, metered spend
├── bench/
│   ├── harness.py     # warm-vs-cold harness; REPEAT_FLOOR; holdout guards
│   ├── cases/         # the 10-case corpus; hidden invariants stay host-side
│   ├── replay.py      # replay schedule vocabulary (case × arm × repeat)
│   ├── protocol.py    # pre-registered EvalProtocol; pre-run validation
│   ├── budgets.py     # Decimal dollar ceilings, spend projection/reconciliation
│   ├── live.py        # the live case runner (spend seam)
│   ├── shadow.py      # non-mutating shadow passes; OverlayRetriever
│   ├── runner.py      # offline KnowledgeRunner (submissions from knowledge)
│   ├── gate_replay.py # HarnessReplayRunner: the memory gate's replay backend
│   └── rundirs.py, prompts.py, probes.py, perturb.py, stats.py
├── ops/               # store operations
│   ├── backup.py      # tar snapshot of the store; fail-closed restore
│   ├── compaction.py  # derived-index rebuild + git gc; audit trail preserved
│   ├── benchmark.py   # synthetic-store compaction benchmark
│   ├── migrations.py  # forward-only schema versioning, applied at store open
│   ├── logging_setup.py # the one logging entry point: level + text|json format
│   └── metrics.py     # in-process counters — no exporters, no sockets
├── console/           # localhost operator console ([console] extra)
│   ├── launch.py      # launch-time refusals: root, identity, loopback bind
│   ├── app.py, views.py, funnel.py   # read-only pages over existing telemetry
│   ├── sources.py     # stores-and-sync selector: local store + sync remotes
│   ├── swarm.py       # the /swarm board over the orchestrator's run directories
│   ├── injection.py   # knowledge-injection panel over reviewed import bundles
│   ├── write.py       # the two approval write routes, identity recorded
│   ├── templates/     # Jinja2 pages (autoescaped), /overview through /loop
│   └── server.py      # uvicorn entry: `evogine console` and the module entry point
└── sync/              # the sync daemon ([server] extra) and its client
    ├── server.py      # evogine-syncd entry: validate → resolve bind → serve
    ├── settings.py    # binding policy: loopback-first; peer-address gate
    ├── app.py         # FastAPI app: git transport + read/query routes
    ├── gitbackend.py  # git http-backend CGI handoff
    ├── stores.py      # the hosted-store registry
    ├── queries.py     # read/query surface over the stores' typed files
    ├── merge.py       # append-only audit merge driver
    └── client.py      # `evogine sync status|push|pull` over configured endpoints
```

Two transport surfaces — the console and the sync daemon — are optional
extras by contract (`[console]`, `[server]`): FastAPI and uvicorn load
only behind them, so the core install gains nothing it does not use.

## The store contract

`Engine` is the single write path to a store root, and the store root is
its own git repository. The contract's load-bearing rules:

- **Promotion-only retrieval.** `retrieve()` scores exactly the promoted
  pool; drafts, candidates, dropped and retired records are invisible to
  warm runs by construction.
- **One commit per lifecycle transition; one append-only audit line per
  event.** `seq` is derived from the live line count of `audit.jsonl` —
  the trail is never rewritten, and there is no second clock.
- **Qualified ids at the boundary.** A store's own record ids are
  unqualified filenames; the `<store-name>:<record-id>` form is reserved
  for cross-scope references, and a draft carrying a qualified id is
  refused at `propose`.
- **Refusal-first writes.** Pre-stamped promotion, replay evidence on a
  draft, out-of-pattern ids, and unknown cross-references are all
  refused before anything is written.
- **Concurrency-safe review.** A per-store advisory lock
  (`review.lock`) serializes lifecycle writes; a push held under the
  same lock can never interleave with a local transition.

## The two gates

**Code gate** — a candidate patch is staged traversal-safe, executed in
a locked-down Docker container (no network, pinned dependencies, capped
CPU/memory/time, sanitized output, no inherited host environment,
seccomp default-on, wall-clock cross-check), and then checked: sandboxed
pytest must pass, differential equivalence must hold (bytecode-verified
when a refactor is claimed), and no evaluated case may regress
(never-worse). The gate fails closed on its own crash; where Docker is
absent, Docker-gated probes skip cleanly.

**Memory gate** — for each candidate learning it replays the recorded
failure cases cold (knowledge absent) and warm (knowledge applied), then
decides through the stages described in the [user
guide](user-guide.md#memory-gates-and-promotion). Since the human-approval
boundary (V2-6), the gate never promotes: a cleared candidate becomes a
rendered approval proposal, and only `evogine approvals approve|reject`
(or the console's two write routes) applies or refuses it. A cleared
review without replay evidence is refused — the evidence requirement is
enforced regardless of the human.

## Federation and the four invariants

Federated memory is per-project stores behind one seam — Option B of
[the governance record](cross-project-memory.md). The store contract is
unchanged everywhere; everything federated lives **above** the store:

- `FederationRegistry` — the named set of per-project stores. Store
  paths are injected at registration; duplicate names are refused
  (fail-closed — silently re-pointing a scope at a different root would
  move every future read under the same name).
- `FederationRouter` — the retrieval seam. Dispatch is by scope: every
  read names exactly one store and is served from that store alone.
- `transfer.py` — the reviewed movement layer: bundles of promoted
  records, landing only through the destination's own write path.

From the governance record (§3), four invariants any implementation must
preserve — all four hold structurally in the code on main:

1. **Scoping is enforced at the storage/retrieval layer — never by
   caller discipline.** A store serves only its own rows; boundaries
   live where the pool is assembled.
2. **Promotion-only, per scope.** Warm runs see exactly what the gate
   proved *for this scope*. Imports land as candidates and must earn the
   destination's own promotion; source evidence never masquerades as
   destination evidence.
3. **Per-store append-only audit, with no invented global ordering.**
   Each trail's internal order is the only order that exists; import
   events reference the source by qualified id only — no stitched
   clocks, no cross-store sequence numbers.
4. **Qualified ids before anything references records across a scope
   boundary.** `<store-name>:<record-id>` is enforced at write time;
   import refuses a bundle arriving with unqualified cross-scope
   references. The `MergedSource` pattern is the standard: provenance
   travels verbatim with the survivor.

## LearningSink-mediated proposal flow

The swarm's three roles have deliberately asymmetric powers, wired
through protocols so no role can reach around its contract:

- **Builder** — reads knowledge only through the engine's promoted-only
  retrieval (`KnowledgeSource` protocol) and re-checks the contract
  defensively (a knowledge source returning anything but promoted
  records raises rather than informing a patch); it applies patches in a
  workspace and **never writes to the engine**.
- **Verifier** — holds the gates and nothing else: no model client, no
  engine. It cannot propose, retrieve, or write even by accident; it
  judges the proposal itself, never the builder's claims about it.
- **Reflector** — diagnoses failures and proposes learnings **only**
  through the `LearningSink` protocol — the engine's `propose` seam.
  Whatever the model claims, the reflector stores `ephemeral` under
  reflector provenance; only the memory gate can move knowledge past
  that, and only a human can move it past approval.

So knowledge enters the store through one narrow seam, is decided by one
objective gate, and is applied by one person — every proposal's path is
the same auditable funnel.

## Sync transport

The daemon is git-native: the transport is git's own smart-HTTP
protocol (`git http-backend` behind FastAPI), not a bespoke replication
format. Consequences:

- Any git client can clone/fetch a hosted store; a push is a normal
  receive-pack, held under the store's advisory lock.
- **Loopback-first binding** with an explicit `--expose-lan` escape for
  read endpoints only; per-request peer checks keep every store-mutating
  endpoint loopback-only regardless of flags — with no auth stack (a
  single-operator scope decision), the bind and peer addresses are the
  trust anchors, enforced structurally.
- The read/query API serves the stores' typed files as JSON (health,
  stores, records, decisions, approvals, funnel, evidence, shadow,
  spend); every refusal is a JSON body naming the reason.
- Concurrent appends from two machines merge through the append-only
  audit merge driver — append-only enforced against the merge base, a
  rewritten prefix exits non-zero so git conflicts visibly. See the
  [runbook](operator-runbook.md#merging-two-machines-histories) for the
  registration and the collision story.

## Operator console

The console is a localhost-only surface over telemetry that already
exists (no second computation): overview, decisions with detail pages,
approvals, the audit timeline, and the proposal-funnel panel are
read-only; guarded write routes decide one proposal each
(`POST /approvals/{id}/approve|reject`) and land reviewed knowledge
bundles (the injection panel's confirm), recording the operator identity
the console was launched with. Launch-time refusals cover a root that is
not a store, an empty operator identity, and any non-loopback bind —
there is no auth stack, so machine possession is the trust anchor and a
non-loopback bind is refused before any socket exists.

The knowledge-injection panel is the read-side view of the same reviewed
transfer flow the `memory export`/`import` CLI drives: it lists export
bundles from a conventional directory (`<store>/imports`, override with
`--imports-dir` at launch), renders each bundle's manifest
— source store, schema version, per-record qualified ids — before any
action, and offers exactly one write: the import command's own dry run,
then a confirm POST that lands candidates through `apply_import` under
the store's advisory lock. Every refusal is a panel with the store
untouched: an invalid manifest, a wrong schema version, duplicate
qualified ids, a destination collision, an import from the store's own
identity, and a confirm that skipped the dry run (an unapproved
injection) each render the transfer module's reason verbatim —
all-or-nothing, like the CLI. Landed records are candidates carrying the
source record's provenance byte-for-byte; promotion still faces this
store's gate and the approvals queue, so injection cannot bypass it.

The stores-and-sync selector chooses which store context the read-only
pages target: the local store, plus any sync remotes read from a config
file passed at launch (the default profile's `sync.endpoints`, resolved
once at launch, refusing a config that does not load). The `evogine
console` command itself takes only `--root`, `--operator`, `--host`, and
`--port` — to serve sync remotes, launch the console's module entry
point, which accepts the config: `python -m evogine.console.server
--root <root> --operator <identity> --config evogine.yaml`. Without it
the selector shows the local store alone. It changes what pages read,
never what they write — writes stay bound to the local store. The
`/sources` page renders per-source health from the existing sync client:
reachability, the query API's decision counts and fleet spend, the last
local audit position (remote trails are never stitched into one
sequence), and an explicit refusal row for a remote that will not answer.
Selecting a remote that refuses re-renders the overview as a refusal with
the client's reason, never a blank page.

## Configuration

One YAML file (`evogine.yaml`), named profiles, `default_profile` naming
the one that applies. The loader is the single entry point: unknown
fields refuse (`extra="forbid"` everywhere), unknown profiles refuse,
money is `Decimal` (a float ceiling is how $25.00 becomes $24.999997
silently), and the provider key is never stored — the config names an
environment variable (`api_key_env`), the value is read at run time.
`evogine init` writes a starter file that loads back through the same
contract; `evogine doctor` reads it and probes everything it declares.

## Refusal-state vocabulary

Promotion decisions carry a stage and a reason, both from a
machine-checked vocabulary — a reason string outside the set refuses to
be recorded (checked at the telemetry boundary, `check_reason`):

| stage | reasons | meaning |
|---|---|---|
| `replay` | `no_flip` | warm replay did not change any verdict |
| `materiality` | `not_material` | the flip did not meet the materiality threshold |
| `never_worse` | `cleared`, `regression` | no regression cleared; a regression refused |
| `approval` | `proposed`, `refused_no_evidence` | proposal rendered for a person; a cleared review without evidence is refused |
| `gate_error` | `gate_error` | replay infrastructure failed — not a knowledge verdict; retryable |

The same vocabulary is what the funnel zero-fills, what `decisions
report` counts, and what the console renders — one vocabulary everywhere
a refusal can be spoken. Beyond the gate, the rest of the system speaks
the same discipline in prose: bind refusals, import refusals
(all-or-nothing, nothing persisted on refusal), config refusals naming
file and defect, and gate failures that fail closed. A refusal always
names what was refused and, where a fix exists, the fix.

## The shipped command surface and what remains limited

`evogine --help` renders the full CLI surface: `run`, `bench`, `shadow`,
`decisions`, `approvals`, `console`, `init`, `doctor`, `decompose`,
`memory`, `store`, `sync`, and `swarm` — plus the `evogine-syncd` daemon
entry point. The swarm run CLI is part of that surface:
`evogine swarm run` drives a full swarm run over a repository task in one
command — decompose → plan → orchestrator → learning sink → promotion
pass — with the builder/verifier/reflector roles, bounded worker pool,
and metered spend described above.

Two operational limits remain, stated plainly:

- Schema migrations are the forward-only machinery in
  `ops/migrations.py`, applied automatically at store open (a store's
  schema version lives in its `schema.json`; older code refuses to open
  a newer store). There is no `evogine migrate` CLI command.
- The metrics surface is deliberately in-process: `ops/metrics.py` keeps
  counters that the console and dashboards read in-process. There is no
  network metrics endpoint — no exporter, no socket.

The console's sync-remotes launch path has one wrinkle worth knowing:
`evogine console` takes no `--config` flag; the selector reads sync
remotes when the console is launched through its module entry point,
which does (`python -m evogine.console.server --config …`) — see
[Operator console](#operator-console).
