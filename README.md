# evogine

evogine is a self-evolving agentic engine: a git-backed knowledge engine that detects
recurring failure patterns, preserves effective engineering practices, and improves code
repair through repeated, auditable, evidence-based evaluation. A task-scaled
builder / verifier / reflector swarm proposes changes and learnings; two objective gates —
a sandboxed **code gate** (sandboxed pytest + differential equivalence + never-worse check)
and a **memory gate** (counterfactual-replay promotion) — decide what lands and what is
remembered; a decay-pressure benchmark measures, warm versus cold, whether accumulated
knowledge actually improves outcomes.

## Layout

```
src/evogine/
├── cli.py             # the `evogine` entry point: run, bench, shadow, decisions,
│                      # approvals, console, init, doctor, decompose, memory,
│                      # store, sync, swarm
├── config.py          # unified config: one YAML, named profiles, Decimal money
├── engine/
│   ├── records.py     # LearningRecord: the versioned knowledge atom (pydantic contract)
│   ├── store.py       # Engine: git-backed lifecycle, promotion-only retrieval
│   ├── retrieval.py   # hybrid scoring (cosine | lexical), honest empty state
│   ├── embedders.py   # HashingEmbedder default; the Embedder protocol
│   ├── federation.py  # scope dispatch above per-project stores
│   └── transfer.py    # cross-project export/import bundles (reviewed landing)
├── gates/
│   ├── code_gate.py   # gate 1: sandboxed pytest + differential equivalence + never-worse
│   ├── equivalence.py # pure gate logic: bytecode signatures, differential flips, never-worse
│   ├── memory_gate.py # gate 2: counterfactual replay promotion of learned knowledge
│   ├── approvals.py   # the human approval boundary: approve/reject application
│   └── funnel.py      # proposal-quality funnel from existing telemetry
├── sandbox/
│   ├── config.py       # sandbox contract: pinned dependencies, CPU/memory/time caps
│   ├── docker_runner.py # Docker execution sandbox: no network, sanitized output
│   ├── seccomp.py      # seccomp default-on
│   ├── sanitize.py     # output redaction/capping before it can influence a verdict
│   └── staging.py      # path-traversal-safe baseline/candidate staging
├── swarm/
│   ├── roles.py       # builder / verifier / reflector responsibilities
│   ├── clients.py     # model client protocol; retrying Anthropic-compatible client
│   ├── patches.py     # unified-diff application
│   ├── decomposition/ # repository decomposer: module graph + criticality scoring
│   └── orchestrator/  # bounded worker pool, planning, checkpoints, metered spend
├── bench/
│   ├── harness.py     # warm-vs-cold decay-pressure benchmark
│   ├── cases/         # the 10-case corpus with hidden host-side invariants
│   ├── replay.py      # replay schedule vocabulary (case × arm × repeat)
│   ├── protocol.py    # pre-registered evaluation protocols
│   ├── live.py        # protocol-gated live case runner
│   └── shadow.py      # shadow pass: profile candidates without mutating state
├── console/           # localhost operator console ([console] extra): overview,
│                      # decisions, approvals, audit, funnel, sources, injection,
│                      # the swarm board (/swarm), and the learning loop (/loop)
├── ops/               # store maintenance: backup/restore/compaction/benchmark,
│                      # forward-only schema migrations, structured logging, metrics
├── sync/              # git-native sync daemon ([server] extra) and its client
├── seeds/             # bootstrap learning records (empty for now)
└── py.typed
```

## Getting started

```bash
pip install -e .   # core, offline
evogine init       # create the learning store, write a starter evogine.yaml, walk through provider key setup
evogine doctor     # git, Docker, provider key, disk space, sync endpoints — every state explicit, with the fix
```

Configuration lives in one YAML file (`evogine.yaml`) with named profiles:
store roots, provider settings, Decimal budget ceilings, worker-pool bounds,
sync endpoints, and logging. The provider key is never
stored in the file: it is read at run time from the environment variable
named by `api_key_env`.

## Documentation

- [Quickstart](docs/quickstart.md) — from clone to first governed run:
  `init`, `doctor`, a seeded candidate, an offline review, the approval,
  and the audit trail. Every command executed against the real CLI.
- [User guide](docs/user-guide.md) — store lifecycle, the approvals CLI,
  retrieval and the honest empty state, memory gates and promotion, bench
  protocols, the proposal funnel, and cross-project export/import.
- [Operator runbook](docs/operator-runbook.md) — backup and restore,
  schema migrations, compaction, sync daemon and client operation
  (loopback-first binding, LAN read exposure and its limits), doctor
  checks, and logs.
- [Architecture](docs/architecture.md) — the module map, the four
  federation invariants, the LearningSink-mediated proposal flow, sync
  transport, and the refusal-state vocabulary.
- [Cross-project memory](docs/cross-project-memory.md) — the federation
  governance record: the architecture fork, and the four invariants.

## Status

Shipped on main: the git-backed learning store with promotion-only
retrieval and the full record lifecycle; the sandboxed code gate
(Docker-gated, fail-closed) and the memory gate behind the human-approval
boundary; the offline warm-vs-cold bench plus protocol-gated live runs
and non-mutating shadow passes; the proposal funnel over promotion
telemetry; the localhost operator console with guarded approval write
paths; federated memory with reviewed cross-project export/import; the
git-native sync daemon (loopback-first binding, read/query API) with the
`evogine sync status`/`push`/`pull` client; store maintenance
(`evogine store backup`/`restore`/`compact`/`benchmark`) with forward-only
schema migrations and structured logging; the repository decomposer;
unified config with `init` and `doctor`; and the swarm run CLI —
`evogine swarm run`, one command chaining decompose, plan, orchestrator,
learning sink, and the promotion pass over a repository task. Details and
the refusal-state vocabulary: [docs/architecture.md](docs/architecture.md).

## Spec

Full v1 design: Blueprint artifact `art_bGBG4tqU` — "Self-evolving engine v1 — design
spec" (Obvious project `prj_knPCZEvt`).
