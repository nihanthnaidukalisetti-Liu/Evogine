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
├── engine/
│   ├── records.py     # LearningRecord: the versioned knowledge atom (pydantic contract)
│   └── store.py       # Engine: git-backed lifecycle, promotion-only retrieval
├── gates/
│   ├── code_gate.py   # gate 1: sandboxed pytest + differential equivalence + never-worse
│   ├── equivalence.py # pure gate logic: bytecode signatures, differential flips, never-worse
│   └── memory_gate.py # gate 2: counterfactual replay promotion of learned knowledge
├── sandbox/
│   ├── config.py       # sandbox contract: pinned dependencies, CPU/memory/time caps
│   ├── docker_runner.py # Docker execution sandbox: no network, sanitized output
│   ├── sanitize.py     # output redaction/capping before it can influence a verdict
│   └── staging.py      # path-traversal-safe baseline/candidate staging
├── swarm/
│   └── roles.py       # builder / verifier / reflector responsibilities
├── bench/
│   ├── harness.py     # warm-vs-cold decay-pressure benchmark
│   ├── cases/         # the 10-case corpus with hidden host-side invariants
│   ├── replay.py      # replay schedule vocabulary (case × arm × repeat)
│   └── shadow.py      # W2 shadow pass: profile candidates without mutating state
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
sync endpoints (forward-looking), and logging. The provider key is never
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
  schema migrations, compaction, sync daemon operation (loopback-first
  binding, LAN read exposure and its limits), doctor checks, and logs.
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
git-native sync daemon (loopback-first binding, read/query API); the
repository decomposer; and unified config with `init` and `doctor`. The
swarm run CLI — driving builder/verifier/reflector over a repository
task — is not on main yet. Details and the refusal-state vocabulary:
[docs/architecture.md](docs/architecture.md).

## Spec

Full v1 design: Blueprint artifact `art_bGBG4tqU` — "Self-evolving engine v1 — design
spec" (Obvious project `prj_knPCZEvt`).
