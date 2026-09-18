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
│   └── cases/         # benchmark cases (empty for now)
├── seeds/             # bootstrap learning records (empty for now)
└── py.typed
```

## Status

The sandboxed code gate is implemented: candidate patches are staged
traversal-safe, verified inside a locked-down Docker container (no network,
pinned dependencies, capped resources, sanitized output, no inherited host
environment), checked for differential equivalence (bytecode-verified when a
refactor claim is made), and enforced never-worse across evaluated cases. The
gate fails closed on its own crash, and Docker-gated probes skip cleanly where
Docker is absent. Everything else remains v1 scaffold skeletons.

## Spec

Full v1 design: Blueprint artifact `art_bGBG4tqU` — "Self-evolving engine v1 — design
spec" (Obvious project `prj_knPCZEvt`).
