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
│   └── memory_gate.py # gate 2: counterfactual replay promotion of learned knowledge
├── swarm/
│   └── roles.py       # builder / verifier / reflector responsibilities
├── bench/
│   ├── harness.py     # warm-vs-cold decay-pressure benchmark
│   └── cases/         # benchmark cases (empty for now)
├── seeds/             # bootstrap learning records (empty for now)
└── py.typed
```

## Status

v1 scaffold: contract-stable skeletons only — pydantic record model, stub engine and
gates, tests, and CI. No engine logic yet.

## Spec

Full v1 design: Blueprint artifact `art_bGBG4tqU` — "Self-evolving engine v1 — design
spec" (Obvious project `prj_knPCZEvt`).
