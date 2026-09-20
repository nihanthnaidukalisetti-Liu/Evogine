# Codebase map — evogine

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, deps, `[dev]` extras, `evogine` console script, ruff config |
| `.github/workflows/ci.yml` | CI: ruff + pytest on Python 3.12 (pushes to main, PRs) |
| `src/evogine/__init__.py` | Package root: exports `LearningRecord`, `__version__` |
| `src/evogine/cli.py` | `evogine` console entry point (stub: `--version`, `--help`) |
| `src/evogine/engine/records.py` | `LearningRecord` + body/provenance/counterfactual models (pydantic) |
| `src/evogine/engine/store.py` | `Engine`: git-backed lifecycle, promotion-only retrieval (stub) |
| `src/evogine/gates/code_gate.py` | Gate 1: sandboxed pytest + differential equivalence + never-worse (stub) |
| `src/evogine/gates/memory_gate.py` | Gate 2: counterfactual-replay promotion of learnings (stub) |
| `src/evogine/swarm/roles.py` | `Builder` / `Verifier` / `Reflector` role contracts (stubs) |
| `src/evogine/bench/harness.py` | Warm-vs-cold decay-pressure benchmark harness (stub) |
| `src/evogine/bench/cases/` | Benchmark cases — empty (`.gitkeep`) |
| `src/evogine/seeds/` | Bootstrap learning records — empty (`.gitkeep`) |
| `tests/test_records.py` | `LearningRecord` model tests |
| `tests/test_smoke.py` | Import / CLI smoke tests |
