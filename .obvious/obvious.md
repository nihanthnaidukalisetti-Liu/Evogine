# obvious.md — evogine

## Repo

**evogine** (v0.1.0) — a self-evolving agentic engine: a git-backed knowledge engine with a
builder/verifier/reflector swarm, two objective gates (code gate, memory gate), and a
decay-pressure benchmark. Current state: **v1 scaffold** — contract-stable skeletons only
(pydantic record model, stub engine/gates/swarm/bench, tests, CI). No engine logic yet.

## Stack

- **Language:** Python >= 3.12 (verified on 3.13.14 in this sandbox)
- **Packaging:** hatchling (`pyproject.toml`), installed editable
- **Package manager:** pip
- **Key deps:** anthropic (unused in scaffold), pydantic>=2, pyyaml
- **Dev deps:** pytest, ruff, mypy
- **External services:** none — no DB, no Redis, no Docker/Compose. Tests make no live API calls.
- **Env vars / secrets:** none required

## Commands

All commands assume the repo venv at `.venv/` (gitignored):

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # one-time setup
.venv/bin/ruff check .        # lint (CI gate)
.venv/bin/pytest -q           # tests (CI gate; 4 tests, ~0.1s)
.venv/bin/mypy src            # typecheck (optional; not part of CI)
.venv/bin/evogine --version   # CLI identity: prints "evogine 0.1.0"
.venv/bin/evogine             # CLI stub: reports scaffold status
```

CI (`.github/workflows/ci.yml`) runs exactly: `pip install -e ".[dev]"` -> `ruff check .` ->
`pytest -q` on Python 3.12 (pushes to main and all PRs).

## Local Verification Summary

Verified 2026-09-18 on the onboarding sandbox (Python 3.13.14), dev_stack_healthy: true:

- `pip install -e ".[dev]"` — clean editable install
- `ruff check .` — all checks passed
- `pytest -q` — 4 passed in 0.09s
- CLI flow: `evogine --version` -> `evogine 0.1.0`; bare `evogine` and `evogine --help` work
- Library flow: `LearningRecord.sample()` constructs and survives a pydantic
  `model_dump` -> `model_validate` round-trip

## Codebase map

See [codebase-map.md](codebase-map.md).

## Notes & gotchas

- The CLI is a deliberate stub: no subcommands exist yet; don't "fix" that.
- `Engine`, `CodeGate`, `MemoryGate`, `BenchHarness`, and the swarm roles raise
  `NotImplementedError` by design — v1 scaffold contract only.
- `src/evogine/bench/cases/` and `src/evogine/seeds/` are intentionally empty
  (`.gitkeep` placeholders).
- Full v1 design spec: Obvious project `prj_knPCZEvt`, artifact `art_bGBG4tqU`.

## Sandbox snapshot

- Snapshot ID: `ws09f9ryqyfv1kopxk53:default`
- Captured: 2026-09-18T15:56:47Z
- Restores the working onboarding environment: `.venv` installed, dev stack healthy.
