---
name: local-dev
---

# local-dev — evogine onboarding record

Recorded by the autobuild onboarding worker, 2026-09-18.

## Environment

- Sandbox computer `cmp_wjPhOf4X`, repo at `/home/user/work/Evogine`, branch `main`
- Python 3.13.14 (`/usr/local/bin/python3`) — satisfies `requires-python >= 3.12` from pyproject
- venv at `.venv/` (gitignored), installed editable

## Steps that worked

1. `python3 -m venv .venv`
2. `.venv/bin/pip install -e ".[dev]"` — installs anthropic, pydantic, pyyaml, pytest, ruff, mypy
3. `.venv/bin/ruff check .` — all checks passed
4. `.venv/bin/pytest -q` — 4 passed in 0.09s
5. CLI flows: `evogine --version` -> `evogine 0.1.0`; bare `evogine`; `evogine --help` all work
6. Library flow: `LearningRecord.sample()` plus pydantic round-trip via
   `LearningRecord.model_validate(r.model_dump(mode="json"))` — equality preserved

## Gotchas

- No external services, no env vars, no secrets — the scaffold suite makes no live API calls.
- No Docker/Compose/Makefile; the venv + pip editable install is the entire dev stack.
- `mypy` is a dev extra but not a CI gate; `ruff check .` + `pytest -q` are the gates.
- Fresh sandboxes: recreate `.venv` with step 1-2; the snapshot restores it otherwise.

## Result

`dev_stack_healthy: true` — captured in sandbox snapshot
`ws09f9ryqyfv1kopxk53:default` (2026-09-18T15:56:47Z).
