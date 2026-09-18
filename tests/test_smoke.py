"""Smoke test: every scaffold module imports cleanly."""

import importlib

MODULES = [
    "evogine",
    "evogine.cli",
    "evogine.engine",
    "evogine.engine.records",
    "evogine.engine.store",
    "evogine.gates",
    "evogine.gates.code_gate",
    "evogine.gates.memory_gate",
    "evogine.swarm",
    "evogine.swarm.roles",
    "evogine.bench",
    "evogine.bench.cases",
    "evogine.bench.harness",
    "evogine.bench.replay",
]


def test_every_module_imports() -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        assert module is not None, f"failed to import {name}"
