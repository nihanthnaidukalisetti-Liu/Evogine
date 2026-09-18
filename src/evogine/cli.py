"""Console entry point for the ``evogine`` command.

v1 scaffold: the CLI exists so ``pip install .`` yields a working ``evogine``
command. Subcommands (run, bench, store inspection) arrive with the engine
implementation phases; this stub only reports identity and version.
"""

from __future__ import annotations

import argparse

from evogine import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the ``evogine`` CLI and return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="evogine",
        description="Self-evolving agentic engine (v1 scaffold).",
    )
    parser.add_argument(
        "--version", action="version", version=f"evogine {__version__}"
    )
    parser.parse_args(argv)
    print(f"evogine {__version__}: scaffold only — no subcommands implemented yet")
    return 0
