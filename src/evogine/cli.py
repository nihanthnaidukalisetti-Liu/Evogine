"""Console entry point for the ``evogine`` command.

Subcommands grow with the engine. ``decisions report`` renders promotion
telemetry — decision counts by gate stage — from a store root's decision
log, so the 0/11-style silent failures of V1 cannot happen again: every
decided proposal's verdict is aggregated here.
"""

from __future__ import annotations

import argparse

from evogine import __version__
from evogine.gates.telemetry import DecisionLog, DecisionStage


def main(argv: list[str] | None = None) -> int:
    """Run the ``evogine`` CLI and return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="evogine",
        description="Self-evolving agentic engine.",
    )
    parser.add_argument(
        "--version", action="version", version=f"evogine {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decisions_parser = subparsers.add_parser(
        "decisions", help="promotion telemetry"
    )
    decisions_subparsers = decisions_parser.add_subparsers(
        dest="decisions_command", required=True
    )
    report_parser = decisions_subparsers.add_parser(
        "report", help="counts of promotion decisions by stage"
    )
    report_parser.add_argument(
        "--root",
        default=".",
        help="engine store root holding the decisions/ log (default: .)",
    )

    args = parser.parse_args(argv)
    if args.command == "decisions" and args.decisions_command == "report":
        return _decisions_report(args.root)
    return 0  # unreachable: subcommands are required


def _decisions_report(root: str) -> int:
    """Render decision counts by stage; every stage is always listed."""
    log = DecisionLog(root)
    decisions = log.all()
    counts = log.counts_by_stage()
    print(f"promotion decisions: {len(decisions)} total")
    for stage in DecisionStage:
        print(f"  {stage.value}: {counts[stage]}")
    promoted = sum(1 for decision in decisions if decision.promoted)
    print(f"promoted: {promoted}")
    return 0
