"""Console entry point for the ``evogine`` command.

Subcommands grow with the engine. ``decisions report`` renders promotion
telemetry — decision counts by gate stage — from a store root's decision
log, so the 0/11-style silent failures of V1 cannot happen again: every
decided proposal's verdict is aggregated here. ``approvals`` is the
human-approval promotion boundary (V2-6): the memory gate renders
promotion proposals into the store's approval log, and
``approvals list|approve|reject`` is how a person reviews and decides
them — approve applies the promotion, reject refuses it and demotes the
record. The gate never promotes on its own.
"""

from __future__ import annotations

import argparse
import sys

from evogine import __version__
from evogine.engine.store import Engine, StoreError
from evogine.gates.approvals import (
    ApprovalError,
    ApprovalLog,
    ApprovalState,
    approve_proposal,
    reject_proposal,
)
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

    approvals_parser = subparsers.add_parser(
        "approvals", help="human-approval promotion boundary (V2-6)"
    )
    approvals_subparsers = approvals_parser.add_subparsers(
        dest="approvals_command", required=True
    )
    approvals_list_parser = approvals_subparsers.add_parser(
        "list", help="promotion proposals rendered by the memory gate"
    )
    approvals_list_parser.add_argument(
        "--root",
        default=".",
        help="engine store root holding the approvals/ log (default: .)",
    )
    approvals_approve_parser = approvals_subparsers.add_parser(
        "approve", help="apply a pending proposal (promotes the record)"
    )
    approvals_approve_parser.add_argument("proposal_id", help="id of the proposal (appr-...)")
    approvals_approve_parser.add_argument(
        "--by", required=True, help="who approved; recorded on the proposal"
    )
    approvals_approve_parser.add_argument(
        "--reason", default=None, help="optional note recorded with the approval"
    )
    approvals_approve_parser.add_argument(
        "--root", default=".", help="engine store root (default: .)"
    )
    approvals_reject_parser = approvals_subparsers.add_parser(
        "reject", help="refuse a pending proposal (demotes the record)"
    )
    approvals_reject_parser.add_argument("proposal_id", help="id of the proposal (appr-...)")
    approvals_reject_parser.add_argument(
        "--by", required=True, help="who rejected; recorded on the proposal"
    )
    approvals_reject_parser.add_argument(
        "--reason", required=True, help="why the proposal is refused"
    )
    approvals_reject_parser.add_argument(
        "--root", default=".", help="engine store root (default: .)"
    )

    args = parser.parse_args(argv)
    if args.command == "decisions" and args.decisions_command == "report":
        return _decisions_report(args.root)
    if args.command == "approvals" and args.approvals_command == "list":
        return _approvals_list(args.root)
    if args.command == "approvals" and args.approvals_command == "approve":
        return _approvals_decide(
            args.root, args.proposal_id, decided_by=args.by, reason=args.reason, reject=False
        )
    if args.command == "approvals" and args.approvals_command == "reject":
        return _approvals_decide(
            args.root, args.proposal_id, decided_by=args.by, reason=args.reason, reject=True
        )
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


def _approvals_list(root: str) -> int:
    """Render the proposal log: every proposal, with pending ones marked."""
    log = ApprovalLog(root)
    proposals = log.all()
    pending = [proposal for proposal in proposals if proposal.state is ApprovalState.PENDING]
    print(f"approval proposals: {len(proposals)} total, {len(pending)} pending")
    for proposal in proposals:
        print(f"  {proposal.proposal_id} [{proposal.state.value}]")
        print(f"    learning: {proposal.learning_id}")
        trail = proposal.decision.reason
        print(f"    trail: {trail} (evidence: {proposal.evidence_ref or 'none'})")
        if proposal.decided_by is not None:
            decided_reason = (
                f": {proposal.decided_reason}" if proposal.decided_reason else ""
            )
            print(f"    decided by {proposal.decided_by}{decided_reason}")
    return 0


def _approvals_decide(
    root: str,
    proposal_id: str,
    *,
    decided_by: str,
    reason: str | None,
    reject: bool,
) -> int:
    """Apply or refuse one proposal; a refused action exits non-zero."""
    engine = Engine(root)
    log = ApprovalLog(root)
    action = "rejected" if reject else "approved"
    try:
        if reject:
            record = reject_proposal(
                engine, log, proposal_id, decided_by=decided_by, reason=reason or ""
            )
        else:
            record = approve_proposal(
                engine, log, proposal_id, decided_by=decided_by, reason=reason
            )
    except (ApprovalError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{proposal_id} {action}: record {record.id} is {record.status}")
    return 0
