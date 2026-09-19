"""The append-only audit merge driver — the documented merge surface.

Two machines can both append to ``audit.jsonl`` between syncs. The brief
makes documenting and testing this the daemon's job because it is the ONE
surface git cannot merge by default (both sides modified the same file);
everything else in the store is one JSON file per record, so disjoint
lifecycles touch disjoint files and merge clean with stock git.

Registration (any store that will *pull*):

    git config merge.evogine-audit.name "Evogine audit union (append-only)"
    git config merge.evogine-audit.driver "python3 -m evogine.sync.merge %O %A %B"
    printf 'audit.jsonl merge=evogine-audit\\n' >> .gitattributes

Semantics: with the common base as the anchor, each side must be the base
lines followed by appended lines (append-only, enforced — a side whose
prefix differs from the base has rewritten history, and the driver exits
nonzero so git reports a conflict instead of silently adopting the
rewrite). The merge is the base, then our appends, then their appends
(exact duplicate lines appended by both sides are kept once), each side's
internal order preserved verbatim. A same-record lifecycle collision (two
machines transitioning the same learning id) is NOT this driver's to
resolve: the per-record JSON conflicts visibly in stock git, and the
operator reconciles — the daemon never rewrites knowledge.

The git history is the reconciliation record: after a merge, both lines
of descent are commits in the hosted store, and a push transfers them in
order. No global ordering across stores is invented (governance invariant
3.3) — each trail's internal order is the only order that exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _lines(text: str) -> list[str]:
    """Non-blank JSONL lines, order preserved."""
    return [line for line in text.splitlines() if line.strip()]


def _appends(base: list[str], side: list[str], side_name: str) -> list[str]:
    """The lines ``side`` appended after the common base, refusing rewrites."""
    prefix = side[: len(base)]
    if prefix != base:
        raise ValueError(
            f"{side_name} diverged from the audit's append-only contract: its first "
            f"{len(base)} line(s) differ from the common base. The driver refuses to "
            "merge rewritten history — reconcile the trail by hand."
        )
    return side[len(base) :]


def merge_audit_lines(base: str, ours: str, theirs: str) -> str:
    """Union of two append-only audit trails against their common base.

    Raises ``ValueError`` when either side is not base-plus-appends, so a
    caller can fail the merge visibly.
    """
    base_lines, ours_lines, theirs_lines = _lines(base), _lines(ours), _lines(theirs)
    ours_appends = _appends(base_lines, ours_lines, "ours")
    theirs_appends = [
        line
        for line in _appends(base_lines, theirs_lines, "theirs")
        if line not in set(ours_appends)
    ]
    merged = base_lines + ours_appends + theirs_appends
    return "".join(f"{line}\n" for line in merged)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m evogine.sync.merge %O %A %B`` (git merge driver).

    Exit 0 with ``%A`` replaced by the union; exit 1 (git reports a real
    conflict) on any append-only violation.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evogine.sync.merge",
        description="Evogine's union merge driver for append-only audit trails.",
    )
    parser.add_argument("base", help="merge base version (%O)")
    parser.add_argument("ours", help="current branch version (%A)")
    parser.add_argument("theirs", help="other branch version (%B)")
    args = parser.parse_args(argv)
    try:
        merged = merge_audit_lines(
            Path(args.base).read_text(encoding="utf-8"),
            Path(args.ours).read_text(encoding="utf-8"),
            Path(args.theirs).read_text(encoding="utf-8"),
        )
        Path(args.ours).write_text(merged, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
