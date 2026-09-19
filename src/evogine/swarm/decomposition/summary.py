"""Human summary rendering for the decomposer's CLI output.

Rendering only — the numbers come from the :class:`DecompositionDAG` and the
layout mirrors the other ``evogine`` table outputs (fixed-width columns, an
honest "rest below the fold" line instead of an unbounded dump on large
repositories).
"""

from __future__ import annotations

from evogine.swarm.decomposition.criticality import DEFAULT_WEIGHTS, ScoreWeights
from evogine.swarm.decomposition.models import DecompositionDAG

DEFAULT_TOP = 20
"""Code modules shown before the summary switches to a count of the rest."""


def render_summary_table(dag: DecompositionDAG, *, top: int = DEFAULT_TOP) -> str:
    """Render the decompose summary: counts, then the criticality ranking."""
    stats = dag.stats
    head = f" (head {dag.head_commit[:12]})" if dag.head_commit else " (no git history)"
    weights: ScoreWeights = DEFAULT_WEIGHTS
    lines = [
        f"repository: {dag.repo_root}{head}",
        (
            f"modules: {stats.code_modules} code, {stats.test_modules} test — "
            f"{stats.import_edges} import edges, {stats.tested_by_edges} test mappings"
        ),
        "",
        (
            f"criticality ranking (churn {weights.churn:.0%} · failure {weights.failure:.0%} · "
            f"coverage {weights.coverage:.0%} · centrality {weights.centrality:.0%}):"
        ),
        "",
        (
            f"  {'crit':>6}  {'module':<40} {'loc':>5} {'churn':>5} {'fail':>5} "
            f"{'gap%':>6}  {'imported_by':>11} {'tested_by':>9}"
        ),
    ]
    ranked = sorted(
        (node for node in dag.modules if node.kind == "code"),
        key=lambda node: (
            -(node.criticality if node.criticality is not None else -1.0),
            node.module,
        ),
    )
    if not ranked:
        lines.append("  (no Python modules found — nothing to decompose)")
        return "\n".join(lines)
    for node in ranked[:top]:
        criticality = "—" if node.criticality is None else f"{node.criticality:.3f}"
        gap = "—" if node.coverage_gap is None else f"{node.coverage_gap:.1f}"
        marker = " !" if node.parse_error else " "
        lines.append(
            f"{marker}{criticality:>6}  {node.module:<40.40} {node.loc:>5} "
            f"{node.churn_commits:>5} {node.failure_commits:>5} {gap:>6}  "
            f"{len(node.imported_by):>11} {len(node.tested_by):>9}"
        )
    if len(ranked) > top:
        lines.append(f"  ({len(ranked) - top} more code modules below the top {top})")
    return "\n".join(lines)
