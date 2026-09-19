"""Repository decomposition: module graph + criticality scoring (V4 swarm).

Pure analysis over a Python repository's working tree — no store writes, no
network, no configuration. The graph (imports, tests-to-code mapping) is the
task DAG the swarm orchestrator runs; per-module criticality (churn, failure
density, coverage gaps, import centrality) is the priority the pool works
in. :func:`decompose` is the one-call entry point;
``evogine decompose --repo PATH`` is its CLI face.
"""

from __future__ import annotations

from evogine.swarm.decomposition.criticality import (
    DEFAULT_WEIGHTS,
    ModuleEvidence,
    NormalizedSignals,
    ScoreWeights,
    normalize_evidence,
    score_all,
    score_signals,
)
from evogine.swarm.decomposition.errors import DecompositionError
from evogine.swarm.decomposition.evidence import (
    GitEvidence,
    collect_coverage_gaps,
    collect_git_evidence,
    is_failure_subject,
)
from evogine.swarm.decomposition.graph import build_module_graph, decompose
from evogine.swarm.decomposition.models import (
    DagEdge,
    DecompositionDAG,
    DecompositionStats,
    ModuleKind,
    ModuleNode,
)
from evogine.swarm.decomposition.summary import render_summary_table

__all__ = [
    "DEFAULT_WEIGHTS",
    "DagEdge",
    "DecompositionDAG",
    "DecompositionError",
    "DecompositionStats",
    "GitEvidence",
    "ModuleEvidence",
    "ModuleKind",
    "ModuleNode",
    "NormalizedSignals",
    "ScoreWeights",
    "build_module_graph",
    "collect_coverage_gaps",
    "collect_git_evidence",
    "decompose",
    "is_failure_subject",
    "normalize_evidence",
    "render_summary_table",
    "score_all",
    "score_signals",
]
