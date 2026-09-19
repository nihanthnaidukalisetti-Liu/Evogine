"""Typed shapes for the decomposition DAG: the orchestrator/dashboard contract.

:class:`ModuleNode` carries one Python module's graph position and criticality
evidence; :class:`DecompositionDAG` carries the whole repository's graph. Both
are plain JSON-serializable pydantic models (``model_dump(mode="json")`` emits
no paths-as-objects, no sets, no datetimes), so the same object feeds the
swarm orchestrator's task scheduler (a module's ``imports`` are its task
prerequisites; ``criticality`` is its priority) and the mission-control
dashboard's DAG view.

Canonical ordering is an invariant of the model, not a builder habit: the
``model_validator`` re-sorts modules and edges on construction and on every
``model_validate`` round-trip, so two decompositions of the same repository
state serialize byte-identically regardless of directory-walk order.

Evidence fields default to the honest zero — ``0``/``None`` means "no
evidence found" (no git history, no ``coverage.json``), never a fabricated
signal.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModuleKind = Literal["code", "test"]
EdgeKind = Literal["import", "tested_by"]


class DagEdge(BaseModel):
    """One directed dependency: ``source`` depends on (or tests) ``target``.

    ``import`` edges are the code-task dependencies the orchestrator
    schedules against; ``tested_by`` edges are the tests-to-code mapping
    (source is always a test-kind module) that coverage-aware scoring and the
    dashboard read. A test module's import of code appears once, as a
    ``tested_by`` edge — never double-counted as an ``import`` edge.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: EdgeKind


class ModuleNode(BaseModel):
    """One Python module with its graph position and criticality evidence.

    ``imports``/``imported_by`` list resolved internal module ids (sorted);
    ``tested_by`` lists the test modules that import this one. ``centrality``
    is the code-only import count normalized to the repo's busiest module;
    ``criticality`` is the weighted criticality score (``None`` when the
    graph was built without scoring).
    """

    model_config = ConfigDict(extra="forbid")

    module: str
    path: str
    kind: ModuleKind
    loc: int = Field(default=0, ge=0)
    external_imports: int = Field(default=0, ge=0)
    parse_error: bool = False
    imports: list[str] = Field(default_factory=list)
    imported_by: list[str] = Field(default_factory=list)
    tested_by: list[str] = Field(default_factory=list)
    churn_commits: int = Field(default=0, ge=0)
    failure_commits: int = Field(default=0, ge=0)
    coverage_gap: float | None = Field(default=None, ge=0, le=100)
    centrality: float = Field(default=0.0, ge=0, le=1)
    criticality: float | None = Field(default=None, ge=0, le=1)


class DecompositionStats(BaseModel):
    """Counts summarizing the graph — the summary line's numbers."""

    model_config = ConfigDict(extra="forbid")

    modules: int = Field(ge=0)
    code_modules: int = Field(ge=0)
    test_modules: int = Field(ge=0)
    import_edges: int = Field(ge=0)
    tested_by_edges: int = Field(ge=0)


class DecompositionDAG(BaseModel):
    """The whole repository as a criticality-weighted module graph.

    ``repo_root`` is the absolute path analyzed and ``head_commit`` the git
    HEAD the evidence was read against (``None`` without git). ``modules`` is
    sorted by module id and ``edges`` by ``(kind, source, target)``; both are
    re-sorted by the validator, so serialization is canonical.
    """

    model_config = ConfigDict(extra="forbid")

    repo_root: str
    head_commit: str | None = None
    modules: list[ModuleNode] = Field(default_factory=list)
    edges: list[DagEdge] = Field(default_factory=list)
    stats: DecompositionStats

    @model_validator(mode="after")
    def _canonical_order(self) -> DecompositionDAG:
        self.modules.sort(key=lambda node: node.module)
        self.edges.sort(key=lambda edge: (edge.kind, edge.source, edge.target))
        return self
