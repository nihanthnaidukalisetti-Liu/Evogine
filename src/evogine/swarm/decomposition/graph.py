"""The module graph: discovery, ast import extraction, tests-to-code mapping.

Builds the graph the swarm orchestrator runs and the dashboard draws, from
three passes over the working tree:

1. **Discover** every ``*.py`` file (deterministic walk; caches, virtualenvs,
   build outputs, and symlinked directories are skipped — a symlinked
   directory can cycle and defeat the walk, and vendored trees are not the
   repository's own modules).
2. **Parse** each file with ``ast`` and resolve its imports against the
   discovered module ids. Only internal edges survive: a module id is its
   importable dotted name (``src/``-layout stripped — ``src/evogine/cli.py``
   is ``evogine.cli``), and resolution walks dotted prefixes, so
   ``from evogine.engine import store`` lands on ``evogine.engine.store``
   while ``from evogine.gates.funnel import FunnelReport`` lands on its
   parent when the attribute is not a module. Relative imports resolve
   against the containing package. Anything unresolved is counted as an
   external import — metadata for the summary, never a graph edge.
3. **Map tests**: a module is test-kind when it lives under a ``tests``/
   ``test`` directory or matches ``test_*.py`` / ``*_test.py`` /
   ``conftest.py``. Its internal imports become ``tested_by`` edges — the
   tests-to-code mapping — not plain import edges, so the orchestrator's
   code-task graph stays clean while the mapping feeds coverage-aware
   scoring and the dashboard.

:func:`build_module_graph` is the graph alone (centrality computed,
criticality ``None``); :func:`decompose` adds the evidence pass — git churn
and failure history, ``coverage.json`` gaps — and scores criticality. Both
are pure analysis over the tree: no store writes, no network, no config.

A file that fails to parse stays in the graph with ``parse_error=True`` and
no import edges — a broken module is a real task for a repair swarm, not a
silent omission. This matters for swarm use: pointing the decomposer at a
tree mid-refactor must not crash it.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

from evogine.swarm.decomposition.criticality import (
    DEFAULT_WEIGHTS,
    ModuleEvidence,
    ScoreWeights,
    score_all,
)
from evogine.swarm.decomposition.errors import DecompositionError
from evogine.swarm.decomposition.evidence import (
    collect_coverage_gaps,
    collect_git_evidence,
)
from evogine.swarm.decomposition.models import (
    DagEdge,
    DecompositionDAG,
    DecompositionStats,
    ModuleKind,
    ModuleNode,
)

IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        ".idea",
        ".vscode",
    }
)

TEST_DIRECTORY_NAMES = frozenset({"tests", "test"})

# A compiler directive, not a dependency: `from __future__ import annotations`
# says nothing about module coupling and would only be dashboard noise.
_FUTURE_MODULE = "__future__"


def _is_test_module(rel_path: PurePosixPath) -> bool:
    """Test-kind: lives in a test directory, or is named like a test module."""
    if any(part in TEST_DIRECTORY_NAMES for part in rel_path.parts[:-1]):
        return True
    name = rel_path.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _iter_python_files(repo: Path) -> list[PurePosixPath]:
    """Every ``*.py`` under ``repo``, repo-relative, in sorted order.

    Deterministic by construction: children are visited in sorted order and
    the result is sorted again, so the walk order never leaks into the graph.
    """
    found: list[PurePosixPath] = []
    stack = [repo]
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.is_dir():
                if not entry.is_symlink() and entry.name not in IGNORED_DIRECTORY_NAMES:
                    stack.append(entry)
            elif entry.suffix == ".py" and entry.is_file():
                found.append(PurePosixPath(entry.relative_to(repo).as_posix()))
    return sorted(found)


def _source_parts(rel_path: PurePosixPath) -> list[str]:
    """Dotted-name parts for a path, dropping the trailing ``__init__``."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _strip_src(parts: list[str]) -> list[str]:
    """Importable parts of a ``src/``-layout path (no-op otherwise)."""
    if len(parts) > 1 and parts[0] == "src":
        return parts[1:]
    return parts


def _module_id(rel_path: PurePosixPath) -> str:
    """The importable dotted name for a file: ``src``-stripped, ``__init__``-aware."""
    stripped = _strip_src(_source_parts(rel_path))
    name = ".".join(stripped)
    return name if name else rel_path.as_posix()


def _alias_names(rel_path: PurePosixPath) -> list[str]:
    """Every dotted name imports can use to reach this file.

    A ``src/``-layout file is importable both as ``evogine.cli`` (canonical)
    and — from tooling that treats ``src`` as a namespace root — as
    ``src.evogine.cli``; indexing both makes import resolution robust to
    either convention.
    """
    parts = _source_parts(rel_path)
    names = [_module_id(rel_path)]
    unstripped = ".".join(parts)
    if parts and parts[0] == "src" and unstripped and unstripped != names[0]:
        names.append(unstripped)
    return names


def _lookup(index: dict[str, str], dotted: str) -> str | None:
    """Resolve a dotted name to a module id, walking up parent packages."""
    candidate = dotted
    while candidate:
        if candidate in index:
            return index[candidate]
        candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""
    return None


def _relative_base(own_id: str, own_is_package: bool, level: int) -> str | None:
    """The absolute base package for a relative import, or ``None`` if invalid.

    ``from . import x`` in ``evogine/cli.py`` bases on ``evogine``; in
    ``evogine/swarm/__init__.py`` (itself a package) it bases on
    ``evogine.swarm``. Each extra level climbs one package; climbing past the
    top, or a relative import from a top-level module, is invalid — the
    import would raise at runtime, so it resolves to no edge.
    """
    parts = own_id.split(".") if own_id else []
    if not own_is_package:
        if not parts:
            return None  # a top-level module has no package to be relative to
        parts = parts[:-1]
    drops = level - 1
    if drops > 0:
        if drops > len(parts):
            return None
        parts = parts[:-drops]
    base = ".".join(parts)
    return base or None


class _ParsedModule:
    """One discovered file's facts before the graph is assembled."""

    def __init__(self, rel_path: PurePosixPath, *, is_package: bool) -> None:
        self.rel_path = rel_path
        self.module_id = _module_id(rel_path)
        self.is_package = is_package
        self.kind: ModuleKind = "test" if _is_test_module(rel_path) else "code"
        self.loc = 0
        self.parse_error = False
        self.internal_targets: set[str] = set()
        self.external_roots: set[str] = set()


def _scan_imports(
    parsed: _ParsedModule, tree: ast.Module, index: dict[str, str]
) -> None:
    """Fill ``parsed.internal_targets`` and ``parsed.external_roots``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _lookup(index, alias.name)
                if hit is not None and hit != parsed.module_id:
                    parsed.internal_targets.add(hit)
                elif hit is None:
                    parsed.external_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base_full = node.module or None
            else:
                rel_base = _relative_base(
                    parsed.module_id, parsed.is_package, node.level
                )
                if rel_base is None:
                    continue
                base_full = f"{rel_base}.{node.module}" if node.module else rel_base
            for alias in node.names:
                if alias.name == "*" and base_full is not None:
                    hit = _lookup(index, base_full)
                    if hit is not None and hit != parsed.module_id:
                        parsed.internal_targets.add(hit)
                    elif hit is None and node.level == 0:
                        parsed.external_roots.add(base_full.split(".", 1)[0])
                    continue
                if base_full is None:
                    continue
                candidate = f"{base_full}.{alias.name}"
                hit = _lookup(index, candidate)
                if hit is None:
                    # The base itself may still be internal (`from pydantic
                    # import BaseModel` is external; `from evogine import x`
                    # with x an attribute is an edge to `evogine`).
                    if node.level == 0:
                        root_hit = _lookup(index, base_full)
                        if root_hit is not None and root_hit != parsed.module_id:
                            parsed.internal_targets.add(root_hit)
                        else:
                            parsed.external_roots.add(base_full.split(".", 1)[0])
                    continue
                if hit != parsed.module_id:
                    parsed.internal_targets.add(hit)
    parsed.external_roots.discard(_FUTURE_MODULE)


def build_module_graph(repo: Path) -> DecompositionDAG:
    """Build the module graph for ``repo`` — structure and centrality only.

    Evidence fields stay at the honest zero and ``criticality`` stays
    ``None``; :func:`decompose` adds the evidence pass.
    """
    resolved_repo = repo.resolve()
    if not resolved_repo.is_dir():
        raise DecompositionError(f"repository path is not a directory: {repo}")
    rel_paths = _iter_python_files(resolved_repo)

    # Index dotted names → module ids. Sorted-path-first wins collisions
    # (a repo with both `pkg/mod.py` and `src/pkg/mod.py`): deterministic,
    # and the loser keeps a distinct id rather than vanishing. Aliases use
    # setdefault so a later file's alias can never hijack an earlier file's
    # canonical id.
    index: dict[str, str] = {}
    claimed: set[str] = set()
    for rel in rel_paths:
        candidates = _alias_names(rel)
        free = [name for name in candidates if name not in claimed]
        module_id = free[0] if free else rel.as_posix()
        claimed.add(module_id)
        for name in candidates:
            index.setdefault(name, module_id)
        index.setdefault(module_id, module_id)

    parsed_modules: list[_ParsedModule] = []
    for rel in rel_paths:
        is_package = rel.name == "__init__.py"
        parsed = _ParsedModule(rel, is_package=is_package)
        source_path = resolved_repo / rel
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DecompositionError(f"cannot read {rel} in {resolved_repo}: {exc}") from exc
        parsed.loc = len(text.splitlines())
        try:
            tree = ast.parse(text, filename=str(rel))
        except (SyntaxError, ValueError, RecursionError):
            parsed.parse_error = True
        else:
            _scan_imports(parsed, tree, index)
        parsed_modules.append(parsed)

    imported_by: dict[str, set[str]] = {mid: set() for mid in claimed}
    code_importers: dict[str, set[str]] = {mid: set() for mid in claimed}
    edges: list[DagEdge] = []
    for parsed in parsed_modules:
        for target in sorted(parsed.internal_targets):
            imported_by[target].add(parsed.module_id)
            if parsed.kind == "code":
                code_importers[target].add(parsed.module_id)
            edges.append(
                DagEdge(
                    source=parsed.module_id,
                    target=target,
                    kind="tested_by" if parsed.kind == "test" else "import",
                )
            )

    max_importers = max((len(v) for v in code_importers.values()), default=0)
    test_module_ids = {
        parsed.module_id for parsed in parsed_modules if parsed.kind == "test"
    }
    nodes = [
        ModuleNode(
            module=parsed.module_id,
            path=parsed.rel_path.as_posix(),
            kind=parsed.kind,
            loc=parsed.loc,
            external_imports=len(parsed.external_roots),
            parse_error=parsed.parse_error,
            imports=sorted(parsed.internal_targets),
            imported_by=sorted(imported_by[parsed.module_id]),
            tested_by=sorted(imported_by[parsed.module_id] & test_module_ids),
            centrality=(
                len(code_importers[parsed.module_id]) / max_importers
                if max_importers
                else 0.0
            ),
        )
        for parsed in parsed_modules
    ]

    kind_by_module = {node.module: node.kind for node in nodes}
    stats = DecompositionStats(
        modules=len(nodes),
        code_modules=sum(1 for kind in kind_by_module.values() if kind == "code"),
        test_modules=sum(1 for kind in kind_by_module.values() if kind == "test"),
        import_edges=sum(1 for edge in edges if edge.kind == "import"),
        tested_by_edges=sum(1 for edge in edges if edge.kind == "tested_by"),
    )
    return DecompositionDAG(
        repo_root=str(resolved_repo),
        head_commit=None,
        modules=nodes,
        edges=edges,
        stats=stats,
    )


def decompose(
    repo: Path,
    *,
    weights: ScoreWeights | None = None,
) -> DecompositionDAG:
    """Build the graph, collect evidence, and score criticality.

    The full decomposition: graph structure plus per-module churn, failure
    density, and coverage gap, with centrality already graph-derived. Pure
    analysis — no store writes, no network, no configuration.
    """
    dag = build_module_graph(repo)
    git_evidence = collect_git_evidence(repo)
    gaps = collect_coverage_gaps(repo)

    # Centrality's evidence basis is code-only importers: being imported by
    # tests maps coverage (tested_by), it does not make a module load-bearing.
    code_importer_counts: dict[str, int] = {node.module: 0 for node in dag.modules}
    for edge in dag.edges:
        if edge.kind == "import" and edge.target in code_importer_counts:
            code_importer_counts[edge.target] += 1

    evidence: list[ModuleEvidence] = []
    updated: list[ModuleNode] = []
    for node in dag.modules:
        gap = gaps.get(node.path)
        evidence.append(
            ModuleEvidence(
                churn_commits=git_evidence.churn.get(node.path, 0),
                failure_commits=git_evidence.failures.get(node.path, 0),
                coverage_gap=gap,
                imported_by_count=code_importer_counts[node.module],
            )
        )
        updated.append(
            node.model_copy(
                update={
                    "churn_commits": evidence[-1].churn_commits,
                    "failure_commits": evidence[-1].failure_commits,
                    "coverage_gap": gap,
                }
            )
        )

    scores = score_all(evidence, weights or DEFAULT_WEIGHTS)
    scored = [
        node.model_copy(update={"criticality": score})
        for node, score in zip(updated, scores)
    ]
    return DecompositionDAG(
        repo_root=dag.repo_root,
        head_commit=git_evidence.head_commit,
        modules=scored,
        edges=dag.edges,
        stats=dag.stats,
    )
