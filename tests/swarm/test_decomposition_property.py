"""Property tests for the decomposer: determinism and perturbation stability.

Disk-shaped properties run over generated module trees in temporary
directories (deterministic walk, no network, no store); the scoring
properties are pure and run at the profile's full example budget.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from evogine.swarm.decomposition import (
    ModuleEvidence,
    build_module_graph,
    decompose,
    score_all,
)
from evogine.swarm.decomposition.models import DecompositionDAG

MODULE_POOL = ("alpha", "beta", "gamma", "delta")
TEST_POOL = ("test_alpha", "test_beta")

# Disk-shaped strategies: each example writes real files, so keep the
# example count low; pure-scoring tests below run at the profile budget.
TREE_EXAMPLES = 15


def _write_module_tree(root: Path, imports: dict[str, list[str]]) -> None:
    for name, targets in imports.items():
        body = "".join(f"import {target}\n" for target in targets)
        (root / f"{name}.py").write_text(f"{body}VALUE = 1\n", encoding="utf-8")


def _test_names(imports: dict[str, list[str]]) -> set[str]:
    return {name for name in imports if name.startswith("test_")}


@st.composite
def module_trees(draw):
    """An import map over a generated module set (code + test modules)."""
    code_names = draw(st.sets(st.sampled_from(MODULE_POOL), min_size=1, max_size=4))
    test_names = draw(st.sets(st.sampled_from(TEST_POOL), max_size=2))
    names = sorted(code_names | test_names)
    imports: dict[str, list[str]] = {}
    for name in names:
        others = [other for other in names if other != name]
        targets: st.SearchStrategy[list[str]] = (
            st.sets(st.sampled_from(others), max_size=min(3, len(others))).map(sorted)
            if others
            else st.just([])
        )
        imports[name] = draw(targets)
    return imports


@settings(max_examples=TREE_EXAMPLES)
@given(imports=module_trees())
def test_graph_invariants_hold_for_any_module_tree(imports):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_module_tree(root, imports)
        dag = build_module_graph(root)

        ids = [node.module for node in dag.modules]
        by_id = {node.module: node for node in dag.modules}

        # Nodes sorted by module id, one per generated file.
        assert ids == sorted(ids)
        assert set(ids) == set(imports)
        for name, targets in imports.items():
            assert by_id[name].imports == targets

        # Edge coherence: edges match node kinds, imported_by is the exact
        # reverse of imports, tested_by are the test-kind importers only.
        for edge in dag.edges:
            assert edge.source in by_id and edge.target in by_id
            if edge.source in _test_names(imports):
                assert edge.kind == "tested_by"
            else:
                assert edge.kind == "import"
        for node in dag.modules:
            expected_importers = sorted(
                name for name, targets in imports.items() if node.module in targets
            )
            assert node.imported_by == expected_importers
            assert node.tested_by == [
                name for name in expected_importers if name in _test_names(imports)
            ]

        # Stats agree with the lists, and code-node centrality stays in the
        # unit interval (test-only leaves legitimately carry centrality 0).
        assert dag.stats.modules == len(ids)
        assert dag.stats.code_modules == len(imports) - len(_test_names(imports))
        assert dag.stats.test_modules == len(_test_names(imports))
        for node in dag.modules:
            if node.kind == "code":
                assert 0.0 <= node.centrality <= 1.0


@settings(max_examples=TREE_EXAMPLES)
@given(imports=module_trees())
def test_rebuilding_the_same_tree_is_byte_identical(imports):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_module_tree(root, imports)

        first = build_module_graph(root)
        second = build_module_graph(root)

        assert first.model_dump_json() == second.model_dump_json()


@settings(max_examples=TREE_EXAMPLES)
@given(imports=module_trees())
def test_adding_an_isolated_module_leaves_every_existing_node_identical(imports):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_module_tree(root, imports)
        before = decompose(root)

        _write_module_tree(root, {"isolated_extra": []})
        after = decompose(root)

        # A fresh non-git tree scores churn 0 and unknown coverage for every
        # module; the isolated addition must not perturb any earlier node.
        old_by_id = {node.module: node for node in before.modules}
        assert {node.module for node in after.modules} - set(old_by_id) == {
            "isolated_extra"
        }
        for module_id, old_node in old_by_id.items():
            new_node = next(n for n in after.modules if n.module == module_id)
            assert new_node.model_dump() == old_node.model_dump()


@settings(max_examples=TREE_EXAMPLES)
@given(imports=module_trees())
def test_adding_one_import_moves_only_its_two_endpoints(imports):
    # alpha → beta must not already exist; both must be present.
    assume("alpha" in imports and "beta" in imports)
    assume("beta" not in imports["alpha"])
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_module_tree(root, imports)
        before = build_module_graph(root)

        imports["alpha"] = sorted(imports["alpha"] + ["beta"])
        _write_module_tree(root, {"alpha": imports["alpha"]})
        after = build_module_graph(root)

        old_by_id = {node.module: node for node in before.modules}
        for node in after.modules:
            old = old_by_id[node.module]
            if node.module in ("alpha", "beta"):
                continue
            # Structure untouched for bystanders; centrality can only be
            # diluted (the importer max may grow), never increased.
            assert node.imports == old.imports
            assert node.imported_by == old.imported_by
            assert node.tested_by == old.tested_by
            assert node.centrality <= old.centrality + 1e-12
        alpha_after = next(n for n in after.modules if n.module == "alpha")
        beta_after = next(n for n in after.modules if n.module == "beta")
        assert "beta" in alpha_after.imports
        assert "alpha" in beta_after.imported_by
        assert _max_code_importers(after) >= _max_code_importers(before)


def _max_code_importers(dag: DecompositionDAG) -> int:
    counts: dict[str, int] = {node.module: 0 for node in dag.modules}
    for edge in dag.edges:
        if edge.kind == "import" and edge.target in counts:
            counts[edge.target] += 1
    return max(counts.values(), default=0)


@settings(max_examples=100)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(0, 50),  # churn commits
            st.integers(0, 20),  # failure commits
            st.one_of(st.none(), st.floats(0, 100)),  # coverage gap
            st.integers(0, 30),  # importers
        ),
        min_size=1,
        max_size=12,
    )
)
def test_score_all_is_order_transparent(rows):
    """Scores follow the evidence, never its position in the list."""
    # None coverage gaps are not orderable against floats — give them a
    # fixed position in the canonical order instead of bare tuple sorting.
    original_rows = sorted(
        rows, key=lambda row: (row[0], row[1], -1.0 if row[2] is None else row[2], row[3])
    )
    original = [_evidence(row) for row in original_rows]
    shuffled = [_evidence(row) for row in rows]

    original_scores = score_all(original)
    shuffled_scores = score_all(shuffled)

    for position, row in enumerate(rows):
        canonical = original_rows.index(row)  # equal rows have equal scores
        assert shuffled_scores[position] == original_scores[canonical]


def _evidence(row: tuple[int, int, float | None, int]) -> ModuleEvidence:
    churn, failures, coverage_gap, importers = row
    return ModuleEvidence(
        churn_commits=churn,
        failure_commits=failures,
        coverage_gap=coverage_gap,
        imported_by_count=importers,
    )


@settings(max_examples=100)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(0, 10_000),
            st.integers(0, 10_000),
            st.one_of(st.none(), st.floats(0, 100)),
            st.integers(0, 10_000),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_every_score_is_bounded_and_the_top_module_is_one(rows):
    evidence = [_evidence(row) for row in rows]

    scores = score_all(evidence)

    assert len(scores) == len(rows)
    assert all(0.0 <= score <= 1.0 for score in scores)
    all_zero = all(
        churn == 0 and failures == 0 and (coverage_gap or 0.0) <= 0.0 and importers == 0
        for churn, failures, coverage_gap, importers in rows
    )
    if all_zero:
        assert scores == [0.0] * len(rows)  # honest zero: no evidence, no score
    else:
        # Some module attains the max of at least one active signal, and the
        # smallest weight is failure's 0.15 — so the top score is bounded
        # below by it.
        assert max(scores) >= 0.15
