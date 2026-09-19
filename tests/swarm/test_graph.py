"""Unit tests for the AST module-graph builder."""

from __future__ import annotations

import pytest
from tree_helpers import write_tree

from evogine.swarm.decomposition import DecompositionError, build_module_graph
from evogine.swarm.decomposition.models import DecompositionDAG, ModuleNode


def node_by_module(dag: DecompositionDAG, module_id: str) -> ModuleNode:
    return next(node for node in dag.modules if node.module == module_id)


def test_module_graph_resolves_flat_imports(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "alpha.py": "import beta\nimport argparse\n",
            "beta.py": "from alpha import VALUE\n",
        },
    )
    dag = build_module_graph(root)
    by_id = {node.module: node for node in dag.modules}

    assert set(by_id) == {"alpha", "beta"}
    assert by_id["alpha"].imports == ["beta"]
    assert by_id["alpha"].external_imports == 1  # argparse
    assert by_id["alpha"].imported_by == ["beta"]
    assert by_id["beta"].imports == ["alpha"]
    assert by_id["beta"].imported_by == ["alpha"]


def test_from_import_resolves_submodule_and_attribute(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/store.py": "VALUE = 1\n",
            "app.py": "from pkg.store import VALUE\nfrom pkg import store\n",
        },
    )
    dag = build_module_graph(root)
    by_id = {node.module: node for node in dag.modules}

    assert set(by_id) == {"pkg", "pkg.store", "app"}
    assert by_id["pkg"].path == "pkg/__init__.py"  # the package initializer node
    # `from pkg.store import VALUE` and `from pkg import store` both bind the
    # submodule, so they collapse to one edge — submodule-wins keeps
    # attribute-style imports from painting spurious package edges.
    assert by_id["app"].imports == ["pkg.store"]


def test_relative_imports_resolve_against_the_containing_package(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "pkg/__init__.py": "from . import mod\n",
            "pkg/mod.py": "from .other import THING\n",
            "pkg/other.py": "THING = 1\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/deep.py": "from ..mod import VALUE\n",
        },
    )
    dag = build_module_graph(root)
    by_id = {node.module: node for node in dag.modules}

    assert by_id["pkg"].imports == ["pkg.mod"]
    assert by_id["pkg.mod"].imports == ["pkg.other"]
    assert by_id["pkg.sub.deep"].imports == ["pkg.mod"]


def test_invalid_relative_import_resolves_to_nothing(tmp_path):
    root = write_tree(tmp_path, {"top.py": "from . import nowhere\n"})
    dag = build_module_graph(root)

    node = node_by_module(dag, "top")
    assert node.imports == []
    assert node.external_imports == 0
    assert node.imported_by == []


def test_src_layout_modules_use_their_import_name(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "src/evogine/cli.py": "from evogine.engine import store\n",
            "src/evogine/engine/__init__.py": "",
            "src/evogine/engine/store.py": "VALUE = 1\n",
        },
    )
    dag = build_module_graph(root)
    by_id = {node.module: node for node in dag.modules}

    assert set(by_id) == {"evogine.cli", "evogine.engine", "evogine.engine.store"}
    assert by_id["evogine.cli"].imports == ["evogine.engine.store"]


def test_tests_to_code_mapping_uses_tested_by_edges(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "core.py": "VALUE = 1\n",
            "tests/test_core.py": "import core\nimport pytest\n",
        },
    )
    dag = build_module_graph(root)
    by_id = {node.module: node for node in dag.modules}

    assert by_id["core"].kind == "code"
    assert by_id["tests.test_core"].kind == "test"
    assert by_id["core"].tested_by == ["tests.test_core"]
    assert by_id["tests.test_core"].tested_by == []
    assert by_id["core"].imported_by == ["tests.test_core"]
    assert by_id["tests.test_core"].external_imports == 1  # pytest
    edge = next(edge for edge in dag.edges if edge.source == "tests.test_core")
    assert edge.kind == "tested_by"


def test_parse_error_keeps_the_module_visible(tmp_path):
    root = write_tree(
        tmp_path,
        {"broken.py": "def (:\n", "ok.py": "import broken\n"},
    )
    dag = build_module_graph(root)
    by_id = {node.module: node for node in dag.modules}

    assert by_id["broken"].parse_error is True
    assert by_id["broken"].imports == []
    assert by_id["ok"].imports == ["broken"]  # the file exists, so it resolves


def test_ignored_directories_are_not_discovered(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "mod.py": "VALUE = 1\n",
            ".venv/x.py": "import mod\n",
            "__pycache__/y.py": "import mod\n",
            "build/z.py": "import mod\n",
        },
    )
    dag = build_module_graph(root)

    assert [node.module for node in dag.modules] == ["mod"]


def test_missing_repo_refuses(tmp_path):
    with pytest.raises(DecompositionError, match="not a directory"):
        build_module_graph(tmp_path / "absent")


def test_repo_path_is_a_file_refuses(tmp_path):
    target = tmp_path / "not-a-dir.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(DecompositionError, match="not a directory"):
        build_module_graph(target)


def test_empty_directory_decomposes_to_honest_zero(tmp_path):
    dag = build_module_graph(tmp_path)

    assert dag.modules == []
    assert dag.edges == []
    assert dag.stats.modules == 0
    assert dag.stats.code_modules == 0
    assert dag.stats.test_modules == 0


def test_graph_json_round_trip(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "alpha.py": "import beta\n",
            "beta.py": "VALUE = 1\n",
            "tests/test_alpha.py": "import alpha\n",
        },
    )
    dag = build_module_graph(root)

    revived = DecompositionDAG.model_validate_json(dag.model_dump_json())
    assert revived == dag
