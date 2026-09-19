"""CLI tests for ``evogine decompose`` and the this-repository invariant test."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from tree_helpers import write_tree

from evogine.cli import main
from evogine.swarm.decomposition import DecompositionDAG, decompose

REPO_ROOT = Path(__file__).resolve().parents[2]

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _repo_is_shallow() -> bool:
    """CI checkouts use fetch-depth 1 — git history assertions only when full.

    A git that cannot answer at all (restricted sandboxes stage candidate
    trees without reachable git metadata) is treated like a shallow clone:
    no history to mine, so history-shaped assertions stay skipped.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return result.stdout.strip() == "true"


def test_decompose_cli_prints_a_summary_table(tmp_path, capsys):
    root = write_tree(
        tmp_path,
        {"alpha.py": "import beta\n", "beta.py": "VALUE = 1\n"},
    )

    code = main(["decompose", "--repo", str(root)])

    assert code == 0
    out = capsys.readouterr().out
    assert "criticality ranking" in out
    assert "alpha" in out
    assert "beta" in out


def test_decompose_cli_json_option_writes_a_dag_file(tmp_path):
    root = write_tree(
        tmp_path,
        {"alpha.py": "import beta\n", "beta.py": "VALUE = 1\n"},
    )
    out_path = tmp_path / "dag.json"

    code = main(["decompose", "--repo", str(root), "--json", str(out_path)])

    assert code == 0
    dag = DecompositionDAG.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert dag.stats.modules == 2
    assert {node.module for node in dag.modules} == {"alpha", "beta"}


def test_decompose_cli_json_stdout_alongside_the_table(tmp_path, capsys):
    root = write_tree(tmp_path, {"alpha.py": "VALUE = 1\n"})

    code = main(["decompose", "--repo", str(root), "--json", "-"])

    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert DecompositionDAG.model_validate(payload).stats.modules == 1


def test_decompose_cli_refuses_a_missing_directory(tmp_path, capsys):
    code = main(["decompose", "--repo", str(tmp_path / "absent")])

    assert code == 1
    assert "not a directory" in capsys.readouterr().err


@pytest.mark.skipif(
    not (REPO_ROOT / "pyproject.toml").is_file(),
    reason="this test needs the Evogine checkout itself",
)
def test_decompose_this_repository_asserts_sane_invariants(tmp_path):
    """The sample decomposition on a real repo: this repository, at HEAD.

    Assertions are drift-proof on purpose (structure, not history-shaped
    numbers): the checkout is a live repo whose module set and ranking move
    with every merge.
    """
    dag = decompose(REPO_ROOT)
    by_id = {node.module: node for node in dag.modules}

    # The graph covers this repository's own surface, including the
    # decomposer itself and the CLI it is registered in.
    assert dag.stats.code_modules >= 40
    assert dag.stats.test_modules >= 40
    assert "evogine.swarm.decomposition.graph" in by_id
    assert "evogine.cli" in by_id

    # Git identity is well-formed when present. The Docker gate stages the
    # suite on trees without reachable git metadata, where the honest value
    # is None — absent evidence is never invented (the model's contract).
    assert dag.head_commit is None or HEX40.match(dag.head_commit)
    store = by_id["evogine.engine.store"]
    assert store.criticality is not None and 0.0 < store.criticality <= 1.0
    assert store.imported_by  # the engine store is load-bearing here

    # Load-bearing modules rank by criticality: the top module is one the
    # pool would genuinely visit first — churned and centrally imported.
    top = max(
        (node for node in dag.modules if node.kind == "code"),
        key=lambda node: node.criticality or -1.0,
    )
    assert top.centrality >= 0.2
    # History-shaped asserts need a resolvable HEAD AND full history: a
    # shallow checkout or a git-less staging area yields honest zeros, and
    # asserting against them would test the runner's environment, not the
    # decomposer.
    if dag.head_commit is not None and not _repo_is_shallow():
        assert store.churn_commits >= 1
        assert top.churn_commits >= 5

    # Full graph coherence on the real tree.
    ids = {node.module for node in dag.modules}
    for edge in dag.edges:
        assert edge.source in ids and edge.target in ids
    for node in dag.modules:
        for target in node.imports:
            assert target in ids
        assert node.tested_by == [
            importer for importer in node.imported_by if importer in _test_ids(dag)
        ]

    # Determinism on the real tree, end to end.
    again = decompose(REPO_ROOT)
    assert again.model_dump_json() == dag.model_dump_json()

    # And the CLI surfaces the same DAG through --json.
    out_path = tmp_path / "self.json"
    assert main(["decompose", "--repo", str(REPO_ROOT), "--json", str(out_path)]) == 0
    assert DecompositionDAG.model_validate_json(out_path.read_text(encoding="utf-8")) == dag


def _test_ids(dag: DecompositionDAG) -> set[str]:
    return {node.module for node in dag.modules if node.kind == "test"}
