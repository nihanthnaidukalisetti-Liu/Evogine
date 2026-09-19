"""Small on-disk builders shared by the decomposition tests.

Kept as plain functions rather than a conftest fixture: a second
``conftest.py`` in the tree collides with the repo-root conftest under the
suite's existing ``from conftest import …`` pattern.
"""

from __future__ import annotations

from pathlib import Path


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``{relative path: content}`` files under root, creating dirs."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root
