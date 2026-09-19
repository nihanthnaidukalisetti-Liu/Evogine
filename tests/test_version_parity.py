"""Version parity: pyproject.toml is the single version source.

The release version lives in pyproject.toml and nowhere else by policy; the
package ``__version__`` must match it exactly. This test fails on drift, so
a release bump can never half-land (bumped in one place, forgotten in the
other).
"""

import tomllib
from pathlib import Path

import evogine

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_package_version_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as f:
        pyproject = tomllib.load(f)
    declared = pyproject["project"]["version"]
    assert declared, "pyproject [project] version must be declared"
    assert evogine.__version__ == declared
