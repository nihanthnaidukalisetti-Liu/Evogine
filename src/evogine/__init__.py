"""evogine: a self-evolving agentic engine.

A git-backed knowledge engine + builder/verifier/reflector swarm + two
objective gates + decay-pressure benchmark, with a localhost operator
console, federated cross-project memory, and a git-native sync daemon.
This module is the installable package root; pyproject.toml holds the
release version (tests/test_version_parity.py enforces parity).
"""

__version__ = "4.0.0"

from evogine.engine.records import LearningRecord

__all__ = ["LearningRecord", "__version__"]
