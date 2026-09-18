"""evogine: a self-evolving agentic engine (v1 scaffold).

A git-backed knowledge engine + builder/verifier/reflector swarm + two
objective gates + decay-pressure benchmark. This module is the installable
package root; the v1 design spec defines the full architecture.
"""

__version__ = "0.1.0"

from evogine.engine.records import LearningRecord

__all__ = ["LearningRecord", "__version__"]
