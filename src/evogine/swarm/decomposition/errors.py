"""Refusal surface for the repository decomposer.

Decomposition is pure analysis over a working tree, so its failure modes are
narrow: a path that is not a readable Python repository, or an evidence file
(``coverage.json``) that exists but cannot be parsed. Every failure is a
typed, visible refusal — never a traceback and never a silently degraded
analysis.
"""

from __future__ import annotations


class DecompositionError(Exception):
    """The repository cannot be decomposed; the reason is the message."""
