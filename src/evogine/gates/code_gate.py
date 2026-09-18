"""Objective gate #1: sandboxed behavioral verification of code changes.

The code gate decides whether a builder-proposed change may land, and it
never trusts the proposing agent's own claims. Its three checks, per the v1
design: a sandboxed pytest run inside the locked-down execution sandbox;
differential equivalence, ensuring the patch is behavior-preserving where it
claims to be; and a never-worse check, ensuring the patched candidate is not
worse than the baseline on any evaluated case. A patch that fails any check
is rejected outright — the gate is objective, not advisory.
"""

from __future__ import annotations


class CodeGate:
    """Objective verifier for candidate code changes (v1 stub)."""

    def evaluate(self, candidate_patch: str, baseline_ref: str) -> bool:
        """Return ``True`` iff ``candidate_patch`` passes all three checks.

        ``candidate_patch`` is the proposed diff text; ``baseline_ref`` is
        the git ref of the evaluated baseline. Runs sandboxed pytest, checks
        differential equivalence, and enforces never-worse-than-baseline.
        """
        raise NotImplementedError
