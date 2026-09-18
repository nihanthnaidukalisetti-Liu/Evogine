"""The task-scaled agent swarm: builder, verifier, reflector.

Three interchangeable roles, in v1 defined by contract only:

- :class:`Builder` — proposes candidate patches and draft learnings, warm-run
  first: it may retrieve only promoted knowledge from the engine.
- :class:`Verifier` — independently assesses proposals through the objective
  gates; it never proposes and never trusts the builder's self-assessment.
- :class:`Reflector` — diagnoses failures and proposes ephemeral learnings
  that the memory gate must prove before they are ever retrieved again.
"""

from __future__ import annotations


class Builder:
    """Proposes candidate patches and draft learnings using promoted knowledge."""

    def describe(self) -> str:
        """Return a one-paragraph statement of the builder's responsibilities."""
        raise NotImplementedError


class Verifier:
    """Independently assesses proposals through the two objective gates."""

    def describe(self) -> str:
        """Return a one-paragraph statement of the verifier's responsibilities."""
        raise NotImplementedError


class Reflector:
    """Diagnoses failures and proposes ephemeral learnings for memory-gate review."""

    def describe(self) -> str:
        """Return a one-paragraph statement of the reflector's responsibilities."""
        raise NotImplementedError
