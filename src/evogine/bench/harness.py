"""Warm-versus-cold benchmark harness with variance-aware reporting.

The harness runs each benchmark case twice — cold (no engine knowledge) and
warm (promoted knowledge retrieved for the task context) — under
decay-pressure conditions such as concurrency, retries, time zones, locale,
overflow, partial failure, and integration drift. It reports variance-aware
outcomes and preserves rollout provenance, so the claim "the engine helps"
is always backed by comparable, reproducible evidence. Hidden host-side
invariants and oracle-leak protection live in the case definitions, not here.
"""

from __future__ import annotations


class BenchHarness:
    """Warm-vs-cold decay-pressure benchmark runner (v1 stub)."""

    def run(self, case_ids: list[str]) -> dict[str, object]:
        """Run ``case_ids`` cold and warm and return a variance-aware report.

        The report maps each case to its cold and warm outcomes plus the
        delta, with rollout provenance preserved for audit.
        """
        raise NotImplementedError
