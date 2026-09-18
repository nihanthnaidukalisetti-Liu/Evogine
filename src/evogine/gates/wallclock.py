"""Verifier wall-clock cross-check: a too-fast pass is a verifier failure.

The verifier's authority is entirely objective execution — a PASS means the
case's command really ran inside the sandbox and exited 0. Real execution
has a physical floor: interpreter startup, imports, the work itself. A PASS
reported below that floor was not earned by execution (a stale cache, a
short-circuited evaluator, a fabricated result), and the only safe response
is to fail closed: the case is recorded as a failure, the gate rejects the
patch, and the audit trail carries the reason. Fail verdicts are exempt —
failing fast is legitimate.

The check is a cross-check, not a replacement for the exit code: the floor
is per-case configuration (``EvalCase.min_pass_seconds``) because only the
case author knows what the command genuinely costs. A missing duration on a
PASS is treated like a too-fast one — an unmeasurable pass cannot be
distinguished from a fabricated one, so it fails closed.
"""

from __future__ import annotations

from evogine.gates.equivalence import CaseVerdict


def crosscheck_too_fast_pass(
    *,
    verdict: CaseVerdict,
    duration_ms: float | None,
    min_pass_seconds: float,
) -> str | None:
    """Return the fail-closed detail when a PASS is implausibly fast, else ``None``.

    The returned string is a rejection detail for the case's ``CaseResult``;
    ``None`` means the verdict survived the cross-check. Only PASS verdicts
    are cross-checked — a too-fast FAIL is an honest fast failure.
    """
    if verdict is not CaseVerdict.PASS:
        return None
    if duration_ms is None:
        return (
            f"wall-clock cross-check: PASS carries no recorded duration "
            f"(floor {min_pass_seconds:g}s) — an unmeasurable pass cannot be "
            "distinguished from a fabricated one; failing closed"
        )
    seconds = duration_ms / 1000.0
    if seconds < min_pass_seconds:
        return (
            f"wall-clock cross-check: PASS in {duration_ms:.0f}ms is below the "
            f"{min_pass_seconds:g}s floor — the case cannot have executed; "
            "failing closed"
        )
    return None


__all__ = ["crosscheck_too_fast_pass"]
