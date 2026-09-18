"""Pure equivalence logic behind the code gate's verdicts.

Three small, deterministic pieces — bytecode signatures for claimed
refactors, differential flips between outcome maps, and the never-worse rule
— kept pure so they are unit-testable without Docker or live models. The gate
(see ``code_gate.py``) orchestrates them; nothing here touches I/O.
"""

from __future__ import annotations

import dis
import hashlib
import types
from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel


class CaseVerdict(str, Enum):
    """Binary outcome of one evaluated case run in one tree."""

    PASS = "pass"
    FAIL = "fail"


class CaseResult(BaseModel):
    """One evaluated case's outcome, with sanitized detail for the audit."""

    case_id: str
    verdict: CaseVerdict
    detail: str = ""


def _render_const(value: object) -> str:
    """Stable rendering of a co_const, immune to code-object addresses."""
    if isinstance(value, types.CodeType):
        return f"<code:{value.co_qualname}>"
    if isinstance(value, tuple):
        return "(" + ", ".join(_render_const(item) for item in value) + ")"
    return repr(value)


def bytecode_signature(source: str, *, filename: str = "<gate>") -> str:
    """SHA-256 over the normalized instruction stream of every code object.

    Line/column positions are stripped, so comment- and whitespace-only
    refactors keep an identical signature; any change to an operation, name,
    or constant changes it. Signatures are only compared between two sources
    compiled on the same interpreter — they are not stable across Python
    versions, which is exactly the scope the gate compares (one run).
    """
    root = compile(source, filename, "exec", dont_inherit=True)
    digest = hashlib.sha256()

    def walk(code: types.CodeType) -> None:
        digest.update(f"code:{code.co_qualname}\n".encode())
        for instruction in dis.get_instructions(code):
            if instruction.opname == "CACHE":
                continue
            if instruction.opname == "LOAD_CONST":
                rendered = _render_const(instruction.argval)
            else:
                rendered = instruction.argrepr or ""
            digest.update(f"{instruction.opname}={rendered}\n".encode())
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                walk(const)

    walk(root)
    return digest.hexdigest()


def _require_same_cases(
    baseline: Mapping[str, CaseResult], candidate: Mapping[str, CaseResult]
) -> None:
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate cover different case sets")


def differential_flips(
    baseline: Mapping[str, CaseResult], candidate: Mapping[str, CaseResult]
) -> list[CaseResult]:
    """Candidate results whose verdict differs from the baseline on the same case.

    Both maps must cover the identical case universe — a mismatch means an
    evaluator lied about coverage, which the gate treats as a gate error
    (fail-closed).
    """
    _require_same_cases(baseline, candidate)
    return [
        result
        for case_id, result in candidate.items()
        if result.verdict is not baseline[case_id].verdict
    ]


def never_worse_regression(
    baseline: Mapping[str, CaseResult], candidate: Mapping[str, CaseResult]
) -> CaseResult | None:
    """First case the candidate broke (baseline pass → candidate fail), if any.

    Improvements (baseline fail → candidate pass) never count as regressions;
    the rule is exactly "not worse on any evaluated case".
    """
    _require_same_cases(baseline, candidate)
    for case_id, result in baseline.items():
        if (
            result.verdict is CaseVerdict.PASS
            and candidate[case_id].verdict is CaseVerdict.FAIL
        ):
            return candidate[case_id]
    return None
