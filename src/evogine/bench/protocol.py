"""V2-7: the pre-registered evaluation protocol, validated before a run starts.

A compounding claim is only as good as the protocol it was registered under.
An :class:`EvalProtocol` pins the full run configuration in writing — fixed
harness config (repeats), replay budgets, the prompt pack, the case corpus
(by fingerprint), the locked held-out family, and the analysis plan — so the
analysis is fixed *before* the first replay is spent, never chosen after
seeing the outcomes.

``pre_run_validate`` is the gate: it refuses a protocol whose corpus was
edited after registration, whose case selection or held-out family does not
resolve, or whose schedule exceeds the declared budgets. Validation is
all-or-nothing and reports every violation at once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.bench.budgets import BudgetExceededError, ReplayBudgets, queue_replays
from evogine.bench.cases import BenchCase
from evogine.bench.replay import REPEAT_FLOOR, build_replay_requests


class ClaimTag(str, Enum):
    """What kind of evidence surface a reported number came from.

    Every reported number carries exactly one tag: ``contaminable-benchmark``
    for the in-corpus benchmark families (the engine may have seen their
    content during development), ``fresh-holdout`` for the locked held-out
    family (never retrieved against during development), and
    ``in-production`` for numbers measured on live production traffic —
    which a bench run can never produce, so the bench report never emits it.
    """

    CONTAMINABLE_BENCHMARK = "contaminable-benchmark"
    FRESH_HOLDOUT = "fresh-holdout"
    IN_PRODUCTION = "in-production"


class AnalysisPlan(BaseModel):
    """The pre-registered analysis: confidence level, bootstrap size, seed.

    The seed is part of the plan so the bootstrap resampling — and therefore
    every reported interval — is bit-for-bit reproducible from the protocol
    file alone.
    """

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    """The confidence level of every emitted interval (95% by default)."""

    bootstrap_iterations: int = Field(default=2000, ge=1)
    seed: int = Field(default=20260918)
    """Fixed resampling seed; same protocol ⇒ same intervals, always."""


class EvalProtocol(BaseModel):
    """The pre-registered run protocol: pinned before the first replay.

    Every field is checked by :func:`pre_run_validate` before a run starts;
    explicit run arguments that contradict the protocol are refused rather
    than silently overriding it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    """``fingerprint_corpus`` of the exact corpus the run must use — a corpus
    edited after registration fails validation."""

    prompt_pack: str = Field(min_length=1)
    """Opaque id/hash of the frozen prompt pack; prompts are pinned pre-run."""

    repeats: int = Field(ge=REPEAT_FLOOR)
    budgets: ReplayBudgets
    case_ids: list[str] | None = None
    """Omitted means the full pinned corpus; a subset must name existing ids."""

    held_out_family: str | None = None
    """The locked family the engine never retrieves against during
    development. Its cases still run warm-vs-cold; their numbers are tagged
    ``fresh-holdout`` and a recall probe refuses the run if the store
    quotes held-out content."""

    benchmark_claim_tag: ClaimTag = ClaimTag.CONTAMINABLE_BENCHMARK
    """Tag for the benchmark-family headline numbers. ``fresh-holdout`` is
    reserved for the derived held-out numbers and is refused here."""

    analysis: AnalysisPlan = Field(default_factory=AnalysisPlan)

    @model_validator(mode="after")
    def _claim_tag_discipline(self) -> EvalProtocol:
        if self.benchmark_claim_tag is ClaimTag.FRESH_HOLDOUT:
            raise ValueError(
                "benchmark_claim_tag='fresh-holdout' is reserved for the locked "
                "held-out family; benchmark numbers cannot self-certify as fresh"
            )
        return self


class ProtocolValidationError(Exception):
    """Pre-run refusal: the protocol does not match the world it describes."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            "protocol failed pre-run validation (nothing was spent):\n- "
            + "\n- ".join(violations)
        )


@dataclass(frozen=True, slots=True)
class ProtocolCheck:
    """The validated run view: resolved cases and the schedule they imply."""

    protocol: EvalProtocol
    cases: tuple[BenchCase, ...]
    heldout_cases: tuple[BenchCase, ...]
    total_replays: int


def fingerprint_corpus(corpus: Sequence[BenchCase]) -> str:
    """Return the stable fingerprint of a corpus: id, family, task context.

    Order-independent (ids sort) and content-sensitive (a one-word edit of a
    task context changes the hash), so a protocol pins the corpus, not a
    file path that can drift under it.
    """
    lines = sorted(
        f"{case.id}\x1f{case.family}\x1f{case.task_context}" for case in corpus
    )
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def pre_run_validate(
    protocol: EvalProtocol, corpus: Sequence[BenchCase]
) -> ProtocolCheck:
    """Validate the protocol against the corpus BEFORE any replay is spent.

    Collects every violation rather than failing on the first: a rejected
    protocol should name everything wrong with it in one pass. Reuses the
    V2-4 queue-time budget check, so an exceeding schedule is refused here
    for the same reason it is refused at queue time.
    """
    violations: list[str] = []

    by_id = {case.id: case for case in corpus}
    if len(by_id) != len(corpus):
        violations.append("duplicate case ids in corpus")

    expected = fingerprint_corpus(list(corpus))
    if protocol.corpus_fingerprint != expected:
        violations.append(
            f"corpus fingerprint mismatch: protocol pins "
            f"{protocol.corpus_fingerprint!r}, corpus hashes to {expected!r} — "
            "the corpus changed after registration"
        )

    resolved: list[BenchCase] = []
    resolvable = not violations
    if protocol.case_ids is not None:
        duplicates = sorted({i for i in protocol.case_ids if protocol.case_ids.count(i) > 1})
        if duplicates:
            violations.append(f"duplicate case ids in selection: {duplicates}")
        unknown = sorted(set(protocol.case_ids) - set(by_id))
        if unknown:
            violations.append(f"unknown case ids {unknown}; corpus has {sorted(by_id)}")
        resolvable = resolvable and not duplicates and not unknown
    if resolvable:
        resolved = [by_id[i] for i in protocol.case_ids] if protocol.case_ids else list(corpus)

    heldout: tuple[BenchCase, ...] = ()
    if protocol.held_out_family is not None:
        families = {case.family for case in corpus}
        if protocol.held_out_family not in families:
            violations.append(
                f"held-out family {protocol.held_out_family!r} does not exist in the "
                f"corpus; families present: {sorted(families)}"
            )
        heldout = tuple(case for case in resolved if case.family == protocol.held_out_family)
        if protocol.held_out_family in families and not heldout:
            violations.append(
                f"held-out family {protocol.held_out_family!r} has no resolved cases — "
                "a declared holdout that never runs cannot back a fresh-holdout claim"
            )

    if resolvable:
        try:
            schedule_requests = build_replay_requests(resolved, protocol.repeats)
            queue_replays(schedule_requests, protocol.budgets)
        except BudgetExceededError as exc:
            violations.append(f"declared budgets refuse the resolved schedule: {exc}")

    if violations:
        raise ProtocolValidationError(violations)

    return ProtocolCheck(
        protocol=protocol,
        cases=tuple(resolved),
        heldout_cases=heldout,
        total_replays=len(resolved) * 2 * protocol.repeats,
    )


def load_protocol(path: str | Path) -> EvalProtocol:
    """Parse and validate a pre-registered protocol from a YAML file.

    Unknown fields are refused (config drift), the repeat count must meet
    the 4-repeat floor, and the claim-tag discipline applies at parse time.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvalProtocol.model_validate(data)


__all__ = [
    "AnalysisPlan",
    "ClaimTag",
    "EvalProtocol",
    "ProtocolCheck",
    "ProtocolValidationError",
    "fingerprint_corpus",
    "load_protocol",
    "pre_run_validate",
]
