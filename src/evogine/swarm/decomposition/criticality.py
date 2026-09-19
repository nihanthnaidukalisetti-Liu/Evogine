"""Pure, deterministic criticality scoring.

The score turns per-module evidence — churn, failure density, coverage gap,
import centrality — into a relative priority in ``[0.0, 1.0]``. Two rules
keep it honest:

- **Pure**: :func:`score_all` touches nothing but its arguments. No clock,
  no randomness, no I/O — the same evidence in the same weights gives the
  same scores out, so property tests can pin determinism exactly.
- **Relative by max**: each signal is normalized by the maximum across the
  repository (a module's churn means nothing absolutely; it means something
  next to its siblings). An all-zero signal — no git evidence, no coverage
  file — normalizes to ``0.0``, never to a fabricated middle value.

Missing evidence is the honest zero, so a repository decomposes identically
whether or not ``coverage.json`` exists; the gap component simply
contributes nothing. Weights are fixed defaults on :class:`ScoreWeights` —
overridable for experiments, but one set per run, chosen before scoring.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCORE_PRECISION = 6
"""Decimals kept on a criticality score: kills float repr noise in the JSON
feed without ever changing a ranking (rank gaps exceed 1e-6 by construction)."""


class ScoreWeights(BaseModel):
    """The four component weights; they must sum to exactly 1.0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    churn: float = Field(default=0.35, ge=0.0, le=1.0)
    failure: float = Field(default=0.15, ge=0.0, le=1.0)
    coverage: float = Field(default=0.20, ge=0.0, le=1.0)
    centrality: float = Field(default=0.30, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> ScoreWeights:
        total = self.churn + self.failure + self.coverage + self.centrality
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"score weights must sum to 1.0 (got {total})")
        return self


DEFAULT_WEIGHTS = ScoreWeights()


class ModuleEvidence(BaseModel):
    """Raw per-module signals, exactly as collected — counts, not scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    churn_commits: int = Field(default=0, ge=0)
    failure_commits: int = Field(default=0, ge=0)
    coverage_gap: float | None = Field(default=None, ge=0.0, le=100.0)
    imported_by_count: int = Field(default=0, ge=0)


class NormalizedSignals(BaseModel):
    """Per-module signals after max-normalization, each in ``[0.0, 1.0]``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    churn: float = Field(ge=0.0, le=1.0)
    failure: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    centrality: float = Field(ge=0.0, le=1.0)


def normalize_evidence(evidence: list[ModuleEvidence]) -> list[NormalizedSignals]:
    """Normalize each signal by its repository-wide maximum.

    Order-transparent: element ``i`` of the output scores element ``i`` of
    the input, and the maxima are order-independent, so permuting the input
    permutes — but never changes — the signals.
    """
    if not evidence:
        return []
    max_churn = max(item.churn_commits for item in evidence)
    max_failure = max(item.failure_commits for item in evidence)
    max_importers = max(item.imported_by_count for item in evidence)
    normalized: list[NormalizedSignals] = []
    for item in evidence:
        churn = item.churn_commits / max_churn if max_churn else 0.0
        failure = item.failure_commits / max_failure if max_failure else 0.0
        # No coverage.json (or no entry for this module) is the honest zero:
        # the coverage component contributes nothing instead of guessing.
        coverage = item.coverage_gap / 100.0 if item.coverage_gap is not None else 0.0
        centrality = item.imported_by_count / max_importers if max_importers else 0.0
        normalized.append(
            NormalizedSignals(
                churn=churn, failure=failure, coverage=coverage, centrality=centrality
            )
        )
    return normalized


def score_signals(signals: NormalizedSignals, weights: ScoreWeights) -> float:
    """Weighted component sum, rounded to :data:`SCORE_PRECISION` decimals."""
    raw = (
        weights.churn * signals.churn
        + weights.failure * signals.failure
        + weights.coverage * signals.coverage
        + weights.centrality * signals.centrality
    )
    # The weighted sum of components in [0, 1] cannot exceed 1, but float
    # addition can land a hair above; clamp before rounding.
    return round(min(raw, 1.0), SCORE_PRECISION)


def score_all(
    evidence: list[ModuleEvidence],
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> list[float]:
    """Score every module's evidence, in input order."""
    return [score_signals(signals, weights) for signals in normalize_evidence(evidence)]
