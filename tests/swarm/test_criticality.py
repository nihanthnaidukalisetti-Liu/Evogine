"""Unit tests for pure criticality scoring (no disk, no git, no clock)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evogine.swarm.decomposition import (
    DEFAULT_WEIGHTS,
    ModuleEvidence,
    NormalizedSignals,
    ScoreWeights,
    normalize_evidence,
    score_all,
    score_signals,
)


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        ScoreWeights(churn=0.5, failure=0.5, coverage=0.5, centrality=0.5)


def test_default_weights_sum_to_one():
    assert (
        DEFAULT_WEIGHTS.churn
        + DEFAULT_WEIGHTS.failure
        + DEFAULT_WEIGHTS.centrality
        + DEFAULT_WEIGHTS.coverage
    ) == pytest.approx(1.0)


def test_all_zero_evidence_scores_zero():
    scores = score_all([ModuleEvidence(), ModuleEvidence(), ModuleEvidence()])

    assert scores == [0.0, 0.0, 0.0]


def test_max_normalization_is_relative_to_siblings():
    evidence = [
        ModuleEvidence(churn_commits=10, imported_by_count=5),
        ModuleEvidence(churn_commits=5, imported_by_count=0),
        ModuleEvidence(churn_commits=0, imported_by_count=0),
    ]
    signals = normalize_evidence(evidence)

    assert signals[0].churn == 1.0
    assert signals[0].centrality == 1.0
    assert signals[1].churn == pytest.approx(0.5)
    assert signals[1].centrality == 0.0
    assert signals[2] == NormalizedSignals(
        churn=0.0, failure=0.0, coverage=0.0, centrality=0.0
    )


def test_top_module_scores_one_and_every_score_is_in_the_unit_interval():
    evidence = [
        ModuleEvidence(
            churn_commits=4, failure_commits=1, coverage_gap=100.0, imported_by_count=3
        ),
        ModuleEvidence(churn_commits=1, imported_by_count=0),
    ]
    scores = score_all(evidence)

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert max(scores) == pytest.approx(1.0)


def test_missing_coverage_gap_contributes_zero():
    covered = ModuleEvidence(coverage_gap=100.0)
    unknown = ModuleEvidence(coverage_gap=None)

    signals = normalize_evidence([covered, unknown])

    assert signals[0].coverage == 1.0
    assert signals[1].coverage == 0.0  # honest zero: no evidence, no invented gap


def test_empty_module_list_scores_to_an_empty_list():
    assert score_all([]) == []


def test_scores_stay_bounded_when_a_signal_overshoots_its_cap():
    evidence = [ModuleEvidence(churn_commits=10_000), ModuleEvidence(churn_commits=1)]
    scores = score_all(evidence)

    # Only the churn signal is active, and it normalizes to exactly 1.0 for
    # the max-holder — so the top score is the churn weight itself, and the
    # tiny-history module scores a small but positive fraction of it.
    assert scores[0] == pytest.approx(0.35)
    assert 0.0 <= scores[1] < 0.001  # tiny but never negative or above one


def test_score_precision_is_stable_under_float_noise():
    signals = NormalizedSignals(churn=0.1, failure=0.2, coverage=0.3, centrality=0.4)
    weights = ScoreWeights(churn=0.25, failure=0.25, coverage=0.25, centrality=0.25)

    expected = round(0.25 * (0.1 + 0.2 + 0.3 + 0.4), 6)
    assert score_signals(signals, weights) == expected
