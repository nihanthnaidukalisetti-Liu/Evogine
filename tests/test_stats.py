"""V2-7 statistics tests: fixed fixtures, deterministic intervals, pairing."""

from __future__ import annotations

import pytest

from evogine.bench.protocol import (
    AnalysisPlan,
    ClaimTag,
    EvalProtocol,
    ReplayBudgets,
)
from evogine.bench.stats import (
    PairedCaseStats,
    cluster_bootstrap,
    cold_rate,
    headline_claims,
    mean_delta,
    pair_verdicts,
    percentile_ci,
    warm_rate,
)


def _pair(
    case_id: str, family: str, cold: tuple[bool, ...], warm: tuple[bool, ...]
) -> PairedCaseStats:
    return pair_verdicts(case_id, family, cold, warm)


def _protocol(**overrides: object) -> EvalProtocol:
    fields: dict[str, object] = {
        "name": "stats fixture",
        "corpus_fingerprint": "sha256:fixture",
        "prompt_pack": "pack",
        "repeats": 4,
        "budgets": ReplayBudgets(
            max_per_proposal=999, max_per_case_family=999, max_per_run=999
        ),
        "analysis": AnalysisPlan(bootstrap_iterations=50, seed=42),
    }
    fields.update(overrides)
    return EvalProtocol.model_validate(fields)


def test_pair_verdicts_refuses_unpaired_arms() -> None:
    with pytest.raises(ValueError, match="identical case instances"):
        pair_verdicts("alpha", "fam-a", (True, False), (True,))


def test_degenerate_outcomes_give_a_point_interval() -> None:
    """Cold always fails, warm always passes: every replicate is the same."""
    pairs = [_pair("alpha", "fam-a", (False, False, False, False), (True,) * 4)]

    assert percentile_ci(
        cluster_bootstrap(pairs, mean_delta, n_boot=25, seed=7), 0.95
    ) == (1.0, 1.0)
    assert percentile_ci(cluster_bootstrap(pairs, cold_rate, n_boot=25, seed=7), 0.95) == (
        0.0,
        0.0,
    )
    assert percentile_ci(cluster_bootstrap(pairs, warm_rate, n_boot=25, seed=7), 0.95) == (
        1.0,
        1.0,
    )


def test_cluster_resampling_moves_whole_families() -> None:
    """Two one-case families: only cluster draws can build each replicate.

    Family A regresses the case (warm worse), family B improves it. Because
    resampling draws FAMILIES, every replicate pools whole families and the
    mean delta can only be -1, 0, or +1 — never a within-family value.
    """
    pairs = [
        _pair("a-case", "fam-a", (True, True, True, True), (False,) * 4),
        _pair("b-case", "fam-b", (False,) * 4, (True, True, True, True)),
    ]

    replicates = cluster_bootstrap(pairs, mean_delta, n_boot=200, seed=11)

    assert set(replicates) <= {-1.0, 0.0, 1.0}
    # With a fixed seed the draw is deterministic: both extremes occur.
    assert -1.0 in replicates and 1.0 in replicates


def test_same_seed_reproduces_the_same_replicates() -> None:
    pairs = [
        _pair("a", "fam-a", (False, True, False, True), (True, True, False, True)),
        _pair("b", "fam-b", (True, True, False, False), (True, False, False, False)),
    ]

    first = cluster_bootstrap(pairs, mean_delta, n_boot=100, seed=2026)
    second = cluster_bootstrap(pairs, mean_delta, n_boot=100, seed=2026)

    assert first == second


def test_interval_brackets_the_point_estimate() -> None:
    pairs = [
        _pair("a", "fam-a", (False, True, False, True), (True, True, False, True)),
        _pair("b", "fam-a", (True, False, False, False), (True, True, True, False)),
        _pair("c", "fam-b", (False, False, True, False), (True, False, True, True)),
    ]

    for pairs_subset in (pairs, pairs[:2], pairs[1:]):
        for statistic in (cold_rate, warm_rate, mean_delta):
            replicates = cluster_bootstrap(
                pairs_subset, statistic, n_boot=200, seed=5
            )
            low, high = percentile_ci(replicates, 0.95)
            point = statistic(pairs_subset)
            assert low <= point <= high


def test_percentile_ci_needs_replicates() -> None:
    with pytest.raises(ValueError, match="empty replicate set"):
        percentile_ci([], 0.95)


def test_run_level_numbers_exclude_the_heldout_family() -> None:
    pairs = [
        _pair("a", "bench", (False,) * 4, (True,) * 4),
        _pair("h", "holdout", (False,) * 4, (True,) * 4),
    ]
    protocol = _protocol(held_out_family="holdout")

    claims = headline_claims(pairs, protocol)

    run_cold = next(c for c in claims if c.metric == "cold_pass_rate" and c.scope == "run")
    holdout_cold = next(
        c for c in claims if c.metric == "cold_pass_rate" and c.scope == "family:holdout"
    )
    # The run-level number covers only benchmark families — the held-out
    # case's identical cold rate must not double-count into it.
    assert run_cold.value == 0.0
    assert run_cold.claim_tag is ClaimTag.CONTAMINABLE_BENCHMARK
    assert holdout_cold.claim_tag is ClaimTag.FRESH_HOLDOUT


def test_every_headline_number_is_tagged_and_intervalled() -> None:
    pairs = [
        _pair("a", "bench", (False, True, False, False), (True, True, False, True)),
        _pair("h", "holdout", (True, False, False, False), (True, True, False, True)),
    ]
    protocol = _protocol(held_out_family="holdout")

    claims = headline_claims(pairs, protocol)

    metrics = {(claim.metric, claim.scope) for claim in claims}
    assert ("cold_pass_rate", "run") in metrics
    assert ("warm_pass_rate", "run") in metrics
    assert ("delta_pass_rate", "run") in metrics
    assert ("never_worse_margin", "run") in metrics
    assert ("never_worse_margin", "family:bench") in metrics
    assert ("never_worse_margin", "family:holdout") in metrics
    assert all(claim.ci_low <= claim.ci_high for claim in claims)
    assert all(claim.claim_tag in ClaimTag for claim in claims)


def test_never_worse_margin_is_the_delta_lower_bound() -> None:
    pairs = [_pair("a", "fam-a", (False, False, True, True), (True, True, True, True))]
    protocol = _protocol()

    claims = headline_claims(pairs, protocol)
    delta = next(c for c in claims if c.metric == "delta_pass_rate")
    margin = next(c for c in claims if c.metric == "never_worse_margin")

    assert margin.value == delta.ci_low
    assert margin.ci_low == delta.ci_low
    assert margin.ci_high == delta.ci_high
    assert margin.never_worse_holds is (delta.ci_low >= 0.0)


def test_never_worse_margin_flags_a_regression() -> None:
    """A family whose warm arm is worse must fail the never-worse check."""
    pairs = [_pair("a", "fam-a", (True,) * 4, (False, False, False, False))]
    protocol = _protocol()

    margin = next(
        c
        for c in headline_claims(pairs, protocol)
        if c.metric == "never_worse_margin" and c.scope == "run"
    )

    assert margin.value < 0.0
    assert margin.never_worse_holds is False


def test_claims_are_reproducible_from_the_protocol_alone() -> None:
    pairs = [
        _pair("a", "bench", (False, True, False, False), (True, True, False, True)),
        _pair("b", "bench", (True, False, False, False), (True, True, True, False)),
    ]
    protocol = _protocol()

    first = headline_claims(pairs, protocol)
    second = headline_claims(pairs, protocol)

    assert [claim.model_dump() for claim in first] == [
        claim.model_dump() for claim in second
    ]


def test_empty_groups_emit_no_claims() -> None:
    pairs = [_pair("a", "fam-a", (False,) * 4, (True,) * 4)]
    protocol = _protocol(held_out_family="nowhere")

    claims = headline_claims(pairs, protocol)

    assert all(claim.scope != "family:nowhere" for claim in claims)
