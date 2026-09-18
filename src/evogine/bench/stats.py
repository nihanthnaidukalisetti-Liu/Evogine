"""V2-7: cluster-bootstrap confidence intervals for bench headline numbers.

A point estimate is not a claim. Every headline number a bench run reports —
cold and warm pass rates, the warm-vs-cold delta, and the never-worse
margins — ships with a 95% bootstrap confidence interval, resampled by case
FAMILY (the cluster unit): whole families are drawn with replacement, so the
interval reflects between-family variance instead of pretending cases are
independent. The resampling seed is part of the pre-registered analysis plan
(:class:`evogine.bench.protocol.AnalysisPlan`), so the same protocol
reproduces the same intervals bit-for-bit.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from evogine.bench.protocol import ClaimTag, EvalProtocol


@dataclass(frozen=True, slots=True)
class PairedCaseStats:
    """One case's verdicts in both arms, paired per repeat index.

    Warm-vs-cold runs on identical case instances, so verdict ``i`` of each
    arm is the same instance under the same repeat — pairing is the
    analysis-side guarantee that the comparison is like-for-like.
    """

    case_id: str
    family: str
    cold: tuple[bool, ...]
    warm: tuple[bool, ...]


def pair_verdicts(
    case_id: str,
    family: str,
    cold: Sequence[bool],
    warm: Sequence[bool],
) -> PairedCaseStats:
    """Pair cold and warm verdicts by repeat index; refuse an unpaired pair.

    Both arms must cover identical case instances — a length mismatch means
    the arms ran different instance sets and no paired comparison is valid.
    """
    if len(cold) != len(warm):
        raise ValueError(
            f"unpaired outcomes for case {case_id!r}: {len(cold)} cold vs "
            f"{len(warm)} warm — both arms must cover identical case instances"
        )
    return PairedCaseStats(case_id, family, tuple(cold), tuple(warm))


class HeadlineClaim(BaseModel):
    """One reported number with its interval and its evidence tag."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    """``cold_pass_rate`` | ``warm_pass_rate`` | ``delta_pass_rate`` |
    ``never_worse_margin``."""

    scope: str
    """``"run"`` for the run-level number, ``"family:<name>"`` otherwise."""

    value: float
    ci_low: float
    ci_high: float
    claim_tag: ClaimTag
    never_worse_holds: bool | None = None
    """Delta metrics only: the lower interval bound clears zero."""


def _pass_rate(verdicts: Sequence[bool]) -> float:
    if not verdicts:
        raise ValueError("pass rate of an empty verdict set is undefined")
    return sum(verdicts) / len(verdicts)


def cold_rate(pairs: Sequence[PairedCaseStats]) -> float:
    """Mean per-case cold pass rate over the given cases."""
    return _pass_rate_of_mean(pairs, "cold")


def warm_rate(pairs: Sequence[PairedCaseStats]) -> float:
    """Mean per-case warm pass rate over the given cases."""
    return _pass_rate_of_mean(pairs, "warm")


def mean_delta(pairs: Sequence[PairedCaseStats]) -> float:
    """Mean per-case warm-minus-cold delta over the given cases."""
    if not pairs:
        raise ValueError("delta over an empty case set is undefined")
    return sum(
        _pass_rate(pair.warm) - _pass_rate(pair.cold) for pair in pairs
    ) / len(pairs)


def _pass_rate_of_mean(pairs: Sequence[PairedCaseStats], arm: str) -> float:
    if not pairs:
        raise ValueError(f"{arm} pass rate over an empty case set is undefined")
    rates = [_pass_rate(pair.cold if arm == "cold" else pair.warm) for pair in pairs]
    return sum(rates) / len(rates)


def cluster_bootstrap(
    pairs: Sequence[PairedCaseStats],
    statistic: Callable[[Sequence[PairedCaseStats]], float],
    *,
    n_boot: int,
    seed: int,
) -> list[float]:
    """Resample whole case families with replacement; return the statistic per replicate.

    The resampling unit is the family (the cluster), never the individual
    case: each replicate draws as many families as exist, pooling every case
    of each drawn family. Same seed ⇒ same replicates, always.
    """
    if not pairs:
        raise ValueError("cluster bootstrap needs at least one case")
    families = sorted({pair.family for pair in pairs})
    grouped: dict[str, list[PairedCaseStats]] = {family: [] for family in families}
    for pair in pairs:
        grouped[pair.family].append(pair)

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_boot):
        pool: list[PairedCaseStats] = []
        for _ in families:
            pool.extend(grouped[families[rng.randrange(len(families))]])
        replicates.append(statistic(pool))
    return replicates


def percentile_ci(
    replicates: Sequence[float], confidence: float
) -> tuple[float, float]:
    """Percentile interval of the replicate distribution.

    Integer-index quantiles (no interpolation): deterministic on any Python,
    and exact on a degenerate distribution (low == high == the value).
    """
    if not replicates:
        raise ValueError("interval of an empty replicate set is undefined")
    ordered = sorted(replicates)
    n = len(ordered)
    alpha = 1.0 - confidence
    low = ordered[int((n - 1) * (alpha / 2))]
    high = ordered[int((n - 1) * (1 - alpha / 2))]
    return low, high


def headline_claims(
    pairs: Sequence[PairedCaseStats], protocol: EvalProtocol
) -> list[HeadlineClaim]:
    """Tag and interval every headline number the run will report.

    Run-level numbers cover the benchmark families only (the held-out family
    reports separately, tagged ``fresh-holdout``); every benchmark family
    also gets its own delta and never-worse margin, so a run-level win
    cannot hide a family-level regression.
    """
    held = protocol.held_out_family
    benchmark = [pair for pair in pairs if pair.family != held] if held else list(pairs)
    heldout = (
        [pair for pair in pairs if pair.family == held] if held is not None else []
    )

    claims = _group_claims(benchmark, "run", protocol.benchmark_claim_tag, protocol)
    for family in sorted({pair.family for pair in benchmark}):
        members = [pair for pair in benchmark if pair.family == family]
        claims += _group_claims(
            members, f"family:{family}", protocol.benchmark_claim_tag, protocol
        )
    claims += _group_claims(heldout, f"family:{held}", ClaimTag.FRESH_HOLDOUT, protocol)
    return claims


def _group_claims(
    pairs: Sequence[PairedCaseStats],
    scope: str,
    tag: ClaimTag,
    protocol: EvalProtocol,
) -> list[HeadlineClaim]:
    """Interval four headline metrics over one group of cases."""
    if not pairs:
        return []
    analysis = protocol.analysis
    n_boot = analysis.bootstrap_iterations
    confidence = analysis.confidence
    seed = analysis.seed

    def interval(
        statistic: Callable[[Sequence[PairedCaseStats]], float],
    ) -> tuple[float, float]:
        replicates = cluster_bootstrap(
            pairs, statistic, n_boot=n_boot, seed=seed
        )
        return percentile_ci(replicates, confidence)

    cold_low, cold_high = interval(cold_rate)
    warm_low, warm_high = interval(warm_rate)
    delta_low, delta_high = interval(mean_delta)
    return [
        HeadlineClaim(
            metric="cold_pass_rate",
            scope=scope,
            value=cold_rate(pairs),
            ci_low=cold_low,
            ci_high=cold_high,
            claim_tag=tag,
        ),
        HeadlineClaim(
            metric="warm_pass_rate",
            scope=scope,
            value=warm_rate(pairs),
            ci_low=warm_low,
            ci_high=warm_high,
            claim_tag=tag,
        ),
        HeadlineClaim(
            metric="delta_pass_rate",
            scope=scope,
            value=mean_delta(pairs),
            ci_low=delta_low,
            ci_high=delta_high,
            claim_tag=tag,
            never_worse_holds=delta_low >= 0.0,
        ),
        HeadlineClaim(
            # The never-worse margin IS the delta's lower bound: how far
            # below zero the warm-minus-cold effect could plausibly reach.
            metric="never_worse_margin",
            scope=scope,
            value=delta_low,
            ci_low=delta_low,
            ci_high=delta_high,
            claim_tag=tag,
            never_worse_holds=delta_low >= 0.0,
        ),
    ]


__all__ = [
    "HeadlineClaim",
    "PairedCaseStats",
    "cluster_bootstrap",
    "cold_rate",
    "headline_claims",
    "mean_delta",
    "pair_verdicts",
    "percentile_ci",
    "warm_rate",
]
