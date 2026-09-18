"""Counterfactual-replay materiality: the corrected V2-2 rule.

V1's ``flip_is_material`` applied a MAD floor to binary replay outcomes,
which degenerates to majority vote: a flaky 2-of-3 flip could "qualify"
through score-shaped machinery that has no meaning for bits (candidates
audit, N3). V2-2 replaces it with one function both outcome kinds share,
so the rules cannot drift apart:

- binary verdicts are material iff any paired repeat flips — no MAD floor
  on bits, ever (regression-tested, including the flaky 2-of-3 case);
- scored verdicts are material iff the paired median delta exceeds the
  configured threshold;
- both arms must cover identical case instances — replay is paired per
  case instance, so variance shows up as repeats, never as unpaired drift.
"""

from __future__ import annotations

from statistics import median

from pydantic import BaseModel, Field, model_validator


class MaterialityError(ValueError):
    """Replay verdict arms cannot be paired or compared."""


class Verdict(BaseModel):
    """Outcome of replaying one case instance in one arm.

    Exactly one of ``passed`` (binary) or ``score`` (scored, higher is
    better) is set. ``instance_id`` names the case instance (case plus
    repeat), so paired arms align by identity, never by list position.
    """

    instance_id: str = Field(min_length=1)
    passed: bool | None = None
    score: float | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> Verdict:
        if (self.passed is None) == (self.score is None):
            raise ValueError(f"verdict {self.instance_id!r} needs exactly one of passed/score")
        return self


class MaterialityVerdict(BaseModel):
    """The materiality decision plus the evidence behind it."""

    material: bool
    detail: str


def is_binary(verdicts: list[Verdict]) -> bool:
    """Return ``True`` iff every verdict in the arm carries a binary outcome."""
    return all(verdict.passed is not None for verdict in verdicts)


def align_verdicts(
    baseline: list[Verdict], warm: list[Verdict]
) -> list[tuple[Verdict, Verdict]]:
    """Pair the two arms by case instance, sorted by ``instance_id``.

    Replay is paired per case instance, so alignment is by identity, never
    position — a flaky difference in ordering must not change what pairs
    with what. Raises :class:`MaterialityError` when an arm is empty,
    repeats an instance, or the arms cover different case instances: the
    two arms must describe exactly the same case set.
    """
    if not baseline or not warm:
        raise MaterialityError("both replay arms must cover at least one case instance")
    baseline_by_id = _by_instance(baseline)
    warm_by_id = _by_instance(warm)
    if baseline_by_id.keys() != warm_by_id.keys():
        baseline_only = sorted(baseline_by_id.keys() - warm_by_id.keys())
        warm_only = sorted(warm_by_id.keys() - baseline_by_id.keys())
        raise MaterialityError(
            "replay arms cover different case instances: "
            f"baseline-only={baseline_only}, warm-only={warm_only}"
        )
    return [(baseline_by_id[i], warm_by_id[i]) for i in sorted(baseline_by_id)]


def _by_instance(verdicts: list[Verdict]) -> dict[str, Verdict]:
    by_id: dict[str, Verdict] = {}
    for verdict in verdicts:
        if verdict.instance_id in by_id:
            raise MaterialityError(
                f"duplicate case instance in one arm: {verdict.instance_id!r}"
            )
        by_id[verdict.instance_id] = verdict
    return by_id


def paired_median_delta(baseline: list[Verdict], warm: list[Verdict]) -> float:
    """Median of per-instance warm−baseline score deltas (paired).

    The median, not the mean: a single large delta must not buy materiality
    that the typical case does not support.
    """
    deltas: list[float] = []
    for baseline_v, warm_v in align_verdicts(baseline, warm):
        if baseline_v.score is None or warm_v.score is None:
            raise MaterialityError(
                "scored deltas require a score on both arms for instance "
                f"{baseline_v.instance_id!r}"
            )
        deltas.append(warm_v.score - baseline_v.score)
    return median(deltas)


def is_material(
    baseline_verdicts: list[Verdict],
    warm_verdicts: list[Verdict],
    scored_threshold: float = 0.15,
) -> MaterialityVerdict:
    """Counterfactual replay materiality, consistent by construction.

    Binary verdicts: material iff any paired repeat changes outcome
    (flip/no-flip) — a flaky 2-of-3 flip IS material noise only if it
    doesn't reproduce; replays are paired per case instance, so variance
    shows up as repeats. Scored verdicts: material iff the paired median
    delta exceeds ``scored_threshold``. Both arms must cover identical
    case instances.
    """
    pairs = align_verdicts(baseline_verdicts, warm_verdicts)
    baseline_binary = is_binary(baseline_verdicts)
    warm_binary = is_binary(warm_verdicts)
    if baseline_binary != warm_binary:
        raise MaterialityError("mixed binary/scored replay arms cannot be compared")
    if baseline_binary:
        flips = sum(baseline_v.passed != warm_v.passed for baseline_v, warm_v in pairs)
        return MaterialityVerdict(material=flips > 0, detail=f"{flips} binary flips")
    if any(v.score is None for v in baseline_verdicts + warm_verdicts):
        raise MaterialityError(
            "scored arms must carry a score on every verdict (arms are mixed binary/scored)"
        )
    delta = paired_median_delta(baseline_verdicts, warm_verdicts)
    return MaterialityVerdict(
        material=delta > scored_threshold,
        detail=f"scored delta {delta:.3f} vs threshold {scored_threshold}",
    )
