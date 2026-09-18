"""Materiality rule tests: the corrected V2-2 binary/scored contract.

The suite pins the three properties the audit demanded: binary flips are
counted with no MAD floor (the flaky 2-of-3 regression), scored outcomes
use the paired median against a threshold, and both arms must cover
identical case instances. All offline — pure functions, no models.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from evogine.gates.materiality import (
    MaterialityError,
    Verdict,
    align_verdicts,
    is_binary,
    is_material,
    paired_median_delta,
)


def binary_arm(outcomes: list[bool], prefix: str = "case") -> list[Verdict]:
    """Binary verdicts for instances ``prefix-0..n`` in list order."""
    return [Verdict(instance_id=f"{prefix}-{i}", passed=o) for i, o in enumerate(outcomes)]


def scored_arm(scores: list[float], prefix: str = "case") -> list[Verdict]:
    """Scored verdicts for instances ``prefix-0..n`` in list order."""
    return [Verdict(instance_id=f"{prefix}-{i}", score=s) for i, s in enumerate(scores)]


# --------------------------------------------------------------- binary rule


def test_no_flip_is_not_material() -> None:
    result = is_material(binary_arm([False, False, False]), binary_arm([False, False, False]))
    assert result.material is False
    assert result.detail == "0 binary flips"


def test_any_single_flip_is_material_no_mad_floor_on_bits() -> None:
    """A 1-of-3 minority flip is material: bits get no MAD floor, ever.

    The old rule's majority-vote machinery discarded minority flips; the
    V2-2 rule counts any paired flip.
    """
    result = is_material(binary_arm([False, False, False]), binary_arm([False, True, False]))
    assert result.material is True
    assert result.detail == "1 binary flips"


def test_flaky_two_of_three_flip_is_material_regression() -> None:
    """Regression for audit N3: the flaky 2-of-3 flip case.

    Two of three paired repeats flip. The V2-2 rule is any-flip-is-material
    — no MAD floor on bits may ever return to reject this.
    """
    result = is_material(binary_arm([False, False, False]), binary_arm([True, True, False]))
    assert result.material is True
    assert result.detail == "2 binary flips"


def test_full_flip_is_material() -> None:
    result = is_material(binary_arm([False, False, False]), binary_arm([True, True, True]))
    assert result.material is True
    assert result.detail == "3 binary flips"


def test_reverse_direction_flip_counts_too() -> None:
    """Flip counting is direction-agnostic; never-worse handles regressions."""
    result = is_material(binary_arm([True, True, True]), binary_arm([True, False, True]))
    assert result.material is True
    assert result.detail == "1 binary flips"


# --------------------------------------------------------------- scored rule


def test_scored_above_threshold_is_material() -> None:
    result = is_material(scored_arm([0.5, 0.5, 0.5]), scored_arm([0.9, 0.9, 0.9]))
    assert result.material is True
    assert result.detail == "scored delta 0.400 vs threshold 0.15"


def test_scored_below_threshold_is_not_material() -> None:
    result = is_material(scored_arm([0.5, 0.5, 0.5]), scored_arm([0.55, 0.55, 0.55]))
    assert result.material is False
    assert result.detail == "scored delta 0.050 vs threshold 0.15"


def test_scored_at_exact_threshold_is_not_material() -> None:
    """Strictly-greater rule: delta equal to the threshold does not qualify.

    0.5 and 0.25 are exact binary floats, so the boundary is exact here.
    """
    result = is_material(scored_arm([1.0, 1.0]), scored_arm([1.0, 1.5]), scored_threshold=0.25)
    assert result.material is False


def test_scored_negative_delta_is_not_material() -> None:
    result = is_material(scored_arm([0.8, 0.8]), scored_arm([0.4, 0.4]))
    assert result.material is False


def test_paired_median_not_mean() -> None:
    """The median, not the mean: one outlier delta cannot buy materiality.

    Deltas [0.0, 0.0, 1.0] have median 0.0 (not material) but mean 0.333
    (which would qualify) — this pins the median choice.
    """
    baseline = scored_arm([1.0, 1.0, 1.0])
    warm = scored_arm([1.0, 1.0, 2.0])
    assert paired_median_delta(baseline, warm) == 0.0
    assert is_material(baseline, warm).material is False


def test_paired_median_even_count_interpolates() -> None:
    """Even counts: the median interpolates — deltas [0.0, 0.5] -> 0.25."""
    baseline = scored_arm([1.0, 1.0])
    warm = scored_arm([1.0, 1.5])
    assert paired_median_delta(baseline, warm) == 0.25


# ------------------------------------------------- paired-instance assertion


def test_arms_missing_an_instance_are_rejected() -> None:
    baseline = binary_arm([False, False, False])
    warm = binary_arm([True, False], prefix="case")
    with pytest.raises(MaterialityError, match="different case instances"):
        is_material(baseline, warm)


def test_arms_with_an_extra_instance_are_rejected() -> None:
    baseline = binary_arm([False, False])
    warm = binary_arm([True, False, True])
    with pytest.raises(MaterialityError, match="different case instances"):
        is_material(baseline, warm)


def test_duplicate_instance_within_an_arm_is_rejected() -> None:
    baseline = binary_arm([False, False])
    warm = [
        Verdict(instance_id="case-0", passed=True),
        Verdict(instance_id="case-0", passed=False),
    ]
    with pytest.raises(MaterialityError, match="duplicate case instance"):
        align_verdicts(baseline, warm)


def test_empty_arm_is_rejected() -> None:
    with pytest.raises(MaterialityError, match="at least one case instance"):
        is_material([], binary_arm([True]))


def test_mixed_arms_are_rejected() -> None:
    with pytest.raises(MaterialityError, match="mixed binary/scored"):
        is_material(binary_arm([False]), scored_arm([0.5]))


def test_mixed_outcomes_within_one_arm_are_rejected() -> None:
    baseline = binary_arm([False, False])
    warm = [Verdict(instance_id="case-0", score=0.5), Verdict(instance_id="case-1", passed=True)]
    with pytest.raises(MaterialityError, match="mixed binary/scored"):
        is_material(baseline, warm)


def test_verdict_requires_exactly_one_outcome() -> None:
    with pytest.raises(ValidationError, match="exactly one of passed/score"):
        Verdict(instance_id="case-0")
    with pytest.raises(ValidationError, match="exactly one of passed/score"):
        Verdict(instance_id="case-0", passed=True, score=0.5)


# --------------------------------------------------------- permutation property


@given(
    st.data(),
    st.lists(st.booleans(), min_size=1, max_size=16),
    st.randoms(),
)
def test_binary_materiality_is_permutation_invariant(data, outcomes, rnd) -> None:
    """Property: binary materiality is invariant to instance order.

    Applying the same permutation to both paired arms must not change the
    verdict — the rule pairs by case instance, so ordering is never a
    hidden input to the decision.
    """
    warm_outcomes = data.draw(st.lists(st.booleans(), min_size=len(outcomes), max_size=len(outcomes)))
    baseline_verdicts = binary_arm(outcomes)
    warm_verdicts = binary_arm(warm_outcomes)

    indices = list(range(len(outcomes)))
    rnd.shuffle(indices)
    baseline_permuted = [baseline_verdicts[i] for i in indices]
    warm_permuted = [warm_verdicts[i] for i in indices]

    assert is_material(baseline_permuted, warm_permuted) == is_material(
        baseline_verdicts, warm_verdicts
    )


def test_scored_materiality_is_permutation_invariant() -> None:
    """Same property for the scored arm: median deltas ignore order."""
    baseline = scored_arm([1.0, 0.9, 1.1, 0.8])
    warm = scored_arm([1.2, 1.0, 1.3, 1.4])

    order = [3, 0, 2, 1]
    baseline_permuted = [baseline[i] for i in order]
    warm_permuted = [warm[i] for i in order]

    assert is_material(baseline_permuted, warm_permuted) == is_material(baseline, warm)


def test_is_binary_distinguishes_arm_kinds() -> None:
    assert is_binary(binary_arm([True, False]))
    assert not is_binary(scored_arm([0.5, 0.7]))
