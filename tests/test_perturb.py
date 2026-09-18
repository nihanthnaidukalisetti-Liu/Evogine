"""V2-7 perturbation tests: deterministic rename/paraphrase variants."""

from __future__ import annotations

from evogine.bench.cases import entries
from evogine.bench.perturb import (
    generate_perturbations,
    paraphrase_variant,
    rename_variant,
)

_PROTECTED_RETRY = frozenset(
    {"backoff", "jitter", "exponential", "idempotent", "idempotency", "cap", "retry"}
)


def _retry_entry():
    return next(e for e in entries() if e.case.id == "retry-jitter-budget")


def test_rename_variant_is_deterministic() -> None:
    case = _retry_entry().case

    first = rename_variant(case, variant_index=0, seed=7, protected=_PROTECTED_RETRY)
    second = rename_variant(case, variant_index=0, seed=7, protected=_PROTECTED_RETRY)

    assert first.task_context == second.task_context
    assert first.id == second.id == "retry-jitter-budget::rename-0"


def test_rename_variant_preserves_family_and_oracle() -> None:
    entry = _retry_entry()

    variant = rename_variant(
        entry.case, variant_index=0, seed=7, protected=_PROTECTED_RETRY
    )

    assert variant.family == entry.case.family
    assert variant.hidden_invariant is entry.case.hidden_invariant


def test_rename_variant_protects_the_gold_vocabulary() -> None:
    entry = _retry_entry()
    context_tokens = set(entry.case.task_context.lower().split())

    variant = rename_variant(
        entry.case, variant_index=0, seed=7, protected=_PROTECTED_RETRY
    )

    renamed = variant.task_context.lower()
    for token in context_tokens:
        # Any protected word that appeared in the prompt must survive verbatim.
        cleaned = token.strip(".,:;")
        if cleaned in _PROTECTED_RETRY and cleaned in context_tokens:
            assert cleaned in renamed


def test_rename_variant_actually_rewords_the_prompt() -> None:
    entry = _retry_entry()

    variant = rename_variant(
        entry.case, variant_index=0, seed=7, protected=_PROTECTED_RETRY
    )

    assert variant.task_context != entry.case.task_context


def test_substitution_is_single_pass() -> None:
    """An alias must never be renamed again by a later pass."""
    from evogine.bench.perturb import _substitute

    swapped = _substitute("pipeline", {"pipeline": "processor", "processor": "broker"})
    assert swapped == "processor"  # a second pass would have renamed it to "broker"


def test_paraphrase_variant_is_deterministic_and_cycling() -> None:
    case = _retry_entry().case

    first = paraphrase_variant(case, variant_index=0)
    again = paraphrase_variant(case, variant_index=0)
    cycled = paraphrase_variant(case, variant_index=3)

    assert first.task_context == again.task_context
    assert first.id == "retry-jitter-budget::paraphrase-0"
    assert cycled.task_context == first.task_context
    assert cycled.id == "retry-jitter-budget::paraphrase-3"


def test_paraphrase_keeps_the_oracle_green() -> None:
    """The substitution tables must not break grading on any corpus entry."""
    for entry in entries():
        for index in range(3):
            variant = paraphrase_variant(entry.case, variant_index=index)
            assert variant.hidden_invariant(entry.good_output), (
                f"good_output broke on {variant.id}"
            )
            assert not variant.hidden_invariant(entry.bad_output), (
                f"bad_output sneaked through {variant.id}"
            )


def test_rename_keeps_the_oracle_green_with_protection() -> None:
    for entry in entries():
        protected = frozenset(
            {
                token
                for token in entry.good_output.lower().split()
                if token.strip(".,:;")
            }
        ) | frozenset(
            {token for token in entry.bad_output.lower().split() if token.strip(".,:;")}
        )
        variant = rename_variant(entry.case, variant_index=0, seed=7, protected=protected)
        assert variant.hidden_invariant(entry.good_output), (
            f"good_output broke on {variant.id}"
        )
        assert not variant.hidden_invariant(entry.bad_output), (
            f"bad_output sneaked through {variant.id}"
        )


def test_generate_perturbations_produces_the_declared_variants_in_order() -> None:
    case = _retry_entry().case

    variants = generate_perturbations(
        case, rename=2, paraphrase=1, seed=7, protected=_PROTECTED_RETRY
    )

    assert [v.id for v in variants] == [
        "retry-jitter-budget::rename-0",
        "retry-jitter-budget::rename-1",
        "retry-jitter-budget::paraphrase-0",
    ]


def test_generate_perturbations_is_reproducible() -> None:
    case = _retry_entry().case

    first = generate_perturbations(
        case, rename=2, paraphrase=2, seed=7, protected=_PROTECTED_RETRY
    )
    second = generate_perturbations(
        case, rename=2, paraphrase=2, seed=7, protected=_PROTECTED_RETRY
    )

    assert [(v.id, v.task_context) for v in first] == [
        (v.id, v.task_context) for v in second
    ]
