"""Anti-gaming separation invariants: judges never grade their own proxies.

V2-8 closes the loop holes that let a system certify itself:

- **Hidden tests distinct from gate-visible tests by construction** — the
  benchmark's hidden oracle cases (V2-7) and the tests a gate can see or
  admit (replay cases, gold-sanity admission ids) live in disjoint
  registries. A test that judges a gate must never be a test the gate
  can see, grade, or influence.
- **Promotion proxy never equals the evaluation verifier** — the
  never-worse replay check that admits knowledge (V2-1) and the scored
  benchmark protocol with hidden invariants (V2-7) are different judges
  on purpose. If they ever collapse into one, passing the gate would be
  the same act as passing the evaluation, and the anti-gaming story
  collapses.

The invariants are executable so the separation is a construction rule,
not a comment: the test fixtures assert them against the real corpus and
the locked proxy/verifier identities from the V2 spec.
"""

from __future__ import annotations

from collections.abc import Iterable


class SeparationError(ValueError):
    """A separation invariant was violated — refuse the construction."""


def assert_test_registries_disjoint(
    gate_tests: Iterable[str],
    hidden_tests: Iterable[str],
) -> None:
    """Refuse any id that appears in both the gate-visible and hidden registries.

    ``gate_tests``: test ids a gate can see or admit (replay case ids,
    gold-sanity admission case ids, gate-visible eval cases).
    ``hidden_tests``: benchmark case ids judged by hidden host-side
    oracles — never visible to any gate.
    """
    overlap = sorted(set(gate_tests) & set(hidden_tests))
    if overlap:
        raise SeparationError(
            "hidden tests must stay distinct from gate-visible tests by "
            f"construction — overlapping ids: {overlap}"
        )


def assert_proxy_is_not_verifier(
    promotion_proxy: str,
    evaluation_verifier: str,
) -> None:
    """Refuse a construction where the promotion proxy is the eval verifier."""
    if promotion_proxy == evaluation_verifier:
        raise SeparationError(
            "the promotion proxy must never equal the evaluation verifier — "
            "one judge certifying itself is the definition of gaming"
        )
