"""V2-7: deterministically generated rename/paraphrase perturbation variants.

A robust knowledge win should survive surface variation of the task prompt.
Variants are pure functions of ``(case, variant_index, seed)`` — no model,
no randomness beyond the seeded hash — so re-running generation reproduces
the exact same variants, and the protocol fingerprint can pin them like any
other corpus.

Renames swap content words for neutral aliases, never touching *protected*
tokens (the gold/bad answer vocabulary callers pass in), so the hidden
oracle's grading surface is invariant by construction. Paraphrases apply a
fixed, indexed substitution table with the same guarantee. The oracle itself
is carried over untouched: a variant differs from its base case in the
prompt text only.
"""

from __future__ import annotations

import hashlib
import re

from evogine.bench.cases import BenchCase

_ALIAS_POOL: tuple[str, ...] = (
    "processor",
    "pipeline",
    "component",
    "handler",
    "service",
    "module",
    "adapter",
    "broker",
)

_PARAPHRASES: tuple[tuple[tuple[str, str], ...], ...] = (
    (("State the", "Describe the"), ("State how", "Describe how"), ("State what", "Describe what")),
    (("must", "needs to"),),
    (("prevents this", "stops this from happening"), ("safeguard", "protection")),
)

_WORD = re.compile(r"[a-z_][a-z0-9_]*")


def _stable_hash(text: str) -> int:
    """Deterministic 64-bit hash (Python's hash() is randomized per process)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _substitute(text: str, mapping: dict[str, str]) -> str:
    """Single-pass whole-word substitution; one pass, so no cascade renames."""
    if not mapping:
        return text
    pattern = re.compile(
        r"\b(?:" + "|".join(map(re.escape, mapping)) + r")\b", re.IGNORECASE
    )
    return pattern.sub(lambda m: mapping.get(m.group(0).lower(), m.group(0)), text)


def rename_variant(
    case: BenchCase,
    *,
    variant_index: int,
    seed: int,
    protected: frozenset[str] = frozenset(),
) -> BenchCase:
    """Return the case with content words swapped for deterministic aliases.

    Renames are a pure function of ``(token, seed, variant_index, position)``
    and skip protected tokens, so oracle vocabulary survives verbatim.
    """
    tokens = sorted(
        {
            token
            for token in _WORD.findall(case.task_context.lower())
            if len(token) >= 4 and token not in protected
        }
    )
    mapping: dict[str, str] = {}
    for position, token in enumerate(tokens):
        alias = _ALIAS_POOL[(_stable_hash(f"{seed}:{variant_index}:{token}") + position) % len(_ALIAS_POOL)]
        if alias == token:
            alias = _ALIAS_POOL[(_stable_hash(f"{seed}:{variant_index}:{token}") + position + 1) % len(_ALIAS_POOL)]
        mapping[token] = alias

    return case.model_copy(
        update={
            "id": f"{case.id}::rename-{variant_index}",
            "task_context": _substitute(case.task_context, mapping),
        }
    )


def paraphrase_variant(case: BenchCase, *, variant_index: int) -> BenchCase:
    """Return the case reworded by the fixed substitution table for the index.

    Tables cycle by index, so variant 0 and variant N reproduce the same
    rewording — and the same protocol can pin both deterministically.
    """
    table = _PARAPHRASES[variant_index % len(_PARAPHRASES)]
    text = case.task_context
    for old, new in table:
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return case.model_copy(
        update={"id": f"{case.id}::paraphrase-{variant_index}", "task_context": text}
    )


def generate_perturbations(
    case: BenchCase,
    *,
    rename: int = 0,
    paraphrase: int = 0,
    seed: int = 0,
    protected: frozenset[str] = frozenset(),
) -> list[BenchCase]:
    """Generate the declared variants: ``rename`` renames then ``paraphrase`` rewordings.

    Same arguments ⇒ byte-identical variant list, every time.
    """
    variants = [
        rename_variant(case, variant_index=index, seed=seed, protected=protected)
        for index in range(rename)
    ]
    variants += [
        paraphrase_variant(case, variant_index=index) for index in range(paraphrase)
    ]
    return variants


__all__ = ["generate_perturbations", "paraphrase_variant", "rename_variant"]
