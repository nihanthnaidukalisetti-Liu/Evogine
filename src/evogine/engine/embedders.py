"""Embedder seam for hybrid retrieval: swap a provider backend in by construction.

The retrieval path must never depend on a live model by default: the
:class:`HashingEmbedder` is a local, fully deterministic stand-in that keeps
CI offline and reproducible. A provider embedding model later implements the
same one-method :class:`Embedder` protocol and is injected at Engine
construction — a config change, not a retrieval rewrite.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

# Shared scoring vocabulary with retrieval's lexical ranker: lowercase
# alphanumeric runs of length >= 3.
_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Text in, fixed-length vectors out — the seam provider backends plug into."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``; ``result[i]`` pairs with ``texts[i]``."""
        ...


def scoring_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (length >= 3), order and multiplicity kept."""
    return [token for token in _WORD_RE.findall(text.lower()) if len(token) >= 3]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of two equal-length vectors; ``0.0`` when either is all zeros."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(component * component for component in a))
    norm_b = math.sqrt(sum(component * component for component in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HashingEmbedder:
    """Deterministic local default: signed feature hashing, L2-normalized.

    Each token's *set membership* hashes to one (dimension, sign) pair via
    MD5 — stable across processes, unlike the builtin ``hash()``, which is
    salted per run and would make ranking flake in CI. Set semantics (not
    term frequency) keep vectors a pure function of a text's vocabulary:
    texts sharing content vocabulary land close together regardless of word
    order, unrelated texts land near-orthogonal. The fixed dimension bound
    keeps every vector the same length, so one batch call serves any input.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in set(scoring_tokens(text)):
            digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]
