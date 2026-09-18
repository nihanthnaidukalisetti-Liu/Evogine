"""Case definitions for the decay-pressure benchmark.

A case pairs the engine-facing task context with its hidden host-side
invariant: the oracle the harness grades submissions against. The invariant
is part of the case definition and is deliberately kept out of every
engine-facing surface — the engine never reads a case's oracle.
"""

from evogine.bench.cases.base import BenchCase
from evogine.bench.cases.corpus import CorpusEntry, default_corpus, entries

__all__ = ["BenchCase", "CorpusEntry", "default_corpus", "entries"]
