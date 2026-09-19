"""Property tests for the retrieval scorer (V3 seam hardening).

Three properties the V3 growth plan presses, all against the deterministic
:class:`HashingEmbedder` (offline by contract):

- **Process-level determinism** — the same inputs produce the same scores
  and the same ranking across repeated calls, fresh embedder/store
  instances, and freshly spawned interpreters. The :class:`HashingEmbedder`
  promises MD5-stable vectors across processes (the builtin ``hash()`` is
  salted per run); this holds that promise as a tested invariant instead of
  a comment.
- **Floor honesty** — retrieval returns *exactly* the records the floor
  admits, checked against an independent restatement of the hybrid rule:
  nothing below the floor is ever surfaced, nothing at-or-above is ever
  suppressed, ordering follows ``(-score, id)``, and ``k`` truncates. An
  empty result below the floor stays a first-class outcome (V2-3).
- **Titles never influence scores** — the anti-gaming continuity (V2-8,
  PR #10): retitling records — even into a copy of the task context's own
  tokens, the gaming shape a rogue title would try — can never move a
  score, a body text, or the ranking.

The store double exposes only ``promoted_records()``, the single method the
scorer reads; ``tests/test_retrieval.py`` pins that free-function retrieval
agrees with the :class:`Engine` method, so the double exercises the same
seam without paying a git commit per lifecycle step.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.engine.embedders import HashingEmbedder, cosine_similarity
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.retrieval import body_text, lexical_overlap, retrieve

#: Shared vocabulary for bodies, titles, and contexts — overlapping tokens
#: make the generated retrieval problems non-trivial (real partial matches).
WORDS = [
    "429",
    "backoff",
    "jittered",
    "exponential",
    "retry",
    "budget",
    "rollout",
    "cache",
    "stale",
    "deadlock",
    "migration",
    "verdict",
    "replay",
    "canary",
    "latency",
]

_WORDS_STRATEGY = st.sampled_from(WORDS)


class _PromotedStore:
    """The one method the retrieval scorer reads: ``promoted_records()``."""

    def __init__(self, records: list[LearningRecord]) -> None:
        self._records = list(records)

    def promoted_records(self) -> list[LearningRecord]:
        return list(self._records)


def _record(record_id: str, *, title: str, body_words: list[str]) -> LearningRecord:
    """A promoted record whose scored content is fully controlled."""
    words = " ".join(body_words)
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=title,
        body=RecordBody(
            signal=words,
            failure_mode=f"{words} amplifies",
            detection=f"canary observes {words}",
            fix_strategy=f"contain {words} with a budget",
        ),
        status="promoted",
        confidence=0.6,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="test-fixture",
            source="unit-test",
            note="deterministic fixture; never touches a live model",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 18, tzinfo=UTC),
        promoted_at=datetime(2026, 9, 18, tzinfo=UTC),
    )


@st.composite
def _scenarios(draw: st.DrawFn, min_size: int = 1, max_size: int = 6) -> tuple[
    list[LearningRecord], str
]:
    """Generated promoted records plus a task context over shared vocabulary."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    context = " ".join(draw(st.lists(_WORDS_STRATEGY, min_size=0, max_size=6)))
    records = [
        _record(
            f"lr-prop-{idx}",
            title=" ".join(draw(st.lists(_WORDS_STRATEGY, min_size=1, max_size=4))),
            body_words=draw(st.lists(_WORDS_STRATEGY, min_size=1, max_size=6)),
        )
        for idx in range(count)
    ]
    return records, context


def _hybrid_score(
    record: LearningRecord, context: str, embedder: HashingEmbedder
) -> float:
    """The hybrid rule restated from the module docstring, for cross-checking."""
    vectors = embedder.embed([context, body_text(record)])
    return max(
        cosine_similarity(vectors[0], vectors[1]),
        lexical_overlap(record, context),
    )


def _expected_ranking(
    records: list[LearningRecord],
    context: str,
    k: int,
    floor: float,
    embedder: HashingEmbedder,
) -> list[str]:
    """The spec's contract: hybrid score, floor cut, (-score, id) order, k cap."""
    if k <= 0 or not records:
        return []
    scored = [(_hybrid_score(r, context, embedder), r.id) for r in records]
    qualified = [(score, record_id) for score, record_id in scored if score >= floor]
    qualified.sort(key=lambda pair: (-pair[0], pair[1]))
    return [record_id for _, record_id in qualified[:k]]


# ------------------------------------------------------- process determinism


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(
    data=_scenarios(),
    k=st.integers(min_value=-2, max_value=10),
    floor=st.floats(min_value=0.0, max_value=1.0),
)
def test_ranking_is_reproducible_across_calls_and_instances(
    data: tuple[list[LearningRecord], str], k: int, floor: float
) -> None:
    """Same inputs, same ranking — repeated calls, fresh embedder, fresh store."""
    records, context = data
    embedder = HashingEmbedder()
    first = retrieve(_PromotedStore(records), context, k=k, floor=floor, embedder=embedder)
    second = retrieve(
        _PromotedStore(records), context, k=k, floor=floor, embedder=HashingEmbedder()
    )
    third = retrieve(
        _PromotedStore(list(reversed(records))), context, k=k, floor=floor, embedder=embedder
    )

    ids = [r.id for r in first]
    assert ids == [r.id for r in second]
    # Record iteration order is the store's business, never the scorer's:
    # a reversed store yields the same ranking.
    assert ids == [r.id for r in third]
    assert ids == _expected_ranking(records, context, k, floor, embedder)

    # The scores themselves — not just the order they induce — repeat exactly.
    scores = [_hybrid_score(r, context, embedder) for r in records]
    assert scores == [_hybrid_score(r, context, HashingEmbedder()) for r in records]


_FRESH_PROCESS_PROGRAM = r"""
import json
import sys

from evogine.engine.embedders import HashingEmbedder, cosine_similarity
from evogine.engine.records import LearningRecord
from evogine.engine.retrieval import DEFAULT_FLOOR, body_text, lexical_overlap, retrieve


class Store:
    def __init__(self, records):
        self._records = records

    def promoted_records(self):
        return list(self._records)


payload = json.loads(sys.stdin.read())
results = []
for scenario in payload:
    records = [LearningRecord.model_validate(r) for r in scenario["records"]]
    context = scenario["context"]
    embedder = HashingEmbedder()
    scores = []
    for record in records:
        vectors = embedder.embed([context, body_text(record)])
        scores.append(
            max(
                cosine_similarity(vectors[0], vectors[1]),
                lexical_overlap(record, context),
            )
        )
    results.append(
        {
            "scores": scores,
            "ranking": [
                r.id
                for r in retrieve(Store(records), context, k=len(records), floor=0.0)
            ],
            "default_floor": [
                r.id
                for r in retrieve(
                    Store(records), context, k=len(records), floor=DEFAULT_FLOOR
                )
            ],
        }
    )
json.dump(results, sys.stdout)
"""


@settings(max_examples=5, deadline=None)  # two fresh interpreters per example
@given(data=_scenarios(min_size=2, max_size=4))
def test_scores_are_stable_across_fresh_processes(
    data: tuple[list[LearningRecord], str],
) -> None:
    """A brand-new interpreter computes the same scores and ranking.

    Spawns two fresh processes for the same batch and compares them against
    this process exactly (JSON floats round-trip bit-for-bit). This is what
    fails if anyone swaps MD5 feature hashing for a salted builtin hash or
    makes record iteration order observable in the scores.
    """
    records, context = data
    payload = [
        {
            "records": [json.loads(r.model_dump_json()) for r in records],
            "context": context,
        }
    ]

    children: list[dict] = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", _FRESH_PROCESS_PROGRAM],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        children.append(json.loads(proc.stdout))

    embedder = HashingEmbedder()
    expected = {
        "scores": [_hybrid_score(r, context, embedder) for r in records],
        "ranking": [
            r.id for r in retrieve(_PromotedStore(records), context, k=len(records), floor=0.0)
        ],
        "default_floor": [
            r.id
            for r in retrieve(_PromotedStore(records), context, k=len(records))
        ],
    }
    for child in children:
        assert child[0] == expected, "fresh process disagreed on scores or ranking"


# ------------------------------------------------------------- floor honesty


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(
    data=_scenarios(),
    k=st.integers(min_value=-2, max_value=10),
    floor=st.floats(min_value=0.0, max_value=1.0),
)
def test_retrieval_returns_exactly_the_records_the_floor_admits(
    data: tuple[list[LearningRecord], str], k: int, floor: float
) -> None:
    """Nothing below the floor is surfaced; nothing at-or-above is suppressed."""
    records, context = data
    embedder = HashingEmbedder()
    hits = retrieve(_PromotedStore(records), context, k=k, floor=floor, embedder=embedder)

    scores = {r.id: _hybrid_score(r, context, embedder) for r in records}
    assert [hit.id for hit in hits] == _expected_ranking(records, context, k, floor, embedder)
    for hit in hits:
        assert scores[hit.id] >= floor, f"{hit.id} surfaced below the floor"
    assert len(hits) <= max(k, 0)


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(data=_scenarios(), context=st.one_of(st.just(""), _WORDS_STRATEGY))
def test_honest_empty_states(data: tuple[list[LearningRecord], str], context: str) -> None:
    """The degenerate cases all return []: impossible floor, empty store, no k."""
    records, _ = data
    embedder = HashingEmbedder()
    store = _PromotedStore(records)

    # No hybrid score can exceed 1.0: an unreachable floor admits nothing.
    assert retrieve(store, context, k=len(records), floor=2.0, embedder=embedder) == []
    # Nothing promoted — the store's honest answer is [], never padding.
    assert retrieve(_PromotedStore([]), context, k=5, floor=0.0, embedder=embedder) == []
    # An empty context scores 0.0 on every signal; the default floor drops it.
    assert retrieve(store, "", k=len(records), embedder=embedder) == []


# ------------------------------------------------------- title independence


@settings(deadline=None)  # example budget from the profile (dev 30 / ci 200)
@given(data=_scenarios(), game=st.booleans())
def test_titles_never_move_scores_or_ranking(
    data: tuple[list[LearningRecord], str], game: bool
) -> None:
    """Retitling — even into the context's own tokens — cannot move anything.

    The gaming shape a rogue record would try is a title quoting the task
    context verbatim; the scorer must be blind to it. The ``game=False``
    arm is the control: untouched titles trivially preserve the ranking.
    """
    records, context = data
    embedder = HashingEmbedder()

    baseline_scores = [_hybrid_score(r, context, embedder) for r in records]
    baseline_ranking = [
        r.id
        for r in retrieve(_PromotedStore(records), context, k=len(records), floor=0.0)
    ]

    gaming_title = f"{context} {context}" if context else " ".join(WORDS)
    retitled = [
        r.model_copy(update={"title": f"{gaming_title} #{position}"}) if game else r
        for position, r in enumerate(records)
    ]

    for original, mutated in zip(records, retitled, strict=True):
        assert body_text(mutated) == body_text(original), "title leaked into the body"
        assert lexical_overlap(mutated, context) == lexical_overlap(original, context)

    mutated_scores = [_hybrid_score(r, context, embedder) for r in retitled]
    assert mutated_scores == baseline_scores, "a title moved a score"

    mutated_ranking = [
        r.id
        for r in retrieve(_PromotedStore(retitled), context, k=len(retitled), floor=0.0)
    ]
    assert mutated_ranking == baseline_ranking, "a title moved the ranking"
