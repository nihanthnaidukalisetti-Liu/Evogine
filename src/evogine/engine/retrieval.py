"""Hybrid retrieval with an honest empty state (V2-3).

Warm runs score promoted records against the task context by two signals
combined with ``max``: cosine similarity of embedder vectors over the
record's *body fields*, and lexical overlap between the context and those
same fields. Below the relevance floor retrieval returns ``[]`` — an empty
result is a first-class outcome, never a fallback to unrelated records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evogine.engine.embedders import HashingEmbedder, cosine_similarity, scoring_tokens
from evogine.engine.records import LearningRecord

if TYPE_CHECKING:
    from evogine.engine.embedders import Embedder
    from evogine.engine.store import Engine

#: The task context is the plain text a warm run is about to act on.
TaskContext = str

#: Below this hybrid score a record is irrelevant: retrieval returns [].
DEFAULT_FLOOR = 0.35


def body_text(record: LearningRecord) -> str:
    """Every scored field of a record: all four body fields, never the title.

    Titles are excluded by design — the v1 defect was title-only matching —
    so scoring reads only the body, and ``signal``, ``failure_mode``,
    ``detection``, and ``fix_strategy`` always participate. A title match
    alone can therefore never qualify a record.
    """
    body = record.body
    return f"{body.signal} {body.failure_mode} {body.detection} {body.fix_strategy}"


def lexical_overlap(record: LearningRecord, context: TaskContext) -> float:
    """Share of the task context's tokens the record's body covers."""
    context_tokens = set(scoring_tokens(context))
    if not context_tokens:
        return 0.0
    body_tokens = set(scoring_tokens(body_text(record)))
    return len(body_tokens & context_tokens) / len(context_tokens)


def retrieve(
    store: Engine,
    task_context: TaskContext,
    k: int = 5,
    floor: float = DEFAULT_FLOOR,
    embedder: Embedder | None = None,
) -> list[LearningRecord]:
    """Top-``k`` promoted records for ``task_context`` — or [] below ``floor``.

    Hybrid score = max(cosine(embedder, record.body), lexical_overlap(record, ctx)).
    Titles alone can never qualify a record; body fields always participate.
    An empty return is a first-class outcome: warm runs must not invent
    relevance the store doesn't hold.
    """
    if k <= 0:
        return []
    active = embedder if embedder is not None else HashingEmbedder()
    records = store.promoted_records()
    if not records:
        return []
    vectors = active.embed([task_context, *(body_text(record) for record in records)])
    context_vector = vectors[0]
    scored = [
        (
            max(
                cosine_similarity(context_vector, vectors[position + 1]),
                lexical_overlap(record, task_context),
            ),
            record,
        )
        for position, record in enumerate(records)
    ]
    qualified = [(score, record) for score, record in scored if score >= floor]
    qualified.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [record for _, record in qualified[:k]]
