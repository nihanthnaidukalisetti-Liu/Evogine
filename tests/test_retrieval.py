"""V2-3 hybrid retrieval: paraphrase match, honest empty state, determinism.

Every test runs against the deterministic :class:`HashingEmbedder` — the
offline fixture embedder; the suite never calls a live model.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from evogine.engine.embedders import HashingEmbedder, cosine_similarity
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.retrieval import DEFAULT_FLOOR, lexical_overlap, retrieve
from evogine.engine.store import Engine

if TYPE_CHECKING:
    from collections.abc import Callable


def build_record(
    record_id: str,
    *,
    title: str,
    signal: str,
    failure_mode: str,
    detection: str,
    fix_strategy: str,
) -> LearningRecord:
    """An ephemeral draft with fully controlled text for scoring tests."""
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=title,
        body=RecordBody(
            signal=signal,
            failure_mode=failure_mode,
            detection=detection,
            fix_strategy=fix_strategy,
        ),
        status="ephemeral",
        confidence=0.6,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="test-fixture",
            source="unit-test",
            note="deterministic fixture; never touches a live model",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 18, tzinfo=UTC),
        promoted_at=None,
    )


def promote_record(
    engine: Engine,
    record: LearningRecord,
    evidence_factory: Callable[[list[str]], RecordCounterfactual],
) -> None:
    """Walk a record through the store's own gates to ``promoted``."""
    engine.propose(record)
    engine.queue_replay(record.id, [f"run-{record.id}"])
    engine.promote(record.id, evidence_factory([f"run-{record.id}"]))


# ------------------------------------------------------------- acceptance


def test_paraphrase_context_matches_promoted_record(tmp_path, evidence_factory):
    """Acceptance: a reordered, reworded context still retrieves the record.

    The paraphrase shares content vocabulary with the body but not sentence
    structure — retrieval is by meaning-bearing overlap, not substring match.
    """
    engine = Engine(tmp_path / "store")
    promote_record(
        engine,
        build_record(
            "lr-ratelimit",
            title="Rate limiter collapse under retry storms",
            signal="429 counts spike across consecutive rollouts",
            failure_mode="unbounded retries hammer the rate limiter once limits tighten",
            detection="counterfactual replay flips the verifier verdict on the degraded case",
            fix_strategy="add jittered exponential backoff with a per-rollout retry budget",
        ),
        evidence_factory,
    )

    hits = engine.retrieve(
        "the rate limiter is hammered by unbounded retries, so backoff must be jittered",
        k=5,
    )
    assert [hit.id for hit in hits] == ["lr-ratelimit"]


def test_unrelated_context_returns_empty(tmp_path, evidence_factory):
    """Acceptance: an unrelated context gets [] — never unrelated records."""
    engine = Engine(tmp_path / "store")
    promote_record(
        engine,
        build_record(
            "lr-1",
            title="Unhandled 429 backoff escalates across rollouts",
            signal="rising 429 backoff counts across consecutive rollouts",
            failure_mode="429 backoff amplifies without jittered backoff",
            detection="counterfactual replay flips the verifier verdict on the degraded case",
            fix_strategy="add jittered exponential backoff with a per-rollout retry budget",
        ),
        evidence_factory,
    )

    assert engine.retrieve("database migration deadlock", k=5) == []


def test_same_input_same_ranking(tmp_path, record_factory, evidence_factory):
    """Acceptance: same store, same context — identical ranking, every time.

    Covers two sources of nondeterminism at once: record iteration order
    (a fresh Engine re-reads the git-backed store from disk) and the
    embedder (vectors must be stable across calls and instances).
    """
    root = tmp_path / "store"
    engine = Engine(root)
    for topic, record_id in (
        ("429 backoff", "lr-backoff"),
        ("deadlock contention", "lr-deadlock"),
        ("connection pool exhaustion", "lr-pool"),
    ):
        promote_record(engine, record_factory(record_id, topic=topic), evidence_factory)

    first = engine.retrieve("429 backoff", k=3)
    second = engine.retrieve("429 backoff", k=3)
    fresh = Engine(root).retrieve("429 backoff", k=3)

    assert [hit.id for hit in first] == [hit.id for hit in second]
    assert [hit.id for hit in first] == [hit.id for hit in fresh]

    embedder_a = HashingEmbedder()
    embedder_b = HashingEmbedder()
    assert embedder_a.embed(["429 backoff", ""]) == embedder_b.embed(["429 backoff", ""])


# -------------------------------------------------- title/body contract


def test_title_only_match_never_qualifies(tmp_path, evidence_factory):
    """Spec failure-mode guard: a title matching every context token, an
    unrelated body — the record must NOT surface. The control proves the
    same record IS retrievable by its body topic, so the guard isn't vacuous.
    """
    engine = Engine(tmp_path / "store")
    record = build_record(
        "lr-title-only",
        title="429 backoff escalates across rollouts",
        signal="cache invalidation storms during deploys",
        failure_mode="stale cache keys accumulate silently",
        detection="synthetic canary observes stale reads",
        fix_strategy="purge cache keys on deploy boundaries",
    )
    promote_record(engine, record, evidence_factory)

    assert engine.retrieve("429 backoff escalates across rollouts", k=5) == []
    assert engine.retrieve("cache invalidation storms during deploys", k=5) == [
        engine.get("lr-title-only")
    ]


def test_body_fields_participate_without_title_help(tmp_path, evidence_factory):
    """Body-only match qualifies: only ``fix_strategy`` shares the context's
    vocabulary — title, signal, failure_mode, detection contribute nothing.
    """
    engine = Engine(tmp_path / "store")
    promote_record(
        engine,
        build_record(
            "lr-body-match",
            title="Completely unrelated headline about networking",
            signal="nothing here matches the context at all",
            failure_mode="likewise unrelated",
            detection="also unrelated",
            fix_strategy="add jittered exponential backoff with a per-rollout retry budget",
        ),
        evidence_factory,
    )

    hits = engine.retrieve("jittered backoff budget", k=5)
    assert [hit.id for hit in hits] == ["lr-body-match"]


def test_embedding_component_rescues_match_below_lexical_floor(
    tmp_path, evidence_factory
):
    """Hybrid is a max(): when lexical coverage alone sits below the floor,
    the embedding signal can still qualify the record.

    Constructed fixture: the context shares 2 of its 6 tokens with a
    5-token body — lexical 1/3 < 0.35, cosine ~0.365 > 0.35 (values fixed
    by the deterministic embedder; asserted against HashingEmbedder directly).
    """
    engine = Engine(tmp_path / "store")
    record = build_record(
        "lr-rescued",
        title="Escalation ladder regression",
        signal="verdict flips",
        failure_mode="unbounded retries",
        detection="replay",
        fix_strategy="verdict",
    )
    promote_record(engine, record, evidence_factory)

    context = "unbounded retries during database migration deadlocks"
    assert lexical_overlap(record, context) == 1 / 3
    assert lexical_overlap(record, context) < DEFAULT_FLOOR
    body = f"{record.body.signal} {record.body.failure_mode} {record.body.detection} {record.body.fix_strategy}"
    vectors = HashingEmbedder().embed([context, body])
    assert cosine_similarity(vectors[0], vectors[1]) >= DEFAULT_FLOOR

    assert [hit.id for hit in engine.retrieve(context, k=5)] == ["lr-rescued"]


# ------------------------------------------------------ floor semantics


def test_below_floor_partial_overlap_returns_empty(tmp_path, evidence_factory):
    """One shared token out of six (lexical ~0.17, cosine ~0.16) stays below
    the floor on BOTH signals: the record is not surfaced.
    """
    engine = Engine(tmp_path / "store")
    record = build_record(
        "lr-1",
        title="Unhandled 429 backoff escalates across rollouts",
        signal="rising 429 backoff counts across consecutive rollouts",
        failure_mode="429 backoff amplifies without jittered backoff",
        detection="counterfactual replay flips the verifier verdict on the degraded case",
        fix_strategy="add jittered exponential backoff with a per-rollout retry budget",
    )
    promote_record(engine, record, evidence_factory)

    context = "429 pagination collapses during traffic spikes"
    assert lexical_overlap(record, context) < DEFAULT_FLOOR
    assert engine.retrieve(context, k=5) == []


def test_floor_parameter_raises_the_bar(tmp_path, evidence_factory):
    """``floor`` is caller-configurable: a 0.5-overlap match passes the
    default floor and is dropped by a raised one.
    """
    engine = Engine(tmp_path / "store")
    promote_record(
        engine,
        build_record(
            "lr-1",
            title="Unhandled 429 backoff escalates across rollouts",
            signal="rising 429 backoff counts across consecutive rollouts",
            failure_mode="429 backoff amplifies without jittered backoff",
            detection="counterfactual replay flips the verifier verdict on the degraded case",
            fix_strategy="add jittered exponential backoff with a per-rollout retry budget",
        ),
        evidence_factory,
    )

    context = "429 pagination"
    assert [hit.id for hit in engine.retrieve(context, k=5)] == ["lr-1"]
    assert engine.retrieve(context, k=5, floor=0.7) == []


# ------------------------------------------------------------- embedder


def test_hashing_embedder_contract():
    """The default embedder is deterministic, unit-norm, and zero on no tokens."""
    embedder = HashingEmbedder()
    vectors = embedder.embed(["429 backoff storms", "", "429 backoff storms"])

    norm = math.sqrt(sum(component * component for component in vectors[0]))
    assert abs(norm - 1.0) < 1e-9
    assert vectors[0] == vectors[2]
    assert all(component == 0.0 for component in vectors[1])
    assert cosine_similarity(vectors[0], vectors[1]) == 0.0
    assert len(vectors[0]) == len(vectors[1]) == 256


def test_module_level_retrieve_matches_engine_method(
    tmp_path, record_factory, evidence_factory
):
    """The spec's free-function contract and the Engine method agree."""
    engine = Engine(tmp_path / "store")
    promote_record(engine, record_factory("lr-1"), evidence_factory)

    context = "429 backoff escalates across rollouts"
    assert retrieve(engine, context, k=5) == engine.retrieve(context, k=5)
