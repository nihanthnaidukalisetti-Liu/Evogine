"""Promotion-only retrieval tests: warm runs see exactly the proven knowledge."""

from evogine.engine.store import Engine


def test_retrieve_returns_only_promoted(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    # All three records share the task context's vocabulary; one is promoted,
    # one is still a candidate, one is still ephemeral.
    engine.propose(record_factory("lr-ephemeral"))
    engine.propose(record_factory("lr-candidate"))
    engine.queue_replay("lr-candidate", ["run-1"])
    engine.propose(record_factory("lr-promoted"))
    engine.queue_replay("lr-promoted", ["run-2"])
    engine.promote("lr-promoted", evidence_factory(["run-2"]))

    hits = engine.retrieve("429 backoff escalates across rollouts", k=5)
    assert [record.id for record in hits] == ["lr-promoted"]


def test_retrieve_returns_empty_when_nothing_is_relevant(
    tmp_path, record_factory, evidence_factory
):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    engine.promote("lr-1", evidence_factory(["run-1"]))

    assert engine.retrieve("database migration deadlock", k=5) == []


def test_retrieve_ranks_by_relevance_and_respects_k(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    for topic, record_id in (("429 backoff", "lr-backoff"), ("deadlock contention", "lr-deadlock")):
        engine.propose(record_factory(record_id, topic=topic))
        engine.queue_replay(record_id, [f"run-{record_id}"])
        engine.promote(record_id, evidence_factory([f"run-{record_id}"]))

    hits = engine.retrieve("429 backoff", k=1)
    assert [record.id for record in hits] == ["lr-backoff"]

    both = engine.retrieve("429 backoff", k=2)
    assert [record.id for record in both] == ["lr-backoff", "lr-deadlock"]


def test_retrieve_empty_store_returns_empty(tmp_path):
    engine = Engine(tmp_path / "store")
    assert engine.retrieve("anything at all", k=5) == []


def test_retrieve_nonpositive_k_returns_empty(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.queue_replay("lr-1", ["run-1"])
    engine.promote("lr-1", evidence_factory(["run-1"]))
    assert engine.retrieve("429 backoff", k=0) == []
