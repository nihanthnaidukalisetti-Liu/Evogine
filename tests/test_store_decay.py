"""Decay tests: configured confidence aging with an evergreen exemption.

V2-5 decay control: confidence follows an exponential half-life curve with
an asymptotic floor, anchored on each record's last confidence stamp so
passes compose exactly and are idempotent. Records flagged ``evergreen``
are exempt — safety-critical knowledge can never decay below retrieval.
"""

import json
import subprocess
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from evogine.engine.records import DecayConfig, decayed_confidence
from evogine.engine.store import Engine

CREATED = datetime(2026, 9, 18, tzinfo=UTC)


def _promoted(engine: Engine, record_factory, evidence_factory, record_id: str, **updates):
    """Propose, replay, and promote one deterministic record."""
    record = record_factory(record_id).model_copy(update=updates)
    engine.propose(record)
    engine.queue_replay(record_id, ["run-1"])
    return engine.promote(record_id, evidence_factory(["run-1"]))


def test_decayed_confidence_unchanged_at_zero_age():
    """Zero elapsed time decays nothing."""
    config = DecayConfig()
    confidence = decayed_confidence(0.6, since=CREATED, now=CREATED, config=config)
    assert confidence == 0.6


def test_decayed_confidence_halves_above_floor_at_one_half_life():
    """At one half-life the margin above the floor is halved, exactly."""
    config = DecayConfig(half_life_days=30.0, floor=0.1)
    confidence = decayed_confidence(
        0.8, since=CREATED, now=CREATED + timedelta(days=30), config=config
    )
    assert confidence == pytest.approx(0.1 + 0.7 / 2)


def test_decayed_confidence_never_below_floor():
    """Arbitrarily old confidence approaches the floor but never crosses it."""
    config = DecayConfig(half_life_days=30.0, floor=0.05)
    confidence = decayed_confidence(
        0.9, since=CREATED, now=CREATED + timedelta(days=100_000), config=config
    )
    assert confidence == pytest.approx(0.05, abs=1e-9)
    assert confidence >= config.floor


def test_decayed_confidence_only_lowers():
    """A confidence already under the floor is left alone, never lifted."""
    config = DecayConfig(half_life_days=30.0, floor=0.05)
    confidence = decayed_confidence(
        0.02, since=CREATED, now=CREATED + timedelta(days=365), config=config
    )
    assert confidence == 0.02


def test_decayed_confidence_ignores_time_travel():
    """A ``now`` before the anchor changes nothing."""
    config = DecayConfig()
    confidence = decayed_confidence(
        0.6, since=CREATED, now=CREATED - timedelta(days=10), config=config
    )
    assert confidence == 0.6


def test_decay_config_validates_bounds():
    """A degenerate half-life or a floor at 1.0 is refused."""
    with pytest.raises(ValidationError, match="half_life_days"):
        DecayConfig(half_life_days=0.0)
    with pytest.raises(ValidationError, match="floor"):
        DecayConfig(floor=1.0)


def test_decay_lowers_aged_confidence_and_stamps_anchor(tmp_path, record_factory, evidence_factory):
    engine = Engine(tmp_path / "store")
    _promoted(engine, record_factory, evidence_factory, "lr-1")
    now = CREATED + timedelta(days=30)

    (decayed,) = engine.decay(now=now)

    assert decayed.confidence == pytest.approx(0.05 + 0.55 / 2)  # one half-life
    assert decayed.confidence_updated_at == now
    assert engine.get("lr-1").confidence == decayed.confidence


def test_decay_composes_exactly(tmp_path, record_factory, evidence_factory):
    """Two passes at t+10d and t+25d equal one pass at t+35d."""
    engine = Engine(tmp_path / "store")
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    engine.decay(now=CREATED + timedelta(days=10))
    composed = engine.decay(now=CREATED + timedelta(days=35))[0]

    direct_engine = Engine(tmp_path / "direct")
    _promoted(direct_engine, record_factory, evidence_factory, "lr-1")
    (direct,) = direct_engine.decay(now=CREATED + timedelta(days=35))

    assert composed.confidence == pytest.approx(direct.confidence)


def test_decay_is_idempotent_at_a_fixed_instant(tmp_path, record_factory, evidence_factory):
    """Re-running the pass at the same instant changes nothing, writes nothing."""
    root = tmp_path / "store"
    engine = Engine(root)
    _promoted(engine, record_factory, evidence_factory, "lr-1")
    now = CREATED + timedelta(days=30)
    audit_lines = lambda: len((root / "audit.jsonl").read_text("utf-8").splitlines())
    commits = lambda: int(
        subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    first = engine.decay(now=now)
    audit_after_first, commits_after_first = audit_lines(), commits()

    second = engine.decay(now=now)

    assert first and second == []
    assert audit_lines() == audit_after_first
    assert commits() == commits_after_first


def test_decay_respects_the_floor(tmp_path, record_factory, evidence_factory):
    """Deep aging lands at the configured floor, never below it."""
    engine = Engine(tmp_path / "store")
    config = DecayConfig(half_life_days=30.0, floor=0.2)
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    engine.decay(config=config, now=CREATED + timedelta(days=100_000))

    record = engine.get("lr-1")
    assert record.confidence == pytest.approx(0.2, abs=1e-9)
    assert record.confidence >= config.floor


def test_evergreen_records_are_exempt_from_decay(tmp_path, record_factory, evidence_factory):
    """The decay-floor exemption: evergreen confidence never moves.

    A safety-critical record and an ordinary twin age identically; after
    ten half-lives the ordinary record sits near the floor while the
    evergreen record is untouched and still retrievable.
    """
    engine = Engine(tmp_path / "store")
    _promoted(engine, record_factory, evidence_factory, "lr-safe", evergreen=True)
    _promoted(
        engine,
        record_factory,
        evidence_factory,
        "lr-mortal",
        title="Unhandled deadlock contention escalates across rollouts",
        body=record_factory("lr-mortal", topic="deadlock contention").body,
    )
    context = f"{record_factory('lr-safe').title} {record_factory('lr-safe').body.signal}"

    engine.decay(now=CREATED + timedelta(days=300))

    safe = engine.get("lr-safe")
    mortal = engine.get("lr-mortal")
    assert safe.confidence == 0.6  # exempt: never aged toward the floor
    assert mortal.confidence < 0.6
    assert mortal.confidence >= DecayConfig().floor
    # The exempt record stays retrievable — it never aged out of consideration.
    assert "lr-safe" in [record.id for record in engine.retrieve(context)]


def test_decay_skips_terminal_states(tmp_path, record_factory, evidence_factory):
    """Dropped and merged records are frozen history, not live belief.

    The survivor keeps its live status after a merge, so it still ages;
    the absorbed record and the dropped record freeze in place.
    """
    engine = Engine(tmp_path / "store")
    engine.propose(record_factory("lr-1"))
    engine.demote("lr-1", "rejected at the gate")
    propose_and_merge(engine, record_factory)

    engine.decay(now=CREATED + timedelta(days=365))

    assert engine.get("lr-1").confidence == 0.6  # dropped: frozen
    assert engine.get("lr-absorbed").confidence == 0.9  # merged: frozen
    assert engine.get("lr-survivor").confidence < 0.9  # live: ages from its merge anchor


def propose_and_merge(engine: Engine, record_factory) -> None:
    """A survivor at confidence 0.9 absorbing an identical duplicate."""
    engine.propose(record_factory("lr-survivor"))
    engine.propose(record_factory("lr-absorbed").model_copy(update={"confidence": 0.9}))
    engine.merge("lr-survivor", "lr-absorbed", "duplicate proposal")


def test_decay_after_merge_anchors_on_the_merge_moment(tmp_path, record_factory):
    """A merged boost decays from the merge, not from record creation."""
    engine = Engine(tmp_path / "store")
    propose_and_merge(engine, record_factory)
    survivor = engine.get("lr-survivor")
    merged_at = survivor.confidence_updated_at
    assert merged_at is not None

    (decayed,) = engine.decay(now=merged_at + timedelta(days=30))

    assert decayed.confidence == pytest.approx(0.05 + 0.85 / 2)  # one half-life of 0.9


def test_decay_is_audited_and_committed(tmp_path, record_factory, evidence_factory):
    root = tmp_path / "store"
    engine = Engine(root)
    _promoted(engine, record_factory, evidence_factory, "lr-1")

    (decayed,) = engine.decay(now=CREATED + timedelta(days=30))

    events = [
        json.loads(line)
        for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event"] == "decay"
    assert events[-1]["record_id"] == "lr-1"
    assert events[-1]["to_status"] == "promoted"
    assert f"{decayed.confidence:.4f}" in events[-1]["reason"]

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return proc.stdout

    assert int(git("rev-list", "--count", "HEAD").strip()) == 4  # propose, queue, promote, decay
    assert git("status", "--porcelain").strip() == ""


def test_noop_decay_writes_nothing(tmp_path, record_factory, evidence_factory):
    """Decaying at the anchor instant is a no-op: no events, no empty commit."""
    root = tmp_path / "store"
    engine = Engine(root)
    _promoted(engine, record_factory, evidence_factory, "lr-1")
    audit_lines = len((root / "audit.jsonl").read_text("utf-8").splitlines())
    commits = int(
        subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    moved = engine.decay(now=CREATED)

    assert moved == []
    assert len((root / "audit.jsonl").read_text("utf-8").splitlines()) == audit_lines
    latest = int(
        subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert latest == commits


def test_decay_honors_custom_config(tmp_path, record_factory, evidence_factory):
    """The engine applies exactly the configured curve."""
    engine = Engine(tmp_path / "store")
    _promoted(engine, record_factory, evidence_factory, "lr-1")
    config = DecayConfig(half_life_days=15.0, floor=0.2)
    now = CREATED + timedelta(days=45)

    (decayed,) = engine.decay(config=config, now=now)

    expected = decayed_confidence(0.6, since=CREATED, now=now, config=config)
    assert decayed.confidence == pytest.approx(expected)
