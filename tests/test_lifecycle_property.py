"""Property test: lifecycle direction under fuzzed V2-5 policy interleavings.

A random walk over the full policy surface — ``decay``, ``retire``,
``reinstate``, ``merge`` — interleaved with the core
propose/queue/promote/demote transitions, including every invalid
interleaving the store must refuse. Two invariants hold after *every single
operation*:

- **Direction** (the V3 seam): a record may appear in ``retrieve`` only
  while a successful ``promote`` call holds for it. Decay, retirement,
  reinstatement, and consolidation reshuffle confidence and status, but
  none of them may manufacture retrievability — merging a promoted record
  away, retiring it, or letting it decay can only remove knowledge from
  retrieval, never add un-proven knowledge to it.
- **Decay scope**: a decay pass changes confidence only — never status —
  and never touches terminal records (``dropped``/``merged``/``retired``)
  or ``evergreen`` records: safety-critical knowledge is never aged toward
  the floor, so it can never decay out of consideration.

Refused transitions must raise (:class:`StoreError`) without mutating; the
invariants are re-checked after refusals too. Process-heavy by nature (one
git commit per transition), so the example count is pinned explicitly — the
CI Hypothesis profile raises the in-memory suites, not this walk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from conftest import make_record
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.engine.records import DecayConfig, LearningRecord, RecordCounterfactual
from evogine.engine.store import Engine, StoreError

TOPICS = ["429 backoff", "deadlock contention", "cache stampede"]
RECORD_IDS = ["lr-life-0", "lr-life-1", "lr-life-2"]
OPERATIONS = ["propose", "queue", "promote", "demote", "decay", "retire", "reinstate", "merge"]
#: Fixed walk birthday: every make_record shares it, so decay ``now`` values
#: are always strictly after every anchor in the walk.
CREATED = datetime(2026, 9, 18, tzinfo=UTC)


def _records_on_disk(root: Path) -> dict[str, LearningRecord]:
    return {
        path.name.removesuffix(".json"): LearningRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in sorted((root / "records").glob("*.json"))
    }


def _audit_seq(root: Path) -> list[int]:
    path = root / "audit.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)["seq"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _context_for(topic_index: int) -> str:
    draft = make_record("lr-walk", topic=TOPICS[topic_index])
    return f"{draft.title} {draft.body.signal} {draft.body.failure_mode}"


@settings(max_examples=25, deadline=None)  # git commits per op: examples pinned
@given(
    ops=st.lists(
        st.tuples(st.sampled_from(OPERATIONS), st.integers(0, 2), st.integers(0, 2)),
        max_size=10,
    ),
    half_life_days=st.floats(min_value=1.0, max_value=60.0),
    decay_floor=st.floats(min_value=0.0, max_value=0.5),
    decay_days=st.floats(min_value=1.0, max_value=400.0),
)
def test_fuzzed_policy_interleavings_never_reach_retrievable_without_promotion(
    ops: list[tuple[str, int, int]],
    half_life_days: float,
    decay_floor: float,
    decay_days: float,
) -> None:
    """Walk the policy surface; check the direction invariant after every op.

    Record ``a`` picks the record, ``b`` picks its topic on ``propose`` (so
    equal ``b`` draws make genuinely-mergeable duplicate pairs) and flags
    odd-indexed records ``evergreen`` so the decay-scope invariant has
    evergreen subjects in every walk.
    """
    root = Path(tempfile.mkdtemp(prefix="evogine-lifecycle-prop-"))
    try:
        engine = Engine(root)
        promoted_now: set[str] = set()
        ever_promoted: set[str] = set()
        retired_from: dict[str, str] = {}

        for op, a, b in ops:
            rid = RECORD_IDS[a]
            try:
                if op == "propose":
                    draft = make_record(rid, topic=TOPICS[b])
                    if a % 2 == 1:  # odd records are evergreen in every walk
                        draft = draft.model_copy(update={"evergreen": True})
                    engine.propose(draft)
                elif op == "queue":
                    engine.queue_replay(rid, [f"run-{rid}"])
                elif op == "promote":
                    engine.promote(
                        rid,
                        RecordCounterfactual(
                            replay_run_ids=[f"run-{rid}"],
                            verdict_change=True,
                            evidence="runs/replay/property-walk.json",
                        ),
                    )
                    promoted_now.add(rid)
                    ever_promoted.add(rid)
                elif op == "demote":
                    engine.demote(rid, f"property-walk drop of {rid}")
                    promoted_now.discard(rid)
                elif op == "decay":
                    before = {
                        rec_id: (record.status, record.confidence, record.evergreen)
                        for rec_id, record in _records_on_disk(root).items()
                    }
                    engine.decay(
                        DecayConfig(half_life_days=half_life_days, floor=decay_floor),
                        now=CREATED + timedelta(days=decay_days),
                    )
                    for rec_id, (status, confidence, evergreen) in before.items():
                        record = engine.get(rec_id)
                        assert record.status == status, "a decay pass changed a status"
                        frozen = evergreen or status in ("dropped", "merged", "retired")
                        if frozen:
                            assert record.confidence == confidence, (
                                f"decay moved {rec_id} (evergreen={evergreen}, "
                                f"status={status})"
                            )
                elif op == "retire":
                    status_before = engine.get(rid).status
                    engine.retire(rid, f"property-walk retirement of {rid}")
                    retired_from[rid] = status_before
                    promoted_now.discard(rid)  # retired is not retrievable
                elif op == "reinstate":
                    engine.reinstate(rid)
                    if retired_from.pop(rid, None) == "promoted":
                        promoted_now.add(rid)
                elif op == "merge":
                    primary_id, duplicate_id = RECORD_IDS[a], RECORD_IDS[b]
                    engine.merge(primary_id, duplicate_id, "property-walk consolidation")
                    # The absorbed record is terminal; only the survivor's own
                    # promotion status decides what retrieval may see.
                    promoted_now.discard(duplicate_id)
            except StoreError:
                # Invalid transitions (duplicate propose, promote of an
                # ephemeral, retire of a dropped record, self-merge, ...) must
                # raise, never mutate. The invariants below still hold.
                pass

            # The direction invariant, after every single operation: nothing
            # that is not actively promoted is ever retrievable.
            for topic_index in range(len(TOPICS)):
                context = _context_for(topic_index)
                for hit in engine.retrieve(context, k=len(RECORD_IDS)):
                    assert hit.id in promoted_now, (
                        f"{hit.id} retrieved with status {hit.status} "
                        "without an active promotion"
                    )

            # The mirror direction on disk: the only route to "promoted" is a
            # successful promote call — no policy pass can fake the state.
            for rec_id, record in _records_on_disk(root).items():
                assert record.status in {
                    "ephemeral",
                    "candidate",
                    "promoted",
                    "dropped",
                    "merged",
                    "retired",
                }
                assert record.status != "promoted" or rec_id in ever_promoted

            # The append-only trail stays gap-free through every interleaving.
            seqs = _audit_seq(root)
            assert seqs == list(range(1, len(seqs) + 1)), "audit sequence has gaps"
    finally:
        shutil.rmtree(root)


@settings(max_examples=25, deadline=None)  # git commits: examples pinned
@given(
    half_life_days=st.floats(min_value=1.0, max_value=60.0),
    decay_floor=st.floats(min_value=0.0, max_value=0.5),
    decay_days=st.floats(min_value=1.0, max_value=400.0),
    repeat_passes=st.integers(min_value=1, max_value=3),
)
def test_reinstated_promotion_is_retrievable_again_and_decay_composes(
    half_life_days: float, decay_floor: float, decay_days: float, repeat_passes: int
) -> None:
    """The retirement round-trip restores retrievability exactly.

    promote -> retire -> reinstate returns a promoted record to retrieval;
    the record's confidence composes across decay passes anchored on each
    pass's own ``confidence_updated_at`` stamp, so re-running decay at the
    same instant is a no-op (the direction property never wobbles).
    """
    root = Path(tempfile.mkdtemp(prefix="evogine-lifecycle-roundtrip-"))
    try:
        engine = Engine(root)
        rid = RECORD_IDS[0]
        engine.propose(make_record(rid, topic=TOPICS[0]))
        engine.queue_replay(rid, [f"run-{rid}"])
        engine.promote(
            rid,
            RecordCounterfactual(
                replay_run_ids=[f"run-{rid}"],
                verdict_change=True,
                evidence="runs/replay/property-walk.json",
            ),
        )
        context = _context_for(0)
        assert [hit.id for hit in engine.retrieve(context, k=3)] == [rid]

        engine.retire(rid, "stale under the current rollout mix")
        assert engine.retrieve(context, k=3) == []

        engine.reinstate(rid)
        assert [hit.id for hit in engine.retrieve(context, k=3)] == [rid]

        # Decay passes compose: each anchors on the previous stamp, and a
        # second pass at the same instant moves nothing.
        moment = CREATED + timedelta(days=decay_days)
        config = DecayConfig(half_life_days=half_life_days, floor=decay_floor)
        for _ in range(repeat_passes):
            moment = moment + timedelta(days=1)
            engine.decay(config, now=moment)
        steady = engine.decay(config, now=moment)
        assert steady == [], "re-decay at the same instant moved confidence"

        assert [hit.id for hit in engine.retrieve(context, k=3)] == [rid]
        assert engine.get(rid).status == "promoted"
    finally:
        shutil.rmtree(root)
