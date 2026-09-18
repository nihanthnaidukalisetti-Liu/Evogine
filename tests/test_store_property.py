"""Property test: no path reaches retrievable without promotion.

Random walks over the full transition surface — including out-of-order and
invalid calls, which the store must refuse — checked against the invariant
after every single operation: a record may appear in ``retrieve`` only while
a successful ``promote`` call holds for it.
"""

import shutil
import tempfile
from pathlib import Path

from conftest import make_record
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.engine.records import LearningRecord, RecordCounterfactual
from evogine.engine.store import Engine, StoreError

TOPICS = ["429 backoff", "deadlock contention", "cache stampede"]
OPERATIONS = ["propose", "queue_replay", "promote", "demote"]


def _context_for(record_id: str) -> str:
    draft = make_record(record_id)
    return f"{draft.title} {draft.body.signal} {draft.body.failure_mode}"


@settings(max_examples=30, deadline=None)
@given(
    ops=st.lists(
        st.tuples(
            st.sampled_from(OPERATIONS),
            st.integers(min_value=0, max_value=len(TOPICS) - 1),
        ),
        max_size=10,
    )
)
def test_no_path_reaches_retrievable_without_promotion(ops):
    root = Path(tempfile.mkdtemp(prefix="evogine-store-prop-"))
    try:
        engine = Engine(root)
        currently_promoted: set[str] = set()
        ever_promoted: set[str] = set()

        for op, idx in ops:
            record_id = f"lr-prop-{idx}"
            try:
                if op == "propose":
                    engine.propose(make_record(record_id, topic=TOPICS[idx]))
                elif op == "queue_replay":
                    engine.queue_replay(record_id, [f"run-{idx}"])
                elif op == "promote":
                    engine.promote(
                        record_id,
                        RecordCounterfactual(
                            replay_run_ids=[f"run-{idx}"],
                            verdict_change=True,
                            evidence="runs/replay/property-walk.json",
                        ),
                    )
                    currently_promoted.add(record_id)
                    ever_promoted.add(record_id)
                elif op == "demote":
                    engine.demote(record_id, f"property-walk drop of {record_id}")
                    currently_promoted.discard(record_id)
            except StoreError:
                # Invalid transitions (duplicate propose, promote of an
                # ephemeral, demote of a dropped record, ...) must raise,
                # never mutate. The invariant below still holds.
                pass

            # The invariant, after every single operation: nothing that is
            # not actively promoted is ever retrievable.
            for topic_idx in range(len(TOPICS)):
                context = _context_for(f"lr-prop-{topic_idx}")
                for hit in engine.retrieve(context, k=len(TOPICS)):
                    assert hit.id in currently_promoted, (
                        f"{hit.id} retrieved with status {hit.status} "
                        "without an active promotion"
                    )

        # The mirror direction on-disk: the only route to status "promoted"
        # is a successful promote call — no transition can fake the state.
        for path in sorted((root / "records").glob("*.json")):
            record = LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))
            assert record.status in {"ephemeral", "candidate", "promoted", "dropped"}
            if record.status == "promoted":
                assert record.id in ever_promoted
    finally:
        shutil.rmtree(root)
