"""V3 W2 property: a shadow pass cannot change what warm retrieval sees.

The acceptance property, generalized across the configuration space: for
arbitrary candidate overlays — records in any lifecycle state, ids that
collide with store records, bodies that may quote the held-out family —
over a store holding every lifecycle state, the shadow pass either
completes with evidence confined to ``runs/shadow/<candidate_id>/`` or
refuses before spending, and in BOTH cases the engine's state is
byte-identical afterwards: same record files, same audit trail, same
promoted pool, same warm retrieval results.

Offline by contract; each example builds its own throwaway store.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from test_shadow import (
    HELDOUT_TOKEN,
    CountingRunner,
    engine_snapshot,
    free_budget,
    live_record,
    shadow_corpus,
    shadow_holdout,
    store_with_lifecycle_states,
)

from evogine.bench.harness import REPEAT_FLOOR
from evogine.bench.rundirs import RunCompleteError
from evogine.bench.shadow import (
    RUNS_SUBDIR,
    ShadowCandidate,
    ShadowRefusalError,
    run_shadow_pass,
)
from evogine.engine.store import Engine

TOTAL_REPLAYS = 2 * len(shadow_corpus()) * REPEAT_FLOOR
CANDIDATE_IDS = ("cand-a", "cand-b")

_LIFECYCLE_STATES = ("ephemeral", "candidate", "promoted", "dropped", "merged", "retired")
# Overlay ids that sometimes collide with real store records, sometimes not.
_RECORD_IDS = ("lr-live", "lr-decayed", "lr-candidate", "lr-ghost", "lr-shadow-1")

_overlay_specs = st.lists(
    st.fixed_dictionaries(
        {
            "record_id": st.sampled_from(_RECORD_IDS),
            "status": st.sampled_from(_LIFECYCLE_STATES),
            "word": st.sampled_from((HELDOUT_TOKEN, "jittered", "wal", "idempotent")),
            "leak": st.booleans(),
        }
    ),
    max_size=3,
    unique_by=lambda spec: spec["record_id"],
)


def _overlay_from(specs: list[dict[str, str | bool]]) -> list:
    return [
        live_record(
            str(spec["record_id"]),
            status=str(spec["status"]),
            topic=str(spec["word"]),
            leak=bool(spec["leak"]),
        )
        for spec in specs
    ]


def _confinement_check(runs_root: Path) -> None:
    """Every path under the shadow root belongs to one of this property's candidates."""
    for path in runs_root.rglob("*"):
        relative = path.relative_to(runs_root).as_posix()
        if relative in CANDIDATE_IDS:  # the candidate dir itself, not a result
            continue
        assert relative.startswith(
            tuple(f"{cid}/" for cid in CANDIDATE_IDS)
        ), f"shadow results leaked outside their run dirs: {relative}"


@settings(max_examples=15, deadline=None)
@given(
    overlay_specs=_overlay_specs,
    k=st.integers(min_value=1, max_value=5),
    floor=st.floats(min_value=0.0, max_value=1.0),
)
def test_shadow_pass_never_changes_what_warm_retrieval_sees(
    overlay_specs: list[dict[str, str | bool]], k: int, floor: float
) -> None:
    """Any overlay × any lifecycle store ⇒ byte-identical state after the pass."""
    workdir = Path(tempfile.mkdtemp(prefix="shadow-prop-"))
    try:
        store_root = workdir / "store"
        store_with_lifecycle_states(store_root)
        before = engine_snapshot(store_root)
        fresh_engine = Engine(str(store_root))
        runner = CountingRunner()
        runs_root = store_root / "runs" / RUNS_SUBDIR
        completed: list[str] = []

        def attempt(candidate: ShadowCandidate) -> None:
            try:
                run_shadow_pass(
                    fresh_engine,
                    candidate,
                    runner,
                    runs_root=runs_root,
                    heldout_cases=shadow_holdout(),
                    corpus=shadow_corpus(),
                    repeats=REPEAT_FLOOR,
                )
            except ShadowRefusalError:
                return
            completed.append(candidate.candidate_id)

        attempt(_candidate("cand-a", _overlay_from(overlay_specs), k, floor))
        # A second candidate accumulates alongside the first without touching it.
        attempt(_candidate("cand-b", [live_record("lr-b-1")], k, floor))
        # A completed run is immutable: rerunning cand-a must not re-evaluate.
        rerun_refused = False
        try:
            attempt(_candidate("cand-a", _overlay_from(overlay_specs), k, floor))
        except RunCompleteError:
            rerun_refused = True

        # 1. The engine — files, audit, promoted pool, warm retrieval — is untouched.
        assert engine_snapshot(store_root) == before

        # 2. Call accounting: only non-refused, non-complete passes evaluate.
        assert runner.calls == TOTAL_REPLAYS * len(completed)

        # 3. Run dirs exist exactly for completed candidates; nothing leaks.
        if "cand-a" not in completed and not rerun_refused:
            assert not (runs_root / "cand-a").exists()
        for candidate_id in CANDIDATE_IDS:
            if candidate_id in completed:
                assert (runs_root / candidate_id / "complete.marker").exists()
        if runs_root.exists():
            _confinement_check(runs_root)

        # 4. A leaking overlay is always refused before spend ($0, no evaluation).
        if any(spec["leak"] for spec in overlay_specs):
            assert "cand-a" not in completed
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _candidate(
    candidate_id: str, records: list, k: int, floor: float
) -> ShadowCandidate:
    return ShadowCandidate(
        candidate_id=candidate_id,
        records=records,
        k=k,
        floor=floor,
        budget=free_budget(),
    )
