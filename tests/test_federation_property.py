"""Property tests: the four federation invariants (governance §3), structurally.

docs/cross-project-memory.md §3 names four invariants any federation must
preserve. Each is checked here as a structural property over generated
federations and random cross-scope operations, against an independent
restatement of the rule — never by re-calling the code under test:

- **§3.1 Scoping is enforced at the storage/retrieval layer** — every row
  any router read returns physically lives in the addressed store's own
  records directory, and a local store refuses to serve another store's
  qualified id.
- **§3.2 Promotion-only, per scope** — a scope's warm retrieval draws only
  from that scope's own promoted pool: identical knowledge sitting
  unpromoted in the addressed scope never surfaces, and knowledge promoted
  in another scope cannot leak in (the explicit import flow is a separate
  later task; until it exists the boundary is absolute).
- **§3.3 Per-store append-only audit, no invented global ordering** —
  every store's trail grows only by append, its ``seq`` is its own
  contiguous 1..n clock (a global sequencer would renumber one store's
  events past the other's count), and an untouched store grows no trail.
- **§3.4 Qualified ids for cross-scope references** — the store's write
  path rejects every unqualified or self-qualified cross-scope reference
  at write time, a rejected write persists nothing, and a well-formed
  foreign reference is stored verbatim (provenance is never rewritten).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from conftest import make_record, replay_evidence
from hypothesis import given, settings
from hypothesis import strategies as st

from evogine.engine.federation import FederationRegistry, FederationRouter
from evogine.engine.records import LearningRecord, MergedSource
from evogine.engine.store import Engine, StoreError, split_qualified_id

SCOPES = ["alpha", "beta"]
TOPICS = ["429 backoff", "deadlock contention", "cache stampede"]

# Independent restatement of the store's record-id grammar (store.py locks
# ids to this before they can become filenames) — the §3.4 oracle below must
# not borrow the code under test's own pattern.
_ID_GRAMMAR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _fresh_federation(
    names: list[str],
) -> tuple[Path, FederationRegistry, dict[str, Engine], FederationRouter]:
    """A federation of freshly git-initialized stores under one temp parent."""
    parent = Path(tempfile.mkdtemp(prefix="evogine-federation-prop-"))
    registry = FederationRegistry()
    engines = {name: registry.register(name, parent / name) for name in names}
    return parent, registry, engines, FederationRouter(registry)


def _promote(engine: Engine, record_id: str, topic: str = "429 backoff") -> None:
    """Propose, queue, and promote one record through the store's own lifecycle."""
    engine.propose(make_record(record_id, topic=topic))
    engine.queue_replay(record_id, [f"run-{record_id}"])
    engine.promote(record_id, replay_evidence([f"run-{record_id}"]))


def _context_for(topic: str) -> str:
    """A task context phrased from the fixture record's own body fields."""
    draft = make_record("context-probe", topic=topic)
    return f"{draft.title} {draft.body.signal} {draft.body.failure_mode}"


def _persisted_state(root: Path) -> tuple[int, int]:
    """(record file count, audit line count) — a refused write changes neither."""
    records = list((root / "records").glob("*.json"))
    audit = root / "audit.jsonl"
    lines = audit.read_text(encoding="utf-8").splitlines() if audit.exists() else []
    return len(records), len(lines)


# --------------------------------------------- invariant 3.1: layer-enforced scoping


@settings(max_examples=15, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(SCOPES), st.sampled_from(TOPICS)), max_size=6))
def test_router_reads_never_cross_the_addressed_scope(ops: list[tuple[str, str]]) -> None:
    parent, _, engines, router = _fresh_federation(SCOPES)
    try:
        promoted: dict[str, set[str]] = {name: set() for name in SCOPES}
        for index, name in enumerate(SCOPES):
            record_id = f"lr-seed-{name}"
            _promote(engines[name], record_id, topic=TOPICS[index % len(TOPICS)])
            promoted[name].add(record_id)

        for scope, topic in ops:
            rows = router.retrieve(scope, topic, k=3)
            # Independent restatement: every returned row is one of THIS
            # scope's promoted ids (the oracle tracked above), and its JSON
            # physically lives in the addressed store's records directory —
            # no cross-store row can ride a retrieval result.
            assert {record.id for record in rows} <= promoted[scope]
            for record in rows:
                assert (engines[scope].root / "records" / f"{record.id}.json").exists()
            # Point reads obey the same boundary: the router dispatches a
            # qualified id to its own scope, and a local store refuses to
            # serve another store's qualified id.
            target = min(promoted[scope])
            assert router.get(scope, target).id == target
            assert router.resolve(f"{scope}:{target}").id == target
            foreign = next(name for name in SCOPES if name != scope)
            with pytest.raises(StoreError, match="outside this store"):
                engines[scope].get(f"{foreign}:{target}")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# ------------------------------------- invariant 3.2: promotion-only, per scope


@settings(max_examples=15, deadline=None)
@given(st.lists(st.sampled_from(TOPICS), min_size=1, max_size=3, unique=True))
def test_warm_retrieval_sees_only_the_own_scopes_promotions(topics: list[str]) -> None:
    parent, _, engines, router = _fresh_federation(SCOPES)
    try:
        # Identical knowledge proposed in BOTH scopes; promoted only in alpha.
        for index, topic in enumerate(topics):
            _promote(engines["alpha"], f"lr-a{index}", topic=topic)
            engines["beta"].propose(make_record(f"lr-b{index}", topic=topic))

        # The leak check, for every topic while beta's pool is still empty:
        # warm retrieval in beta must be honestly empty — same words, but its
        # gate proved nothing — and alpha's warm run draws only from alpha's
        # own promoted pool, its proven record surfacing for the matching
        # context. (Exact ranking is the scorer's contract, not this one:
        # fixture bodies share vocabulary, so several rows can clear the
        # floor.)
        for index, topic in enumerate(topics):
            context = _context_for(topic)
            assert router.retrieve("beta", context, k=5) == []
            assert engines["beta"].promoted_records() == []
            alpha_ids = [record.id for record in router.retrieve("alpha", context, k=5)]
            assert set(alpha_ids) <= {f"lr-a{other}" for other in range(len(topics))}
            assert f"lr-a{index}" in alpha_ids

        # The counterfactual: once beta's own gate promotes its twins
        # (already proposed — queue and promote), beta's pool answers from
        # its own rows — proving the empty results above were
        # promotion-state-driven, not text-driven.
        for other in range(len(topics)):
            engines["beta"].queue_replay(f"lr-b{other}", [f"run-lr-b{other}"])
            engines["beta"].promote(f"lr-b{other}", replay_evidence([f"run-lr-b{other}"]))
        for index, topic in enumerate(topics):
            context = _context_for(topic)
            beta_ids = [record.id for record in router.retrieve("beta", context, k=5)]
            assert set(beta_ids) <= {f"lr-b{other}" for other in range(len(topics))}
            assert f"lr-b{index}" in beta_ids
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# ------------------- invariant 3.3: per-store append-only audit, no global order


@settings(max_examples=15, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from(SCOPES), st.sampled_from(["propose", "queue", "promote"])),
        max_size=8,
    )
)
def test_each_store_keeps_its_own_append_only_audit(ops: list[tuple[str, str]]) -> None:
    parent, _, engines, _router = _fresh_federation(SCOPES)
    try:
        seen: dict[str, list[str]] = {name: [] for name in SCOPES}
        touched: set[str] = set()

        for step, (scope, op) in enumerate(ops):
            record_id = f"lr-{step}"
            try:
                if op == "propose":
                    engines[scope].propose(make_record(record_id))
                elif op == "queue":
                    engines[scope].queue_replay(record_id, [f"run-{step}"])
                elif op == "promote":
                    engines[scope].promote(record_id, replay_evidence([f"run-{step}"]))
                touched.add(scope)
            except StoreError:
                pass  # invalid transition — must leave every trail untouched

            for name in SCOPES:
                trail = engines[name].audit_path
                lines = trail.read_text(encoding="utf-8").splitlines() if trail.exists() else []
                # Append-only: everything previously observed is an exact
                # prefix — no federation operation ever rewrites a trail.
                assert lines[: len(seen[name])] == seen[name]
                # Per-store clock: seq is that store's own contiguous 1..n.
                # A global sequencer would renumber one store's events past
                # the other store's count — this assert is the pin.
                seqs = [json.loads(line)["seq"] for line in lines]
                assert seqs == list(range(1, len(lines) + 1))
                seen[name] = lines

        # A scope that never saw a successful op never grew a trail: no
        # global clock dragged it along.
        for name in SCOPES:
            if name not in touched:
                assert not engines[name].audit_path.exists()
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# -------------------------- invariant 3.4: qualified ids for cross-scope references


def _reference_should_pass(reference: str, *, local: set[str], store: str) -> bool:
    """Independent restatement of the store's reference rule (invariant 3.4)."""
    if ":" in reference:
        try:
            qualified_store, _ = split_qualified_id(reference)
        except StoreError:
            return False  # qualified-shaped but malformed
        return qualified_store != store  # self-qualification is always refused
    return bool(_ID_GRAMMAR.fullmatch(reference)) and reference in local


def _draft_with_reference(record_id: str, reference: str) -> LearningRecord:
    draft = make_record(record_id)
    absorbed = make_record("absorbed")
    return draft.model_copy(
        update={
            "merged_sources": [
                MergedSource(
                    record_id=reference,
                    title=absorbed.title,
                    provenance=absorbed.provenance,
                    confidence=absorbed.confidence,
                    merged_at=absorbed.created,
                )
            ]
        }
    )


@settings(max_examples=50, deadline=None)
@given(
    reference=st.one_of(
        st.sampled_from(
            [
                "lr-local-1",
                "ghost",
                "",
                "lr:1",
                "alpha:lr-local-1",
                "ALPHA:ghost",
                "beta:lr-9",
                "BETA:lr-9",
                "beta:",
                ":lr-1",
                "beta:lr:1",
                "al pha:lr-1",
            ]
        ),
        st.builds(
            lambda store_name, record_id: f"{store_name}:{record_id}",
            st.sampled_from(["alpha", "beta", "zeta", ""]),
            st.sampled_from(["lr-local-1", "ghost", ""]),
        ),
    )
)
def test_cross_scope_references_are_rejected_unless_qualified(reference: str) -> None:
    parent = Path(tempfile.mkdtemp(prefix="evogine-fed-ref-prop-"))
    try:
        engine = Engine(parent / "alpha", name="alpha")
        engine.propose(make_record("lr-local-1"))
        before = _persisted_state(parent / "alpha")

        draft = _draft_with_reference("lr-new", reference)
        if _reference_should_pass(reference, local={"lr-local-1"}, store="alpha"):
            stored = engine.propose(draft)
            # A well-formed foreign reference is kept verbatim: normalization
            # decides identity, it never rewrites provenance.
            assert stored.merged_sources[0].record_id == reference
        else:
            with pytest.raises(StoreError):
                engine.propose(draft)
            # A rejected write persists nothing: no record file, no audit line.
            assert _persisted_state(parent / "alpha") == before
    finally:
        shutil.rmtree(parent, ignore_errors=True)
