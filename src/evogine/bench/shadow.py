"""V3 W2: the shadow pass — profile candidate configurations without mutating.

A shadow pass answers "how would retrieval and prompting behave under this
candidate configuration?" over the case corpus while touching nothing the
engine's warm surface can ever see. It is the V3 spec's core non-negotiable:
**shadow evaluation must never be able to change what warm retrieval sees.**

The guarantee is structural, in layers — not a comment:

- The orchestrator's only handle on the store is :class:`ShadowStore`, a
  read-only protocol. Every lifecycle mutator (``promote``, ``demote``,
  ``merge``, ``decay``, ``retire``, ...) is absent from that type, so a
  call to one is a mypy-strict type error, not a discipline.
- Evaluation reads a materialized copy of the promoted pool taken once at
  pass start (:class:`OverlayRetriever`), so it cannot even observe store
  changes mid-pass, let alone cause them.
- Results go only into ``runs/shadow/<candidate_id>/`` under the PR #11
  run-dir conventions (resume by candidate id, ``complete.marker``
  immutability). Warm retrieval draws from ``promoted_records()`` — the
  store's ``records/`` glob — so nothing under ``runs/`` can ever enter
  the pool (the governance record, ``docs/cross-project-memory.md`` §1.1,
  is why the store's single flat pool makes that the load-bearing line).

Both refusals run before any evaluation: the contamination recall probe
against the pool as the shadow run would serve it, and the dollar-ceiling
projection (:class:`~evogine.bench.budgets.DollarBudget`). A contaminated
or over-ceiling shadow run exits non-zero having spent $0 and written
nothing anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.bench.budgets import DollarBudget, SpendCeilingExceededError, queue_spend
from evogine.bench.cases import BenchCase, default_corpus
from evogine.bench.harness import (
    REPEAT_FLOOR,
    BenchHarness,
    CaseReport,
    CaseRunner,
    RunArtifacts,
    RunSummary,
)
from evogine.bench.probes import ProbeReport, recall_probe
from evogine.bench.replay import ReplaySchedule, build_replay_requests
from evogine.bench.rundirs import RunDirectory, open_run
from evogine.engine.embedders import Embedder
from evogine.engine.records import LearningRecord
from evogine.engine.retrieval import DEFAULT_FLOOR
from evogine.engine.retrieval import retrieve as hybrid_retrieve
from evogine.engine.store import Engine

RUNS_SUBDIR = "shadow"
"""Where shadow results live: ``<runs_root>/<RUNS_SUBDIR>/<candidate_id>/``.

Store-root-relative, the ref is ``runs/shadow/<candidate_id>/`` — beside the
gate evidence at ``runs/gate/`` and the trajectories at ``runs/trajectory/``,
and invisible to warm retrieval, which reads ``records/`` only.
"""


@runtime_checkable
class ShadowStore(Protocol):
    """The read-only store surface a shadow pass may hold.

    The orchestrator sees the engine through this protocol and nothing
    else: two read members, no mutators. The static check below the
    protocol pins that the real store satisfies it structurally — the
    reverse (a shadow pass reaching a mutator) cannot typecheck.
    """

    embedder: Embedder
    """The store's own embedder: profiling must score with exactly what warm
    retrieval scores with, so the shadow numbers describe the served system."""

    def promoted_records(self) -> list[LearningRecord]:
        """Return the promoted records — the only pool retrieval may draw from."""
        ...


# Static protocol proof: mypy verifies the real store satisfies the read-only
# shadow seam at this assignment (the runtime twin lives in tests/test_shadow.py).
_protocol_check: type[ShadowStore] = Engine


class _PoolView:
    """Minimal promoted-pool view over a plain list, for hybrid scoring."""

    def __init__(self, records: list[LearningRecord]) -> None:
        self._records = list(records)

    def promoted_records(self) -> list[LearningRecord]:
        return list(self._records)


class OverlayRetriever:
    """Hybrid retrieval over a pool snapshot plus overlay records.

    The generalization of the V2 gate-replay overlay (the counterfactual
    warm arm): overlay records — the knowledge being profiled as if it were
    promoted — lead the ranking, deduplicated by id; remaining slots fill
    from the pool's hybrid-scored records under the configured floor. The
    pool is a plain list captured at construction, so evaluation reads a
    snapshot: it can neither observe nor cause store changes after that
    point.

    Overlay records are served regardless of the floor — they are the
    counterfactual being profiled, not pool members to be filtered; pool
    records below the configured floor never qualify, exactly as warm
    retrieval would refuse them.
    """

    def __init__(
        self,
        *,
        pool: list[LearningRecord],
        overlay: list[LearningRecord],
        embedder: Embedder | None = None,
        k: int = 5,
        floor: float = DEFAULT_FLOOR,
    ) -> None:
        self._pool = list(pool)
        self._embedder = embedder
        self._k = k
        self._floor = floor
        self._overlay: list[LearningRecord] = []
        seen: set[str] = set()
        for record in overlay:
            if record.id not in seen:
                seen.add(record.id)
                self._overlay.append(record)

    @property
    def overlay_record_ids(self) -> list[str]:
        """The overlay records actually served, in order, deduplicated."""
        return [record.id for record in self._overlay]

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Overlay-first retrieval; the configured width is authoritative.

        The harness calls with its default ``k``; a shadow pass profiles a
        pinned configuration, so the configured width — not the caller's —
        bounds every result.
        """
        if self._k <= 0:
            return []
        pool_hits = hybrid_retrieve(
            _PoolView(self._pool),
            task_context,
            k=self._k,
            floor=self._floor,
            embedder=self._embedder,
        )
        overlay_ids = {record.id for record in self._overlay}
        served = [*self._overlay, *(hit for hit in pool_hits if hit.id not in overlay_ids)]
        return served[: self._k]


class ShadowCandidate(BaseModel):
    """One candidate configuration to profile, loaded from a declaration file.

    The ``records`` are the hypothetical knowledge: overlaid as if promoted
    whatever lifecycle status they hold on disk — a dropped record profiled
    for resurrection is a legitimate candidate. The dollar budget is
    required (an ambient budget is the failure mode V2-4 exists to fix),
    and the optional ``heldout_family`` names the corpus family the recall
    probe must find fresh.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    """Becomes the shadow run id: the directory under ``runs/shadow/``."""
    records: list[LearningRecord] = Field(default_factory=list)
    k: int = Field(default=5, ge=1)
    """Retrieval width the shadow run pins."""
    floor: float = Field(default=DEFAULT_FLOOR, ge=0.0, le=1.0)
    """Relevance floor for pool records; overlay records are never floored."""
    budget: DollarBudget
    heldout_family: str | None = None
    """Corpus family the recall probe must find fresh; ``None`` probes none."""

    @model_validator(mode="after")
    def _unique_overlay_ids(self) -> ShadowCandidate:
        ids = [record.id for record in self.records]
        duplicates = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate overlay record ids: {duplicates}")
        return self


class ShadowRefusalError(Exception):
    """Pre-evaluation refusal: nothing evaluated, nothing written, $0 spent."""


class ShadowContaminationRefused(ShadowRefusalError):
    """The recall probe found the pool quoting held-out-unique content."""

    def __init__(self, report: ProbeReport) -> None:
        self.report = report
        finding = report.findings[0]
        super().__init__(
            f"shadow run refused: record {finding.record_id!r} quotes held-out tokens "
            f"{finding.matched_tokens} — the pool is not fresh against the declared "
            "holdout (nothing was evaluated, nothing was written, $0 spent)"
        )


class ShadowBudgetRefused(ShadowRefusalError):
    """The projected spend exceeds the declared dollar ceiling."""

    def __init__(self, projected: Decimal, allowed: Decimal, replays: int) -> None:
        self.projected = projected
        self.allowed = allowed
        self.replays = replays
        super().__init__(
            f"shadow run refused: {replays} replays project ${projected} against "
            f"a ${allowed} ceiling (nothing was evaluated, nothing was written, $0 spent)"
        )


class ShadowReport(BaseModel):
    """What a completed shadow pass found, persisted inside its run directory."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    k: int
    floor: float
    overlay_record_ids: list[str]
    contamination: ProbeReport
    replays: int
    projected_cost_usd: Decimal
    ceiling_usd: Decimal
    summary: RunSummary
    cases: list[CaseReport]
    run_ref: str
    """Store-root-relative ref: ``runs/shadow/<candidate_id>`` — the only
    place this pass's results exist."""


def load_shadow_candidate(path: str | Path) -> ShadowCandidate:
    """Parse and validate a shadow candidate from a YAML (or JSON) file.

    Mirrors :func:`evogine.bench.budgets.load_plan`: unknown fields are
    refused (config drift), and the declared dollar budget is required.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ShadowCandidate.model_validate(data)


def run_shadow_pass(
    store: ShadowStore,
    candidate: ShadowCandidate,
    runner: CaseRunner,
    *,
    runs_root: Path,
    heldout_cases: Sequence[BenchCase],
    corpus: Sequence[BenchCase] | None = None,
    repeats: int = REPEAT_FLOOR,
) -> ShadowReport:
    """Profile ``candidate`` over the corpus; write nothing outside the run dir.

    Order is the contract: recall probe, then dollar projection, then —
    only after both pass — the run directory opens and any evaluation is
    spent. A refusal raises before ``open_run`` touches the filesystem, so
    a refused pass is not even an empty directory, and the store never
    changes (the type cannot; :class:`ShadowStore` has no mutators).
    """
    RunDirectory(runs_root, candidate.candidate_id)  # pure id validation, no writes
    cases = list(corpus) if corpus is not None else default_corpus()
    pool = list(store.promoted_records())
    overlay = OverlayRetriever(
        pool=pool,
        overlay=candidate.records,
        embedder=store.embedder,
        k=candidate.k,
        floor=candidate.floor,
    )

    # The probe saturates — every pool and overlay record, no relevance floor —
    # because shadow evidence must survive a later k change: a record quoting
    # the holdout is refused even when the candidate's own retrieval width
    # would never serve it.
    saturation = OverlayRetriever(
        pool=pool,
        overlay=candidate.records,
        embedder=store.embedder,
        k=max(1, len(pool) + len(candidate.records)),
        floor=0.0,
    )

    probe = recall_probe(saturation, heldout_cases, corpus=cases)
    if probe.contaminated:
        raise ShadowContaminationRefused(probe)

    schedule = ReplaySchedule(requests=build_replay_requests(cases, repeats))
    try:
        projected = queue_spend(schedule, candidate.budget)
    except SpendCeilingExceededError as exc:
        raise ShadowBudgetRefused(exc.projected, exc.allowed, exc.replays) from exc

    run = open_run(runs_root, run_id=candidate.candidate_id)
    report = BenchHarness(overlay, runner).run(
        corpus=cases, repeats=repeats, artifacts=RunArtifacts(run)
    )

    shadow = ShadowReport(
        candidate_id=candidate.candidate_id,
        k=candidate.k,
        floor=candidate.floor,
        overlay_record_ids=overlay.overlay_record_ids,
        contamination=probe,
        replays=report.summary.replays,
        projected_cost_usd=projected,
        ceiling_usd=candidate.budget.max_run_cost_usd,
        summary=report.summary,
        cases=report.cases,
        run_ref=f"runs/{RUNS_SUBDIR}/{run.run_id}",
    )
    run.write_artifact("report", shadow.model_dump_json(indent=2) + "\n")
    run.mark_complete()
    return shadow


__all__ = [
    "RUNS_SUBDIR",
    "OverlayRetriever",
    "ShadowBudgetRefused",
    "ShadowCandidate",
    "ShadowContaminationRefused",
    "ShadowRefusalError",
    "ShadowReport",
    "ShadowStore",
    "load_shadow_candidate",
    "run_shadow_pass",
]
