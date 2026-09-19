"""The runner: composes planning, the bounded pool, metering, and the sink.

``SwarmOrchestrator.run`` is the whole story of a run, in order:

1. Open (or resume) the run directory — ``complete.marker`` freezes it.
2. Read checkpoint markers; every already-done item emits a ``skipped``
   status event and is never re-executed (no duplicated spend).
3. With no work left, write the ``complete`` event and freeze.
4. Refuse before any call if the provider key is missing from the
   environment — a visible ``refusal`` event, fail-closed.
5. Spawn the bounded pool (bounds from the unified config). Every model
   call from every worker flows through the meter; when the per-run
   Decimal ceiling cannot fit the next call, the run refuses: a
   ``refusal`` event lands, in-flight items settle, no new work starts,
   and the run directory stays resumable.
6. Each item: ``Builder`` proposes a patch through the metered client; a
   wired verifier (the ops layer adapts the code gate) decides
   ``verified`` vs. rejection; any model or gate failure proposes a
   learning through the ``LearningSink`` — the store is never touched
   directly by this package.
7. All items settled: the ``complete`` event lands and the run freezes.

Run configuration comes from the unified config's ``ProfileSettings``:
worker bounds from ``workers``, the dollar ceiling from ``budget.
max_run_cost_usd``, and the key's variable name from ``provider.
api_key_env`` (resolved against the environment at run time — the config
never holds secrets).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.bench.rundirs import RunDirectory, open_run
from evogine.config import EvogineConfig
from evogine.engine.records import LearningRecord
from evogine.swarm.clients import ModelClient, ModelClientError
from evogine.swarm.orchestrator.checkpoint import CheckpointStore, ItemResult
from evogine.swarm.orchestrator.errors import MissingApiKeyRefused
from evogine.swarm.orchestrator.events import (
    CompleteEvent,
    EventLog,
    RefusalEvent,
    SpendEvent,
    StatusEvent,
)
from evogine.swarm.orchestrator.planning import WorkItem, WorkPlan
from evogine.swarm.orchestrator.pool import Scheduler
from evogine.swarm.orchestrator.spend import MeteredModelClient, SpendMeter, SpendPolicy
from evogine.swarm.roles import (
    Builder,
    KnowledgeSource,
    LearningSink,
    ProposedPatch,
    Reflector,
    ReflectorOutputError,
)

ItemVerifier = Callable[[ProposedPatch], bool]
"""The ops layer's verification seam: True keeps the item, False rejects it."""


class RunSettings(BaseModel):
    """A run's bounded pool, dollar ceiling, and key variable name.

    Construct directly for tests, or via :meth:`from_config` so the values
    flow from the unified config's named profile — the orchestrator never
    reads ``EvogineConfig`` fields scattered through its code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_workers: int = Field(ge=1)
    max_workers: int = Field(ge=1)
    spend: SpendPolicy
    api_key_env: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> RunSettings:
        if self.max_workers < self.min_workers:
            raise ValueError(
                f"max_workers ({self.max_workers}) must be >= min_workers ({self.min_workers})"
            )
        return self

    @classmethod
    def from_config(
        cls,
        config: EvogineConfig,
        *,
        profile: str | None = None,
        input_price_per_mtok_usd: Decimal,
        output_price_per_mtok_usd: Decimal,
        max_call_input_tokens: int,
    ) -> RunSettings:
        """Resolve pool bounds, ceiling, and key name from a config profile.

        Prices are the caller's declared per-million-token table — the
        config deliberately holds no provider pricing.
        """
        resolved = config.profile(profile)
        return cls(
            min_workers=resolved.workers.min_workers,
            max_workers=resolved.workers.max_workers,
            api_key_env=resolved.provider.api_key_env,
            spend=SpendPolicy(
                ceiling_usd=resolved.budget.max_run_cost_usd,
                input_price_per_mtok_usd=input_price_per_mtok_usd,
                output_price_per_mtok_usd=output_price_per_mtok_usd,
                max_call_input_tokens=max_call_input_tokens,
            ),
        )


class RunOutcome(BaseModel):
    """What a run did: its state, its counts, its exact spend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    outcome: Literal["completed", "refused", "failed"]
    detail: str = ""
    items_total: int = Field(ge=0)
    items_succeeded: int = Field(ge=0)
    items_failed: int = Field(ge=0)
    resumed_items: int = Field(ge=0)
    workers_spawned: int = Field(ge=0)
    spent_usd: Decimal = Field(ge=0)
    ceiling_usd: Decimal = Field(gt=0)
    items: list[ItemResult] = Field(default_factory=list)


class _EmptyKnowledge:
    """The honest empty state: no knowledge source wired, none pretended."""

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        return []


def resolve_api_key(env: Mapping[str, str], api_key_env: str) -> str:
    """Read the provider key from the environment, refusing when absent."""
    key = env.get(api_key_env, "")
    if not key.strip():
        raise MissingApiKeyRefused(
            f"environment variable {api_key_env!r} is not set — refusing to "
            "run without provider credentials"
        )
    return key


class SwarmOrchestrator:
    """Executes a :class:`WorkPlan` under bounds, ceilings, and the sink.

    Constructor seams (all keyword-only, all fakeable offline):

    - ``client`` — the provider seam; every worker wraps it in a
      :class:`MeteredModelClient`, so the ceiling is unskippable.
    - ``sink`` — the **only** store path. Learnings reach the engine
      exclusively through ``LearningSink.propose``; this package has no
      other import of the store and never writes it directly.
    - ``knowledge`` — optional promoted-only retrieval for builders;
      absent means an honest empty retrieval, not a shortcut.
    - ``verify`` — optional item verifier (the code gate's adapter).
      ``None`` means items complete ``unverified`` — recorded honestly.
    """

    def __init__(
        self,
        *,
        settings: RunSettings,
        client: ModelClient,
        sink: LearningSink,
        knowledge: KnowledgeSource | None = None,
        verify: ItemVerifier | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sink = sink
        self._knowledge: KnowledgeSource = knowledge if knowledge is not None else _EmptyKnowledge()
        self._verify = verify
        self._env: Mapping[str, str] = env if env is not None else os.environ

    def run(
        self,
        plan: WorkPlan,
        runs_root: Path,
        *,
        resume_run_id: str | None = None,
    ) -> RunOutcome:
        """Run (or resume) ``plan`` in a fresh-or-resumed run directory."""
        run = open_run(runs_root, resume_run_id)
        store = CheckpointStore(run)
        log = EventLog(run)
        done = store.completed()

        resumed = {item.item_id for item in plan.items if item.item_id in done}
        for item in plan.items:
            if item.item_id in done:
                log.append(
                    StatusEvent(
                        worker_id=None,
                        item_id=item.item_id,
                        status="skipped",
                        detail="resumed from checkpoint marker; not re-executed",
                    )
                )

        ceiling = self._settings.spend.ceiling_usd
        initial_spend = sum((result.spend_usd for result in done.values()), Decimal(0))

        def record_spend(event: SpendEvent) -> None:
            log.append(event)

        meter = SpendMeter(
            self._settings.spend,
            initial_spent_usd=initial_spend,
            on_event=record_spend,
        )
        pending = [item for item in plan.items if item.item_id not in done]

        if pending and self._key_missing():
            detail = (
                f"environment variable {self._settings.api_key_env!r} is not set — "
                "refusing to run without provider credentials"
            )
            log.append(
                RefusalEvent(
                    reason="missing_api_key",
                    detail=detail,
                    spent_usd=meter.spent_usd,
                    projected_usd=None,
                    ceiling_usd=ceiling,
                )
            )
            return self._outcome(
                run,
                "refused",
                items_total=len(plan.items),
                results=self._merged(plan, done, {}),
                resumed_items=len(resumed),
                workers_spawned=0,
                spent=meter.spent_usd,
                ceiling=ceiling,
                detail=detail,
            )

        if not pending:
            log.append(
                CompleteEvent(
                    items_total=len(plan.items),
                    items_completed=len(plan.items),
                    resumed_items=len(resumed),
                    spent_usd=meter.spent_usd,
                    ceiling_usd=ceiling,
                )
            )
            run.mark_complete()
            return self._outcome(
                run,
                "completed",
                items_total=len(plan.items),
                results=self._merged(plan, done, {}),
                resumed_items=len(resumed),
                workers_spawned=0,
                spent=meter.spent_usd,
                ceiling=ceiling,
            )

        def work_for(item: WorkItem, worker_id: int) -> ItemResult:
            result = self._execute_item(meter, log, item, worker_id)
            # Every settled item gets its resume marker: a resumed run never
            # re-bills an item whose attempt budget is exhausted. The refusal
            # path raises past this line and leaves the item unmarked — the
            # one thing a resume must re-attempt.
            store.mark_completed(result)
            return result

        scheduler = Scheduler(
            plan,
            run,
            log,
            min_workers=self._settings.min_workers,
            max_workers=self._settings.max_workers,
            work_fn=work_for,
        )
        results = scheduler.run(frozenset(done))
        if scheduler.refusal is not None:
            refusal = scheduler.refusal
            log.append(
                RefusalEvent(
                    reason="spend_ceiling",
                    detail=str(refusal),
                    spent_usd=meter.spent_usd,
                    projected_usd=None,
                    ceiling_usd=ceiling,
                )
            )
            not_run = sum(
                1 for item in plan.items if item.item_id not in done and item.item_id not in results
            )
            return self._outcome(
                run,
                "refused",
                items_total=len(plan.items),
                results=self._merged(plan, done, results),
                resumed_items=len(resumed),
                workers_spawned=scheduler.spawned,
                spent=meter.spent_usd,
                ceiling=ceiling,
                detail=(
                    f"spend ceiling refused before {not_run} remaining item(s) "
                    "could run; the run directory is resumable"
                ),
            )

        log.append(
            CompleteEvent(
                items_total=len(plan.items),
                items_completed=sum(
                    1 for r in results.values() if r.outcome in ("verified", "unverified")
                ),
                resumed_items=len(resumed),
                spent_usd=meter.spent_usd,
                ceiling_usd=ceiling,
            )
        )
        run.mark_complete()
        any_failed = any(r.outcome == "failed" for r in results.values())
        return self._outcome(
            run,
            "failed" if any_failed else "completed",
            items_total=len(plan.items),
            results=self._merged(plan, done, results),
            resumed_items=len(resumed),
            workers_spawned=scheduler.spawned,
            spent=meter.spent_usd,
            ceiling=ceiling,
        )

    def _key_missing(self) -> bool:
        try:
            resolve_api_key(self._env, self._settings.api_key_env)
        except MissingApiKeyRefused:
            return True
        return False

    def _merged(
        self,
        plan: WorkPlan,
        done: dict[str, ItemResult],
        results: dict[str, ItemResult],
    ) -> list[ItemResult]:
        merged = {**done, **results}
        return [merged[item.item_id] for item in plan.items if item.item_id in merged]

    def _outcome(
        self,
        run: RunDirectory,
        outcome: Literal["completed", "refused", "failed"],
        *,
        items_total: int,
        results: list[ItemResult],
        resumed_items: int,
        workers_spawned: int,
        spent: Decimal,
        ceiling: Decimal,
        detail: str = "",
    ) -> RunOutcome:
        return RunOutcome(
            run_id=run.run_id,
            outcome=outcome,
            detail=detail,
            items_total=items_total,
            items_succeeded=sum(1 for r in results if r.outcome in ("verified", "unverified")),
            items_failed=sum(1 for r in results if r.outcome == "failed"),
            resumed_items=resumed_items,
            workers_spawned=workers_spawned,
            spent_usd=spent,
            ceiling_usd=ceiling,
            items=results,
        )

    def _execute_item(
        self,
        meter: SpendMeter,
        log: EventLog,
        item: WorkItem,
        worker_id: int,
    ) -> ItemResult:
        client = MeteredModelClient(meter, self._client, worker_id=worker_id)
        client.set_item(item.item_id)
        log.append(
            StatusEvent(
                worker_id=worker_id,
                item_id=item.item_id,
                status="started",
                detail=f"criticality {item.criticality:.3f}, tier {item.review_tier}",
            )
        )
        try:
            patch = Builder(client=client, engine=self._knowledge).propose_patch(
                _task_context(item)
            )
        except ModelClientError as exc:
            learning_id, learn_note = self._propose_learning(
                client, item, f"model calls failed on item {item.item_id}: {exc}"
            )
            outcome: Literal["learning_proposed", "failed"] = (
                "learning_proposed" if learning_id else "failed"
            )
            log.append(
                StatusEvent(
                    worker_id=worker_id,
                    item_id=item.item_id,
                    status="failed",
                    detail=f"model failure: {exc}; {learn_note}".strip(),
                )
            )
            return ItemResult(
                item_id=item.item_id,
                modules=list(item.modules),
                outcome=outcome,
                worker_id=worker_id,
                calls=client.call_count,
                spend_usd=client.item_spent_usd,
                learning_id=learning_id,
                detail=f"model failure: {exc}; {learn_note}".strip(),
            )

        if self._verify is None:
            log.append(
                StatusEvent(
                    worker_id=worker_id,
                    item_id=item.item_id,
                    status="completed",
                    detail="patch proposed; no verifier wired — unverified",
                )
            )
            return ItemResult(
                item_id=item.item_id,
                modules=list(item.modules),
                outcome="unverified",
                worker_id=worker_id,
                calls=client.call_count,
                spend_usd=client.item_spent_usd,
                detail="no verifier wired; patch recorded without a gate verdict",
            )

        try:
            passed = self._verify(patch)
        except Exception as exc:  # noqa: BLE001 - a broken gate is an item detail, not a crash
            log.append(
                StatusEvent(
                    worker_id=worker_id,
                    item_id=item.item_id,
                    status="completed",
                    detail=f"verifier error: {exc}",
                )
            )
            return ItemResult(
                item_id=item.item_id,
                modules=list(item.modules),
                outcome="unverified",
                worker_id=worker_id,
                calls=client.call_count,
                spend_usd=client.item_spent_usd,
                detail=f"verifier raised {type(exc).__name__}: {exc}",
            )
        if passed:
            log.append(
                StatusEvent(
                    worker_id=worker_id,
                    item_id=item.item_id,
                    status="completed",
                    detail="patch proposed and verified",
                )
            )
            return ItemResult(
                item_id=item.item_id,
                modules=list(item.modules),
                outcome="verified",
                worker_id=worker_id,
                calls=client.call_count,
                spend_usd=client.item_spent_usd,
            )

        learning_id, learn_note = self._propose_learning(
            client, item, f"verifier rejected the proposed patch for {item.module}"
        )
        outcome = "learning_proposed" if learning_id else "failed"
        detail = f"verifier rejected patch; {learn_note}".strip("; ")
        log.append(
            StatusEvent(
                worker_id=worker_id,
                item_id=item.item_id,
                status="failed" if outcome == "failed" else "completed",
                detail=detail,
            )
        )
        return ItemResult(
            item_id=item.item_id,
            modules=list(item.modules),
            outcome=outcome,
            worker_id=worker_id,
            calls=client.call_count,
            spend_usd=client.item_spent_usd,
            learning_id=learning_id,
            detail=detail,
        )

    def _propose_learning(
        self, client: MeteredModelClient, item: WorkItem, failure_context: str
    ) -> tuple[str | None, str]:
        """Propose a learning through the sink; the only store path there is.

        A learning that cannot be proposed (its model call fails, or the
        model's output is not a valid draft) is recorded in the item's
        detail — never silently dropped, never allowed to crash the pool.
        """
        try:
            record = Reflector(client=client, engine=self._sink).propose_learning(failure_context)
        except (ModelClientError, ReflectorOutputError) as exc:
            return None, f"learning proposal failed: {type(exc).__name__}: {exc}"
        return record.id, f"learning {record.id} proposed through the sink"


def _task_context(item: WorkItem) -> str:
    tests = [module for module in item.modules if module != item.module]
    covered = ", ".join(tests) if tests else "none recorded"
    return (
        f"Repair or implement module {item.module} in the Evogine repository. "
        f"Work item {item.item_id} (criticality {item.criticality:.3f}, "
        f"review tier {item.review_tier}). Covering test modules: {covered}."
    )


__all__ = [
    "ItemVerifier",
    "RunOutcome",
    "RunSettings",
    "SwarmOrchestrator",
    "resolve_api_key",
]
