"""Bounded dynamic spawning: the worker pool and its spawn policy.

Bounds come from the unified config (``WorkerPoolSettings.min_workers`` /
``max_workers``) and are enforced structurally:

- The run spawns **once**, up front, with
  :func:`compute_pool_size` — a pure function of plan size and criticality
  concentration, clamped to ``[min_workers, max_workers]``. Concurrency
  can therefore never exceed the configured maximum under any module
  count, and every item runs on one of the spawned workers.
- Workers **never grow** the pool. A worker that finds nothing ready
  waits on a condition variable and is woken by completions; it exits
  only when every item is settled. (A worker finishing its last item
  ends the run's concurrency — the min bound is the operator's floor for
  spawn count, not a guarantee that idle threads persist.)

Dependencies come from the plan. Python really does contain import
cycles, so when every pending item is blocked and none is running, the
scheduler designates the highest-priority pending item a **cycle
breaker** and runs it liveness-first: starvation is impossible, and the
breaker designation is visible in the returned stats. Without a cycle the
breaker logic never triggers — every item then runs strictly after its
dependencies complete.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

from evogine.bench.rundirs import RunDirectory
from evogine.swarm.orchestrator.checkpoint import ItemResult
from evogine.swarm.orchestrator.errors import OrchestratorRefused
from evogine.swarm.orchestrator.events import EventLog, SpawnEvent
from evogine.swarm.orchestrator.planning import WorkItem, WorkPlan

AVERAGE_ITEMS_PER_WORKER = 4
"""Plan size per worker: 40 items ask for a 10-worker pool (before clamping)."""


def compute_pool_size(
    total_items: int,
    *,
    min_workers: int,
    max_workers: int,
    tight_share: float,
) -> int:
    """The spawn count for a run: scaled by size and criticality, bounded.

    ``total_items`` drives the pool up (``AVERAGE_ITEMS_PER_WORKER`` items
    per worker); ``tight_share`` — the fraction of items in the tight
    review tier — pushes a criticality-concentrated plan toward its
    midpoint so the most-reviewed work spreads across more workers. The
    result is always within ``[min_workers, max_workers]`` (0 only for an
    empty plan), whatever the module count.
    """
    if min_workers < 1:
        raise ValueError(f"min_workers must be >= 1, got {min_workers}")
    if max_workers < min_workers:
        raise ValueError(f"max_workers ({max_workers}) must be >= min_workers ({min_workers})")
    if total_items <= 0:
        return 0
    size_for_items = math.ceil(total_items / AVERAGE_ITEMS_PER_WORKER)
    midpoint = (min_workers + max_workers) // 2
    size_for_criticality = midpoint if tight_share >= 0.5 else min_workers
    return min(max_workers, max(min_workers, size_for_items, size_for_criticality))


class _ReadyQueue:
    """Dependency-aware take/complete queue over a plan's items.

    Not thread-safe by itself: the scheduler guards it with its condition
    variable's lock.
    """

    def __init__(self, plan: WorkPlan, completed_ids: frozenset[str]) -> None:
        self._order = plan.items
        self._by_id = {item.item_id: item for item in plan.items}
        self._done: set[str] = set(completed_ids)
        self._failed: set[str] = set()
        self._running: set[str] = set()
        self._breakers: set[str] = set()

    @property
    def total(self) -> int:
        return len(self._order)

    @property
    def breakers(self) -> frozenset[str]:
        return frozenset(self._breakers)

    def settled_count(self) -> int:
        return len(self._done) + len(self._failed)

    def all_settled(self) -> bool:
        return self.settled_count() + len(self._running) == len(self._order)

    def _unblocked(self, item: WorkItem) -> bool:
        return all(
            dep in self._done or dep in self._failed or dep in self._breakers
            for dep in item.dependencies
        )

    def _take_locked(self) -> WorkItem | None:
        for item in self._order:
            if (
                item.item_id in self._done
                or item.item_id in self._failed
                or item.item_id in self._running
            ):
                continue
            if item.item_id in self._breakers or self._unblocked(item):
                self._running.add(item.item_id)
                return item
        return None

    def take(self) -> WorkItem | None:
        """Claim the next runnable item, or ``None`` when nothing is ready."""
        return self._take_locked()

    def break_cycle_if_stuck(self) -> str | None:
        """Designate the top-priority blocked item a cycle breaker.

        Called when no item is takeable, not everything is settled, and no
        breaker is pending — the signature of an import cycle among the
        remaining items. Returns the designated item id, or ``None`` when
        a breaker is already designated or nothing remains.
        """
        if self._breakers:
            return None
        for item in self._order:
            if (
                item.item_id in self._done
                or item.item_id in self._failed
                or item.item_id in self._running
            ):
                continue
            if self._unblocked(item):
                return None  # something is runnable; not stuck
            self._breakers.add(item.item_id)
            return item.item_id
        return None

    def complete(self, item_id: str, failed: bool) -> None:
        self._running.discard(item_id)
        (self._failed if failed else self._done).add(item_id)


class Scheduler:
    """Runs a plan on a bounded pool of worker threads; returns per-item results.

    ``work_fn`` is the runner's per-item body — it owns the model client,
    metering, events, and checkpointing, and returns the item's
    :class:`ItemResult`. It runs on the worker thread; a raised exception
    fails that item (recorded with its text, never swallowed) and the pool
    moves on.
    """

    def __init__(
        self,
        plan: WorkPlan,
        run: RunDirectory,
        log: EventLog,
        *,
        min_workers: int,
        max_workers: int,
        work_fn: Callable[[WorkItem, int], ItemResult],
    ) -> None:
        self._plan = plan
        self._run = run
        self._log = log
        self._min = min_workers
        self._max = max_workers
        self._work_fn = work_fn
        self._cond = threading.Condition()
        self._results: dict[str, ItemResult] = {}
        self._spawned = 0
        self._refusal: OrchestratorRefused | None = None

    @property
    def spawned(self) -> int:
        """How many workers the run spawned (the bounded number)."""
        return self._spawned

    @property
    def results(self) -> dict[str, ItemResult]:
        return dict(self._results)

    @property
    def refusal(self) -> OrchestratorRefused | None:
        """The run-level refusal that stopped the pool, if one occurred."""
        return self._refusal

    def run(self, completed_ids: frozenset[str]) -> dict[str, ItemResult]:
        """Spawn the bounded pool and drive every item to a settled state."""
        plan_size = len(self._plan.items)
        tight_share = self._plan.tight_count / plan_size if plan_size else 0.0
        pool_size = compute_pool_size(
            plan_size,
            min_workers=self._min,
            max_workers=self._max,
            tight_share=tight_share,
        )
        if pool_size == 0:
            return {}
        queue = _ReadyQueue(self._plan, completed_ids)
        threads: list[threading.Thread] = []
        for worker_id in range(pool_size):
            self._log.append(SpawnEvent(worker_id=worker_id, pool_size=pool_size))
            thread = threading.Thread(
                target=self._worker_loop,
                args=(queue, worker_id),
                name=f"evogine-worker-{worker_id}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            self._spawned += 1
        for thread in threads:
            thread.join()
        return self.results

    def _worker_loop(self, queue: _ReadyQueue, worker_id: int) -> None:
        while True:
            with self._cond:
                if self._refusal is not None:
                    return  # the run was refused; no worker takes new work
                item = queue.take()
                while item is None:
                    if queue.all_settled():
                        return
                    # A fresh designation skips the wait: the designating
                    # worker must take the breaker itself, or a pool where
                    # every worker sleeps (pool_size == 1) waits forever.
                    if queue.break_cycle_if_stuck() is None:
                        self._cond.wait()
                    if self._refusal is not None:
                        return
                    item = queue.take()
            try:
                result = self._work_fn(item, worker_id)
            except OrchestratorRefused as exc:
                # A fail-closed refusal (spend ceiling) stops the whole run:
                # record it, release every waiter, spawn no further work.
                with self._cond:
                    if self._refusal is None:
                        self._refusal = exc
                    queue.complete(item.item_id, failed=True)
                    self._results[item.item_id] = ItemResult(
                        item_id=item.item_id,
                        modules=list(item.modules),
                        outcome="failed",
                        worker_id=worker_id,
                        detail=f"run refused before this item completed: {exc}",
                    )
                    self._cond.notify_all()
                return
            except Exception as exc:  # noqa: BLE001 - one item's failure is recorded, the pool moves on
                result = ItemResult(
                    item_id=item.item_id,
                    modules=list(item.modules),
                    outcome="failed",
                    worker_id=worker_id,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                with self._cond:
                    queue.complete(item.item_id, failed=True)
                    self._results[item.item_id] = result
                    self._cond.notify_all()
                continue
            with self._cond:
                queue.complete(item.item_id, result.outcome == "failed")
                self._results[item.item_id] = result
                self._cond.notify_all()


__all__ = ["AVERAGE_ITEMS_PER_WORKER", "Scheduler", "compute_pool_size"]
