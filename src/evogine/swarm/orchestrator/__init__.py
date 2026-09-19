"""Swarm orchestrator: bounded dynamic spawning over a decomposition (V4).

Consumes the decomposition's :class:`~evogine.swarm.decomposition.
DecompositionDAG` and executes it as criticality-ordered, PR-sized work
items on a bounded worker pool. Every model call is metered against the
unified config's exact-``Decimal`` dollar ceiling (fail-closed — a
reservation that cannot fit refuses the run before the call is made), run
state persists as immutable per-item resume markers (a killed run resumes
from its last checkpoint with no duplicated spend), and worker proposals
reach the engine's store exclusively through the ``LearningSink``.

The run's typed Pydantic events (spawn, status, spend, complete, refusal)
land as JSONL in the run directory — the mission-control dashboard's
contract, documented in :mod:`evogine.swarm.orchestrator.events`.
"""

from __future__ import annotations

from evogine.swarm.orchestrator.checkpoint import (
    MARKER_PREFIX,
    CheckpointStore,
    ItemResult,
    marker_name,
)
from evogine.swarm.orchestrator.errors import (
    CheckpointError,
    EventLogError,
    MissingApiKeyRefused,
    OrchestratorError,
    OrchestratorRefused,
)
from evogine.swarm.orchestrator.events import (
    CompleteEvent,
    EventLog,
    RefusalEvent,
    RunEvent,
    SpawnEvent,
    SpendEvent,
    StatusEvent,
    read_events,
)
from evogine.swarm.orchestrator.planning import (
    TIGHT_FRACTION,
    WorkItem,
    WorkPlan,
    item_id_for,
    plan_work,
)
from evogine.swarm.orchestrator.pool import (
    AVERAGE_ITEMS_PER_WORKER,
    Scheduler,
    compute_pool_size,
)
from evogine.swarm.orchestrator.runner import (
    ItemVerifier,
    RunOutcome,
    RunSettings,
    SwarmOrchestrator,
    resolve_api_key,
)
from evogine.swarm.orchestrator.spend import (
    MeteredModelClient,
    SpendCeilingRefused,
    SpendMeter,
    SpendPolicy,
)

__all__ = [
    "AVERAGE_ITEMS_PER_WORKER",
    "MARKER_PREFIX",
    "TIGHT_FRACTION",
    "CheckpointError",
    "CheckpointStore",
    "CompleteEvent",
    "EventLog",
    "EventLogError",
    "ItemResult",
    "ItemVerifier",
    "MeteredModelClient",
    "MissingApiKeyRefused",
    "OrchestratorError",
    "OrchestratorRefused",
    "RefusalEvent",
    "RunEvent",
    "RunOutcome",
    "RunSettings",
    "Scheduler",
    "SpawnEvent",
    "SpendCeilingRefused",
    "SpendEvent",
    "SpendMeter",
    "SpendPolicy",
    "StatusEvent",
    "SwarmOrchestrator",
    "WorkItem",
    "WorkPlan",
    "compute_pool_size",
    "item_id_for",
    "marker_name",
    "plan_work",
    "read_events",
    "resolve_api_key",
]
