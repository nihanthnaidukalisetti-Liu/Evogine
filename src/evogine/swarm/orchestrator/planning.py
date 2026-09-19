"""Planning: a DecompositionDAG becomes ordered, tiered, PR-sized work items.

Planning is pure — the DAG in, a :class:`WorkPlan` out, no I/O, no clock —
so the same decomposition always plans identically and property tests can
pin every invariant.

- **Partition**: every code module owns exactly one work item, sized one
  module (+ the test modules that exercise it) per item — bounded and
  PR-sized by construction, whatever the repository's shape. A test module
  nobody's ``tested_by`` lists stands alone as its own item.
- **Order**: items are canonically ordered by criticality (descending,
  item id breaking ties) — the priority the pool works in.
- **Dependencies**: a module's ``imports`` are its prerequisites
  (:mod:`evogine.swarm.decomposition.models`): an item waits on the items
  owning its import targets, so downstream modules never start before
  their dependencies complete. Import cycles exist in real Python; the
  scheduler breaks them liveness-first (see :mod:`evogine.swarm.orchestrator.pool`).
- **Review tier**: the top quarter of code items by criticality (ties
  included) is flagged ``tight`` — the highest-criticality modules get the
  tightest review, regardless of how the score distributes.
"""

from __future__ import annotations

import hashlib
import math
import re

from pydantic import BaseModel, ConfigDict, Field

from evogine.swarm.decomposition.models import DecompositionDAG, ModuleNode

TIGHT_FRACTION = 0.25
"""The share of code items (by criticality rank) flagged for tight review."""

_SAFE_SEGMENT = re.compile(r"[0-9A-Za-z][0-9A-Za-z_-]*\Z")
"""The path-segment shape run directories demand of marker names."""

_ID_HASH_WIDTH = 8


class WorkItem(BaseModel):
    """One bounded unit of swarm work: a module, its tests, its priority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    """Segment-safe id (derived from the owning module — see module docstring)."""
    module: str
    """The owning module: a code module, or the standalone test module."""
    kind: str = "code"
    """``code`` when a code module owns the item, ``test`` for standalone tests."""
    modules: list[str] = Field(min_length=1)
    """The owning module plus every test module attached to it, sorted."""
    criticality: float = Field(ge=0.0, le=1.0)
    """The maximum member criticality; missing evidence scores the honest zero."""
    review_tier: str = "standard"
    """``tight`` for the top criticality tier; ``standard`` otherwise."""
    dependencies: list[str] = Field(default_factory=list)
    """Item ids this item waits for (sorted, never self-referencing)."""
    loc: int = Field(default=0, ge=0)
    """Total lines of code across members — the item's PR-size evidence."""


class WorkPlan(BaseModel):
    """The whole run's work, canonically ordered — criticality first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[WorkItem]

    @property
    def tight_count(self) -> int:
        """How many items carry the tight review tier."""
        return sum(1 for item in self.items if item.review_tier == "tight")


def item_id_for(module_id: str) -> str:
    """A segment-safe, deterministic, injective id for a module id.

    Module ids are dotted names (``evogine.cli``), which are not safe run
    directory segments. Safe ids pass through unchanged; anything else is
    sanitized and suffixed with a short digest of the original — the suffix
    keeps the mapping injective (``a.b`` and ``a_b`` sanitize alike, but
    their digests differ).
    """
    if _SAFE_SEGMENT.fullmatch(module_id):
        return module_id
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", module_id)
    if not sanitized or not _SAFE_SEGMENT.fullmatch(sanitized):
        sanitized = f"m{sanitized}"
    digest = hashlib.sha256(module_id.encode("utf-8")).hexdigest()[:_ID_HASH_WIDTH]
    return f"{sanitized}-{digest}"


def plan_work(
    dag: DecompositionDAG,
    *,
    tight_fraction: float = TIGHT_FRACTION,
) -> WorkPlan:
    """Partition ``dag`` into work items: criticality-ordered, tiered, total."""
    if not 0 < tight_fraction <= 1:
        raise ValueError(f"tight_fraction must be in (0, 1], got {tight_fraction}")
    nodes: dict[str, ModuleNode] = {node.module: node for node in dag.modules}

    # Ownership: each code module owns its item; each test module attaches
    # to the lexicographically-first code module listing it in tested_by
    # (deterministic), or stands alone when no code module lists it.
    owner: dict[str, str] = {module_id: module_id for module_id in nodes}
    for test_id in sorted(nodes):
        if nodes[test_id].kind != "test":
            continue
        hosts = sorted(
            module_id
            for module_id, node in nodes.items()
            if node.kind == "code" and test_id in node.tested_by
        )
        if hosts:
            owner[test_id] = hosts[0]

    members_of: dict[str, list[str]] = {}
    for module_id, host in owner.items():
        members_of.setdefault(host, []).append(module_id)

    code_hosts = sorted(host for host in members_of if nodes[host].kind == "code")
    tight_ids = _tight_tier(code_hosts, nodes, tight_fraction)

    items: list[WorkItem] = []
    for host in code_hosts:
        items.append(_build_item(host, members_of[host], nodes, owner, tight_ids))
    # Standalone test items (tests that test nothing the graph can see).
    for host in sorted(members_of):
        if nodes[host].kind != "code":
            items.append(_build_item(host, members_of[host], nodes, owner, tight_ids))

    items.sort(key=lambda item: (-item.criticality, item.item_id))
    return WorkPlan(items=items)


def _tight_tier(
    code_hosts: list[str],
    nodes: dict[str, ModuleNode],
    tight_fraction: float,
) -> set[str]:
    """The item ids in the tight tier: the top ``tight_fraction`` by rank.

    Rank-based, not score-based, so the tier survives any criticality
    distribution; ties at the cut score are all included.
    """
    if not code_hosts:
        return set()
    ranked = sorted(
        code_hosts,
        key=lambda module_id: (-(nodes[module_id].criticality or 0.0), module_id),
    )
    cut = max(0, math.ceil(len(ranked) * tight_fraction) - 1)
    threshold = ranked[cut]
    threshold_score = nodes[threshold].criticality or 0.0
    return {
        module_id
        for module_id in code_hosts
        if (nodes[module_id].criticality or 0.0) >= threshold_score
    }


def _build_item(
    host: str,
    members: list[str],
    nodes: dict[str, ModuleNode],
    owner: dict[str, str],
    tight_ids: set[str],
) -> WorkItem:
    ordered_members = sorted(set(members))
    criticality = max((nodes[m].criticality or 0.0) for m in ordered_members)
    dependencies: set[str] = set()
    for member in ordered_members:
        for target in nodes[member].imports:
            if target == member:
                continue
            target_owner = owner.get(target)
            if target_owner is not None and target_owner != host:
                dependencies.add(item_id_for(target_owner))
    return WorkItem(
        item_id=item_id_for(host),
        module=host,
        kind="code" if nodes[host].kind == "code" else "test",
        modules=ordered_members,
        criticality=criticality,
        review_tier="tight" if host in tight_ids else "standard",
        dependencies=sorted(dependencies),
        loc=sum(nodes[m].loc for m in ordered_members),
    )


__all__ = ["TIGHT_FRACTION", "WorkItem", "WorkPlan", "item_id_for", "plan_work"]
