"""Shared offline helpers for orchestrator tests.

Fake model clients (scripted, recording), a recording sink, and DAG/plan
builders — structural stand-ins for the real seams, in the repo's
fake-client idiom: plain classes, no inheritance from the protocols.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from decimal import Decimal

from evogine.swarm.clients import ModelClientError, ModelRequest, ModelResponse
from evogine.swarm.decomposition.models import (
    DecompositionDAG,
    DecompositionStats,
    ModuleNode,
)
from evogine.swarm.orchestrator.planning import WorkItem, WorkPlan
from evogine.swarm.orchestrator.runner import RunSettings
from evogine.swarm.orchestrator.spend import SpendPolicy

BUILDER_REPLY = (
    "diff --git a/evogine/example.py b/evogine/example.py\n"
    "--- a/evogine/example.py\n"
    "+++ b/evogine/example.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
)

_REFLECTOR_BODY = {
    "signal": "swarm item failed",
    "failure_mode": "model failure",
    "detection": "metered call raised",
    "fix_strategy": "bounded retry then skip",
}
VALID_REFLECTOR_JSON = json.dumps(
    {
        "type": "failure-pattern",
        "title": "Swarm item failed",
        "body": _REFLECTOR_BODY,
        "confidence": 0.4,
        "note": "from the orchestrator test fake",
    }
)


class FakeModelClient:
    """Scripted ModelClient: per-role replies, marker-triggered failures.

    Builder-shaped prompts get ``builder_text``; reflector-shaped prompts
    get a valid learning JSON. ``fail_on`` markers fail **builder** calls
    only: the reflector's failure context quotes the failed item's id, so
    a marker would otherwise silence the learning too and the item could
    never settle as ``learning_proposed``.
    """

    def __init__(
        self,
        *,
        usage: tuple[int, int] = (1_000, 500),
        fail_on: Sequence[str] = (),
        builder_text: str = BUILDER_REPLY,
    ) -> None:
        self.usage = usage
        self.fail_on = list(fail_on)
        self.builder_text = builder_text
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        is_builder = request.system.startswith("You are the builder")
        if is_builder and any(marker in request.user for marker in self.fail_on):
            raise ModelClientError(f"scripted failure: {request.user[:60]!r}")
        if request.system.startswith("You are the reflector"):
            return ModelResponse(
                text=VALID_REFLECTOR_JSON,
                input_tokens=self.usage[0],
                output_tokens=self.usage[1],
            )
        return ModelResponse(
            text=self.builder_text,
            input_tokens=self.usage[0],
            output_tokens=self.usage[1],
        )


class FakeSink:
    """Recording LearningSink: proposals land in ``proposed``, returned intact."""

    def __init__(self) -> None:
        self.proposed: list[object] = []
        self.calls = 0

    def propose(self, draft: object) -> object:
        self.calls += 1
        self.proposed.append(draft)
        return draft


class CrashingModelClient(FakeModelClient):
    """A client that simulates a killed process after N recorded calls."""

    def __init__(self, *, exit_after_calls: int, exit_code: int = 67, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.exit_after_calls = exit_after_calls
        self.exit_code = exit_code

    def complete(self, request: ModelRequest) -> ModelResponse:
        if len(self.requests) >= self.exit_after_calls:
            # Simulate SIGKILL mid-run: no cleanup, no checkpoint, no marker.
            os._exit(self.exit_code)
        return super().complete(request)


def make_dag(spec: dict[str, dict[str, object]]) -> DecompositionDAG:
    """A DecompositionDAG from ``{module_id: {kind, imports, tested_by, loc, criticality}}``."""
    modules = [
        ModuleNode(
            module=module_id,
            path=str(entry.get("path", f"src/{module_id.replace('.', '/')}.py")),
            kind=entry.get("kind", "code"),  # type: ignore[arg-type]
            loc=int(entry.get("loc", 100)),
            imports=list(entry.get("imports", [])),  # type: ignore[arg-type]
            tested_by=list(entry.get("tested_by", [])),  # type: ignore[arg-type]
            criticality=entry.get("criticality"),  # type: ignore[arg-type]
        )
        for module_id, entry in spec.items()
    ]
    stats = DecompositionStats(
        modules=len(modules),
        code_modules=sum(1 for node in modules if node.kind == "code"),
        test_modules=sum(1 for node in modules if node.kind == "test"),
        import_edges=sum(len(node.imports) for node in modules),
        tested_by_edges=sum(len(node.tested_by) for node in modules),
    )
    return DecompositionDAG(modules=modules, edges=[], repo_root="/test-repo", stats=stats)


def make_plan(
    count: int = 4,
    *,
    prefix: str = "m",
    criticalities: Sequence[float] | None = None,
    dependencies: dict[str, list[str]] | None = None,
) -> WorkPlan:
    """A synthetic plan: ``count`` code items, optional criticalities/deps."""
    deps = dependencies or {}
    items = [
        WorkItem(
            item_id=f"{prefix}{index}",
            module=f"evogine.{prefix}{index}",
            kind="code",
            modules=[f"evogine.{prefix}{index}"],
            criticality=(criticalities[index] if criticalities is not None else 0.5),
            review_tier="standard",
            dependencies=deps.get(f"{prefix}{index}", []),
            loc=100,
        )
        for index in range(count)
    ]
    return WorkPlan(items=items)


def make_settings(
    *,
    min_workers: int = 1,
    max_workers: int = 2,
    ceiling_usd: str = "10.00",
    input_price_per_mtok_usd: str = "1.00",
    output_price_per_mtok_usd: str = "1.00",
    max_call_input_tokens: int = 1_000,
) -> RunSettings:
    """RunSettings with declared prices and a ``Decimal`` ceiling."""
    return RunSettings(
        min_workers=min_workers,
        max_workers=max_workers,
        api_key_env="EVOGINE_TEST_API_KEY",
        spend=SpendPolicy(
            ceiling_usd=Decimal(ceiling_usd),
            input_price_per_mtok_usd=Decimal(input_price_per_mtok_usd),
            output_price_per_mtok_usd=Decimal(output_price_per_mtok_usd),
            max_call_input_tokens=max_call_input_tokens,
        ),
    )


__all__ = [
    "BUILDER_REPLY",
    "VALID_REFLECTOR_JSON",
    "CrashingModelClient",
    "FakeModelClient",
    "FakeSink",
    "make_dag",
    "make_plan",
    "make_settings",
]
