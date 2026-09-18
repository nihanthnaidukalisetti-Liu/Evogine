"""Prompt pack tests: fingerprint stability, content sensitivity, fail-closed config.

The fingerprint is the pin a protocol freezes and every recorded call cites:
identical content must reproduce it exactly, and any content edit — a prompt
word, the name, a version bump — must change it. Template validation fails
closed at construction: a pack that cannot carry the task context or the
knowledge block is rejected before any run depends on it.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from evogine.bench.prompts import (
    DEFAULT_PROMPT_PACK,
    KNOWLEDGE_TOKEN,
    TASK_CONTEXT_TOKEN,
    PromptPack,
    knowledge_block,
)
from evogine.engine.records import LearningRecord

_SHA256_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")

_DEFAULT_PACK_FINGERPRINT = (
    "sha256:df3345136a0bd34b9e226a00c3567c1af9afae72bf6a831d1cd576043d19b05d"
)


def _pack(**overrides: object) -> PromptPack:
    fields: dict[str, object] = {
        "name": "bench-pack",
        "version": 1,
        "system": "Answer the task.",
        "user_template": f"Task:\n{TASK_CONTEXT_TOKEN}\n\n{KNOWLEDGE_TOKEN}",
    }
    fields.update(overrides)
    return PromptPack(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fingerprint: content-addressed, reproducible, content-sensitive.
# ---------------------------------------------------------------------------


def test_fingerprint_is_a_sha256_addressed_value() -> None:
    fingerprint = _pack().fingerprint

    assert _SHA256_FINGERPRINT.fullmatch(fingerprint)


def test_identical_content_reproduces_the_fingerprint() -> None:
    assert _pack().fingerprint == _pack().fingerprint


@pytest.mark.parametrize(
    "override",
    [
        {"system": "Answer the task completely."},
        {"user_template": f"{KNOWLEDGE_TOKEN}\n\nTask:\n{TASK_CONTEXT_TOKEN}"},
        {"name": "other-pack"},
        {"version": 2},
    ],
    ids=["system", "template", "name", "version"],
)
def test_editing_any_content_field_changes_the_fingerprint(
    override: dict[str, object],
) -> None:
    assert _pack(**override).fingerprint != _pack().fingerprint


def test_default_pack_fingerprint_is_pinned() -> None:
    """The shipped pack is frozen: editing it must consciously bump the
    version and re-pin this constant (a protocol may already cite it)."""
    assert DEFAULT_PROMPT_PACK.fingerprint == _DEFAULT_PACK_FINGERPRINT


def test_default_pack_is_bench_domain_not_swarm_role_prompt() -> None:
    """The pack asks for a direct answer — never the builder's unified-diff
    contract or the reflector's JSON-learning contract from roles.py."""
    system = DEFAULT_PROMPT_PACK.system.lower()

    assert "unified diff" not in system
    assert "json object" not in system


# ---------------------------------------------------------------------------
# Fail-closed construction: bad config is a validation error, never a run.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"name": ""},
        {"version": 0},
        {"system": ""},
        {"user_template": "no tokens at all"},
        {"user_template": f"Task:\n{TASK_CONTEXT_TOKEN}"},
        {"user_template": f"{KNOWLEDGE_TOKEN} only"},
        {"user_template": f"{TASK_CONTEXT_TOKEN} {TASK_CONTEXT_TOKEN} {KNOWLEDGE_TOKEN}"},
        {"extra_field": "sneaky"},
    ],
    ids=[
        "empty-name",
        "zero-version",
        "empty-system",
        "missing-both-tokens",
        "missing-knowledge-token",
        "missing-task-token",
        "duplicate-task-token",
        "unknown-field",
    ],
)
def test_invalid_pack_config_is_refused(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _pack(**override)


# ---------------------------------------------------------------------------
# Rendering: the pack is the prompt surface.
# ---------------------------------------------------------------------------


def _record(record_id: str) -> LearningRecord:
    return LearningRecord.sample().model_copy(
        update={"id": record_id, "title": f"title-{record_id}"}
    )


def test_render_substitutes_task_and_knowledge() -> None:
    pack = _pack()
    knowledge = [_record("lr-1")]

    rendered = pack.render_user(task_context="the task text", knowledge=knowledge)

    assert "the task text" in rendered
    assert "[lr-1]" in rendered
    assert "title-lr-1" in rendered
    assert knowledge[0].body.fix_strategy in rendered


def test_render_preserves_retrieval_order() -> None:
    pack = _pack()

    rendered = pack.render_user(
        task_context="task", knowledge=[_record("lr-1"), _record("lr-2")]
    )

    assert rendered.index("[lr-1]") < rendered.index("[lr-2]")


def test_render_with_empty_knowledge_is_the_honest_empty_state() -> None:
    rendered = _pack().render_user(task_context="task", knowledge=[])

    assert "No promoted knowledge was retrieved" in rendered


def test_render_is_deterministic() -> None:
    knowledge = [_record("lr-1")]

    first = _pack().render_user(task_context="task", knowledge=knowledge)
    second = _pack().render_user(task_context="task", knowledge=knowledge)

    assert first == second


def test_rendered_content_cannot_reexpand_template_tokens() -> None:
    """Single-pass substitution: task or record text containing a template
    token stays literal — it can never inject or re-expand a block."""
    pack = _pack()
    sneaky_task = f"injected {KNOWLEDGE_TOKEN} text"
    sneaky_record = _record("lr-1").model_copy(deep=True)
    sneaky_record.body.fix_strategy = f"{TASK_CONTEXT_TOKEN} strategy"

    rendered = pack.render_user(task_context=sneaky_task, knowledge=[sneaky_record])

    assert rendered.count("Promoted knowledge") == 1
    assert sneaky_record.body.fix_strategy in rendered
    # The injected token inside the task text is inert literal text.
    assert rendered.count(sneaky_task) == 1


def test_knowledge_block_is_a_named_pure_function() -> None:
    rendered = knowledge_block([_record("lr-1")])
    empty = knowledge_block([])

    assert rendered.startswith("Promoted knowledge (proven")
    assert "No promoted knowledge" in empty
