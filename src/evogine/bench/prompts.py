"""The fingerprinted prompt pack: the bench's pinned prompt artifact (V3 W0).

A live bench run spends real money on prompts, so the prompt text is part of
the run's pre-registered configuration, not an ambient implementation detail.
This module owns that artifact: the exact system prompt and user template a
live runner sends, carried by a versioned :class:`PromptPack` whose SHA-256
fingerprint is content-addressed — identical content reproduces the
fingerprint bit-for-bit, and any edit (including a version bump) changes it.
A protocol pins the fingerprint (``EvalProtocol.prompt_pack``), and every
recorded model call cites the fingerprint of the pack that produced it, so a
run's evidence always names the prompt text behind it.

Deliberately distinct from the swarm role prompts in
:mod:`evogine.swarm.roles`: those drive the builder (unified diffs) and the
reflector (learning drafts) — role plumbing, not benchmark surface. The pack
asks for a bench case's direct answer, and its text is data, not code.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evogine.engine.records import LearningRecord

TASK_CONTEXT_TOKEN = "{{task_context}}"
"""User-template placeholder for the case's engine-facing task context."""

KNOWLEDGE_TOKEN = "{{knowledge}}"
"""User-template placeholder for the rendered promoted-knowledge block."""

_EMPTY_KNOWLEDGE_BLOCK = "(No promoted knowledge was retrieved for this case.)"


def knowledge_block(knowledge: list[LearningRecord]) -> str:
    """Render the served records as the prompt's knowledge block.

    Deterministic in the given order — retrieval order is meaningful and is
    preserved verbatim. Empty knowledge renders the honest empty state: the
    model is never told knowledge exists when none was served.
    """
    if not knowledge:
        return _EMPTY_KNOWLEDGE_BLOCK
    rendered = "\n".join(
        f"- [{record.id}] {record.title}: {record.body.fix_strategy}"
        for record in knowledge
    )
    return f"Promoted knowledge (proven; prefer it where it applies):\n{rendered}"


class PromptPack(BaseModel):
    """A versioned, content-addressed bench prompt pack.

    The pack is the frozen prompt surface of a live run: the system prompt
    plus a user template with exactly one ``{{task_context}}`` and one
    ``{{knowledge}}`` token. The fingerprint covers the pack's full content —
    name, version, and both prompt surfaces — so a protocol pinning the
    fingerprint pins the exact prompt text, and every recorded call cites it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    """The pack's version. Deliberately inside the fingerprint: a version
    bump is a content change and must produce a new fingerprint."""
    system: str = Field(min_length=1)
    user_template: str = Field(min_length=1)
    """The user-turn template; must contain exactly one
    ``{{task_context}}`` and one ``{{knowledge}}`` token (fail-closed:
    a template that cannot carry either is a construction error, not a
    runtime surprise)."""

    @model_validator(mode="after")
    def _template_tokens(self) -> PromptPack:
        for token in (TASK_CONTEXT_TOKEN, KNOWLEDGE_TOKEN):
            if self.user_template.count(token) != 1:
                raise ValueError(
                    f"user_template must contain exactly one {token} token"
                )
        return self

    @property
    def fingerprint(self) -> str:
        """Return the content-addressed fingerprint: ``sha256:<hex>``.

        SHA-256 over canonicalized JSON (sorted keys, no insignificant
        whitespace) of the pack's full content. Identical content reproduces
        the fingerprint; any edit — a prompt word, the name, a version bump —
        changes it.
        """
        canonical = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "system": self.system,
                "user_template": self.user_template,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def render_user(self, *, task_context: str, knowledge: list[LearningRecord]) -> str:
        """Render the user turn: task context and knowledge into the template.

        Substitution is single-pass over the template's literal segments, so
        task or record content that itself contains a template token stays
        literal text and can never re-expand.
        """
        head, tail = self.user_template.split(KNOWLEDGE_TOKEN)
        return (
            head.replace(TASK_CONTEXT_TOKEN, task_context)
            + knowledge_block(knowledge)
            + tail.replace(TASK_CONTEXT_TOKEN, task_context)
        )


DEFAULT_PROMPT_PACK = PromptPack(
    name="evogine-bench",
    version=1,
    system=(
        "You are answering a benchmark case about production engineering "
        "failures. Answer the stated task directly and completely in plain "
        "prose: name the exact rule, guard, or discipline the task asks for "
        "and why it holds. "
        "Reply with the answer only — no preamble, no code fences."
    ),
    user_template=(
        "Task:\n{{task_context}}\n\n{{knowledge}}\n\nAnswer the task directly."
    ),
)


__all__ = [
    "DEFAULT_PROMPT_PACK",
    "KNOWLEDGE_TOKEN",
    "TASK_CONTEXT_TOKEN",
    "PromptPack",
    "knowledge_block",
]
