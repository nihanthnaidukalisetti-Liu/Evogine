"""Refusal and failure surface for the swarm orchestrator.

Every refusal is typed and visible: a run that cannot proceed names the
exact defect and stops — it never degrades silently into unbounded spend,
an unwritten checkpoint, or a half-executed plan. The base class lets
callers catch any orchestrator refusal; the concrete classes carry the
details (the ceiling numbers, the missing variable's name).
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for orchestrator refusals and contract violations."""


class OrchestratorRefused(OrchestratorError):
    """A fail-closed refusal state: the run stops before further work."""


class MissingApiKeyRefused(OrchestratorRefused):
    """The configured provider key environment variable is not set.

    Raised before any model call is made — a key refusal spends nothing.
    The run directory (already open when work was pending) records the
    refusal as a visible ``refusal`` event and stays resumable.
    """


class EventLogError(OrchestratorError):
    """The event log is corrupted mid-file — real corruption, never hidden."""


class CheckpointError(OrchestratorError):
    """A checkpoint marker is unusable: unreadable payload or a key collision."""
