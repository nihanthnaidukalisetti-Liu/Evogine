"""The bench case atom: an engine-facing prompt plus a hidden host-side oracle."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


class BenchCase(BaseModel):
    """One benchmark case.

    ``task_context`` is the only surface the engine ever sees — the warm arm
    retrieves promoted knowledge against exactly this text. The
    ``hidden_invariant`` is the host-side oracle: the harness grades every
    submission with it locally, and it never travels to the engine, the
    store, or any gate (oracle-leak protection by construction).

    ``proposal_id`` attributes the case's replay spend to the learning under
    test; when omitted, the case itself is the attribution unit.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    task_context: str
    hidden_invariant: Callable[[str], bool]
    proposal_id: str | None = None

    def proposal(self) -> str:
        """Return the proposal this case's replay spend is attributed to."""
        return self.proposal_id if self.proposal_id is not None else self.id

    def engine_view(self) -> dict[str, str]:
        """Return the engine-facing projection: id, family, task context — never the oracle."""
        return {
            "id": self.id,
            "family": self.family,
            "task_context": self.task_context,
        }

    def grade(self, output: str) -> bool:
        """Evaluate the hidden host-side invariant against a submission."""
        return bool(self.hidden_invariant(output))
