"""V3 W0: the live case runner — the bench's fail-closed doorway to models.

The first :class:`evogine.bench.harness.CaseRunner` that drives the swarm
:class:`evogine.swarm.clients.ModelClient` seam. Its contract is fail-closed
end to end:

- A client exception — raised only after the client's own bounded retries
  and provider failover are exhausted — becomes a *recorded failing
  outcome*: the call is persisted with its error, the replay's submission
  is a sentinel that cannot pass a sane oracle, and the run continues.
  A model outage degrades evidence quality; it never destroys the run and
  never silently passes a case.
- Every model call — success or failure — is recorded with its model
  identity and token counts, so post-run spend reconciliation
  (:func:`evogine.bench.budgets.reconcile_spend`) prices what actually
  happened, and the record shows exactly which prompt text produced it
  (the pack's fingerprint travels with every call).
- Exceptions outside the seam's error types mean a broken provider
  implementation; per the client contract they propagate untouched —
  a bug is never converted into benchmark evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evogine.bench.budgets import DollarBudget, SpendReconciliation, reconcile_spend
from evogine.bench.cases import BenchCase
from evogine.bench.harness import CaseRunner
from evogine.bench.prompts import PromptPack
from evogine.bench.replay import Arm
from evogine.bench.rundirs import RunDirectory
from evogine.engine.records import LearningRecord
from evogine.swarm.clients import ModelClient, ModelClientError, ModelRequest

FAILURE_OUTPUT = (
    "[live runner: model call failed after retries and failover — "
    "recorded as a failing outcome]"
)
"""The submission a failed call returns instead of raising.

Deliberately not empty: an empty string could be mistaken for a real
(degenerate) answer by a sloppy oracle, while this sentinel names what
happened and cannot satisfy any invariant a case would grade.
"""


class LiveCallRecord(BaseModel):
    """The outcome record of exactly one model call, persisted per call.

    Successes carry the answering model and its token counts; failures
    carry the error and ``None`` for model and tokens — a failed call is
    never fabricated into usage data, it is counted as evidence that spent
    nothing billable here.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    arm: Arm
    repeat: int
    call_index: int
    """1-based count of calls this runner has made across the whole run."""
    prompt_pack_fingerprint: str
    """Which prompt text produced this call — the pack is content-addressed."""
    ok: bool
    model: str | None = None
    """Model identity as reported by the provider; ``None`` when the call failed."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class LiveCaseRunner:
    """Produce case submissions through the swarm model client seam.

    The client passed here owns resilience — compose
    :class:`~evogine.swarm.clients.RetryingModelClient` and
    :class:`~evogine.swarm.clients.FailoverModelClient` around the raw
    provider yourself. By the time ``complete`` raises, retries and
    failover are exhausted; the runner's fail-closed mapping takes over.
    When ``run`` is given, every call record is also persisted into the run
    directory so evidence survives the process.
    """

    def __init__(
        self,
        client: ModelClient,
        pack: PromptPack,
        *,
        run: RunDirectory | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._pack = pack
        self._run = run
        self._max_tokens = max_tokens
        self._call_index = 0
        self.calls: list[LiveCallRecord] = []

    def execute(
        self,
        case: BenchCase,
        *,
        arm: Arm,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        """Return the submission for one replay, recording the call either way."""
        request = ModelRequest(
            system=self._pack.system,
            user=self._pack.render_user(task_context=case.task_context, knowledge=knowledge),
            max_tokens=self._max_tokens,
        )
        try:
            response = self._client.complete(request)
        except ModelClientError as exc:
            # Terminal by the time it reaches us: the client already ran its
            # bounded retries and provider failover. Record, fail the replay,
            # keep the run alive.
            self._record(
                case=case,
                arm=arm,
                repeat=repeat,
                ok=False,
                error=str(exc),
            )
            return FAILURE_OUTPUT
        self._record(
            case=case,
            arm=arm,
            repeat=repeat,
            ok=True,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return response.text

    def reconcile(self, budget: DollarBudget) -> SpendReconciliation:
        """Price this run's recorded calls against the declared spend table."""
        return reconcile_spend(budget, self.calls)

    def _record(
        self,
        *,
        case: BenchCase,
        arm: Arm,
        repeat: int,
        ok: bool,
        error: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self._call_index += 1
        record = LiveCallRecord(
            case_id=case.id,
            arm=arm,
            repeat=repeat,
            call_index=self._call_index,
            prompt_pack_fingerprint=self._pack.fingerprint,
            ok=ok,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )
        self.calls.append(record)
        if self._run is not None:
            self._run.write_artifact(
                f"call-{case.id}-{arm}-{repeat:04d}-{record.call_index:04d}",
                record.model_dump_json(),
            )


def read_call_records(run_dir: RunDirectory) -> list[LiveCallRecord]:
    """Load the persisted per-call records back out of a run directory.

    Spend reconciliation reads these from disk — not from the runner's
    memory — so the report also accounts for work loaded back from a
    resumed run.
    """
    records: list[LiveCallRecord] = []
    for path in sorted((run_dir.directory / "artifacts").glob("call-*.json")):
        records.append(LiveCallRecord.model_validate_json(path.read_text(encoding="utf-8")))
    return records


__all__ = [
    "FAILURE_OUTPUT",
    "LiveCallRecord",
    "LiveCaseRunner",
    "read_call_records",
]

# Static protocol proof: mypy verifies LiveCaseRunner satisfies CaseRunner
# structurally at this assignment; the runtime isinstance twin lives in
# tests/test_live_runner.py.
_protocol_check: type[CaseRunner] = LiveCaseRunner
