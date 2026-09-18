"""Gold-sanity admission: a generated test proves it can judge before it judges.

The reflector's ``check`` outputs — and any other system-generated test —
can become judges of builder work. An unsanity-checked judge is an attack
surface: a test rigged by its own author (asserting the broken behavior,
or asserting nothing) would otherwise grade everything downstream. The
V2-8 rule: before a generated test can judge anything, it must first run
against a known-good solution and survive two checks —

1. **Known-good run**: the test must PASS on the gold tree. A test that
   fails the correct behavior is broken, not strict.
2. **Known-broken run**: the test must FAIL on a reference tree with the
   known defect the test is supposed to catch. A test that passes both
   trees discriminates nothing — it is rigged to wave everything through.

The gold run goes first, literally: a test that cannot pass the
known-good solution never earns the second run. Admission fails closed —
an evaluation error is a refusal, never an admission, because a test that
cannot be sanity-checked must not judge.

Execution goes through the same :class:`~evogine.gates.code_gate.CaseEvaluator`
seam the code gate uses, so the unit suite stays offline with
deterministic stubs and the real Docker runner plugs in as configuration.
The caller stages the generated test source into both trees — identically
in each, or the comparison is meaningless — and hands the gate the
command that runs it; the gate judges the command's behavior on each
tree, never the proposing agent's claims about it.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evogine.gates.code_gate import CaseEvaluator, CaseResult, CaseVerdict, EvalCase
from evogine.sandbox.sanitize import sanitize_output


class GeneratedTest(BaseModel):
    """A system-generated test before admission: id, command, provenance."""

    model_config = ConfigDict(extra="forbid")

    test_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    """The exact command that runs the test on a staged tree."""
    proposed_by: str = Field(min_length=1)
    """Which role generated it (e.g. ``reflector``) — judges carry lineage."""


class GoldStage(str, Enum):
    """Where gold-sanity stopped (or admitted) a generated test."""

    ADMITTED = "admitted"
    FAILS_GOLD = "fails_gold"
    NO_DISCRIMINATION = "no_discrimination"
    GATE_ERROR = "gate_error"


class GoldSanityVerdict(BaseModel):
    """Typed admission verdict for one generated test, with the evidence runs."""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    admitted: bool
    stage: GoldStage
    reason: str
    gold_result: CaseResult | None = None
    broken_result: CaseResult | None = None


class GoldSanityGate:
    """Admits a generated test only after it survives the known-good run.

    ``gold_tree`` stages the known-good (reference) solution; ``broken_tree``
    stages a reference with the known defect the test is supposed to catch.
    Both trees are prepared by the caller, exactly as the code gate's
    staging prepares baseline and candidate trees.
    """

    def __init__(
        self,
        evaluator: CaseEvaluator,
        *,
        gold_tree: Path,
        broken_tree: Path,
    ) -> None:
        self._evaluator = evaluator
        self._gold_tree = gold_tree
        self._broken_tree = broken_tree

    def admit(self, test: GeneratedTest) -> GoldSanityVerdict:
        """Run the two-arm sanity check and admit or refuse the test.

        The known-good run is always first; a test that fails it is
        refused before the known-broken run is spent.
        """
        case = EvalCase(case_id=f"gold-sanity:{test.test_id}", command=test.command)

        gold: CaseResult
        try:
            gold = self._evaluator.run_case("gold", self._gold_tree, case)
        except Exception as exc:  # noqa: BLE001 — fail-closed: an evaluator crash
            # must never become an admission (documented per-method exception).
            return self._gate_error(test, f"gold run failed: {exc}")
        if gold.verdict is not CaseVerdict.PASS:
            return GoldSanityVerdict(
                test_id=test.test_id,
                admitted=False,
                stage=GoldStage.FAILS_GOLD,
                reason=(
                    "generated test fails the known-good solution — broken, "
                    f"not strict: {sanitize_output(gold.detail, max_chars=200)}"
                ),
                gold_result=gold,
            )

        broken: CaseResult
        try:
            broken = self._evaluator.run_case("broken", self._broken_tree, case)
        except Exception as exc:  # noqa: BLE001 — fail-closed, as above.
            return self._gate_error(test, f"known-broken run failed: {exc}")
        if broken.verdict is CaseVerdict.PASS:
            return GoldSanityVerdict(
                test_id=test.test_id,
                admitted=False,
                stage=GoldStage.NO_DISCRIMINATION,
                reason=(
                    "generated test passes the known-broken reference — it "
                    "discriminates nothing and would wave every defect through"
                ),
                gold_result=gold,
                broken_result=broken,
            )

        return GoldSanityVerdict(
            test_id=test.test_id,
            admitted=True,
            stage=GoldStage.ADMITTED,
            reason="passes the known-good solution, fails the known-broken reference",
            gold_result=gold,
            broken_result=broken,
        )

    @staticmethod
    def _gate_error(test: GeneratedTest, detail: str) -> GoldSanityVerdict:
        return GoldSanityVerdict(
            test_id=test.test_id,
            admitted=False,
            stage=GoldStage.GATE_ERROR,
            reason=(
                f"gold-sanity could not evaluate the test, refusing (fail-closed): "
                f"{sanitize_output(detail, max_chars=200)}"
            ),
        )
