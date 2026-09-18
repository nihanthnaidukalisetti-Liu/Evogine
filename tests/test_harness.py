"""Bench harness tests: stub cases and deterministic runners — never live models."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from evogine.bench.cases import BenchCase, CorpusEntry, default_corpus, entries
from evogine.bench.harness import REPEAT_FLOOR, BenchHarness
from evogine.engine.records import LearningRecord


class ScriptedRunner:
    """Deterministic runner stub: (case_id, arm) -> per-repeat submission outputs."""

    def __init__(self, outputs: dict[tuple[str, str], list[str]] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, str, int, int]] = []

    def execute(
        self,
        case: BenchCase,
        *,
        arm: str,
        knowledge: list[LearningRecord],
        repeat: int,
    ) -> str:
        self.calls.append((case.id, arm, repeat, len(knowledge)))
        sequence = self.outputs.get((case.id, arm), ["ok"])
        return sequence[repeat % len(sequence)]


class StubRetriever:
    """Promotion-only retrieval stub, offline by construction."""

    def __init__(self, records: dict[str, list[LearningRecord]] | None = None) -> None:
        self.records = records or {}
        self.queries: list[str] = []

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        self.queries.append(task_context)
        return list(self.records.get(task_context, []))[:k]


def _stub_case(
    case_id: str, family: str, invariant: Callable[[str], bool]
) -> BenchCase:
    return BenchCase(
        id=case_id,
        family=family,
        task_context=f"context-for-{case_id}",
        hidden_invariant=invariant,
    )


def _stub_cases() -> list[BenchCase]:
    return [
        _stub_case("alpha", "fam-a", lambda output: "ok" in output),
        _stub_case("beta", "fam-a", lambda output: output == "perfect"),
        _stub_case("gamma", "fam-b", lambda output: "fix" in output.lower()),
    ]


def test_integration_warm_vs_cold_over_stub_cases() -> None:
    """Full warm-vs-cold run over stub cases: outcomes, deltas, provenance, spend."""
    cases = _stub_cases()
    retriever = StubRetriever({"context-for-beta": [LearningRecord.sample()]})
    runner = ScriptedRunner(
        {
            ("beta", "cold"): ["no"] * 4,
            ("beta", "warm"): ["perfect"] * 4,
            ("alpha", "cold"): ["ok", "no", "ok", "no"],
            ("alpha", "warm"): ["ok", "no", "ok", "no"],
        }
    )

    report = BenchHarness(retriever, runner).run(
        [case.id for case in cases], corpus=cases
    )

    assert report.repeats == REPEAT_FLOOR
    assert [case.case_id for case in report.cases] == ["alpha", "beta", "gamma"]
    assert report.summary.replays == 24

    # Spend: every case × both arms × 4 repeats; cold saw no knowledge, warm
    # saw the retrieved record only for the case it was retrieved for.
    assert len(runner.calls) == 24
    assert all(knowledge == 0 for _, arm, _, knowledge in runner.calls if arm == "cold")
    warm_knowledge = {
        case_id: k for case_id, arm, _, k in runner.calls if arm == "warm"
    }
    assert warm_knowledge == {"alpha": 0, "beta": 1, "gamma": 0}
    assert retriever.queries == [case.task_context for case in cases]

    alpha, beta, gamma = report.cases
    assert beta.cold.pass_rate == 0.0
    assert beta.warm.pass_rate == 1.0
    assert beta.delta_pass_rate == 1.0
    assert alpha.cold.pass_rate == pytest.approx(0.5)
    assert alpha.delta_pass_rate == pytest.approx(0.0)
    assert gamma.delta_pass_rate == pytest.approx(0.0)

    # Rollout provenance: the graded output and the knowledge that informed it.
    first_alpha_cold = alpha.cold.outcomes[0]
    assert first_alpha_cold.output == "ok"
    assert first_alpha_cold.knowledge_ids == []
    first_beta_warm = beta.warm.outcomes[0]
    assert first_beta_warm.output == "perfect"
    assert first_beta_warm.knowledge_ids == [LearningRecord.sample().id]


def test_report_carries_variance_across_repeats() -> None:
    """A flip-flopping case reports its dispersion; a stable one reports zero."""
    cases = [_stub_case("solo", "fam-x", lambda output: "ok" in output)]
    runner = ScriptedRunner({("solo", "cold"): ["ok", "no", "ok", "no"]})

    report = BenchHarness(StubRetriever(), runner).run(corpus=cases)

    case_report = report.cases[0]
    assert case_report.cold.pass_rate == pytest.approx(0.5)
    assert case_report.cold.variance == pytest.approx(0.25)
    assert case_report.warm.pass_rate == pytest.approx(1.0)
    assert case_report.warm.variance == pytest.approx(0.0)


def test_regressions_listed_when_warm_loses() -> None:
    """A case where warm does worse lands in summary.regressions, never silently."""
    cases = [_stub_case("solo", "fam-x", lambda output: "ok" in output)]
    runner = ScriptedRunner(
        {
            ("solo", "cold"): ["ok"] * 4,
            ("solo", "warm"): ["no"] * 4,
        }
    )

    report = BenchHarness(StubRetriever(), runner).run(corpus=cases)

    assert report.summary.regressions == ["solo"]
    assert report.summary.improvements == []
    assert report.cases[0].delta_pass_rate == pytest.approx(-1.0)


def test_summary_aggregates_improvements_and_rates() -> None:
    cases = _stub_cases()
    retriever = StubRetriever({"context-for-beta": [LearningRecord.sample()]})
    runner = ScriptedRunner(
        {
            ("beta", "cold"): ["no"] * 4,
            ("beta", "warm"): ["perfect"] * 4,
            ("alpha", "cold"): ["ok", "no", "ok", "no"],
            ("alpha", "warm"): ["ok", "no", "ok", "no"],
        }
    )

    report = BenchHarness(retriever, runner).run(corpus=cases)

    assert report.summary.cold_pass_rate == pytest.approx(1 / 6)
    assert report.summary.warm_pass_rate == pytest.approx(0.5)
    assert report.summary.mean_delta_pass_rate == pytest.approx(1 / 3)
    assert report.summary.improvements == ["beta"]
    assert report.summary.regressions == []


def test_repeat_floor_is_enforced() -> None:
    with pytest.raises(ValueError, match="4-repeat floor"):
        BenchHarness(StubRetriever(), ScriptedRunner()).run(
            corpus=_stub_cases(), repeats=REPEAT_FLOOR - 1
        )


def test_repeats_default_to_floor() -> None:
    report = BenchHarness(StubRetriever(), ScriptedRunner()).run(corpus=_stub_cases())
    assert report.repeats == REPEAT_FLOOR


def test_unknown_case_id_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown case ids"):
        BenchHarness(StubRetriever(), ScriptedRunner()).run(
            ["nope"], corpus=_stub_cases()
        )


def test_empty_case_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        BenchHarness(StubRetriever(), ScriptedRunner()).run([], corpus=_stub_cases())


def test_duplicate_case_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        BenchHarness(StubRetriever(), ScriptedRunner()).run(
            ["alpha", "alpha"], corpus=_stub_cases()
        )


def test_engine_view_never_carries_the_invariant() -> None:
    """The engine-facing projection is id/family/task_context — the oracle stays host-side."""
    all_cases = _stub_cases() + default_corpus()
    for case in all_cases:
        assert set(case.engine_view()) == {"id", "family", "task_context"}


def test_corpus_has_ten_cases_across_seven_families() -> None:
    cases = default_corpus()
    assert len(cases) == 10
    assert len({case.id for case in cases}) == 10
    assert {case.family for case in cases} == {
        "time-zones",
        "retries",
        "concurrency",
        "locale",
        "overflow",
        "partial-failure",
        "integration-drift",
    }


def test_corpus_gold_sanity() -> None:
    """Every corpus oracle accepts its known-good output and rejects its known-bad one."""
    sanity: list[CorpusEntry] = list(entries())
    assert len(sanity) == 10
    for entry in sanity:
        assert entry.case.grade(entry.good_output) is True, entry.case.id
        assert entry.case.grade(entry.bad_output) is False, entry.case.id


def test_corpus_invariants_are_deterministic() -> None:
    for entry in entries():
        assert entry.case.grade(entry.good_output) == entry.case.grade(
            entry.good_output
        )
        assert entry.case.grade(entry.bad_output) == entry.case.grade(entry.bad_output)


def test_full_default_corpus_runs() -> None:
    """The shipped corpus runs end to end with default selection and stub spend."""
    report = BenchHarness(StubRetriever(), ScriptedRunner()).run()
    assert len(report.cases) == 10
    for case_report in report.cases:
        assert len(case_report.cold.outcomes) == REPEAT_FLOOR
        assert len(case_report.warm.outcomes) == REPEAT_FLOOR
        assert case_report.cold.passes == case_report.warm.passes  # scripted "ok"
