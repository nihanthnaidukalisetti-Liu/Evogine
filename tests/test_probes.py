"""V2-7 probe tests: memorization-recall contamination on fixed fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

from evogine.bench.cases import BenchCase, entries
from evogine.bench.probes import (
    content_tokens,
    memorization_probe,
    recall_probe,
)
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)


class RecordingRetriever:
    """Retrieval stub that returns the same records for every query."""

    def __init__(self, records: list[LearningRecord]) -> None:
        self.records = records
        self.queries: list[str] = []

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        self.queries.append(task_context)
        return list(self.records)[:k]


def _record(record_id: str, text: str) -> LearningRecord:
    """A promoted record whose searchable text is ``text``."""
    return LearningRecord(
        id=record_id,
        type="failure-pattern",
        title=text[:60],
        body=RecordBody(
            signal=text,
            failure_mode=text,
            detection="spotted in counterfactual replay",
            fix_strategy=text,
        ),
        status="promoted",
        confidence=0.9,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="test-fixture",
            source="unit-test",
            note="deterministic fixture; never touches a live model",
        ),
        counterfactual=RecordCounterfactual(
            replay_run_ids=[], verdict_change=None, evidence=None
        ),
        created=datetime(2026, 9, 18, tzinfo=UTC),
    )


def _holdout_case(case_id: str, context: str) -> BenchCase:
    return BenchCase(
        id=case_id,
        family="holdout-fam",
        task_context=context,
        hidden_invariant=lambda output: True,
    )


def test_content_tokens_drop_stopwords_and_short_words() -> None:
    tokens = content_tokens("The handler must retry with jitter and cap")
    assert "handler" in tokens and "jitter" in tokens
    assert "the" not in tokens and "must" not in tokens and "with" not in tokens


def test_recall_probe_clean_when_store_holds_only_benchmark_knowledge() -> None:
    holdout = _holdout_case("gamma", "context quoting quartz Keops-448 axes")
    benchmark = _holdout_case("alpha", "context quoting graphite Delphi-113 nodes")
    retriever = RecordingRetriever([_record("lr-1", "generic graphite backoff guidance")])

    report = recall_probe(retriever, [holdout], corpus=[holdout, benchmark])

    assert report.contaminated is False
    assert report.findings == []
    assert report.probed_cases == 1


def test_recall_probe_flags_records_quoting_holdout_unique_content() -> None:
    holdout = _holdout_case("gamma", "the scheduler quoted quartz Keops-448 to the audit")
    benchmark = _holdout_case("alpha", "the scheduler quoted graphite Delphi-113 to the audit")
    dirty = _record("lr-9", "quartz Keops-448 escalation playbook")
    retriever = RecordingRetriever([dirty])

    report = recall_probe(retriever, [holdout], corpus=[holdout, benchmark])

    assert report.contaminated is True
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.record_id == "lr-9"
    assert finding.case_id == "gamma"
    assert "keops" in finding.matched_tokens and "quartz" in finding.matched_tokens


def test_recall_probe_never_flags_shared_benchmark_vocabulary() -> None:
    """A record quoting words the benchmark families also use is not contaminated."""
    holdout = _holdout_case("gamma", "handle the idempotency-key overlap on retry storms")
    benchmark = _holdout_case("alpha", "handle the idempotency-key overlap on queue storms")
    retriever = RecordingRetriever([_record("lr-2", "idempotency-key overlap guidance")])

    report = recall_probe(retriever, [holdout], corpus=[holdout, benchmark])

    assert report.contaminated is False


def test_recall_probe_queries_each_heldout_case() -> None:
    holdout_a = _holdout_case("gamma-a", "context about stellar nautical tables")
    holdout_b = _holdout_case("gamma-b", "context about verdant mahogany inlets")
    retriever = RecordingRetriever([])

    recall_probe(retriever, [holdout_a, holdout_b], corpus=[holdout_a, holdout_b])

    assert retriever.queries == [holdout_a.task_context, holdout_b.task_context]


def test_memorization_probe_flags_records_quoting_the_gold_answer() -> None:
    retry_entry = next(e for e in entries() if e.case.id == "retry-jitter-budget")
    memorized = _record(
        "lr-3",
        "use exponential backoff with jitter and a hard cap of five attempts",
    )
    retriever = RecordingRetriever([memorized])

    report = memorization_probe(retriever, [retry_entry])

    assert report.contaminated is True
    assert report.findings[0].record_id == "lr-3"
    assert "jitter" in report.findings[0].matched_tokens
    assert "backoff" in report.findings[0].matched_tokens


def test_memorization_probe_stays_silent_on_generalizable_knowledge() -> None:
    retry_entry = next(e for e in entries() if e.case.id == "retry-jitter-budget")
    general = _record(
        "lr-4",
        "throttle clients degrade under sustained pressure; cap retry depth",
    )
    retriever = RecordingRetriever([general])

    report = memorization_probe(retriever, [retry_entry])

    assert report.contaminated is False


def test_memorization_probe_runs_over_the_whole_default_corpus() -> None:
    retriever = RecordingRetriever([])

    report = memorization_probe(retriever, list(entries()))

    assert report.probed_cases == 10
    assert report.contaminated is False


def test_probe_reports_round_trip_through_pydantic() -> None:
    holdout = _holdout_case("gamma", "context quoting quartz Keops-448 axes")
    benchmark = _holdout_case("alpha", "context quoting graphite Delphi-113 nodes")
    retriever = RecordingRetriever([_record("lr-9", "quartz Keops-448 playbook")])

    report = recall_probe(retriever, [holdout], corpus=[holdout, benchmark])

    dumped = report.model_dump(mode="json")
    restored = type(report).model_validate(dumped)
    assert restored == report
