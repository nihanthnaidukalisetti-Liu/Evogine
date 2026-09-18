"""V2-7 protocol tests: pre-registration, fingerprints, pre-run validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evogine.bench.budgets import ReplayBudgets
from evogine.bench.cases import BenchCase, default_corpus
from evogine.bench.harness import BenchHarness
from evogine.bench.protocol import (
    ClaimTag,
    EvalProtocol,
    ProtocolValidationError,
    fingerprint_corpus,
    load_protocol,
    pre_run_validate,
)


def _budgets(per_run: int = 999) -> ReplayBudgets:
    return ReplayBudgets(
        max_per_proposal=999, max_per_case_family=999, max_per_run=per_run
    )


def _protocol(corpus: list[BenchCase], **overrides: object) -> EvalProtocol:
    fields: dict[str, object] = {
        "name": "v2-7 discipline check",
        "corpus_fingerprint": fingerprint_corpus(corpus),
        "prompt_pack": "pack-2026-09-18",
        "repeats": 4,
        "budgets": _budgets(),
    }
    fields.update(overrides)
    return EvalProtocol.model_validate(fields)


def test_fingerprint_is_order_independent_and_content_sensitive() -> None:
    corpus = default_corpus()
    shuffled = list(reversed(corpus))
    assert fingerprint_corpus(corpus) == fingerprint_corpus(shuffled)

    edited = [case.model_copy() for case in corpus]
    edited[0].task_context = edited[0].task_context + " One word changed."
    assert fingerprint_corpus(corpus) != fingerprint_corpus(edited)


def test_pre_run_validate_resolves_selection_and_schedule() -> None:
    corpus = default_corpus()
    protocol = _protocol(corpus, case_ids=["tz-offset-drift", "retry-jitter-budget"])

    check = pre_run_validate(protocol, corpus)

    assert [case.id for case in check.cases] == [
        "tz-offset-drift",
        "retry-jitter-budget",
    ]
    assert check.total_replays == 2 * 2 * 4  # cases × arms × repeats
    assert check.heldout_cases == ()


def test_pre_run_validate_reports_every_violation_at_once() -> None:
    corpus = default_corpus()
    protocol = _protocol(
        corpus,
        corpus_fingerprint="sha256:deadbeef",
        case_ids=["no-such-case", "tz-offset-drift"],
        held_out_family="no-such-family",
    )

    with pytest.raises(ProtocolValidationError) as excinfo:
        pre_run_validate(protocol, corpus)

    message = str(excinfo.value)
    assert "corpus fingerprint mismatch" in message
    assert "unknown case ids" in message
    assert "does not exist in the corpus" in message


def test_pre_run_validate_refuses_budget_exceeding_schedule() -> None:
    corpus = default_corpus()
    # Full corpus at 4 repeats spends 80 replays; 10 cannot cover it.
    protocol = _protocol(corpus, budgets=_budgets(per_run=10))

    with pytest.raises(ProtocolValidationError) as excinfo:
        pre_run_validate(protocol, corpus)

    assert "budgets refuse the resolved schedule" in str(excinfo.value)


def test_pre_run_validate_refunds_nothing_and_spends_nothing_on_failure() -> None:
    """A refused protocol leaves the runner untouched: refusal is pre-spend."""
    corpus = default_corpus()

    class CountingRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, case: BenchCase, **_: object) -> str:
            self.calls.append(case.id)
            return "ok"

    class StubRetriever:
        def retrieve(self, task_context: str, k: int = 5) -> list[object]:
            return []

    protocol = _protocol(corpus, budgets=_budgets(per_run=1))
    runner = CountingRunner()
    with pytest.raises(ProtocolValidationError):
        BenchHarness(StubRetriever(), runner).run(protocol=protocol)
    assert runner.calls == []


def test_pre_run_validate_requires_holdout_cases_to_run() -> None:
    corpus = default_corpus()
    protocol = _protocol(
        corpus,
        case_ids=["tz-offset-drift"],  # excludes every integration-drift case
        held_out_family="integration-drift",
    )

    with pytest.raises(ProtocolValidationError) as excinfo:
        pre_run_validate(protocol, corpus)

    assert "has no resolved cases" in str(excinfo.value)


def test_benchmark_claim_tag_cannot_self_certify_as_fresh() -> None:
    corpus = default_corpus()
    with pytest.raises(ValidationError, match="fresh-holdout"):
        _protocol(corpus, benchmark_claim_tag=ClaimTag.FRESH_HOLDOUT)


def test_repeats_below_the_floor_are_refused_at_construction() -> None:
    corpus = default_corpus()
    with pytest.raises(ValidationError, match="greater than or equal to 4"):
        _protocol(corpus, repeats=3)


def test_heldout_cases_resolve_by_family() -> None:
    corpus = default_corpus()
    protocol = _protocol(corpus, held_out_family="integration-drift")

    check = pre_run_validate(protocol, corpus)

    assert {case.id for case in check.heldout_cases} == {
        "schema-drift-pin",
        "pagination-truncation",
    }


def test_load_protocol_round_trips_yaml(tmp_path: Path) -> None:
    corpus = default_corpus()
    protocol = _protocol(corpus, held_out_family="integration-drift")
    path = tmp_path / "protocol.yml"
    path.write_text(yaml.safe_dump(protocol.model_dump(mode="json")), encoding="utf-8")

    loaded = load_protocol(path)

    assert loaded == protocol


def test_load_protocol_refuses_unknown_fields(tmp_path: Path) -> None:
    corpus = default_corpus()
    protocol = _protocol(corpus)
    data = protocol.model_dump(mode="json")
    data["surprise_knob"] = True
    path = tmp_path / "protocol.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_protocol(path)
