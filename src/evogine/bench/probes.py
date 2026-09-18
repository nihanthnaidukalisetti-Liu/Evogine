"""V2-7: memorization-recall contamination probes.

Two probes ask whether a warm win is really knowledge working:

- **Recall probe** (held-out families): the store must quote nothing unique
  to the held-out cases. A record that reproduces held-out content could
  only have gotten it by seeing the holdout — the family is no longer fresh
  and any fresh-holdout claim built on it is void. The harness runs this
  probe pre-run and refuses a contaminated holdout before spending.
- **Memorization probe** (benchmark cases): a record that quotes the case's
  gold answer marks the warm arm as a memorization risk — the win may be
  answer recall, not generalizable knowledge. Advisory: the report lists
  matched tokens so a reviewer can adjudicate generic-vocabulary overlaps.

Both probes are pure functions over a retriever and fixed fixtures; they
never touch a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from evogine.bench.cases import BenchCase, CorpusEntry
from evogine.engine.records import LearningRecord

_STOPWORDS = frozenset(
    {
        "the", "and", "with", "must", "that", "this", "from", "when", "what",
        "how", "state", "produce", "including", "instead", "because", "every",
        "never", "into", "over", "under", "same", "their", "them", "they",
        "have", "been", "were", "will", "would", "could", "should", "your",
        "than", "then", "once", "only", "also", "such", "each", "both",
    }
)


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase content tokens of a text: length >= 4, stopwords removed."""
    return frozenset(re.findall(r"[a-z_][a-z0-9_]{3,}", text.lower())) - _STOPWORDS


class ProbeRetriever(Protocol):
    """The seam the probes need: the same promotion-only retrieval surface."""

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Return promoted records relevant to ``task_context``."""
        ...


class ProbeFinding(BaseModel):
    """One record quoting content it should not hold."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    record_id: str
    matched_tokens: list[str]


class ProbeReport(BaseModel):
    """Outcome of one contamination probe over a set of cases."""

    model_config = ConfigDict(extra="forbid")

    probe: str
    probed_cases: int
    findings: list[ProbeFinding]
    contaminated: bool
    """True when any record quoted content it should not hold."""


def _record_text(record: LearningRecord) -> str:
    return (
        f"{record.title} {record.body.signal} {record.body.failure_mode} "
        f"{record.body.detection} {record.body.fix_strategy}"
    )


def recall_probe(
    retriever: ProbeRetriever,
    heldout_cases: Sequence[BenchCase],
    *,
    corpus: Sequence[BenchCase],
) -> ProbeReport:
    """Does the store quote anything unique to the held-out cases?

    A token is *unique* when it appears in a held-out case's task context and
    nowhere else in the corpus. Generic domain vocabulary shared with the
    benchmark families cannot flag a record; quoting held-out-specific words
    can only mean the store saw the holdout.
    """
    heldout_ids = {case.id for case in heldout_cases}
    others = [case for case in corpus if case.id not in heldout_ids]
    other_tokens = frozenset().union(
        *(content_tokens(case.task_context) for case in others)
    )

    findings: list[ProbeFinding] = []
    for case in heldout_cases:
        unique = content_tokens(case.task_context) - other_tokens
        for record in retriever.retrieve(case.task_context):
            hits = sorted(content_tokens(_record_text(record)) & unique)
            if hits:
                findings.append(
                    ProbeFinding(
                        case_id=case.id, record_id=record.id, matched_tokens=hits
                    )
                )
    return ProbeReport(
        probe="recall-holdout",
        probed_cases=len(heldout_cases),
        findings=findings,
        contaminated=bool(findings),
    )


def memorization_probe(
    retriever: ProbeRetriever,
    entries: Sequence[CorpusEntry],
) -> ProbeReport:
    """Does retrieved knowledge quote the case's gold answer?

    The comparison set is the gold output minus the bad output — the tokens
    that discriminate a correct answer — so shared domain phrasing that both
    answers contain cannot flag a record.
    """
    findings: list[ProbeFinding] = []
    for entry in entries:
        gold = content_tokens(entry.good_output) - content_tokens(entry.bad_output)
        for record in retriever.retrieve(entry.case.task_context):
            hits = sorted(content_tokens(_record_text(record)) & gold)
            if hits:
                findings.append(
                    ProbeFinding(
                        case_id=entry.case.id,
                        record_id=record.id,
                        matched_tokens=hits,
                    )
                )
    return ProbeReport(
        probe="memorization-gold",
        probed_cases=len(entries),
        findings=findings,
        contaminated=bool(findings),
    )


class ContaminationError(Exception):
    """Pre-run refusal: the holdout is not fresh — the store quotes its content."""


__all__ = [
    "ContaminationError",
    "ProbeFinding",
    "ProbeReport",
    "ProbeRetriever",
    "content_tokens",
    "memorization_probe",
    "recall_probe",
]
