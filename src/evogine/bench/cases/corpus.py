"""The v1 10-case decay-pressure corpus.

Ten cases across the seven pressure families the harness measures:
concurrency, retries, time zones, locale, overflow, partial failure, and
integration drift. Each entry pairs the engine-facing case with known-good
and known-bad sample outputs used by the gold-sanity test; the hidden
invariant itself lives on the case and never reaches engine-facing surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from evogine.bench.cases.base import BenchCase


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """A corpus case plus gold-sanity outputs that pin its oracle's behavior."""

    case: BenchCase
    good_output: str
    bad_output: str


def _grade_utc_offset(output: str) -> bool:
    return "-08:00" in output and "09:30" in output


def _grade_retry_jitter(output: str) -> bool:
    lowered = output.lower()
    return "jitter" in lowered and "cap" in lowered


def _grade_race_idempotency(output: str) -> bool:
    lowered = output.lower()
    return "idempoten" in lowered and "lock" in lowered


def _grade_locale_decimal(output: str) -> bool:
    lowered = output.lower()
    return "locale" in lowered and "decimal" in lowered


def _grade_overflow_guard(output: str) -> bool:
    lowered = output.lower()
    return "bound" in lowered and "overflow" in lowered


def _grade_partial_write(output: str) -> bool:
    lowered = output.lower()
    # ``resum`` stems resume/resumable/resumption; ``idempoten`` stems
    # idempotent/idempotency.
    return "checkpoint" in lowered and ("resum" in lowered or "idempoten" in lowered)


def _grade_schema_pin(output: str) -> bool:
    lowered = output.lower()
    return "schema" in lowered and "pin" in lowered


def _grade_monotonic_clock(output: str) -> bool:
    return "monotonic" in output.lower()


def _grade_pagination_truncation(output: str) -> bool:
    lowered = output.lower()
    return "page" in lowered and "truncat" in lowered


def _grade_dup_submit(output: str) -> bool:
    return "idempoten" in output.lower()


_CASES: tuple[CorpusEntry, ...] = (
    CorpusEntry(
        case=BenchCase(
            id="tz-offset-drift",
            family="time-zones",
            task_context=(
                "A scheduler stored the local time '2026-03-08 01:30' with zone "
                "America/Los_Angeles. Produce the correct UTC instant and state the UTC "
                "offset in effect at that local moment — a daylight-saving transition "
                "happens that night."
            ),
            hidden_invariant=_grade_utc_offset,
        ),
        good_output=(
            "2026-03-08 01:30 America/Los_Angeles is 2026-03-08T09:30:00Z; the offset in "
            "effect is -08:00 (PST), because the spring-forward transition has not "
            "happened yet."
        ),
        bad_output=(
            "2026-03-08 01:30 America/Los_Angeles is 2026-03-08T08:30:00Z; the offset "
            "is -07:00 all year."
        ),
    ),
    CorpusEntry(
        case=BenchCase(
            id="retry-jitter-budget",
            family="retries",
            task_context=(
                "A client hammers a rate-limited API with tight, immediate retries and "
                "turns a small stall into an outage. State the retry policy the client "
                "must use, including how attempts are bounded."
            ),
            hidden_invariant=_grade_retry_jitter,
        ),
        good_output=(
            "Use exponential backoff with jitter and a hard cap of five attempts; stop "
            "and surface the error when the cap is hit."
        ),
        bad_output="Retry immediately in a tight loop until the request succeeds.",
    ),
    CorpusEntry(
        case=BenchCase(
            id="race-idempotent-lock",
            family="concurrency",
            task_context=(
                "Two workers can pick up the same queue message at once and "
                "double-charge the customer. State how the handler must be made safe."
            ),
            hidden_invariant=_grade_race_idempotency,
        ),
        good_output=(
            "Take a per-message lock and make the handler idempotent, so a redelivery "
            "can never double-charge."
        ),
        bad_output=(
            "The queue probably won't deliver the same message twice, so no extra "
            "work is needed."
        ),
    ),
    CorpusEntry(
        case=BenchCase(
            id="locale-decimal-parse",
            family="locale",
            task_context=(
                "Users in de-DE enter '1.234,56' and the form crashes. State the "
                "parsing rule the input layer must follow."
            ),
            hidden_invariant=_grade_locale_decimal,
        ),
        good_output=(
            "Parse the number with the locale's decimal separator: '1.234,56' is a "
            "thousands separator plus a comma decimal mark, never a hardcoded period."
        ),
        bad_output=(
            "Strip the dots and parse the rest as a float with the period as the "
            "decimal point."
        ),
    ),
    CorpusEntry(
        case=BenchCase(
            id="int-overflow-guard",
            family="overflow",
            task_context=(
                "A cast from an untrusted JSON number to a fixed-width integer "
                "silently wraps on huge values. State the guard the code must apply "
                "before casting."
            ),
            hidden_invariant=_grade_overflow_guard,
        ),
        good_output=(
            "Bounds-check the value against the target width before the cast so "
            "overflow is rejected instead of wrapping silently."
        ),
        bad_output="Cast the value directly; JSON numbers are always small enough.",
    ),
    CorpusEntry(
        case=BenchCase(
            id="partial-write-resume",
            family="partial-failure",
            task_context=(
                "A batch importer crashed halfway through a 10k-row file; the rerun "
                "duplicated every row it had already written. State the write "
                "discipline that prevents this."
            ),
            hidden_invariant=_grade_partial_write,
        ),
        good_output=(
            "Commit per batch with a resumable checkpoint, so a crashed rerun "
            "continues from the last committed batch instead of duplicating rows."
        ),
        bad_output=(
            "Rerun the whole file from the top; duplicates can be cleaned up manually "
            "afterwards."
        ),
    ),
    CorpusEntry(
        case=BenchCase(
            id="schema-drift-pin",
            family="integration-drift",
            task_context=(
                "A partner API added a field and the client began mis-parsing every "
                "response. State the contract the client must enforce against drift."
            ),
            hidden_invariant=_grade_schema_pin,
        ),
        good_output=(
            "Pin the expected schema version and fail fast when a response drifts "
            "from it, instead of guessing at new shapes."
        ),
        bad_output="Parse whatever fields arrive and hope the meaning stays the same.",
    ),
    CorpusEntry(
        case=BenchCase(
            id="monotonic-elapsed",
            family="time-zones",
            task_context=(
                "A profiler computes elapsed time from wall-clock timestamps and "
                "reports negative durations around DST and NTP adjustments. State what "
                "it must measure with instead."
            ),
            hidden_invariant=_grade_monotonic_clock,
        ),
        good_output=(
            "Measure elapsed time with a monotonic clock, never wall time, so DST and "
            "NTP steps cannot produce negative durations."
        ),
        bad_output="Subtract wall-clock timestamps; they are precise enough for profiling.",
    ),
    CorpusEntry(
        case=BenchCase(
            id="pagination-truncation",
            family="integration-drift",
            task_context=(
                "An exporter stopped at the first page when the upstream raised its "
                "page size, and nobody noticed for weeks. State the safeguard."
            ),
            hidden_invariant=_grade_pagination_truncation,
        ),
        good_output=(
            "Iterate pages until exhaustion and fail loudly on truncation, so an "
            "incomplete export can never pass silently."
        ),
        bad_output=(
            "Read the first page; it usually contains everything worth exporting."
        ),
    ),
    CorpusEntry(
        case=BenchCase(
            id="dup-submit-idempotency",
            family="retries",
            task_context=(
                "A timeout retry resubmitted a payment form and the customer was "
                "billed twice for one order. State the guard."
            ),
            hidden_invariant=_grade_dup_submit,
        ),
        good_output=(
            "Attach a client-generated idempotency key to the submission so a retry "
            "collapses into the original charge."
        ),
        bad_output="Remove the retry so the form can only ever be submitted once.",
    ),
)


def default_corpus() -> list[BenchCase]:
    """Return the full 10-case corpus in canonical order."""
    return [entry.case for entry in _CASES]


def entries() -> tuple[CorpusEntry, ...]:
    """Return all corpus entries with their gold-sanity outputs."""
    return _CASES
