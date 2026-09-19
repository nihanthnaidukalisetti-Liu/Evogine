"""Structured logging: the one entry point, config-driven sinks, typed fields."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from evogine.config import LoggingSettings
from evogine.ops.logging_setup import (
    JsonFormatter,
    StructuredFieldError,
    TextFormatter,
    configure_logging,
    log_event,
)


def _buffer_logger(stream: StringIO) -> logging.Logger:
    return logging.getLogger("evogine.test.harness")


@pytest.fixture(autouse=True)
def _restore_root_logging() -> None:
    """Leave the root logger exactly as pytest set it after every test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_json_sink_emits_one_object_per_event() -> None:
    stream = StringIO()
    configure_logging(LoggingSettings(level="info", format="json"), stream=stream)
    logger = _buffer_logger(stream)
    log_event(logger, logging.INFO, "store.transition", "propose lr-1", record_id="lr-1")
    payload = json.loads(stream.getvalue())
    assert payload["event"] == "store.transition"
    assert payload["record_id"] == "lr-1"
    assert payload["level"] == "info"
    assert payload["logger"] == "evogine.test.harness"
    assert payload["message"] == "propose lr-1"
    assert "ts" in payload


def test_text_sink_appends_typed_fields() -> None:
    stream = StringIO()
    configure_logging(LoggingSettings(level="info", format="text"), stream=stream)
    logger = _buffer_logger(stream)
    log_event(logger, logging.INFO, "gate.decision", "gate verdict", stage="replay")
    line = stream.getvalue()
    assert "gate verdict" in line
    assert "stage=replay" in line
    assert "event=gate.decision" in line


def test_level_setting_filters_below_it() -> None:
    stream = StringIO()
    configure_logging(LoggingSettings(level="warning", format="json"), stream=stream)
    logger = _buffer_logger(stream)
    log_event(logger, logging.INFO, "store.transition", "not logged")
    assert stream.getvalue() == ""
    log_event(logger, logging.ERROR, "gate.error", "logged")
    assert "logged" in stream.getvalue()


def test_reconfiguration_replaces_handlers_never_stacks() -> None:
    stream = StringIO()
    configure_logging(LoggingSettings(format="json"), stream=stream)
    configure_logging(LoggingSettings(format="json"), stream=stream)
    root = logging.getLogger()
    handlers = [handler for handler in root.handlers if isinstance(handler, logging.StreamHandler)]
    assert len(handlers) == 1
    logger = _buffer_logger(stream)
    log_event(logger, logging.INFO, "store.transition", "once")
    assert stream.getvalue().count("once") == 1


def test_defaults_without_settings_are_info_text() -> None:
    stream = StringIO()
    configure_logging(None, stream=stream)
    logger = _buffer_logger(stream)
    log_event(logger, logging.INFO, "store.transition", "visible")
    assert "visible" in stream.getvalue()
    stream2 = StringIO()
    configure_logging(LoggingSettings(level="error", format="json"), stream=stream2)
    assert logging.getLogger().level == logging.ERROR


def test_non_string_fields_are_call_site_bugs() -> None:
    logger = _buffer_logger(StringIO())
    with pytest.raises(StructuredFieldError, match="must be a string"):
        log_event(logger, logging.INFO, "bad", "message", count=3)  # type: ignore[arg-type]


def test_json_formatter_merges_event_fields_at_top_level() -> None:
    record = logging.LogRecord(
        name="evogine",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.evogine_fields = {"event": "custom", "store": "core"}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "custom"
    assert payload["store"] == "core"


def test_text_formatter_without_fields_has_no_trailing_space() -> None:
    record = logging.LogRecord(
        name="evogine",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="plain",
        args=(),
        exc_info=None,
    )
    rendered = TextFormatter().format(record)
    assert "plain" in rendered
    assert not rendered.endswith(" ")
