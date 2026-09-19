"""Structured logging: the one setup entry point, wired to the config module.

Every process configures logging exactly once, here:
:func:`configure_logging` reads the profile's
:class:`~evogine.config.LoggingSettings` and owns the root logger — no
``basicConfig`` anywhere, no module configuring its own handlers. Library
code (engine, gates, ops) only *emits*: ``logging.getLogger(__name__)`` and
:func:`log_event` with typed fields; the format and level are the config
file's decision, applied at the process entry point.

Two sinks, zero dependencies:

- ``text`` (default) — human-oriented single lines for interactive use.
- ``json`` — one JSON object per line (``ts``, ``level``, ``logger``,
  ``message``, plus the event's structured fields), for log shippers and
  the dashboard's log view.

Structured fields ride the standard ``extra`` mechanism under one key and
are validated to a flat ``str -> str`` mapping — a field that is not a
string is a call-site bug and raises, because silently stringifying would
let types drift where dashboards parse.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO

from evogine.config import LoggingSettings

#: The ``extra`` key carrying a structured event's typed fields.
_FIELDS_KEY = "evogine_fields"

_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


class StructuredFieldError(ValueError):
    """A structured log field is not a plain string (a call-site bug)."""


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    /,
    **fields: str,
) -> None:
    """Emit one structured event with typed fields.

    The single emit path for ops/library events: ``event`` names the
    machine-readable event type, ``message`` the human sentence, and every
    keyword a string field. Fields must be flat strings — anything else
    raises :class:`StructuredFieldError` at the call site rather than
    shipping type drift into the JSON sink.
    """
    for key, value in fields.items():
        if not isinstance(value, str):
            raise StructuredFieldError(
                f"log field {key!r} must be a string, got {type(value).__name__}"
            )
    logger.log(level, message, extra={_FIELDS_KEY: {"event": event, **fields}})


class JsonFormatter(logging.Formatter):
    """One JSON object per record, event fields merged at the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields: object = getattr(record, _FIELDS_KEY, None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human lines; structured fields appended as ``key=value`` pairs."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        fields: object = getattr(record, _FIELDS_KEY, None)
        if isinstance(fields, dict) and fields:
            pairs = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
            rendered = f"{rendered} {pairs}"
        if record.exc_info:
            rendered = f"{rendered}\n{self.formatException(record.exc_info)}"
        return rendered


def configure_logging(
    settings: LoggingSettings | None = None,
    *,
    stream: IO[str] | None = None,
) -> None:
    """Configure the process root logger from the config module's settings.

    The ONE logging entry point: call it once at the process boundary
    (the CLI's ``main``). Re-running it reconfigures — handlers are
    replaced, never stacked, so repeated entry (tests, embedded runs)
    cannot duplicate lines. ``settings=None`` means the environment
    defaults (``info``/``text``) — the same defaults
    :class:`~evogine.config.LoggingSettings` documents, so "no config
    file" and "a config without a logging section" behave identically.

    ``stream`` exists for tests (inject a buffer); production leaves it
    unset and logs to stderr — stdout is reserved for command output.
    """
    resolved = settings if settings is not None else LoggingSettings()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if resolved.format == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(_LEVELS[resolved.level])


__all__ = [
    "JsonFormatter",
    "StructuredFieldError",
    "TextFormatter",
    "configure_logging",
    "log_event",
]
