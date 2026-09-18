"""Sanitized gate output: candidate code never speaks unfiltered to a verdict.

Security review: candidate output must not flow unfiltered into gate artifacts.
Everything leaving the sandbox is stripped of terminal escapes, secret-shaped
tokens, and exact host strings (staging roots, allowlisted env values), then
hard-capped in size.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_SECRET_SHAPED = (
    # Provider-style keys ("sk-...", per the H1 live-key lesson).
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    # Labeled assignments: api_key = "...", SECRET_TOKEN: "...".
    re.compile(
        r"(?i)(?:api|secret|token)[_-]?key[\"']?\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
)


def sanitize_output(
    text: str,
    *,
    max_chars: int = 20_000,
    redactions: Mapping[str, str] | None = None,
) -> str:
    """Redact, de-escape, and cap ``text`` before it enters gate artifacts.

    ``redactions`` maps exact host strings (staging roots, allowlisted env
    values) to placeholders and is applied first, so no later step can
    re-expose them. Secret-shaped tokens are replaced next; the result is
    finally capped at ``max_chars``.
    """
    for needle, placeholder in (redactions or {}).items():
        if needle:
            text = text.replace(needle, placeholder)
    text = _ANSI_ESCAPE.sub("", text)
    for pattern in _SECRET_SHAPED:
        text = pattern.sub("<redacted>", text)
    if len(text) > max_chars:
        dropped = len(text) - max_chars
        text = text[:max_chars] + f"\n<truncated: {dropped} more characters>"
    return text
