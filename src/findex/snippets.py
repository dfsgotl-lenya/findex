"""Text snippets with case-insensitive term highlighting."""

from __future__ import annotations

import re
from pathlib import Path


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def make_snippet(
    path: str,
    terms: tuple[str, ...],
    *,
    width: int = 80,
) -> str:
    """Return a ±``width`` character window around the earliest query match."""
    text = _read_text(path).replace("\r", "").replace("\n", " ")
    if not text or not terms:
        return ""

    lowered = text.casefold()
    best_position: int | None = None
    for term in terms:
        position = lowered.find(term.casefold())
        if position >= 0 and (best_position is None or position < best_position):
            best_position = position

    if best_position is None:
        return text[: 2 * width]

    start = max(0, best_position - width)
    end = min(len(text), best_position + width)
    snippet = text[start:end]

    escaped = sorted(
        (re.escape(term) for term in set(terms) if term),
        key=len,
        reverse=True,
    )
    if not escaped:
        return snippet
    pattern = re.compile("|".join(escaped), re.IGNORECASE)
    return pattern.sub(lambda match: f"**{match.group(0)}**", snippet)
