"""Text snippets with case-insensitive term highlighting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast


def _read_text(path: str) -> str:
    marker = "::jsonl-offset:"
    try:
        if marker in path:
            jsonl_path, details = path.split(marker, maxsplit=1)
            offset_text = details.split("::url:", maxsplit=1)[0]
            with Path(jsonl_path).open("rb") as handle:
                handle.seek(int(offset_text))
                parsed: object = json.loads(
                    handle.readline().decode("utf-8", errors="strict")
                )
            if not isinstance(parsed, dict):
                return ""
            record = cast(dict[str, object], parsed)
            text = record.get("text", "")
            return text if isinstance(text, str) else ""
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
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
