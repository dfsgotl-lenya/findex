"""Text snippets with case-insensitive term highlighting."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import cast


@lru_cache(maxsize=512)
def _read_text(path: str) -> str:
    """Full text behind a ``DocMeta.path``; cached (Lab 8 hotspot #2).

    Every search result used to reopen the JSONL file and ``json.loads`` the
    whole record. Popular documents come back again and again, so the most
    recent 512 texts (~9 MB for PEPs) stay in memory.
    """
    return _read_text_uncached(path)


def clear_text_cache() -> None:
    _read_text.cache_clear()


def _read_text_uncached(path: str) -> str:
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


def read_document_text(path: str) -> str:
    """Return the full text behind a ``DocMeta.path`` (``""`` if unreadable)."""
    return _read_text(path)


def source_url(path: str) -> str | None:
    """Extract the original URL stored in a JSONL ``DocMeta.path``, if any."""
    marker = "::url:"
    if marker not in path:
        return None
    url = path.split(marker, maxsplit=1)[1]
    return url if url.startswith(("http://", "https://")) else None


def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    escaped = sorted(
        (re.escape(term) for term in set(terms) if term),
        key=len,
        reverse=True,
    )
    if not escaped:
        return None
    # Match at the start of a word so "py" does not light up "happy".
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + ")", re.IGNORECASE)


def highlight_spans(text: str, terms: tuple[str, ...]) -> list[tuple[str, bool]]:
    """Split *text* into ``(fragment, is_match)`` pairs for safe HTML rendering.

    Unlike :func:`make_snippet` this never inserts markup into the text, so the
    web layer can escape every fragment and wrap only the matches in ``<mark>``.
    """
    pattern = _term_pattern(terms)
    if pattern is None or not text:
        return [(text, False)] if text else []
    spans: list[tuple[str, bool]] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            spans.append((text[cursor : match.start()], False))
        spans.append((match.group(0), True))
        cursor = match.end()
    if cursor < len(text):
        spans.append((text[cursor:], False))
    return spans


def snippet_spans(
    path: str,
    terms: tuple[str, ...],
    *,
    width: int = 120,
) -> list[tuple[str, bool]]:
    """Window of ±``width`` characters around the first match, as spans."""
    text = " ".join(_read_text(path).split())
    if not text:
        return []
    pattern = _term_pattern(terms)
    matches = list(pattern.finditer(text)) if pattern is not None else []
    if not matches:
        window = text[: 2 * width]
        return [(window + ("…" if len(text) > len(window) else ""), False)]
    # Centre the window on the spot where the most *different* query terms
    # appear close together, not simply on the first match of any term.
    # Sliding window over the (sorted) matches: O(number of matches).
    best = matches[0]
    best_count = 0
    in_window: Counter[str] = Counter()
    right = 0
    for left, candidate in enumerate(matches):
        while (
            right < len(matches) and matches[right].start() < candidate.start() + width
        ):
            in_window[matches[right].group(0).casefold()] += 1
            right += 1
        if len(in_window) > best_count:
            best, best_count = candidate, len(in_window)
        key = matches[left].group(0).casefold()
        in_window[key] -= 1
        if not in_window[key]:
            del in_window[key]
    anchor = best.start()
    start = max(0, anchor - width // 2)
    end = min(len(text), start + 2 * width)
    if start > 0:
        # Do not cut a word in half at the left edge.
        space = text.find(" ", start)
        start = space + 1 if 0 <= space < anchor else start
    spans = highlight_spans(text[start:end], terms)
    if start > 0:
        spans.insert(0, ("…", False))
    if end < len(text):
        spans.append(("…", False))
    return spans
