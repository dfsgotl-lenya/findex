"""Text snippets with case-insensitive term highlighting."""

from __future__ import annotations

import json
import re
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
    _flat_text.cache_clear()


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


@lru_cache(maxsize=1024)
def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    escaped = sorted(
        (re.escape(term) for term in set(terms) if term),
        key=len,
        reverse=True,
    )
    if not escaped:
        return None
    # Whole words only: "in" must not light up "introduced" or "within".
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)


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


@lru_cache(maxsize=512)
def _flat_text(path: str) -> tuple[str, str]:
    """``(text, lowered)``: whitespace collapsed once, lower-cased once (cached)."""
    text = " ".join(_read_text(path).split())
    return text, text.lower()


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _word_positions(
    text: str, lowered: str, term: str, start: int = 0, end: int | None = None
) -> list[int]:
    """Start offsets of *term* as a whole word inside ``[start, end)``.

    ``str.find`` on pre-lowered text runs in C and was ~25x faster than an
    IGNORECASE regex with look-arounds on 100 KB PEPs. If lowering changed the
    length (rare non-ASCII cases) offsets would not line up, so fall back to
    the regex there.
    """
    needle = term.lower()
    stop = len(lowered) if end is None else min(end, len(lowered))
    if len(lowered) != len(text) or not needle:
        pattern = _term_pattern((term,))
        if pattern is None:
            return []
        return [m.start() for m in pattern.finditer(text, start, stop)]
    found: list[int] = []
    at = lowered.find(needle, start, stop)
    while at >= 0:
        after = at + len(needle)
        if (at == 0 or not _is_word_char(lowered[at - 1])) and (
            after == len(lowered) or not _is_word_char(lowered[after])
        ):
            found.append(at)
        at = lowered.find(needle, at + 1, stop)
    return found


def _best_anchor(
    text: str, lowered: str, terms: tuple[str, ...], width: int
) -> int | None:
    """Position where the most *different* query terms occur within ``width``.

    Anchors are the occurrences of the term that is rarest *in this document*
    (``str.count`` in C picks it); for each anchor the other terms are only
    searched inside the window. (The first version slid a window over *every*
    match in Python — for "python" in a long PEP that is thousands of
    iterations per result; the profiler showed it as ~65 % of a web search.)
    """
    unique = [t for t in dict.fromkeys(terms) if t and t.lower() in lowered]
    if not unique:
        return None
    unique.sort(key=lambda term: lowered.count(term.lower()))
    rarest, others = unique[0], unique[1:]
    anchors = _word_positions(text, lowered, rarest)
    if not anchors:
        # Only sub-word hits for the rarest term; try the next one.
        return _best_anchor(text, lowered, tuple(others), width) if others else None
    best, best_count = anchors[0], 0
    for anchor in anchors[:200]:
        lo, hi = max(0, anchor - width), anchor + width
        count = 1 + sum(1 for t in others if _word_positions(text, lowered, t, lo, hi))
        if count > best_count:
            best, best_count = anchor, count
            if count == len(unique):
                break
    return best


def snippet_spans(
    path: str,
    terms: tuple[str, ...],
    *,
    width: int = 120,
) -> list[tuple[str, bool]]:
    """About ``2·width`` characters around the densest cluster of query terms."""
    text, lowered = _flat_text(path)
    if not text:
        return []
    anchor = _best_anchor(text, lowered, terms, width)
    if anchor is None:
        window = text[: 2 * width]
        return [(window + ("…" if len(text) > len(window) else ""), False)]
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
