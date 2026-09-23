"""Streaming Unicode-aware tokenization rules for findex."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# Unicode word characters, excluding underscore, with an optional internal
# apostrophe. The pattern keeps digits and letters together, e.g. ``python3``.
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens from *text* without building a token list.

    Normalization rules:
    - Unicode NFC normalization is applied first.
    - ``casefold()`` provides caseless matching (stronger than ``lower()``).
    - Letters and digits are kept; mixed tokens such as ``python3`` stay whole.
    - An apostrophe is kept only when it is internal to a token, supporting
      both ASCII ``'`` and typographic ``’`` apostrophes (e.g. ``п'ять``).
    - Hyphens are separators, so ``state-of-the-art`` becomes three tokens.
    - Punctuation and symbols are separators.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in _WORD_RE.finditer(normalized):
        yield match.group(0)
