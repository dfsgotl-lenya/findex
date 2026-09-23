"""Lazy loading of text documents from a directory tree."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus document produced by :func:`iter_documents`."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield UTF-8 ``.txt`` documents one file at a time.

    The function is a generator: directory entries are discovered lazily and
    each file is opened only when it is about to be yielded. A decoding or I/O
    failure is logged and skipped so one broken file does not stop indexing.
    """
    root = Path(root)

    if root.is_file() and root.suffix.lower() == ".txt":
        paths: Iterator[Path] = iter((root,))
    elif root.is_dir():
        paths = root.rglob("*.txt")
    else:
        logger.warning("Corpus path is not a directory or .txt file: %s", root)
        return

    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                text = handle.read()
        except (OSError, UnicodeError) as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        yield Document(
            doc_id=path.relative_to(root).as_posix() if root.is_dir() else path.name,
            path=path,
            text=text,
        )
