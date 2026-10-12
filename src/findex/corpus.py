"""Lazy loading of text and crawler JSONL documents."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus document produced by :func:`iter_documents`."""

    doc_id: str
    path: Path
    text: str
    title: str | None = None
    source_url: str | None = None
    jsonl_offset: int | None = None


@dataclass(frozen=True, slots=True)
class JsonlDocumentRef:
    """Lightweight pointer to one JSONL line; it never contains page text."""

    path: Path
    line_number: int
    byte_offset: int


DocumentRef = Path | JsonlDocumentRef


def _eligible_paths(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".txt", ".jsonl"}:
        return [root]
    if root.is_dir():
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".jsonl"}
        )
    return []


def iter_document_refs(root: Path, limit: int | None = None) -> Iterator[DocumentRef]:
    """Yield text paths and JSONL byte-offset refs without retaining bodies."""
    root = Path(root)
    if limit is not None and limit <= 0:
        return
    emitted = 0
    for path in _eligible_paths(root):
        if path.suffix.lower() == ".txt":
            yield path
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            continue

        try:
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    try:
                        parsed: object = json.loads(
                            line.decode("utf-8", errors="strict")
                        )
                        if not isinstance(parsed, dict):
                            raise ValueError(
                                "JSONL record must be an object with a string "
                                "text field"
                            )
                        record = cast(dict[str, object], parsed)
                        if not isinstance(record.get("text"), str):
                            raise ValueError(
                                "JSONL record must be an object with a string "
                                "text field"
                            )
                    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                        logger.warning("Skipping %s:%d: %s", path, line_number, exc)
                        continue
                    yield JsonlDocumentRef(path, line_number, offset)
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return
        except OSError as exc:
            logger.warning("Skipping %s: %s", path, exc)


def read_document_reference(
    reference: DocumentRef, root: Path | None = None
) -> Document:
    """Read one reference from disk, parsing only its JSONL record if needed."""
    if isinstance(reference, JsonlDocumentRef):
        with reference.path.open("rb") as handle:
            handle.seek(reference.byte_offset)
            raw_line = handle.readline()
        parsed: object = json.loads(raw_line.decode("utf-8", errors="strict"))
        if not isinstance(parsed, dict):
            raise ValueError(
                f"JSONL row {reference.line_number} in "
                f"{reference.path} is not an object"
            )
        record = cast(dict[str, object], parsed)
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"JSONL row {reference.line_number} in {reference.path} "
                "has no text field"
            )
        url_value = record.get("url")
        url = url_value if isinstance(url_value, str) and url_value else None
        title_value = record.get("title")
        title = title_value if isinstance(title_value, str) and title_value else None
        fallback_id = f"{reference.path.name}:{reference.line_number}"
        doc_id = url or str(record.get("doc_id") or fallback_id)
        return Document(
            doc_id=doc_id,
            path=reference.path,
            text=text,
            title=title,
            source_url=url,
            jsonl_offset=reference.byte_offset,
        )

    path = Path(reference)
    text = path.read_text(encoding="utf-8", errors="strict")
    if root is not None and Path(root).is_dir():
        doc_id = path.relative_to(root).as_posix()
    else:
        doc_id = path.name
    return Document(doc_id=doc_id, path=path, text=text)


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield UTF-8 ``.txt`` or JSONL documents lazily, one document at a time.

    JSONL records are expected to contain at least ``text``; crawler output also
    contains ``url``, ``title``, ``fetched_at``, ``status``, ``bytes`` and
    ``elapsed``. One JSONL line becomes one search document. Read/parse failures
    are logged and skipped so a single damaged line does not stop indexing.
    """
    root = Path(root)
    if not _eligible_paths(root):
        logger.warning(
            "Corpus path is not a directory, .txt file, or .jsonl file: %s", root
        )
        return

    for reference in iter_document_refs(root):
        try:
            yield read_document_reference(reference, root=root)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            location = (
                f"{reference.path}:{reference.line_number}"
                if isinstance(reference, JsonlDocumentRef)
                else str(reference)
            )
            logger.warning("Skipping %s: %s", location, exc)
