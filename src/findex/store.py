"""Serialization and context-manager support for Lab 3."""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .index import DocMeta, Index, Posting
from .timing import timed

logger = logging.getLogger(__name__)


def _detect_format(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        return fmt
    return "json" if path.suffix.lower() == ".json" else "pickle"


@timed
def save(index: Index, path: Path, *, format: str | None = None) -> None:
    """Save the index as pickle or JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _detect_format(path, format)

    if fmt == "pickle":
        with path.open("wb") as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fmt == "json":
        payload = {
            "doc_meta": {
                str(doc_id): {
                    "doc_id": meta.doc_id,
                    "path": meta.path,
                    "title": meta.title,
                    "length": meta.length,
                }
                for doc_id, meta in index.doc_meta.items()
            },
            "postings": {
                term: [
                    {
                        "doc_id": posting.doc_id,
                        "tf": posting.tf,
                        "positions": list(posting.positions),
                    }
                    for posting in postings
                ]
                for term, postings in index.postings.items()
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return

    raise ValueError("format must be 'pickle' or 'json'")


@timed
def load(path: Path, *, format: str | None = None) -> Index:
    """Load a trusted index from pickle or JSON."""
    path = Path(path)
    fmt = _detect_format(path, format)

    if fmt == "pickle":
        # Never unpickle an untrusted file: pickle can execute arbitrary code
        # during deserialization via custom object reducers.
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if not isinstance(result, Index):
            raise TypeError("pickle does not contain a findex Index")
        return result

    if fmt == "json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        doc_meta = {
            int(doc_id): DocMeta(
                doc_id=int(meta["doc_id"]),
                path=str(meta["path"]),
                title=str(meta["title"]),
                length=int(meta["length"]),
            )
            for doc_id, meta in payload["doc_meta"].items()
        }
        postings = {
            term: tuple(
                Posting(
                    doc_id=int(item["doc_id"]),
                    tf=int(item["tf"]),
                    positions=tuple(
                        int(pos) for pos in item.get("positions", [])
                    ),
                )
                for item in items
            )
            for term, items in payload["postings"].items()
        }
        return Index(postings=postings, doc_meta=doc_meta)

    raise ValueError("format must be 'pickle' or 'json'")


@contextmanager
def open_index(
    path: Path,
    *,
    format: str | None = None,
) -> Iterator[Index]:
    """Load an index and always close it when the context exits."""
    index = load(path, format=format)
    try:
        yield index
    finally:
        index.close()
        logger.info("index closed: %s", path)
