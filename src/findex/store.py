"""Serialization helpers for the inverted index."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from .index import DocMeta, InvertedIndex, Posting


def _detect_format(path: Path, format: str | None) -> str:
    if format is not None:
        return format
    return "json" if path.suffix.lower() == ".json" else "pickle"


def save(index: InvertedIndex, path: Path, *, format: str | None = None) -> None:
    """Save *index* as pickle or JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _detect_format(path, format)

    if fmt == "pickle":
        # WARNING: pickle.load() executes constructors from the serialized
        # object graph. Never load a pickle received from an untrusted source.
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
                    {"doc_id": posting.doc_id, "tf": posting.tf}
                    for posting in postings
                ]
                for term, postings in index.postings.items()
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        return

    raise ValueError("format must be 'pickle' or 'json'")


def load(path: Path, *, format: str | None = None) -> InvertedIndex:
    """Load an index from pickle or JSON."""
    path = Path(path)
    fmt = _detect_format(path, format)

    if fmt == "pickle":
        # WARNING: only load pickle files produced and trusted by this project.
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if not isinstance(result, InvertedIndex):
            raise TypeError("pickle does not contain a findex InvertedIndex")
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
                Posting(doc_id=int(item["doc_id"]), tf=int(item["tf"]))
                for item in items
            )
            for term, items in payload["postings"].items()
        }
        return InvertedIndex(postings=postings, doc_meta=doc_meta)

    raise ValueError("format must be 'pickle' or 'json'")
