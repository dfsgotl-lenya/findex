"""Build and persist an inverted index from the lazy Lab 1 corpus pipeline."""

from __future__ import annotations

import argparse
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .corpus import Document, iter_documents
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    """One term occurrence summary for a single document."""

    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Metadata needed to present a matching document to the user."""

    doc_id: int
    path: str
    title: str
    length: int


@dataclass(frozen=True, slots=True)
class InvertedIndex:
    """Immutable inverted index with postings sorted by document id."""

    postings: dict[str, tuple[Posting, ...]]
    doc_meta: dict[int, DocMeta]

    @property
    def doc_ids(self) -> tuple[int, ...]:
        """Return document IDs in sorted order."""
        return tuple(sorted(self.doc_meta))

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def posting_count(self) -> int:
        return sum(len(items) for items in self.postings.values())


def _flush_document(
    postings: defaultdict[str, list[Posting]],
    document: Document,
    doc_id: int,
    counts: Counter[str],
    length: int,
    doc_meta: dict[int, DocMeta],
) -> None:
    for term, tf in counts.items():
        postings[term].append(Posting(doc_id=doc_id, tf=tf))

    doc_meta[doc_id] = DocMeta(
        doc_id=doc_id,
        path=document.path.as_posix(),
        title=document.path.stem,
        length=length,
    )


def build_index(
    documents: Iterable[Document] | Iterator[Document],
    *,
    limit: int | None = None,
) -> InvertedIndex:
    """Build the index in one pass over a lazy document iterator.

    Only the current document's term frequencies are materialized temporarily.
    The inverted index and document metadata are the persistent result.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    count = 0
    for document in documents:
        if limit is not None and count >= limit:
            break

        counts: Counter[str] = Counter()
        length = 0
        for token in tokenize(document.text):
            counts[token] += 1
            length += 1

        _flush_document(postings, document, count, counts, length, doc_meta)
        count += 1

    frozen_postings = {
        term: tuple(sorted(items, key=lambda posting: posting.doc_id))
        for term, items in postings.items()
    }
    return InvertedIndex(postings=frozen_postings, doc_meta=doc_meta)


def build_index_from_path(root: Path, limit: int | None = None) -> InvertedIndex:
    """Build an index directly from the Lab 1 lazy corpus generator."""
    return build_index(iter_documents(root), limit=limit)


def _format_mib(value: int) -> str:
    return f"{value / (1024**2):.2f} MiB"


def _measure_build_and_save(
    root: Path,
    output: Path,
    limit: int | None,
    fmt: str | None,
) -> tuple[InvertedIndex, float, int, float, float]:
    """Measure the full build+save CLI path under tracemalloc."""
    from .store import save

    tracemalloc.start()
    start = perf_counter()
    index = build_index_from_path(root, limit=limit)
    built_at = perf_counter()
    save(index, output, format=fmt)
    finished = perf_counter()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, finished - start, peak, built_at - start, finished - built_at


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save a findex inverted index."
    )
    parser.add_argument("root", type=Path, help="Directory containing .txt files")
    parser.add_argument(
        "--out", type=Path, default=Path("index.bin"), help="Output index path"
    )
    parser.add_argument(
        "--format",
        choices=("pickle", "json"),
        default=None,
        help="Serialization format; inferred from .json when omitted",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Index only the first N documents"
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    index, total_elapsed, peak, build_elapsed, save_elapsed = _measure_build_and_save(
        args.root, args.out, args.limit, args.format
    )

    print(f"Documents: {len(index.doc_meta)}")
    print(f"Vocabulary: {index.vocabulary_size}")
    print(f"Postings: {index.posting_count}")
    print(f"Build time: {build_elapsed:.4f} s")
    print(f"Save time: {save_elapsed:.4f} s")
    print(f"Elapsed: {total_elapsed:.4f} s")
    print(f"Peak memory: {peak:,} bytes ({_format_mib(peak)})")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
