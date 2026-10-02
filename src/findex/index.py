"""Object-oriented inverted index for Lab 3."""

from __future__ import annotations

import argparse
import logging
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

from .corpus import Document, iter_documents
from .timing import timed
from .tokenize import tokenize

if TYPE_CHECKING:
    from .ranking import Scorer, SearchResult

logger = logging.getLogger(__name__)


class CacheInfo(Protocol):
    """Structural type for ``functools.lru_cache`` statistics."""

    @property
    def hits(self) -> int: ...

    @property
    def misses(self) -> int: ...

    @property
    def maxsize(self) -> int | None: ...

    @property
    def currsize(self) -> int: ...


class QueryCache(Protocol):
    """Callable query cache exposing the standard LRU cache controls."""

    def __call__(self, query: str) -> frozenset[int]: ...

    def cache_info(self) -> CacheInfo: ...

    def cache_clear(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Posting:
    """A term posting for one document."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Metadata required for ranking and result presentation."""

    doc_id: int
    path: str
    title: str
    length: int


class Index(Mapping[str, tuple[Posting, ...]]):
    """Read-mostly inverted index implementing normal mapping protocols."""

    def __init__(
        self,
        postings: Mapping[str, Iterable[Posting]],
        doc_meta: Mapping[int, DocMeta],
    ) -> None:
        self._postings: dict[str, tuple[Posting, ...]] = {
            str(term): tuple(sorted(items, key=lambda posting: posting.doc_id))
            for term, items in sorted(postings.items(), key=lambda item: str(item[0]))
        }
        self._doc_meta: dict[int, DocMeta] = dict(sorted(doc_meta.items()))
        self._closed = False
        self._cached_query = self._make_cached_query()

    @property
    def postings(self) -> Mapping[str, tuple[Posting, ...]]:
        """Return the posting mapping for read-only inspection."""
        return self._postings

    @property
    def doc_meta(self) -> Mapping[int, DocMeta]:
        """Return document metadata for read-only inspection."""
        return self._doc_meta

    @property
    def num_docs(self) -> int:
        return len(self._doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        """Average document length, computed once per index instance."""
        if not self._doc_meta:
            return 0.0
        return sum(meta.length for meta in self._doc_meta.values()) / self.num_docs

    @property
    def doc_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._doc_meta))

    def doc_length(self, doc_id: int) -> int:
        return self._doc_meta[doc_id].length

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def __len__(self) -> int:
        """Return the number of indexed terms (vocabulary size)."""
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __getitem__(self, term: str) -> tuple[Posting, ...]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):,}, docs={self.num_docs:,})"

    def close(self) -> None:
        """Mark the loaded index as closed."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def cache_info(self) -> CacheInfo:
        """Return cache statistics for the query-to-document-ID path."""
        return self._cached_query.cache_info()

    def clear_cache(self) -> None:
        self._cached_query.cache_clear()

    def evaluate_query(self, query: str) -> frozenset[int]:
        """Parse and evaluate a query, caching the resulting document IDs."""
        return self._cached_query(query)

    def search(
        self,
        query: str,
        *,
        scorer: Scorer | None = None,
        k: int = 10,
    ) -> list[SearchResult] | list[int]:
        """Convenience wrapper around :func:`findex.search.search`."""
        from .ranking import BM25
        from .search import search

        return search(self, query, scorer=scorer or BM25(), k=k)

    def __getstate__(self) -> dict[str, object]:
        return {
            "postings": self._postings,
            "doc_meta": self._doc_meta,
            "closed": self._closed,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self._postings = cast(dict[str, tuple[Posting, ...]], state["postings"])
        self._doc_meta = cast(dict[int, DocMeta], state["doc_meta"])
        self._closed = bool(state.get("closed", False))
        self._cached_query = self._make_cached_query()

    def _make_cached_query(self) -> QueryCache:
        from functools import lru_cache

        from .query import parse

        @lru_cache(maxsize=256)
        def cached(query: str) -> frozenset[int]:
            return frozenset(parse(query).evaluate(self))

        return cached


# Compatibility alias from Lab 2.
InvertedIndex = Index


def _limited_documents(
    documents: Iterable[Document], limit: int | None
) -> Iterator[Document]:
    """Yield at most ``limit`` documents without materializing the input."""
    for count, document in enumerate(documents):
        if limit is not None and count >= limit:
            break
        yield document


@timed
def build_index(
    documents: Iterable[Document],
    *,
    limit: int | None = None,
    positions: bool = False,
) -> Index:
    """Build an inverted index in one pass over the lazy document stream."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, document in enumerate(_limited_documents(documents, limit)):
        counts: Counter[str] = Counter()
        position_map: defaultdict[str, list[int]] = defaultdict(list)
        length = 0

        for token in tokenize(document.text):
            counts[token] += 1
            if positions:
                position_map[token].append(length)
            length += 1

        for term, tf in counts.items():
            postings[term].append(
                Posting(
                    doc_id=doc_id,
                    tf=tf,
                    positions=(tuple(position_map[term]) if positions else ()),
                )
            )

        doc_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            path=document.path.as_posix(),
            title=document.path.stem,
            length=length,
        )

    return Index(postings=postings, doc_meta=doc_meta)


def build_index_from_path(
    root: Path,
    limit: int | None = None,
    *,
    positions: bool = False,
) -> Index:
    """Build an index from Lab 1's lazy document generator."""
    return build_index(
        iter_documents(root),
        limit=limit,
        positions=positions,
    )


def _format_mib(value: int) -> str:
    return f"{value / (1024**2):.2f} MiB"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build and save a findex inverted index."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Store token positions for phrase queries.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    from .store import save

    tracemalloc.start()
    started = perf_counter()
    index = build_index_from_path(
        args.root,
        limit=args.limit,
        positions=args.positions,
    )
    built = perf_counter()
    save(index, args.out, format=args.format)
    finished = perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info("index: %s", index)
    logger.info("documents: %s", index.num_docs)
    logger.info("vocabulary: %s", len(index))
    logger.info("average document length: %.2f", index.avg_doc_length)
    logger.info("build time: %.4f s", built - started)
    logger.info("save time: %.4f s", finished - built)
    logger.info("elapsed: %.4f s", finished - started)
    logger.info("peak memory: %s", _format_mib(peak))
    logger.info("saved: %s", args.out)


if __name__ == "__main__":
    main()
