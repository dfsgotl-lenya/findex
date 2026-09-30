"""Streaming corpus statistics and eager-vs-lazy benchmark."""

from __future__ import annotations

import argparse
import logging
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path
from time import perf_counter

from .corpus import Document, iter_documents
from .tokenize import tokenize


def _limited_documents(root: Path, limit: int | None) -> Iterator[Document]:
    docs = iter_documents(root)
    return islice(docs, limit) if limit is not None else docs


def collect_stats(
    root: Path, limit: int | None = None
) -> tuple[int, int, Counter[str]]:
    """Collect document/token/vocabulary statistics in one lazy pass."""
    doc_count = 0
    token_count = 0
    counts: Counter[str] = Counter()

    for document in _limited_documents(root, limit):
        doc_count += 1
        for token in tokenize(document.text):
            token_count += 1
            counts[token] += 1

    return doc_count, token_count, counts


def eager_stats(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    """Deliberately eager implementation for the lab comparison."""
    documents = list(_limited_documents(root, limit))
    token_lists = [list(tokenize(document.text)) for document in documents]
    counts = Counter(token for tokens in token_lists for token in tokens)
    return len(documents), sum(len(tokens) for tokens in token_lists), counts


def measure(
    func: Callable[[Path, int | None], tuple[int, int, Counter[str]]],
    root: Path,
    limit: int | None,
) -> tuple[tuple[int, int, Counter[str]], float, int]:
    """Run *func* under tracemalloc and perf_counter."""
    tracemalloc.start()
    start = perf_counter()
    result = func(root, limit)
    elapsed = perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _format_mib(value: int) -> str:
    return f"{value / (1024**2):.2f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream findex corpus statistics.")
    parser.add_argument("root", type=Path, help="Directory containing .txt files")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    (doc_count, token_count, counts), elapsed, peak = measure(
        collect_stats, args.root, args.limit
    )

    logger = logging.getLogger(__name__)
    logger.info("documents: %s", doc_count)
    logger.info("tokens: %s", token_count)
    logger.info("vocabulary: %s", len(counts))
    logger.info("top 50 terms: %s", counts.most_common(50))
    logger.info("elapsed: %.4f s", elapsed)
    logger.info("peak memory: %s", _format_mib(peak))


if __name__ == "__main__":
    main()
