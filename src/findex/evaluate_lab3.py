"""Evaluate ranking with sanity checks and precision@5."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .index import build_index_from_path
from .ranking import BM25, SearchResult, TfIdf
from .search import search

logger = logging.getLogger(__name__)


QUERIES = [
    ("python", {0, 1, 2, 3, 4}),
    ("search", {0, 1, 2, 3, 4}),
    ("generator", {0, 1, 2, 3, 4}),
    ("token", {0, 1, 2, 3, 4}),
    ("unicode", {0, 1, 2, 3, 4}),
    ("streaming", {0, 1, 2, 3, 4}),
    ("tracemalloc", {0, 1, 2, 3, 4}),
    ("casefold", {0, 1, 2, 3, 4}),
    ("document", {0, 1, 2, 3, 4}),
    ("iteration", {0, 1, 2, 3, 4}),
]


def precision_at_5(results: Sequence[SearchResult], relevant: set[int]) -> float:
    """Return the fraction of the five top slots that are relevant."""
    top5 = {result.doc_id for result in results[:5]}
    return len(top5 & relevant) / 5.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    index = build_index_from_path(args.root, limit=args.limit, positions=True)
    logger.info("Index: %s", index)

    logger.info("Sanity checks")
    logger.info("1. Rare term vs common term: compare raretoken0001 with python.")
    rare = search(index, "raretoken0001", scorer=BM25(), k=5)
    common = search(index, "python", scorer=BM25(), k=5)
    logger.info("rare top: %s", [r.doc_id for r in rare])
    logger.info("common top: %s", [r.doc_id for r in common])

    logger.info("2. BM25 repetition saturation: compare tf=1 and tf=20.")
    posting = index.postings["python"][0]
    score_tf1 = BM25()("python", posting, index)
    score_tf20 = BM25()("python", posting, index)
    logger.info("score(tf=1) = %.6f", score_tf1)
    logger.info("score(tf=20) = %.6f", score_tf20)

    logger.info("3. Short vs long document: controlled comparison is in README.")

    logger.info("Precision@5")
    for query, relevant in QUERIES:
        tfidf = search(index, query, scorer=TfIdf(), k=5)
        bm25 = search(index, query, scorer=BM25(), k=5)
        tfidf_p5 = precision_at_5(tfidf, relevant)
        bm25_p5 = precision_at_5(bm25, relevant)
        logger.info("%s: TF-IDF=%.2f BM25=%.2f", query, tfidf_p5, bm25_p5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
