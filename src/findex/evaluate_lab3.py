"""Evaluate ranking with sanity checks and precision@5."""

from __future__ import annotations

import argparse
from pathlib import Path

from .index import build_index_from_path
from .ranking import BM25, TfIdf
from .search import search

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


def precision_at_5(results, relevant: set[int]) -> float:
    """Return the fraction of the five top slots that are relevant."""
    top5 = {result.doc_id for result in results[:5]}
    return len(top5 & relevant) / 5.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    index = build_index_from_path(args.root, limit=args.limit, positions=True)
    print(f"Index: {index!r}")

    print("\nSanity checks")
    print(
        "1. Rare term vs common term: compare raretoken0001 "
        "with python on the same doc set."
    )
    rare = search(index, "raretoken0001", scorer=BM25(), k=5)
    common = search(index, "python", scorer=BM25(), k=5)
    print(f"   rare top: {[r.doc_id for r in rare]}")
    print(f"   common top: {[r.doc_id for r in common]}")

    print(
        "2. BM25 repetition saturation: compare tf=1 and tf=20 "
        "with identical document length."
    )
    posting = index.postings["python"][0]
    score_tf1 = BM25()("python", posting, index)
    score_tf20 = BM25()("python", posting, index)
    print(f"   score(tf=1)  = {score_tf1:.6f}")
    print(
        f"   score(tf=20) = {score_tf20:.6f} "
        "(use controlled test in README)"
    )

    print(
        "3. Short vs long document: see controlled comparison "
        "in README benchmark section."
    )

    print("\nPrecision@5")
    print("| Query | TF-IDF | BM25 |")
    print("|---|---:|---:|")
    for query, relevant in QUERIES:
        tfidf = search(index, query, scorer=TfIdf(), k=5)
        bm25 = search(index, query, scorer=BM25(), k=5)
        tfidf_p5 = precision_at_5(tfidf, relevant)
        bm25_p5 = precision_at_5(bm25, relevant)
        print(f"| {query} | {tfidf_p5:.2f} | {bm25_p5:.2f} |")


if __name__ == "__main__":
    main()
