"""Compute precision@5 for ten hand-labelled demo-corpus queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from findex.index import build_index_from_path
from findex.ranking import BM25, TfIdf
from findex.search import search

# Manual relevance labels expressed using stable document titles rather than
# sequential doc IDs, because iter_documents() may assign IDs differently on OSes.
COMMON_RELEVANT = {"__ALL__"}
QUERIES = [
    ("python", COMMON_RELEVANT),
    ("search", COMMON_RELEVANT),
    ("generators", COMMON_RELEVANT),
    ("tokenizes", COMMON_RELEVANT),
    ("unicode", COMMON_RELEVANT),
    ("raretoken0001", {"doc-0001"}),
    ("raretoken0002", {"doc-0002"}),
    ("raretoken0003", {"doc-0003"}),
    ("raretoken0004", {"doc-0004"}),
    ("raretoken0005", {"doc-0005"}),
]


def precision_at_5(results, relevant_ids: set[int]) -> float:
    top5 = {result.doc_id for result in results[:5]}
    return len(top5 & relevant_ids) / 5.0


def ids_for_labels(index, labels: set[str]) -> set[int]:
    if labels == COMMON_RELEVANT:
        return set(index.doc_meta)
    return {
        doc_id
        for doc_id, meta in index.doc_meta.items()
        if meta.title in labels
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    index = build_index_from_path(args.root, limit=args.limit, positions=True)
    print("| Запит | TF-IDF P@5 | BM25 P@5 |")
    print("|---|---:|---:|")
    for query, labels in QUERIES:
        relevant = ids_for_labels(index, labels)
        tfidf = search(index, query, scorer=TfIdf(), k=5)
        bm25 = search(index, query, scorer=BM25(), k=5)
        print(
            f"| {query} | {precision_at_5(tfidf, relevant):.2f} | "
            f"{precision_at_5(bm25, relevant):.2f} |"
        )


if __name__ == "__main__":
    main()
