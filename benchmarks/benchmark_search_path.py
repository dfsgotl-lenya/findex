"""Lab 8: end-to-end ranked search (parse + Boolean + rank + snippets) per fix.

uv run python benchmarks/benchmark_search_path.py data/peps.bin
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from findex.index import Index
from findex.query import parse
from findex.ranking import BM25
from findex.search import rank
from findex.snippets import clear_text_cache, make_snippet
from findex.store import load

QUERIES = ("python type annotations", "python", "walrus")


def one(index: Index, query: str, *, vectorized: bool, cache: bool) -> None:
    if not cache:
        clear_text_cache()
    index.clear_cache()
    tree = parse(query)
    matched = index.evaluate_query(query)
    terms = tree.positive_terms()
    for result in rank(index, terms, matched, BM25(), 10, vectorized=vectorized):
        make_snippet(index.doc_meta[result.doc_id].path, terms)


def main() -> None:
    logging.disable(logging.CRITICAL)
    index = load(Path(sys.argv[1] if len(sys.argv) > 1 else "data/peps.bin"))
    index.vector_scorer()
    variants = [
        ("Lab 3 scorer, no text cache (= lab-07)", False, False),
        ("NumPy scorer, no text cache", True, False),
        ("NumPy scorer + text cache", True, True),
    ]
    print("| Variant | ms per search (mean of 600) |")
    print("|---|---:|")
    for label, vectorized, cache in variants:
        for query in QUERIES:
            one(index, query, vectorized=vectorized, cache=cache)
        rounds = 200
        started = time.perf_counter()
        for _ in range(rounds):
            for query in QUERIES:
                one(index, query, vectorized=vectorized, cache=cache)
        per_search = (time.perf_counter() - started) / (rounds * len(QUERIES))
        print(f"| {label} | {per_search * 1000:.2f} |")


if __name__ == "__main__":
    main()
