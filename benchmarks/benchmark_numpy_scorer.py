"""Lab 8 M2: Lab 3 per-posting scorer vs. the NumPy scorer.

Times only ranking (score + top-k); parsing, Boolean evaluation and snippets
are done once outside the timed region, so the table compares like with like.

    uv run python benchmarks/benchmark_numpy_scorer.py data/peps.bin
"""

from __future__ import annotations

import argparse
import logging
import timeit
from pathlib import Path

from findex.arrays import vector_top_k
from findex.index import DocMeta, Index, Posting
from findex.query import parse
from findex.ranking import BM25, Scorer, TfIdf, score_terms, top_k
from findex.store import load


def _best_ms(fn: object, number: int, repeat: int = 7) -> float:
    times = timeit.repeat(fn, number=number, repeat=repeat)  # type: ignore[arg-type]
    return min(times) / number * 1000


def bench(index: Index, query: str, scorer: Scorer, k: int) -> tuple[int, float, float]:
    terms = parse(query).positive_terms()
    matched = index.evaluate_query(query)
    vectors = index.vector_scorer()

    def python_path() -> None:
        top_k(score_terms(index, terms, matched, scorer), index, k)

    def numpy_path() -> None:
        vector_top_k(vectors, terms, matched, scorer, k)

    expected = top_k(score_terms(index, terms, matched, scorer), index, k)
    actual = vector_top_k(vectors, terms, matched, scorer, k)
    assert [r.doc_id for r in expected] == [r.doc_id for r in actual]

    number = max(5, int(2000 / max(len(matched), 1)))
    return len(matched), _best_ms(python_path, number), _best_ms(numpy_path, number)


def replicate(index: Index, times: int) -> Index:
    """The same corpus ``times`` over (doc ids shifted): longer postings lists
    with an unchanged vocabulary, to see how the speedup scales with df."""
    n = index.num_docs
    postings = {
        term: [
            Posting(p.doc_id + copy * n, p.tf) for copy in range(times) for p in items
        ]
        for term, items in index.postings.items()
    }
    meta = {
        m.doc_id + copy * n: DocMeta(m.doc_id + copy * n, m.path, m.title, m.length)
        for copy in range(times)
        for m in index.doc_meta.values()
    }
    return Index(postings, meta)


def main() -> None:
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, nargs="?", default=Path("data/peps.bin"))
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--replicate", type=int, default=1)
    args = parser.parse_args()

    index = load(args.index)
    if args.replicate > 1:
        index = replicate(index, args.replicate)
    rare = min(
        (
            term
            for term in index
            if index.df(term) == 3 * args.replicate and term.isalpha()
        ),
        key=len,
    )
    queries = [
        ("common term", "python"),
        ("mid-frequency term", "generator"),
        (f"rare term (df={3 * args.replicate})", rare),
        ("3-term query", "python type annotations"),
        ("3-term OR query", "python OR type OR annotations"),
    ]
    for scorer in (BM25(), TfIdf()):
        print(f"\n{type(scorer).__name__}, k={args.k}, {index.num_docs} docs\n")
        print(
            "| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |"
        )
        print("|---|---:|---:|---:|---:|")
        for label, query in queries:
            matched, py_ms, np_ms = bench(index, query, scorer, args.k)
            print(
                f"| {label} `{query}` | {matched} | {py_ms:.3f} | {np_ms:.3f} | "
                f"{py_ms / np_ms:.1f}× |"
            )


if __name__ == "__main__":
    main()
