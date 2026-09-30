"""Benchmark merge vs set Boolean search on the commonest and rarest terms."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from findex.index import build_index_from_path
from findex.search import search


def _term_extremes(index):
    by_df = sorted((len(posts), term) for term, posts in index.postings.items())
    rarest = [term for _df, term in by_df[:2]]
    common = [term for _df, term in reversed(by_df[-2:])]
    return common, rarest


def _bench(index, query: str, engine: str, repeats: int) -> float:
    started = perf_counter()
    for _ in range(repeats):
        search(index, query, engine=engine)
    return perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    index = build_index_from_path(args.root, limit=args.limit)
    common, rarest = _term_extremes(index)

    print("Two most common terms:")
    for term in common:
        print(f"  {term}: df={len(index.postings[term])}")
    print("Two rarest terms:")
    for term in rarest:
        print(f"  {term}: df={len(index.postings[term])}")
    print()
    print("Term | Group | Engine | Matches | Total time (s) | Avg (ms)")
    print("--- | --- | --- | ---: | ---: | ---:")

    for group, terms in (("common", common), ("rare", rarest)):
        for term in terms:
            for engine in ("merge", "set"):
                matches = search(index, term, engine=engine)
                elapsed = _bench(index, term, engine, args.repeats)
                print(
                    f"{term} | {group} | {engine} | {len(matches)} | "
                    f"{elapsed:.6f} | {elapsed / args.repeats * 1000:.4f}"
                )


if __name__ == "__main__":
    main()
