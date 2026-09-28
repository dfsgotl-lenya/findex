"""Boolean search over sorted inverted-index postings."""

from __future__ import annotations

import argparse
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .index import DocMeta, InvertedIndex, Posting
from .store import load
from .tokenize import tokenize


def _posting_ids(postings: Sequence[Posting]) -> list[int]:
    return [posting.doc_id for posting in postings]


def _and_ids(left: Sequence[int], right: Sequence[int]) -> list[int]:
    i = j = 0
    result: list[int] = []
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


def _or_ids(left: Sequence[int], right: Sequence[int]) -> list[int]:
    i = j = 0
    result: list[int] = []
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_and(left: Sequence[Posting], right: Sequence[Posting]) -> list[int]:
    """Intersect two postings lists with a two-pointer merge."""
    return _and_ids(_posting_ids(left), _posting_ids(right))


def merge_or(left: Sequence[Posting], right: Sequence[Posting]) -> list[int]:
    """Union two postings lists with a two-pointer merge."""
    return _or_ids(_posting_ids(left), _posting_ids(right))


def merge_not(all_doc_ids: Sequence[int], excluded: Sequence[Posting]) -> list[int]:
    """Return all document IDs except those in *excluded*, both sorted."""
    excluded_ids = _posting_ids(excluded)
    result: list[int] = []
    j = 0
    for doc_id in all_doc_ids:
        while j < len(excluded_ids) and excluded_ids[j] < doc_id:
            j += 1
        if j >= len(excluded_ids) or excluded_ids[j] != doc_id:
            result.append(doc_id)
    return result


def search(index: InvertedIndex, query: str, *, engine: str = "merge") -> list[int]:
    """Evaluate a simple left-to-right Boolean query.

    Adjacent terms imply AND. ``OR`` performs union and ``NOT`` subtracts the
    right-hand term. Query words use the same normalization rules as indexing.
    """
    if engine not in {"merge", "set"}:
        raise ValueError("engine must be 'merge' or 'set'")

    terms = list(tokenize(query))
    if not terms:
        return []

    current: list[int] = []
    operator = "AND"
    initialized = False

    for term in terms:
        if term.upper() in {"OR", "NOT"} and initialized:
            operator = term.upper()
            continue

        posting_list = index.postings.get(term, ())
        right_ids = _posting_ids(posting_list)

        if not initialized:
            current = right_ids
            initialized = True
            operator = "AND"
            continue

        if engine == "merge":
            if operator == "AND":
                current = _and_ids(current, right_ids)
            elif operator == "OR":
                current = _or_ids(current, right_ids)
            else:
                current = _and_ids(
                    current,
                    merge_not(index.doc_ids, posting_list),
                )
        elif operator == "AND":
            current = sorted(set(current) & set(right_ids))
        elif operator == "OR":
            current = sorted(set(current) | set(right_ids))
        else:
            current = sorted(set(current) - set(right_ids))
        operator = "AND"

    return current


def search_with_meta(
    index: InvertedIndex, query: str, *, engine: str = "merge"
) -> list[tuple[int, DocMeta]]:
    """Return matching IDs together with display metadata."""
    result = search(index, query, engine=engine)
    return [(doc_id, index.doc_meta[doc_id]) for doc_id in result]


def _format_mib(value: int) -> str:
    return f"{value / (1024**2):.2f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a saved findex index.")
    parser.add_argument("index", type=Path, help="Path to saved index")
    parser.add_argument("query", help="Boolean query")
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    args = parser.parse_args()

    tracemalloc.start()
    started = perf_counter()
    load_started = perf_counter()
    index = load(args.index, format=args.format)
    loaded = perf_counter()
    results = search(index, args.query, engine=args.engine)
    finished = perf_counter()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Engine: {args.engine}")
    print(f"Query: {args.query}")
    print(f"Load time: {loaded - load_started:.4f} s")
    print(f"Search time: {finished - loaded:.4f} s")
    print(f"Elapsed: {finished - started:.4f} s")
    print(f"Peak memory: {peak:,} bytes ({_format_mib(peak)})")
    print(f"Results: {len(results)}")
    for doc_id in results:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id}] {meta.title} — {meta.path} ({meta.length} tokens)")


if __name__ == "__main__":
    main()
