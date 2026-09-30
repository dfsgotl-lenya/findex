"""Boolean compatibility search and ranked Lab 3 search."""

from __future__ import annotations

import argparse
import logging
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal, overload

from .index import DocMeta, Index, Posting
from .query import And, Not, Or, Phrase, QueryNode, Term, parse
from .ranking import BM25, Scorer, SearchResult, TfIdf, score_terms, top_k
from .snippets import make_snippet
from .store import open_index
from .timing import timed

logger = logging.getLogger(__name__)


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
    return result + list(left[i:]) + list(right[j:])


def merge_and(left: Sequence[Posting], right: Sequence[Posting]) -> list[int]:
    return _and_ids(_posting_ids(left), _posting_ids(right))


def merge_or(left: Sequence[Posting], right: Sequence[Posting]) -> list[int]:
    return _or_ids(_posting_ids(left), _posting_ids(right))


def merge_not(all_doc_ids: Sequence[int], excluded: Sequence[Posting]) -> list[int]:
    excluded_ids = _posting_ids(excluded)
    result: list[int] = []
    j = 0
    for doc_id in all_doc_ids:
        while j < len(excluded_ids) and excluded_ids[j] < doc_id:
            j += 1
        if j >= len(excluded_ids) or excluded_ids[j] != doc_id:
            result.append(doc_id)
    return result


def _evaluate_merge(node: QueryNode, index: Index) -> list[int]:
    if isinstance(node, Term):
        return _posting_ids(index.postings.get(node.value, ()))
    if isinstance(node, Phrase):
        return sorted(node.evaluate(index))
    if isinstance(node, And):
        return _and_ids(
            _evaluate_merge(node.left, index),
            _evaluate_merge(node.right, index),
        )
    if isinstance(node, Or):
        return _or_ids(
            _evaluate_merge(node.left, index),
            _evaluate_merge(node.right, index),
        )
    if isinstance(node, Not):
        child_ids = _evaluate_merge(node.child, index)
        return _subtract_sorted(index.doc_ids, child_ids)
    raise TypeError(f"Unsupported query node: {type(node)!r}")


def _subtract_sorted(
    all_doc_ids: Sequence[int],
    excluded_ids: Sequence[int],
) -> list[int]:
    result: list[int] = []
    j = 0
    for doc_id in all_doc_ids:
        while j < len(excluded_ids) and excluded_ids[j] < doc_id:
            j += 1
        if j >= len(excluded_ids) or excluded_ids[j] != doc_id:
            result.append(doc_id)
    return result


def boolean_search(
    index: Index,
    query: str,
    *,
    engine: Literal["merge", "set"] = "merge",
) -> list[int]:
    """Evaluate a Boolean query with the Lab 2 merge or set engine."""
    if engine not in {"merge", "set"}:
        raise ValueError("engine must be 'merge' or 'set'")
    tree = parse(query)
    if engine == "merge":
        return _evaluate_merge(tree, index)
    return sorted(tree.evaluate(index))


def search_with_meta(
    index: Index,
    query: str,
    *,
    engine: Literal["merge", "set"] = "merge",
) -> list[tuple[int, DocMeta]]:
    ids = boolean_search(index, query, engine=engine)
    return [(doc_id, index.doc_meta[doc_id]) for doc_id in ids]


@overload
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer,
    k: int = 10,
    engine: Literal["merge", "set"] | None = None,
) -> list[SearchResult]: ...


@overload
def search(
    index: Index,
    query: str,
    *,
    scorer: None = None,
    k: int = 10,
    engine: Literal["merge", "set"] | None = None,
) -> list[int]: ...


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    engine: Literal["merge", "set"] | None = None,
) -> list[SearchResult] | list[int]:
    """Run Boolean compatibility search or ranked search."""
    if scorer is None:
        return boolean_search(index, query, engine=engine or "merge")
    if k < 1:
        raise ValueError("k must be >= 1")

    tree = parse(query)
    matched = index.evaluate_query(query)
    terms = tree.positive_terms()
    scores = score_terms(index, terms, matched, scorer)
    results = top_k(scores, index, k)

    enriched: list[SearchResult] = []
    for result in results:
        meta = index.doc_meta[result.doc_id]
        enriched.append(
            SearchResult(
                score=result.score,
                doc_id=result.doc_id,
                title=result.title,
                snippet=make_snippet(meta.path, terms, width=80),
            )
        )
    logger.info("query cache: %s", index.cache_info())
    return enriched


def _scorer_from_name(name: str) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def _format_mib(value: int) -> str:
    return f"{value / (1024**2):.2f} MiB"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Search a saved findex index.")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument(
        "--scorer",
        choices=("bm25", "tfidf"),
        default="bm25",
    )
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("-k must be >= 1")

    tracemalloc.start()
    started = perf_counter()
    with open_index(args.index, format=args.format) as index:
        loaded = perf_counter()
        results = search(
            index,
            args.query,
            scorer=_scorer_from_name(args.scorer),
            k=args.k,
            engine=args.engine,
        )
        finished = perf_counter()

        logger.info("index: %s", index)
        logger.info("scorer: %s", args.scorer)
        logger.info("query: %s", args.query)
        logger.info("load time: %.4f s", loaded - started)
        logger.info("search time: %.4f s", finished - loaded)
        logger.info("elapsed: %.4f s", finished - started)
        logger.info("results: %s", len(results))
        for result in results:
            logger.info("result: %s", result)

        first = index.cache_info()
        search(
            index,
            args.query,
            scorer=_scorer_from_name(args.scorer),
            k=args.k,
            engine=args.engine,
        )
        second = index.cache_info()
        logger.info("cache before repeat: %s", first)
        logger.info("cache after repeat: %s", second)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info("peak memory: %s", _format_mib(peak))


if __name__ == "__main__":
    main()
