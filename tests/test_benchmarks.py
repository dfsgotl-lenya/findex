"""Lab 8: pytest-benchmark for ``search`` and ``build_index`` on the fixture
corpus. CI runs them with ``--benchmark-only`` and publishes the table, so a
slowdown shows up in every pull request. Normal ``pytest`` runs them once each
(``--benchmark-disable``) just to keep them working."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import cast

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import Document
from findex.index import Index, build_index
from findex.query import parse
from findex.ranking import BM25, TfIdf, score_terms, top_k
from findex.search import rank, search


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    # @timed logs every call at INFO; that would dominate a micro-benchmark.
    logging.disable(logging.INFO)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def documents(zipf_docs: list[Document]) -> list[Document]:
    return zipf_docs


def test_bench_build_index(
    benchmark: BenchmarkFixture, documents: list[Document]
) -> None:
    index = cast(Index, benchmark(build_index, documents, positions=False))
    assert index.num_docs == len(documents)


@pytest.mark.parametrize("query", ["w0", "w1 w2 w3", "w0 OR w1 OR w2"])
def test_bench_search_bm25(
    benchmark: BenchmarkFixture, zipf_index: Index, query: str
) -> None:
    def run() -> object:
        zipf_index.clear_cache()
        return search(zipf_index, query, scorer=BM25(), k=10)

    results = cast(list[object], benchmark(run))
    assert results


def test_bench_rank_numpy_vs_python(
    benchmark: BenchmarkFixture, zipf_index: Index
) -> None:
    query = "w0 OR w1 OR w2"
    terms = parse(query).positive_terms()
    matched = zipf_index.evaluate_query(query)
    zipf_index.vector_scorer()
    benchmark(rank, zipf_index, terms, matched, TfIdf(), 10)


def test_numpy_scorer_not_slower_than_lab3(zipf_index: Index) -> None:
    """A machine-independent regression guard: on long postings lists the
    vectorized path must stay clearly faster than the Lab 3 loop."""
    import timeit

    query = "w0 OR w1 OR w2"
    terms = parse(query).positive_terms()
    matched = zipf_index.evaluate_query(query)
    vectors = zipf_index.vector_scorer()
    assert vectors is not None
    python_s = min(
        timeit.repeat(
            lambda: top_k(
                score_terms(zipf_index, terms, matched, BM25()), zipf_index, 10
            ),
            number=20,
            repeat=5,
        )
    )
    numpy_s = min(
        timeit.repeat(
            lambda: rank(zipf_index, terms, matched, BM25(), 10), number=20, repeat=5
        )
    )
    assert numpy_s * 2 < python_s, (python_s, numpy_s)
