"""Lab 8 M2: the NumPy scorer must rank exactly like the Lab 3 scorer."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from findex.arrays import ArrayIndex, VectorScorer, top_k_arrays, vector_top_k
from findex.index import DocMeta, Index, Posting
from findex.query import parse
from findex.ranking import BM25, Scorer, TfIdf, score_terms, top_k
from findex.search import rank, search


@pytest.fixture
def corpus(zipf_index: Index) -> Index:
    return zipf_index


QUERIES = [
    "w0",  # in almost every document
    "w1 w2",
    "w0 OR w399",
    "w5 w17 w60",  # 3-term AND
    "w3 NOT w0",
    "w250",  # rare
    "(w10 OR w11) w12",
    "NOT w0",  # matched documents without positive terms -> all zero scores
]


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), BM25(k1=1.2, b=0.5)])
@pytest.mark.parametrize("query", QUERIES)
def test_numpy_ranking_identical_to_lab3(
    corpus: Index, query: str, scorer: Scorer
) -> None:
    tree = parse(query)
    matched = corpus.evaluate_query(query)
    terms = tree.positive_terms()
    for k in (1, 5, 10, len(matched) + 3):
        expected = top_k(score_terms(corpus, terms, matched, scorer), corpus, k)
        actual = vector_top_k(corpus.vector_scorer(), terms, matched, scorer, k)
        assert [r.doc_id for r in actual] == [r.doc_id for r in expected]
        # Same operation order and math.log -> bit-identical floats.
        assert [r.score for r in actual] == [r.score for r in expected]
        assert [r.title for r in actual] == [r.title for r in expected]


def test_rank_switch_gives_same_results(corpus: Index) -> None:
    query = "w1 OR w2 OR w3"
    tree = parse(query)
    matched = corpus.evaluate_query(query)
    fast = rank(corpus, tree.positive_terms(), matched, BM25(), 20)
    slow = rank(corpus, tree.positive_terms(), matched, BM25(), 20, vectorized=False)
    assert fast == slow


def test_search_uses_vectors_and_matches_fixture(tiny_index: Index) -> None:
    results = search(tiny_index, "python", scorer=BM25(), k=5)
    assert [r.doc_id for r in results] == [1, 0]


def test_array_layout(tiny_index: Index) -> None:
    arrays = ArrayIndex.from_index(tiny_index)
    assert arrays.doc_ids.dtype == np.int32
    assert arrays.tfs.dtype == np.int32
    assert arrays.doc_len.dtype == np.int32
    assert arrays.offsets[-1] == sum(len(p) for p in tiny_index.postings.values())
    ids, tfs = arrays.postings("python")
    assert ids.tolist() == [0, 1]
    assert tfs.tolist() == [1, 2]
    assert arrays.df("python") == tiny_index.df("python")
    assert arrays.df("missing") == 0
    assert arrays.postings("missing")[0].size == 0
    assert arrays.avg_doc_length == tiny_index.avg_doc_length


def test_npz_roundtrip(corpus: Index, tmp_path: Path) -> None:
    arrays = ArrayIndex.from_index(corpus)
    path = tmp_path / "index.npz"
    arrays.save(path)
    loaded = ArrayIndex.load(path)
    assert loaded.terms == arrays.terms
    assert loaded.titles == arrays.titles
    for name in ("offsets", "doc_ids", "tfs", "doc_len"):
        assert np.array_equal(getattr(loaded, name), getattr(arrays, name))
    a = VectorScorer(arrays).score(["w3", "w9"], range(corpus.num_docs), BM25())
    b = VectorScorer(loaded).score(["w3", "w9"], range(corpus.num_docs), BM25())
    assert np.array_equal(a[1], b[1])


def test_top_k_tie_break_by_doc_id() -> None:
    ids = np.array([5, 2, 9, 1, 7], dtype=np.int32)
    scores = np.array([1.0, 3.0, 3.0, 3.0, 0.5])
    assert top_k_arrays(ids, scores, 2) == [(1, 3.0), (2, 3.0)]
    assert top_k_arrays(ids, scores, 10)[-1] == (7, 0.5)
    assert top_k_arrays(ids, scores, 0) == []


def test_sparse_doc_ids_fall_back_to_python() -> None:
    sparse = Index(
        postings={"a": [Posting(3, 1), Posting(10, 2)], "b": [Posting(10, 1)]},
        doc_meta={
            3: DocMeta(3, "x", "three", 4),
            10: DocMeta(10, "y", "ten", 9),
        },
    )
    with pytest.raises(ValueError):
        sparse.vector_scorer()
    results = search(sparse, "a", scorer=BM25(), k=5)
    assert [r.doc_id for r in results] == [10, 3]


def test_posting_pickle_roundtrip_uses_constructor() -> None:
    posting = Posting(doc_id=4, tf=2, positions=(1, 7))
    assert pickle.loads(pickle.dumps(posting)) == posting
    assert posting.__reduce__() == (Posting, (4, 2, (1, 7)))


def test_vector_scorer_not_pickled(corpus: Index) -> None:
    corpus.vector_scorer()
    clone = pickle.loads(pickle.dumps(corpus))
    assert clone._vectors is None  # pyright: ignore[reportPrivateUsage]
    assert clone.search("w1", scorer=BM25(), k=3) == corpus.search(
        "w1", scorer=BM25(), k=3
    )
