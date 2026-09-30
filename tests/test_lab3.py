from pathlib import Path
from typing import Literal

import pytest

from findex.corpus import Document
from findex.index import Index, build_index
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.ranking import BM25, SearchResult, TfIdf
from findex.search import search
from findex.store import load, open_index, save
from findex.timing import timed


def make_docs() -> list[Document]:
    return [
        Document("a.txt", Path("a.txt"), "Python async event loop search"),
        Document("b.txt", Path("b.txt"), "Python search search engine"),
        Document("c.txt", Path("c.txt"), "java engine"),
    ]


def test_index_python_protocols_and_cached_property() -> None:
    index = build_index(make_docs(), positions=True)
    assert len(index) == len(index.postings)
    assert "python" in index
    assert [posting.doc_id for posting in index["python"]] == [0, 1]
    assert list(index)
    assert repr(index).startswith("Index(terms=")
    assert index.num_docs == 3
    assert index.df("python") == 2
    first = index.avg_doc_length
    assert index.avg_doc_length == first
    assert "avg_doc_length" in index.__dict__


def test_missing_term_is_key_error() -> None:
    index = build_index(make_docs())
    with pytest.raises(KeyError):
        _ = index["missing"]


def test_query_tree_equality_and_operator_overloads() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert (Term("a") & (Term("b") | Term("c"))) == And(
        Term("a"),
        Or(Term("b"), Term("c")),
    )
    assert ~Term("java") == Not(Term("java"))
    assert parse('"event loop"') == Phrase(("event", "loop"))


def test_query_parser_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse("a @ b")
    with pytest.raises(ValueError):
        parse("(a OR b")


def test_phrase_evaluation_requires_positions() -> None:
    index = build_index(make_docs(), positions=False)
    with pytest.raises(ValueError):
        parse('"event loop"').evaluate(index)


def test_phrase_evaluation_with_positions() -> None:
    index = build_index(make_docs(), positions=True)
    assert parse('"event loop"').evaluate(index) == {0}
    assert parse('"loop event"').evaluate(index) == set()


def test_tfidf_and_bm25_return_ranked_results() -> None:
    index = build_index(make_docs(), positions=True)
    for scorer in (TfIdf(), BM25()):
        results = search(index, "python", scorer=scorer, k=2)
        assert all(isinstance(result, SearchResult) for result in results)
        assert results[0].score >= results[1].score


def test_scorers_are_call_compatible() -> None:
    index = build_index(make_docs(), positions=True)
    posting = index["python"][0]
    assert TfIdf()("python", posting, index) >= 0
    assert BM25()("python", posting, index) >= 0


def test_lru_cache_hits_on_repeated_query() -> None:
    index = build_index(make_docs(), positions=True)
    before = index.cache_info()
    search(index, "python", scorer=BM25(), k=3)
    middle = index.cache_info()
    search(index, "python", scorer=BM25(), k=3)
    after = index.cache_info()
    assert middle.misses == before.misses + 1
    assert after.hits >= middle.hits + 1


def test_timed_preserves_metadata() -> None:
    @timed
    def sample(x: int) -> int:
        """sample doc"""
        return x + 1

    assert sample.__name__ == "sample"
    assert sample.__doc__ == "sample doc"
    assert sample(1) == 2


def test_context_manager_closes_index_on_exception(tmp_path: Path) -> None:
    index = build_index(make_docs(), positions=True)
    path = tmp_path / "index.bin"
    save(index, path)
    opened: Index | None = None
    with pytest.raises(RuntimeError):
        with open_index(path) as current:
            opened = current
            assert current.closed is False
            raise RuntimeError("boom")
    assert opened is not None
    assert opened.closed is True


def test_pickle_and_json_round_trip(tmp_path: Path) -> None:
    index = build_index(make_docs(), positions=True)
    cases: tuple[tuple[str, Literal["pickle", "json"]], ...] = (
        (".bin", "pickle"),
        (".json", "json"),
    )
    for suffix, fmt in cases:
        path = tmp_path / f"index{suffix}"
        save(index, path, format=fmt)
        restored = load(path, format=fmt)
        assert restored.postings == index.postings
        assert restored.doc_meta == index.doc_meta
