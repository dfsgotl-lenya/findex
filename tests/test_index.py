from pathlib import Path

from findex.corpus import Document
from findex.index import DocMeta, Posting, build_index
from findex.search import merge_and, merge_not, merge_or, search
from findex.store import load, save


def make_docs() -> list[Document]:
    return [
        Document("a.txt", Path("a.txt"), "Python python search"),
        Document("b.txt", Path("b.txt"), "Python engine search"),
        Document("c.txt", Path("c.txt"), "engine only"),
    ]


def test_posting_and_docmeta_are_hashable() -> None:
    assert hash(Posting(1, 2)) is not None
    assert hash(DocMeta(1, "a.txt", "a", 3)) is not None


def test_index_contains_sorted_postings_and_tf() -> None:
    index = build_index(make_docs())
    assert [p.doc_id for p in index.postings["python"]] == [0, 1]
    assert [p.tf for p in index.postings["python"]] == [2, 1]
    assert index.doc_meta[0].length == 3


def test_merge_boolean_operations() -> None:
    left = tuple(Posting(i, 1) for i in [0, 2, 4])
    right = tuple(Posting(i, 1) for i in [2, 3, 4])
    assert merge_and(left, right) == [2, 4]
    assert merge_or(left, right) == [0, 2, 3, 4]
    assert merge_not([0, 1, 2, 3, 4], right) == [0, 1]


def test_boolean_search_merge_and_set_match() -> None:
    index = build_index(make_docs())
    for query in ("python search", "python OR engine", "python NOT engine"):
        assert search(index, query, engine="merge") == search(
            index, query, engine="set"
        )


def test_pickle_round_trip(tmp_path) -> None:
    index = build_index(make_docs())
    path = tmp_path / "index.bin"
    save(index, path, format="pickle")
    restored = load(path, format="pickle")
    assert restored == index


def test_json_round_trip(tmp_path) -> None:
    index = build_index(make_docs())
    path = tmp_path / "index.json"
    save(index, path, format="json")
    restored = load(path, format="json")
    assert restored == index


def test_unknown_term_returns_empty() -> None:
    index = build_index(make_docs())
    assert search(index, "missing") == []


def test_limit_stops_after_requested_number_of_documents() -> None:
    index = build_index(make_docs(), limit=2)
    assert len(index.doc_meta) == 2
    assert index.doc_ids == (0, 1)
