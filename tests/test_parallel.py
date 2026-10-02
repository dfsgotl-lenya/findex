from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.parallel import (
    ExecutorName,
    build_parallel_index,
    build_partial,
    collect_document_paths,
    merge,
)

EXECUTORS: tuple[ExecutorName, ...] = ("serial", "threads", "processes")


def _pickle_bytes(index: Index) -> bytes:
    return pickle.dumps(index, protocol=5)


def test_serial_partial_merge_matches_lab4(tiny_corpus: Path) -> None:
    paths = collect_document_paths(tiny_corpus)
    lab4_index = build_index(iter_documents(tiny_corpus), positions=True)
    parallel_index = build_parallel_index(
        tiny_corpus,
        executor="serial",
        workers=1,
        positions=True,
    ).index
    merged = merge([build_partial(paths, 0, True)])
    assert _pickle_bytes(parallel_index) == _pickle_bytes(merged)
    assert _pickle_bytes(lab4_index) == _pickle_bytes(merged)


def test_partial_doc_id_ranges_are_globally_unique(tiny_corpus: Path) -> None:
    paths = collect_document_paths(tiny_corpus)
    first = build_partial(paths[:1], 0)
    second = build_partial(paths[1:], 1)
    assert sorted(first.doc_meta) == [0]
    assert sorted(second.doc_meta) == [1, 2]
    assert set(first.doc_meta).isdisjoint(second.doc_meta)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_all_executors_produce_same_index(
    tiny_corpus: Path, executor: ExecutorName
) -> None:
    result = build_parallel_index(
        tiny_corpus,
        executor=executor,
        workers=2,
        positions=True,
    ).index
    expected = build_index(iter_documents(tiny_corpus), positions=True)
    assert _pickle_bytes(result) == _pickle_bytes(expected)


def test_process_executor_propagates_worker_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        build_parallel_index(
            tmp_path,
            executor="processes",
            workers=2,
        )


def test_limit_is_applied_before_chunking(tiny_corpus: Path) -> None:
    result = build_parallel_index(
        tiny_corpus,
        executor="threads",
        workers=8,
        limit=2,
    ).index
    assert result.num_docs == 2
    assert sorted(result.doc_meta) == [0, 1]
