"""Parallel index construction for Lab 5."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from .index import DocMeta, Index, Posting
from .tokenize import tokenize

ExecutorName = Literal["serial", "threads", "processes"]


@dataclass(frozen=True, slots=True)
class PartialIndex:
    """A deterministic partial index produced by one worker."""

    postings: dict[str, tuple[Posting, ...]]
    doc_meta: dict[int, DocMeta]


@dataclass(frozen=True, slots=True)
class ParallelBuildResult:
    """Index and timing values produced by a build."""

    index: Index
    build_time: float
    merge_time: float


def collect_document_paths(root: Path, limit: int | None = None) -> list[Path]:
    """Collect sorted UTF-8 text paths without reading document contents."""
    root = Path(root)
    if root.is_file():
        paths = [root] if root.suffix.lower() == ".txt" else []
    elif root.is_dir():
        paths = sorted(root.rglob("*.txt"))
    else:
        paths = []

    return paths[:limit] if limit is not None else paths


def _index_one_file(
    path: Path,
    doc_id: int,
    positions: bool,
) -> tuple[DocMeta, dict[str, Posting]]:
    """Read one file and build its term-frequency map."""
    text = path.read_text(encoding="utf-8", errors="strict")
    counts: dict[str, int] = {}
    position_map: dict[str, list[int]] = {}
    length = 0

    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1
        if positions:
            position_map.setdefault(token, []).append(length)
        length += 1

    per_document = {
        term: Posting(
            doc_id=doc_id,
            tf=tf,
            positions=tuple(position_map[term]) if positions else (),
        )
        for term, tf in counts.items()
    }
    meta = DocMeta(
        doc_id=doc_id,
        path=path.as_posix(),
        title=path.stem,
        length=length,
    )
    return meta, per_document


def build_partial(
    doc_paths: list[Path],
    start_doc_id: int = 0,
    positions: bool = False,
) -> PartialIndex:
    """Build a partial index from paths reserved to one worker.

    Only file paths are sent to workers. The worker reads and tokenizes its
    own files, so large document bodies never have to be pickled into the
    process pool.
    """
    posting_lists: dict[str, list[Posting]] = {}
    doc_meta: dict[int, DocMeta] = {}

    for offset, path in enumerate(doc_paths):
        doc_id = start_doc_id + offset
        meta, per_document = _index_one_file(Path(path), doc_id, positions)
        doc_meta[doc_id] = meta
        for term, posting in per_document.items():
            posting_lists.setdefault(term, []).append(posting)

    canonical = {
        term: tuple(sorted(items, key=lambda item: item.doc_id))
        for term, items in sorted(posting_lists.items())
    }
    return PartialIndex(postings=canonical, doc_meta=dict(sorted(doc_meta.items())))


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Merge partial indexes deterministically by term and ``doc_id``."""
    posting_lists: dict[str, list[Posting]] = {}
    doc_meta: dict[int, DocMeta] = {}

    for partial in partials:
        doc_meta.update(partial.doc_meta)
        for term, postings in partial.postings.items():
            posting_lists.setdefault(term, []).extend(postings)

    canonical = {
        term: tuple(sorted(items, key=lambda item: item.doc_id))
        for term, items in sorted(posting_lists.items())
    }
    return Index(postings=canonical, doc_meta=dict(sorted(doc_meta.items())))


def _make_chunks(paths: list[Path], workers: int) -> list[tuple[list[Path], int]]:
    """Split paths while reserving contiguous global document-ID ranges."""
    if not paths:
        return []
    worker_count = max(1, min(workers, len(paths)))
    size, remainder = divmod(len(paths), worker_count)
    chunks: list[tuple[list[Path], int]] = []
    cursor = 0
    start_doc_id = 0

    for worker in range(worker_count):
        count = size + (1 if worker < remainder else 0)
        chunk = paths[cursor : cursor + count]
        chunks.append((chunk, start_doc_id))
        cursor += count
        start_doc_id += count

    return chunks


def _run_serial(
    chunks: list[tuple[list[Path], int]],
    positions: bool,
    on_complete: Callable[[], None] | None,
) -> list[PartialIndex]:
    results: list[PartialIndex] = []
    for paths, start_doc_id in chunks:
        results.append(build_partial(paths, start_doc_id, positions))
        if on_complete is not None:
            on_complete()
    return results


def _run_pool(
    executor: ThreadPoolExecutor | ProcessPoolExecutor,
    chunks: list[tuple[list[Path], int]],
    positions: bool,
    on_complete: Callable[[], None] | None,
) -> list[PartialIndex]:
    futures: dict[Future[PartialIndex], int] = {
        executor.submit(build_partial, paths, start_doc_id, positions): index
        for index, (paths, start_doc_id) in enumerate(chunks)
    }
    results: list[PartialIndex | None] = [None] * len(chunks)

    for future in as_completed(futures):
        index = futures[future]
        results[index] = future.result()
        if on_complete is not None:
            on_complete()

    return [result for result in results if result is not None]


def build_parallel_index(
    root: Path,
    *,
    workers: int = 1,
    executor: ExecutorName = "serial",
    positions: bool = False,
    limit: int | None = None,
    on_partial_complete: Callable[[], None] | None = None,
) -> ParallelBuildResult:
    """Build an index with one of three execution backends."""
    if workers < 1:
        raise ValueError("workers must be >= 1")

    paths = collect_document_paths(Path(root), limit=limit)
    chunks = _make_chunks(paths, workers if executor != "serial" else 1)

    started = perf_counter()
    if executor == "serial":
        partials = _run_serial(chunks, positions, on_partial_complete)
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partials = _run_pool(
                pool,
                chunks,
                positions,
                on_partial_complete,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as pool:
            partials = _run_pool(
                pool,
                chunks,
                positions,
                on_partial_complete,
            )
    built = perf_counter()

    index = merge(partials)
    finished = perf_counter()
    return ParallelBuildResult(
        index=index,
        build_time=built - started,
        merge_time=finished - built,
    )
