"""Measure the three posting-storage variants required by Lab 2."""

from __future__ import annotations

import argparse
import gc
import pickle
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotsPosting:
    doc_id: int
    tf: int


def _build_variant(root: Path, variant: str, limit: int | None):
    postings = defaultdict(list)
    document_count = 0
    doc_lengths = {}

    for document in iter_documents(root):
        if limit is not None and document_count >= limit:
            break
        counts = Counter(tokenize(document.text))
        for term, tf in counts.items():
            if variant == "plain":
                postings[term].append(PlainPosting(document_count, tf))
            elif variant == "slots":
                postings[term].append(SlotsPosting(document_count, tf))
            elif variant == "array":
                if term not in postings:
                    postings[term] = array("I")
                postings[term].extend((document_count, tf))
            else:
                raise ValueError(variant)
        doc_lengths[document_count] = sum(counts.values())
        document_count += 1

    return dict(postings), doc_lengths, document_count


def _measure(root: Path, variant: str, limit: int | None):
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    value = _build_variant(root, variant, limit)
    elapsed = perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak


def _pickle_measure(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    save_time = perf_counter() - started

    started = perf_counter()
    with path.open("rb") as handle:
        pickle.load(handle)
    load_time = perf_counter() - started
    return path.stat().st_size, save_time, load_time


def _mib(value: int) -> float:
    return value / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/lab2-bench"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(
        "Variant | Documents | Peak memory (MiB) | Pickle size (MiB) | "
        "Save (s) | Load (s)"
    )
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for variant in ("plain", "slots", "array"):
        value, _elapsed, peak = _measure(args.root, variant, args.limit)
        size, save_time, load_time = _pickle_measure(
            value, args.out / f"{variant}.pickle"
        )
        docs = value[2]
        print(
            f"{variant} | {docs} | {_mib(peak):.2f} | {_mib(size):.2f} | "
            f"{save_time:.6f} | {load_time:.6f}"
        )


if __name__ == "__main__":
    main()
