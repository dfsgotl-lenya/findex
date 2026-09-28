"""Measure save/load time and file size for pickle and JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from findex.index import build_index_from_path
from findex.store import load, save


def _mib(value: int) -> float:
    return value / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/lab2-storage"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    index = build_index_from_path(args.root, limit=args.limit)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Format | Size (MiB) | Save (s) | Load (s)")
    print("--- | ---: | ---: | ---:")
    for fmt, filename in (("pickle", "index.bin"), ("json", "index.json")):
        path = args.out / filename
        started = perf_counter()
        save(index, path, format=fmt)
        save_elapsed = perf_counter() - started
        started = perf_counter()
        load(path, format=fmt)
        load_elapsed = perf_counter() - started
        print(
            f"{fmt} | {_mib(path.stat().st_size):.2f} | "
            f"{save_elapsed:.6f} | {load_elapsed:.6f}"
        )


if __name__ == "__main__":
    main()
