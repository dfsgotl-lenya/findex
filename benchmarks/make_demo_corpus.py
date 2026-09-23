"""Create a deterministic local corpus for benchmarking the lab implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = (
    "Python generators are useful for streaming text through a search engine.\n"
    "A lazy pipeline reads one document, normalizes Unicode, tokenizes words, "
    "and updates a counter.\n"
    "Iteration uses iter and next, while StopIteration signals the end of a stream.\n"
    "Unicode NFC and casefold help equivalent text compare consistently.\n"
    "This document contains Cyrillic text too: пошук, документ, генератор, токен.\n"
    "Lab 01 measures memory with tracemalloc and elapsed time with perf_counter.\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--documents", type=int, default=300)
    parser.add_argument("--copies-per-document", type=int, default=80)
    args = parser.parse_args()

    if args.documents < 1 or args.copies_per_document < 1:
        parser.error("documents and copies-per-document must be positive")

    args.root.mkdir(parents=True, exist_ok=True)
    body = TEMPLATE * args.copies_per_document
    for index in range(1, args.documents + 1):
        path = args.root / f"doc-{index:04d}.txt"
        path.write_text(body, encoding="utf-8")

    total_bytes = sum(path.stat().st_size for path in args.root.glob("*.txt"))
    print(f"Created {args.documents} documents: {total_bytes / (1024**2):.2f} MiB")


if __name__ == "__main__":
    main()
