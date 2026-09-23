"""Compare eager and lazy corpus processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .stats import _measure, collect_stats, eager_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark eager vs lazy findex.")
    parser.add_argument("root", type=Path, help="Directory containing .txt files")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N documents for both measurements",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    (lazy_result, lazy_time, lazy_peak) = _measure(
        collect_stats, args.root, args.limit
    )
    (eager_result, eager_time, eager_peak) = _measure(
        eager_stats, args.root, args.limit
    )

    print("Version | Documents | Tokens | Vocabulary | Peak memory | Elapsed")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    print(
        f"eager (lists) | {eager_result[0]} | {eager_result[1]} | "
        f"{len(eager_result[2])} | "
        f"{eager_peak / (1024**2):.2f} MiB | {eager_time:.4f} s"
    )
    print(
        f"lazy (generators) | {lazy_result[0]} | {lazy_result[1]} | "
        f"{len(lazy_result[2])} | {lazy_peak / (1024**2):.2f} MiB | {lazy_time:.4f} s"
    )


if __name__ == "__main__":
    main()
