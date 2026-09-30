"""Compare eager and lazy corpus processing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .stats import collect_stats, eager_stats, measure

logger = logging.getLogger(__name__)


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

    (lazy_result, lazy_time, lazy_peak) = measure(collect_stats, args.root, args.limit)
    (eager_result, eager_time, eager_peak) = measure(eager_stats, args.root, args.limit)

    logger.info("Version | Documents | Tokens | Vocabulary | Peak memory | Elapsed")
    logger.info("--- | ---: | ---: | ---: | ---: | ---:")
    logger.info(
        "eager (lists) | %s | %s | %s | %.2f MiB | %.4f s",
        eager_result[0],
        eager_result[1],
        len(eager_result[2]),
        eager_peak / (1024**2),
        eager_time,
    )
    logger.info(
        "lazy (generators) | %s | %s | %s | %.2f MiB | %.4f s",
        lazy_result[0],
        lazy_result[1],
        len(lazy_result[2]),
        lazy_peak / (1024**2),
        lazy_time,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
