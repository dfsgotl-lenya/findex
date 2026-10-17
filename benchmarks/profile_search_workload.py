"""Workload for Scalene / py-spy: load the index, then run 200 searches.

uv run scalene run --cpu-only benchmarks/profile_search_workload.py
uv run scalene view --cli
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from findex.ranking import BM25
from findex.search import search
from findex.store import load

QUERIES = ("python type annotations", "python", "walrus")


def main() -> None:
    logging.disable(logging.CRITICAL)
    index_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/peps.bin")
    index = load(index_path)
    for _ in range(200):
        for query in QUERIES:
            index.clear_cache()
            search(index, query, scorer=BM25(), k=10)


if __name__ == "__main__":
    main()
