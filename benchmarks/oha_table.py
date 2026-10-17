"""Turn ``oha --output-format json`` files into the Lab 7 README table.

uv run python benchmarks/oha_table.py docs/lab07_oha/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast


def row(path: Path) -> str:
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    summary = cast(dict[str, Any], data["summary"])
    pct = cast(dict[str, Any], data["latencyPercentiles"])
    codes = cast(dict[str, int], data.get("statusCodeDistribution", {}))
    ok = codes.get("200", 0)
    total = sum(codes.values())
    return (
        f"| {path.stem} | {summary['requestsPerSec']:.1f} | "
        f"{pct['p50'] * 1000:.0f} | {pct['p95'] * 1000:.0f} | "
        f"{pct['p99'] * 1000:.0f} | {ok}/{total} |"
    )


def main() -> None:
    print("| Target | RPS | p50 (ms) | p95 (ms) | p99 (ms) | 200 OK |")
    print("|---|---:|---:|---:|---:|---:|")
    for name in sys.argv[1:]:
        print(row(Path(name)))


if __name__ == "__main__":
    main()
