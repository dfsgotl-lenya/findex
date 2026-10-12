"""Benchmark findex crawl at concurrency 1, 5, and 20 against local mirror."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil


def run_once(
    seed: str,
    *,
    concurrency: int,
    max_pages: int,
    contact: str,
    out_path: Path,
    log_path: Path,
    per_host: int,
    per_host_delay: float,
) -> tuple[float, int, int, float]:
    command = [
        sys.executable,
        "-m",
        "findex.cli",
        "crawl",
        seed,
        "--contact",
        contact,
        "--max-pages",
        str(max_pages),
        "--concurrency",
        str(concurrency),
        "--per-host",
        str(per_host),
        "--per-host-delay",
        str(per_host_delay),
        "--out",
        str(out_path),
        "--log-file",
        str(log_path),
    ]
    started = time.perf_counter()
    stdout_path = out_path.with_suffix(".stdout.txt")
    with stdout_path.open("w", encoding="utf-8") as stdout_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        child = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, child.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.02)
    try:
        peak_rss = max(peak_rss, child.memory_info().rss)
    except psutil.Error:
        pass
    process.wait()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        output = stdout_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"crawl command failed (exit={process.returncode}):\n{output}"
        )

    page_count = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    page_count += 1
    errors = 0
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if " action=retry " in line:
                continue
            status_error = bool(re.search(r"status=(?:429|5\d\d)\b", line)) or any(
                status in line
                for status in (
                    "status=timeout",
                    "status=network_error",
                    "status=worker_error",
                    "status=robots_denied",
                    "status=redirect_loop",
                )
            )
            if (
                " bytes=" in line and status_error
            ) or "status=robots_unavailable" in line:
                errors += 1
    return elapsed, page_count, errors, peak_rss / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="http://127.0.0.1:8765/page/0")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--contact", required=True, help="Your real contact email.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--per-host", type=int, default=20)
    parser.add_argument("--per-host-delay", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/lab06-benchmark"))
    args = parser.parse_args()
    if args.max_pages < 200:
        parser.error("--max-pages must be at least 200")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.per_host < 1 or args.per_host_delay < 0:
        parser.error("invalid per-host limit/delay")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seed: {args.seed}")
    print(f"Pages cap: {args.max_pages}; repetitions: {args.repeats}")
    print(
        "Benchmark profile: LOCAL mirror; per-host limit relaxed for "
        "throughput comparison."
    )
    print()
    print(
        "| Concurrency | Pages (median) | Wall (s, median) | Pages/s | "
        "Errors (median) | Peak RSS (MiB, median) |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    markdown_rows: list[str] = []
    for concurrency in (1, 5, 20):
        times: list[float] = []
        pages: list[int] = []
        errors: list[int] = []
        rss: list[float] = []
        for repeat in range(1, args.repeats + 1):
            out_path = args.out_dir / f"crawl-c{concurrency}-r{repeat}.jsonl"
            log_path = args.out_dir / f"crawl-c{concurrency}-r{repeat}.log"
            wall, page_count, error_count, peak = run_once(
                args.seed,
                concurrency=concurrency,
                max_pages=args.max_pages,
                contact=args.contact,
                out_path=out_path,
                log_path=log_path,
                per_host=args.per_host,
                per_host_delay=args.per_host_delay,
            )
            times.append(wall)
            pages.append(page_count)
            errors.append(error_count)
            rss.append(peak)
            print(
                f"  run {repeat}/{args.repeats}: concurrency={concurrency}, "
                f"pages={page_count}, wall={wall:.3f}s, errors={error_count}, "
                f"peak={peak:.2f} MiB"
            )
        median_wall = statistics.median(times)
        median_pages = int(statistics.median(pages))
        median_errors = int(statistics.median(errors))
        median_rss = statistics.median(rss)
        page_rate = median_pages / median_wall if median_wall > 0 else 0.0
        row = (
            f"| {concurrency} | {median_pages} | {median_wall:.3f} | "
            f"{page_rate:.2f} | {median_errors} | {median_rss:.2f} |"
        )
        markdown_rows.append(row)
        print(row)
    table_lines = [
        "| Concurrency | Pages (median) | Wall (s, median) | Pages/s | "
        "Errors (median) | Peak RSS (MiB, median) |",
        "|---:|---:|---:|---:|---:|---:|",
        *markdown_rows,
    ]
    (args.out_dir / "table.md").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )
    print(f"\nMarkdown table saved to {args.out_dir / 'table.md'}")
    print(
        "The log from each run is saved beside its JSONL output. "
        "Do not commit the generated data corpus."
    )


if __name__ == "__main__":
    main()
