"""Benchmark parallel index construction for Lab 5."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import psutil

from findex.parallel import ExecutorName, build_parallel_index


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Measurements for one execution."""

    wall: float
    cpu: float
    peak_rss: int
    merge: float


class ResourceSampler:
    """Sample parent RSS and child CPU time while a run is active."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.peak_rss = 0
        self._child_cpu_max: dict[int, float] = {}

    def start(self) -> None:
        self._sample_once()
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        self.thread.join()
        self._sample_once()
        return sum(self._child_cpu_max.values())

    def _sample_loop(self) -> None:
        while not self.stop_event.wait(0.01):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        except psutil.Error:
            return

        try:
            children = self.process.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                cpu = child.cpu_times()
                total = cpu.user + cpu.system
                self._child_cpu_max[child.pid] = max(
                    self._child_cpu_max.get(child.pid, 0.0),
                    total,
                )
            except psutil.Error:
                continue


def machine_description() -> str:
    """Return the machine information requested by the lab."""
    logical = os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    processor = platform.processor() or platform.uname().processor or "unknown"
    return (
        f"OS: {platform.platform()}\n"
        f"CPU: {processor}\n"
        f"Cores: {physical} physical / {logical} logical"
    )


def worker_values(cpu_count: int) -> list[int]:
    """Return the requested worker counts, deduplicated and capped."""
    values = [1, 2, 4, 8, cpu_count]
    return sorted({max(1, min(value, cpu_count)) for value in values})


def run_once(
    corpus: Path,
    executor: ExecutorName,
    workers: int,
    limit: int | None,
) -> RunMetrics:
    """Run one build and collect wall, CPU, RSS, and merge metrics."""
    sampler = ResourceSampler()
    sampler.start()
    parent_cpu_start = time.process_time()
    wall_start = time.perf_counter()
    result = build_parallel_index(
        corpus,
        executor=executor,
        workers=workers,
        positions=False,
        limit=limit,
    )
    wall = time.perf_counter() - wall_start
    child_cpu = sampler.stop()
    parent_cpu = time.process_time() - parent_cpu_start
    return RunMetrics(
        wall=wall,
        cpu=parent_cpu + child_cpu,
        peak_rss=sampler.peak_rss,
        merge=result.merge_time,
    )


def median_metrics(runs: Iterable[RunMetrics]) -> RunMetrics:
    """Take the median of each metric."""
    values = list(runs)
    return RunMetrics(
        wall=statistics.median(item.wall for item in values),
        cpu=statistics.median(item.cpu for item in values),
        peak_rss=int(statistics.median(item.peak_rss for item in values)),
        merge=statistics.median(item.merge for item in values),
    )


def markdown_table(rows: list[dict[str, object]]) -> str:
    """Render benchmark rows as a Markdown table."""
    lines = [
        "| Executor | Workers | Wall (s, median) | CPU (s) | "
        "Peak RSS (MiB) | Merge (s) | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            (
                "| {executor} | {workers} | {wall:.4f} | {cpu:.4f} | "
                "{rss:.2f} | {merge:.4f} | {speedup:.2f}× |"
            ).format(
                executor=row["executor"],
                workers=row["workers"],
                wall=row["wall"],
                cpu=row["cpu"],
                rss=row["rss"] / (1024**2),
                merge=row["merge"],
                speedup=row["speedup"],
            )
        )
    return "\n".join(lines)


def make_plot(rows: list[dict[str, object]], out: Path) -> None:
    """Create speedup-vs-workers plot with an ideal linear reference."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for executor in ("serial", "threads", "processes"):
        subset = [row for row in rows if row["executor"] == executor]
        if not subset:
            continue
        subset.sort(key=lambda row: int(row["workers"]))
        x = [int(row["workers"]) for row in subset]
        y = [float(row["speedup"]) for row in subset]
        ax.plot(x, y, label=executor)

    ideal_x = sorted({int(row["workers"]) for row in rows})
    ax.plot(ideal_x, ideal_x, label="ідеальна лінія")
    ax.set_title("Speedup індексації від кількості воркерів")
    ax.set_xlabel("Воркерів")
    ax.set_ylabel("Прискорення (×)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--executor",
        choices=("all", "serial", "threads", "processes"),
        default="all",
        help="Обмежити benchmark одним executor.",
    )
    parser.add_argument(
        "--label",
        default="CPython 3.13 (GIL)",
        help="Мітка інтерпретатора для виводу.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("docs/lab05_speedup.png"),
    )
    args = parser.parse_args()

    if args.repeats < 3:
        parser.error("--repeats має бути >= 3")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit має бути >= 1")

    cpu_count = os.cpu_count() or 1
    workers = worker_values(cpu_count)
    if args.executor == "serial":
        specs = [("serial", 1)]
    elif args.executor == "all":
        specs = [("serial", 1)]
        specs.extend(
            (executor, count)
            for executor in ("threads", "processes")
            for count in workers
        )
    else:
        specs = [(args.executor, count) for count in workers]

    print(machine_description())
    print(f"Python: {platform.python_version()}")
    print(f"Variant: {args.label}")
    print(f"Corpus: {args.corpus}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Runs per cell: {args.repeats}")
    print("\nWarm-up + measured runs")

    rows: list[dict[str, object]] = []
    baseline: float | None = None
    for executor, count in specs:
        effective_workers = min(count, cpu_count)
        run_once(args.corpus, executor, effective_workers, args.limit)
        runs = [
            run_once(args.corpus, executor, effective_workers, args.limit)
            for _ in range(args.repeats)
        ]
        metrics = median_metrics(runs)
        if baseline is None:
            baseline = metrics.wall
        speedup = baseline / metrics.wall if baseline else 1.0
        row = {
            "executor": executor,
            "workers": effective_workers,
            "wall": metrics.wall,
            "cpu": metrics.cpu,
            "rss": metrics.peak_rss,
            "merge": metrics.merge,
            "speedup": speedup,
        }
        rows.append(row)
        print(
            f"{executor:9s} workers={effective_workers:3d} "
            f"wall={metrics.wall:.4f}s cpu={metrics.cpu:.4f}s "
            f"rss={metrics.peak_rss / (1024**2):.2f}MiB "
            f"merge={metrics.merge:.4f}s speedup={speedup:.2f}x"
        )

    print("\nMarkdown для README:")
    print(markdown_table(rows))
    serial_row = next(row for row in rows if row["executor"] == "serial")
    merge_fraction = float(serial_row["merge"]) / float(serial_row["wall"])
    if merge_fraction > 0:
        print(
            f"\nAmdahl: serial merge fraction = {merge_fraction:.2%}; "
            f"theoretical limit ≈ {1 / merge_fraction:.2f}×"
        )
    make_plot(rows, args.out_plot)
    print(f"\nГрафік: {args.out_plot}")


if __name__ == "__main__":
    main()
