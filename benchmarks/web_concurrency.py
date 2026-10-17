"""Lab 7 M2: 20 concurrent /search requests, ``def`` vs naive ``async def``.

For each variant the script starts a real uvicorn process (one worker) with
``SEARCH_ROUTE=def`` or ``SEARCH_ROUTE=async``, fires N concurrent searches,
and — while they run — keeps pinging ``/health`` to see whether the event loop
is still free to answer cheap requests.

    uv run python benchmarks/web_concurrency.py --index data/peps.bin
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx


async def _wait_ready(base: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{base}/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError("server did not become healthy")


async def _burst(
    base: str, query: str, concurrency: int
) -> tuple[float, list[float], list[float]]:
    """Return (wall seconds, search latencies ms, health latencies ms)."""
    limits = httpx.Limits(max_connections=concurrency + 5)
    async with httpx.AsyncClient(limits=limits, timeout=120) as client:
        done = asyncio.Event()
        health: list[float] = []

        async def one_search() -> float:
            started = time.perf_counter()
            response = await client.get(f"{base}/search", params={"q": query, "k": 10})
            response.raise_for_status()
            return (time.perf_counter() - started) * 1000

        async def ping_health() -> None:
            while not done.is_set():
                started = time.perf_counter()
                await client.get(f"{base}/health")
                health.append((time.perf_counter() - started) * 1000)
                await asyncio.sleep(0.005)

        pinger = asyncio.create_task(ping_health())
        await asyncio.sleep(0.02)
        started = time.perf_counter()
        latencies = await asyncio.gather(*(one_search() for _ in range(concurrency)))
        wall = time.perf_counter() - started
        done.set()
        await pinger
    return wall, list(latencies), health


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def run_variant(
    variant: str, index: Path, query: str, concurrency: int, repeats: int, port: int
) -> dict[str, float]:
    env = {**os.environ, "INDEX_PATH": str(index), "SEARCH_ROUTE": variant}
    env["LOG_LEVEL"] = "WARNING"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "findex.web:app",
            "--port",
            str(port),
            "--workers",
            "1",
            "--no-access-log",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        asyncio.run(_wait_ready(base))
        asyncio.run(_burst(base, query, concurrency))  # warm-up
        walls: list[float] = []
        searches: list[float] = []
        health: list[float] = []
        for _ in range(repeats):
            wall, lat, hl = asyncio.run(_burst(base, query, concurrency))
            walls.append(wall)
            searches.extend(lat)
            health.extend(hl)
    finally:
        server.terminate()
        server.wait(timeout=15)
    return {
        "wall_ms": statistics.median(walls) * 1000,
        "search_p50": _pct(searches, 0.50),
        "search_max": max(searches),
        "search_min": min(searches),
        "health_p50": _pct(health, 0.50),
        "health_max": max(health),
        "health_n": len(health) / repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/peps.bin"))
    parser.add_argument("--query", default="python type annotations")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    rows: list[tuple[str, dict[str, float]]] = []
    for variant in ("def", "async"):
        stats = run_variant(
            variant, args.index, args.query, args.concurrency, args.repeats, args.port
        )
        rows.append((variant, stats))

    print(
        f"\n{args.concurrency} concurrent GET /search?q={args.query!r}, "
        f"1 uvicorn worker, median of {args.repeats} bursts\n"
    )
    print(
        "| Route | Wall for 20 (ms) | Search fastest (ms) | Search p50 (ms) | "
        "Search slowest (ms) | /health p50 during burst (ms) | "
        "/health max (ms) | /health answers per burst |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant, s in rows:
        label = "`def` (thread pool)" if variant == "def" else "`async def` (on loop)"
        print(
            f"| {label} | {s['wall_ms']:.0f} | {s['search_min']:.0f} | "
            f"{s['search_p50']:.0f} | {s['search_max']:.0f} | "
            f"{s['health_p50']:.1f} | {s['health_max']:.0f} | {s['health_n']:.0f} |"
        )


if __name__ == "__main__":
    main()
