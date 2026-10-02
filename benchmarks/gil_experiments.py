"""Small reproducible experiments for the GIL section of Lab 5."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def crunch(n: int) -> int:
    """CPU-bound pure-Python workload."""
    return sum(i * i for i in range(n))


def _crunch(args: tuple[int, int]) -> int:
    """Picklable process-pool wrapper."""
    _, n = args
    return crunch(n)


def _run_threads(n: int) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(crunch, [n, n]))


def _run_processes(n: int) -> None:
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(_crunch, [(0, n), (1, n)]))


def timed(label: str, fn: Callable[[], object]) -> None:
    """Measure one experiment with wall-clock time."""
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    print(f"{label:16s} {elapsed:.4f} s")


def race(iterations: int, use_lock: bool, yield_every: int) -> int:
    """Run a deliberately racy counter and then a locked version."""
    shared = [0]
    lock = threading.Lock()

    def add() -> None:
        for i in range(iterations):
            if use_lock:
                with lock:
                    shared[0] += 1
                continue

            value = shared[0]
            if yield_every and i % yield_every == 0:
                time.sleep(0)
            shared[0] = value + 1

    threads = [threading.Thread(target=add) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return shared[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2_000_000)
    parser.add_argument("--iterations", type=int, default=500_000)
    parser.add_argument("--yield-every", type=int, default=1_000)
    args = parser.parse_args()

    gil_probe = getattr(sys, "_is_gil_enabled", None)
    gil_enabled = gil_probe() if gil_probe is not None else "unavailable"
    print(f"python: {sys.version.split()[0]}")
    print(f"sys._is_gil_enabled(): {gil_enabled}")
    print(f"sys.getswitchinterval(): {sys.getswitchinterval():.6f} s")

    timed("one", lambda: crunch(args.n))
    timed("two serial", lambda: (crunch(args.n), crunch(args.n)))
    timed("two threads", lambda: _run_threads(args.n))
    timed("two processes", lambda: _run_processes(args.n))

    expected = args.iterations * 2
    unsafe = race(args.iterations, use_lock=False, yield_every=args.yield_every)
    safe = race(args.iterations, use_lock=True, yield_every=0)
    print(f"race expected:      {expected}")
    print(f"race unsafe result: {unsafe}")
    print(f"race locked result: {safe}")


if __name__ == "__main__":
    main()
