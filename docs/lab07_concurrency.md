# Lab 7: 20 concurrent /search requests, def vs async def

Command: `uv run python benchmarks/web_concurrency.py --index data/peps.bin` (2 vCPU, Python 3.13).


| Route | Wall for 20 (ms) | Search fastest (ms) | Search p50 (ms) | Search slowest (ms) | /health p50 during burst (ms) | /health max (ms) | /health answers per burst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `def` (thread pool) | 472 | 56 | 435 | 480 | 38.2 | 141 | 9 |
| `async def` (on loop) | 372 | 24 | 352 | 384 | 5.6 | 348 | 4 |
