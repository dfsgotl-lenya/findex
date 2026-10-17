# Lab 8 M2: Lab 3 scorer vs NumPy scorer

Command: `uv run python benchmarks/benchmark_numpy_scorer.py data/peps.bin [--replicate 10]`. Best of 7 repeats; ranking only (score + top-k). 2 vCPU, Python 3.13, NumPy 2.x.

## Real corpus (744 PEPs)

BM25, k=10, 744 docs

| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---:|---:|---:|---:|
| common term `python` | 729 | 0.645 | 0.070 | 9.2× |
| mid-frequency term `generator` | 100 | 0.100 | 0.036 | 2.8× |
| rare term (df=3) `ax` | 3 | 0.005 | 0.022 | 0.3× |
| 3-term query `python type annotations` | 68 | 0.242 | 0.079 | 3.1× |
| 3-term OR query `python OR type OR annotations` | 744 | 1.214 | 0.111 | 10.9× |

TfIdf, k=10, 744 docs

| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---:|---:|---:|---:|
| common term `python` | 729 | 0.454 | 0.067 | 6.8× |
| mid-frequency term `generator` | 100 | 0.079 | 0.057 | 1.4× |
| rare term (df=3) `ax` | 3 | 0.005 | 0.017 | 0.3× |
| 3-term query `python type annotations` | 68 | 0.202 | 0.057 | 3.5× |
| 3-term OR query `python OR type OR annotations` | 744 | 0.792 | 0.082 | 9.6× |

## Same corpus replicated 10x (7 440 docs, postings 10x longer)

BM25, k=10, 7440 docs

| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---:|---:|---:|---:|
| common term `python` | 7290 | 5.786 | 0.327 | 17.7× |
| mid-frequency term `generator` | 1000 | 0.853 | 0.077 | 11.0× |
| rare term (df=30) `ax` | 30 | 0.039 | 0.033 | 1.2× |
| 3-term query `python type annotations` | 680 | 2.070 | 0.242 | 8.6× |
| 3-term OR query `python OR type OR annotations` | 7440 | 11.926 | 0.486 | 24.5× |

TfIdf, k=10, 7440 docs

| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---:|---:|---:|---:|
| common term `python` | 7290 | 3.965 | 0.310 | 12.8× |
| mid-frequency term `generator` | 1000 | 0.615 | 0.066 | 9.3× |
| rare term (df=30) `ax` | 30 | 0.030 | 0.029 | 1.0× |
| 3-term query `python type annotations` | 680 | 1.573 | 0.198 | 7.9× |
| 3-term OR query `python OR type OR annotations` | 7440 | 7.848 | 0.458 | 17.1× |
