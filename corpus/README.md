# Demo corpus: Python Enhancement Proposals

`peps.jsonl.gz` — 744 PEPs from [python/peps](https://github.com/python/peps)
at commit `6068058b896441388bbf22f2607ab681193f6a3d` (2026-10-16), one PEP per
JSONL line in the same `{"url", "title", "text"}` shape the Lab 6 crawler writes.

PEPs are placed in the public domain or under CC0-1.0 by their authors, so the
snapshot can be redistributed. It is committed (4 MB compressed) so that the
Docker build is hermetic: no network access is needed to build the index.

Regenerate:

```bash
git clone --depth 1 https://github.com/python/peps /tmp/peps
uv run python benchmarks/make_pep_corpus.py /tmp/peps --out data/peps.jsonl
gzip -9 -c data/peps.jsonl > corpus/peps.jsonl.gz
```

Build the index locally:

```bash
gunzip -c corpus/peps.jsonl.gz > data/peps.jsonl   # PowerShell: see README
uv run findex index data/peps.jsonl --out data/peps.bin --positions
```
