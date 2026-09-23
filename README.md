# findex — Lab 01: Iterators, Generators, and the Corpus

A small text-search engine project started from the corpus-intake stage. This lab focuses on Python's iteration protocol, generators, lazy pipelines, `pathlib`, `re`, Unicode normalization, and memory measurement with `tracemalloc`.

The implementation follows the Lab 01 requirements: a lazy `iter_documents()` generator, a streaming `tokenize()` generator using NFC + `casefold()` + `re.finditer`, a one-pass statistics pipeline, an eager comparison, and tests.

## Project structure

```text
findex/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/                  # local corpus; ignored by Git
├── src/
│   └── findex/
│       ├── __init__.py
│       ├── corpus.py
│       ├── tokenize.py
│       ├── stats.py
│       └── benchmark.py
├── tests/
│   └── test_tokenize.py
└── benchmarks/
    └── make_demo_corpus.py
```

## Corpus

Recommended course corpus: **Project Gutenberg plain-text books**. The course explicitly lists Gutenberg as one valid source and requires the local corpus to live under `data/`, with `data/` excluded from Git.

For the first lab, a small subset is enough while developing. Later labs can scale toward about 1,000 documents or about 50 MB of text.

Put `.txt` files in `data/`, for example:

```text
data/
├── book-001.txt
├── book-002.txt
└── ...
```

Do **not** commit the corpus itself.

## Setup with uv

The course standard is Python 3.12+ and `uv`, with an `src/` layout and `ruff`/`pytest` development tooling.

```powershell
uv sync
uv run pytest
uv run ruff check .
```

## M1 — Lazy document stream

`src/findex/corpus.py` exposes:

```python
iter_documents(root: Path) -> Iterator[Document]
```

`Document` contains `doc_id`, `path`, and `text`. The function is a generator, discovers `.txt` files lazily with `Path.rglob()`, opens one file at a time, and logs/skips files that cannot be read as UTF-8. This avoids materializing the whole corpus in memory. The lab specifically asks for a one-document-at-a-time generator.

## M2 — Streaming tokenizer

`tokenize(text)` first applies:

1. Unicode NFC normalization;
2. `casefold()` for case-insensitive matching;
3. `re.finditer()` to yield matches lazily.

Token policy:

- letters and digits are retained;
- mixed alphanumeric forms such as `Python3` stay one token;
- internal ASCII `'` and typographic `’` are retained (`don't`, `п'ять`);
- hyphens separate words (`state-of-the-art` → `state`, `of`, `the`, `art`);
- punctuation and symbols are separators;
- underscores are not treated as word characters.

These choices are explicit because the lab requires documented policies for apostrophes, hyphens, and digits.

## M3 — Statistics pipeline

Run:

```powershell
uv run python -m findex.stats data/
```

Limit processing during development:

```powershell
uv run python -m findex.stats data/ --limit 100
```

The pipeline computes in one pass:

- document count;
- total token count;
- vocabulary size;
- top-50 terms;
- elapsed wall-clock time;
- peak traced memory.

The `--limit` option uses `itertools.islice`, as required by the lab.

## M4 — Eager vs lazy measurement

The intentionally eager implementation reads documents into a list and then creates a list of token lists. The lazy implementation keeps documents and tokens flowing through generators and lets only the `Counter` grow.

Run the comparison on the same corpus slice:

```powershell
uv run python -m findex.benchmark data/ --limit 100
```

### Measurement table

The following values are a real reference run for this implementation in the current Python 3.13.5 Linux environment. Before submission, rerun the same benchmark on your Windows machine and replace the table values, because the lab requires measurements from your machine.

| Version | Documents | Tokens | Vocabulary | Peak memory | Elapsed |
|---|---:|---:|---:|---:|---:|
| eager (lists) | 300 | 1,656,000 | 57 | 107.25 MiB | 3.1548 s |
| lazy (generators) | 300 | 1,656,000 | 57 | 0.65 MiB | 3.6225 s |

The eager version keeps every `Document`, every token list, and all token strings alive at the same time, so memory grows with the corpus slice. In the lazy version, upstream stages keep only the current document/token work plus the `Counter`; old documents and tokens can be released after they are consumed. Lazy processing therefore avoids the large intermediate lists, although its peak is not zero because one document's text, generator state, regular-expression objects, and the growing `Counter` still require memory. This is the central memory claim the lab asks you to measure rather than guess.

## Local benchmark corpus

For a reproducible test without committing data, create a deterministic local corpus outside `data/` or directly inside ignored `data/`:

```powershell
uv run python benchmarks/make_demo_corpus.py data/benchmark --documents 300 --copies-per-document 80
uv run python -m findex.benchmark data/benchmark --limit 300
```

The generated benchmark corpus is only a measurement fixture. For the semester project, replace it with your chosen real corpus (for example, Project Gutenberg books).

## Generator checks

You can demonstrate that both main stages are generators:

```python
import inspect
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

print(inspect.isgeneratorfunction(iter_documents))
print(inspect.isgeneratorfunction(tokenize))

print(next(iter_documents(Path("data"))))
print(next(tokenize("Hello, world!")))
```

The lab uses generator identity and the ability to obtain a next value before reading the whole corpus as evidence of laziness.

## Tests

```powershell
uv run pytest
```

The tokenizer tests cover mixed case, Cyrillic, a combining-mark accent, apostrophes, hyphens, digits, punctuation, and empty input, matching the requested categories.

## Reflection — defense notes

### 1. What does `for x in xs` really do?

Conceptually:

```python
it = iter(xs)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
```

An **iterable** can produce an iterator with `__iter__()`. An **iterator** also implements `__next__()` and remembers its current position. A generator object is an iterator created by a generator function.

### 2. Why can a list be reused but a generator cannot?

A list is reusable because `iter(list)` creates a fresh iterator starting at the beginning. A generator is its own progressing iterator; once exhausted, subsequent `next()` calls continue to raise `StopIteration`. To make processing restartable, call the generator function again or materialize the data into a list.

### 3. What is in memory around the 10,000th token?

The lazy pipeline holds the current document text, the current generator/regex state, the `Counter`, and small supporting objects. Previously consumed documents and tokens are not retained by the pipeline. The `Counter` grows because it is the actual result being produced.

### 4. Where does a tokenizer exception appear?

A generator function does not execute its body when the generator object is created. Execution begins when the generator is consumed, so an exception inside the body surfaces at the consuming operation (`next()`, `for`, `Counter`, etc.), not at the original call that created the generator.

### 5. Why `casefold()` and NFC?

`casefold()` is designed for caseless matching and can make forms such as `Straße` and `STRASSE` compare consistently. NFC normalization makes canonically equivalent Unicode representations share the same composed form; for example, `café` and `cafe\u0301` should tokenize to the same normalized token.

### 6. `findall` vs `finditer`

Both use the same pattern, but `findall` materializes the matches in a list, while `finditer` yields match objects lazily. On a very large document, the list of all matches adds a potentially large memory cost.

### 7. Why isn't lazy memory zero?

Laziness removes the large intermediate corpus/token lists; it does not remove the memory needed for the current document, current generator state, the regex machinery, Python object overhead, and the growing `Counter`.

## Deliverable checklist

- [x] `src/` project layout
- [x] `uv` project configuration
- [x] `data/` ignored by Git
- [x] `iter_documents()` generator
- [x] `tokenize()` generator
- [x] NFC + `casefold()` + `re.finditer()`
- [x] documented apostrophe/hyphen/digit policy
- [x] tokenizer tests
- [x] one-pass statistics with `--limit` / `islice`
- [x] eager-vs-lazy benchmark implementation
- [ ] replace benchmark table placeholders with your own machine's measurements
- [ ] add your real corpus to local `data/`
- [ ] make repository public and push to GitHub
- [ ] create Git tag `lab-01`

The official lab's definition of done requires the eager-vs-lazy table to contain real numbers from the student's machine and the repository to be tagged `lab-01`.

### Updating the table

After running `uv run python -m findex.benchmark data/benchmark --limit 300`, copy the two output rows into the table above. Keep the same document count for both versions so the comparison is fair.
