from __future__ import annotations

import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, build_index


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text(
        "Python async event loop search engine", encoding="utf-8"
    )
    (root / "b.txt").write_text("Python Python search engine token", encoding="utf-8")
    (root / "c.txt").write_text("Java engine only", encoding="utf-8")
    return root


@pytest.fixture
def tiny_index() -> Index:
    documents = [
        Document("a.txt", Path("a.txt"), "Python async event loop search engine"),
        Document("b.txt", Path("b.txt"), "Python Python search engine token"),
        Document("c.txt", Path("c.txt"), "Java engine only"),
    ]
    return build_index(documents, positions=True)


def zipf_documents(seed: int, docs: int = 300, vocab: int = 400) -> list[Document]:
    """Deterministic random corpus with a Zipf-like vocabulary: frequent and
    rare terms, many tf ties and very different document lengths."""
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    weights = [1 / (rank + 1) for rank in range(vocab)]
    return [
        Document(
            f"d{i}",
            Path(f"d{i}.txt"),
            " ".join(rng.choices(words, weights, k=rng.randint(3, 400))),
            title=f"doc {i}",
        )
        for i in range(docs)
    ]


@pytest.fixture(scope="session")
def zipf_index() -> Index:
    """300-document random corpus shared by the Lab 8 ranking/benchmark tests."""
    return build_index(zipf_documents(seed=7), positions=True)


@pytest.fixture(scope="session")
def zipf_docs() -> list[Document]:
    """Documents (not yet indexed) for the build_index benchmark."""
    return zipf_documents(seed=11)
