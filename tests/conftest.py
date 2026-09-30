from __future__ import annotations

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
