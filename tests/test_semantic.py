"""Lab 8 M3: chunking, embeddings, best-chunk-per-doc, RRF, modes in CLI/API."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.index import Index, build_index_from_path
from findex.semantic import (
    FloatMatrix,
    SemanticIndex,
    WordLlamaEmbedder,
    build_semantic_index,
    chunk_spans,
    has_operators,
    reciprocal_rank_fusion,
    search_mode,
)
from findex.store import save
from findex.web import create_app
from findex.web.settings import Settings
from findex.web.state import SemanticDeps, get_index, get_loaded_index, get_semantic

TOPICS = ("cat", "car", "sea")


class TopicEmbedder:
    """Deterministic toy model: one dimension per topic word + a bias axis.

    "kitten" and "feline" count as "cat", so a query can match a document by
    meaning without sharing its words — exactly what the real model is for.
    """

    name = "toy"
    synonyms = {"kitten": "cat", "feline": "cat", "automobile": "car", "ocean": "sea"}

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        rows: list[list[float]] = []
        for text in texts:
            words = [self.synonyms.get(w, w) for w in text.lower().split()]
            rows.append([float(words.count(t)) for t in TOPICS] + [0.1])
        matrix = np.asarray(rows, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def semantic_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text(
        "cat cat cat sleeps here\n\nthe sea is calm today", encoding="utf-8"
    )
    (root / "b.txt").write_text("car car engine and wheels", encoding="utf-8")
    (root / "c.txt").write_text("sea sea waves\n\ncat", encoding="utf-8")
    return root


@pytest.fixture
def index(semantic_corpus: Path) -> Index:
    return build_index_from_path(semantic_corpus, positions=True)


@pytest.fixture
def semantic(index: Index) -> SemanticIndex:
    return build_semantic_index(index, TopicEmbedder(), max_words=5)


# ------------------------------------------------------------------ chunking


def test_chunk_spans_pack_paragraphs_and_split_long_ones() -> None:
    text = "one two\n\nthree four five\n\n" + " ".join(f"w{i}" for i in range(12))
    spans = list(chunk_spans(text, max_words=5))
    pieces = [text[a:b] for a, b in spans]
    assert pieces[0] == "one two\n\nthree four five"  # packed: 5 words
    assert pieces[1] == "w0 w1 w2 w3 w4"  # long paragraph cut by words
    assert pieces[-1] == "w10 w11"
    assert all(len(p.split()) <= 5 for p in pieces)


def test_chunk_spans_empty_text() -> None:
    assert list(chunk_spans("   \n\n  ")) == []


# --------------------------------------------------------------- the index


def test_build_semantic_index_shapes(index: Index, semantic: SemanticIndex) -> None:
    assert semantic.embeddings.dtype == np.float32
    assert semantic.chunks.shape == (semantic.num_chunks, 3)
    assert semantic.num_chunks >= index.num_docs
    norms = np.linalg.norm(semantic.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert set(semantic.chunks[:, 0].tolist()) == set(index.doc_ids)


def test_best_chunk_per_document(index: Index, semantic: SemanticIndex) -> None:
    query = TopicEmbedder().embed(["kitten"])[0]
    hits = semantic.search(query, k=3)
    titles = [index.doc_meta[h.doc_id].title for h in hits]
    # a's best chunk is all cats; c's best chunk is the lone "cat" paragraph.
    assert titles[:2] == ["a", "c"]
    assert len({h.doc_id for h in hits}) == len(hits)  # one hit per document
    assert "cat" in semantic.passage(index, hits[0].chunk)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_save_and_load_roundtrip(semantic: SemanticIndex, tmp_path: Path) -> None:
    semantic.save(tmp_path / "sem")
    for mmap in (False, True):
        loaded = SemanticIndex.load(tmp_path / "sem", mmap=mmap)
        assert loaded.model == "toy"
        assert np.array_equal(loaded.embeddings, semantic.embeddings)
        assert np.array_equal(loaded.chunks, semantic.chunks)


# --------------------------------------------------------------------- RRF


def test_rrf_formula_and_ties() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert [doc for doc, _ in fused] == [1, 3, 2]
    # Equal fused score -> smaller doc id first.
    assert [d for d, _ in reciprocal_rank_fusion([[5], [4]])] == [4, 5]


def test_has_operators() -> None:
    assert has_operators("a AND b")
    assert has_operators('"exact phrase"')
    assert not has_operators("android orange notebook")


# ------------------------------------------------------------------- modes


def test_keyword_mode_is_lab3(index: Index) -> None:
    found = search_mode(index, "cat", mode="keyword", k=5)
    assert found.total == 2
    assert all(hit.chunk is None for hit in found.hits)


def test_semantic_mode_finds_synonyms(index: Index, semantic: SemanticIndex) -> None:
    keyword = search_mode(index, "automobile", mode="keyword", k=3)
    assert keyword.total == 0  # the word is not in any document
    found = search_mode(
        index,
        "automobile",
        mode="semantic",
        k=1,
        semantic=semantic,
        embedder=TopicEmbedder(),
    )
    assert index.doc_meta[found.hits[0].doc_id].title == "b"
    assert found.hits[0].chunk is not None


def test_hybrid_mode_fuses_both_lists(index: Index, semantic: SemanticIndex) -> None:
    found = search_mode(
        index,
        "sea kitten",
        mode="hybrid",
        k=3,
        semantic=semantic,
        embedder=TopicEmbedder(),
    )
    # "sea kitten" has no AND match, but hybrid's keyword side is an OR.
    titles = [index.doc_meta[h.doc_id].title for h in found.hits]
    assert set(titles[:2]) == {"a", "c"}
    assert found.hits[0].score > 0


def test_semantic_mode_without_embeddings_raises(index: Index) -> None:
    with pytest.raises(LookupError):
        search_mode(index, "cat", mode="semantic")


# --------------------------------------------------------------------- API


def _client(index: Index, deps: SemanticDeps) -> TestClient:
    application = create_app(Settings(index_path=Path("unused.bin")))
    application.dependency_overrides[get_index] = lambda: index
    application.dependency_overrides[get_loaded_index] = lambda: index
    application.dependency_overrides[get_semantic] = lambda: deps
    return TestClient(application)


def test_api_semantic_and_hybrid(index: Index, semantic: SemanticIndex) -> None:
    client = _client(index, SemanticDeps(semantic, TopicEmbedder()))
    body = client.get("/search", params={"q": "ocean", "mode": "semantic"}).json()
    assert body["mode"] == "semantic"
    assert body["results"][0]["title"] in {"a", "c"}
    assert body["results"][0]["snippet"]
    body = client.get("/search", params={"q": "ocean", "mode": "hybrid"}).json()
    assert body["mode"] == "hybrid"
    assert body["total"] >= 2


def test_api_semantic_unavailable_is_503(index: Index) -> None:
    client = _client(index, SemanticDeps())
    response = client.get("/search", params={"q": "cat", "mode": "semantic"})
    assert response.status_code == 503
    assert (
        client.get("/search", params={"q": "cat", "mode": "fuzzy"}).status_code == 422
    )
    assert client.get("/search", params={"q": "cat"}).json()["mode"] == "keyword"


def test_ui_mode_toggle(index: Index, semantic: SemanticIndex) -> None:
    with_semantic = _client(index, SemanticDeps(semantic, TopicEmbedder()))
    page = with_semantic.get("/", params={"q": "kitten", "mode": "semantic"})
    assert 'value="semantic" checked' in page.text
    assert "closest in meaning" in page.text
    without = _client(index, SemanticDeps())
    page = without.get("/", params={"q": "cat", "mode": "semantic"})
    assert page.status_code == 200
    assert 'name="mode"' not in page.text  # toggle hidden, keyword used


# ------------------------------------------------------- real model + CLI


def test_wordllama_embeddings_capture_meaning() -> None:
    embedder = WordLlamaEmbedder()
    vectors = embedder.embed(
        [
            "the cat sat on the mat",
            "a feline rested on the rug",
            "the stock market fell",
        ]
    )
    assert vectors.shape == (3, 256)
    sims = vectors @ vectors.T
    assert sims[0, 1] > sims[0, 2]
    assert sims[0, 1] > sims[1, 2]


def test_cli_embed_and_search_modes(semantic_corpus: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "index.bin"
    save(build_index_from_path(semantic_corpus, positions=True), index_path)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["embed", str(index_path), "--chunk-words", "20"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "index.semantic" / "embeddings.npy").exists()
    for mode in ("semantic", "hybrid"):
        result = runner.invoke(
            cli_app, ["search", str(index_path), "kitten", "--mode", mode, "--json"]
        )
        assert result.exit_code == 0, result.output
        assert result.stdout.count("\n") == 3


def test_cli_semantic_without_embeddings_fails(
    semantic_corpus: Path, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.bin"
    save(build_index_from_path(semantic_corpus), index_path)
    result = CliRunner().invoke(
        cli_app, ["search", str(index_path), "cat", "--mode", "semantic"]
    )
    assert result.exit_code == 1
    assert "findex embed" in result.output
