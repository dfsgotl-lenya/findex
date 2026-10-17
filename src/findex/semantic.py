"""Lab 8 M3: semantic search over chunk embeddings, and hybrid search via RRF.

Pipeline (``findex embed``)::

    document text -> chunks of ~CHUNK_WORDS words -> embedder -> L2-normalize
    -> E: float32[N_chunks, d]  +  chunks: int32[N_chunks, 3] (doc, start, end)

Search: ``sims = E @ q`` (one matrix-vector product = cosine similarity for
unit vectors), best chunk per document, top-k documents. Hybrid search fuses
the BM25 list and the semantic list with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from .index import Index
from .ranking import Scorer
from .snippets import read_document_text

FloatMatrix = npt.NDArray[np.float32]
Int32Array = npt.NDArray[np.int32]

SearchMode = Literal["keyword", "semantic", "hybrid"]
CHUNK_WORDS = 200
RRF_K = 60
DEFAULT_MODEL = "wordllama"


# ----------------------------------------------------------------- embedders


class Embedder(Protocol):
    """Anything that turns texts into L2-normalized float32 row vectors."""

    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> FloatMatrix: ...


def _normalize(matrix: npt.NDArray[np.floating]) -> FloatMatrix:
    vectors = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class WordLlamaEmbedder:
    """Default model: WordLlama ``l2_supercat`` (256-d, ~16 MB, NumPy only).

    Its weights ship inside the PyPI wheel, so it works offline, in Docker and
    on a 512 MB free instance. It is a *static* model (token embeddings taken
    from Llama 2 and fine-tuned, then pooled), weaker than a transformer like
    all-MiniLM-L6-v2 but ~100x cheaper to run.
    """

    name = "wordllama"

    def __init__(self) -> None:
        from importlib import import_module

        wordllama = import_module("wordllama")  # untyped third-party package
        # The package bundles weights + tokenizer under its own directory;
        # pointing cache_dir there means no network access is ever needed.
        bundled = Path(str(wordllama.__file__)).parent
        self._model = wordllama.WordLlama.load(cache_dir=bundled, disable_download=True)

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.embed(list(texts), norm=False)
        return _normalize(cast(npt.NDArray[np.floating], vectors))


class SentenceTransformerEmbedder:
    """Optional: any sentence-transformers model, e.g. ``all-MiniLM-L6-v2``.

    ``uv pip install sentence-transformers`` first; the model (~90 MB) is
    downloaded from Hugging Face on first use.
    """

    def __init__(self, model_name: str) -> None:
        from importlib import import_module

        module = import_module("sentence_transformers")
        self.name = model_name
        self._model = module.SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True
        )
        return _normalize(cast(npt.NDArray[np.floating], vectors))


def load_embedder(name: str = DEFAULT_MODEL) -> Embedder:
    if name == "wordllama":
        return WordLlamaEmbedder()
    return SentenceTransformerEmbedder(name)


# ------------------------------------------------------------------ chunking

_WORD_RE = re.compile(r"\S+")


def chunk_spans(text: str, max_words: int = CHUNK_WORDS) -> Iterator[tuple[int, int]]:
    """Split *text* into ``(start, end)`` character spans of ≤ ``max_words``.

    Paragraphs (blank-line separated) are packed together until the next one
    would overflow; a paragraph longer than the limit is cut on word
    boundaries. Character offsets (not copies) are stored, so a hit can show
    its passage later without keeping the text in the index.
    """
    paragraphs = [
        (match.start(), match.end())
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.DOTALL)
    ]
    start: int | None = None
    end = 0
    words = 0
    for p_start, p_end in paragraphs:
        p_words = [m.span() for m in _WORD_RE.finditer(text, p_start, p_end)]
        if len(p_words) > max_words:
            if start is not None:
                yield start, end
                start, words = None, 0
            for i in range(0, len(p_words), max_words):
                piece = p_words[i : i + max_words]
                yield piece[0][0], piece[-1][1]
            continue
        if start is not None and words + len(p_words) > max_words:
            yield start, end
            start, words = None, 0
        if start is None:
            start = p_start
        end = p_end
        words += len(p_words)
    if start is not None:
        yield start, end


# ------------------------------------------------------------- the index


@dataclass(frozen=True, slots=True, eq=False)
class SemanticHit:
    doc_id: int
    score: float
    chunk: int


@dataclass(frozen=True, slots=True, eq=False)
class SemanticIndex:
    """Chunk embedding matrix + chunk → (doc, start, end) map."""

    model: str
    embeddings: FloatMatrix  # (N_chunks, d), unit rows
    chunks: Int32Array  # (N_chunks, 3): doc_id, start char, end char

    @property
    def num_chunks(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        np.save(directory / "chunks.npy", self.chunks)
        meta = {"model": self.model, "chunks": self.num_chunks, "dim": self.dim}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, *, mmap: bool = False) -> SemanticIndex:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        mode: Literal["r"] | None = "r" if mmap else None
        embeddings = np.load(directory / "embeddings.npy", mmap_mode=mode)
        chunks = np.load(directory / "chunks.npy")
        return cls(
            model=str(meta["model"]),
            embeddings=cast(FloatMatrix, embeddings),
            chunks=cast(Int32Array, chunks.astype(np.int32)),
        )

    def search(self, query_vector: FloatMatrix, k: int) -> list[SemanticHit]:
        """Top-k documents by their best chunk's cosine similarity."""
        if k <= 0 or self.num_chunks == 0:
            return []
        sims = self.embeddings @ query_vector.reshape(-1)  # (N_chunks,)
        docs = self.chunks[:, 0]
        # Sort chunks by (doc, -sim): the first chunk of every doc group is
        # that document's best chunk ("max pooling" over its passages).
        order = np.lexsort((-sims, docs))
        sorted_docs = docs[order]
        first = np.ones(order.shape[0], dtype=np.bool_)
        first[1:] = sorted_docs[1:] != sorted_docs[:-1]
        best_chunks = order[first]
        best_scores = sims[best_chunks]
        best_docs = docs[best_chunks]
        take = min(k, best_chunks.shape[0])
        top = np.argpartition(-best_scores, take - 1)[:take]
        # score desc, then doc id asc — same tie rule as the keyword ranking
        top = top[np.lexsort((best_docs[top], -best_scores[top]))]
        return [
            SemanticHit(
                doc_id=int(best_docs[i]),
                score=float(best_scores[i]),
                chunk=int(best_chunks[i]),
            )
            for i in top
        ]

    def passage(self, index: Index, chunk: int) -> str:
        doc_id, start, end = (int(value) for value in self.chunks[chunk])
        text = read_document_text(index.doc_meta[doc_id].path)
        return text[start:end]


def build_semantic_index(
    index: Index,
    embedder: Embedder,
    *,
    max_words: int = CHUNK_WORDS,
    batch_size: int = 256,
    on_progress: Callable[[int, int], None] | None = None,
) -> SemanticIndex:
    """Chunk every document of *index*, embed the chunks, stack the matrix."""
    rows: list[tuple[int, int, int]] = []
    texts: list[str] = []
    for doc_id in index.doc_ids:
        meta = index.doc_meta[doc_id]
        text = read_document_text(meta.path)
        for start, end in chunk_spans(text, max_words):
            rows.append((doc_id, start, end))
            # The title gives every passage its context ("PEP 572 – ...").
            texts.append(f"{meta.title}\n{text[start:end]}")
    parts: list[FloatMatrix] = []
    for begin in range(0, len(texts), batch_size):
        parts.append(embedder.embed(texts[begin : begin + batch_size]))
        if on_progress is not None:
            on_progress(min(begin + batch_size, len(texts)), len(texts))
    if parts:
        embeddings = np.vstack(parts).astype(np.float32)
    else:
        embeddings = np.zeros((0, 1), dtype=np.float32)
    chunks = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
    return SemanticIndex(model=embedder.name, embeddings=embeddings, chunks=chunks)


# ------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """``score(d) = Σ_lists 1 / (k + rank_d)`` with 1-based ranks.

    Uses only positions, never raw scores, so BM25 (unbounded) and cosine
    (-1..1) need no normalization. Ties: smaller doc id first.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


# --------------------------------------------------- one entry point, 3 modes

_OPERATOR_RE = re.compile(r'\b(AND|OR|NOT)\b|["()]')


@dataclass(frozen=True, slots=True)
class ModeHit:
    doc_id: int
    score: float
    chunk: int | None = None  # best passage for semantic/hybrid hits


@dataclass(frozen=True, slots=True)
class ModeResults:
    hits: list[ModeHit]
    total: int
    terms: tuple[str, ...]


def has_operators(query: str) -> bool:
    return _OPERATOR_RE.search(query) is not None


def search_mode(
    index: Index,
    query: str,
    *,
    mode: SearchMode = "keyword",
    scorer: Scorer | None = None,
    k: int = 10,
    semantic: SemanticIndex | None = None,
    embedder: Embedder | None = None,
    depth: int = 100,
) -> ModeResults:
    """Ranked search in one of three modes; returns at most ``k`` hits.

    * ``keyword`` — Lab 3 semantics: the Boolean query (implicit AND) filters,
      BM25/TF-IDF ranks. ``total`` = documents that match.
    * ``semantic`` — the query is embedded and compared with every chunk;
      the Boolean language is ignored. ``total`` is capped at ``depth``.
    * ``hybrid`` — RRF of the keyword list and the semantic list (each cut at
      ``depth``). The keyword side is recall-oriented: a plain query is
      treated as ``term OR term …``; explicit operators are kept as written.

    Raises ``ValueError`` for an unparsable query and ``LookupError`` if the
    semantic index / embedder is missing for a mode that needs it.
    """
    from .query import parse
    from .ranking import BM25
    from .search import rank

    scorer = scorer or BM25()
    tree = parse(query)
    terms = tree.positive_terms()

    if mode == "keyword":
        matched = index.evaluate_query(query)
        results = rank(index, terms, matched, scorer, k)
        return ModeResults(
            [ModeHit(r.doc_id, r.score) for r in results], len(matched), terms
        )

    if semantic is None or embedder is None:
        raise LookupError("semantic index is not loaded")
    text = " ".join(terms) if has_operators(query) else query
    vector = embedder.embed([text])[0]
    semantic_hits = semantic.search(vector, max(k, depth))

    if mode == "semantic":
        hits = [ModeHit(h.doc_id, h.score, h.chunk) for h in semantic_hits[:k]]
        return ModeResults(hits, len(semantic_hits), terms)

    if has_operators(query):
        matched = index.evaluate_query(query)
    else:
        matched = index.evaluate_query(" OR ".join(dict.fromkeys(terms)))
    keyword = rank(index, terms, matched, scorer, depth)
    fused = reciprocal_rank_fusion(
        [[r.doc_id for r in keyword], [h.doc_id for h in semantic_hits]]
    )
    best_chunk = {h.doc_id: h.chunk for h in semantic_hits}
    hits = [
        ModeHit(doc_id, score, best_chunk.get(doc_id)) for doc_id, score in fused[:k]
    ]
    return ModeResults(hits, len(fused), terms)
