"""Glue between the HTTP layer and the Lab 2-3 engine.

Everything here is plain synchronous, CPU-bound Python. The routes decide how
to run it (thread pool vs. event loop); see ``app.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..index import Index
from ..ranking import BM25, Scorer, SearchResult, TfIdf
from ..semantic import Embedder, SearchMode, SemanticIndex, search_mode
from ..snippets import highlight_spans, read_document_text, snippet_spans, source_url
from .models import (
    DocumentResponse,
    ScorerName,
    SearchHit,
    SearchResponse,
)

Spans = list[tuple[str, bool]]


class QueryError(ValueError):
    """The query string could not be parsed or evaluated."""


class SemanticUnavailable(LookupError):
    """mode=semantic/hybrid requested but no embeddings are loaded."""


class UnknownDocument(LookupError):
    """No document with the requested id exists in the index."""


@dataclass(frozen=True, slots=True)
class Hit:
    """A ranked Lab 3 ``SearchResult`` plus what the web layer shows with it."""

    result: SearchResult
    rank: int
    url: str | None
    spans: Spans


@dataclass(frozen=True, slots=True)
class ResultPage:
    query: str
    scorer: ScorerName
    mode: SearchMode
    page: int
    k: int
    total: int
    took_ms: float
    terms: tuple[str, ...]
    hits: tuple[Hit, ...]

    @property
    def pages(self) -> int:
        return math.ceil(self.total / self.k) if self.total else 0


def scorer_from_name(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def _passage_spans(passage: str, terms: tuple[str, ...], width: int = 240) -> Spans:
    """Snippet for a semantic hit: the start of its best passage, highlighted."""
    text = " ".join(passage.split())
    window = text[:width]
    if len(text) > width:
        cut = window.rfind(" ")
        window = (window[:cut] if cut > width // 2 else window) + "…"
    return highlight_spans(window, terms)


def search_page(
    index: Index,
    query: str,
    *,
    scorer: ScorerName = "bm25",
    k: int = 10,
    page: int = 1,
    mode: SearchMode = "keyword",
    semantic: SemanticIndex | None = None,
    embedder: Embedder | None = None,
) -> ResultPage:
    """Rank documents in the requested mode and return one page of results.

    Ranking keeps only ``page * k`` items (``argpartition`` / heap), and
    snippets are built only for the page that is returned.
    """
    started = time.perf_counter()
    try:
        found = search_mode(
            index,
            query,
            mode=mode,
            scorer=scorer_from_name(scorer),
            k=page * k,
            semantic=semantic,
            embedder=embedder,
        )
    except LookupError as exc:
        raise SemanticUnavailable(str(exc)) from exc
    except ValueError as exc:
        raise QueryError(str(exc)) from exc

    hits: list[Hit] = []
    window = found.hits[(page - 1) * k : page * k]
    for offset, mode_hit in enumerate(window):
        meta = index.doc_meta[mode_hit.doc_id]
        if mode_hit.chunk is not None and semantic is not None:
            spans = _passage_spans(semantic.passage(index, mode_hit.chunk), found.terms)
        else:
            spans = snippet_spans(meta.path, found.terms)
        hits.append(
            Hit(
                result=SearchResult(
                    doc_id=mode_hit.doc_id, score=mode_hit.score, title=meta.title
                ),
                rank=(page - 1) * k + offset + 1,
                url=source_url(meta.path),
                spans=spans,
            )
        )
    took_ms = (time.perf_counter() - started) * 1000
    return ResultPage(
        query=query,
        scorer=scorer,
        mode=mode,
        page=page,
        k=k,
        total=found.total,
        took_ms=took_ms,
        terms=found.terms,
        hits=tuple(hits),
    )


def to_response(result_page: ResultPage) -> SearchResponse:
    """Convert internal dataclasses to the public Pydantic schema."""
    return SearchResponse(
        query=result_page.query,
        scorer=result_page.scorer,
        mode=result_page.mode,
        page=result_page.page,
        k=result_page.k,
        total=result_page.total,
        pages=result_page.pages,
        took_ms=round(result_page.took_ms, 3),
        results=[
            SearchHit(
                doc_id=hit.result.doc_id,
                title=hit.result.title,
                score=hit.result.score,
                snippet="".join(text for text, _ in hit.spans),
                url=hit.url,
            )
            for hit in result_page.hits
        ],
    )


def get_document(index: Index, doc_id: int) -> DocumentResponse:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        raise UnknownDocument(doc_id)
    return DocumentResponse(
        doc_id=meta.doc_id,
        title=meta.title,
        url=source_url(meta.path),
        length=meta.length,
        text=read_document_text(meta.path),
    )
