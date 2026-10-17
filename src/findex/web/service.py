"""Glue between the HTTP layer and the Lab 2-3 engine.

Everything here is plain synchronous, CPU-bound Python. The routes decide how
to run it (thread pool vs. event loop); see ``app.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..index import Index
from ..query import parse
from ..ranking import BM25, Scorer, SearchResult, TfIdf
from ..search import rank
from ..snippets import read_document_text, snippet_spans, source_url
from .models import (
    DocumentResponse,
    ScorerName,
    SearchHit,
    SearchResponse,
)

Spans = list[tuple[str, bool]]


class QueryError(ValueError):
    """The query string could not be parsed or evaluated."""


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


def search_page(
    index: Index,
    query: str,
    *,
    scorer: ScorerName = "bm25",
    k: int = 10,
    page: int = 1,
) -> ResultPage:
    """Rank every matching document and return one page of results.

    ``top_k`` keeps a heap of ``page * k`` items, so deep pages cost
    O(n log(page·k)); snippets are only built for the page that is returned.
    """
    started = time.perf_counter()
    try:
        tree = parse(query)
        matched = index.evaluate_query(query)
    except ValueError as exc:
        raise QueryError(str(exc)) from exc

    terms = tree.positive_terms()
    ranked = rank(index, terms, matched, scorer_from_name(scorer), page * k)
    window = ranked[(page - 1) * k : page * k]

    hits: list[Hit] = []
    for offset, result in enumerate(window):
        meta = index.doc_meta[result.doc_id]
        hits.append(
            Hit(
                result=result,
                rank=(page - 1) * k + offset + 1,
                url=source_url(meta.path),
                spans=snippet_spans(meta.path, terms),
            )
        )
    took_ms = (time.perf_counter() - started) * 1000
    return ResultPage(
        query=query,
        scorer=scorer,
        page=page,
        k=k,
        total=len(matched),
        took_ms=took_ms,
        terms=terms,
        hits=tuple(hits),
    )


def to_response(result_page: ResultPage) -> SearchResponse:
    """Convert internal dataclasses to the public Pydantic schema."""
    return SearchResponse(
        query=result_page.query,
        scorer=result_page.scorer,
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
