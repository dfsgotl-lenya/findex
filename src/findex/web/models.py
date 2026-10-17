"""Pydantic models at the HTTP boundary.

Inside the engine everything stays the frozen dataclasses from Labs 2-3
(``Posting``, ``DocMeta``, ``SearchResult``). These models exist only to
validate what comes in and to describe/serialize what goes out; the
conversion happens in :mod:`findex.web.service`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScorerName = Literal["bm25", "tfidf"]


class SearchHit(BaseModel):
    doc_id: int = Field(ge=0)
    title: str
    score: float
    snippet: str = Field(description="Plain-text window around the first match.")
    url: str | None = Field(default=None, description="Original source URL.")


class SearchResponse(BaseModel):
    query: str
    scorer: ScorerName
    page: int = Field(ge=1)
    k: int = Field(ge=1, description="Results per page.")
    total: int = Field(ge=0, description="Documents matching the query.")
    pages: int = Field(ge=0)
    took_ms: float
    results: list[SearchHit]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    url: str | None
    length: int = Field(description="Document length in tokens.")
    text: str


class StatsResponse(BaseModel):
    version: str
    documents: int
    vocabulary: int
    postings: int
    avg_doc_length: float
    index_path: str
    index_bytes: int
    load_seconds: float
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    documents: int | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
