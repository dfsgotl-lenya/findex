"""TF-IDF and BM25 ranking models for Lab 3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from heapq import nlargest
from typing import Protocol

from .index import Index, Posting


class Scorer(Protocol):
    """Protocol for interchangeable scoring models."""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """TF-IDF with logarithmic TF and IDF."""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or index.num_docs == 0 or posting.tf <= 0:
            return 0.0
        tf = 1.0 + math.log(posting.tf)
        idf = math.log(index.num_docs / df)
        return tf * idf


@dataclass(frozen=True, slots=True)
class BM25:
    """BM25 with the lab defaults ``k1=1.5`` and ``b=0.75``."""

    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or index.num_docs == 0 or posting.tf <= 0:
            return 0.0
        avgdl = index.avg_doc_length or 1.0
        idf = math.log(
            1.0 + (index.num_docs - df + 0.5) / (df + 0.5)
        )
        dl = index.doc_length(posting.doc_id)
        denominator = posting.tf + self.k1 * (
            1.0 - self.b + self.b * dl / avgdl
        )
        tf_component = posting.tf * (self.k1 + 1.0) / denominator
        return idf * tf_component


def score_terms(
    index: Index,
    terms: tuple[str, ...],
    matched_ids: frozenset[int] | set[int],
    scorer: Scorer,
) -> dict[int, float]:
    """Accumulate scores only for documents matched by the query tree."""
    scores = {doc_id: 0.0 for doc_id in matched_ids}
    for term in terms:
        for posting in index.postings.get(term, ()):
            if posting.doc_id in matched_ids:
                scores[posting.doc_id] += scorer(term, posting, index)
    return scores


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    """Ranked result; ``top_k`` returns higher scores first."""

    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)

    def __str__(self) -> str:
        return (
            f"[{self.doc_id}] {self.title} "
            f"score={self.score:.4f} — {self.snippet}"
        )


def top_k(
    scores: dict[int, float],
    index: Index,
    k: int,
) -> list[SearchResult]:
    """Return top-k results with ``heapq.nlargest`` in O(n log k)."""
    if k <= 0:
        return []
    items = nlargest(
        k,
        scores.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return [
        SearchResult(
            doc_id=doc_id,
            score=score,
            title=index.doc_meta[doc_id].title,
        )
        for doc_id, score in items
    ]
