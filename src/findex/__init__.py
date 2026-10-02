"""findex: a small educational full-text search engine."""

from typing import TYPE_CHECKING

from .corpus import Document, iter_documents
from .tokenize import tokenize

if TYPE_CHECKING:
    from .index import (
        DocMeta,
        Index,
        InvertedIndex,
        Posting,
        build_index,
        build_index_from_path,
    )
    from .parallel import build_parallel_index, build_partial, merge
    from .query import And, Not, Or, Phrase, QueryNode, Term, parse
    from .ranking import BM25, Scorer, SearchResult, TfIdf


__all__ = [
    "And",
    "BM25",
    "DocMeta",
    "Document",
    "Index",
    "InvertedIndex",
    "Not",
    "Or",
    "Phrase",
    "Posting",
    "QueryNode",
    "Scorer",
    "SearchResult",
    "Term",
    "TfIdf",
    "build_index",
    "build_index_from_path",
    "build_parallel_index",
    "build_partial",
    "iter_documents",
    "merge",
    "parse",
    "tokenize",
]


def __getattr__(name: str):
    """Lazily expose heavier project objects."""
    if name in {
        "DocMeta",
        "Index",
        "InvertedIndex",
        "Posting",
        "build_index",
        "build_index_from_path",
    }:
        from .index import (
            DocMeta,
            Index,
            InvertedIndex,
            Posting,
            build_index,
            build_index_from_path,
        )

        return {
            "DocMeta": DocMeta,
            "Index": Index,
            "InvertedIndex": InvertedIndex,
            "Posting": Posting,
            "build_index": build_index,
            "build_index_from_path": build_index_from_path,
        }[name]

    if name in {"build_parallel_index", "build_partial", "merge"}:
        from .parallel import build_parallel_index, build_partial, merge

        return {
            "build_parallel_index": build_parallel_index,
            "build_partial": build_partial,
            "merge": merge,
        }[name]

    if name in {"And", "Not", "Or", "Phrase", "QueryNode", "Term", "parse"}:
        from .query import And, Not, Or, Phrase, QueryNode, Term, parse

        return {
            "And": And,
            "Not": Not,
            "Or": Or,
            "Phrase": Phrase,
            "QueryNode": QueryNode,
            "Term": Term,
            "parse": parse,
        }[name]

    if name in {"BM25", "Scorer", "SearchResult", "TfIdf"}:
        from .ranking import BM25, Scorer, SearchResult, TfIdf

        return {
            "BM25": BM25,
            "Scorer": Scorer,
            "SearchResult": SearchResult,
            "TfIdf": TfIdf,
        }[name]

    raise AttributeError(name)
