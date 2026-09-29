"""findex: a small educational full-text search engine."""

from .corpus import Document, iter_documents
from .tokenize import tokenize

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
    "iter_documents",
    "parse",
    "tokenize",
]


def __getattr__(name: str):
    """Lazily expose heavier Lab 2/3 objects without CLI import warnings."""
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
