"""findex: streaming corpus intake for a small full-text search engine."""

from .corpus import Document, iter_documents
from .tokenize import tokenize

__all__ = ["Document", "iter_documents", "tokenize"]
