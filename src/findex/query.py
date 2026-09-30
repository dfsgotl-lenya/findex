"""Recursive-descent query language and query-tree objects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tokenize import tokenize

if TYPE_CHECKING:
    from .index import Index

_TOKEN_RE = re.compile(
    r'"(?P<phrase>[^"\n]+)"|(?P<lpar>\()|(?P<rpar>\))|'
    r'(?P<op>AND|OR|NOT)\b|(?P<term>[^\s()"]+)',
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class QueryNode:
    """Base node supporting ``&``, ``|`` and ``~`` composition."""

    def __and__(self, other: QueryNode) -> QueryNode:
        return And(self, other)

    def __or__(self, other: QueryNode) -> QueryNode:
        return Or(self, other)

    def __invert__(self) -> QueryNode:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        return {posting.doc_id for posting in index.postings.get(self.value, ())}

    def positive_terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if any(index.df(term) == 0 for term in self.terms):
            return set()

        posting_maps = {
            term: {posting.doc_id: posting for posting in index[term]}
            for term in self.terms
        }
        candidates = set(posting_maps[self.terms[0]])
        for term in self.terms[1:]:
            candidates &= set(posting_maps[term])

        result: set[int] = set()
        for doc_id in candidates:
            postings = [posting_maps[term][doc_id] for term in self.terms]
            if any(not posting.positions for posting in postings):
                raise ValueError(
                    "Phrase queries require an index built with --positions"
                )
            if _has_consecutive_positions([posting.positions for posting in postings]):
                result.add(doc_id)
        return result

    def positive_terms(self) -> tuple[str, ...]:
        return self.terms


def _has_consecutive_positions(position_lists: list[tuple[int, ...]]) -> bool:
    """Check a phrase using repeated two-pointer merges over positions."""
    starts = position_lists[0]
    for offset, positions in enumerate(position_lists[1:], start=1):
        next_starts: list[int] = []
        i = j = 0
        while i < len(starts) and j < len(positions):
            target = starts[i] + offset
            if target == positions[j]:
                next_starts.append(starts[i])
                i += 1
                j += 1
            elif target < positions[j]:
                i += 1
            else:
                j += 1
        starts = tuple(next_starts)
        if not starts:
            return False
    return bool(starts)


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    child: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids) - self.child.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return ()


class _Parser:
    def __init__(self, query: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        consumed = 0
        for match in _TOKEN_RE.finditer(query):
            if query[consumed : match.start()].strip():
                raise ValueError("Unexpected characters in query")
            consumed = match.end()

            if match.group("phrase") is not None:
                self.tokens.append(("phrase", match.group("phrase")))
            elif match.group("lpar") is not None:
                self.tokens.append(("lpar", "("))
            elif match.group("rpar") is not None:
                self.tokens.append(("rpar", ")"))
            elif match.group("op") is not None:
                self.tokens.append(("op", match.group("op").upper()))
            else:
                self.tokens.append(("term", match.group("term")))

        if query[consumed:].strip():
            raise ValueError("Unexpected characters in query")
        self.pos = 0

    def current(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> tuple[str, str]:
        token = self.current()
        if token is None:
            raise ValueError("Unexpected end of query")
        self.pos += 1
        return token

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise ValueError("Empty query")
        node = self.parse_or()
        current = self.current()
        if current is not None:
            raise ValueError(f"Unexpected token: {current[1]}")
        return node

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while self.current() == ("op", "OR"):
            self.consume()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_not()
        while True:
            current = self.current()
            if current == ("op", "AND"):
                self.consume()
                node = And(node, self.parse_not())
            elif current and (
                current[0] in {"term", "phrase", "lpar"} or current == ("op", "NOT")
            ):
                node = And(node, self.parse_not())
            else:
                break
        return node

    def parse_not(self) -> QueryNode:
        if self.current() == ("op", "NOT"):
            self.consume()
            return Not(self.parse_atom())
        return self.parse_atom()

    def parse_atom(self) -> QueryNode:
        kind, value = self.consume()
        if kind == "term":
            normalized = next(tokenize(value), "")
            if not normalized:
                raise ValueError(f"Invalid term: {value}")
            return Term(normalized)
        if kind == "phrase":
            terms = tuple(tokenize(value))
            if not terms:
                raise ValueError("Empty phrase")
            return Phrase(terms)
        if kind == "lpar":
            node = self.parse_or()
            if self.current() != ("rpar", ")"):
                raise ValueError("Missing closing parenthesis")
            self.consume()
            return node
        raise ValueError(f"Unexpected token: {value}")


def parse(query: str) -> QueryNode:
    """Parse query syntax with implicit AND, OR, NOT, parentheses and phrases."""
    return _Parser(query).parse()
