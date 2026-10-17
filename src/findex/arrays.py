"""Lab 8: postings as NumPy arrays and vectorized BM25 / TF-IDF.

Layout (CSR, "compressed sparse row"): all postings of all terms live in two
flat ``int32`` arrays; term ``t`` owns the slice ``offsets[t]:offsets[t + 1]``.

    terms     {"python": 0, "typing": 1, ...}
    offsets   int64[V + 1]
    doc_ids   int32[P]        sorted inside each term's slice
    tfs       int32[P]
    doc_len   int32[N]

Scoring one term is then a handful of whole-array operations executed in C
instead of a Python call per posting. The formulas below keep *exactly* the
operation order of :class:`findex.ranking.BM25` / :class:`TfIdf`, and logs are
taken with :func:`math.log` (not ``np.log``), so scores are bit-identical to
Lab 3 and the ranking is the same — that is tested, not assumed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .index import Index
from .ranking import BM25, Scorer, SearchResult, TfIdf

Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class ArrayIndex:
    """Read-only NumPy view of an :class:`Index` used for ranking."""

    terms: dict[str, int]
    offsets: Int64Array
    doc_ids: Int32Array
    tfs: Int32Array
    doc_len: Int32Array
    titles: tuple[str, ...]

    # ------------------------------------------------------------ building

    @classmethod
    def from_index(cls, index: Index) -> ArrayIndex:
        """Copy postings into flat arrays (doc ids must be 0..N-1)."""
        num_docs = index.num_docs
        if index.doc_ids != tuple(range(num_docs)):
            raise ValueError("ArrayIndex needs dense document ids 0..N-1")
        postings = index.postings
        terms = {term: number for number, term in enumerate(postings)}
        lengths = np.fromiter(
            (len(items) for items in postings.values()),
            dtype=np.int64,
            count=len(postings),
        )
        offsets = np.zeros(len(postings) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])
        doc_ids = np.fromiter(
            (posting.doc_id for items in postings.values() for posting in items),
            dtype=np.int32,
            count=total,
        )
        tfs = np.fromiter(
            (posting.tf for items in postings.values() for posting in items),
            dtype=np.int32,
            count=total,
        )
        if total and int(doc_ids.max()) >= num_docs:
            raise ValueError("posting refers to a document outside 0..N-1")
        doc_len = np.fromiter(
            (index.doc_meta[doc_id].length for doc_id in range(num_docs)),
            dtype=np.int32,
            count=num_docs,
        )
        titles = tuple(index.doc_meta[doc_id].title for doc_id in range(num_docs))
        return cls(terms, offsets, doc_ids, tfs, doc_len, titles)

    def save(self, path: Path) -> None:
        """Store as ``.npz``: a few raw buffers, loaded in milliseconds."""
        vocabulary = np.array(list(self.terms), dtype=np.str_)
        np.savez(
            path,
            vocabulary=vocabulary,
            offsets=self.offsets,
            doc_ids=self.doc_ids,
            tfs=self.tfs,
            doc_len=self.doc_len,
            titles=np.array(self.titles, dtype=np.str_),
        )

    @classmethod
    def load(cls, path: Path) -> ArrayIndex:
        with np.load(path, allow_pickle=False) as data:
            vocabulary = [str(term) for term in data["vocabulary"]]
            return cls(
                terms={term: number for number, term in enumerate(vocabulary)},
                offsets=data["offsets"].astype(np.int64),
                doc_ids=data["doc_ids"].astype(np.int32),
                tfs=data["tfs"].astype(np.int32),
                doc_len=data["doc_len"].astype(np.int32),
                titles=tuple(str(title) for title in data["titles"]),
            )

    # ------------------------------------------------------------ queries

    @property
    def num_docs(self) -> int:
        return int(self.doc_len.shape[0])

    @property
    def avg_doc_length(self) -> float:
        # Same summation as Index.avg_doc_length (Python ints, then one divide).
        if self.num_docs == 0:
            return 0.0
        return sum(int(length) for length in self.doc_len) / self.num_docs

    def postings(self, term: str) -> tuple[Int32Array, Int32Array]:
        """``(doc_ids, tfs)`` views for one term; empty arrays if unknown."""
        number = self.terms.get(term)
        if number is None:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        start, end = int(self.offsets[number]), int(self.offsets[number + 1])
        return self.doc_ids[start:end], self.tfs[start:end]

    def df(self, term: str) -> int:
        number = self.terms.get(term)
        if number is None:
            return 0
        return int(self.offsets[number + 1] - self.offsets[number])


class VectorScorer:
    """Scores whole postings lists at once. Built once per index."""

    def __init__(self, arrays: ArrayIndex) -> None:
        self.arrays = arrays
        self._avgdl = arrays.avg_doc_length
        max_tf = int(arrays.tfs.max()) if arrays.tfs.size else 0
        # 1 + log(tf) via a lookup table filled by math.log: bit-identical to
        # TfIdf, and one gather instead of a transcendental per posting.
        table = [0.0] + [1.0 + math.log(tf) for tf in range(1, max_tf + 1)]
        self._log_tf: FloatArray = np.asarray(table, dtype=np.float64)

    def term_scores(self, term: str, scorer: Scorer) -> tuple[Int32Array, FloatArray]:
        doc_ids, tfs = self.arrays.postings(term)
        num_docs = self.arrays.num_docs
        df = doc_ids.shape[0]
        if df == 0 or num_docs == 0:
            return doc_ids, np.empty(0, dtype=np.float64)
        if isinstance(scorer, BM25):
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            avgdl = self._avgdl or 1.0
            tf = tfs.astype(np.float64)
            dl = self.arrays.doc_len[doc_ids]  # fancy indexing: gather lengths
            denominator = tf + scorer.k1 * (1.0 - scorer.b + scorer.b * dl / avgdl)
            tf_component = tf * (scorer.k1 + 1.0) / denominator
            return doc_ids, idf * tf_component
        if isinstance(scorer, TfIdf):
            idf = math.log(num_docs / df)
            return doc_ids, self._log_tf[tfs] * idf
        raise TypeError(f"no vectorized form for {type(scorer).__name__}")

    def score(
        self,
        terms: Sequence[str],
        matched: Iterable[int],
        scorer: Scorer,
    ) -> tuple[Int32Array, FloatArray]:
        """Accumulate scores for the matched documents only.

        Returns ``(candidate_doc_ids, scores)``; like Lab 3's ``score_terms``
        every matched document is a candidate, even with score 0 (``NOT`` only).
        """
        num_docs = self.arrays.num_docs
        candidates = np.fromiter(matched, dtype=np.int32)
        candidates.sort()
        totals = np.zeros(num_docs, dtype=np.float64)
        mask = np.zeros(num_docs, dtype=np.bool_)
        mask[candidates] = True
        for term in terms:
            doc_ids, scores = self.term_scores(term, scorer)
            if doc_ids.size == 0:
                continue
            keep = mask[doc_ids]
            # doc_ids are unique within one term, so += with fancy indexing is
            # safe (np.add.at would only be needed for repeated indices).
            totals[doc_ids[keep]] += scores[keep]
        return candidates, totals[candidates]


def top_k_arrays(
    doc_ids: Int32Array,
    scores: FloatArray,
    k: int,
) -> list[tuple[int, float]]:
    """Top-k by score (desc), ties by doc id (asc) — Lab 3's order.

    ``np.argpartition`` finds the k-th best score in O(n); only the
    candidates at or above it (k plus any ties) are sorted.
    """
    n = int(doc_ids.shape[0])
    if k <= 0 or n == 0:
        return []
    if k < n:
        kth = np.argpartition(-scores, k - 1)[k - 1]
        threshold = scores[kth]
        selected = np.flatnonzero(scores >= threshold)
    else:
        selected = np.arange(n)
    # lexsort: last key is primary -> score descending, then doc id ascending.
    order = selected[np.lexsort((doc_ids[selected], -scores[selected]))][:k]
    return [(int(doc_ids[i]), float(scores[i])) for i in order]


def vector_top_k(
    vectors: VectorScorer,
    terms: Sequence[str],
    matched: Iterable[int],
    scorer: Scorer,
    k: int,
) -> list[SearchResult]:
    """Vectorized replacement for ``score_terms`` + ``ranking.top_k``."""
    doc_ids, scores = vectors.score(terms, matched, scorer)
    titles = vectors.arrays.titles
    return [
        SearchResult(doc_id=doc_id, score=score, title=titles[doc_id])
        for doc_id, score in top_k_arrays(doc_ids, scores, k)
    ]
