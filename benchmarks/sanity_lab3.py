"""Run the three ranking sanity checks required by Lab 3."""

from __future__ import annotations

import math

from findex.index import DocMeta, Index, Posting
from findex.ranking import BM25, TfIdf


def controlled_index(
    postings: dict[str, tuple[Posting, ...]],
    lengths: dict[int, int],
) -> Index:
    meta = {
        doc_id: DocMeta(
            doc_id,
            f"doc-{doc_id}.txt",
            f"doc-{doc_id}",
            length,
        )
        for doc_id, length in lengths.items()
    }
    return Index(postings, meta)


def main() -> None:
    index = controlled_index(
        {
            "common": (Posting(0, 1), Posting(1, 1), Posting(2, 1)),
            "rare": (Posting(1, 1),),
        },
        {0: 10, 1: 10, 2: 10},
    )
    tfidf = TfIdf()
    bm25 = BM25()

    rare_score = tfidf("rare", Posting(1, 1), index)
    common_score = tfidf("common", Posting(0, 1), index)
    print("1. Rare term vs common term")
    print(f"   rare TF-IDF score:   {rare_score:.6f}")
    print(f"   common TF-IDF score: {common_score:.6f}")

    index2 = controlled_index(
        {"python": (Posting(0, 1),)},
        {0: 100},
    )
    score1 = bm25("python", Posting(0, 1), index2)
    score19 = bm25("python", Posting(0, 19), index2)
    score20 = bm25("python", Posting(0, 20), index2)
    print("2. BM25 repetition saturation")
    print(f"   tf=1:  {score1:.6f}")
    print(f"   tf=19: {score19:.6f}")
    print(f"   tf=20: {score20:.6f}")
    print(f"   20th incremental gain: {(score20 - score19):.6f}")

    index3 = controlled_index(
        {"python": (Posting(0, 1), Posting(1, 1))},
        {0: 10, 1: 1000},
    )
    short_score = bm25("python", index3.postings["python"][0], index3)
    long_score = bm25("python", index3.postings["python"][1], index3)
    ratio = short_score / max(long_score, math.ulp(1.0))
    print("3. Short document vs long document")
    print(f"   short score: {short_score:.6f}")
    print(f"   long score:  {long_score:.6f}")
    print(f"   ratio short/long: {ratio:.2f}")


if __name__ == "__main__":
    main()
