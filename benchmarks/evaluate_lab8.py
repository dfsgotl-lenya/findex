"""Lab 8 M3: precision@5 of keyword / semantic / hybrid search on the PEP corpus.

Relevance labels are fixed *before* looking at results:

* topic queries — every PEP whose ``Topic:`` header (kept out of the searchable
  text, see ``make_pep_corpus.py``) contains the topic is relevant;
* feature queries — an explicit list of PEP numbers about that feature.

The 5 paraphrase queries deliberately avoid the key words of their topic.

    uv run findex embed data/peps.bin
    uv run python benchmarks/evaluate_lab8.py data/peps.bin data/peps.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from findex.index import Index
from findex.ranking import BM25
from findex.semantic import SearchMode, SemanticIndex, load_embedder, search_mode
from findex.store import load

PATTERN = {622, 634, 635, 636, 642, 653}
GIL = {703, 684, 779, 554, 734}
ASYNC = {492, 525, 530, 3156, 3152, 789}
WALRUS = {572, 379, 577}
INTERPOLATION = {215, 292, 498, 501, 502, 536, 701, 750, 3101, 787}
DATACLASSES = {557, 681}


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    relevant: str | frozenset[int]  # a Topic name or explicit PEP numbers
    paraphrase: bool = False


QUERIES = [
    Query("type hints", "Typing"),
    Query("packaging metadata", "Packaging"),
    Query("release schedule", "Release"),
    Query("steering council", "Governance"),
    Query("structural pattern matching", frozenset(PATTERN)),
    Query("global interpreter lock", frozenset(GIL)),
    Query("async await coroutines", frozenset(ASYNC)),
    Query("assignment expressions", frozenset(WALRUS)),
    Query("string interpolation", frozenset(INTERPOLATION)),
    Query("data classes", frozenset(DATACLASSES)),
    # Paraphrases: none of "type", "hint", "annotation", "package", "wheel",
    # "match", "pattern", "GIL", "lock", "governance", "council".
    Query("describe which kinds of values a variable may hold", "Typing", True),
    Query("bundle a library so people can install it", "Packaging", True),
    Query("branch on the shape of nested data and pull out parts", PATTERN, True),
    Query("run python threads truly in parallel on several cores", GIL, True),
    Query("who has the final say over changes to the language", "Governance", True),
]


def pep_numbers(index: Index) -> dict[int, int]:
    numbers: dict[int, int] = {}
    for doc_id, meta in index.doc_meta.items():
        match = re.match(r"PEP (\d+)", meta.title)
        if match:
            numbers[doc_id] = int(match.group(1))
    return numbers


def topics_by_pep(corpus: Path) -> dict[int, set[str]]:
    topics: dict[int, set[str]] = {}
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            number = int(re.findall(r"pep-(\d+)", record["url"])[0])
            topics[number] = {t.strip() for t in record.get("topic", "").split(",")}
    return topics


def main() -> None:
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, nargs="?", default=Path("data/peps.bin"))
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("data/peps.jsonl"))
    parser.add_argument("--semantic", type=Path, default=None)
    args = parser.parse_args()

    index = load(args.index)
    semantic = SemanticIndex.load(args.semantic or args.index.with_suffix(".semantic"))
    embedder = load_embedder(semantic.model)
    numbers = pep_numbers(index)
    topics = topics_by_pep(args.corpus)

    def relevant(query: Query) -> set[int]:
        if isinstance(query.relevant, str):
            return {n for n, t in topics.items() if query.relevant in t}
        return set(query.relevant)

    modes: tuple[SearchMode, ...] = ("keyword", "semantic", "hybrid")
    totals: dict[tuple[bool, SearchMode], list[float]] = {}
    print(f"model: {semantic.model}, {semantic.num_chunks} chunks\n")
    print("| # | Query | Kind | Relevant | Keyword P@5 | Semantic P@5 | Hybrid P@5 |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for number, query in enumerate(QUERIES, start=1):
        rel = relevant(query)
        row: list[str] = []
        for mode in modes:
            found = search_mode(
                index,
                query.text,
                mode=mode,
                scorer=BM25(),
                k=5,
                semantic=semantic,
                embedder=embedder,
            )
            top = [numbers[hit.doc_id] for hit in found.hits[:5]]
            p5 = sum(1 for pep in top if pep in rel) / 5
            totals.setdefault((query.paraphrase, mode), []).append(p5)
            row.append(f"{p5:.2f}")
        kind = "paraphrase" if query.paraphrase else "keyword"
        print(
            f"| {number} | {query.text} | {kind} | {len(rel)} | "
            + " | ".join(row)
            + " |"
        )
    for paraphrase, label in ((False, "10 keyword queries"), (True, "5 paraphrases")):
        means = [
            sum(totals[(paraphrase, m)]) / len(totals[(paraphrase, m)]) for m in modes
        ]
        print(
            f"| | **mean, {label}** | | | "
            + " | ".join(f"**{v:.2f}**" for v in means)
            + " |"
        )
    all_means = [
        sum(totals[(False, m)] + totals[(True, m)]) / len(QUERIES) for m in modes
    ]
    print(
        "| | **mean, all 15** | | | "
        + " | ".join(f"**{v:.2f}**" for v in all_means)
        + " |"
    )


if __name__ == "__main__":
    main()
