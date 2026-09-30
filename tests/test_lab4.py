from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import Document
from findex.index import DocMeta, Index, Posting, build_index
from findex.search import merge_and, merge_or
from findex.store import load, save
from findex.tokenize import tokenize

runner = CliRunner()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD", ["hello", "world"]),
        ("Привіт УКРАЇНО", ["привіт", "україно"]),
        ("cafe\u0301", ["café"]),
        ("don't п'ять", ["don't", "п'ять"]),
        ("state-of-the-art", ["state", "of", "the", "art"]),
        ("Python3 2026", ["python3", "2026"]),
    ],
)
def test_tokenize_parametrized(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tiny_corpus_fixture_is_built(tiny_index: Index) -> None:
    assert tiny_index.num_docs == 3
    assert tiny_index.df("python") == 2


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout


def test_cli_index_command(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.bin"
    result = runner.invoke(
        app,
        ["index", str(tiny_corpus), "--out", str(out), "--positions"],
    )
    assert result.exit_code == 0
    assert out.exists()


def test_cli_search_json(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.bin"
    build = runner.invoke(
        app,
        ["index", str(tiny_corpus), "--out", str(out), "--positions"],
    )
    assert build.exit_code == 0
    result = runner.invoke(
        app,
        ["search", str(out), "python", "--json", "-k", "2"],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert rows
    assert all("doc_id" in row and "score" in row for row in rows)


def test_cli_stats(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.bin"
    build = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(out)])
    assert build.exit_code == 0
    result = runner.invoke(app, ["stats", str(out)])
    assert result.exit_code == 0
    assert "Документи" in result.stdout


def test_cli_missing_index_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert result.exit_code != 0
    assert "Помилка:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_bad_query_is_friendly(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.bin"
    build = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(out)])
    assert build.exit_code == 0
    result = runner.invoke(app, ["search", str(out), "("])
    assert result.exit_code != 0
    assert "Помилка:" in result.stderr
    assert "Traceback" not in result.stderr


def test_json_load_save_round_trip(tiny_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    save(tiny_index, path, format="json")
    restored = load(path, format="json")
    assert restored.postings == tiny_index.postings
    assert restored.doc_meta == tiny_index.doc_meta


def test_bm25_rare_term_has_higher_score(tiny_index: Index) -> None:
    from findex.ranking import BM25

    postings = {
        "common": (Posting(0, 1), Posting(1, 1), Posting(2, 1)),
        "rare": (Posting(0, 1),),
    }
    index = Index(
        postings,
        {
            0: DocMeta(0, "a.txt", "a", 3),
            1: DocMeta(1, "b.txt", "b", 3),
            2: DocMeta(2, "c.txt", "c", 3),
        },
    )
    scorer = BM25()
    assert scorer("rare", index["rare"][0], index) > scorer(
        "common", index["common"][0], index
    )


def test_bm25_repetition_saturates() -> None:
    from findex.ranking import BM25

    meta = {0: DocMeta(0, "a.txt", "a", 100)}
    one = Index({"python": (Posting(0, 1),)}, meta)
    nineteen = Index({"python": (Posting(0, 19),)}, meta)
    twenty = Index({"python": (Posting(0, 20),)}, meta)
    scorer = BM25()
    score_one = scorer("python", one["python"][0], one)
    score_nineteen = scorer("python", nineteen["python"][0], nineteen)
    score_twenty = scorer("python", twenty["python"][0], twenty)
    assert score_twenty > score_nineteen > score_one
    assert score_twenty - score_nineteen < score_nineteen - score_one


def test_bm25_prefers_shorter_document_for_same_tf() -> None:
    from findex.ranking import BM25

    index = Index(
        {"python": (Posting(0, 1), Posting(1, 1))},
        {
            0: DocMeta(0, "short.txt", "short", 5),
            1: DocMeta(1, "long.txt", "long", 100),
        },
    )
    scorer = BM25()
    assert scorer("python", index["python"][0], index) > scorer(
        "python", index["python"][1], index
    )


@given(st.text())
def test_property_tokenize_never_yields_empty(text: str) -> None:
    assert all(token for token in tokenize(text))


@given(
    rows=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=12),
            st.text(max_size=80),
        ),
        min_size=0,
        max_size=8,
    )
)
def test_property_index_round_trip(
    rows: list[tuple[str, str]],
) -> None:
    documents = [
        Document(f"doc-{i}.txt", Path(f"doc-{i}.txt"), text)
        for i, (_, text) in enumerate(rows)
    ]

    index = build_index(documents, positions=True)

    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "index.json"

        save(index, path, format="json")
        restored = load(path, format="json")

    assert restored.postings == index.postings
    assert restored.doc_meta == index.doc_meta


@given(
    st.lists(st.integers(min_value=0, max_value=100), unique=True),
    st.lists(st.integers(min_value=0, max_value=100), unique=True),
)
def test_property_merge_matches_sets(left: list[int], right: list[int]) -> None:
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    left_postings = tuple(Posting(doc_id=x, tf=1) for x in left_sorted)
    right_postings = tuple(Posting(doc_id=x, tf=1) for x in right_sorted)
    assert merge_and(left_postings, right_postings) == sorted(set(left) & set(right))
    assert merge_or(left_postings, right_postings) == sorted(set(left) | set(right))


def test_postings_are_sorted(tiny_index: Index) -> None:
    assert all(
        [p.doc_id for p in postings] == sorted(p.doc_id for p in postings)
        for postings in tiny_index.postings.values()
    )


@given(st.lists(st.text(max_size=60), min_size=0, max_size=10))
def test_property_postings_are_sorted(texts: list[str]) -> None:
    documents = [
        Document(f"doc-{i}.txt", Path(f"doc-{i}.txt"), text)
        for i, text in enumerate(texts)
    ]
    index = build_index(documents)
    assert all(
        [posting.doc_id for posting in postings]
        == sorted(posting.doc_id for posting in postings)
        for postings in index.postings.values()
    )


def test_cli_verbose_flag(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.bin"
    result = runner.invoke(
        app,
        ["-vv", "index", str(tiny_corpus), "--out", str(out)],
    )
    assert result.exit_code == 0


def test_cli_limit(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "index.bin"
    result = runner.invoke(
        app,
        ["index", str(tiny_corpus), "--out", str(out), "--limit", "2"],
    )
    assert result.exit_code == 0
    restored = load(out)
    assert restored.num_docs == 2
