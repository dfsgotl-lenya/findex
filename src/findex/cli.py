"""Typer/Rich command-line interface for the packaged findex tool."""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .corpus import Document, iter_documents
from .index import Index, build_index
from .ranking import BM25, Scorer, SearchResult, TfIdf
from .search import search
from .store import open_index, save

Engine = Literal["merge", "set"]
ScorerName = Literal["bm25", "tfidf"]
IndexFormat = Literal["pickle", "json"]

app = typer.Typer(
    name="findex",
    help="Навчальний повнотекстовий пошуковий рушій.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)
logger = logging.getLogger(__name__)


def configure_logging(verbosity: int) -> None:
    """Configure application logging once at the CLI entry point."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _validate_corpus_path(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"корпус не знайдено: {path}")
    if path.is_file() and path.suffix.lower() != ".txt":
        raise ValueError(f"очікувався каталог або .txt файл: {path}")


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit < 0:
        raise ValueError("--limit має бути >= 0")


def _count_documents(root: Path, limit: int | None) -> int | None:
    if root.is_file():
        return 1 if limit is None or limit > 0 else 0
    total = sum(1 for _ in root.rglob("*.txt"))
    return min(total, limit) if limit is not None else total


def _iter_with_progress(
    documents: Iterable[Document],
    progress: Progress,
    task_id: TaskID,
) -> Iterator[Document]:
    for document in documents:
        progress.advance(task_id)
        yield document


def _index_documents(
    corpus: Path,
    *,
    limit: int | None,
    positions: bool,
) -> Index:
    total = _count_documents(corpus, limit)
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = TaskID(
            progress.add_task(
                "Побудова індексу",
                total=total,
            )
        )
        if limit is not None:
            documents: Iterator[Document] = islice(iter_documents(corpus), limit)
        else:
            documents = iter_documents(corpus)
        return build_index(
            _iter_with_progress(documents, progress, task_id),
            positions=positions,
        )


def _scorer_from_name(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def _render_results(results: list[SearchResult]) -> None:
    table = Table(title="Результати пошуку")
    table.add_column("#", justify="right")
    table.add_column("Doc ID", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Заголовок")
    table.add_column("Сніпет")
    for rank, result in enumerate(results, start=1):
        table.add_row(
            str(rank),
            str(result.doc_id),
            f"{result.score:.4f}",
            result.title,
            result.snippet,
        )
    console.print(table)


def _json_lines(results: list[SearchResult]) -> None:
    for result in results:
        typer.echo(json.dumps(asdict(result), ensure_ascii=False))


@app.callback()
def _root(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Підвищити рівень логування; -vv вмикає DEBUG.",
        ),
    ] = 0,
) -> None:
    configure_logging(verbose)


@app.command("index")
def index_command(
    corpus: Annotated[Path, typer.Argument(help="Каталог або .txt файл корпусу.")],
    out: Annotated[
        Path,
        typer.Option("--out", help="Куди зберегти індекс."),
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Зберігати позиції токенів для фраз."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Обробити не більше N документів."),
    ] = None,
    format: Annotated[
        IndexFormat | None,
        typer.Option("--format", help="Формат серіалізації."),
    ] = None,
) -> None:
    """Побудувати та зберегти інвертований індекс."""
    try:
        _validate_corpus_path(corpus)
        _validate_limit(limit)
        index = _index_documents(corpus, limit=limit, positions=positions)
        save(index, out, format=format)
        console.print(
            f"Індекс збережено: {out} | документів: {index.num_docs} | "
            f"термів: {len(index):,}"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        error_console.print(f"Помилка: {exc}")
        raise typer.Exit(code=1) from None


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(help="Файл збереженого індексу.")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит.")],
    k: Annotated[int, typer.Option("-k", help="Кількість результатів.")] = 10,
    scorer: Annotated[
        ScorerName,
        typer.Option("--scorer", help="Модель ранжування."),
    ] = "bm25",
    engine: Annotated[
        Engine,
        typer.Option("--engine", help="Boolean engine."),
    ] = "merge",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="JSON Lines у stdout."),
    ] = False,
    format: Annotated[
        IndexFormat | None,
        typer.Option("--format", help="Формат серіалізації."),
    ] = None,
) -> None:
    """Виконати ранжований пошук у збереженому індексі."""
    try:
        if k < 1:
            raise ValueError("-k має бути >= 1")
        with open_index(index_path, format=format) as index:
            results = search(
                index,
                query,
                scorer=_scorer_from_name(scorer),
                k=k,
                engine=engine,
            )
        if as_json:
            _json_lines(results)
        else:
            _render_results(results)
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        error_console.print(f"Помилка: {exc}")
        raise typer.Exit(code=1) from None


@app.command("stats")
def stats_command(
    index_path: Annotated[Path, typer.Argument(help="Файл збереженого індексу.")],
    format: Annotated[
        IndexFormat | None,
        typer.Option("--format", help="Формат серіалізації."),
    ] = None,
) -> None:
    """Показати статистику збереженого індексу."""
    try:
        with open_index(index_path, format=format) as index:
            counts: Counter[str] = Counter(
                {
                    term: sum(posting.tf for posting in postings)
                    for term, postings in index.postings.items()
                }
            )
            table = Table(title="Статистика індексу")
            table.add_column("Показник")
            table.add_column("Значення", justify="right")
            table.add_row("Документи", f"{index.num_docs:,}")
            table.add_row("Словник", f"{len(index):,}")
            table.add_row("Середня довжина", f"{index.avg_doc_length:.2f}")
            table.add_row("Токени", f"{sum(counts.values()):,}")
            console.print(table)

            top = Table(title="Топ-10 термів")
            top.add_column("Терм")
            top.add_column("Частота", justify="right")
            for term, count in counts.most_common(50):
                top.add_row(term, f"{count:,}")
            console.print(top)
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        error_console.print(f"Помилка: {exc}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
