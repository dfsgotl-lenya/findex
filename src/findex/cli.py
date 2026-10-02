"""Typer/Rich command-line interface for the packaged findex tool."""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .parallel import ExecutorName, build_parallel_index, collect_document_paths
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


def _count_documents(root: Path, limit: int | None) -> int:
    return len(collect_document_paths(root, limit=limit))


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
        typer.echo(
            json.dumps(
                {
                    "doc_id": result.doc_id,
                    "score": result.score,
                    "title": result.title,
                    "snippet": result.snippet,
                },
                ensure_ascii=False,
            )
        )


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
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Кількість воркерів."),
    ] = 1,
    executor: Annotated[
        ExecutorName,
        typer.Option("--executor", help="Модель виконання."),
    ] = "processes",
) -> None:
    """Побудувати та зберегти інвертований індекс."""
    try:
        _validate_corpus_path(corpus)
        _validate_limit(limit)
    except (OSError, ValueError, TypeError) as exc:
        error_console.print(f"Помилка: {exc}")
        raise typer.Exit(code=1) from None

    total = _count_documents(corpus, limit)
    progress_total = max(min(workers, total) if executor != "serial" else 1, 1)
    completed_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"Побудова індексу: {executor}/{workers}",
            total=progress_total,
        )

        def on_partial_complete() -> None:
            nonlocal completed_chunks
            completed_chunks += 1
            progress.update(
                task_id,
                completed=min(progress_total, completed_chunks),
            )

        started = time.perf_counter()
        # Worker exceptions are intentionally not caught here: the caller
        # must see the propagated worker traceback instead of a silent hang.
        result = build_parallel_index(
            corpus,
            workers=workers,
            executor=executor,
            positions=positions,
            limit=limit,
            on_partial_complete=on_partial_complete,
        )
        save_started = time.perf_counter()
        try:
            save(result.index, out, format=format)
        except (
            OSError,
            ValueError,
            TypeError,
            EOFError,
            pickle.UnpicklingError,
        ) as exc:
            error_console.print(f"Помилка: {exc}")
            raise typer.Exit(code=1) from None
        save_time = time.perf_counter() - save_started
        wall = time.perf_counter() - started

    if total == 0:
        console.print("Індекс порожній: у корпусі немає .txt документів.")
    else:
        console.print(
            f"Індекс збережено: {out} | документів: "
            f"{result.index.num_docs} | термів: {len(result.index):,}"
        )
    logger.info("executor: %s", executor)
    logger.info("workers: %d", workers)
    logger.info("build time: %.4f s", result.build_time)
    logger.info("merge time: %.4f s", result.merge_time)
    logger.info("save time: %.4f s", save_time)
    logger.info("wall time: %.4f s", wall)


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

            top = Table(title="Топ-50 термів")
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
