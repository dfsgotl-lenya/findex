"""Index lifecycle (loaded once per process) and the dependencies that expose it."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request, status

from ..index import Index
from ..semantic import Embedder, SemanticIndex, load_embedder
from ..store import load

logger = logging.getLogger("findex.web")

LoadStatus = Literal["loading", "ok", "error"]


@dataclass(slots=True)
class IndexState:
    """Mutable holder stored on ``app.state``; only the lifespan writes to it."""

    path: Path
    semantic_dir: Path | None = None
    status: LoadStatus = "loading"
    index: Index | None = None
    error: str | None = None
    load_seconds: float = 0.0
    index_bytes: int = 0
    postings: int = 0
    semantic: SemanticIndex | None = None
    embedder: Embedder | None = None
    semantic_error: str | None = None
    started_at: float = field(default_factory=time.monotonic)


def _load_blocking(state: IndexState) -> None:
    started = time.perf_counter()
    index = load(state.path)
    state.postings = sum(len(items) for items in index.postings.values())
    state.index_bytes = state.path.stat().st_size
    index.vector_scorer()  # build the NumPy postings once, not on 1st request
    _load_semantic(state)
    state.load_seconds = time.perf_counter() - started
    state.index = index
    state.status = "ok"


def _load_semantic(state: IndexState) -> None:
    """Optional: embeddings for mode=semantic/hybrid. Never fatal."""
    directory = state.semantic_dir
    if directory is None or not (directory / "meta.json").exists():
        state.semantic_error = f"no semantic index at {directory}"
        logger.info("semantic search disabled: %s", state.semantic_error)
        return
    try:
        semantic = SemanticIndex.load(directory)
        state.embedder = load_embedder(semantic.model)
        state.semantic = semantic
        logger.info(
            "semantic index: %d chunks x %d (%s)",
            semantic.num_chunks,
            semantic.dim,
            semantic.model,
        )
    except Exception as exc:  # noqa: BLE001 - keyword search still works
        state.semantic_error = f"{type(exc).__name__}: {exc}"
        logger.exception("semantic index failed to load: %s", directory)


async def load_index(state: IndexState) -> None:
    """Load the index in a worker thread so the server can answer /health meanwhile."""
    logger.info("loading index from %s", state.path)
    try:
        await asyncio.to_thread(_load_blocking, state)
    except Exception as exc:  # noqa: BLE001 - reported through /health
        state.status = "error"
        state.error = f"{type(exc).__name__}: {exc}"
        logger.exception("index load failed: %s", state.path)
        return
    index = state.index
    assert index is not None
    logger.info(
        "index loaded: %d docs, %d terms in %.2f s",
        index.num_docs,
        len(index),
        state.load_seconds,
    )


def get_state(request: Request) -> IndexState:
    state = getattr(request.app.state, "findex", None)
    if not isinstance(state, IndexState):
        # Lifespan has not run (e.g. TestClient used without ``with``).
        return IndexState(path=Path("<not started>"), status="loading")
    return state


def get_loaded_index(
    state: Annotated[IndexState, Depends(get_state)],
) -> Index | None:
    """The index if it is ready, else ``None``. Used by /health."""
    return state.index if state.status == "ok" else None


def get_index(
    index: Annotated[Index | None, Depends(get_loaded_index)],
) -> Index:
    """Dependency for every route that needs the index; 503 until it is loaded."""
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index is not loaded yet",
        )
    return index


IndexDep = Annotated[Index, Depends(get_index)]


@dataclass(frozen=True, slots=True)
class SemanticDeps:
    """What mode=semantic/hybrid needs; both ``None`` when unavailable."""

    index: SemanticIndex | None = None
    embedder: Embedder | None = None


def get_semantic(state: Annotated[IndexState, Depends(get_state)]) -> SemanticDeps:
    return SemanticDeps(state.semantic, state.embedder)


SemanticDep = Annotated[SemanticDeps, Depends(get_semantic)]
