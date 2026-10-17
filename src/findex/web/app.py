"""FastAPI application: JSON API + server-rendered search page."""

from __future__ import annotations

import asyncio
import logging
import textwrap
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..index import Index
from ..snippets import highlight_spans, source_url
from ..tokenize import tokenize
from .logs import configure_logging, request_id_var
from .models import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    ScorerName,
    SearchResponse,
    StatsResponse,
)
from .service import (
    QueryError,
    ResultPage,
    UnknownDocument,
    get_document,
    search_page,
    to_response,
)
from .settings import Settings, get_settings
from .state import IndexDep, IndexState, get_loaded_index, get_state, load_index

logger = logging.getLogger("findex.web")

HERE = Path(__file__).parent
MAX_K = 50
MAX_PAGE = 100

try:
    __version__ = version("findex")
except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
    __version__ = "0.0.0"

# Shared, validated query parameters (Pydantic reads the type hints).
QueryParam = Annotated[
    str, Query(min_length=1, max_length=200, description="Query; AND/OR/NOT, quotes")
]
KParam = Annotated[int, Query(ge=1, le=MAX_K, description="Results per page")]
PageParam = Annotated[int, Query(ge=1, le=MAX_PAGE, description="1-based page")]
ScorerParam = Annotated[ScorerName, Query(description="Ranking model")]


def _query_error(exc: QueryError) -> HTTPException:
    return HTTPException(status_code=422, detail=f"Invalid query: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load the index once per process; close it on shutdown (SIGTERM)."""
    settings: Settings = app.state.settings
    state = IndexState(path=settings.index_path)
    app.state.findex = state
    loader = asyncio.create_task(load_index(state))
    try:
        yield
    finally:
        loader.cancel()
        with suppress(asyncio.CancelledError):
            await loader
        if state.index is not None:
            state.index.close()
        logger.info("shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title="findex",
        version=__version__,
        summary="Full-text search over Python Enhancement Proposals.",
        lifespan=lifespan,
        swagger_ui_oauth2_redirect_url=None,
        responses={500: {"model": ErrorResponse}},
    )
    app.state.settings = settings
    app.dependency_overrides[get_settings] = lambda: settings

    templates = Jinja2Templates(directory=HERE / "templates")
    template_globals = cast(dict[str, object], templates.env.globals)
    template_globals["site_name"] = settings.site_name
    template_globals["version"] = __version__
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Request id + access log + a clean 500 for anything unexpected."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error on %s %s", request.method, request.url)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "request_id": request_id},
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d %.1f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        request_id_var.reset(token)
        return response

    # ------------------------------------------------------------------ API

    if settings.search_route == "def":

        @app.get("/search", response_model=SearchResponse, tags=["api"])
        def search(  # pyright: ignore[reportUnusedFunction]
            index: IndexDep,
            q: QueryParam,
            k: KParam = 10,
            scorer: ScorerParam = "bm25",
            page: PageParam = 1,
        ) -> SearchResponse:
            """Ranked search. A plain ``def``: FastAPI runs it in its thread pool,
            so the CPU-bound scoring never blocks the event loop."""
            try:
                result = search_page(index, q, scorer=scorer, k=k, page=page)
            except QueryError as exc:
                raise _query_error(exc) from None
            return to_response(result)

    else:

        @app.get("/search", response_model=SearchResponse, tags=["api"])
        async def search_async(  # pyright: ignore[reportUnusedFunction]
            index: IndexDep,
            q: QueryParam,
            k: KParam = 10,
            scorer: ScorerParam = "bm25",
            page: PageParam = 1,
        ) -> SearchResponse:
            """Experiment only (``SEARCH_ROUTE=async``): CPU work directly inside
            ``async def`` — it runs on the event loop and blocks it."""
            try:
                result = search_page(index, q, scorer=scorer, k=k, page=page)
            except QueryError as exc:
                raise _query_error(exc) from None
            return to_response(result)

    @app.get(
        "/docs/{doc_id}",
        response_model=DocumentResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["api"],
    )
    def document(index: IndexDep, doc_id: int) -> DocumentResponse:  # pyright: ignore[reportUnusedFunction]
        """Full text and metadata of one document."""
        try:
            return get_document(index, doc_id)
        except UnknownDocument:
            raise HTTPException(
                status_code=404, detail=f"Document {doc_id} not found"
            ) from None

    @app.get("/stats", response_model=StatsResponse, tags=["api"])
    def stats(  # pyright: ignore[reportUnusedFunction]
        index: IndexDep,
        state: Annotated[IndexState, Depends(get_state)],
    ) -> StatsResponse:
        """Index size, vocabulary and process uptime."""
        return StatsResponse(
            version=__version__,
            documents=index.num_docs,
            vocabulary=len(index),
            postings=state.postings
            or sum(len(items) for items in index.postings.values()),
            avg_doc_length=round(index.avg_doc_length, 2),
            index_path=str(state.path),
            index_bytes=state.index_bytes,
            load_seconds=round(state.load_seconds, 3),
            uptime_seconds=round(time.monotonic() - state.started_at, 1),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["api"],
    )
    def health(  # pyright: ignore[reportUnusedFunction]
        index: Annotated[Index | None, Depends(get_loaded_index)],
        state: Annotated[IndexState, Depends(get_state)],
    ) -> JSONResponse:
        """200 once the index is in memory; 503 while loading or after a failure."""
        if index is None:
            status_name = "error" if state.status == "error" else "loading"
            body = HealthResponse(status=status_name, detail=state.error)
            return JSONResponse(status_code=503, content=body.model_dump())
        body = HealthResponse(status="ok", documents=index.num_docs)
        return JSONResponse(status_code=200, content=body.model_dump())

    # ------------------------------------------------------------------- UI

    def _render_search(
        request: Request,
        index: Index,
        q: str,
        k: int,
        scorer: ScorerName,
        page: int,
        template: str,
    ) -> HTMLResponse:
        result: ResultPage | None = None
        error: str | None = None
        q = q.strip()
        if q:
            try:
                result = search_page(index, q, scorer=scorer, k=k, page=page)
            except QueryError as exc:
                error = str(exc)
        return templates.TemplateResponse(
            request,
            template,
            {
                "q": q,
                "k": k,
                "scorer": scorer,
                "page": page,
                "result": result,
                "error": error,
                "num_docs": index.num_docs,
            },
            status_code=422 if error else 200,
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        index: IndexDep,
        q: Annotated[str, Query(max_length=200)] = "",
        k: KParam = 10,
        scorer: ScorerParam = "bm25",
        page: PageParam = 1,
    ) -> HTMLResponse:
        return _render_search(request, index, q, k, scorer, page, "index.html")

    @app.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def results_fragment(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        index: IndexDep,
        q: Annotated[str, Query(max_length=200)] = "",
        k: KParam = 10,
        scorer: ScorerParam = "bm25",
        page: PageParam = 1,
    ) -> HTMLResponse:
        """htmx target: only the results block, for search-as-you-type."""
        response = _render_search(request, index, q, k, scorer, page, "_results.html")
        # Keep the address bar on the shareable page URL, not the fragment URL.
        query_string = request.url.query
        response.headers["HX-Push-Url"] = f"/?{query_string}" if q else "/"
        return response

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_page(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        index: IndexDep,
        doc_id: int,
        q: Annotated[str, Query(max_length=200)] = "",
    ) -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"q": q, "doc_id": doc_id},
                status_code=404,
            )
        doc = get_document(index, doc_id)
        terms = tuple(dict.fromkeys(tokenize(q))) if q else ()
        paragraphs: list[tuple[list[tuple[str, bool]], bool]] = []
        for block in doc.text.split("\n\n"):
            if not block.strip():
                continue
            lines = block.splitlines()
            # Indented reST blocks are code/literals: keep their line breaks.
            is_code = all(line.startswith((" ", "\t")) for line in lines if line)
            text = textwrap.dedent(block) if is_code else " ".join(block.split())
            paragraphs.append((highlight_spans(text.strip("\n"), terms), is_code))
        return templates.TemplateResponse(
            request,
            "doc.html",
            {
                "q": q,
                "doc": doc,
                "paragraphs": paragraphs,
                "url": source_url(meta.path),
            },
        )

    return app
