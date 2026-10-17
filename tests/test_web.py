"""Lab 7: HTTP API and UI tests. No real index file: the tiny fixture index is
injected through ``app.dependency_overrides``."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.index import Index, build_index_from_path
from findex.store import save
from findex.web import create_app
from findex.web.settings import Settings
from findex.web.state import get_index, get_loaded_index


@pytest.fixture
def corpus_index(tiny_corpus: Path) -> Index:
    # Built from real files so snippets and document text can be read back.
    return build_index_from_path(tiny_corpus, positions=True)


def _app(index: Index | None, **settings: object) -> FastAPI:
    application = create_app(
        Settings(index_path=Path("does-not-matter.bin"), **settings)  # type: ignore[arg-type]
    )
    if index is not None:
        application.dependency_overrides[get_index] = lambda: index
        application.dependency_overrides[get_loaded_index] = lambda: index
    return application


@pytest.fixture
def client(corpus_index: Index) -> Iterator[TestClient]:
    # No ``with``: the lifespan (and so the real loader) never runs.
    yield TestClient(_app(corpus_index))


def test_search_success(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "python"
    assert body["scorer"] == "bm25"
    assert body["total"] == 2
    assert body["pages"] == 1
    titles = [hit["title"] for hit in body["results"]]
    assert titles == ["b", "a"]  # tf=2 beats tf=1 in BM25
    assert "Python" in body["results"][0]["snippet"]
    assert body["results"][0]["score"] > body["results"][1]["score"] > 0
    assert response.headers["X-Request-ID"]


def test_search_tfidf_and_boolean_query(client: TestClient) -> None:
    response = client.get("/search", params={"q": "engine NOT java", "scorer": "tfidf"})
    assert response.status_code == 200
    assert {hit["title"] for hit in response.json()["results"]} == {"a", "b"}


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "engine", "k": 2, "page": 1}).json()
    second = client.get("/search", params={"q": "engine", "k": 2, "page": 2}).json()
    beyond = client.get("/search", params={"q": "engine", "k": 2, "page": 3}).json()
    assert first["total"] == second["total"] == 3
    assert first["pages"] == 2
    assert len(first["results"]) == 2
    assert len(second["results"]) == 1
    assert beyond["results"] == []
    seen = {hit["doc_id"] for hit in first["results"] + second["results"]}
    assert len(seen) == 3


def test_search_no_matches(client: TestClient) -> None:
    body = client.get("/search", params={"q": "haskell"}).json()
    assert body["total"] == 0
    assert body["pages"] == 0
    assert body["results"] == []


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "x" * 201},
        {"q": "python", "k": "abc"},
        {"q": "python", "k": 0},
        {"q": "python", "k": 51},
        {"q": "python", "page": 0},
        {"q": "python", "page": 101},
        {"q": "python", "scorer": "pagerank"},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, object]) -> None:
    response = client.get("/search", params=params)  # type: ignore[arg-type]
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_unparsable_query_is_422_with_detail(client: TestClient) -> None:
    response = client.get("/search", params={"q": "(python"})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("Invalid query")


def test_document_found(client: TestClient) -> None:
    response = client.get("/docs/0")
    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == 0
    assert body["text"].startswith("Python async")


def test_document_404(client: TestClient) -> None:
    response = client.get("/docs/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document 999 not found"}


def test_document_id_must_be_int(client: TestClient) -> None:
    assert client.get("/docs/abc").status_code == 422


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 3
    assert body["vocabulary"] > 0
    assert body["postings"] > body["documents"]


def test_health_ok_when_index_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_503_when_index_not_loaded() -> None:
    application = _app(None)
    application.dependency_overrides[get_loaded_index] = lambda: None
    response = TestClient(application).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "loading"


def test_routes_503_until_index_loaded() -> None:
    client = TestClient(_app(None))
    assert client.get("/search", params={"q": "python"}).status_code == 503
    assert client.get("/stats").status_code == 503


def test_unexpected_exception_is_clean_500(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    application = client.app
    assert isinstance(application, FastAPI)

    @application.get("/boom")
    def boom() -> None:  # pyright: ignore[reportUnusedFunction]
        raise RuntimeError("secret internal detail")

    response = client.get("/boom", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal Server Error",
        "request_id": "req-123",
    }
    assert "secret" not in response.text
    assert "Traceback" not in response.text
    logged = [r for r in caplog.records if r.levelname == "ERROR"]
    assert logged and getattr(logged[0], "request_id", None) == "req-123"
    assert logged[0].exc_info is not None


def test_async_variant_returns_same_results(corpus_index: Index) -> None:
    sync_client = TestClient(_app(corpus_index))
    async_client = TestClient(_app(corpus_index, search_route="async"))
    params = {"q": "search engine", "k": 3}
    sync_body = sync_client.get("/search", params=params).json()
    async_body = async_client.get("/search", params=params).json()
    sync_body.pop("took_ms")
    async_body.pop("took_ms")
    assert sync_body == async_body


def test_openapi_lists_api_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------- lifespan


def _wait_for_health(client: TestClient, expected: int) -> dict[str, object]:
    deadline = time.monotonic() + 10
    while True:
        response = client.get("/health")
        if response.status_code == expected or time.monotonic() > deadline:
            assert response.status_code == expected, response.text
            return response.json()
        time.sleep(0.02)


def test_lifespan_loads_index_once(corpus_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.bin"
    save(corpus_index, path)
    application = create_app(Settings(index_path=path))
    with TestClient(application) as client:
        body = _wait_for_health(client, 200)
        assert body["documents"] == 3
        stats = client.get("/stats").json()
        assert stats["index_bytes"] == path.stat().st_size
        assert client.get("/search", params={"q": "token"}).json()["total"] == 1


def test_lifespan_missing_index_reports_error(tmp_path: Path) -> None:
    application = create_app(Settings(index_path=tmp_path / "missing.bin"))
    with TestClient(application) as client:
        body = _wait_for_health(client, 503)
        deadline = time.monotonic() + 10
        while body["status"] == "loading" and time.monotonic() < deadline:
            body = client.get("/health").json()
        assert body["status"] == "error"
        assert "FileNotFoundError" in str(body["detail"])


# ---------------------------------------------------------------------- UI


def test_home_page_without_query(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Search all 3" in response.text


def test_home_page_highlights_matches(client: TestClient) -> None:
    response = client.get("/", params={"q": "token"})
    assert response.status_code == 200
    assert "<mark>token</mark>" in response.text
    assert 'href="/doc/' in response.text


def test_home_page_escapes_user_input(client: TestClient) -> None:
    response = client.get("/", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_home_page_bad_query_explains(client: TestClient) -> None:
    response = client.get("/", params={"q": "(python"})
    assert response.status_code == 422
    assert "could not be read" in response.text


def test_results_fragment_for_htmx(client: TestClient) -> None:
    response = client.get("/ui/results", params={"q": "python", "k": 1})
    assert response.status_code == 200
    assert "<html" not in response.text
    assert response.headers["HX-Push-Url"].startswith("/?q=python")
    assert 'rel="next"' in response.text


def test_document_page(client: TestClient) -> None:
    response = client.get("/doc/1", params={"q": "token"})
    assert response.status_code == 200
    assert "<mark>token</mark>" in response.text
    assert client.get("/doc/42").status_code == 404


def test_static_css_served(client: TestClient) -> None:
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert "--mark" in response.text


# --------------------------------------------------------------------- CLI


def test_serve_command_wraps_uvicorn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    calls: list[dict[str, object]] = []

    def fake_run(target: str, **kwargs: object) -> None:
        calls.append({"target": target, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setenv("INDEX_PATH", "placeholder.bin")
    index_path = tmp_path / "x.bin"
    result = CliRunner().invoke(
        cli_app,
        ["serve", "--index", str(index_path), "--port", "9000", "--workers", "4"],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "target": "findex.web:app",
            "host": "127.0.0.1",
            "port": 9000,
            "workers": 4,
            "reload": False,
            "access_log": False,
            "proxy_headers": True,
            "forwarded_allow_ips": "*",
            "timeout_graceful_shutdown": 10,
        }
    ]
    import os

    assert os.environ["INDEX_PATH"] == str(index_path)
