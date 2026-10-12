from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import JsonlDocumentRef, iter_documents
from findex.crawler import Page, canonicalize_url, crawl, fetch
from findex.crawler.limiter import RequestLimiter
from findex.crawler.robots import RobotsCache
from findex.parallel import build_parallel_index, collect_document_paths

USER_AGENT = "findex-lab6-test/1.0 (contact=mailto:test@example.org)"


def test_canonicalize_url_normalizes_scheme_host_slash_query_and_fragment() -> None:
    assert canonicalize_url("HTTP://Example.COM:80/a/?z=2&a=1#section") == (
        "http://example.com/a?a=1&z=2"
    )
    assert canonicalize_url("https://EXAMPLE.com/#part") == "https://example.com/"


def test_canonicalize_url_rejects_non_http_and_credentials() -> None:
    with pytest.raises(ValueError):
        canonicalize_url("ftp://example.org/file")
    with pytest.raises(ValueError):
        canonicalize_url("https://user:pass@example.org/")


def test_fetch_retries_429_honors_retry_after_and_records_attempts() -> None:
    count = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"}, request=request)
        return httpx.Response(200, content=b"ok", request=request)

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.org/",
                max_attempts=3,
                sleeper=fake_sleep,
                random_value=lambda: 0.0,
            )
        assert result.status == 200
        assert result.body == b"ok"
        assert result.attempts == 2

    asyncio.run(run())
    assert count == 2
    assert sleeps == [0.25]


def test_fetch_retries_5xx_and_honors_retry_after() -> None:
    count = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(503, headers={"Retry-After": "0.4"}, request=request)
        return httpx.Response(200, content=b"available", request=request)

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.org/temporary",
                max_attempts=3,
                sleeper=fake_sleep,
                random_value=lambda: 0.0,
            )
        assert result.status == 200
        assert result.attempts == 2

    asyncio.run(run())
    assert count == 2
    assert sleeps == [0.4]


def test_fetch_retries_timeout_but_stops_after_three_attempts() -> None:
    count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        await asyncio.sleep(0.05)
        return httpx.Response(200, content=b"late", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.org/slow",
                timeout=0.005,
                max_attempts=3,
                sleeper=_no_wait,
                random_value=lambda: 0.0,
            )
        assert result.status == "timeout"
        assert result.attempts == 3

    asyncio.run(run())
    assert count == 3


def test_fetch_does_not_retry_non_timeout_network_error() -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        raise httpx.ConnectError("connection refused", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.org/offline",
                max_attempts=3,
                sleeper=_no_wait,
            )
        assert result.status == "network_error"
        assert result.attempts == 1

    asyncio.run(run())
    assert count == 1


def test_fetch_does_not_retry_regular_4xx() -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(404, content=b"missing", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.org/missing",
                max_attempts=3,
                sleeper=_no_wait,
            )
        assert result.status == 404
        assert result.error == "http_404"
        assert result.attempts == 1

    asyncio.run(run())
    assert count == 1


async def _no_wait(_seconds: float) -> None:
    return None


def test_robots_cache_is_per_origin_and_disallows_private_paths() -> None:
    counts: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counts[path] += 1
        if path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private\nAllow: /",
                request=request,
            )
        return httpx.Response(200, text="ok", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT}
        ) as client:
            robots = RobotsCache(client, USER_AGENT)
            assert await robots.allowed("https://example.org/public")
            assert not await robots.allowed("https://example.org/private/secret")
            assert await robots.allowed("https://example.org/another")

    asyncio.run(run())
    assert counts["/robots.txt"] == 1


def test_crawl_uses_frontier_without_duplicate_canonical_urls() -> None:
    requested: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        requested[
            request.url.path
            + (f"?{request.url.query.decode()}" if request.url.query else "")
        ] += 1
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200, text="User-agent: *\nAllow: /\n", request=request
            )
        pages = {
            "/page/0": (
                "<title>Zero</title><body>zero page "
                '<a href="/page/1/?z=2&amp;a=1#top">one</a>'
                '<a href="/page/1?a=1&amp;z=2">duplicate</a>'
                '<a href="https://outside.example/page">outside</a></body>'
            ),
            "/page/1": (
                "<title>One</title><body>one page "
                '<a href="/page/2#fragment">two</a></body>'
            ),
            "/page/2": "<title>Two</title><body>two page</body>",
        }
        return httpx.Response(
            200, text=pages.get(request.url.path, "<body>other</body>"), request=request
        )

    async def run() -> list[Page]:
        found: list[Page] = []
        async for page in crawl(
            ["https://example.org/page/0#initial"],
            max_pages=3,
            concurrency=3,
            per_host=3,
            per_host_delay=0,
            timeout=1,
            allowed_domains=["example.org"],
            user_agent=USER_AGENT,
            transport=httpx.MockTransport(handler),
        ):
            found.append(page)
        return found

    pages = asyncio.run(run())
    assert [page.title for page in pages] == ["Zero", "One", "Two"]
    assert requested["/robots.txt"] == 1
    assert requested["/page/1?a=1&z=2"] == 1
    assert not any("outside.example" in path for path in requested)


def test_request_limiter_caps_per_host_concurrency_and_spacing() -> None:
    active = 0
    peak_active = 0
    starts: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        starts.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(200, content=b"ok", request=request)

    async def run() -> None:
        limiter = RequestLimiter(concurrency=4, per_host=2, per_host_delay=0.01)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.gather(
                *[
                    fetch(
                        client,
                        f"https://example.org/page/{index}",
                        request_guard=limiter.request,
                        max_attempts=1,
                    )
                    for index in range(6)
                ]
            )

    asyncio.run(run())
    assert peak_active <= 2
    assert len(starts) == 6
    assert all(
        second - first >= 0.008
        for first, second in zip(starts, starts[1:], strict=False)
    )


def test_crawl_honors_robots_and_page_cap() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /blocked\nAllow: /\n",
                request=request,
            )
        if request.url.path == "/start":
            return httpx.Response(
                200,
                text=(
                    "<title>Start</title>"
                    '<a href="/blocked">blocked</a>'
                    '<a href="/ok">ok</a>'
                ),
                request=request,
            )
        return httpx.Response(200, text="<title>OK</title>content", request=request)

    async def run() -> list[Page]:
        pages: list[Page] = []
        async for page in crawl(
            ["https://example.org/start"],
            max_pages=3,
            concurrency=2,
            per_host=2,
            per_host_delay=0,
            allowed_domains=["example.org"],
            user_agent=USER_AGENT,
            transport=httpx.MockTransport(handler),
        ):
            pages.append(page)
        return pages

    pages = asyncio.run(run())
    assert len(pages) <= 3
    assert "/blocked" not in requested


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_crawl_jsonl_documents_can_be_parallel_indexed(
    tmp_path: Path, executor: str
) -> None:
    jsonl = tmp_path / "crawl.jsonl"
    jsonl.write_text(
        '{"url":"https://example.org/one","title":"First page",'
        '"text":"python asyncio"}\n'
        '{"url":"https://example.org/two","title":"Second page",'
        '"text":"python crawler"}\n'
        '{"url":"https://example.org/three","title":"Third page",'
        '"text":"search engine"}\n',
        encoding="utf-8",
    )

    documents = list(iter_documents(jsonl))
    refs = collect_document_paths(jsonl)
    assert len(documents) == 3
    assert isinstance(refs[0], JsonlDocumentRef)
    assert documents[0].doc_id == "https://example.org/one"
    assert documents[0].title == "First page"

    built = build_parallel_index(jsonl, workers=2, executor=executor)  # type: ignore[arg-type]
    assert built.index.num_docs == 3
    assert built.index.df("python") == 2
    assert "::jsonl-offset:" in built.index.doc_meta[0].path
    assert built.index.doc_meta[0].title == "First page"
    from findex.snippets import make_snippet

    snippet = make_snippet(built.index.doc_meta[0].path, ("python",))
    assert "**python**" in snippet


def test_iter_documents_keeps_plain_txt_compatibility(tmp_path: Path) -> None:
    text_path = tmp_path / "plain.txt"
    text_path.write_text("Hello crawler", encoding="utf-8")
    docs = list(iter_documents(text_path))
    assert len(docs) == 1
    assert docs[0].text == "Hello crawler"
    assert docs[0].title is None


def test_fetch_stops_redirect_loop_without_retries() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/a":
            return httpx.Response(302, headers={"Location": "/b"}, request=request)
        return httpx.Response(302, headers={"Location": "/a"}, request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.org/a",
                allowed_domains=frozenset({"example.org"}),
                max_attempts=3,
                sleeper=_no_wait,
            )
        assert result.status == "redirect_loop"
        assert result.error == "redirect loop detected"
        assert result.attempts == 1

    asyncio.run(run())
    assert requested == ["/a", "/b"]


def test_cli_exposes_crawl_command_and_jsonl_index_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "crawl" in result.stdout
    index_help = runner.invoke(app, ["index", "--help"])
    assert index_help.exit_code == 0
    assert ".jsonl" in index_help.stdout
