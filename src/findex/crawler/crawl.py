"""Queue-based asynchronous crawl with bounded workers and streaming results."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlsplit

import httpx

from .fetch import fetch
from .limiter import RequestLimiter
from .models import CrawlStats, Page
from .parser import parse_html
from .robots import RobotsCache
from .urls import canonicalize_url, host_allowed, normalize_domains

logger = logging.getLogger("findex.crawler")
_STOP: Final[None] = None
ProgressCallback = Callable[[CrawlStats], None]


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.2,
    timeout: float = 10.0,
    allowed_domains: Iterable[str] | None = None,
    user_agent: str,
    transport: httpx.AsyncBaseTransport | None = None,
    on_progress: ProgressCallback | None = None,
    retry_attempts: int = 3,
) -> AsyncIterator[Page]:
    """Yield pages as they are downloaded, parsed, and accepted.

    The frontier is a queue consumed by ``TaskGroup`` workers. The total number
    of scheduled URLs is capped at ``max_pages`` (so failures cannot cause an
    unbounded frontier); the output queue is bounded to apply backpressure and
    keep page contents from accumulating in memory. Closing the generator
    cancels workers and closes the shared HTTP client.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if concurrency < 1 or per_host < 1:
        raise ValueError("concurrency and per_host must be >= 1")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    if not user_agent.strip():
        raise ValueError("user_agent must not be empty")

    seed_urls: list[str] = []
    for seed in seeds:
        canonical = canonicalize_url(seed)
        if canonical not in seed_urls:
            seed_urls.append(canonical)
        if len(seed_urls) >= max_pages:
            break
    if not seed_urls:
        raise ValueError("at least one seed URL is required")

    if allowed_domains is None:
        domains = normalize_domains(urlsplit(url).hostname or "" for url in seed_urls)
    else:
        domains = normalize_domains(allowed_domains)
    seed_urls = [url for url in seed_urls if host_allowed(url, domains)]
    if not seed_urls:
        raise ValueError("no seed URL matches the allowed domain list")
    seed_urls = seed_urls[:max_pages]

    stats = CrawlStats()
    frontier: asyncio.Queue[str] = asyncio.Queue()
    output: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=max(1, concurrency))
    seen = set(seed_urls)
    pending = len(seed_urls)
    finished = False
    state_lock = asyncio.Lock()
    for seed in seed_urls:
        frontier.put_nowait(seed)

    def notify() -> None:
        stats.queue_depth = frontier.qsize()
        if on_progress is not None:
            on_progress(stats)

    def request_started(_url: str) -> None:
        stats.requests += 1
        stats.in_flight += 1
        notify()

    def request_ended(_url: str) -> None:
        stats.in_flight = max(0, stats.in_flight - 1)
        notify()

    def request_failed(_url: str) -> None:
        stats.errors += 1
        notify()

    limiter = RequestLimiter(concurrency, per_host, per_host_delay)
    limits = httpx.Limits(
        max_connections=max(concurrency, per_host),
        max_keepalive_connections=max(concurrency, per_host),
    )

    async with httpx.AsyncClient(
        transport=transport,
        headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
        timeout=None,
        follow_redirects=False,
        max_redirects=10,
        limits=limits,
    ) as client:
        robots = RobotsCache(
            client,
            user_agent,
            timeout=timeout,
            request_guard=limiter.request,
            on_request_start=request_started,
            on_request_end=request_ended,
        )

        async def worker() -> None:
            nonlocal pending, finished
            while True:
                url = await frontier.get()
                links_to_schedule: list[str] = []
                try:
                    if not await robots.allowed(url):
                        stats.errors += 1
                        logger.info(
                            "url=%s status=robots_denied bytes=0 elapsed=0.000s",
                            url,
                        )
                    else:
                        result = await fetch(
                            client,
                            url,
                            timeout=timeout,
                            max_attempts=retry_attempts,
                            allowed_domains=domains,
                            allowed_url=robots.allowed,
                            request_guard=limiter.request,
                            on_request_start=request_started,
                            on_request_end=request_ended,
                            on_error=request_failed,
                        )
                        if (
                            isinstance(result.status, int)
                            and 200 <= result.status < 300
                            and result.error is None
                        ):
                            html = result.body.decode("utf-8", errors="replace")
                            title, text, extracted_links = await asyncio.to_thread(
                                parse_html, html, result.url
                            )
                            canonical_links: list[str] = []
                            for link in extracted_links:
                                try:
                                    canonical = canonicalize_url(link)
                                except ValueError:
                                    continue
                                if host_allowed(canonical, domains):
                                    canonical_links.append(canonical)
                            page = Page(
                                url=result.url,
                                title=title,
                                text=text,
                                fetched_at=datetime.now(UTC).isoformat(),
                                status=result.status,
                                bytes_received=result.bytes_received,
                                elapsed=result.elapsed,
                            )
                            stats.pages += 1
                            notify()
                            links_to_schedule = canonical_links
                            await output.put(page)
                        elif result.error is None:
                            request_failed(url)
                            logger.warning(
                                "url=%s status=%s action=skip-page "
                                "error=unexpected-status",
                                url,
                                result.status,
                            )
                        else:
                            logger.warning(
                                "url=%s status=%s action=skip-page error=%s",
                                url,
                                result.status,
                                result.error,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # one bad page must not kill sibling workers
                    request_failed(url)
                    logger.exception("url=%s status=worker_error error=%s", url, exc)
                finally:
                    frontier.task_done()
                    async with state_lock:
                        for link in links_to_schedule:
                            if len(seen) >= max_pages:
                                break
                            if link not in seen and host_allowed(link, domains):
                                seen.add(link)
                                pending += 1
                                frontier.put_nowait(link)
                        pending -= 1
                        notify()
                        if pending == 0 and not finished:
                            finished = True
                            await output.put(_STOP)

        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(worker(), name=f"findex-crawl-{i}")
                for i in range(concurrency)
            ]
            notify()
            try:
                while True:
                    item = await output.get()
                    if item is None:
                        break
                    yield item
            finally:
                for task in tasks:
                    task.cancel()
