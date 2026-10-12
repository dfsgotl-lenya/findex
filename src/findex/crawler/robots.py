"""Per-origin robots.txt cache using :mod:`urllib.robotparser`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .fetch import RequestGuard, fetch

logger = logging.getLogger("findex.crawler")


class RobotsCache:
    """Fetch robots.txt once per origin and fail closed on ambiguous failures.

    A missing robots.txt (404/410) allows crawling. Successful 200 responses
    are parsed with ``RobotFileParser``. Other HTTP failures and network errors
    deny crawling for that origin so a temporary robots outage cannot silently
    turn into permission to crawl.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        *,
        timeout: float = 10.0,
        request_guard: RequestGuard | None = None,
        on_request_start: Callable[[str], None] | None = None,
        on_request_end: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.user_agent = user_agent
        self.product_token = user_agent.split(" ", maxsplit=1)[0]
        self.timeout = timeout
        self.request_guard = request_guard
        self.on_request_start = on_request_start
        self.on_request_end = on_request_end
        self._cache: dict[str, RobotFileParser | bool] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _origin(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), "", "", ""))

    async def _load(self, url: str) -> RobotFileParser | bool:
        origin = self._origin(url)
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin in self._cache:
                return self._cache[origin]
            robots_url = f"{origin}/robots.txt"
            host = urlsplit(origin).hostname or ""
            result = await fetch(
                self.client,
                robots_url,
                timeout=self.timeout,
                max_attempts=3,
                allowed_domains=frozenset({host}),
                request_guard=self.request_guard,
                on_request_start=self.on_request_start,
                on_request_end=self.on_request_end,
            )
            if result.status in {404, 410}:
                policy: RobotFileParser | bool = True
            elif result.status == 200 and result.error is None:
                parser = RobotFileParser(robots_url)
                parser.parse(result.body.decode("utf-8", errors="replace").splitlines())
                policy = parser
            else:
                policy = False
                logger.warning(
                    "url=%s status=robots_unavailable action=deny-crawl error=%s",
                    robots_url,
                    result.error or result.status,
                )
            self._cache[origin] = policy
            return policy

    async def allowed(self, url: str) -> bool:
        """Return whether the configured crawler identity may fetch ``url``."""
        policy = await self._load(url)
        if policy is True:
            return True
        if policy is False:
            return False
        return policy.can_fetch(self.product_token, url)
