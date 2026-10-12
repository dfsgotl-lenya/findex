"""Global/per-host request limits and a per-host minimum request interval."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import monotonic
from urllib.parse import urlsplit


class RequestLimiter:
    """Apply global and per-origin concurrency and polite start intervals."""

    def __init__(
        self,
        concurrency: int,
        per_host: int,
        per_host_delay: float,
    ) -> None:
        if concurrency < 1 or per_host < 1:
            raise ValueError("concurrency and per_host must be at least 1")
        if per_host_delay < 0:
            raise ValueError("per_host_delay must be >= 0")
        self.global_semaphore = asyncio.Semaphore(concurrency)
        self.per_host = per_host
        self.per_host_delay = per_host_delay
        self._host_semaphores: dict[str, asyncio.Semaphore] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_started: dict[str, float] = {}

    def _key(self, url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}"

    @asynccontextmanager
    async def request(self, url: str) -> AsyncGenerator[None, None]:
        """Acquire the host limit, respect delay, then acquire global capacity."""
        key = self._key(url)
        host_semaphore = self._host_semaphores.setdefault(
            key, asyncio.Semaphore(self.per_host)
        )
        host_lock = self._host_locks.setdefault(key, asyncio.Lock())

        async with host_semaphore:
            async with self.global_semaphore:
                async with host_lock:
                    now = monotonic()
                    last = self._last_started.get(key)
                    if last is not None:
                        remaining = self.per_host_delay - (now - last)
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                    self._last_started[key] = monotonic()
                yield
