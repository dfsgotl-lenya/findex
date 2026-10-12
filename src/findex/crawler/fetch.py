"""Polite HTTP fetching with timeout, bounded retries, and structured logs."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import perf_counter
from urllib.parse import urljoin

import httpx

from .models import FetchResult
from .urls import canonicalize_url, host_allowed

logger = logging.getLogger("findex.crawler")
RequestGuard = Callable[[str], AbstractAsyncContextManager[None]]
AllowedURL = Callable[[str], Awaitable[bool]]
CounterCallback = Callable[[str], None]


@asynccontextmanager
async def _no_guard(_url: str) -> AsyncGenerator[None, None]:
    yield


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After as delta seconds or an HTTP-date."""
    if not value:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _domain_is_allowed(url: str, domains: frozenset[str] | None) -> bool:
    if domains is None:
        return True
    return host_allowed(url, domains)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 8.0,
    allowed_domains: frozenset[str] | None = None,
    allowed_url: AllowedURL | None = None,
    request_guard: RequestGuard | None = None,
    on_request_start: CounterCallback | None = None,
    on_request_end: CounterCallback | None = None,
    on_error: CounterCallback | None = None,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_value: Callable[[], float] = random.random,
    max_redirects: int = 10,
) -> FetchResult:
    """Fetch one URL with bounded retry and safe manual redirect handling.

    Only 429, 5xx, and timeouts are retried. A valid ``Retry-After`` header
    takes precedence over exponential backoff. Redirects
    are followed manually so an off-allowlist redirect is never requested.
    ``max_attempts`` includes the initial attempt (default: three total tries).
    """
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    if max_attempts < 1 or max_attempts > 3:
        raise ValueError("max_attempts must be between 1 and 3")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("retry delays must be >= 0")

    requested_url = canonicalize_url(url)
    started = perf_counter()
    bytes_received = 0
    last_status: int | str = "not_started"
    last_error: str | None = None
    last_body = b""
    last_headers: dict[str, str] = {}
    current_url = requested_url
    attempt = 0
    guard_factory = request_guard or _no_guard

    for attempt in range(1, max_attempts + 1):
        current_url = requested_url
        redirect_seen: set[str] = set()
        retry_after: float | None = None
        retryable = False
        try:
            async with asyncio.timeout(timeout):
                for redirect_count in range(max_redirects + 1):
                    current_url = canonicalize_url(current_url)
                    if not _domain_is_allowed(current_url, allowed_domains):
                        last_status = "redirect_blocked"
                        last_error = "redirect target is outside the allowed domains"
                        logger.warning(
                            "url=%s status=%s bytes=0 elapsed=0.000s "
                            "attempt=%d error=%s",
                            current_url,
                            last_status,
                            attempt,
                            last_error,
                        )
                        if on_error is not None:
                            on_error(current_url)
                        return FetchResult(
                            url=current_url,
                            status=last_status,
                            bytes_received=bytes_received,
                            elapsed=perf_counter() - started,
                            body=last_body,
                            headers=last_headers,
                            error=last_error,
                            attempts=attempt,
                        )
                    if current_url in redirect_seen:
                        last_status = "redirect_loop"
                        last_error = "redirect loop detected"
                        logger.warning(
                            "url=%s status=%s bytes=0 elapsed=0.000s "
                            "attempt=%d error=%s",
                            current_url,
                            last_status,
                            attempt,
                            last_error,
                        )
                        if on_error is not None:
                            on_error(current_url)
                        return FetchResult(
                            url=current_url,
                            status=last_status,
                            bytes_received=bytes_received,
                            elapsed=perf_counter() - started,
                            body=last_body,
                            headers=last_headers,
                            error=last_error,
                            attempts=attempt,
                        )
                    redirect_seen.add(current_url)
                    if allowed_url is not None and not await allowed_url(current_url):
                        last_status = "robots_denied"
                        last_error = "URL disallowed by robots.txt"
                        logger.info(
                            "url=%s status=%s bytes=0 elapsed=0.000s attempt=%d",
                            current_url,
                            last_status,
                            attempt,
                        )
                        if on_error is not None:
                            on_error(current_url)
                        return FetchResult(
                            url=current_url,
                            status=last_status,
                            bytes_received=bytes_received,
                            elapsed=perf_counter() - started,
                            body=b"",
                            headers={},
                            error=last_error,
                            attempts=attempt,
                        )

                    request_started = perf_counter()
                    async with guard_factory(current_url):
                        if on_request_start is not None:
                            on_request_start(current_url)
                        try:
                            response = await client.get(
                                current_url,
                                follow_redirects=False,
                            )
                        finally:
                            if on_request_end is not None:
                                on_request_end(current_url)
                    request_elapsed = perf_counter() - request_started
                    response_bytes = len(response.content)
                    bytes_received += response_bytes
                    last_status = response.status_code
                    last_body = response.content
                    last_headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    logger.info(
                        "url=%s status=%s bytes=%d elapsed=%.3fs attempt=%d",
                        current_url,
                        response.status_code,
                        response_bytes,
                        request_elapsed,
                        attempt,
                    )

                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            last_error = "redirect response has no Location header"
                            return FetchResult(
                                url=current_url,
                                status=response.status_code,
                                bytes_received=bytes_received,
                                elapsed=perf_counter() - started,
                                body=response.content,
                                headers=last_headers,
                                error=last_error,
                                attempts=attempt,
                            )
                        if redirect_count >= max_redirects:
                            last_status = "redirect_loop"
                            last_error = f"more than {max_redirects} redirects"
                            logger.warning(
                                "url=%s status=%s bytes=0 elapsed=0.000s "
                                "attempt=%d error=%s",
                                current_url,
                                last_status,
                                attempt,
                                last_error,
                            )
                            return FetchResult(
                                url=current_url,
                                status=last_status,
                                bytes_received=bytes_received,
                                elapsed=perf_counter() - started,
                                body=b"",
                                headers=last_headers,
                                error=last_error,
                                attempts=attempt,
                            )
                        next_url = canonicalize_url(urljoin(current_url, location))
                        if next_url in redirect_seen:
                            last_status = "redirect_loop"
                            last_error = "redirect loop detected"
                            logger.warning(
                                "url=%s status=%s bytes=0 elapsed=0.000s "
                                "attempt=%d error=%s",
                                next_url,
                                last_status,
                                attempt,
                                last_error,
                            )
                            return FetchResult(
                                url=current_url,
                                status=last_status,
                                bytes_received=bytes_received,
                                elapsed=perf_counter() - started,
                                body=b"",
                                headers=last_headers,
                                error=last_error,
                                attempts=attempt,
                            )
                        if not _domain_is_allowed(next_url, allowed_domains):
                            last_status = "redirect_blocked"
                            last_error = (
                                "redirect target is outside the allowed domains"
                            )
                            logger.warning(
                                "url=%s status=%s bytes=0 elapsed=0.000s "
                                "attempt=%d error=%s",
                                next_url,
                                last_status,
                                attempt,
                                last_error,
                            )
                            return FetchResult(
                                url=current_url,
                                status=last_status,
                                bytes_received=bytes_received,
                                elapsed=perf_counter() - started,
                                body=b"",
                                headers=last_headers,
                                error=last_error,
                                attempts=attempt,
                            )
                        current_url = next_url
                        continue

                    if response.status_code == 429 or response.status_code >= 500:
                        if on_error is not None:
                            on_error(current_url)
                        retryable = True
                        retry_after = _retry_after_seconds(
                            response.headers.get("Retry-After")
                        )
                        last_error = f"http_{response.status_code}"
                        break
                    if response.status_code >= 400:
                        last_error = f"http_{response.status_code}"
                        if on_error is not None:
                            on_error(current_url)
                    else:
                        last_error = None
                    return FetchResult(
                        url=current_url,
                        status=response.status_code,
                        bytes_received=bytes_received,
                        elapsed=perf_counter() - started,
                        body=response.content,
                        headers=last_headers,
                        error=last_error,
                        attempts=attempt,
                    )
        except (TimeoutError, httpx.TimeoutException) as exc:
            last_status = "timeout"
            last_error = type(exc).__name__
            retryable = True
            if on_error is not None:
                on_error(current_url)
            logger.warning(
                "url=%s status=timeout bytes=0 elapsed=%.3fs attempt=%d error=%s",
                current_url,
                perf_counter() - started,
                attempt,
                last_error,
            )
        except httpx.RequestError as exc:
            # The lab contract retries only HTTP 429, 5xx, and timeouts.
            last_status = "network_error"
            last_error = type(exc).__name__
            retryable = False
            if on_error is not None:
                on_error(current_url)
            logger.warning(
                "url=%s status=network_error action=stop bytes=0 "
                "elapsed=%.3fs attempt=%d error=%s",
                current_url,
                perf_counter() - started,
                attempt,
                last_error,
            )

        if not retryable or attempt >= max_attempts:
            break
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = min(max_delay, delay + random_value() * delay * 0.25)
        logger.warning(
            "url=%s status=%s action=retry delay=%.3fs attempt=%d/%d",
            requested_url,
            last_status,
            delay,
            attempt,
            max_attempts,
        )
        await sleeper(delay)

    return FetchResult(
        url=current_url,
        status=last_status,
        bytes_received=bytes_received,
        elapsed=perf_counter() - started,
        body=last_body,
        headers=last_headers,
        error=last_error or str(last_status),
        attempts=min(max_attempts, attempt),
    )
