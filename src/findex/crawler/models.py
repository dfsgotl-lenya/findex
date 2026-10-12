"""Data models shared by the asynchronous crawler."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of one URL fetch, including failed attempts and response bytes."""

    url: str
    status: int | str
    bytes_received: int
    elapsed: float
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    error: str | None = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class Page:
    """Parsed HTML page that can be streamed to JSON Lines."""

    url: str
    title: str
    text: str
    fetched_at: str
    status: int
    bytes_received: int
    elapsed: float

    def as_json_dict(self) -> dict[str, str | int | float]:
        """Return the stable JSONL schema used by ``findex index``."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "fetched_at": self.fetched_at,
            "status": self.status,
            "bytes": self.bytes_received,
            "elapsed": round(self.elapsed, 6),
        }


@dataclass(slots=True)
class CrawlStats:
    """Mutable counters updated by workers; consumed by the Rich display."""

    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    requests: int = 0
    started_at: float = field(default_factory=perf_counter)

    @property
    def elapsed(self) -> float:
        return max(perf_counter() - self.started_at, 1e-9)

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.elapsed
