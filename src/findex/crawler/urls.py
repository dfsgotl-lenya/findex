"""URL validation, canonicalization, and domain allowlist helpers."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    """Return a stable HTTP(S) URL without fragments or incidental variants.

    The scheme and host are lowercased, trailing path slashes are removed
    (except for the root), and query pairs are sorted without discarding
    repeated keys or blank values.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parts.scheme!r}")
    if not parts.hostname:
        raise ValueError(f"URL has no hostname: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs containing credentials are not supported")

    hostname = parts.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = parts.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query_items = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_items, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_domains(domains: Iterable[str]) -> frozenset[str]:
    """Normalize host/domain names supplied by the CLI or caller."""
    normalized: set[str] = set()
    for domain in domains:
        value = domain.strip().lower().rstrip(".")
        if value.startswith("http://") or value.startswith("https://"):
            value = urlsplit(value).hostname or ""
        if value:
            normalized.add(value.strip("[]"))
    return frozenset(normalized)


def host_allowed(url: str, allowed_domains: Iterable[str]) -> bool:
    """Check exact hosts and subdomains against an explicit allowlist."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    domains = normalize_domains(allowed_domains)
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)
