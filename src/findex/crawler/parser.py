"""Small HTML parser based on the Python standard library."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin

from .urls import canonicalize_url


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if lowered == "a":
            href = next((value for key, value in attrs if key.lower() == "href"), None)
            if href:
                try:
                    link = canonicalize_url(urljoin(self.base_url, href))
                except ValueError:
                    return
                if link not in self.links:
                    self.links.append(link)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = False
        if lowered in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        compact = " ".join(data.split())
        if not compact:
            return
        if self._in_title:
            self.title_parts.append(compact)
        if not self._skip_depth:
            self.text_parts.append(compact)


def parse_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Extract title, readable text, and canonicalized anchor URLs."""
    parser = _PageParser(base_url)
    parser.feed(html)
    parser.close()
    title = " ".join(parser.title_parts).strip() or base_url
    text = " ".join(parser.text_parts).strip()
    return title, text, parser.links
