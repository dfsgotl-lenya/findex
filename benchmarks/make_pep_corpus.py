"""Turn a checkout of github.com/python/peps into a findex JSONL corpus.

PEPs are placed in the public domain (or CC0) by their authors, so they make a
real, redistributable English corpus for the web demo of Labs 7-8. Each PEP
becomes one JSONL line in the same shape the Lab 6 crawler writes:
``{"url", "title", "text"}``.

Usage::

    git clone --depth 1 https://github.com/python/peps /tmp/peps
    uv run python benchmarks/make_pep_corpus.py /tmp/peps --out data/peps.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_HEADER_RE = re.compile(r"^([A-Za-z-]+):\s*(.*)$")
_UNDERLINE_RE = re.compile(r"^\s*([=\-~^*+#`'\"])\1{2,}\s*$")
_DIRECTIVE_RE = re.compile(r"^\s*\.\. [a-zA-Z:-]+::.*$")
_COMMENT_RE = re.compile(r"^\s*\.\.(\s.*)?$")
_FIELD_RE = re.compile(r"^\s+:[a-zA-Z-]+:.*$")
_ROLE_RE = re.compile(r":[a-zA-Z:]+:`([^`<]*?)(?:\s*<[^>]*>)?`")
_LINK_RE = re.compile(r"`([^`<]+?)\s*<[^>]+>`_{1,2}")
_REF_RE = re.compile(r"`([^`]+)`_{1,2}")
_LITERAL_RE = re.compile(r"``([^`]+)``")
_EMPHASIS_RE = re.compile(r"\*{1,2}([^*\n]+)\*{1,2}")


def _parse_headers(lines: list[str]) -> tuple[dict[str, str], int]:
    headers: dict[str, str] = {}
    last_key = ""
    for number, line in enumerate(lines):
        if not line.strip():
            return headers, number + 1
        match = _HEADER_RE.match(line)
        if match:
            last_key = match.group(1)
            headers[last_key] = match.group(2).strip()
        elif last_key:
            headers[last_key] += " " + line.strip()
    return headers, len(lines)


def clean_rst(body: str) -> str:
    """Drop reST markup that would pollute the vocabulary; keep the prose."""
    kept: list[str] = []
    for line in body.splitlines():
        if _UNDERLINE_RE.match(line):
            continue
        if _DIRECTIVE_RE.match(line) or _FIELD_RE.match(line):
            continue
        if _COMMENT_RE.match(line):
            continue
        text = _ROLE_RE.sub(r"\1", line)
        text = _LINK_RE.sub(r"\1", text)
        text = _REF_RE.sub(r"\1", text)
        text = _LITERAL_RE.sub(r"\1", text)
        text = _EMPHASIS_RE.sub(r"\1", text)
        kept.append(text.rstrip())
    cleaned = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def pep_record(path: Path) -> dict[str, str] | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers, body_start = _parse_headers(lines)
    number = headers.get("PEP")
    title = headers.get("Title")
    if not number or not title or not number.isdigit():
        return None
    pep = int(number)
    body = clean_rst("\n".join(lines[body_start:]))
    status = headers.get("Status", "")
    pep_type = headers.get("Type", "")
    created = headers.get("Created", "")
    meta_line = f"Status: {status}. Type: {pep_type}. Created: {created}."
    return {
        "url": f"https://peps.python.org/pep-{pep:04d}/",
        "title": f"PEP {pep} – {title}",
        # Kept out of "text" on purpose: Lab 8 uses it as an objective
        # relevance label, so it must not be searchable.
        "topic": headers.get("Topic", ""),
        "text": f"{title}\n\n{meta_line}\n\n{body}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkout", type=Path, help="Path to a python/peps clone.")
    parser.add_argument("--out", type=Path, default=Path("data/peps.jsonl"))
    args = parser.parse_args()

    source = args.checkout / "peps"
    files = sorted(source.glob("pep-*.rst"))
    if not files:
        parser.error(f"no pep-*.rst files under {source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            record = pep_record(path)
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    size_mib = args.out.stat().st_size / 1024**2
    print(f"{written} PEPs -> {args.out} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
