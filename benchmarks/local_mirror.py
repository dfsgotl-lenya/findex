"""Serve an authorized local HTML mirror for repeatable crawler experiments."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class MirrorHandler(BaseHTTPRequestHandler):
    server_version = "findex-local-mirror/1.0"
    request_counts: Counter[str] = Counter()
    pages = 250
    response_delay = 0.01

    def log_message(self, format_string: str, *args: object) -> None:
        # Keep the demo mirror quiet; the crawler writes its own crawl.log.
        return

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        type(self).request_counts[path] += 1
        time.sleep(type(self).response_delay)

        if path == "/robots.txt":
            self._respond(200, "User-agent: *\nAllow: /\n", "text/plain; charset=utf-8")
            return
        if path == "/health":
            self._respond(200, "ok\n", "text/plain; charset=utf-8")
            return
        if path == "/flaky429":
            if type(self).request_counts[path] == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._respond(
                200,
                "<html><title>Recovered after 429</title><body>retry works "
                '<a href="/page/0">start crawl</a></body></html>',
            )
            return
        if path in {"/redirect-loop-a", "/redirect-loop-b"}:
            target = "/redirect-loop-b" if path.endswith("a") else "/redirect-loop-a"
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path.startswith("/page/"):
            try:
                page_id = int(path.rsplit("/", maxsplit=1)[1])
            except ValueError:
                self._respond(404, "not found", "text/plain; charset=utf-8")
                return
            if not 0 <= page_id < type(self).pages:
                self._respond(404, "not found", "text/plain; charset=utf-8")
                return

            links: list[str] = []
            for next_id in range(page_id + 1, min(type(self).pages, page_id + 21)):
                # Check query order, slash, and fragment canonicalization.
                links.append(
                    f'<a href="/page/{next_id}/?z=2&amp;a=1#part">page {next_id}</a>'
                )
                links.append(
                    f'<a href="/page/{next_id}?a=1&amp;z=2">duplicate {next_id}</a>'
                )
            body = (
                f"<html><head><title>Mirror page {page_id}</title></head><body>"
                f"local mirror document {page_id} describes asyncio queues, HTTP "
                f"requests, "
                f"robots rules, retries, event loops and search indexing. "
                f"{' '.join(links)}</body></html>"
            )
            self._respond(200, body)
            return
        self._respond(404, "not found", "text/plain; charset=utf-8")

    def _respond(
        self,
        status: int,
        body: str,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback interface only.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pages", type=int, default=250)
    parser.add_argument(
        "--delay", type=float, default=0.01, help="Local response delay in seconds."
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the lab mirror must bind to loopback only")
    if args.pages < 201:
        parser.error("--pages must be at least 201 for the Lab 6 benchmark")
    if args.delay < 0:
        parser.error("--delay must be >= 0")

    MirrorHandler.pages = args.pages
    MirrorHandler.response_delay = args.delay
    MirrorHandler.request_counts.clear()
    server = ThreadingHTTPServer((args.host, args.port), MirrorHandler)
    print(f"Local mirror ready at http://{args.host}:{args.port}/page/0")
    print(f"Pages: {args.pages}; response delay: {args.delay:.3f}s")
    print("robots.txt allows this local mirror; Ctrl+C stops the server.")
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        print("Stopping local mirror...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
