"""A tiny real HTTP server on loopback, for tests that must cross a socket.

``serve(handler)`` starts a ``ThreadingHTTPServer`` on 127.0.0.1 with a free
port and yields ``(base_url, requests)``. ``handler(method, path, body)``
returns ``(status, payload)``: a dict or list is sent as JSON, a string as
text/event-stream. Every request is recorded as ``(method, path, headers,
body)`` so a test can assert what reached the "provider".
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Captured at import, before the hermetic fixture patches it for a test.
_REAL_CREATE_CONNECTION = socket.create_connection


@contextlib.contextmanager
def serve(handler):
    requests: list[tuple[str, str, dict, dict | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def _handle(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = None
            requests.append((method, self.path, dict(self.headers), body))
            status, payload = handler(method, self.path, body)
            if isinstance(payload, str):
                data, kind = payload.encode(), "text/event-stream"
            else:
                data, kind = json.dumps(payload).encode(), "application/json"
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - http.server naming
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def log_message(self, *args):  # keep test output quiet
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", requests
    finally:
        httpd.shutdown()
        httpd.server_close()


def allow_loopback(monkeypatch) -> None:
    """Let ``socket.create_connection`` reach 127.0.0.1 only.

    The suite's hermetic fixture blocks ``create_connection`` outright, which
    is what ``urllib`` and sync ``httpx`` use. The fake servers here live on
    loopback, so this re-opens exactly that and nothing else; the credential
    scrubbing of the hermetic fixture stays in force.
    """
    from tests.conftest import BlockedNetwork, _is_loopback

    real = _REAL_CREATE_CONNECTION

    def create_connection(address, *args, **kwargs):
        if not _is_loopback(address):
            raise BlockedNetwork(f"the test suite may not reach {address!r}")
        return real(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", create_connection)
