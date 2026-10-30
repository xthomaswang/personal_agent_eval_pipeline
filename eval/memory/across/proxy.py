"""A timing proxy between localmelo and the model server (stdlib only).

localmelo's chat and embedding URLs point at this proxy, which forwards each request to the real server and
streams the response back unchanged, byte for byte. For every upstream request it appends one JSON line to a
log: the path, the model and stream flag read from the request body, whether the request offers tools (the
agent's chat call does, a recall selection does not), and four clock readings (seconds since the epoch, the
same clock as the gateway's events and the driver's turn windows):

    t_start        the request is read from the client
    t_headers      the upstream answered with its status line and headers
    t_first_byte   the first byte of the response body (for a stream: the first event; for a reply that is
                   not a stream, which is sent whole, about the end of it)
    t_end          the last byte went back to the client

plus status, bytes (response body), req_bytes and error. A request that cannot reach the server is answered
502 and logged. The proxy adds a few milliseconds to every request, the same for all arms.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host"}
CHUNK = 65536
JSON_BODY_LIMIT = 8 * 1024 * 1024


def describe(path: str, body: bytes) -> dict[str, Any]:
    """What the log says about a request, from its path and JSON body (None where it does not apply)."""
    info: dict[str, Any] = {"model": None, "stream": None, "tools": None, "n_input": None}
    if not body or len(body) > JSON_BODY_LIMIT:
        return info
    try:
        doc = json.loads(body)
    except ValueError:
        return info
    if not isinstance(doc, dict):
        return info
    info["model"] = doc.get("model") if isinstance(doc.get("model"), str) else None
    if path.endswith("/chat/completions"):
        info["stream"] = bool(doc.get("stream"))
        info["tools"] = bool(doc.get("tools"))
    elif path.endswith("/embeddings"):
        value = doc.get("input")
        info["n_input"] = len(value) if isinstance(value, list) else 1
    return info


class TimingProxy:
    """Forward to ``upstream`` (``http://host:port[/prefix]``) and log every request to ``log_path``."""

    def __init__(self, upstream: str, log_path: Path, *, upstream_timeout: float = 1800.0) -> None:
        url = urllib.parse.urlsplit(upstream)
        if url.scheme != "http" or not url.hostname:
            raise ValueError(f"the proxy forwards to an http URL, got {upstream!r}")
        self.host, self.port, self.prefix = url.hostname, url.port or 80, url.path.rstrip("/")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.upstream_timeout = upstream_timeout
        self._log_lock = threading.Lock()
        self._active = 0  # requests being forwarded
        self._idle = threading.Condition()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                proxy.forward(self)

            do_POST = do_PUT = do_DELETE = do_PATCH = do_GET

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.listen_port = int(self.server.server_address[1])
        #: what localmelo is given in place of the model server's URL; the path prefix is kept so requests match
        self.url = f"http://127.0.0.1:{self.listen_port}{self.prefix}"
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    def stop(self, drain: float = 2.0) -> None:
        """Stop listening; requests already under way get up to ``drain`` seconds to finish and be logged."""
        self.server.shutdown()
        with self._idle:
            self._idle.wait_for(lambda: self._active == 0, timeout=drain)
        self.server.server_close()

    def log(self, row: dict[str, Any]) -> None:
        with self._log_lock, self.log_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def read_body(h: BaseHTTPRequestHandler) -> bytes:
        if "chunked" in (h.headers.get("Transfer-Encoding") or "").lower():
            parts = []
            while True:
                size = int((h.rfile.readline().split(b";")[0].strip() or b"0"), 16)
                if size == 0:
                    h.rfile.readline()
                    return b"".join(parts)
                parts.append(h.rfile.read(size))
                h.rfile.readline()
        return h.rfile.read(int(h.headers.get("Content-Length") or 0))

    def forward(self, h: BaseHTTPRequestHandler) -> None:
        t_start = time.time()
        with self._idle:
            self._active += 1
        try:
            self._forward(h, t_start)
        finally:
            with self._idle:
                self._active -= 1
                self._idle.notify_all()

    def _forward(self, h: BaseHTTPRequestHandler, t_start: float) -> None:
        body = self.read_body(h)
        row: dict[str, Any] = {"method": h.command, "path": h.path, **describe(h.path, body), "req_bytes": len(body),
                               "t_start": t_start, "t_headers": None, "t_first_byte": None, "t_end": None, "status": None, "bytes": 0, "error": ""}
        headers = {k: v for k, v in h.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "content-length"}
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.upstream_timeout)
        try:
            try:
                conn.request(h.command, h.path, body=body or None, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                row.update(status=502, error=f"{type(exc).__name__}: {exc}", t_end=time.time())
                raw = json.dumps({"error": {"message": "the model server did not answer", "detail": row["error"]}}).encode()
                h.send_response(502)
                h.send_header("Content-Type", "application/json")
                h.send_header("Content-Length", str(len(raw)))
                h.end_headers()
                h.wfile.write(raw)
                return
            row.update(status=resp.status, t_headers=time.time())
            length = resp.getheader("Content-Length")
            h.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                if key.lower() not in HOP_BY_HOP and key.lower() != "content-length":
                    h.send_header(key, value)
            if length is not None:
                h.send_header("Content-Length", length)
            else:
                h.send_header("Transfer-Encoding", "chunked")
            h.end_headers()
            self.stream_body(h, resp, row, chunked=length is None)
        except (BrokenPipeError, ConnectionResetError) as exc:
            row["error"] = f"client went away: {type(exc).__name__}"
            h.close_connection = True
        except (OSError, http.client.HTTPException) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            h.close_connection = True
        finally:
            conn.close()
            row["t_end"] = row["t_end"] or time.time()
            self.log(row)

    @staticmethod
    def stream_body(h: BaseHTTPRequestHandler, resp: http.client.HTTPResponse, row: dict[str, Any], *, chunked: bool) -> None:
        while True:
            data = resp.read1(CHUNK)
            if not data:
                break
            if row["t_first_byte"] is None:
                row["t_first_byte"] = time.time()
            row["bytes"] += len(data)
            h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n" if chunked else data)
            h.wfile.flush()
        if chunked:
            h.wfile.write(b"0\r\n\r\n")
            h.wfile.flush()
        row["t_end"] = time.time()


def read_log(path: Path) -> list[dict[str, Any]]:
    """The requests a proxy logged, in the order they ended; unreadable lines are skipped."""
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
