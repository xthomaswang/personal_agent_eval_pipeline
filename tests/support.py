"""Test support: an in-process stand-in for oMLX and servers on OS-assigned ports."""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
import time
import zlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from bench.proxy import ProxyConfig

MODEL = "Qwen3.8-27B-8bit"

# Exactly what oMLX 0.3.9 sends as keepalive with sse_keepalive_mode="chunk".
OPENAI_KEEPALIVE = (
    b'data: {"id":"chatcmpl-keepalive","object":"chat.completion.chunk","created":0,'
    b'"model":"keepalive","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}\n\n'
)
ANTHROPIC_PING = b'event: ping\ndata: {"type":"ping"}\n\n'

OPENAI_USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "prompt_tokens_details": {"cached_tokens": 3},
}
# oMLX with prefix caching on: input_tokens=0, the prompt is split into
# cache_creation + cache_read.
ANTHROPIC_USAGE = {
    "input_tokens": 0,
    "output_tokens": 9,
    "cache_creation_input_tokens": 8,
    "cache_read_input_tokens": 4,
}

ERROR_RESPONSES: dict[str, tuple[int, Any]] = {
    "error-400": (400, {"detail": "Chat template error: Unexpected reasoning effort high."}),
    "error-422": (
        422,
        {
            "detail": [
                {
                    "type": "missing",
                    "loc": ["body", "max_tokens"],
                    "msg": "Field required",
                    "input": {"messages": [{"role": "user", "content": "TOP-SECRET-PROMPT"}]},
                }
            ]
        },
    ),
    "error-503": (503, {"detail": "Server is busy with oQ quantization. Please try again later."}),
}


def sse(obj: Any, event: str | None = None) -> bytes:
    data = obj if isinstance(obj, str) else json.dumps(obj)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n".encode()


def openai_chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    return sse(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def anthropic_event(obj: dict[str, Any]) -> bytes:
    return sse(obj, obj["type"])


def openai_stream(include_usage: bool) -> list[bytes]:
    events = [
        OPENAI_KEEPALIVE,
        openai_chunk({"role": "assistant", "content": ""}),
        openai_chunk({"reasoning_content": "Thinking about it."}),
        openai_chunk({"content": "Hello"}),
        openai_chunk({"content": " world"}),
        openai_chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
                ]
            }
        ),
        openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]}),
        openai_chunk(
            {
                "tool_calls": [
                    {"index": 1, "id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
                ]
            }
        ),
        openai_chunk({}, "tool_calls"),
    ]
    if include_usage:
        events.append(
            sse(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [],
                    "usage": OPENAI_USAGE,
                }
            )
        )
    events.append(b"data: [DONE]\n\n")
    return events


def anthropic_stream() -> list[bytes]:
    return [
        ANTHROPIC_PING,
        anthropic_event(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    # an estimate; message_delta carries the real counts
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }
        ),
        anthropic_event(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": "omlx-reasoning"}}
        ),
        anthropic_event({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Hmm."}}),
        anthropic_event({"type": "content_block_stop", "index": 0}),
        anthropic_event({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        anthropic_event({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Reading."}}),
        anthropic_event({"type": "content_block_stop", "index": 1}),
        anthropic_event(
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}}
        ),
        anthropic_event(
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "/tmp/a"}'}}
        ),
        anthropic_event({"type": "content_block_stop", "index": 2}),
        anthropic_event(
            {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {}}}
        ),
        anthropic_event(
            {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'}}
        ),
        anthropic_event({"type": "content_block_stop", "index": 3}),
        anthropic_event({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": ANTHROPIC_USAGE}),
        anthropic_event({"type": "message_stop"}),
    ]


def openai_tool_only_stream() -> list[bytes]:
    """Thinking off and only a tool call: oMLX sends it after generation."""
    return [
        OPENAI_KEEPALIVE,
        openai_chunk({"role": "assistant", "content": ""}),
        openai_chunk(
            {"tool_calls": [{"index": 0, "id": "call_9", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}
        ),
        openai_chunk({}, "tool_calls"),
        b"data: [DONE]\n\n",
    ]


def anthropic_tool_only_stream() -> list[bytes]:
    start = {
        "type": "message_start",
        "message": {"id": "msg_t", "type": "message", "role": "assistant", "model": MODEL, "content": [], "usage": {"input_tokens": 5, "output_tokens": 0}},
    }
    return [
        ANTHROPIC_PING,
        anthropic_event(start),
        anthropic_event(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "Read", "input": {}}}
        ),
        anthropic_event({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        anthropic_event({"type": "content_block_stop", "index": 0}),
        anthropic_event({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4, "input_tokens": 5}}),
        anthropic_event({"type": "message_stop"}),
    ]


def png(width: int, height: int, fill: int = 0) -> bytes:
    """A valid RGB PNG of one color."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes([fill]) * (3 * width)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")


def jpeg_header(width: int, height: int) -> bytes:
    """SOI, a JFIF APP0 segment and a baseline frame header: enough for sizing."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app0 + b"\xff\xff" + sof0 + b"\xff\xd9"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def openai_gated_parts() -> list[bytes]:
    return [
        OPENAI_KEEPALIVE + openai_chunk({"content": "part-0"}),
        openai_chunk({"content": "part-1"}),
        openai_chunk({"content": "part-2"}) + openai_chunk({}, "stop") + b"data: [DONE]\n\n",
    ]


def anthropic_gated_parts() -> list[bytes]:
    start = {
        "type": "message_start",
        "message": {"id": "msg_g", "type": "message", "role": "assistant", "model": MODEL, "content": [], "usage": {"input_tokens": 5, "output_tokens": 0}},
    }
    return [
        ANTHROPIC_PING
        + anthropic_event(start)
        + anthropic_event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        + anthropic_event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "part-0"}}),
        anthropic_event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "part-1"}}),
        anthropic_event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "part-2"}})
        + anthropic_event({"type": "content_block_stop", "index": 0})
        + anthropic_event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3, "input_tokens": 5}})
        + anthropic_event({"type": "message_stop"}),
    ]


OPENAI_COMPLETION = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 1,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello world",
                "reasoning_content": "Thinking about it.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": OPENAI_USAGE,
}

ANTHROPIC_MESSAGE = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [
        {"type": "thinking", "thinking": "Hmm.", "signature": "omlx-reasoning"},
        {"type": "text", "text": "Reading."},
        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/tmp/a"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": ANTHROPIC_USAGE,
}


class FakeUpstream:
    """Records every request and answers like oMLX 0.3.9 would.

    The ``x-fake-scenario`` request header (relayed by the proxy like any
    other header) selects error answers, a gated stream or a broken stream.
    """

    def __init__(self, gate_timeout: float = 5.0) -> None:
        self.requests: list[dict[str, Any]] = []
        self.gate_timeout = gate_timeout
        self.gates = [threading.Event() for _ in range(4)]
        self.gate_timeouts = 0
        self.stream_cancelled = threading.Event()
        self.sent = bytearray()  # bytes yielded by the last streamed body
        self.url = ""
        self.app = Starlette(
            routes=[Route("/{path:path}", self.dispatch, methods=["GET", "HEAD", "POST", "PUT", "DELETE"])]
        )

    def release_gates(self) -> None:
        for gate in self.gates:
            gate.set()

    async def dispatch(self, request: Request) -> Response:
        body = await request.body()
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        self.requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": body,
                "json": parsed,
            }
        )
        scenario = request.headers.get("x-fake-scenario", "")
        if scenario in ERROR_RESPONSES:
            status, payload = ERROR_RESPONSES[scenario]
            return JSONResponse(payload, status_code=status)

        path = request.url.path
        if path == "/v1/models":
            return JSONResponse({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "omlx"}]})
        if path not in ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if not isinstance(parsed, dict):
            return JSONResponse(
                {"detail": [{"type": "json_invalid", "loc": ["body", 0], "msg": "JSON decode error", "input": {}}]},
                status_code=422,
            )
        if path == "/v1/messages/count_tokens":
            return JSONResponse({"input_tokens": 42})

        openai = path == "/v1/chat/completions"
        if scenario == "gated":
            return self._stream(openai_gated_parts() if openai else anthropic_gated_parts(), gated=True)
        if scenario == "abort":
            parts = openai_stream(False)[:4] if openai else anthropic_stream()[:4]
            return self._stream(parts, fail_after=True)
        if scenario == "tool-only":
            return self._stream(openai_tool_only_stream() if openai else anthropic_tool_only_stream())
        if parsed.get("stream"):
            if openai:
                options = parsed.get("stream_options") or {}
                return self._stream(openai_stream(include_usage=options.get("include_usage") is True))
            return self._stream(anthropic_stream())
        # oMLX sends non-streaming answers as a stream too: keepalive spaces
        # first, then the JSON body.
        return self._stream([b" ", json.dumps(OPENAI_COMPLETION if openai else ANTHROPIC_MESSAGE).encode()], media_type="application/json")

    def _stream(
        self,
        parts: list[bytes],
        *,
        gated: bool = False,
        fail_after: bool = False,
        media_type: str = "text/event-stream",
    ) -> StreamingResponse:
        self.sent = bytearray()

        async def body() -> AsyncIterator[bytes]:
            try:
                for i, part in enumerate(parts):
                    self.sent += part
                    yield part
                    if gated and i < len(parts) - 1:
                        await self._wait_gate(i)
                if fail_after:
                    raise RuntimeError("simulated upstream crash")
            except BaseException:
                self.stream_cancelled.set()
                raise

        return StreamingResponse(body(), media_type=media_type)

    async def _wait_gate(self, index: int) -> None:
        # Poll instead of blocking a thread, so that cancellation (the proxy
        # closing the connection) interrupts the wait immediately.
        deadline = time.monotonic() + self.gate_timeout
        while not self.gates[index].is_set():
            if time.monotonic() > deadline:
                self.gate_timeouts += 1
                return
            await anyio.sleep(0.002)


class ServerThread:
    """Serve an ASGI app with uvicorn on an OS-assigned port in a thread."""

    def __init__(self, app: Any) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on", timeout_graceful_shutdown=1)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("test server did not start")
            time.sleep(0.005)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.sock.close()


@dataclass
class ProxyHandle:
    url: str
    config: ProxyConfig

    @property
    def run_dir(self) -> Path:
        return self.config.run_dir

    @property
    def calls_path(self) -> Path:
        return self.run_dir / "calls.jsonl"

    def records(self) -> list[dict[str, Any]]:
        if not self.calls_path.exists():
            return []
        return [json.loads(line) for line in self.calls_path.read_text().splitlines() if line]

    def record(self, index: int = -1) -> dict[str, Any]:
        return self.records()[index]

    def wait_for_record(self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 5.0) -> dict[str, Any]:
        """Poll for a record that is written after the client has gone away."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for record in self.records():
                if predicate(record):
                    return record
            time.sleep(0.01)
        raise AssertionError(f"no matching record in {self.calls_path}: {self.records()}")


def free_port() -> int:
    """A port with nothing listening on it (bound once, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
