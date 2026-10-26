"""Starlette app that relays requests to the upstream and records each call."""

from __future__ import annotations

import contextlib
import json
import secrets
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from .config import ProxyConfig
from .observe import MODEL_ENDPOINTS, ResponseObserver, clip
from .recorder import CallRecorder
from .request_info import GENERATION_ENDPOINTS, THINKING_ENDPOINTS, classify, summarize_request
from .sampling import apply_sampling, sampling_view
from .steps import counts_as_step
from .thinking import rewrite_thinking, thinking_view

SCHEMA_VERSION = 2
BUDGET_ERROR = "bench_budget_exhausted"
RUN_ID_HEADER = "x-bench-run-id"
CALL_INDEX_HEADER = b"x-bench-call-index"
BENCH_HEADER_PREFIX = "x-bench-"
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
# httpx sets host and content-length itself (the body may have been
# rewritten); identity encoding keeps relayed bytes parseable; the request
# body has already been read, so "expect: 100-continue" is moot.
DROP_REQUEST_HEADERS = HOP_BY_HOP | {"host", "content-length", "accept-encoding", "expect"}
# uvicorn adds its own server and date headers. content-length is dropped so
# that every relayed body is chunked: the client then sees the end of a
# response only with the terminating chunk, which is sent after the call
# record has been written.
DROP_RESPONSE_HEADERS = HOP_BY_HOP | {"server", "date", "content-length"}
SECRET_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"}
)


def _dump_json(obj: Any) -> bytes:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates survive only as \u escapes
        return json.dumps(obj, separators=(",", ":")).encode("ascii")


def _keep_one_reasoning_field(body: dict[str, Any]) -> int:
    """Leave one of `reasoning_content` and `reasoning` on each assistant message that has both.

    The non-empty one is kept, `reasoning_content` first (Hermes stores the same text in both).
    The messages list and the changed messages are copies, so the parsed request stays intact for
    the record. Returns the number of messages changed.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    out: list[Any] = []
    dropped = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant" and {"reasoning_content", "reasoning"} <= message.keys():
            content = message.get("reasoning_content")
            drop = "reasoning" if isinstance(content, str) and content else "reasoning_content"
            message = {k: v for k, v in message.items() if k != drop}
            dropped += 1
        out.append(message)
    if dropped:
        body["messages"] = out
    return dropped


def _sanitized(headers: list[tuple[str, str]]) -> list[list[str]]:
    return [[k, "<redacted>" if k.lower() in SECRET_HEADERS else v] for k, v in headers]


def _error_body(protocol: str | None, status: int, kind: str, message: str) -> bytes:
    """An error answered by the proxy itself, in the client's protocol."""
    text = f"bench proxy: {message}"
    if protocol == "anthropic":
        error_type = "invalid_request_error" if status < 500 else "api_error"
        return _dump_json({"type": "error", "error": {"type": error_type, "message": text}})
    return _dump_json({"error": {"message": text, "type": kind, "code": kind}})


@dataclass
class RunState:
    """Counters of one run (one run_id: the CLI value or an X-Bench-Run-Id)."""

    images_seen: set[str] = field(default_factory=set)
    main_calls: int = 0  # main calls so far, i.e. the last main_index
    steps_done: int = 0  # finished main calls that count as steps (n_llm)
    steps_in_flight: int = 0  # admitted main calls that have not finished


class Call:
    """Everything the proxy tracks about one request while it is in flight."""

    def __init__(
        self,
        index: int,
        record: dict[str, Any],
        protocol: str | None,
        endpoint: str,
        client_headers: list[tuple[str, str]],
        body: bytes,
    ) -> None:
        self.index = index
        self.record = record
        self.protocol = protocol
        self.endpoint = endpoint
        self.client_headers = client_headers
        self.body = body
        self.forwarded = body
        self.method = "GET"
        self.url = ""
        self.upstream_headers: list[tuple[str, str]] = []
        self.observer: ResponseObserver | None = None
        self.response_headers: list[tuple[bytes, bytes]] = []
        self.response_body: bytearray | None = None
        self.response_bytes = 0
        self.first_byte: tuple[float, float] | None = None
        self.upstream_done = False
        self.error: dict[str, Any] | None = None
        self.finalized = False
        self.run_state: RunState | None = None
        # True while this main call holds one step of the budget.
        self.step_reserved = False

    def start_response(self, response: httpx.Response, keep_body: bool) -> None:
        content_type = response.headers.get("content-type", "")
        self.record["status"] = response.status_code
        self.record["content_type"] = content_type or None
        self.observer = ResponseObserver(self.protocol, self.endpoint, response.status_code, content_type)
        self.response_headers = list(response.headers.raw)
        if keep_body:
            self.response_body = bytearray()

    def client_response_headers(self) -> list[tuple[bytes, bytes]]:
        headers = [
            (k, v)
            for k, v in self.response_headers
            if k.decode("latin-1").lower() not in DROP_RESPONSE_HEADERS
        ]
        headers.append((CALL_INDEX_HEADER, str(self.index).encode()))
        return headers

    def on_chunk(self, chunk: bytes) -> None:
        wall, mono = time.time(), time.monotonic()
        if self.first_byte is None:
            self.first_byte = (wall, mono)
        self.response_bytes += len(chunk)
        if self.observer is not None:
            self.observer.feed(chunk, wall, mono)
        if self.response_body is not None:
            self.response_body += chunk


class ProxyApp:
    def __init__(self, config: ProxyConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.recorder = CallRecorder(config.run_dir, config.log_bodies)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._runs: dict[str, RunState] = {}

    # -- lifecycle --------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            c = self.config
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=c.connect_timeout, read=c.read_timeout, write=c.read_timeout, pool=c.read_timeout
                ),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=32),
                # Never route loopback traffic through HTTP(S)_PROXY settings.
                trust_env=False,
                follow_redirects=False,
                transport=self._transport,
            )
        return self._client

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        _ = self.client
        try:
            yield
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    # -- request handling -----------------------------------------------

    async def handle(self, request: Request) -> Any:
        wall, mono = time.time(), time.monotonic()
        index = self.recorder.next_index()
        path = request.url.path
        query = request.url.query
        protocol, endpoint = classify(path)
        header_run_id = request.headers.get(RUN_ID_HEADER, "").strip()
        run_id = header_run_id or self.config.run_id
        record = self._new_record(
            index=index,
            run_id=run_id,
            run_id_source="header" if header_run_id else "cli",
            protocol=protocol,
            endpoint=endpoint,
            request=request,
            wall=wall,
            mono=mono,
        )
        client_headers = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in request.headers.raw]
        body = await request.body()
        call = Call(index, record, protocol, endpoint, client_headers, body)
        record["request_bytes"] = len(body)

        if endpoint == "hello":
            # Claude Code's connectivity probe; oMLX has no such route.
            record["handled_locally"] = True
            record["status"] = 200
            self.finalize(call, disconnected=False)
            return Response(status_code=200, headers={CALL_INDEX_HEADER.decode(): str(index)})

        if protocol is not None and request.method == "POST" and body:
            call.forwarded = self._prepare_body(call)
        record["forwarded_bytes"] = len(call.forwarded)
        if not self._admit(call, run_id):
            return self._refuse_over_budget(call)

        raw_path = request.scope.get("raw_path") or path.encode()
        call.method = request.method
        call.url = self.config.upstream + raw_path.decode("latin-1") + (f"?{query}" if query else "")
        token = self.config.upstream_token
        call.upstream_headers = [
            (k, v)
            for k, v in client_headers
            if k.lower() not in DROP_REQUEST_HEADERS
            and not k.lower().startswith(BENCH_HEADER_PREFIX)
            and not (token and k.lower() == "authorization")
        ] + [("accept-encoding", "identity")]
        if token:
            call.upstream_headers.append(("authorization", f"Bearer {token}"))
        return RelayResponse(self, call)

    def _prepare_body(self, call: Call) -> bytes:
        """Summarize the request and apply the thinking and usage rewrites."""
        record = call.record
        try:
            parsed = json.loads(call.body)
        except ValueError:
            record["warnings"].append("request body is not JSON; forwarded unchanged")
            return call.body
        if not isinstance(parsed, dict):
            record["warnings"].append("request body is not a JSON object; forwarded unchanged")
            return call.body

        protocol, endpoint = call.protocol, call.endpoint
        record.update(summarize_request(protocol, endpoint, parsed))
        # Rewrites replace top-level keys only, so a shallow copy keeps the
        # original intact for the record.
        outgoing = dict(parsed)
        if endpoint in THINKING_ENDPOINTS:
            record["thinking_original"] = thinking_view(protocol, parsed)
            record["warnings"] += rewrite_thinking(protocol, outgoing, self.config.thinking)
            record["thinking_forwarded"] = thinking_view(protocol, outgoing)
            record["thinking_rewritten"] = record["thinking_forwarded"] != record["thinking_original"]
        if endpoint in GENERATION_ENDPOINTS:
            record["sampling_original"] = sampling_view(parsed)
            apply_sampling(outgoing, self.config.sampling)
            if self.config.random_seed and protocol == "openai":
                outgoing["seed"] = secrets.randbelow(2**31)
            record["sampling_forwarded"] = sampling_view(outgoing)
        if endpoint == "chat_completions" and self.config.one_reasoning_field:
            dropped = _keep_one_reasoning_field(outgoing)
            if dropped:
                record["warnings"].append(f"one reasoning field: dropped a duplicate from {dropped} assistant message(s)")
        if endpoint == "chat_completions" and self.config.inject_stream_usage and parsed.get("stream") is True:
            options = outgoing.get("stream_options")
            if not (isinstance(options, dict) and options.get("include_usage") is True):
                outgoing["stream_options"] = {**(options if isinstance(options, dict) else {}), "include_usage": True}
                record["usage_injected"] = True
        if outgoing == parsed:
            return call.body
        return _dump_json(outgoing)

    def _admit(self, call: Call, run_id: str) -> bool:
        """Count the call in its run; return False when the step budget refuses it.

        A main call is beyond the budget when the run's steps (finished main
        calls that succeeded, as in n_llm) plus the main calls still in
        flight already reach max_steps. Failed main calls give their step
        back, so upstream errors do not eat the budget.
        """
        record = call.record
        if call.endpoint not in GENERATION_ENDPOINTS:
            return True
        state = self._runs.setdefault(run_id, RunState())
        call.run_state = state
        if record["role"] == "main":
            state.main_calls += 1
            record["main_index"] = state.main_calls
            limit = self.config.max_steps
            if limit is not None and state.steps_done + state.steps_in_flight >= limit:
                record["budget_exhausted"] = True
                if self.config.budget_mode == "reject":
                    return False
            state.steps_in_flight += 1
            call.step_reserved = True
        if record["image_sha256"] is not None:
            # Images the model has not been sent before in this run.
            new_sides: list[int | None] = []
            for sha, wh in zip(record["image_sha256"], record["image_wh"]):
                if sha is None or sha in state.images_seen:
                    continue
                state.images_seen.add(sha)
                new_sides.append(max(wh) if wh else None)
            record["images_new"] = len(new_sides)
            record["images_new_long_side"] = new_sides
        return True

    def _refuse_over_budget(self, call: Call) -> Response:
        record = call.record
        message = f"{BUDGET_ERROR}: the run has used its {self.config.max_steps} steps"
        call.error = {"source": "proxy", "type": BUDGET_ERROR, "message": message}
        body = _error_body(call.protocol, 400, BUDGET_ERROR, message)
        record["handled_locally"] = True
        record["status"] = 400
        record["content_type"] = "application/json"
        if self.config.log_bodies:
            call.response_body = bytearray(body)
        self.finalize(call, disconnected=False)
        return Response(
            body,
            status_code=400,
            media_type="application/json",
            headers={CALL_INDEX_HEADER.decode(): str(call.index)},
        )

    def _new_record(
        self,
        *,
        index: int,
        run_id: str,
        run_id_source: str,
        protocol: str | None,
        endpoint: str,
        request: Request,
        wall: float,
        mono: float,
    ) -> dict[str, Any]:
        beta = request.headers.get("anthropic-beta")
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_id_source": run_id_source,
            "call_index": index,
            "harness": self.config.harness,
            "protocol": protocol,
            "endpoint": endpoint,
            "role": None,
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query or None,
            "upstream": self.config.upstream,
            "handled_locally": False,
            "model": None,
            "stream": False,
            "request_bytes": 0,
            "forwarded_bytes": 0,
            "messages": None,
            "tools": None,
            "request_tool_names": None,
            "request_tools_sha256": None,
            "request_tools_raw_sha256": None,
            "max_tokens": None,
            "images": None,
            "image_sha256": None,
            "image_wh": None,
            "images_new": None,
            "images_new_long_side": None,
            "user_agent": request.headers.get("user-agent"),
            "anthropic_beta": [b.strip() for b in beta.split(",") if b.strip()] if beta else None,
            "thinking_mode": str(self.config.thinking),
            "thinking_original": None,
            "thinking_forwarded": None,
            "thinking_rewritten": False,
            "sampling_policy": str(self.config.sampling) + ("+random-seed" if self.config.random_seed else ""),
            "sampling_original": None,
            "sampling_forwarded": None,
            "usage_injected": False,
            "main_index": None,
            "max_steps": self.config.max_steps,
            "budget_exhausted": False,
            "status": None,
            "content_type": None,
            "response_bytes": 0,
            "usage": None,
            "usage_raw": None,
            "tool_calls": None,
            "tool_names": None,
            "stop_reason": None,
            "output_text_chars": None,
            "output_reasoning_chars": None,
            "counted_input_tokens": None,
            "sse_events": None,
            "sse_keepalives": None,
            "stream_complete": None,
            "client_disconnected": False,
            "error": None,
            "warnings": [],
            "ts_start": wall,
            "ts_first_byte": None,
            "ts_first_output": None,
            "ts_first_content": None,
            "ts_end": None,
            "mono_start": mono,
            "mono_first_byte": None,
            "mono_first_output": None,
            "mono_first_content": None,
            "mono_end": None,
            "ttfb_s": None,
            "ttft_s": None,
            "ttft_valid": None,
            "duration_s": None,
            "bodies": None,
        }

    # -- completion -------------------------------------------------------

    def finalize(self, call: Call, disconnected: bool) -> None:
        """Complete the record and write it. Runs exactly once per call."""
        if call.finalized:
            return
        call.finalized = True
        wall, mono = time.time(), time.monotonic()
        record = call.record
        observer = call.observer
        if observer is not None:
            try:
                observer.finish()
            except Exception as exc:  # observation must never break relaying
                record["warnings"].append(f"response observation failed: {exc!r}")
            record.update(observer.results())
            if record["model"] is None and observer.model:
                record["model"] = observer.model
        record["response_bytes"] = call.response_bytes
        record["client_disconnected"] = disconnected
        record["error"] = self._final_error(call, disconnected)

        start = record["mono_start"]
        record["ts_end"], record["mono_end"] = wall, mono
        record["duration_s"] = round(mono - start, 6)
        if call.first_byte is not None:
            record["ts_first_byte"], record["mono_first_byte"] = call.first_byte
            record["ttfb_s"] = round(call.first_byte[1] - start, 6)
        if observer is not None and observer.first_output is not None:
            record["ts_first_output"], record["mono_first_output"] = observer.first_output
            record["ttft_s"] = round(observer.first_output[1] - start, 6)
            record["ttft_valid"] = observer.ttft_valid
        if observer is not None and observer.first_content is not None:
            record["ts_first_content"], record["mono_first_content"] = observer.first_content

        if call.step_reserved and call.run_state is not None:
            call.step_reserved = False
            call.run_state.steps_in_flight -= 1
            if counts_as_step(record):
                call.run_state.steps_done += 1

        if self.config.log_bodies:
            record["bodies"] = self._write_bodies(call)
        try:
            self.recorder.write(record)
        except OSError as exc:
            print(f"[proxy] failed to write call record: {exc}", file=sys.stderr, flush=True)
        if self.config.echo:
            print(_summary_line(record), file=sys.stderr, flush=True)

    @staticmethod
    def _final_error(call: Call, disconnected: bool) -> dict[str, Any] | None:
        observer = call.observer
        error = call.error or (observer.error if observer is not None else None)
        if error is not None or observer is None:
            return error
        complete = observer.complete if observer.is_sse else call.upstream_done
        if disconnected:
            # A client may close right after [DONE] / message_stop; that is fine.
            return None if complete else {"source": "client", "type": "client_disconnected"}
        if not call.upstream_done:
            return {"source": "proxy", "type": "aborted"}
        if observer.is_sse and call.endpoint in MODEL_ENDPOINTS and not observer.complete and observer.status < 400:
            return {"source": "upstream", "type": "incomplete_stream"}
        return None

    def _write_bodies(self, call: Call) -> dict[str, str] | None:
        rec = self.recorder
        paths: dict[str, str] = {}
        try:
            if call.body:
                ext = "json" if call.protocol is not None else "bin"
                paths["request"] = rec.write_body(call.index, "request", ext, call.body)
            if call.forwarded is not call.body and call.forwarded != call.body:
                paths["forwarded"] = rec.write_body(call.index, "forwarded", "json", call.forwarded)
            if call.response_body:
                content_type = (call.record["content_type"] or "").lower()
                ext = "sse" if "event-stream" in content_type else "json" if "json" in content_type else "bin"
                paths["response"] = rec.write_body(call.index, "response", ext, bytes(call.response_body))
            headers = {
                "request": _sanitized(call.client_headers),
                "response": _sanitized([(k.decode("latin-1"), v.decode("latin-1")) for k, v in call.response_headers]),
            }
            paths["headers"] = rec.write_body(
                call.index, "headers", "json", json.dumps(headers, indent=2, ensure_ascii=False).encode()
            )
        except OSError as exc:
            call.record["warnings"].append(f"failed to write bodies: {exc}")
        return paths or None


class RelayResponse:
    """ASGI response that relays one upstream response (see _Relay)."""

    def __init__(self, proxy: ProxyApp, call: Call) -> None:
        self.proxy = proxy
        self.call = call

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _Relay(self.proxy, self.call, receive, send).run()


class _Relay:
    """Forward one call upstream and stream the answer back unbuffered.

    A watcher task waits for the client's ``http.disconnect``. When the client
    goes away (for example the user aborts a turn), the relay is cancelled and
    the upstream connection is closed at once, which makes oMLX stop
    generating. Without this the GPU would stay busy and skew the next call:
    uvicorn's ``send()`` silently drops data after a disconnect, so a relay
    that only writes would never notice.
    """

    def __init__(self, proxy: ProxyApp, call: Call, receive: Receive, send: Send) -> None:
        self.proxy = proxy
        self.call = call
        self.receive = receive
        self.send = send
        # Set just before the final body message: uvicorn reports
        # http.disconnect once a response is complete, which is not a client
        # disconnect.
        self.finished = False
        self.disconnected = False

    async def run(self) -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._watch_disconnect, tg.cancel_scope)
            try:
                await self._forward()
            finally:
                tg.cancel_scope.cancel()

    async def _watch_disconnect(self, scope: anyio.CancelScope) -> None:
        while True:
            message = await self.receive()
            if message["type"] == "http.disconnect":
                if not self.finished:
                    self.disconnected = True
                    scope.cancel()
                return

    async def _forward(self) -> None:
        proxy, call = self.proxy, self.call
        response: httpx.Response | None = None
        try:
            request = proxy.client.build_request(
                call.method, call.url, headers=call.upstream_headers, content=call.forwarded or None
            )
            try:
                response = await proxy.client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                await self._local_error(504, "upstream_timeout", exc)
                return
            except httpx.HTTPError as exc:
                await self._local_error(502, "upstream_unreachable", exc)
                return
            call.start_response(response, keep_body=proxy.config.log_bodies)
            await self.send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": call.client_response_headers(),
                }
            )
            async for chunk in response.aiter_raw():
                call.on_chunk(chunk)
                await self.send({"type": "http.response.body", "body": chunk, "more_body": True})
            call.upstream_done = True
        except httpx.HTTPError as exc:
            # The upstream broke off mid-response (or stalled past the read
            # timeout). The client connection is aborted below.
            call.error = {"source": "upstream", "type": type(exc).__name__, "message": clip(str(exc))}
        except OSError:
            self.disconnected = True
        finally:
            if response is not None:
                with anyio.CancelScope(shield=True):
                    await response.aclose()
            # Record before the response is finished, so the record exists
            # once the client has seen the end of the response.
            proxy.finalize(call, disconnected=self.disconnected)
        if call.upstream_done and not self.disconnected:
            self.finished = True
            await self.send({"type": "http.response.body", "body": b"", "more_body": False})
        # Otherwise return without completing the response: uvicorn then
        # closes the connection and the client sees a broken response, as it
        # would have without the proxy.

    async def _local_error(self, status: int, kind: str, exc: Exception) -> None:
        call = self.call
        message = clip(f"upstream {self.proxy.config.upstream}: {type(exc).__name__}: {exc}")
        call.error = {"source": "proxy", "type": kind, "message": message}
        body = _error_body(call.protocol, status, kind, message)
        call.record["status"] = status
        call.record["content_type"] = "application/json"
        if self.proxy.config.log_bodies:
            call.response_body = bytearray(body)
        self.proxy.finalize(call, disconnected=False)
        self.finished = True
        await self.send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (CALL_INDEX_HEADER, str(call.index).encode()),
                ],
            }
        )
        await self.send({"type": "http.response.body", "body": body, "more_body": False})


def _summary_line(record: dict[str, Any]) -> str:
    parts = [f"#{record['call_index']}", record["protocol"] or "-", record["endpoint"]]
    if record["role"] == "main":
        parts.append(f"main#{record['main_index']}")
    elif record["role"]:
        parts.append(record["role"])
    parts.append(str(record["status"]))
    if record["stream"]:
        parts.append("stream")
    if record["budget_exhausted"]:
        parts.append("over-budget")
    parts.append(f"{record['duration_s']:.2f}s")
    if record["ttft_s"] is not None:
        parts.append(f"ttft={record['ttft_s']:.2f}s")
    usage = record["usage"] or {}
    if usage.get("prompt_tokens") is not None:
        parts.append(f"in={usage['prompt_tokens']}")
    if usage.get("cached_tokens"):
        parts.append(f"cached={usage['cached_tokens']}")
    if usage.get("completion_tokens") is not None:
        parts.append(f"out={usage['completion_tokens']}")
    if record["tool_names"]:
        parts.append("tools=" + ",".join(str(name) for name in record["tool_names"]))
    if record["stop_reason"]:
        parts.append(f"stop={record['stop_reason']}")
    if record["error"]:
        error = record["error"]
        parts.append(f"error={error.get('type') or error.get('source')}")
    return "[proxy] " + " ".join(parts)


def create_app(config: ProxyConfig, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """Build the proxy ASGI app. ``transport`` overrides the upstream transport (tests)."""
    proxy = ProxyApp(config, transport)
    app = Starlette(routes=[Route("/{path:path}", proxy.handle, methods=METHODS)], lifespan=proxy.lifespan)
    app.state.proxy = proxy
    return app
