"""Mock OpenAI + Anthropic SSE server for tests (stdlib asyncio, chunked HTTP/1.1).

Legacy mode (no responder): the scenario is chosen by the `x-mock-scenario` request header
(default "text"). With a `responder` (see agentbench.mock_smart.SmartResponder) every
request is answered by it instead. Bind to an ephemeral port (port=0) in tests, or
8790-8799 for manual runs; never to 8500, 8600-8699 or the frameworks' 8700-8706.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

D = 0.02  # default inter-event delay (s)


def _oa(delta: dict[str, Any] | None = None, finish: str | None = None, usage: dict | None = None,
        choices: bool = True, **extra: Any) -> str:
    obj: dict[str, Any] = {"object": "chat.completion.chunk", "model": "mock", **extra}
    obj["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []
    if usage is not None:
        obj["usage"] = usage
    return "data: " + json.dumps(obj) + "\n\n"


USAGE = {"prompt_tokens": 1200, "completion_tokens": 5, "total_tokens": 1205,
         "prompt_tokens_details": {"cached_tokens": 1024}}


def openai_events(scenario: str) -> list[tuple[float, str]]:
    """(delay_before, raw SSE text) pairs."""
    if scenario == "text":
        ev = [(D, _oa({"role": "assistant", "content": ""})), (D, ": keep-alive\n\n"),
              (D, _oa(choices=False, prompt_progress={"total": 1200, "cache": 1024, "processed": 600})),
              (D, "event: progress\ndata: {\"progress\": 0.9}\n\n"),
              (0.1, _oa({"content": "Hello"})), (D, _oa({"content": " world"})), (D, _oa({"content": "!"})),
              (D, _oa({}, finish="stop")), (D, _oa(choices=False, usage=USAGE)), (0, "data: [DONE]\n\n")]
        return ev
    if scenario == "tools":
        tc0 = {"index": 0, "id": "call_a", "type": "function", "function": {"name": "fs_read_file", "arguments": ""}}
        tc1 = {"index": 1, "id": "call_b", "type": "function", "function": {"name": "fs_read_file", "arguments": ""}}
        return [(D, _oa({"role": "assistant"})), (D, _oa({"tool_calls": [tc0]})),
                (D, _oa({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": \"~/pro"}}]})),
                (D, _oa({"tool_calls": [{"index": 0, "function": {"arguments": "ject/README.md\"}"}}]})),
                (D, _oa({"tool_calls": [tc1]})),
                (D, _oa({"tool_calls": [{"index": 1, "function": {"arguments": "{\"path\": \"~/project/setup.cfg\"}"}}]})),
                (D, _oa({}, finish="tool_calls")), (D, _oa(choices=False, usage=USAGE)), (0, "data: [DONE]\n\n")]
    if scenario == "reasoning":
        return [(D, _oa({"role": "assistant", "content": None})), (D, _oa({"reasoning_content": "Think"})),
                (D, _oa({"reasoning_content": "ing."})), (D, _oa({"content": "Answer"})),
                (D, _oa({}, finish="stop")), (0, "data: [DONE]\n\n")]
    if scenario == "inline_think":
        return [(D, _oa({"role": "assistant"})), (D, _oa({"content": "<think>\nplan"})),
                (D, _oa({"content": "\n</think>\n\nDone"})), (D, _oa({}, finish="stop")), (0, "data: [DONE]\n\n")]
    if scenario == "leak":
        return [(D, _oa({"content": "<tool_call>\n<function=fs_read_file>\n<parameter=path>\nx\n</parameter>"})),
                (D, _oa({}, finish="stop")), (0, "data: [DONE]\n\n")]
    if scenario == "stall":
        return [(D, _oa({"content": "a"})), (5.0, _oa({"content": "b"}))]
    if scenario == "error_event":
        return [(D, _oa({"content": "a"})), (D, "data: " + json.dumps({"error": {"message": "boom"}}) + "\n\n")]
    raise KeyError(scenario)


def anthropic_events(scenario: str) -> list[tuple[float, str]]:
    def e(name: str, obj: dict[str, Any]) -> str:
        return f"event: {name}\ndata: {json.dumps(dict(obj, type=name))}\n\n"

    start = e("message_start", {"message": {"id": "msg_1", "role": "assistant", "content": [],
                                             "usage": {"input_tokens": 900, "cache_read_input_tokens": 800, "output_tokens": 1}}})
    if scenario == "error_event":
        return [(D, start), (D, e("error", {"error": {"type": "overloaded_error", "message": "busy"}}))]
    return [(D, start), (D, e("ping", {})),
            (D, e("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}})),
            (0.1, e("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}})),
            (D, e("content_block_stop", {"index": 0})),
            (D, e("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}})),
            (D, e("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "Reading it."}})),
            (D, e("content_block_stop", {"index": 1})),
            (D, e("content_block_start", {"index": 2, "content_block": {"type": "tool_use", "id": "tu_1",
                                                                         "name": "fs_read_file", "input": {}}})),
            (D, e("content_block_delta", {"index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"path\": "}})),
            (D, e("content_block_delta", {"index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"~/a.txt\"}"}})),
            (D, e("content_block_stop", {"index": 2})),
            (D, e("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 17}})),
            (0, e("message_stop", {}))]


class MockServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, responder=None) -> None:
        self.host, self.port = host, port
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.base_events.Server | None = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode().split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            hdrs = {k.lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
            body = await reader.readexactly(int(hdrs.get("content-length", 0)))
            req = json.loads(body) if body else {}
            if self.responder is not None:
                await self._respond(writer, await self.responder(method, path, hdrs, req))
                return
            scen = hdrs.get("x-mock-scenario", "text")
            self.requests.append({"path": path, "scenario": scen, "body": req, "t": asyncio.get_running_loop().time()})
            if scen.startswith("http_"):
                code = int(scen[5:])
                payload = json.dumps({"error": {"message": f"mock {code}"}}).encode()
                writer.write(f"HTTP/1.1 {code} Error\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                             "Connection: close\r\n\r\n".encode() + payload)
                await writer.drain()
                return
            if scen == "slow_headers":
                await asyncio.sleep(5)
            events = anthropic_events(scen) if path.startswith("/v1/messages") else openai_events(scen)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            for delay, raw in events:
                if delay:
                    await asyncio.sleep(delay)
                data = raw.encode()
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, resp: tuple) -> None:
        if resp[0] == "json":
            _, code, obj = resp
            payload = json.dumps(obj).encode()
            writer.write(f"HTTP/1.1 {code} {'OK' if code < 400 else 'Error'}\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
            return
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
                     b"Connection: close\r\n\r\n")
        await writer.drain()
        gen = resp[1]
        try:
            async for raw in gen:
                data = raw.encode()
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            await gen.aclose()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()

        async def start() -> None:
            self._server = await asyncio.start_server(self._handle, self.host, self.port)
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    def start(self) -> "MockServer":
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(5)
        return self

    def stop(self) -> None:
        """Close the listener, cancel in-flight handlers (they close their sockets while the
        loop still runs), then stop the loop."""
        if not self._loop:
            return

        async def shutdown() -> None:
            if self._server:
                self._server.close()
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(5)
        except Exception:  # noqa: BLE001 - best effort
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
