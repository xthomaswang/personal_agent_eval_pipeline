"""Async streaming clients for OpenAI /v1/chat/completions and Anthropic /v1/messages.

Every request produces a RequestRecord with send time, TTFT, per-chunk timestamps,
usage (raw + normalized cached tokens), finish reason, assembled outputs, and errors.

TTFT definition (METHODOLOGY.md): time from send to the first SSE chunk that carries
a generated token: non-empty `content`, `reasoning_content`/`reasoning`, or any
`tool_calls` delta (OpenAI); the first non-empty text/thinking/input_json delta
(Anthropic). Role-only chunks, SSE comments/keep-alives and progress events are ignored.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .sse import SSEParser

LEAK_PATTERNS = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")


@dataclass
class ServerConfig:
    base_url: str = "http://127.0.0.1:8701"
    model: str = ""
    max_tokens_field: str = "max_tokens"
    extra_body: dict[str, Any] = field(default_factory=dict)
    stream_options: bool = True
    api_key: str = ""
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 1800.0  # 64K prefill can take minutes before the first byte
    total_timeout_s: float = 3600.0
    openai_path: str = "/v1/chat/completions"
    anthropic_path: str = "/v1/messages"
    anthropic_version: str = "2023-06-01"


@dataclass
class Chunk:
    t: float  # seconds since send
    kind: str  # content | reasoning | tool_call | text | thinking | input_json
    text: str


@dataclass
class RequestRecord:
    api: str
    url: str
    t_send_wall: float = 0.0
    t_headers: float | None = None
    ttft: float | None = None
    t_last: float | None = None
    e2e: float | None = None
    http_status: int | None = None
    chunks: list[Chunk] = field(default_factory=list)
    n_ignored: dict[str, int] = field(default_factory=lambda: {"role_only": 0, "keepalive": 0, "progress": 0, "empty": 0})
    content: str = ""
    reasoning: str = ""
    reasoning_inline: bool = False
    reasoning_fields: list[str] = field(default_factory=list)  # e.g. ["reasoning_content"] or ["reasoning"]
    t_first_data: float | None = None  # first SSE data event of any kind (diagnostic only)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage_raw: dict[str, Any] | None = None
    timings_raw: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    error: dict[str, Any] | None = None
    parse_errors: int = 0
    tool_call_mode: str = "tool_calls"  # or "structured_output" (calls parsed from JSON content)
    structured_raw: str | None = None
    structured_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["chunks"] = [[round(c.t, 6), c.kind, c.text] for c in self.chunks]
        return d

    @property
    def ok(self) -> bool:
        return self.error is None

    def markup_leak(self) -> list[str]:
        return [p for p in LEAK_PATTERNS if p in self.content]


def extract_cached_tokens(usage: dict[str, Any] | None, timings: dict[str, Any] | None = None) -> int | None:
    """Cached prompt tokens from the common server-specific usage shapes."""
    if usage:
        for path in (("prompt_tokens_details", "cached_tokens"), ("input_tokens_details", "cached_tokens"),
                     ("cached_tokens",), ("cache_read_input_tokens",), ("prompt_cache_hit_tokens",)):
            v: Any = usage
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                return int(v)
    if timings and isinstance(timings.get("cache_n"), (int, float)):  # llama.cpp
        return int(timings["cache_n"])
    return None


_THINK_RE = re.compile(r"<think>(.*?)(</think>|$)", re.S)


def split_inline_think(content: str) -> tuple[str, str, bool]:
    """Separate inline reasoning. Handles `<think>..</think>` and a bare `</think>`
    (the template already opened `<think>` in the prompt)."""
    if "<think>" in content:
        reasoning = "".join(m.group(1) for m in _THINK_RE.finditer(content))
        rest = _THINK_RE.sub("", content)
        return rest.lstrip(), reasoning.strip(), True
    if "</think>" in content:
        head, _, tail = content.partition("</think>")
        return tail.lstrip(), head.strip(), True
    return content, "", False


class _ToolAcc:
    def __init__(self) -> None:
        self.calls: dict[int, dict[str, Any]] = {}

    def add(self, deltas: list[dict[str, Any]]) -> None:
        for pos, d in enumerate(deltas):
            idx = d.get("index", pos)
            cur = self.calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
            if d.get("id") and not cur["id"]:
                cur["id"] = d["id"]
            fn = d.get("function") or {}
            name = fn.get("name")
            if name and name != cur["name"]:
                cur["name"] = name if not cur["name"] else cur["name"] + name
            args = fn.get("arguments")
            if isinstance(args, dict):  # non-compliant servers send objects
                cur["arguments"] += json.dumps(args, ensure_ascii=False)
                cur["arguments_was_object"] = True
            elif args:
                cur["arguments"] += args

    def result(self) -> list[dict[str, Any]]:
        out = []
        for idx in sorted(self.calls):
            c = dict(self.calls[idx], index=idx)
            try:
                c["arguments_parsed"] = json.loads(c["arguments"]) if c["arguments"] else {}
            except json.JSONDecodeError as e:
                c["arguments_parsed"], c["arguments_error"] = None, str(e)
            out.append(c)
        return out


def _is_progress(obj: dict[str, Any], event: str) -> bool:
    return event in ("progress", "prompt_progress") or "prompt_progress" in obj or obj.get("object") == "progress"


class StreamClient:
    """Shared httpx.AsyncClient wrapper. Use one per suite run."""

    def __init__(self, cfg: ServerConfig, max_connections: int = 8) -> None:
        self.cfg = cfg
        timeout = httpx.Timeout(connect=cfg.connect_timeout_s, read=cfg.read_timeout_s, write=60.0, pool=60.0)
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self.http = httpx.AsyncClient(base_url=cfg.base_url, timeout=timeout, headers=headers, trust_env=False,
                                      limits=httpx.Limits(max_connections=max_connections))

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> "StreamClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- OpenAI
    def openai_body(self, messages: list[dict[str, Any]], max_tokens: int, tools: list | None = None,
                    extra: dict[str, Any] | None = None, temperature: float = 0.0) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "stream": True,
                                self.cfg.max_tokens_field: max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = tools
        if self.cfg.stream_options:
            body["stream_options"] = {"include_usage": True}
        deep_merge(body, self.cfg.extra_body)
        deep_merge(body, extra or {})
        return body

    async def chat(self, body: dict[str, Any]) -> RequestRecord:
        rec = RequestRecord(api="openai", url=self.cfg.base_url + self.cfg.openai_path)
        try:
            async with asyncio.timeout(self.cfg.total_timeout_s):
                await self._chat(body, rec)
        except TimeoutError:
            rec.error = {"type": "timeout", "phase": "total", "limit_s": self.cfg.total_timeout_s}
        return rec

    async def _chat(self, body: dict[str, Any], rec: RequestRecord) -> None:
        tools = _ToolAcc()
        parser = SSEParser()
        content: list[str] = []
        reasoning: list[str] = []
        rec.t_send_wall = time.time()
        t0 = time.perf_counter()
        try:
            async with self.http.stream("POST", self.cfg.openai_path, json=body) as resp:
                rec.t_headers = time.perf_counter() - t0
                rec.http_status = resp.status_code
                if resp.status_code >= 400:
                    raw = (await resp.aread()).decode("utf-8", "replace")
                    rec.error = {"type": "http", "status": resp.status_code, "body": raw[:4000]}
                    return
                async for line in resp.aiter_lines():
                    now = time.perf_counter() - t0
                    ev = parser.feed_line(line)
                    if ev is None:
                        continue
                    if rec.t_first_data is None:
                        rec.t_first_data = now
                    if ev.data.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(ev.data)
                    except json.JSONDecodeError:
                        rec.parse_errors += 1
                        continue
                    if ev.event == "error" or "error" in obj and not obj.get("choices"):
                        rec.error = {"type": "stream_error", "detail": obj.get("error", obj)}
                        continue
                    if _is_progress(obj, ev.event):
                        rec.n_ignored["progress"] += 1
                    if obj.get("usage"):
                        rec.usage_raw = obj["usage"]
                    if obj.get("timings"):
                        rec.timings_raw = obj["timings"]
                    bearing = False
                    for ch in obj.get("choices") or []:
                        delta = ch.get("delta") or {}
                        c = delta.get("content")
                        r = delta.get("reasoning_content") or delta.get("reasoning")
                        if r:
                            fld = "reasoning_content" if delta.get("reasoning_content") else "reasoning"
                            if fld not in rec.reasoning_fields:
                                rec.reasoning_fields.append(fld)
                        tc = delta.get("tool_calls")
                        if delta.get("function_call"):
                            tc = [{"index": 0, "function": delta["function_call"]}]
                        if r:
                            reasoning.append(r)
                            rec.chunks.append(Chunk(now, "reasoning", r))
                            bearing = True
                        if c:
                            content.append(c)
                            rec.chunks.append(Chunk(now, "content", c))
                            bearing = True
                        if tc:
                            tools.add(tc)
                            frag = "".join((d.get("function") or {}).get("name") or "" for d in tc) + "".join(
                                a if isinstance(a := (d.get("function") or {}).get("arguments"), str) else json.dumps(a)
                                for d in tc if (d.get("function") or {}).get("arguments"))
                            rec.chunks.append(Chunk(now, "tool_call", frag))
                            bearing = True
                        if ch.get("finish_reason"):
                            rec.finish_reason = ch["finish_reason"]
                        if not bearing and set(delta) <= {"role", "content", "reasoning_content", "reasoning", "tool_calls"}:
                            if delta.get("role"):
                                rec.n_ignored["role_only"] += 1
                            elif not ch.get("finish_reason"):
                                rec.n_ignored["empty"] += 1
                    if bearing:
                        if rec.ttft is None:
                            rec.ttft = now
                        rec.t_last = now
                tail = parser.flush()
                if tail is not None and tail.data.strip() not in ("", "[DONE]"):
                    rec.parse_errors += 1
        except httpx.TimeoutException as e:
            rec.error = {"type": "timeout", "phase": type(e).__name__}
        except httpx.HTTPError as e:
            rec.error = {"type": "transport", "detail": f"{type(e).__name__}: {e}"}
        finally:
            rec.e2e = time.perf_counter() - t0
            rec.n_ignored["keepalive"] = parser.n_comments
            raw_content = "".join(content)
            rec.content, inline, rec.reasoning_inline = split_inline_think(raw_content)
            if rec.reasoning_inline and "inline_think" not in rec.reasoning_fields:
                rec.reasoning_fields.append("inline_think")
            rec.reasoning = "".join(reasoning) or inline
            if rec.reasoning_inline and reasoning:
                rec.reasoning = "".join(reasoning) + "\n" + inline
            rec.tool_calls = tools.result()
            u = rec.usage_raw or {}
            rec.prompt_tokens = u.get("prompt_tokens") if u else (rec.timings_raw or {}).get("prompt_n")
            rec.completion_tokens = u.get("completion_tokens") if u else (rec.timings_raw or {}).get("predicted_n")
            rec.cached_tokens = extract_cached_tokens(rec.usage_raw, rec.timings_raw)

    # ------------------------------------------------------------- Anthropic
    async def messages(self, body: dict[str, Any], headers: dict[str, str] | None = None,
                       add_model: bool = True) -> RequestRecord:
        """`add_model=False` sends the body as given (profile shaping already decided on `model`)."""
        rec = RequestRecord(api="anthropic", url=self.cfg.base_url + self.cfg.anthropic_path)
        try:
            async with asyncio.timeout(self.cfg.total_timeout_s):
                await self._messages(body, rec, headers or {}, add_model)
        except TimeoutError:
            rec.error = {"type": "timeout", "phase": "total", "limit_s": self.cfg.total_timeout_s}
        return rec

    async def _messages(self, body: dict[str, Any], rec: RequestRecord, headers: dict[str, str],
                        add_model: bool = True) -> None:
        body = dict(body, stream=True)
        if add_model and self.cfg.model:
            body.setdefault("model", self.cfg.model)
        hdrs = {"anthropic-version": self.cfg.anthropic_version, "x-api-key": self.cfg.api_key or "local", **headers}
        parser = SSEParser()
        blocks: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        rec.t_send_wall = time.time()
        t0 = time.perf_counter()
        try:
            async with self.http.stream("POST", self.cfg.anthropic_path, json=body, headers=hdrs) as resp:
                rec.t_headers = time.perf_counter() - t0
                rec.http_status = resp.status_code
                if resp.status_code >= 400:
                    raw = (await resp.aread()).decode("utf-8", "replace")
                    rec.error = {"type": "http", "status": resp.status_code, "body": raw[:4000]}
                    return
                async for line in resp.aiter_lines():
                    now = time.perf_counter() - t0
                    ev = parser.feed_line(line)
                    if ev is None:
                        continue
                    try:
                        obj = json.loads(ev.data)
                    except json.JSONDecodeError:
                        rec.parse_errors += 1
                        continue
                    typ = obj.get("type", ev.event)
                    if typ == "ping":
                        rec.n_ignored["keepalive"] += 1
                    elif typ == "error":
                        rec.error = {"type": "stream_error", "detail": obj.get("error", obj)}
                    elif typ == "message_start":
                        usage.update((obj.get("message") or {}).get("usage") or {})
                    elif typ == "content_block_start":
                        cb = dict(obj.get("content_block") or {})
                        cb.setdefault("text", "") if cb.get("type") == "text" else None
                        cb["_json"] = ""
                        blocks[obj.get("index", len(blocks))] = cb
                    elif typ == "content_block_delta":
                        d = obj.get("delta") or {}
                        b = blocks.setdefault(obj.get("index", 0), {"type": "text", "text": "", "_json": ""})
                        kind, text = {"text_delta": ("text", d.get("text")), "thinking_delta": ("thinking", d.get("thinking")),
                                      "input_json_delta": ("input_json", d.get("partial_json"))}.get(d.get("type"), (None, None))
                        if kind == "text":
                            b["text"] = b.get("text", "") + (text or "")
                        elif kind == "thinking":
                            b["thinking"] = b.get("thinking", "") + (text or "")
                        elif kind == "input_json":
                            b["_json"] += text or ""
                        if kind and text:
                            rec.chunks.append(Chunk(now, kind, text))
                            if rec.ttft is None:
                                rec.ttft = now
                            rec.t_last = now
                    elif typ == "message_delta":
                        rec.finish_reason = (obj.get("delta") or {}).get("stop_reason") or rec.finish_reason
                        usage.update(obj.get("usage") or {})
                    else:
                        rec.n_ignored["empty"] += 1
        except httpx.TimeoutException as e:
            rec.error = {"type": "timeout", "phase": type(e).__name__}
        except httpx.HTTPError as e:
            rec.error = {"type": "transport", "detail": f"{type(e).__name__}: {e}"}
        finally:
            rec.e2e = time.perf_counter() - t0
            rec.n_ignored["keepalive"] += parser.n_comments
            for idx in sorted(blocks):
                b = blocks[idx]
                if b.get("type") == "text":
                    rec.content += b.get("text", "")
                elif b.get("type") == "thinking":
                    rec.reasoning += b.get("thinking", "")
                    if b.get("thinking") and "thinking_block" not in rec.reasoning_fields:
                        rec.reasoning_fields.append("thinking_block")
                elif b.get("type") == "tool_use":
                    raw = b["_json"]
                    call = {"id": b.get("id"), "name": b.get("name", ""), "arguments": raw, "index": idx}
                    try:
                        call["arguments_parsed"] = json.loads(raw) if raw else (b.get("input") or {})
                    except json.JSONDecodeError as e:
                        call["arguments_parsed"], call["arguments_error"] = None, str(e)
                    rec.tool_calls.append(call)
            rec.usage_raw = usage or None
            rec.prompt_tokens = usage.get("input_tokens")
            rec.completion_tokens = usage.get("output_tokens")
            rec.cached_tokens = extract_cached_tokens(usage)


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


async def run_concurrent(coros_factories: list, ) -> tuple[list[RequestRecord], float]:
    """Start several requests at the same moment (asyncio.Barrier). Each factory is a
    zero-arg callable returning an awaitable RequestRecord. Returns (records, start
    spread in seconds between the first and last send)."""
    barrier = asyncio.Barrier(len(coros_factories))
    starts: list[float] = []

    async def one(f):
        await barrier.wait()
        starts.append(time.perf_counter())
        return await f()

    recs = await asyncio.gather(*(one(f) for f in coros_factories))
    return list(recs), (max(starts) - min(starts)) if starts else 0.0
