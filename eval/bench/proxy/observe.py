"""Watch an upstream response on its way to the client.

The observer sees the same bytes the client gets and extracts what the call
record needs: usage, tool-call names, stop reason, the amount of generated
text and thinking, and the time of the first generated token. It never keeps
generated text.
"""

from __future__ import annotations

import json
from typing import Any

from .sse import SSEEvent, SSEParser

# oMLX keepalive frames (sse_keepalive_mode="chunk") are a no-op chat chunk
# for OpenAI and "event: ping" for Anthropic. SSE comment lines (mode
# "comment") count as keepalives too. Non-streaming oMLX responses start with
# keepalive spaces before the JSON body.
OPENAI_KEEPALIVE_IDS = frozenset({"chatcmpl-keepalive"})
OPENAI_KEEPALIVE_MODEL = "keepalive"

MODEL_ENDPOINTS = frozenset({"chat_completions", "messages"})

# A non-streaming body larger than this is relayed but not parsed.
MAX_PARSED_BODY = 32 * 1024 * 1024
MAX_TEXT = 300


def clip(text: str, limit: int = MAX_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def summarize_error(obj: Any, status: int | None, source: str = "upstream") -> dict[str, Any]:
    """Type and message of an error body, without echoing request content.

    FastAPI validation errors (HTTP 422) carry an ``input`` member with the
    offending part of the request, i.e. possibly message content, so only
    ``type``, ``loc`` and ``msg`` are kept.
    """
    out: dict[str, Any] = {"source": source}
    if status is not None:
        out["status"] = status
    if not isinstance(obj, dict):
        return out
    err = obj.get("error")
    if isinstance(err, dict):
        if isinstance(err.get("type"), str):
            out["type"] = err["type"]
        if err.get("code") is not None:
            out["code"] = clip(str(err["code"]), 100)
        if err.get("message") is not None:
            out["message"] = clip(str(err["message"]))
    elif isinstance(err, str):
        out["message"] = clip(err)
    detail = obj.get("detail")
    if isinstance(detail, str):
        out["message"] = clip(detail)
    elif isinstance(detail, list):
        out["type"] = "validation_error"
        details = []
        for item in detail[:10]:
            if not isinstance(item, dict):
                continue
            entry: dict[str, Any] = {}
            if isinstance(item.get("type"), str):
                entry["type"] = item["type"]
            if isinstance(item.get("loc"), list):
                entry["loc"] = [x for x in item["loc"] if isinstance(x, (str, int))]
            if item.get("msg") is not None:
                entry["msg"] = clip(str(item["msg"]))
            details.append(entry)
        out["details"] = details
    return out


def normalize_usage(protocol: str | None, raw: Any) -> dict[str, int | None] | None:
    """Map OpenAI or Anthropic usage onto prompt/completion/cached tokens.

    With prefix caching on, oMLX splits Anthropic input usage into the
    disjoint fields input_tokens + cache_creation_input_tokens +
    cache_read_input_tokens; their sum is the prompt length.
    """
    if not isinstance(raw, dict):
        return None
    if protocol == "openai":
        details = raw.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        return {
            "prompt_tokens": _int(raw.get("prompt_tokens")),
            "completion_tokens": _int(raw.get("completion_tokens")),
            "cached_tokens": _int(cached),
        }
    parts = [
        _int(raw.get(key))
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    ]
    known = [p for p in parts if p is not None]
    return {
        "prompt_tokens": sum(known) if known else None,
        "completion_tokens": _int(raw.get("output_tokens")),
        "cached_tokens": _int(raw.get("cache_read_input_tokens")),
    }


class ResponseObserver:
    def __init__(self, protocol: str | None, endpoint: str, status: int, content_type: str) -> None:
        self.protocol = protocol
        self.endpoint = endpoint
        self.status = status
        self.is_sse = "text/event-stream" in content_type.lower()
        self._parser = SSEParser() if self.is_sse else None
        # Non-SSE bodies are kept for parsing when they are model responses
        # or errors; other passthrough bodies are only relayed.
        self._keep_body = not self.is_sse and (protocol is not None or status >= 400)
        self._body = bytearray()
        self._body_too_large = False
        self._last_time: tuple[float | None, float | None] = (None, None)

        self.events = 0
        self.keepalives = 0
        self.parse_errors = 0
        self.complete = False  # SSE end marker ([DONE] / message_stop) seen
        self.usage_raw: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self.text_chars = 0
        self.reasoning_chars = 0
        self.model: str | None = None
        self.error: dict[str, Any] | None = None
        self.counted_input_tokens: int | None = None
        # (wall, monotonic) of the first generated token of any kind, and of
        # the first token that is not thinking (text or a tool call).
        self.first_output: tuple[float, float] | None = None
        self.first_content: tuple[float, float] | None = None
        self.first_output_kind: str | None = None
        self._tool_names: list[str | None] = []
        self._stream_tools: dict[tuple[Any, Any], str | None] = {}

    # -- feeding --------------------------------------------------------

    def feed(self, chunk: bytes, wall: float, mono: float) -> None:
        self._last_time = (wall, mono)
        if self._parser is not None:
            for event in self._parser.feed(chunk):
                self._on_event(event, wall, mono)
        elif self._keep_body:
            if len(self._body) + len(chunk) <= MAX_PARSED_BODY:
                self._body += chunk
            else:
                self._body_too_large = True

    def finish(self) -> None:
        if self._parser is not None:
            wall, mono = self._last_time
            for event in self._parser.close():
                self._on_event(event, wall or 0.0, mono or 0.0)
        elif self._keep_body:
            self._parse_body()

    def results(self) -> dict[str, Any]:
        model_call = self.endpoint in MODEL_ENDPOINTS
        names = list(self._stream_tools.values()) if self._stream_tools else list(self._tool_names)
        return {
            "usage": normalize_usage(self.protocol, self.usage_raw) if model_call else None,
            "usage_raw": self.usage_raw if model_call else None,
            "tool_calls": len(names) if model_call else None,
            "tool_names": names if model_call else None,
            "stop_reason": self.stop_reason,
            "output_text_chars": self.text_chars if model_call else None,
            "output_reasoning_chars": self.reasoning_chars if model_call else None,
            "counted_input_tokens": self.counted_input_tokens,
            "sse_events": self.events if self.is_sse else None,
            "sse_keepalives": self.keepalives if self.is_sse else None,
            "stream_complete": self.complete if self.is_sse else None,
        }

    # -- helpers --------------------------------------------------------

    def _mark(self, wall: float, mono: float, kind: str) -> None:
        """Note generated output of kind "reasoning", "text" or "tool"."""
        if self.first_output is None:
            self.first_output = (wall, mono)
            self.first_output_kind = kind
        if kind != "reasoning" and self.first_content is None:
            self.first_content = (wall, mono)

    @property
    def ttft_valid(self) -> bool | None:
        """Whether first_output is a real time to first token.

        oMLX sends tool calls only after generation has finished, so when a
        response starts with a tool call (no thinking or text streamed
        first), the first output arrives at the end and says nothing about
        the first token. None when nothing was generated or not streamed.
        """
        if self.first_output_kind is None:
            return None
        return self.first_output_kind != "tool"

    def _parse_body(self) -> None:
        if self._body_too_large:
            if self.status >= 400:
                self.error = {"source": "upstream", "status": self.status, "type": "error_body_too_large"}
            return
        text = bytes(self._body).strip()  # drops oMLX's leading keepalive spaces
        if not text:
            if self.status >= 400:
                self.error = {"source": "upstream", "status": self.status}
            return
        try:
            obj = json.loads(text)
        except ValueError:
            if self.status >= 400:
                message = clip(text.decode("utf-8", "replace"))
                self.error = {"source": "upstream", "status": self.status, "message": message}
            elif self.endpoint in MODEL_ENDPOINTS:
                self.error = {"source": "upstream", "status": self.status, "type": "invalid_json"}
            return
        if self.status >= 400:
            self.error = summarize_error(obj, self.status)
            return
        if not isinstance(obj, dict):
            return
        if self.endpoint == "chat_completions":
            self._openai_completion(obj)
        elif self.endpoint == "messages":
            self._anthropic_message(obj)
        elif self.endpoint == "count_tokens":
            self.counted_input_tokens = _int(obj.get("input_tokens"))

    def _openai_completion(self, obj: dict[str, Any]) -> None:
        if isinstance(obj.get("model"), str):
            self.model = obj["model"]
        if isinstance(obj.get("usage"), dict):
            self.usage_raw = obj["usage"]
        if obj.get("error"):
            self.error = summarize_error(obj, self.status)
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    self.text_chars += len(content)
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                if isinstance(reasoning, str):
                    self.reasoning_chars += len(reasoning)
                for call in message.get("tool_calls") or []:
                    if isinstance(call, dict):
                        function = call.get("function")
                        self._tool_names.append(function.get("name") if isinstance(function, dict) else None)
            if choice.get("finish_reason") and self.stop_reason is None:
                self.stop_reason = choice["finish_reason"]

    def _anthropic_message(self, obj: dict[str, Any]) -> None:
        if obj.get("type") == "error":
            self.error = summarize_error(obj, self.status)
            return
        if isinstance(obj.get("model"), str):
            self.model = obj["model"]
        if isinstance(obj.get("usage"), dict):
            self.usage_raw = obj["usage"]
        self.stop_reason = obj.get("stop_reason")
        for block in obj.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                self.text_chars += len(block.get("text") or "")
            elif kind == "thinking":
                self.reasoning_chars += len(block.get("thinking") or "")
            elif kind == "tool_use":
                self._tool_names.append(block.get("name"))

    # -- SSE --------------------------------------------------------------

    def _on_event(self, event: SSEEvent, wall: float, mono: float) -> None:
        if event.comment:
            self.keepalives += 1
            return
        data = event.data.strip()
        if not data:
            return
        if self.protocol == "openai" and data == "[DONE]":
            self.complete = True
            return
        try:
            obj = json.loads(data)
        except ValueError:
            self.parse_errors += 1
            return
        if not isinstance(obj, dict):
            return
        if self.protocol == "openai":
            self._openai_event(obj, wall, mono)
        elif self.protocol == "anthropic":
            self._anthropic_event(obj, event.event, wall, mono)
        else:
            self.events += 1

    def _openai_event(self, obj: dict[str, Any], wall: float, mono: float) -> None:
        if obj.get("id") in OPENAI_KEEPALIVE_IDS or obj.get("model") == OPENAI_KEEPALIVE_MODEL:
            self.keepalives += 1
            return
        self.events += 1
        if obj.get("error"):
            self.error = summarize_error(obj, None, source="upstream_stream")
        if self.model is None and isinstance(obj.get("model"), str):
            self.model = obj["model"]
        if isinstance(obj.get("usage"), dict):
            self.usage_raw = obj["usage"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    self.text_chars += len(content)
                    self._mark(wall, mono, "text")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    self.reasoning_chars += len(reasoning)
                    self._mark(wall, mono, "reasoning")
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    # The name arrives in the first delta of each tool call;
                    # later deltas for the same index carry argument pieces.
                    key = (choice.get("index", 0), call.get("index", call.get("id", len(self._stream_tools))))
                    function = call.get("function")
                    name = function.get("name") if isinstance(function, dict) else None
                    if not self._stream_tools.get(key):
                        self._stream_tools[key] = name or None
                    self._mark(wall, mono, "tool")
            if choice.get("finish_reason"):
                self.stop_reason = choice["finish_reason"]

    def _anthropic_event(self, obj: dict[str, Any], event_name: str | None, wall: float, mono: float) -> None:
        kind = obj.get("type") or event_name
        if kind == "ping":
            self.keepalives += 1
            return
        self.events += 1
        if kind == "message_start":
            message = obj.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("model"), str):
                    self.model = message["model"]
                self._merge_usage(message.get("usage"))
        elif kind == "content_block_start":
            block = obj.get("content_block")
            if not isinstance(block, dict):
                return
            block_type = block.get("type")
            if block_type == "tool_use":
                self._tool_names.append(block.get("name"))
                self._mark(wall, mono, "tool")
            elif block_type == "text" and block.get("text"):
                self.text_chars += len(block["text"])
                self._mark(wall, mono, "text")
            elif block_type == "thinking" and block.get("thinking"):
                self.reasoning_chars += len(block["thinking"])
                self._mark(wall, mono, "reasoning")
        elif kind == "content_block_delta":
            delta = obj.get("delta")
            if not isinstance(delta, dict):
                return
            delta_type = delta.get("type")
            if delta_type == "text_delta" and delta.get("text"):
                self.text_chars += len(delta["text"])
                self._mark(wall, mono, "text")
            elif delta_type == "thinking_delta" and delta.get("thinking"):
                self.reasoning_chars += len(delta["thinking"])
                self._mark(wall, mono, "reasoning")
            elif delta_type == "input_json_delta" and delta.get("partial_json"):
                self._mark(wall, mono, "tool")
        elif kind == "message_delta":
            delta = obj.get("delta")
            if isinstance(delta, dict) and delta.get("stop_reason"):
                self.stop_reason = delta["stop_reason"]
            # message_start carries an estimate; message_delta the final counts.
            self._merge_usage(obj.get("usage"))
        elif kind == "message_stop":
            self.complete = True
        elif kind == "error":
            self.error = summarize_error(obj, None, source="upstream_stream")

    def _merge_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        merged = dict(self.usage_raw or {})
        merged.update({k: v for k, v in usage.items() if v is not None})
        self.usage_raw = merged
