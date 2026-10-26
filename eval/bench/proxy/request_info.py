"""Classify requests and summarize request bodies without keeping content."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import Any

from .images import describe_image

# Endpoints whose bodies the proxy understands. Everything else is relayed
# untouched and recorded with protocol=None, endpoint="other".
ENDPOINTS: dict[str, tuple[str | None, str]] = {
    "/v1/chat/completions": ("openai", "chat_completions"),
    "/v1/messages": ("anthropic", "messages"),
    "/v1/messages/count_tokens": ("anthropic", "count_tokens"),
    # Claude Code probes this on startup; the proxy answers it locally.
    "/api/hello": (None, "hello"),
}

# Requests that make the model generate. Only these get a role, a sampling
# policy and an images_new count; count_tokens and hello are never steps.
GENERATION_ENDPOINTS = frozenset({"chat_completions", "messages"})

# Endpoints whose thinking settings the proxy may rewrite.
THINKING_ENDPOINTS = frozenset({"chat_completions", "messages", "count_tokens"})

# Content-part types that carry an image. Anthropic images can also sit inside
# a tool_result block's content list (Claude Code's Read tool returns them
# that way), so the walk recurses through nested "content" lists.
IMAGE_TYPES = {
    "openai": frozenset({"image_url", "input_image"}),
    "anthropic": frozenset({"image"}),
}


def classify(path: str) -> tuple[str | None, str]:
    """Return ``(protocol, endpoint)`` for a request path."""
    return ENDPOINTS.get(path, (None, "other"))


def iter_image_blocks(messages: list[Any], image_types: frozenset[str]) -> Iterator[dict[str, Any]]:
    def walk(content: Any) -> Iterator[dict[str, Any]]:
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in image_types:
                yield block
            yield from walk(block.get("content"))

    for message in messages:
        if isinstance(message, dict):
            yield from walk(message.get("content"))


def count_images(messages: list[Any], image_types: frozenset[str]) -> int:
    return sum(1 for _ in iter_image_blocks(messages, image_types))


def request_tool_names(tools: Any) -> list[str]:
    """Sorted tool names of a request (OpenAI function tools or Anthropic tools)."""
    names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else tool.get("name")
            if isinstance(name, str):
                names.append(name)
    return sorted(names)


def tool_names_sha256(names: Iterable[str]) -> str:
    """sha256 of the sorted names as a compact JSON array, e.g. '["Bash","Read"]'.

    Use this function wherever a tool list is compared with this hash (for
    example an allowlist), so the canonical form stays the same.
    """
    canonical = json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tools_raw_sha256(tools: Any) -> str | None:
    """sha256 of the tools array as sent: request order, key order and schemas included.

    Unlike ``tool_names_sha256``, it changes when a harness reorders or edits its tools between
    calls, which also invalidates the server's prefix cache from the tools onward.
    """
    if not isinstance(tools, list):
        return None
    raw = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def request_role(endpoint: str, body: dict[str, Any]) -> str | None:
    """``main`` for a generation request with a non-empty tools list, else ``aux``.

    Other endpoints (count_tokens, hello, passthrough paths) have no role.
    """
    if endpoint not in GENERATION_ENDPOINTS:
        return None
    tools = body.get("tools")
    return "main" if isinstance(tools, list) and tools else "aux"


def summarize_request(protocol: str, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    """Counts, names and hashes of a model request; never message content."""
    messages = body.get("messages")
    tools = body.get("tools")
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    model = body.get("model")
    names = request_tool_names(tools)
    summary: dict[str, Any] = {
        "model": model if isinstance(model, str) else None,
        "stream": body.get("stream") is True,
        "role": request_role(endpoint, body),
        "messages": len(messages) if isinstance(messages, list) else None,
        "tools": len(tools) if isinstance(tools, list) else 0,
        "request_tool_names": names,
        "request_tools_sha256": tool_names_sha256(names),
        "request_tools_raw_sha256": tools_raw_sha256(tools),
        "max_tokens": max_tokens if isinstance(max_tokens, int) else None,
    }
    if isinstance(messages, list):
        infos = [describe_image(block) for block in iter_image_blocks(messages, IMAGE_TYPES[protocol])]
        summary["images"] = len(infos)
        summary["image_sha256"] = [sha for sha, _ in infos]
        summary["image_wh"] = [list(wh) if wh else None for _, wh in infos]
    return summary
