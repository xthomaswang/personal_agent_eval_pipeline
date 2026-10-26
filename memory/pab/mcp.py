"""A minimal MCP server over stdio: newline-delimited JSON-RPC 2.0, tools only.

It answers initialize, ping, tools/list and tools/call, and ignores notifications. That is all a
tool-only server needs, and it keeps the bench's own server free of an SDK dependency.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO, TextIO

SUPPORTED = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
DEFAULT_VERSION = "2025-06-18"


class ToolError(Exception):
    """A tool call the server refuses: the model gets the message as an isError result."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[[dict[str, Any]], Any]  # returns a JSON-serializable value, sent as one text block

    def listing(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


def _text(value: Any, is_error: bool = False) -> dict[str, Any]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle(msg: Any, tools: dict[str, Tool], name: str, version: str) -> dict[str, Any] | None:
    """The response to one message, or None for a notification."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        return {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                "error": {"code": -32600, "message": "invalid request"}}
    if "id" not in msg:
        return None
    method, params, rid = msg["method"], msg.get("params") or {}, msg["id"]

    def ok(result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def err(code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok({"protocolVersion": asked if asked in SUPPORTED else DEFAULT_VERSION,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": name, "version": version}})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [t.listing() for t in tools.values()]})
    if method == "tools/call":
        tool = tools.get(params.get("name", ""))
        if tool is None:
            return err(-32602, f"unknown tool: {params.get('name')}")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return ok(_text("arguments must be an object", True))
        try:
            return ok(_text(tool.fn(args)))
        except ToolError as e:
            return ok(_text(str(e), True))
    return err(-32601, f"method not found: {method}")


def serve(tools: list[Tool], name: str, version: str, stdin: BinaryIO | None = None,
          stdout: TextIO | None = None) -> None:
    """Serve until stdin closes."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout
    by_name = {t.name: t for t in tools}
    for raw in stdin:
        if not raw.strip():
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            reply: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(msg, by_name, name, version)
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
