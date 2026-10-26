"""Structured-output tool calling for servers without a Qwen tool-call parser (MLC-LLM).

Instead of `tools`, the request carries the tool schemas inlined at the start of the system
message plus an instruction to answer with one JSON object, and a `response_format` whose
schema is a union over `{"tool": <const name>, "arguments": <that tool's parameters>}`,
`{"tool_calls": [...]}` (parallel calls) and `{"final": <string>}`. The server enforces it
with a grammar (xgrammar in MLC). The JSON reply is parsed back into the normal
`tool_calls` record, so gate / M4 / M7 work unchanged. Records are marked
`tool_call_mode: "structured_output"`.

Two facts from MLC's sources (mlc-llm main 8978ea9a, xgrammar d4f57c4) set the defaults:
- MLC's `response_format.type` is `Literal["text", "json_object"]` and the schema is a JSON
  *string* in `schema`; the OpenAI `json_schema` shape fails validation. Default style
  `mlc` = `{"type": "json_object", "schema": "<json>"}`; `openai` and `none` are selectable.
- The pinned xgrammar does not implement `oneOf` (a top-level `oneOf` compiles to "any
  JSON"); it does implement `anyOf`, `const` and `$ref` to `#/$defs/...`. The branches are
  mutually exclusive (distinct `tool` constants), so `anyOf` is equivalent and the default.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

STEP_INSTRUCTION = (
    "Reply with exactly one JSON object and nothing else:\n"
    "- {\"tool\": \"<tool name>\", \"arguments\": {...}} to call one tool;\n"
    "- {\"tool_calls\": [{\"tool\": \"<tool name>\", \"arguments\": {...}}, ...]} to call several independent "
    "tools at once;\n"
    "- {\"final\": \"<answer>\"} to answer without calling a tool.")
_NONCE_RE = re.compile(r"^\[run (\d{12})\] ")


def _strip_descriptions(schema: Any) -> Any:
    """Grammar schemas do not need descriptions; dropping them keeps the grammar small and
    independent of the nonce (which lives in a description)."""
    if isinstance(schema, dict):
        return {k: _strip_descriptions(v) for k, v in schema.items() if k != "description"}
    if isinstance(schema, list):
        return [_strip_descriptions(v) for v in schema]
    return schema


def agent_step_schema(tools: list[dict[str, Any]], union: str = "anyOf", parallel: bool = True) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    refs = []
    for t in tools:
        fn = t["function"]
        key = "call_" + fn["name"]
        defs[key] = {"type": "object", "properties": {"tool": {"const": fn["name"]},
                                                      "arguments": _strip_descriptions(fn["parameters"])},
                     "required": ["tool", "arguments"], "additionalProperties": False}
        refs.append({"$ref": f"#/$defs/{key}"})
    branches = list(refs)
    branches.append({"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"],
                     "additionalProperties": False})
    if parallel:
        branches.append({"type": "object", "properties": {"tool_calls": {"type": "array", "items": {union: refs},
                                                                         "minItems": 2}},
                         "required": ["tool_calls"], "additionalProperties": False})
    return {"$defs": defs, union: branches}


def response_format(schema: dict[str, Any], style: str) -> dict[str, Any] | None:
    if style == "mlc":
        return {"type": "json_object", "schema": json.dumps(schema, ensure_ascii=False, separators=(",", ":"))}
    if style == "openai":
        return {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema, "strict": True}}
    if style == "none":
        return None
    raise ValueError(f"unknown structured-output style {style!r}")


def inline_tools_text(tools: list[dict[str, Any]]) -> str:
    listing = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                "parameters": t["function"]["parameters"]} for t in tools]
    return ("# Tools\n\nYou can call the following tools. Each entry gives the tool's name, its description and the "
            "JSON schema of its arguments:\n\n" + json.dumps(listing, ensure_ascii=False, separators=(",", ":"))
            + "\n\n" + STEP_INSTRUCTION)


def step_object(calls: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for c in calls:
        fn = c.get("function", c)
        args = fn.get("arguments")
        items.append({"tool": fn["name"], "arguments": json.loads(args) if isinstance(args, str) else (args or {})})
    return items[0] if len(items) == 1 else {"tool_calls": items}


def _text(content: Any) -> Any:
    if isinstance(content, list) and all(p.get("type") == "text" for p in content):
        return "".join(p["text"] for p in content)
    return content


def flatten_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No `tool` messages and no assistant `tool_calls`: calls become the JSON step object
    the grammar would produce, tool results become user messages."""
    out: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                names[tc.get("id", "")] = tc.get("function", tc).get("name", "?")
            out.append({"role": "assistant", "content": json.dumps(step_object(m["tool_calls"]), ensure_ascii=False)})
        elif m.get("role") == "tool":
            head = f"Tool result ({names.get(m.get('tool_call_id', ''), 'tool')}):\n"
            content = _text(m.get("content"))
            if isinstance(content, str):
                out.append({"role": "user", "content": head + content})
            else:
                out.append({"role": "user", "content": [{"type": "text", "text": head}] + list(content or [])})
        else:
            out.append(dict(m))
    return out


def to_structured_request(body: dict[str, Any], style: str = "mlc", union: str = "anyOf",
                          parallel: bool = True) -> tuple[dict[str, Any], list[str]]:
    b = copy.deepcopy(body)
    notes: list[str] = []
    tools = b.pop("tools", None)
    for k in ("tool_choice", "parallel_tool_calls"):
        b.pop(k, None)
    msgs = b.get("messages", [])
    if any(m.get("role") == "tool" or m.get("tool_calls") for m in msgs):
        msgs = flatten_messages(msgs)
        notes.append("structured output: tool history flattened (calls as JSON step objects, results as user messages)")
    if tools:
        nonce = None
        first = tools[0]["function"]
        m = _NONCE_RE.match(first.get("description", ""))
        if m:
            nonce = m.group(1)
            first["description"] = first["description"][m.end():]
        head = (f"[run {nonce}]\n" if nonce else "") + inline_tools_text(tools)
        if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
            msgs = [dict(msgs[0], content=head + "\n\n" + msgs[0]["content"])] + msgs[1:]
        else:
            msgs = [{"role": "system", "content": head}] + msgs
        rf = response_format(agent_step_schema(tools, union, parallel), style)
        if rf is not None:
            b["response_format"] = rf
        notes.append(f"structured output: {len(tools)} tools inlined into the system message, `tools` not sent, "
                     f"response_format={style} ({union})" + ("; nonce moved to the start" if nonce else ""))
    b["messages"] = msgs
    return b, notes


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse_step(content: str) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """(tool_calls, final_text, error) from a structured reply."""
    text = content.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return [], None, f"content is not JSON: {e}"
    if not isinstance(obj, dict):
        return [], None, "content is not a JSON object"
    if isinstance(obj.get("final"), str) and "tool" not in obj and "tool_calls" not in obj:
        return [], obj["final"], None
    raw = [obj] if "tool" in obj else obj.get("tool_calls") if isinstance(obj.get("tool_calls"), list) else None
    if raw is None:
        return [], None, f"unexpected JSON shape (keys: {sorted(obj)})"
    calls = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or not isinstance(c.get("tool"), str):
            return [], None, f"tool call {i} has no tool name"
        args = c.get("arguments")
        call = {"id": None, "name": c["tool"], "arguments": json.dumps(args, ensure_ascii=False), "index": i,
                "arguments_parsed": args if isinstance(args, dict) else None}
        if not isinstance(args, dict):
            call["arguments_error"] = "arguments is not a JSON object"
        calls.append(call)
    return calls, None, None


def apply_to_record(rec) -> None:
    """Replace the record's content with the parsed structured reply."""
    rec.structured_raw = rec.content
    calls, final, err = parse_step(rec.content)
    if err:
        rec.structured_error = err
        return
    rec.tool_calls = calls
    rec.content = final or ""
