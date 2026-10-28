"""Scripted HTTP model for the native T4 smoke: MCP ingestion, retrieval and null probe answers.

This is a plumbing check, not a model-quality measurement. It never consults gold probe answers.
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator


def observed_fact(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] != "tool":
            continue
        data = json.loads(message["content"])
        if not isinstance(data, dict) or not data.get("results"):
            continue
        record = data["results"][0]
        if record["kind"] == "event":
            location = f" at {record['location']}" if record.get("location") else ""
            fact = f"{record['title']} starts {record['start']}{location}."
        else:
            fact = f"{record.get('subject') or record.get('sender')}: {record.get('body') or record.get('text')}"
        return " ".join(fact.split())[:190]
    raise ValueError("submit must follow an actual feed_day result")


def response(body: dict) -> dict:
    messages = body["messages"]
    tools = {t["function"]["name"] for t in body.get("tools", [])}
    text = json.dumps(messages)
    tool, args = None, {}
    if str(messages[0].get("content", "")).startswith("You answer questions about"):
        content = '{"answer": null}'
    elif not tools and "Return at most" in str(messages[-1].get("content", "")):
        content = json.dumps({"ids": list(dict.fromkeys(re.findall(r"\[(m\d+)\]", text)))})
    elif tools:
        used = {c["function"]["name"] for m in messages for c in m.get("tool_calls", [])}
        feed = next(n for n in tools if n.endswith("__feed_day"))
        submit = next(n for n in tools if n.endswith("__submit"))
        if feed not in used:
            tool, content = feed, "I will read today's records."
        elif submit not in used:
            tool, args = submit, {"status": "done"}
            content = f"<note>\nplan: Submit today's session.\nfound: {observed_fact(messages)}\nremember: yes\n</note>"
        else:
            content = "Done."
    else:
        content = json.dumps({"progress_summary": "scripted", "task_still_feasible": False,
                              "next_step_is_concrete": False, "next_step_is_novel": False,
                              "recommended_action": "stop", "rationale": "scripted stop"})
    message = {"role": "assistant", "content": content, "reasoning_content": "Scripted native T4 smoke."}
    if tool:
        message["tool_calls"] = [{"id": f"call-{len(messages)}", "type": "function",
                                  "function": {"name": tool, "arguments": json.dumps(args)}}]
    return {"id": f"chatcmpl-stub-{len(messages)}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "scripted-native-t4"),
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


@contextmanager
def server(log: Path) -> Iterator[str]:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            with lock, log.open("a") as out:
                out.write(json.dumps(body) + "\n")
            data = json.dumps(response(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            data = json.dumps({"object": "list", "data": [{"id": "scripted-native-t4", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
