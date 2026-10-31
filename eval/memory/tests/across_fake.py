"""A stand-in for localmelo's console API and for an oMLX server, for the across-chats tests (stdlib only).

``FakeGateway`` serves the routes the driver uses (health, create a chat with its first turn, a turn in a
chat, the last turn, the event stream, delete a chat) on a loopback port, with a bearer token, and keeps a
memory under ``<home>/.cache/localmelo/memory/memory.json`` like the real thing keeps files there: a
statement (any message that is not a question) becomes a record, a finished turn becomes a document of
its chat, and a question recalls records and the documents of other chats that share enough words. The
arm comes from the ``-X`` settings it is started with. ``melo_main`` is the same thing behind the
``melo`` command line, for tests that start a real subprocess.

With ``store=True`` it also keeps what the real gateway keeps in SQLite and the in-chat checks read: a working window
per chat (the ``[agent] window_messages`` of the cell's config.toml, default 64, dropped in chunks of 8 down from the cap) in the ``windows`` table of
``gateway/sessions.db`` next to the stored ``events`` (``run.finished`` carries the cap), and a ``long_term`` row per
note in ``memory/records.db``. ``tool_words`` makes turns whose text holds a key call that read-tier tool (four
messages instead of two), ``save_only`` limits which statements become notes, and ``answers`` scripts replies. A
question the memory does not answer is answered from the chat's own window when a user message in it matches.
``memory.context_query=on`` makes its recall looser (one shared word is enough, so a pronoun follow-up finds its
note); ``new_build=True`` reports ``recall.context`` and ``recall.block_chars`` as newer localmelo builds do.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import sqlite3
import sys
import threading
import time
import tomllib
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
STOP_WORDS = {"what", "when", "where", "which", "who", "the", "and", "for", "are", "was", "does", "have", "with", "from"}
TOKEN = "fake-token"
KNOWN_SETTINGS = {"memory.retriever": "auto", "memory.across_chats": "true", "agent.window_messages": "64", "memory.context_query": "off"}


def terms(text: str) -> set[str]:
    words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP_WORDS}
    cjk = re.findall(r"[一-鿿]+", text)
    return words | {run[i : i + 2] for run in cjk for i in range(len(run) - 1)}


class FakeGateway:
    def __init__(
        self, home: Path, x_pairs: Sequence[str] = (), *, port: int = 0, delay: float = 0.01,
        recall_timings: bool = True, synonyms: Mapping[str, str] | None = None, drop_stream: bool = False,
        busy_deletes: int = 0, fail_texts: Sequence[str] = (), token: str = TOKEN, model_calls: bool = False,
        approval_texts: Sequence[str] = (), hang_texts: Sequence[str] = (), hang_seconds: float = 3.0,
        store: bool = False, tool_words: Mapping[str, str] | None = None, save_only: Sequence[str] | None = None,
        answers: Mapping[str, str] | None = None, new_build: bool = False,
    ) -> None:
        settings = dict(p.split("=", 1) for p in x_pairs)
        self.retriever = settings.get("memory.retriever", "auto")
        self.across = settings.get("memory.across_chats", "true") != "false"
        if "agent.window_messages" in settings:  # as `melo serve` answers it: the cap is a config.toml entry for a gateway
            raise ValueError("melo: error: only for direct mode (a query without --serve): --window-messages")
        self.store, self.tool_words, self.save_only = store, dict(tool_words or {}), None if save_only is None else tuple(save_only)
        self.answers = dict(answers or {})
        self.context_query = settings.get("memory.context_query")  # None: the build has no such setting
        self.new_build = new_build
        self.x_pairs = tuple(x_pairs)
        self.home = Path(home)
        self.delay, self.recall_timings, self.synonyms = delay, recall_timings, dict(synonyms or {})
        self.drop_stream, self.busy_deletes, self.fail_texts = drop_stream, busy_deletes, tuple(fail_texts)
        self.token = token
        self.model_calls, self.approval_texts = model_calls, tuple(approval_texts)
        self.hang_texts, self.hang_seconds = tuple(hang_texts), hang_seconds
        self.approvals: dict[str, dict[str, Any]] = {}
        self.cancels: dict[str, threading.Event] = {}
        self.chat_url = self.embedding_url = ""
        self.chat_model = self.embedding_model = ""
        config = self.home / ".cache" / "localmelo" / "config.toml"
        document = tomllib.loads(config.read_text()) if config.exists() else {}
        self.window_cap = int(document.get("agent", {}).get("window_messages", 64))
        if model_calls and config.exists():  # the model calls go where the driver's config says, as localmelo's do
            section = document.get("vllm", {})
            self.chat_url, self.chat_model = section.get("chat_url", ""), section.get("chat_model", "")
            self.embedding_url, self.embedding_model = section.get("embedding_url", ""), section.get("embedding_model", "")
        self.stopped = False
        self.sessions: dict[str, dict[str, Any]] = {}
        self.created = 0
        self.requests: list[tuple[str, str]] = []
        self.lock = threading.Condition()
        self.memory_path = self.home / ".cache" / "localmelo" / "memory" / "memory.json"
        auth = self.home / ".cache" / "localmelo" / "gateway" / "auth"
        auth.mkdir(parents=True, exist_ok=True)
        (auth / "token").write_text(token)
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # the event stream is chunked, as the real one is

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                gateway.handle(self, "GET")

            def do_POST(self) -> None:
                gateway.handle(self, "POST")

            def do_DELETE(self) -> None:
                gateway.handle(self, "DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            self.server.shutdown()
            self.server.server_close()

    # --- the SQLite store (store=True) ---

    @property
    def sessions_db(self) -> Path:
        return self.home / ".cache" / "localmelo" / "gateway" / "sessions.db"

    @property
    def records_db(self) -> Path:
        return self.home / ".cache" / "localmelo" / "memory" / "records.db"

    SCHEMAS = {
        "sessions": (
            "CREATE TABLE IF NOT EXISTS windows (session_id TEXT PRIMARY KEY, turn INTEGER NOT NULL, version INTEGER NOT NULL, "
            "data TEXT NOT NULL, created_at REAL NOT NULL, thinking_since REAL)",
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, turn INTEGER, "
            "type TEXT NOT NULL, event TEXT NOT NULL)"),
        "records": (
            "CREATE TABLE IF NOT EXISTS long_term (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
            "kind TEXT NOT NULL DEFAULT 'profile', source TEXT NOT NULL DEFAULT 'user', status TEXT NOT NULL DEFAULT 'active', "
            "session_id TEXT NOT NULL DEFAULT '')",),
    }

    def sql(self, db: Path, statement: str, args: Sequence[Any] = ()) -> None:
        """One write on its own connection, closed at once, so the file is whole on disk for a read-only reader."""
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        try:
            for create in self.SCHEMAS["sessions" if db == self.sessions_db else "records"]:
                conn.execute(create)
            conn.execute(statement, tuple(args))
            conn.commit()
        finally:
            conn.close()

    # --- memory ---

    def load(self) -> dict[str, list[dict[str, Any]]]:
        if self.memory_path.exists():
            return json.loads(self.memory_path.read_text())
        return {"records": [], "documents": []}

    def save(self, memory: Mapping[str, Any]) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(memory))

    def recall(self, text: str, session_id: str, memory: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not self.across or self.retriever == "off":
            return []
        query = terms(text)
        if self.retriever == "embedding":
            query |= {extra for t in list(query) for extra in terms(self.synonyms.get(t, ""))}
        need = max(1, int(0.6 * len(query) + 0.999)) if query and self.context_query != "on" else 1
        used = []
        for record in memory["records"]:
            if len(query & terms(record["text"])) >= need:
                used.append({"kind": "record", "id": record["id"], "text": record["text"], "source": "you said it", "session_id": record["session_id"], "turn": 1, "at": 0})
        said = {u["session_id"] for u in used}  # a record already says what its chat said
        for doc in memory["documents"]:
            if doc["session_id"] != session_id and doc["session_id"] not in said and len(query & terms(doc["text"])) >= need:
                used.append({"kind": "chat", "id": doc["id"], "text": doc["text"], "source": "an earlier chat", "session_id": doc["session_id"], "turn": 1, "at": 0})
        return used[:5]

    # --- routes ---

    def reply(self, h: BaseHTTPRequestHandler, status: int, body: Any = None) -> None:
        raw = b"" if body is None else json.dumps(body).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)

    def handle(self, h: BaseHTTPRequestHandler, method: str) -> None:
        url = urlsplit(h.path)
        with self.lock:
            self.requests.append((method, url.path))
        if h.headers.get("Authorization") != f"Bearer {self.token}":
            return self.reply(h, 401, {"error": {"code": "unauthorized"}})
        body = json.loads(h.rfile.read(int(h.headers.get("Content-Length") or 0)) or b"null")
        path, query = url.path, parse_qs(url.query)
        parts = path.strip("/").split("/")
        if method == "GET" and path == "/api/health":
            return self.reply(h, 200, {"ok": True, "configured": True, "sessions": {"store": "ok"}})
        if method == "POST" and path == "/api/sessions":
            turn = (body or {}).get("turn") or {}
            return self.start(h, None, turn, 201, (body or {}).get("workspace"))
        if method == "GET" and path == "/api/memory/import/scan":
            return self.reply(h, 200, self.import_scan())
        if method == "POST" and path == "/api/memory/import":
            if not self.across:
                return self.reply(h, 400, {"error": {"code": "invalid_request"}})
            ids = list((body or {}).get("source_ids") or [])
            kept = {src["id"]: src["items_kept"] for src in self.import_scan()["sources"]}
            return self.reply(h, 200, {"batch_id": (body or {}).get("batch_id"), "skipped": 0,
                                       "written_ids": list(range(1, 1 + sum(kept.get(i, 0) for i in ids)))})
        if method == "POST" and len(parts) == 3 and parts[:2] == ["api", "approvals"]:
            return self.answer(h, parts[2], body or {})
        if method == "POST" and len(parts) == 4 and parts[:2] == ["api", "turns"] and parts[3] == "cancel":
            with self.lock:
                flag = self.cancels.get(parts[2])
            if flag is None or flag.is_set():
                return self.reply(h, 409, {"error": {"code": "turn_not_running"}})
            flag.set()
            return self.reply(h, 202, {"turn_id": parts[2], "cancelling": True})
        if len(parts) >= 3 and parts[:2] == ["api", "sessions"]:
            sid = parts[2]
            session = self.sessions.get(sid)
            if session is None:
                return self.reply(h, 404, {"error": {"code": "not_found"}})
            tail = parts[3] if len(parts) > 3 else ""
            if method == "GET" and not tail:
                with self.lock:
                    pending = [{"request_id": r, "tool_name": a["tool"]} for r, a in self.approvals.items() if a["session"] == sid and a["outcome"] is None]
                return self.reply(h, 200, {"session": {"id": sid}, "pending_approvals": pending})
            if method == "POST" and tail == "turns":
                return self.start(h, sid, body, 202, None)
            if method == "GET" and tail == "turns":
                return self.reply(h, 200, {"turns": [dict(t) for t in session["turns"]][-int(query.get("limit", ["10"])[0]):], "has_earlier": False})
            if method == "GET" and tail == "events":
                return self.events(h, session, int(query.get("after", ["0"])[0]), query.get("follow", ["true"])[0] != "false")
            if method == "DELETE" and not tail:
                with self.lock:
                    if self.busy_deletes > 0:
                        self.busy_deletes -= 1
                        return self.reply(h, 409, {"error": {"code": "session_busy"}})
                    if self.across:
                        memory = self.load()
                        memory["documents"] = [d for d in memory["documents"] if d["session_id"] != sid]
                        self.save(memory)
                    del self.sessions[sid]
                    if self.store:
                        self.sql(self.sessions_db, "DELETE FROM windows WHERE session_id = ?", (sid,))
                h.send_response(204)
                h.send_header("Content-Length", "0")
                h.end_headers()
                return None
        return self.reply(h, 404, {"error": {"code": "not_found"}})

    def import_scan(self) -> dict[str, Any]:
        """What localmelo's memory import would find in this HOME: the instruction files and the folders of notes, one
        kept item per non-empty line."""
        found = [self.home / ".claude" / "CLAUDE.md", self.home / ".codex" / "AGENTS.md",
                 *sorted((self.home / ".claude" / "projects").glob("*/memory"))]
        sources = []
        for n, path in enumerate(f for f in found if f.exists()):
            files = sorted(path.glob("*.md")) if path.is_dir() else [path]
            lines = [x for f in files for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            sources.append({"id": f"claude-{n}", "harness": "claude", "path": str(path), "scope": "user",
                            "workspace": "", "items_found": len(lines), "items_kept": len(lines), "filtered": {}, "sample": lines[:3]})
        return {"batch_id": "b1", "sources": sources, "partial": False}

    def answer(self, h: BaseHTTPRequestHandler, request_id: str, body: Mapping[str, Any]) -> None:
        with self.lock:
            asked = self.approvals.get(request_id)
            if asked is None:
                return self.reply(h, 404, {"error": {"code": "not_found"}})
            if asked["outcome"] is not None:
                return self.reply(h, 409, {"error": {"code": "already_answered"}})
            asked["outcome"] = "allowed" if body.get("choice") in ("once", "session") else "denied"
            asked["reason"] = body.get("reason", "")
            asked["event"].set()
        return self.reply(h, 200, {"request_id": request_id, "outcome": asked["outcome"], "remember": False})

    def start(self, h: BaseHTTPRequestHandler, sid: str | None, turn: Mapping[str, Any], status: int, workspace: Any) -> None:
        text, key = str(turn.get("text") or ""), turn.get("client_request_id")
        if not text.strip() or (key is not None and not KEY_RE.fullmatch(str(key))):
            return self.reply(h, 400, {"error": {"code": "invalid_request"}})
        with self.lock:
            if sid is None:
                self.created += 1  # never reused, like the real gateway's random ids: a deleted chat's id is gone for good
                sid = f"s{self.created:03d}x{os.getpid() % 1000}"
                self.sessions[sid] = {"id": sid, "turns": [], "events": [], "workspace": workspace}
            session = self.sessions[sid]
            row = {"id": f"t{sid}-{len(session['turns']) + 1}", "turn": len(session["turns"]) + 1, "user_text": text, "status": "running", "cause": "", "answer": "", "result": None, "client_request_id": key}
            session["turns"].append(row)
            self.cancels[row["id"]] = threading.Event()
        threading.Thread(target=self.run_turn, args=(session, row), daemon=True).start()
        body: dict[str, Any] = {"turn_id": row["id"], "turn": dict(row)}
        if status == 201:
            body["session"] = {"id": sid, "workspace": workspace}
        self.reply(h, status, body)

    def add(self, session: dict[str, Any], row: Mapping[str, Any], type_: str, data: Mapping[str, Any]) -> None:
        with self.lock:
            event = {"id": len(session["events"]) + 1, "session_id": session["id"], "turn": row["turn"], "v": 1, "type": type_, "run_id": "", "seq": None, "ts": time.time(), "data": dict(data)}
            session["events"].append(event)
            if self.store:
                self.sql(self.sessions_db, "INSERT INTO events (session_id, turn, type, event) VALUES (?, ?, ?, ?)", (session["id"], row["turn"], type_, json.dumps(event)))
            self.lock.notify_all()

    # --- the working window (store=True) ---

    def push(self, session: dict[str, Any], *messages: Mapping[str, Any]) -> None:
        """Append one unit (a tool call with its result, or one message) to the chat's window and drop the oldest units
        when it holds more than the cap: down to cap - 8 (at most half the cap), whole units, never the newest."""
        units = session.setdefault("units", [])
        units.append([dict(m) for m in messages])
        if sum(len(u) for u in units) > self.window_cap:
            target = self.window_cap - min(8, self.window_cap // 2)
            while sum(len(u) for u in units) > target and len(units) > 1:
                units.pop(0)

    def window(self, session: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [m for u in session.get("units", []) for m in u]

    def window_match(self, session: Mapping[str, Any], text: str) -> str | None:
        """The newest user message in the chat's window, other than ``text``, that shares enough words with it."""
        query = terms(text)
        need = max(1, int(0.6 * len(query) + 0.999))
        return next((m["content"] for m in reversed(self.window(session)) if m["role"] == "user" and m["content"] != text and len(query & terms(m["content"])) >= need), None)

    def call_model(self, base: str, path: str, body: Mapping[str, Any], *, stream: bool = False) -> float:
        """One request to the model server named in the config (through the driver's proxy); milliseconds taken."""
        url = urlsplit(base)
        t0 = time.perf_counter()
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
        try:
            conn.request("POST", url.path.rstrip("/") + path, json.dumps({**body, **({"stream": True} if stream else {})}), {"Content-Type": "application/json"})
            conn.getresponse().read()
        except (OSError, http.client.HTTPException):
            pass  # a model call that fails is not what these tests are about
        finally:
            conn.close()
        return (time.perf_counter() - t0) * 1000.0

    def run_turn(self, session: dict[str, Any], row: dict[str, Any]) -> None:
        text, sid = row["user_text"], session["id"]
        cancel = self.cancels[row["id"]]
        self.add(session, row, "turn.started", {"turn": row["turn"], "user_text": text})
        if self.store:
            self.push(session, {"role": "user", "content": text})
        time.sleep(self.delay)
        statement = not text.rstrip().endswith(("?", "？"))
        used: list[dict[str, Any]] = []
        remembered: list[dict[str, Any]] = []
        recall: dict[str, Any] | None = None
        embeds = bool(self.embedding_url) and self.retriever in ("embedding", "auto")
        if self.across:
            embed_ms = select_ms = 0
            if embeds:
                embed_ms = int(self.call_model(self.embedding_url, "/embeddings", {"model": self.embedding_model, "input": [text]})) if self.model_calls else 12
            with self.lock:
                memory = self.load()
                used = self.recall(text, sid, memory)
            if self.retriever == "llm":  # the chat model chooses among the candidates
                select_ms = int(self.call_model(self.chat_url, "/chat/completions", {"model": self.chat_model, "messages": []})) if self.model_calls else 800
            with self.lock:
                memory = self.load()
                if statement and (self.save_only is None or any(w in text for w in self.save_only)):
                    subject = text.lower().split()[:2]
                    memory["records"] = [r for r in memory["records"] if r["text"].lower().split()[:2] != subject]
                    record = {"id": max([r["id"] for r in memory["records"]] + [0]) + 1, "text": text, "session_id": sid}
                    memory["records"].append(record)
                    remembered.append({"id": record["id"], "text": text, "topic": "about"})
                    self.save(memory)
                    if self.store:
                        self.sql(self.records_db, "INSERT INTO long_term (text, session_id) VALUES (?, ?)", (text, sid))
            shown = self.retriever if self.retriever != "auto" else ("embedding" if embeds else "bm25")
            recall = {"retriever": shown, "fallback": "none", "embed_ms": embed_ms, "search_ms": 1, "select_ms": select_ms,
                      "select_tokens": {"prompt": 640 if self.retriever == "llm" else 0, "completion": 12 if self.retriever == "llm" else 0},
                      "candidates": len(used), "used": len(used)}
            if self.new_build:
                recall.update(context="used" if self.context_query == "on" else "off", block_chars=sum(len(u["text"]) + len(u["source"]) for u in used))
            if self.recall_timings or used or remembered:  # a build without the timings sent the event only for something
                data: dict[str, Any] = {"turn": row["turn"], "turn_id": row["id"], "used": used, "remembered": remembered, "refused": ""}
                if self.recall_timings:
                    data["recall"] = recall
                self.add(session, row, "turn.memory", data)
        recall = recall if self.recall_timings else None
        self.add(session, row, "run.started", {"query": text})
        time.sleep(self.delay)
        denied = False
        if any(m in text for m in self.approval_texts):  # the model tries to save the fact to a file
            request_id = f"ap-{sid}-{row['turn']}"
            asked = {"session": sid, "tool": "file_write", "outcome": None, "event": threading.Event()}
            with self.lock:
                self.approvals[request_id] = asked
            self.add(session, row, "approval.requested", {"request_id": request_id, "call_id": "c1", "tool_name": "file_write", "arguments": {}, "preview": "write notes.txt", "workspace_root": None})
            denied = asked["event"].wait(5.0) and asked["outcome"] == "denied"
            self.add(session, row, "approval.resolved", {"request_id": request_id, "tool_name": "file_write", "outcome": asked["outcome"] or "timeout"})
        cancelled = False
        if any(m in text for m in self.hang_texts):  # a turn that does not come back
            cancelled = cancel.wait(self.hang_seconds)
        tool = next((name for word, name in self.tool_words.items() if word in text), "")
        step = 1
        if tool:  # a read-tier tool call: the call, its result, and then the answer
            call = {"id": f"call_{sid}_{row['turn']}", "name": tool, "arguments": {}}
            self.add(session, row, "assistant.message", {"step": 1, "content": "", "tool_calls": [call], "usage": {"completion_tokens": 7}, "duration_s": self.delay})
            self.add(session, row, "tool.started", {"step": 1, "call_id": call["id"], "name": tool, "arguments": {}})
            self.add(session, row, "tool.finished", {"step": 1, "call_id": call["id"], "name": tool, "status": "success", "output": "ok", "error": ""})
            if self.store:
                self.push(session, {"role": "assistant", "content": "", "tool_calls": [call]}, {"role": "tool", "content": "ok", "tool_call_id": call["id"]})
            step = 2
        if self.model_calls and self.chat_url and not cancelled:
            window = self.window(session) if self.store else []
            self.call_model(self.chat_url, "/chat/completions", {"model": self.chat_model, "messages": window, "tools": [{"type": "function"}]}, stream=True)
        failed = any(f in text for f in self.fail_texts) or (bool(self.approval_texts) and any(m in text for m in self.approval_texts) and not denied)
        scripted = next((reply for key, reply in self.answers.items() if key in text), None)
        own = self.window_match(session, text) if self.store and not statement and not used else None
        if failed or cancelled:
            answer = ""
        elif scripted is not None:
            answer = scripted
        else:
            answer = f"From my notes: {used[0]['text']}" if used else f"From this chat: {own}" if own else ("Noted." if statement else "I don't know.")
        self.add(session, row, "assistant.message", {"step": step, "content": answer, "usage": {"completion_tokens": 7}, "duration_s": self.delay})
        if self.store:
            self.push(session, {"role": "assistant", "content": answer})
        with self.lock:
            if self.across and answer:
                memory = self.load()
                memory["documents"].append({"id": len(memory["documents"]) + 1, "session_id": sid, "text": f"User: {text}\nmelo: {answer}"})
                self.save(memory)
            result = {"usage": {"completion_tokens": 7}, "error": "boom" if failed else ""}
            if recall is not None:
                result["memory_recall"] = recall
            row.update(status="cancelled" if cancelled else "failed" if failed else "completed", cause="run", answer=answer, result=result)
        if self.store:  # the end of a turn: run.finished (with the cap the chat ran with) and the window checkpoint
            self.add(session, row, "run.finished", {"result": {"status": row["status"], "config": {"window_messages": self.window_cap}}})
            self.sql(self.sessions_db, "INSERT OR REPLACE INTO windows (session_id, turn, version, data, created_at, thinking_since) VALUES (?, ?, 1, ?, ?, NULL)",
                     (sid, row["turn"], json.dumps({"v": 1, "messages": self.window(session)}), time.time()))
        self.add(session, row, "turn.finished", {"turn": row["turn"], "status": row["status"], "cause": "run"})
        cancel.set()  # a finished turn can no longer be cancelled (409)
        if self.model_calls and embeds and answer and self.across:  # the finished turn is indexed in the background
            threading.Thread(target=self.call_model, args=(self.embedding_url, "/embeddings", {"model": self.embedding_model, "input": [answer], "slow": True}), daemon=True).start()

    def events(self, h: BaseHTTPRequestHandler, session: dict[str, Any], after: int, follow: bool) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True

        def chunk(data: bytes) -> None:
            h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            h.wfile.flush()

        chunk(b"retry: 2000\n\n")
        sent = after
        deadline = time.time() + 30
        try:
            while time.time() < deadline:
                with self.lock:
                    fresh = [e for e in session["events"] if e["id"] > sent]
                    if not fresh and follow:
                        self.lock.wait(0.5)
                        continue
                for event in fresh:
                    chunk(f"id: {event['id']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                    sent = event["id"]
                    if self.drop_stream and follow:
                        return  # cut off without the closing chunk
                    if follow and event["type"] == "turn.finished":
                        break
                else:
                    if follow:
                        continue
                break
            h.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


class FakeModelServer:
    """An oMLX stand-in: `GET /v1/models`, the status routes, `POST /v1/chat/completions` (whole or streamed, with a
    delay before the first byte) and `POST /v1/embeddings` (with a delay); every POST is kept in `calls`."""

    def __init__(self, ids: Sequence[str], *, first_delay: float = 0.0, chunk_delay: float = 0.0, embed_delay: float = 0.0, slow_embed_delay: float | None = None, loaded: Sequence[str] | None = None,
        unload_delay: float = 0.0, stuck: bool = False, status_routes: bool = True,
    ) -> None:
        self.ids = list(ids)
        self.loaded = list(ids if loaded is None else loaded)
        self.first_delay, self.chunk_delay, self.embed_delay = first_delay, chunk_delay, embed_delay
        self.slow_embed_delay = embed_delay if slow_embed_delay is None else slow_embed_delay  # for a request marked "slow": the indexing
        self.calls: list[dict[str, Any]] = []
        self.unloads: list[str] = []  # the models asked to unload, in order
        self.unload_delay, self.stuck, self.status_routes = unload_delay, stuck, status_routes  # stuck: never lets go of a model
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def send_json(self, body: Any) -> None:
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def send_error_json(self, status: int) -> None:
                raw = b'{"detail": "no"}'
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                if owner.status_routes and self.path.endswith("/api/status"):
                    return self.send_json({"status": "ok", "version": "test", "models_loaded": len(owner.loaded), "loaded_models": owner.loaded, "model_memory_used": 1000 * len(owner.loaded)})
                if owner.status_routes and self.path.endswith("/models/status"):
                    return self.send_json({"models": [{"id": i, "loaded": i in owner.loaded, "is_loading": False, "actual_size": 1000} for i in owner.ids]})
                return self.send_json({"object": "list", "data": [{"id": i, "owned_by": "omlx"} for i in owner.ids]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path.endswith("/unload"):  # POST /v1/models/{id}/unload
                    model = self.path.split("/")[-2]
                    with owner.lock:
                        owner.unloads.append(model)
                        known, loaded = model in owner.ids, model in owner.loaded
                    if not known:
                        return self.send_error_json(404)
                    if not loaded:
                        return self.send_error_json(400)
                    if not owner.stuck:
                        threading.Timer(owner.unload_delay, lambda: owner.loaded.remove(model) if model in owner.loaded else None).start()
                    return self.send_json({"status": "ok", "model_id": model})
                with owner.lock:
                    if body.get("model") in owner.ids and body["model"] not in owner.loaded:
                        owner.loaded.append(body["model"])  # a request loads the model it names
                    owner.calls.append({"path": self.path, "stream": bool(body.get("stream")), "tools": bool(body.get("tools")), "model": body.get("model")})
                if self.path.endswith("/embeddings"):
                    time.sleep(owner.slow_embed_delay if body.get("slow") else owner.embed_delay)
                    return self.send_json({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
                time.sleep(owner.first_delay)
                if not body.get("stream"):
                    return self.send_json({"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for piece in ("data: " + json.dumps({"choices": [{"delta": {"content": w}}]}) + "\n\n" for w in ("o", "k")):
                    self.wfile.write(f"{len(piece.encode()):x}\r\n".encode() + piece.encode() + b"\r\n")
                    self.wfile.flush()
                    time.sleep(owner.chunk_delay)
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.host_port = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.base_url = f"{self.host_port}/v1"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def melo_main(argv: Sequence[str]) -> int:
    """The `melo` command line of the fake: `config get KEY` and `serve --port N [-X K=V]...`."""
    args = list(argv)
    if args[:2] == ["config", "get"]:
        value = KNOWN_SETTINGS.get(args[2])
        if value is None:
            print(f"melo: unknown setting {args[2]}", file=sys.stderr)
            return 1
        print(value)
        return 0
    if args[:1] == ["serve"]:
        port = int(args[args.index("--port") + 1])
        x_pairs = [args[i + 1] for i, a in enumerate(args) if a == "-X"]
        home = Path(os.environ["HOME"])
        (home / "env.json").write_text(json.dumps({
            "home": str(home), "browser": os.environ.get("BROWSER"), "stdin_tty": sys.stdin.isatty() if sys.stdin else False,
            "cwd": os.getcwd(), "secret_visible": "ACROSS_TEST_SECRET_KEY" in os.environ, "argv": args,
        }))
        gateway = FakeGateway(home, x_pairs, port=port)
        done = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: done.set())
        done.wait()
        gateway.stop()
        return 0
    return 2


def mini_raw() -> dict[str, Any]:
    """A small valid task set (4 facts, 5 questions) for the driver and report tests."""
    def fact(i: int, block: str, lang: str, role: str, text: str, value: str, **extra: str) -> dict[str, Any]:
        return {"id": f"F{i}", "order": i, "block": block, "lang": lang, "role": role, "text": text, "value": value, **extra}

    def question(i: int, kind: str, lang: str, facts: list[str], text: str, answer: str = "", **extra: Any) -> dict[str, Any]:
        return {"id": f"Q{i}", "kind": kind, "lang": lang, "facts": facts, "text": text, **({"answer": answer} if answer else {}), **extra}

    return {
        "version": 1, "protocol": "mini", "ask_seed": 1,
        "facts": [
            fact(1, "B01", "en", "plain", "My cat is called Biscuit.", "Biscuit"),
            fact(2, "B01", "zh", "plain", "我的室友是小满。", "小满"),
            fact(3, "B02", "en", "old", "I live in Lisbon.", "Lisbon"),
            fact(4, "B02", "en", "new", "I live in Utrecht.", "Utrecht", supersedes="F3"),
        ],
        "questions": [
            question(1, "exact", "en", ["F1"], "What is my cat called?", "Biscuit"),
            question(2, "exact", "en", ["F1"], "What is my cat called these days?", "Biscuit"),
            question(3, "crosslang", "en", ["F2"], "Who is my roommate?", "小满", aliases=["Xiaoman"]),
            question(4, "superseded", "en", ["F4"], "Where do I live?", "Utrecht", wrong_facts=["F3"], reject=["Lisbon"]),
            question(5, "nothing", "en", [], "What is my favourite board game?"),
        ],
    }


MINI_TOOL_WORDS = {"Read ": "file_read", "List ": "list_files"}  # for FakeGateway(tool_words=...)
MINI_SAVE_ONLY = ("Priya", "advisor", "Ortiz", "班主任", "周老师")  # the fake saves only these statements: "Today's call is in room 4B." is not a note


def mini_raw_v2() -> dict[str, Any]:
    """A small valid v2 task set (2 long-range and 2 follow-up items, one filler set of 40 turns) for the driver and
    report tests. Run it with ``FakeGateway(store=True, tool_words=MINI_TOOL_WORDS, save_only=MINI_SAVE_ONLY)``."""
    files = ["todo.md", "notes.txt", "docs/faq.md"]
    tools = [{"id": f"ft{n:02d}", "lang": "en", "tool": "file_read", "text": f"Read {files[n % 3]} and report line {n}. Answer in one sentence."} for n in range(16)]
    tools += [{"id": f"ft{16 + n:02d}", "lang": "en", "tool": "list_files", "text": f"List the files, check number {n}. Answer in one sentence."} for n in range(6)]
    plain = [{"id": f"fp{n:02d}", "lang": "en", "text": f"Quick fact number {n}, what is it? Answer in one sentence."} for n in range(18)]
    order = []
    for n in range(22):  # a plain turn after every tool turn while there are plain turns
        order.append(tools[n]["id"])
        if n < 18:
            order.append(plain[n]["id"])

    def ask(text: str, answer: str, **extra: Any) -> dict[str, Any]:
        return {"text": text, "answer": answer, **extra}

    return {
        "version": 2, "protocol": "mini-v2", "ask_seed": 5,
        "fixtures": [{"path": p, "text": f"Sample {p}.\n"} for p in files],
        "filler": tools + plain, "sets": {"en": order},
        "items": [
            {"id": "L1", "kind": "longrange", "lang": "en", "note_expected": False, "filler": "en",
             "opener": "Hi, long day of calls ahead. Yesterday's was in room 2A.", "fact": "Today's call is in room 4B.",
             "asks": [ask("Which room is today's call in?", "4B", reject=["2A"])]},
            {"id": "L2", "kind": "longrange", "lang": "en", "note_expected": True, "filler": "en",
             "opener": "Hey, got a minute?", "bridge": "Quick one: what is a synonym for big? Answer in one sentence.",
             "fact": "My sister's name is Priya.", "asks": [ask("What is my sister's name?", "Priya")]},
            {"id": "P1", "kind": "followup", "lang": "en", "facts": ["My advisor is Dr. Lena Ortiz.", "Dr. Ortiz's office is room 512."],
             "asks": [ask("Who is my advisor?", "Ortiz"), ask("And where is her office?", "512")]},
            {"id": "P2", "kind": "followup", "lang": "zh", "facts": ["我的班主任是周老师。", "周老师的办公室在文科楼305。"],
             "asks": [ask("我的班主任是谁？", "周老师"), ask("那她的办公室在哪儿？", "305", aliases=["文科楼"])]},
        ],
    }
