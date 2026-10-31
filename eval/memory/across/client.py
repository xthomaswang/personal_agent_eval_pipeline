"""A small client for localmelo's console API (API v2), enough to run chats and read what memory did.

The shapes, read at localmelo 05ff696c (support/gateway/console/routes_turns.py and routes_sessions.py):

    POST   /api/sessions                    {"workspace": <dir>|null, "turn": {"text", "client_request_id"}}
                                            201 {"session": {"id", ...}, "turn_id", "turn": {...}}
    POST   /api/sessions/{id}/turns         {"text", "client_request_id"}   202 {"turn_id", "turn": {...}}
    GET    /api/sessions/{id}/turns?limit=1 {"turns": [{"id", "turn", "status", "cause", "answer", "result", ...}]}
    GET    /api/sessions/{id}/events        SSE: one "data: <json>" line per event, from event id `after`;
                                            {"id", "session_id", "turn", "type", "ts", "data": {...}}
    DELETE /api/sessions/{id}?notes=keep    204 (the chat's index entries go; its notes stay unless notes=forget)
    GET    /api/health                      {"ok", "configured", "sessions": {"store": "ok"}, ...}
    GET    /api/sessions/{id}               {"session": {...}, "pending_approvals": [{"request_id", "tool_name", ...}], ...}
    POST   /api/approvals/{request_id}      {"choice": "deny", "reason"}   200 {"request_id", "outcome": "denied"}
                                            (404 unknown, 409 answered already)
    POST   /api/turns/{turn_id}/cancel      202 {"turn_id", "cancelling": true}; 409 when the turn is not running

A tool call that needs approval (the chat model tries to save a fact to a file) sets an `approval.requested`
event and leaves the turn `running` until someone answers. The study runs no tools, so `run_turn` denies every
request and counts it.

A turn is done when its row's ``status`` is no longer ``running``; ``turn.finished`` is written in the same
transaction, so the row read after that event already holds the answer. Every request carries the gateway
token as a bearer. The gateway accepts loopback clients only, and so does this client: it uses no proxy and
follows no redirect, so the token goes nowhere else.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import re
import time
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
TOKEN_PATH = Path(".cache") / "localmelo" / "gateway" / "auth" / "token"
KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")  # the gateway's client_request_id grammar
DENY_REASON = "memory study: no tool runs in these chats; answer in text"
TURN_OK = (200, 201, 202)  # 200: the same client_request_id again, answered with the turn it started


class GatewayError(RuntimeError):
    def __init__(self, status: int, message: str, body: Any = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


def read_token(home: Path) -> str:
    return (Path(home) / TOKEN_PATH).read_text(encoding="ascii").strip()


class Gateway:
    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0) -> None:
        url = urllib.parse.urlsplit(base_url)
        if url.scheme != "http" or url.hostname not in LOOPBACK or url.path not in ("", "/") or url.query:
            raise ValueError(f"the gateway URL must be http://<loopback>:<port>, got {base_url!r}")
        self.host = url.hostname or "127.0.0.1"
        self.port = url.port or 80
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self._token = token

    def _connection(self, timeout: float | None = None) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)

    def _headers(self, body: bytes | None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._token}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return headers

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        conn = self._connection()
        try:
            conn.request(method, path, body=data, headers=self._headers(data))
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            conn.close()
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = {"raw": raw.decode("utf-8", "replace")[:500]}
        return resp.status, parsed

    def call(self, method: str, path: str, body: Any = None, ok: tuple[int, ...] = (200,)) -> Any:
        status, parsed = self.request(method, path, body)
        if status not in ok:
            raise GatewayError(status, f"{method} {path}: {json.dumps(parsed, ensure_ascii=False)[:300]}", parsed)
        return parsed

    def health(self) -> dict[str, Any]:
        return self.call("GET", "/api/health")

    def delete_session(self, session_id: str) -> None:
        """Delete a chat and its index entries; ``notes=keep`` is the default and is asked for, since a question has none."""
        self.call("DELETE", f"/api/sessions/{session_id}?notes=keep", ok=(204,))

    def pending_approvals(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.call("GET", f"/api/sessions/{session_id}").get("pending_approvals")
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def deny(self, request_id: str, reason: str) -> bool:
        """Deny one approval request; True when this call denied it (False: unknown or answered already)."""
        status, _ = self.request("POST", f"/api/approvals/{request_id}", {"choice": "deny", "reason": reason})
        if status in (404, 409):
            return False
        if status != 200:
            raise GatewayError(status, f"POST approvals/{request_id}")
        return True

    def cancel_turn(self, turn_id: str) -> bool:
        status, _ = self.request("POST", f"/api/turns/{turn_id}/cancel")
        return status == 202

    def last_turn(self, session_id: str) -> dict[str, Any] | None:
        turns = self.call("GET", f"/api/sessions/{session_id}/turns?limit=1")["turns"]
        return turns[-1] if turns else None

    def events_once(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        """The stored events after ``after``, then the response ends (``follow=false``)."""
        conn = self._connection()
        try:
            conn.request("GET", f"/api/sessions/{session_id}/events?after={after}&follow=false", headers=self._headers(None))
            resp = conn.getresponse()
            if resp.status != 200:
                raise GatewayError(resp.status, f"GET events: {resp.read()[:200]!r}")
            return list(_frames(resp))
        finally:
            conn.close()

    def stream(self, session_id: str, after: int = 0, timeout: float = 60.0) -> Iterator[dict[str, Any]]:
        """Follow the session's events as they are stored; the caller stops reading when it has enough."""
        conn = self._connection(timeout)
        try:
            conn.request("GET", f"/api/sessions/{session_id}/events?after={after}", headers=self._headers(None))
            resp = conn.getresponse()
            if resp.status != 200:
                raise GatewayError(resp.status, f"GET events: {resp.read()[:200]!r}")
            yield from _frames(resp)
        finally:
            conn.close()


def _frames(resp: Any) -> Iterator[dict[str, Any]]:
    """One dict per ``data:`` line of an event stream; other lines (retry, keepalive, ids) are skipped."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


@dataclass
class TurnResult:
    session_id: str
    turn_id: str
    turn_no: int
    status: str  # the turn row's status, or "timeout"
    cause: str = ""
    answer: str = ""
    error: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)  # each with "t_ms", ms since the POST
    wall_ms: float = 0.0
    wall_source: str = "stream"  # "stream": the turn.finished frame arrived; "poll": the row was seen finished
    http_status: int = 0
    denied: list[dict[str, str]] = field(default_factory=list)  # approval requests this call denied: request_id, tool
    cancelled: bool = False  # a turn that timed out was asked to stop
    memory_recall: dict[str, Any] | None = None  # the turn row's result.memory_recall (the same as turn.memory's recall)
    started_epoch: float = 0.0  # seconds since the epoch, just before the POST and when the turn was seen finished
    ended_epoch: float = 0.0

    @property
    def ttfe_ms(self) -> float | None:
        """Milliseconds from the POST to the first event frame; only a stream measures it."""
        return self.events[0].get("t_ms") if self.events and self.wall_source == "stream" else None

    def event(self, type_: str) -> dict[str, Any] | None:
        return next((e for e in self.events if e.get("type") == type_), None)

    def memory(self) -> dict[str, Any] | None:
        """The ``data`` of this turn's last ``turn.memory`` event (``used``, ``remembered``, ``refused``, and
        ``recall`` when the build reports it); None when memory did nothing and sent no event. Each event is
        the turn's whole memory state so far: the first one comes with recall, and a note saved by a tool
        sends another, so only the last has every ``remembered`` item."""
        event = next((e for e in reversed(self.events) if e.get("type") == "turn.memory"), None)
        data = event.get("data") if event else None
        return data if isinstance(data, dict) else None

    def server_ms(self) -> dict[str, float | None]:
        """Where the turn spent its time by the gateway's own event clock (``ts``, seconds), from turn.started."""
        start = self.event("turn.started")
        base = start.get("ts") if start else None

        def at(type_: str) -> float | None:
            event = self.event(type_)
            ts = event.get("ts") if event else None
            if not isinstance(ts, (int, float)) or not isinstance(base, (int, float)):
                return None
            return round((ts - base) * 1000.0, 1)

        return {
            "memory": at("turn.memory"),
            "run_started": at("run.started"),
            "first_message": at("assistant.message"),
            "finished": at("turn.finished"),
        }


def run_turn(
    gw: Gateway,
    text: str,
    key: str,
    *,
    session_id: str | None = None,
    workspace: str | None = None,
    timeout_s: float = 900.0,
    poll_s: float = 0.25,
    use_stream: bool = True,
    deny_approvals: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    epoch: Callable[[], float] = time.time,
) -> TurnResult:
    """Send one message (a new chat when ``session_id`` is None) and wait for the turn to end.

    The wall time runs from just before the POST to the ``turn.finished`` frame (stream) or to the poll that
    saw the row finished. A turn that does not end in ``timeout_s`` is asked to stop and comes back with status
    ``timeout``. With ``deny_approvals`` every approval request of the turn is denied, so a tool call cannot
    keep the turn waiting; the denials are in ``TurnResult.denied``.
    """
    if not KEY_RE.fullmatch(key):
        raise ValueError(f"client_request_id {key!r} breaks the gateway's grammar")
    t0, e0 = clock(), epoch()
    if session_id is None:
        status, body = gw.request("POST", "/api/sessions", {"workspace": workspace, "turn": {"text": text, "client_request_id": key}})
        if status not in TURN_OK or not isinstance(body, dict):
            raise GatewayError(status, f"POST /api/sessions: {json.dumps(body, ensure_ascii=False)[:300]}", body)
        session_id = str(body["session"]["id"])
    else:
        status, body = gw.request("POST", f"/api/sessions/{session_id}/turns", {"text": text, "client_request_id": key})
        if status not in TURN_OK or not isinstance(body, dict):
            raise GatewayError(status, f"POST turns: {json.dumps(body, ensure_ascii=False)[:300]}", body)
    turn_id = str(body["turn_id"])
    turn_no = int((body.get("turn") or {}).get("turn") or 1)
    result = TurnResult(session_id, turn_id, turn_no, "running", http_status=status, started_epoch=e0)
    denied_ids: set[str] = set()

    def deny(request_id: str, tool: str) -> None:
        if deny_approvals and request_id and request_id not in denied_ids:
            denied_ids.add(request_id)
            try:
                if gw.deny(request_id, DENY_REASON):
                    result.denied.append({"request_id": request_id, "tool": tool})
            except (GatewayError, OSError, http.client.HTTPException):
                denied_ids.discard(request_id)  # asked again at the next poll

    finished_at: float | None = None
    if use_stream:
        frames = gw.stream(session_id, timeout=min(60.0, timeout_s))
        try:
            for event in frames:
                now = clock()
                if event.get("turn") not in (None, turn_no):
                    continue
                result.events.append({**event, "t_ms": round((now - t0) * 1000.0, 1)})
                if event.get("type") == "approval.requested":
                    asked = event.get("data") if isinstance(event.get("data"), dict) else {}
                    deny(str(asked.get("request_id") or ""), str(asked.get("tool_name") or ""))
                if event.get("type") == "turn.finished":
                    finished_at, result.ended_epoch = now, epoch()
                    break
                if now - t0 > timeout_s:
                    break
        except (OSError, http.client.HTTPException, GatewayError):
            result.events = []  # a broken stream: read the events again after the turn
        finally:
            frames.close()
    if finished_at is None:
        result.wall_source = "poll"
        result.events = []  # the frames seen so far are partial: read all of them once the turn has ended
    while True:
        row = gw.last_turn(session_id)
        if row is not None and row.get("status") != "running":
            if finished_at is None:
                finished_at, result.ended_epoch = clock(), epoch()
            break
        if deny_approvals:
            for pending in gw.pending_approvals(session_id):
                deny(str(pending.get("request_id") or ""), str(pending.get("tool_name") or ""))
        if clock() - t0 > timeout_s:
            break
        sleep(poll_s)
    if finished_at is None or row is None or row.get("status") == "running":
        result.ended_epoch = epoch()
        result.status = "timeout"
        result.error = f"the turn did not finish in {timeout_s:g} s"
        result.wall_ms = round((clock() - t0) * 1000.0, 1)
        with contextlib.suppress(GatewayError, OSError, http.client.HTTPException):
            result.cancelled = gw.cancel_turn(turn_id)
        return result
    result.wall_ms = round((finished_at - t0) * 1000.0, 1)
    if not result.events:
        stored = gw.events_once(session_id)
        result.events = [{**e, "t_ms": None} for e in stored if e.get("turn") in (None, turn_no)]
    result.status = str(row.get("status") or "")
    result.cause = str(row.get("cause") or "")
    result.answer = str(row.get("answer") or "")
    summary = row.get("result") if isinstance(row.get("result"), dict) else {}
    result.error = str(summary.get("error") or "")
    recall = summary.get("memory_recall")
    result.memory_recall = recall if isinstance(recall, dict) else None
    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
    result.usage = {k: v for k, v in usage.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return result
