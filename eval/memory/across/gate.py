"""The driver side of the note-gate scenarios (kind ``notegate`` of data_v2.py): fixtures, the messages, the turn digest.

A scenario is 1-3 messages in one chat. Its pages and files reach the chat in one of two ways (``--fixtures``):

- ``files`` (the default): every file of the scenario is written to the chat's folder under the cell's HOME, and
  ``http://{{FIXTURE}}/<path>`` in a message becomes the plain relative path ``<path>``;
- ``http``: the scenario's files are served by a stdlib server on 127.0.0.1 and a free port, started before the cell's
  gateway and stopped after it (one server per cell, showing the files of the scenario that is running), and
  ``{{FIXTURE}}`` becomes ``127.0.0.1:<port>``. The gateway is told to let that one host through its ``web_fetch`` by a
  setting whose name is ``--fixture-setting`` (``tools.web_fetch_allow_local``, a name that is not final) with the value
  ``127.0.0.1:<port>``. A file a message names by its bare path (``resume.md``) is also written to the chat's folder,
  since no URL reaches it.

Nothing here reads a note: the score is made after the cell from the cell's SQLite files (report_gate.py).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from . import client, data_v2

ARMS_FILE = Path(__file__).resolve().parent / "data" / "arms_notegate.json"
FIXTURE_MODES = ("files", "http")
FIXTURE_SETTING = "tools.web_fetch_allow_local"  # the name is not final; --fixture-setting replaces it
HOST = "127.0.0.1"
PHASE = "gate"  # the journal's phase of a scenario's turns; the item is "<scenario>.t<nn>"
NOTE_TOOLS = ("remember",)  # the tool that keeps a note (TM27 may rename it: then add its name here)
REFUSAL_PREFIX = "not remembered"  # what localmelo's remember answers when it keeps nothing ("Not remembered: ...")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8", ".md": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8"}


# --- the messages and the files ---


def localize(text: str, host: str | None) -> str:
    """``text`` with the scenario's placeholder resolved: ``http://{{FIXTURE}}/`` is dropped (files mode, ``host`` None) or
    ``{{FIXTURE}}`` becomes ``host`` (http mode)."""
    return text.replace(data_v2.GATE_URL, "") if host is None else text.replace(data_v2.GATE_PLACEHOLDER, host)


def step_label(item: data_v2.Item, number: int) -> str:
    """The journal's item of message ``number`` (from 1) of a scenario."""
    return f"{item.id}.m{number:02d}"


def messages(item: data_v2.Item, host: str | None) -> list[str]:
    return [localize(t.user, host) for t in item.turns]


def workspace_files(item: data_v2.Item, host: str | None) -> list[data_v2.Fixture]:
    """The files written to the chat's folder: all of them in files mode; in http mode (``host`` given) those a message names
    by their bare path outside any URL, since no URL reaches them."""
    if host is None:
        return [data_v2.Fixture(f.path, localize(f.text, None)) for f in item.files]
    outside = " ".join(data_v2.without_gate_urls(t.user) for t in item.turns)
    return [data_v2.Fixture(f.path, localize(f.text, host)) for f in item.files if f.path in outside]


def prepare_workspace(item: data_v2.Item, path: Path, host: str | None) -> None:
    """The scenario's folder with its files in it; a file that is there already is left as it is (a chat that a stop
    interrupted continues in its folder)."""
    path.mkdir(parents=True, exist_ok=True)
    for fixture in workspace_files(item, host):
        target = path / fixture.path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fixture.text, encoding="utf-8")


# --- the fixture server (http mode) ---


class FixtureServer:
    """The scenario's files on 127.0.0.1 and a free port, until ``stop``. ``show`` replaces the files it serves."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                path = unquote(urlsplit(self.path).path).lstrip("/")
                owner.requests.append(path)
                body = owner.files.get(path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer((HOST, 0), Handler)  # port 0: the system picks one that is free
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        self.host = f"{HOST}:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def show(self, files: Mapping[str, str]) -> None:
        self.files = {path: text.encode("utf-8") for path, text in files.items()}

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@dataclass
class Fixtures:
    """How a cell's scenarios reach the chat: the mode, and in http mode the running server."""

    mode: str
    server: FixtureServer | None = None

    @property
    def host(self) -> str | None:
        return self.server.host if self.server else None

    def start_scenario(self, item: data_v2.Item, folder: Path) -> list[str]:
        """Put the scenario's files where its chat can reach them and answer its messages."""
        if self.server:
            self.server.show({f.path: localize(f.text, self.host) for f in item.files})
        prepare_workspace(item, folder, self.host)
        return messages(item, self.host)

    def stop(self) -> None:
        if self.server:
            self.server.stop()


def open_fixtures(mode: str) -> Fixtures:
    """Fixtures for a cell: nothing to start in files mode, a running server in http mode."""
    if mode not in FIXTURE_MODES:
        raise ValueError(f"fixtures must be one of {FIXTURE_MODES}, not {mode!r}")
    return Fixtures(mode, FixtureServer() if mode == "http" else None)


def serve_pair(setting: str, fixtures: Fixtures | None) -> tuple[str, ...]:
    """The ``-X`` setting that lets the gateway's web_fetch reach the fixture server; none in files mode."""
    return (f"{setting}={fixtures.host}",) if fixtures and fixtures.host else ()


def fixture_tools(fixtures: Fixtures | None) -> tuple[str, ...]:
    """The approval a chat asks before web_fetch reads a listed local server (``browser:<origin>``): the driver allows
    that one once and still denies every other request. Without it the fetch is refused like any denied tool."""
    return (f"browser:http://{fixtures.host}",) if fixtures and fixtures.host else ()


# --- what a turn of a scenario did ---


def steps_used(events: list[dict[str, Any]]) -> int | None:
    """The planning calls of a turn: ``run.finished``'s ``result.steps``, else the highest step of its messages; None
    when the events say neither."""
    for event in events:
        if event.get("type") == "run.finished":
            value = (((event.get("data") or {}).get("result")) or {}).get("steps")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    steps = [(e.get("data") or {}).get("step") for e in events if e.get("type") == "assistant.message"]
    steps = [s for s in steps if isinstance(s, int) and not isinstance(s, bool)]
    return max(steps) if steps else None


NOTE_CALL_CHARS = 1000  # what a row keeps of a note call's arguments, output (the gate's whole refusal) and error


def note_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The turn's calls of the note tool, as ``tool.finished`` reports them, with the arguments of their ``tool.started``
    (what the model asked to keep): the output, which holds the gate's refusal and its reasons, and the error, each cut at
    NOTE_CALL_CHARS characters."""
    asked = {(e.get("data") or {}).get("call_id"): (e.get("data") or {}).get("arguments")
             for e in events if e.get("type") == "tool.started"}
    out = []
    for event in events:
        data = event.get("data") or {}
        if event.get("type") == "tool.finished" and data.get("name") in NOTE_TOOLS:
            arguments = asked.get(data.get("call_id"))
            out.append({
                "name": data["name"], "status": data.get("status"),
                "arguments": json.dumps(arguments, ensure_ascii=False)[:NOTE_CALL_CHARS] if arguments is not None else None,
                "output": str(data.get("output") or "")[:NOTE_CALL_CHARS], "error": str(data.get("error") or "")[:NOTE_CALL_CHARS]})
    return out


def is_refusal(call: Mapping[str, Any]) -> bool:
    """Whether a recorded note-tool call kept nothing because the gate said no: its answer starts with "Not remembered"."""
    return str(call.get("output") or "").strip().casefold().startswith(REFUSAL_PREFIX)


def turn_digest(res: client.TurnResult) -> dict[str, Any]:
    """What the report reads of a scenario's turn besides its memory event: the steps used and the note-tool calls."""
    return {"steps": steps_used(res.events), "note_calls": note_calls(res.events)}

