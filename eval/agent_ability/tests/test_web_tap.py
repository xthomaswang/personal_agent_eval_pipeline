"""The web tap (web/tap.py): tool lists per mode, filtering, URL rewriting, the screen mapping and the refusals.

The tap's child (chrome-devtools-mcp) is replaced by a fake MCP server script, so no browser and no node are needed.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from web import tap

TAP = Path(tap.__file__)

USERINFO = "User:pw" + "@"  # built, so the file holds no literal URL with user info
SNAPSHOT = (
    "## Latest page snapshot\n"
    'uid=1_0 RootWebArea "Results" url="https://www.example.com/search?q=a&sxsrf=AKL5y%3A1&ved=2ahUK"\n'
    '  uid=1_1 link "One" url="https://en.wikipedia.org/wiki/One_(number)?x=1#top"\n'
    f'  uid=1_2 link "Two" url="http://{USERINFO}Sub.Example.org:8080/two"\n'
    '  uid=1_3 link "Blank" url="about:blank"\n'
    '  uid=1_4 link "Data" url="data:image/png;base64,AAAA"\n'
    '  uid=1_5 button "Search" description="a url=elsewhere"\n'
)

# The child's tools: every name of chrome-devtools-mcp 1.11.0 the tap must filter, with the parameters it drops.
PAGE_ID = {"pageId": {"type": "number"}}
CHILD_TOOLS: dict[str, dict[str, Any]] = {
    "navigate_page": {
        "type": {"type": "string"},
        "url": {"type": "string"},
        "initScript": {"type": "string"},
        "timeout": {"type": "integer"},
    },
    "new_page": {"url": {"type": "string"}, "isolatedContext": {"type": "string"}, "background": {"type": "boolean"}},
    "list_pages": {},
    "select_page": {**PAGE_ID, "bringToFront": {"type": "boolean"}},
    "close_page": PAGE_ID,
    "wait_for": {"text": {"type": "array"}, "timeout": {"type": "integer"}},
    "take_snapshot": {"verbose": {"type": "boolean"}, "filePath": {"type": "string"}},
    "click": {"uid": {"type": "string"}},
    "hover": {"uid": {"type": "string"}},
    "fill": {"uid": {"type": "string"}, "value": {"type": "string"}},
    "type_text": {"text": {"type": "string"}, "submitKey": {"type": "string"}},
    "press_key": {"key": {"type": "string"}},
    "handle_dialog": {"action": {"type": "string"}},
    "take_screenshot": {"format": {"type": "string"}, "quality": {"type": "number"}},
    "click_at": {"x": {"type": "number"}, "y": {"type": "number"}},
    "evaluate_script": {"function": {"type": "string"}},
    "drag": {},
    "emulate": {},
    "upload_file": {},
    "lighthouse_audit": {},
}

FAKE_CHILD = r'''
"""A fake chrome-devtools-mcp: one JSON-RPC line per message on stdin and stdout; every tools/call is logged."""
import base64, json, os, struct, sys

CALLS = os.environ["FAKE_CHILD_CALLS"]
TOOLS = json.load(open(os.environ["FAKE_CHILD_TOOLS"]))
SNAPSHOT = os.environ["FAKE_CHILD_SNAPSHOT"]
VIEWPORT_FILE = os.environ.get("FAKE_VIEWPORT_FILE", "")
OPEN_PAGE = os.environ.get("FAKE_OPEN_PAGE")
CRASH_ON = os.environ.get("FAKE_CRASH_ON")
SIZES = {int(k): int(v) for k, v in (p.split(":") for p in os.environ.get("FAKE_SHOT_SIZES", "80:30000").split(","))}

with open(os.environ["FAKE_CHILD_PID"], "a") as f:
    f.write(str(os.getpid()) + "\n")


def jpeg(size, width=1280, height=713):
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    data = b"\xff\xd8" + sof
    while len(data) < size - 2:
        chunk = min(60000, size - 2 - len(data) - 4)
        if chunk <= 0:
            break
        data += b"\xff\xfe" + struct.pack(">H", chunk + 2) + b"\x00" * chunk
    return data + b"\xff\xd9"


def reply(rid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": rid}
    msg["error" if error else "result"] = error or result
    print(json.dumps(msg), flush=True)


def text(t, error=False):
    r = {"content": [{"type": "text", "text": t}]}
    if error:
        r["isError"] = True
    return r


for line in sys.stdin:
    msg = json.loads(line)
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params", {})
    if rid is None:
        continue
    if method == "initialize":
        reply(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}})
    elif method == "tools/list":
        reply(rid, {"tools": [
            {"name": n, "description": n + " tool", "inputSchema": {
                "type": "object", "properties": p, "required": [k for k in p if k in ("url", "pageId", "uid")],
                "additionalProperties": False}}
            for n, p in TOOLS.items()]})
    elif method == "tools/call":
        name, args = params["name"], params.get("arguments", {})
        with open(CALLS, "a") as f:
            f.write(json.dumps({"name": name, "arguments": args}) + "\n")
        if name == CRASH_ON:
            os._exit(3)
        if name == "take_snapshot":
            reply(rid, text(SNAPSHOT))
        elif name == "take_screenshot":
            size = SIZES[min(SIZES, key=lambda q: abs(q - args.get("quality", 80)))]
            data = base64.b64encode(jpeg(size)).decode()
            reply(rid, {"content": [{"type": "text", "text": "Took a screenshot."},
                                    {"type": "image", "data": data, "mimeType": "image/jpeg"}]})
        elif name == "evaluate_script":
            viewport = open(VIEWPORT_FILE).read() if os.path.exists(VIEWPORT_FILE) else os.environ.get("FAKE_VIEWPORT", "1280,713")
            reply(rid, text("Script ran on page and returned:\n```json\n[" + viewport + "]\n```"))
        elif name == "click_at" and OPEN_PAGE:
            reply(rid, text("Successfully clicked at the coordinates\n## Opened pages\n2: B page (http://x.test/b)"))
        elif name == "wait_for" and "boom" in args.get("text", []):
            reply(rid, text("Error: Timed out waiting for text", error=True))
        elif name == "hover":
            reply(rid, error={"code": -32602, "message": "Invalid params for hover"})
        else:
            reply(rid, text("Successfully ran " + name + " page https://long.example.com/a/b?c=d"))
'''


class Client:
    """A minimal MCP client over the tap's stdio."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(TAP), *argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.next_id = 0
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {}})
        assert init["result"]["serverInfo"]["name"] == "web"
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put("")

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        line = self.lines.get(timeout=30)
        assert line, "the tap closed its stdout"
        message = json.loads(line)
        assert message["id"] == self.next_id
        return message

    def tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list")["result"]["tools"]

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        message = self.request("tools/call", {"name": name, "arguments": arguments})
        assert "result" in message, message  # a refusal or a child error is a tool error, not a protocol error
        return message["result"]

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait(timeout=20)


class Rig:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.calls = tmp_path / "child-calls.jsonl"
        self.log = tmp_path / "tap-log.jsonl"
        self.pids = tmp_path / "child-pids.txt"
        self.script = tmp_path / "fake_child.py"
        self.script.write_text(FAKE_CHILD)
        (tmp_path / "tools.json").write_text(json.dumps({n: p for n, p in CHILD_TOOLS.items()}))
        self.env = {
            **os.environ,
            "FAKE_CHILD_CALLS": str(self.calls),
            "FAKE_CHILD_TOOLS": str(tmp_path / "tools.json"),
            "FAKE_CHILD_SNAPSHOT": SNAPSHOT,
            "FAKE_CHILD_PID": str(self.pids),
        }
        self.clients: list[Client] = []

    def start(self, mode: str, **env: str) -> Client:
        argv = ["--mode", mode, "--log", str(self.log)]
        if mode != "none":
            argv += ["--", sys.executable, str(self.script)]
        client = Client(argv, {**self.env, **env})
        self.clients.append(client)
        return client

    def child_calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def child_call_names(self) -> list[str]:
        return [c["name"] for c in self.child_calls()]

    def log_entries(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]


@pytest.fixture
def rig(tmp_path: Path):
    rig = Rig(tmp_path)
    yield rig
    for client in rig.clients:
        if client.proc.poll() is None:
            client.proc.kill()


def image_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in result["content"] if b["type"] == "image"]


def text_of(result: dict[str, Any]) -> str:
    return "\n".join(b["text"] for b in result["content"] if b["type"] == "text")


# --- pure helpers ---


def test_urls_are_cut_to_hosts_in_the_url_attribute_only() -> None:
    text, count = tap.rewrite_urls(SNAPSHOT)
    assert 'RootWebArea "Results" url="www.example.com"' in text
    assert 'link "One" url="en.wikipedia.org"' in text
    assert 'link "Two" url="sub.example.org:8080"' in text  # lower-cased, with the port, without the credentials
    assert 'link "Blank" url="about:blank"' in text  # no host: kept
    assert 'link "Data" url="data:"' in text  # a data: URL can be huge
    assert 'description="a url=elsewhere"' in text  # not a url="..." attribute
    assert count == 4
    assert len(text) < len(SNAPSHOT)
    assert tap.rewrite_urls("no urls here https://x.test/a") == ("no urls here https://x.test/a", 0)
    assert tap.host_of("not a url") is None


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://www.google.com/?hl=en", True),
        ("HTTP://example.com", True),
        ("about:blank", True),
        ("google.com", True),  # no scheme: the child decides
        ("file:///etc/hosts", False),
        ("javascript:alert(1)", False),
        ("data:text/html,hi", False),
        ("chrome://settings", False),
        ("view-source:https://example.com", False),
        ("ftp://example.com/x", False),
    ],
)
def test_navigation_schemes(url: str, allowed: bool) -> None:
    assert tap.url_allowed(url) is allowed


def fake_jpeg(width: int, height: int) -> bytes:
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00" + sof + b"\xff\xd9"


def test_jpeg_size_reads_the_frame_header() -> None:
    assert tap.jpeg_size(fake_jpeg(1280, 713)) == (1280, 713)
    assert tap.jpeg_size(b"\x89PNG") is None
    assert tap.jpeg_size(b"\xff\xd8") is None


def test_normalized_coordinates_map_to_viewport_pixels() -> None:
    assert tap.map_point(0, 0, (1280, 713)) == (0, 0)
    assert tap.map_point(1000, 1000, (1280, 713)) == (1279, 712)  # the last pixel, not one past it
    assert tap.map_point(500, 500, (800, 600)) == (400, 300)
    assert tap.map_point(250, 750, (1280, 713)) == (320, 535)


@pytest.mark.parametrize(
    ("spec", "normalized"),
    [
        ("Return", "Enter"),
        ("cmd+a", "Meta+a"),
        ("ctrl+SHIFT+Esc", "Control+Shift+Escape"),
        ("page_down", "PageDown"),
        ("f5", "F5"),
        ("Control++", "Control++"),
        (" ", " "),
        ("Space", " "),
        ("a", "a"),
    ],
)
def test_key_names(spec: str, normalized: str) -> None:
    assert tap.normalize_key(spec) == normalized


@pytest.mark.parametrize(
    ("spec", "why"),
    [
        ("Control+V", "paste"),
        ("cmd+v", "paste"),
        ("Shift+Insert", "paste"),
        ("Meta+Q", "quitting or closing the window"),
        ("Control+W", "quitting or closing the window"),
        ("Alt+F4", "quitting or closing the window"),
        ("F12", "opening DevTools"),
        ("Control+Shift+I", "opening DevTools"),
        ("Meta+Alt+J", "opening DevTools"),
        ("Meta+Shift+N", "opening a window, a tab or a private window"),
        ("Control+T", "opening a window, a tab or a private window"),
        ("Control+U", "viewing the page source"),
    ],
)
def test_refused_keys(spec: str, why: str) -> None:
    assert tap.refused_key(spec) == why


@pytest.mark.parametrize("spec", ["Enter", "Control+A", "Meta+C", "Shift+Tab", "PageDown", "a", "Escape", "Control+F"])
def test_ordinary_keys_are_not_refused(spec: str) -> None:
    assert tap.refused_key(spec) is None


@pytest.mark.parametrize(
    ("spec", "action"),
    [
        ("Alt+ArrowLeft", "back"),
        ("Meta+[", "back"),
        ("alt+right", "forward"),
        ("F5", "reload"),
        ("Control+R", "reload"),
        ("Enter", None),
        ("Control+ArrowLeft", None),
        ("Alt+Shift+ArrowLeft", None),
    ],
)
def test_navigation_shortcuts(spec: str, action: str | None) -> None:
    assert tap.navigation_shortcut(spec) == action


def test_child_command_per_mode() -> None:
    dom = tap.child_command("dom", "http://127.0.0.1:9222", "/x/chrome-devtools-mcp")
    screen = tap.child_command("screen", "http://127.0.0.1:9222", "/x/chrome-devtools-mcp")
    for command in (dom, screen):
        assert command[:3] == ["/x/chrome-devtools-mcp", "--browserUrl", "http://127.0.0.1:9222"]
        assert "--no-page-id-routing" in command
        for flag in ("--usageStatistics=false", "--performanceCrux=false", "--categoryEmulation=false"):
            assert flag in command
        assert {"--categoryPerformance=false", "--categoryNetwork=false", "--categoryMemory=false"} <= set(command)
    assert "--experimentalVision" not in dom
    assert {"--javascriptEvaluation=false", "--fileNavigations=false"} <= set(dom)
    assert "--experimentalVision" in screen
    assert tap.CHILD_ENV == {
        "CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1",
        "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
    }
    with pytest.raises(ValueError, match="browser-url"):
        tap.child_command("dom", None)
    with pytest.raises(ValueError, match="no child"):
        tap.child_command("none", "http://127.0.0.1:1")


# --- the dom mode (D) ---


def test_dom_lists_the_thirteen_tools_with_the_dropped_parameters_removed(rig: Rig) -> None:
    client = rig.start("dom")
    tools = client.tools()
    assert [t["name"] for t in tools] == list(tap.DOM_TOOLS)
    assert len(tools) == 13
    by_name = {t["name"]: t for t in tools}
    assert set(by_name["new_page"]["inputSchema"]["properties"]) == {"url", "background"}
    assert "initScript" not in by_name["navigate_page"]["inputSchema"]["properties"]
    assert "filePath" not in by_name["take_snapshot"]["inputSchema"]["properties"]
    assert by_name["click"]["inputSchema"]["required"] == ["uid"]  # the rest of each schema is untouched
    assert by_name["type_text"]["inputSchema"]["properties"]["submitKey"] == {"type": "string"}
    client.close()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("new_page", {"url": "https://example.com", "isolatedContext": "ctx"}),
        ("new_page", {"url": "https://example.com", "isolatedContext": None}),
        ("navigate_page", {"type": "url", "url": "https://example.com", "initScript": "1"}),
        ("take_snapshot", {"filePath": "/data/x"}),
    ],
)
def test_dom_refuses_the_dropped_parameters(rig: Rig, name: str, arguments: dict[str, Any]) -> None:
    client = rig.start("dom")
    result = client.call(name, **arguments)
    assert result["isError"] is True and text_of(result).startswith("Refused:")
    assert rig.child_calls() == []
    client.close()


@pytest.mark.parametrize(
    "name", ["take_screenshot", "click_at", "evaluate_script", "drag", "upload_file", "screenshot"]
)
def test_dom_never_exposes_pixels_or_scripts(rig: Rig, name: str) -> None:
    client = rig.start("dom")
    assert name not in [t["name"] for t in client.tools()]
    result = client.call(name, x=1, y=1)
    assert result["isError"] is True and "no tool" in text_of(result)
    assert rig.child_calls() == []
    client.close()


@pytest.mark.parametrize("url", ["file:///etc/hosts", "javascript:alert(1)", "chrome://settings", "data:text/html,x"])
def test_dom_refuses_other_than_http_urls(rig: Rig, url: str) -> None:
    client = rig.start("dom")
    for name in ("navigate_page", "new_page"):
        result = client.call(name, url=url, **({"type": "url"} if name == "navigate_page" else {}))
        assert result["isError"] is True and "http" in text_of(result)
    assert rig.child_calls() == []
    client.close()


def test_dom_text_results_have_hosts_instead_of_urls(rig: Rig) -> None:
    client = rig.start("dom")
    result = client.call("take_snapshot")
    assert "isError" not in result
    text = text_of(result)
    assert 'url="www.example.com"' in text and "sxsrf" not in text
    assert len(text) < len(SNAPSHOT)
    # a result that is not a url="..." attribute is passed as the child wrote it
    assert "page https://long.example.com/a/b?c=d" in text_of(client.call("click", uid="1_1"))
    assert rig.child_calls()[0] == {"name": "take_snapshot", "arguments": {}}
    client.close()
    entry = next(e for e in rig.log_entries() if e.get("tool") == "take_snapshot")
    assert entry["chars_raw"] == len(SNAPSHOT) and entry["chars_out"] < entry["chars_raw"]
    assert entry["urls_rewritten"] == 4


def test_dom_arguments_reach_the_child_unchanged(rig: Rig) -> None:
    client = rig.start("dom")
    client.call("fill", uid="1_16", value="capital of Australia")
    client.call("navigate_page", type="url", url="https://www.example.com/x?y=1")
    assert rig.child_calls() == [
        {"name": "fill", "arguments": {"uid": "1_16", "value": "capital of Australia"}},
        {"name": "navigate_page", "arguments": {"type": "url", "url": "https://www.example.com/x?y=1"}},
    ]
    client.close()


def test_child_errors_come_back_as_tool_errors(rig: Rig) -> None:
    client = rig.start("dom")
    result = client.call("wait_for", text=["boom"])  # the child's own isError result
    assert result["isError"] is True and "Timed out" in text_of(result)
    result = client.call("hover", uid="1_1")  # a JSON-RPC error of the child
    assert result["isError"] is True and "Invalid params for hover" in text_of(result)
    assert "isError" not in client.call("take_snapshot")  # and the server goes on
    client.close()
    statuses = {e["tool"]: e["status"] for e in rig.log_entries() if e.get("event") == "call"}
    assert statuses == {"wait_for": "tool_error", "hover": "rpc_error", "take_snapshot": "ok"}


def test_a_dead_child_is_a_tool_error_and_is_started_again(rig: Rig) -> None:
    client = rig.start("dom", FAKE_CRASH_ON="hover")
    assert "isError" not in client.call("take_snapshot")
    result = client.call("hover", uid="1_1")  # the child exits in the middle of the call
    assert result["isError"] is True and "exited" in text_of(result)
    assert "isError" not in client.call("take_snapshot")
    assert len(rig.pids.read_text().split()) == 2
    client.close()


# --- the screen mode (S-cdp) ---


def test_screen_lists_exactly_five_bench_tools(rig: Rig) -> None:
    client = rig.start("screen")
    tools = client.tools()
    assert [t["name"] for t in tools] == ["screenshot", "click", "type_text", "press_keys", "scroll"]
    assert all(t["inputSchema"]["type"] == "object" and t["description"] for t in tools)
    assert rig.child_calls() == []  # listing needs no child call
    client.close()


def test_screenshot_is_one_jpeg_image_block_under_the_size_cap(rig: Rig) -> None:
    client = rig.start("screen", FAKE_SHOT_SIZES="80:400000,60:300000,40:200000,25:100000")
    result = client.call("screenshot")
    assert [b["type"] for b in result["content"]] == ["image"]  # no text block, no structuredContent
    assert "structuredContent" not in result and "isError" not in result
    block = result["content"][0]
    assert block["mimeType"] == "image/jpeg"
    data = base64.b64decode(block["data"])
    assert data[:2] == b"\xff\xd8" and len(data) <= 256 * 1024
    assert tap.jpeg_size(data) == (1280, 713)
    shots = [c["arguments"] for c in rig.child_calls() if c["name"] == "take_screenshot"]
    assert shots == [{"format": "jpeg", "quality": q} for q in (80, 60, 40)]  # the quality steps down until it fits
    client.close()
    entry = next(e for e in rig.log_entries() if e.get("tool") == "screenshot")
    assert entry["quality"] == 40 and entry["bytes"] == len(data) and entry["viewport"] == [1280, 713]
    assert entry["n_images"] == 1 and entry["chars_out"] == 0


def test_a_screenshot_that_never_fits_is_an_error(rig: Rig) -> None:
    client = rig.start("screen", FAKE_SHOT_SIZES="80:400000,60:390000,40:380000,25:370000")
    result = client.call("screenshot")
    assert result["isError"] is True and not image_blocks(result)
    client.close()


def test_click_maps_the_thousand_grid_to_viewport_pixels(rig: Rig) -> None:
    client = rig.start("screen", FAKE_VIEWPORT="800,600")
    client.call("screenshot")
    for x, y in ((0, 0), (500, 500), (1000, 1000), (125.5, 999)):
        assert text_of(client.call("click", x=x, y=y)) == "Clicked."
    clicks = [c["arguments"] for c in rig.child_calls() if c["name"] == "click_at"]
    assert clicks == [{"x": 0, "y": 0}, {"x": 400, "y": 300}, {"x": 799, "y": 599}, {"x": 100, "y": 599}]
    client.close()
    entry = [e for e in rig.log_entries() if e.get("tool") == "click"][1]
    assert entry["norm"] == [500.0, 500.0] and entry["css"] == [400, 300] and entry["viewport"] == [800, 600]


def test_the_viewport_comes_from_the_latest_screenshot(rig: Rig) -> None:
    viewport = rig.tmp / "viewport.txt"
    client = rig.start("screen", FAKE_VIEWPORT_FILE=str(viewport))
    viewport.write_text("1280,713")
    client.call("screenshot")
    viewport.write_text("800,600")  # the window changes, but the agent has only seen the old screenshot
    client.call("click", x=1000, y=1000)
    client.call("screenshot")
    client.call("click", x=1000, y=1000)
    clicks = [c["arguments"] for c in rig.child_calls() if c["name"] == "click_at"]
    assert clicks == [{"x": 1279, "y": 712}, {"x": 799, "y": 599}]
    client.close()


def test_the_image_size_is_the_viewport_when_the_script_answer_is_unreadable(rig: Rig) -> None:
    client = rig.start("screen", FAKE_VIEWPORT="not a size")
    client.call("screenshot")
    client.call("click", x=500, y=500)
    assert rig.child_calls()[-1] == {"name": "click_at", "arguments": {"x": 640, "y": 356}}  # from the 1280x713 JPEG
    client.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"x": -1, "y": 5},
        {"x": 5, "y": 1001},
        {"x": "5", "y": 5},
        {"x": 5},
        {"x": True, "y": 5},
        {"x": 5, "y": 5, "z": 1},
    ],
)
def test_click_refuses_bad_coordinates(rig: Rig, arguments: dict[str, Any]) -> None:
    client = rig.start("screen")
    client.call("screenshot")
    result = client.call("click", **arguments)
    assert result["isError"] is True and text_of(result).startswith("Refused:")
    assert "click_at" not in rig.child_call_names()
    client.close()


def test_click_needs_a_screenshot_first(rig: Rig) -> None:
    client = rig.start("screen")
    result = client.call("click", x=500, y=500)
    assert result["isError"] is True and "screenshot" in text_of(result)
    assert rig.child_calls() == []
    client.close()


def test_type_press_and_scroll_map_to_child_tools(rig: Rig) -> None:
    client = rig.start("screen")
    assert text_of(client.call("type_text", text="capital of Australia")) == "Typed."
    assert text_of(client.call("press_keys", keys=["Return", "cmd+a", "Control+Shift+Tab"])) == (
        "Pressed Return, cmd+a, Control+Shift+Tab."
    )
    assert text_of(client.call("press_keys", keys="Escape")) == "Pressed Escape."  # a bare string is one key
    assert text_of(client.call("scroll", direction="down", amount=2)) == "Scrolled down 2 screens."
    assert text_of(client.call("scroll", direction="up")) == "Scrolled up 1 screen."
    assert rig.child_calls() == [
        {"name": "type_text", "arguments": {"text": "capital of Australia"}},
        {"name": "press_key", "arguments": {"key": "Enter"}},
        {"name": "press_key", "arguments": {"key": "Meta+a"}},
        {"name": "press_key", "arguments": {"key": "Control+Shift+Tab"}},
        {"name": "press_key", "arguments": {"key": "Escape"}},
        {"name": "press_key", "arguments": {"key": "PageDown"}},
        {"name": "press_key", "arguments": {"key": "PageDown"}},
        {"name": "press_key", "arguments": {"key": "PageUp"}},
    ]
    client.close()


def test_back_forward_and_reload_go_through_navigate_page(rig: Rig) -> None:
    client = rig.start("screen")
    client.call("press_keys", keys=["Alt+ArrowLeft", "alt+right", "F5", "Enter"])
    assert rig.child_calls() == [
        {"name": "navigate_page", "arguments": {"type": "back"}},
        {"name": "navigate_page", "arguments": {"type": "forward"}},
        {"name": "navigate_page", "arguments": {"type": "reload"}},
        {"name": "press_key", "arguments": {"key": "Enter"}},
    ]
    client.close()


def test_screen_results_do_not_repeat_what_the_child_said(rig: Rig) -> None:
    """The child's text names URLs and page titles; only the screenshot may show the page to the S arm."""
    client = rig.start("screen")
    client.call("screenshot")
    for result in (
        client.call("click", x=1, y=1),
        client.call("type_text", text="x"),
        client.call("press_keys", keys=["Enter"]),
        client.call("scroll", direction="down"),
    ):
        assert "long.example.com" not in text_of(result) and "Successfully" not in text_of(result)
    client.close()


def test_a_click_that_opens_a_tab_shows_the_new_tab(rig: Rig) -> None:
    client = rig.start("screen", FAKE_OPEN_PAGE="1")
    client.call("screenshot")
    result = client.call("click", x=100, y=100)
    assert text_of(result) == "Clicked."  # nothing of "## Opened pages" reaches the agent
    assert rig.child_calls()[-1] == {"name": "select_page", "arguments": {"pageId": 2, "bringToFront": True}}
    client.close()


@pytest.mark.parametrize(
    "keys", [["Control+V"], ["Meta+Q"], ["F12"], ["Meta+Alt+I"], ["Control+Shift+N"], ["Enter", "cmd+v"]]
)
def test_press_keys_refuses_paste_quit_devtools_and_incognito(rig: Rig, keys: list[str]) -> None:
    client = rig.start("screen")
    result = client.call("press_keys", keys=keys)
    assert result["isError"] is True and "is not allowed" in text_of(result)
    assert rig.child_calls() == []  # nothing was pressed, not even the keys before the refused one
    client.close()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("press_keys", {"keys": []}),
        ("press_keys", {"keys": ["a"] * 11}),
        ("press_keys", {"keys": [1]}),
        ("press_keys", {}),
        ("type_text", {"text": ""}),
        ("type_text", {"text": "x", "submit": True}),
        ("scroll", {"direction": "left"}),
        ("scroll", {"direction": "down", "amount": 0}),
        ("scroll", {"direction": "down", "amount": 11}),
        ("scroll", {"direction": "down", "amount": 1.5}),
    ],
)
def test_screen_tools_refuse_malformed_arguments(rig: Rig, name: str, arguments: dict[str, Any]) -> None:
    client = rig.start("screen")
    result = client.call(name, **arguments)
    assert result["isError"] is True and text_of(result).startswith("Refused:")
    assert rig.child_calls() == []
    client.close()


@pytest.mark.parametrize(
    "name",
    [
        "navigate_page",
        "take_snapshot",
        "take_screenshot",
        "click_at",
        "evaluate_script",
        "press_key",
        "fill",
        "new_page",
    ],
)
def test_screen_refuses_every_other_tool(rig: Rig, name: str) -> None:
    client = rig.start("screen")
    result = client.call(name, url="https://example.com", x=1, y=1)
    assert result["isError"] is True and "no tool" in text_of(result)
    assert rig.child_calls() == []
    client.close()


# --- the none mode (C), the log, and the shutdown ---


def test_none_has_no_tools_and_starts_no_child(rig: Rig) -> None:
    client = rig.start("none")
    assert client.tools() == []
    result = client.call("take_snapshot")
    assert result["isError"] is True and "no tool" in text_of(result)
    client.close()
    assert not rig.pids.exists() and rig.child_calls() == []


def test_every_call_is_logged_with_its_status(rig: Rig) -> None:
    client = rig.start("dom")
    client.tools()
    client.call("take_snapshot")
    client.call("take_screenshot")
    client.call("new_page", url="https://example.com", isolatedContext="x")
    client.call("hover", uid="1_1")
    client.close()
    entries = rig.log_entries()
    assert [e["event"] for e in entries if e["event"] != "call"] == ["start", "initialize", "tools_list", "stop"]
    calls = [e for e in entries if e["event"] == "call"]
    assert [(e["tool"], e["status"]) for e in calls] == [
        ("take_snapshot", "ok"),
        ("take_screenshot", "refused"),
        ("new_page", "refused"),
        ("hover", "rpc_error"),
    ]
    assert calls[2]["args"] == {"url": "https://example.com", "isolatedContext": "x"}
    assert all(e["mode"] == "dom" and e["dur_s"] >= 0 and e["t"] > 0 for e in calls)


def test_the_log_comes_from_the_environment_too(rig: Rig, tmp_path: Path) -> None:
    log = tmp_path / "from-env.jsonl"
    client = Client(
        ["--", sys.executable, str(rig.script)], {**rig.env, "WEB_TAP_MODE": "dom", "WEB_TAP_LOG": str(log)}
    )
    rig.clients.append(client)
    client.call("take_snapshot")
    client.close()
    assert [json.loads(line).get("tool") for line in log.read_text().splitlines()][2:3] == ["take_snapshot"]


def test_the_child_is_shut_down_when_stdin_closes(rig: Rig) -> None:
    client = rig.start("screen")
    client.call("type_text", text="x")
    (child_pid,) = [int(p) for p in rig.pids.read_text().split()]
    os.kill(child_pid, 0)  # it runs
    client.close()  # EOF on the tap's stdin
    assert client.proc.returncode == 0
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the child is still running after the tap exited")


def test_protocol_basics(rig: Rig) -> None:
    client = rig.start("none")
    assert client.request("ping")["result"] == {}
    assert client.request("resources/list")["error"]["code"] == -32601
    assert client.request("tools/call", {})["error"]["code"] == -32602
    client.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}})  # a notification: no answer
    client.proc.stdin.write("this is not json\n")  # type: ignore[union-attr]
    client.proc.stdin.flush()  # type: ignore[union-attr]
    assert json.loads(client.lines.get(timeout=10))["error"]["code"] == -32700
    assert client.request("ping")["result"] == {}  # still serving
    client.close()
