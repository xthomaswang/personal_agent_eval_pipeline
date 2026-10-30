"""Tests of the web tap's cua mode (S-cua): the five screen tools served by cua-driver on a headed Chrome's window.

cua-driver is replaced by a fake MCP server script, so no app, no window and no permission is needed. The fake
returns a real PNG, so the PNG-to-JPEG step runs ``sips`` (macOS).

    cd eval
    uv run pytest -q agent_ability/tests/test_web_tap_cua.py
"""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest
from test_web_tap import Client, text_of
from web import tap

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="the cua mode re-encodes with macOS's sips")

CHROME_PID = 4242
MAIN = {"window_id": 7, "layer": 0, "on_current_space": True, "bounds": {"x": 0, "y": 33, "width": 1280, "height": 800}}
SMALL = {"window_id": 8, "layer": 0, "on_current_space": True, "bounds": {"x": 9, "y": 9, "width": 400, "height": 300}}
ELSEWHERE = {
    "window_id": 9,
    "layer": 0,
    "on_current_space": False,
    "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
}
SECRET = "Eiffel Tower - Wikipedia"  # what cua-driver's text names; the screen arm must never see it


def png(width: int, height: int) -> bytes:
    """A valid grey PNG (stdlib only)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\x80\x80\x80" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


FAKE_CUA = r'''
"""A fake cua-driver MCP server: one JSON-RPC line per message; every tools/call is logged."""
import json, os, sys

CALLS = os.environ["FAKE_CUA_CALLS"]
STATE = os.environ["FAKE_CUA_STATE"]  # JSON: windows, space, refuse (tool names), shot_error
PNG = open(os.environ["FAKE_CUA_PNG"], "rb").read()
SECRET = os.environ["FAKE_SECRET"]


def reply(rid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": rid}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


import base64
for line in sys.stdin:
    msg = json.loads(line)
    rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if rid is None:
        continue
    if method == "initialize":
        reply(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake-cua"}})
        continue
    name, args = params.get("name"), params.get("arguments") or {}
    with open(CALLS, "a") as f:
        f.write(json.dumps({"name": name, "args": args}) + "\n")
    state = json.load(open(STATE))
    if name == "list_windows":
        reply(rid, {"content": [{"type": "text", "text": SECRET}],
                    "structuredContent": {"windows": state["windows"], "current_space_id": state["space"]}})
    elif name == "get_window_state":
        if state.get("shot_error"):
            reply(rid, {"content": [{"type": "text", "text": "No content produced"}], "isError": True})
        else:
            reply(rid, {"content": [{"type": "image", "mimeType": "image/png", "data": base64.b64encode(PNG).decode()},
                                    {"type": "text", "text": SECRET}]})
    else:
        effect = "refused" if name in state.get("refuse", []) else "unverifiable"
        reply(rid, {"content": [{"type": "text", "text": "Pressed link " + SECRET}],
                    "structuredContent": {"effect": effect, "route": "accessibility"}})
'''


class CuaRig:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.calls = tmp_path / "cua-calls.jsonl"
        self.state = tmp_path / "state.json"
        self.log = tmp_path / "tap-log.jsonl"
        (tmp_path / "shot.png").write_bytes(png(1280, 800))
        (tmp_path / "fake_cua.py").write_text(FAKE_CUA)
        self.set(windows=[MAIN, SMALL, ELSEWHERE], space=400)
        self.env = {
            **os.environ,
            "FAKE_CUA_CALLS": str(self.calls),
            "FAKE_CUA_STATE": str(self.state),
            "FAKE_CUA_PNG": str(tmp_path / "shot.png"),
            "FAKE_SECRET": SECRET,
        }
        self.clients: list[Client] = []

    def set(self, **state: Any) -> None:
        current = json.loads(self.state.read_text()) if self.state.exists() else {}
        self.state.write_text(json.dumps({**current, **state}))

    def start(self) -> Client:
        argv = ["--mode", "cua", "--log", str(self.log), "--chrome-pid", str(CHROME_PID)]
        client = Client([*argv, "--", sys.executable, str(self.tmp / "fake_cua.py")], self.env)
        self.clients.append(client)
        return client

    def calls_named(self, *names: str) -> list[dict[str, Any]]:
        lines = self.calls.read_text().splitlines() if self.calls.exists() else []
        return [c for c in map(json.loads, lines) if not names or c["name"] in names]


@pytest.fixture
def rig(tmp_path: Path):
    rig = CuaRig(tmp_path)
    yield rig
    for client in rig.clients:
        if client.proc.poll() is None:
            client.proc.kill()


# --- pure helpers ---


@pytest.mark.parametrize(
    ("spec", "combo"),
    [
        ("Enter", ["return"]),
        ("Return", ["return"]),
        ("Escape", ["escape"]),
        ("Tab", ["tab"]),
        ("Backspace", ["delete"]),
        ("ArrowDown", ["down"]),
        ("PageDown", ["pagedown"]),
        ("F3", ["f3"]),
        ("A", ["a"]),
        ("Control+a", ["ctrl", "a"]),
        ("cmd+f", ["cmd", "f"]),
        ("Shift+Tab", ["shift", "tab"]),
        ("Alt+ArrowLeft", ["cmd", "["]),
        ("Meta+]", ["cmd", "]"]),
        ("F5", ["cmd", "r"]),
    ],
)
def test_keys_become_cua_driver_names(spec: str, combo: list[str]) -> None:
    assert tap.cua_key(spec) == combo


def test_the_cua_child_is_cua_drivers_mcp_server() -> None:
    assert tap.child_command("cua", None, cua_bin="/x/cua-driver") == ["/x/cua-driver", "mcp"]


def test_the_cua_mode_needs_a_chrome_pid(tmp_path: Path) -> None:
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "WEB_TAP_CHROME_PID"}
    out = subprocess.run(
        [sys.executable, str(Path(tap.__file__)), "--mode", "cua"], capture_output=True, text=True, env=env, timeout=30
    )
    assert out.returncode == 2 and "--chrome-pid" in out.stderr


# --- the tools ---


def test_cua_offers_the_same_five_tools_as_the_cdp_screen_arm(rig: CuaRig) -> None:
    assert rig.start().tools() == tap.SCREEN_TOOL_DEFS


def test_a_screenshot_is_one_jpeg_of_the_largest_window_on_screen(rig: CuaRig) -> None:
    client = rig.start()
    result = client.call("screenshot")
    assert [b["type"] for b in result["content"]] == ["image"]  # no text: cua-driver's names stay out
    block = result["content"][0]
    data = base64.b64decode(block["data"])
    assert block["mimeType"] == "image/jpeg" and tap.jpeg_size(data) == (1280, 800)
    assert len(data) <= tap.MAX_IMAGE_BYTES
    (shot,) = rig.calls_named("get_window_state")
    session = shot["args"]["session"]
    assert shot["args"] == {
        "pid": CHROME_PID,
        "window_id": MAIN["window_id"],  # not the small window, not the larger one on another Space
        "include_accessibility_tree": False,
        "max_image_dimension": tap.CUA_IMAGE_DIM,
        "session": session,
    }
    assert rig.calls_named("set_agent_cursor_enabled") == [
        {"name": "set_agent_cursor_enabled", "args": {"enabled": False, "session": session}}
    ]
    client.call("screenshot")
    assert len(rig.calls_named("set_agent_cursor_enabled")) == 1  # once per tap
    assert {c["args"]["session"] for c in rig.calls_named()} == {session}


def test_a_failed_screenshot_is_a_tool_error(rig: CuaRig) -> None:
    rig.set(shot_error=True)
    result = rig.start().call("screenshot")
    assert result.get("isError") and SECRET not in text_of(result)


def test_click_needs_a_screenshot_and_maps_the_grid_to_its_pixels(rig: CuaRig) -> None:
    client = rig.start()
    refused = client.call("click", x=500, y=500)
    assert refused.get("isError") and "screenshot first" in text_of(refused)
    assert rig.calls_named("click") == []
    client.call("screenshot")
    result = client.call("click", x=500, y=250)
    assert text_of(result) == "Clicked." and not result.get("isError")
    (click,) = rig.calls_named("click")
    assert {k: click["args"][k] for k in ("pid", "window_id", "x", "y")} == {
        "pid": CHROME_PID,
        "window_id": MAIN["window_id"],
        "x": 640,
        "y": 200,
    }
    entry = [e for e in map(json.loads, rig.log.read_text().splitlines()) if e.get("tool") == "click"][-1]
    assert entry["px"] == [640, 200] and entry["route"] == "accessibility"


def test_the_screen_arm_checks_apply(rig: CuaRig) -> None:
    client = rig.start()
    client.call("screenshot")
    for name, args in [
        ("click", {"x": 1001, "y": 3}),
        ("click", {"x": 1, "y": 2, "button": "right"}),
        ("type_text", {"text": ""}),
        ("press_keys", {"keys": ["Enter", "cmd+v"]}),  # nothing is pressed when one key is refused
        ("scroll", {"direction": "left"}),
        ("screenshot", {"full": True}),
    ]:
        assert client.call(name, **args).get("isError"), (name, args)
    assert rig.calls_named("click", "type_text", "press_key", "hotkey", "scroll") == []


def test_keys_text_and_scroll_reach_the_window(rig: CuaRig) -> None:
    client = rig.start()
    client.call("screenshot")
    assert text_of(client.call("type_text", text="eiffel tower")) == "Typed."
    assert text_of(client.call("press_keys", keys=["Enter", "Control+a", "Alt+ArrowLeft"])) == (
        "Pressed Enter, Control+a, Alt+ArrowLeft."
    )
    assert text_of(client.call("scroll", direction="down", amount=2)) == "Scrolled down 2 screens."
    sent = [
        {k: v for k, v in c["args"].items() if k not in ("session", "pid", "window_id")}
        | {"tool": c["name"], "target": (c["args"]["pid"], c["args"]["window_id"])}
        for c in rig.calls_named("type_text", "press_key", "hotkey", "scroll")
    ]
    target = (CHROME_PID, MAIN["window_id"])
    assert sent == [
        {"tool": "type_text", "text": "eiffel tower", "target": target},
        {"tool": "press_key", "key": "return", "target": target},
        {"tool": "hotkey", "keys": ["ctrl", "a"], "target": target},
        {"tool": "hotkey", "keys": ["cmd", "["], "target": target},
        {"tool": "scroll", "direction": "down", "by": "page", "amount": 2, "target": target},
    ]


def test_actions_are_refused_while_the_browsers_desktop_is_not_on_screen(rig: CuaRig) -> None:
    client = rig.start()
    client.call("screenshot")
    rig.set(space=111)  # the user switched to another desktop
    for name, args in [("click", {"x": 5, "y": 5}), ("type_text", {"text": "a"}), ("screenshot", {})]:
        result = client.call(name, **args)
        assert result.get("isError") and "not on screen" in text_of(result)
    assert rig.calls_named("click", "type_text", "get_window_state")[1:] == []  # only the first screenshot
    rig.set(space=400)
    assert not client.call("click", x=5, y=5).get("isError")


def test_a_refused_action_is_an_error_without_the_drivers_text(rig: CuaRig) -> None:
    rig.set(refuse=["click"])
    client = rig.start()
    client.call("screenshot")
    result = client.call("click", x=10, y=10)
    assert result.get("isError") and SECRET not in text_of(result)
    assert SECRET not in text_of(client.call("type_text", text="x"))


def test_a_closed_window_needs_a_new_screenshot(rig: CuaRig) -> None:
    client = rig.start()
    client.call("screenshot")
    rig.set(windows=[SMALL])
    result = client.call("click", x=1, y=1)
    assert result.get("isError") and "take a screenshot" in text_of(result)
    client.call("screenshot")
    assert rig.calls_named("get_window_state")[-1]["args"]["window_id"] == SMALL["window_id"]
