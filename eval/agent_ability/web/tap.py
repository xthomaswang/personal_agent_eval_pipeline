#!/usr/bin/env python3
"""The web tap: a stdlib stdio MCP relay that is the one MCP server (named ``web``) of every arm of the web suite.

    python3 tap.py --mode dom    --log calls.jsonl --browser-url http://127.0.0.1:9222
    python3 tap.py --mode screen --log calls.jsonl --browser-url http://127.0.0.1:9222
    python3 tap.py --mode cua    --log calls.jsonl --chrome-pid 4242
    python3 tap.py --mode none   --log calls.jsonl

The modes are the arms:

- ``dom`` (D): chrome-devtools-mcp is the child, attached to the run's Chrome. Only 13 of its tools are exposed
  (``DOM_TOOLS``). ``new_page.isolatedContext`` (it bypasses History) is dropped from the schema and refused; so are
  ``navigate_page.initScript`` (script injection) and ``take_snapshot.filePath``. Every ``url="<full url>"`` in a text
  result becomes ``url="<host>"``. ``take_screenshot``, ``click_at`` and ``evaluate_script`` are never exposed.
- ``screen`` (S-cdp): chrome-devtools-mcp with ``--experimentalVision`` is the child; the agent sees five bench-defined
  tools (``screenshot``, ``click``, ``type_text``, ``press_keys``, ``scroll``). Coordinates are 0-1000 on the latest
  screenshot and mapped to the CSS pixels of the viewport. Screenshots are JPEGs of at most 256 KB, returned as an
  image block only. Nothing else is exposed, and the child's text is never shown (it holds URLs and page titles).
- ``cua`` (S-cua): the same five tools, served by cua-driver (``cua-driver mcp``, which acts through CuaDriver.app
  and its permissions) on the window of a headed Chrome given by ``--chrome-pid``. Screenshots are of the whole
  window, browser UI included; input is OS-level and sent in the background (no pointer move, no focus change). The
  window must be on the Space on screen: off-Space windows give no screenshot, so a call is refused while the Space
  that was current at the first screenshot is not.
- ``none`` (C): no tools.

Every tools/call is appended to the JSONL file given by ``--log`` (or ``WEB_TAP_LOG``). ``tools/list`` is answered from
the filtered set; errors of the child come back as tool errors; the child is shut down when stdin closes.

The child command is built from ``--mcp-bin`` and ``--browser-url`` (``--cua-bin`` in the cua mode); anything after ``--`` replaces it (the tests put
a fake MCP server there). This file uses the standard library only, and runs on any Python 3.9 or later.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Any

SERVER_NAME = "web"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MCP_BIN = os.path.join(HERE, "mcp", "node_modules", ".bin", "chrome-devtools-mcp")

DOM_TOOLS = (
    "navigate_page",
    "new_page",
    "list_pages",
    "select_page",
    "close_page",
    "wait_for",
    "take_snapshot",
    "click",
    "hover",
    "fill",
    "type_text",
    "press_key",
    "handle_dialog",
)
# Parameters taken out of the schema; a call that names one (even with a null) is refused.
DOM_DROPPED_PARAMS = {
    "new_page": ("isolatedContext",),
    "navigate_page": ("initScript",),
    "take_snapshot": ("filePath",),
}

# Child flags. The category flags drop tools the suite never uses; usage statistics and the CrUX lookup are off.
COMMON_CHILD_FLAGS = (
    "--no-page-id-routing",
    "--usageStatistics=false",
    "--performanceCrux=false",
    "--categoryPerformance=false",
    "--categoryNetwork=false",
    "--categoryMemory=false",
    "--categoryEmulation=false",
)
DOM_CHILD_FLAGS = ("--javascriptEvaluation=false", "--fileNavigations=false")
# The screen child keeps script evaluation: the tap itself reads the viewport size with it. The agent never can.
SCREEN_CHILD_FLAGS = ("--experimentalVision", "--screenshotMaxWidth=1280", "--screenshotMaxHeight=800")
CHILD_ENV = {"CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1", "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1"}

CUA_BUNDLE_ID = "com.trycua.driver"
CUA_EXECUTABLE = os.path.join("Contents", "MacOS", "cua-driver")
CUA_IMAGE_DIM = 1280  # the long edge of a cua screenshot: a 1280x800 window comes back at 1280x800
MAX_IMAGE_BYTES = 256 * 1024
JPEG_QUALITIES = (80, 60, 40, 25)
SCROLL_SETTLE_S = 0.5  # Chrome animates PageDown; a screenshot taken at once would catch it half way
MAX_SCROLL_AMOUNT = 10
MAX_KEYS = 10
CALL_TIMEOUT_S = 240.0
START_TIMEOUT_S = 60.0
LOG_ARG_CHARS = 2000
LOG_HEAD_CHARS = 300

ALLOWED_URL_SCHEMES = ("http", "https")


# --- small pure helpers ---


def default_cua_bin() -> str:
    """The cua-driver binary: the app with cua-driver's bundle id (Spotlight); one in /Applications wins."""
    try:
        out = subprocess.run(
            ["mdfind", f"kMDItemCFBundleIdentifier == '{CUA_BUNDLE_ID}'"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    apps = [line for line in out.splitlines() if line.endswith(".app")]
    if not apps:
        raise ValueError(f"no app with bundle id {CUA_BUNDLE_ID}; pass --cua-bin")
    app = min(apps, key=lambda a: (os.path.dirname(a) != "/Applications", len(a), a))
    return os.path.join(app, CUA_EXECUTABLE)


def child_command(
    mode: str, browser_url: str | None, mcp_bin: str | None = None, cua_bin: str | None = None
) -> list[str]:
    """The default command line of the child for a mode: chrome-devtools-mcp, or cua-driver's MCP server."""
    if mode == "cua":
        return [cua_bin or default_cua_bin(), "mcp"]
    if mode not in ("dom", "screen"):
        raise ValueError(f"mode {mode!r} has no child")
    if not browser_url:
        raise ValueError("--browser-url is required for the dom and screen modes")
    flags = DOM_CHILD_FLAGS if mode == "dom" else SCREEN_CHILD_FLAGS
    return [mcp_bin or DEFAULT_MCP_BIN, "--browserUrl", browser_url, *COMMON_CHILD_FLAGS, *flags]


_URL_ATTR = re.compile(r'url="([^"]*)"')


def host_of(url: str) -> str | None:
    """The host (and port, if any) of an absolute URL, lower-cased; None if the URL has no host."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    netloc = parts.netloc.rpartition("@")[2]
    return netloc.lower() or None


def rewrite_urls(text: str) -> tuple[str, int]:
    """Replace each ``url="<full url>"`` by ``url="<host>"``. URLs without a host (about:blank) are kept; data: and
    blob: URLs, which can be huge, are cut to their scheme. Returns the new text and how many were rewritten."""
    count = 0

    def sub(match: re.Match[str]) -> str:
        nonlocal count
        url = match.group(1)
        host = host_of(url)
        if host is not None:
            new = host
        elif url.lower().startswith(("data:", "blob:", "javascript:")):
            new = url.split(":", 1)[0].lower() + ":"
        else:
            return match.group(0)
        count += 1
        return f'url="{new}"'

    return _URL_ATTR.sub(sub, text), count


def url_allowed(url: Any) -> bool:
    """Navigations go to http(s) pages (or about:blank) only; a bare ``host`` without a scheme is left to the child."""
    if not isinstance(url, str):
        return True  # the child's schema check rejects it
    stripped = url.strip().lower()
    if stripped == "about:blank":
        return True
    try:
        scheme = urllib.parse.urlsplit(stripped).scheme
    except ValueError:
        return False
    return scheme == "" or scheme in ALLOWED_URL_SCHEMES


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of a JPEG from its first SOFn marker, or None."""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        i += 2 + length
    return None


def map_point(x: float, y: float, viewport: tuple[int, int]) -> tuple[int, int]:
    """0-1000 coordinates on a screenshot to CSS pixels of the viewport (clamped to its last pixel)."""
    width, height = viewport
    px = min(width - 1, max(0, round(x / 1000.0 * width)))
    py = min(height - 1, max(0, round(y / 1000.0 * height)))
    return px, py


_MODIFIER_NAMES = {
    "ctrl": "Control",
    "control": "Control",
    "cmd": "Meta",
    "command": "Meta",
    "meta": "Meta",
    "super": "Meta",
    "win": "Meta",
    "alt": "Alt",
    "option": "Alt",
    "opt": "Alt",
    "shift": "Shift",
}
_KEY_NAMES = {
    "return": "Enter",
    "enter": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "del": "Delete",
    "delete": "Delete",
    "backspace": "Backspace",
    "tab": "Tab",
    "space": " ",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "pagedown": "PageDown",
    "page_down": "PageDown",
    "pgdn": "PageDown",
    "pageup": "PageUp",
    "page_up": "PageUp",
    "pgup": "PageUp",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
}


def key_parts(spec: str) -> list[str]:
    """Split ``Control+Shift+A`` into its parts; a trailing ``+`` is the plus key (``Control++``, ``+``)."""
    if spec == "+":
        return ["+"]
    if spec.endswith("++"):
        return [*spec[:-2].split("+"), "+"]
    return spec.split("+")


def normalize_key(spec: str) -> str:
    """``cmd+a`` -> ``Meta+a``, ``Return`` -> ``Enter``: the names the child understands. Unknown names pass through."""
    out = []
    for part in key_parts(spec):
        part = part if part == " " else part.strip()
        low = part.lower()
        if low in _MODIFIER_NAMES:
            out.append(_MODIFIER_NAMES[low])
        elif low in _KEY_NAMES:
            out.append(_KEY_NAMES[low])
        elif re.fullmatch(r"f\d{1,2}", low):
            out.append(low.upper())
        else:
            out.append(part)
    return "+".join(out)


def split_key(spec: str) -> tuple[frozenset[str], str]:
    """(modifiers, lower-cased key) of a key spec."""
    parts = key_parts(normalize_key(spec))
    mods = frozenset(p for p in parts[:-1] if p in ("Control", "Meta", "Alt", "Shift"))
    return mods, parts[-1].lower()


def refused_key(spec: str) -> str | None:
    """Why a key combination is refused (paste, quit, DevTools, incognito and new windows), or None."""
    mods, key = split_key(spec)
    command = bool(mods & {"Control", "Meta"})
    if (command and key == "v") or (mods == {"Shift"} and key == "insert"):
        return "paste"
    if (command and key in ("q", "w")) or (mods == {"Alt"} and key == "f4"):
        return "quitting or closing the window"
    if (
        key == "f12"
        or (command and "Shift" in mods and key in ("i", "j", "c"))
        or (mods >= {"Meta", "Alt"} and key in ("i", "j", "c"))
    ):
        return "opening DevTools"
    if command and key in ("n", "t"):
        return "opening a window, a tab or a private window"
    if command and key == "u":
        return "viewing the page source"
    return None


# Browser shortcuts that Chrome handles in its own UI, not in the page: CDP key events never reach them, so the tap
# maps the usual ones to the child's navigate_page.
_NAV_SHORTCUTS = {
    ("Alt", "arrowleft"): "back",
    ("Meta", "arrowleft"): "back",
    ("Meta", "["): "back",
    ("Alt", "arrowright"): "forward",
    ("Meta", "arrowright"): "forward",
    ("Meta", "]"): "forward",
    ("Control", "r"): "reload",
    ("Meta", "r"): "reload",
}


def navigation_shortcut(spec: str) -> str | None:
    """``back``, ``forward`` or ``reload`` for the shortcuts of ``_NAV_SHORTCUTS`` (and F5), else None."""
    mods, key = split_key(spec)
    if len(mods) == 1:
        return _NAV_SHORTCUTS.get((next(iter(mods)), key))
    return "reload" if not mods and key == "f5" else None


# cua-driver's names for keys and modifiers, and Chrome's own shortcuts on macOS for the navigation keys.
_CUA_MODIFIERS = {"Control": "ctrl", "Meta": "cmd", "Alt": "option", "Shift": "shift"}
_CUA_KEYS = {
    "Enter": "return",
    "Escape": "escape",
    "Tab": "tab",
    " ": "space",
    "Backspace": "delete",
    "Delete": "delete",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Home": "home",
    "End": "end",
}
_CUA_NAV = {"back": ["cmd", "["], "forward": ["cmd", "]"], "reload": ["cmd", "r"]}


def cua_key(spec: str) -> list[str]:
    """A key spec as cua-driver's names, modifiers first: ``Control+a`` -> ``["ctrl", "a"]``, ``Return`` ->
    ``["return"]``. The navigation shortcuts become Chrome's own (``Alt+ArrowLeft`` -> ``["cmd", "["]``)."""
    nav = navigation_shortcut(spec)
    if nav:
        return list(_CUA_NAV[nav])
    parts = key_parts(normalize_key(spec))
    mods = [_CUA_MODIFIERS[p] for p in parts[:-1] if p in _CUA_MODIFIERS]
    key = parts[-1]
    if key in _CUA_KEYS:
        key = _CUA_KEYS[key]
    elif re.fullmatch(r"F\d{1,2}", key) or len(key) == 1:
        key = key.lower()
    return [*mods, key]


def png_to_jpeg(png: bytes, quality: int, workdir: str) -> bytes:
    """Re-encode a PNG as a JPEG with macOS's ``sips`` (the tap has no image library)."""
    src = os.path.join(workdir, "shot.png")
    dst = os.path.join(workdir, "shot.jpg")
    with open(src, "wb") as f:
        f.write(png)
    subprocess.run(
        ["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality), src, "--out", dst],
        capture_output=True,
        check=True,
        timeout=30,
    )
    with open(dst, "rb") as f:
        return f.read()


# --- results ---


def text_result(text: str, error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if error:
        result["isError"] = True
    return result


def result_text(result: dict[str, Any]) -> str:
    return "\n".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")


def first_image(result: dict[str, Any]) -> dict[str, Any] | None:
    return next((b for b in result.get("content", []) if b.get("type") == "image"), None)


class Refused(Exception):
    """A call the tap does not pass on."""


# --- the child MCP server ---


class ChildError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Pending:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: ChildError | None = None


class Child:
    """A stdio MCP server as a subprocess, spoken to as a client. Thread-safe; started on first use."""

    def __init__(self, cmd: list[str], env: dict[str, str] | None = None, stderr_path: str | None = None) -> None:
        self.cmd = cmd
        self.env = env
        self.stderr_path = stderr_path
        self.proc: subprocess.Popen[str] | None = None
        self._start_lock = threading.Lock()
        self._state_lock = threading.Lock()  # the pending table and the ids
        self._write_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 0
        self.starts = 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_started(self) -> None:
        with self._start_lock:
            if not self.alive():
                self._start()

    def _start(self) -> None:
        stderr: Any = subprocess.DEVNULL
        if self.stderr_path:
            stderr = open(self.stderr_path, "ab")  # noqa: SIM115 - closed below, once the child has its copy
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=self.env,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise ChildError(-32000, f"cannot start the browser server: {exc}") from exc
        finally:
            if self.stderr_path:
                stderr.close()
        self.starts += 1
        threading.Thread(target=self._read_loop, args=(self.proc,), daemon=True).start()
        init = {
            "protocolVersion": SUPPORTED_PROTOCOLS[0],
            "capabilities": {},
            "clientInfo": {"name": SERVER_NAME + "-tap", "version": SERVER_VERSION},
        }
        self._request("initialize", init, START_TIMEOUT_S)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _read_loop(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # the child must only speak JSON-RPC on stdout; anything else is ignored
            if not isinstance(msg, dict):
                continue
            if "method" in msg and "id" in msg:  # a request from the child: this client offers nothing
                with contextlib.suppress(ChildError):
                    self._send(
                        {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "not supported"}}
                    )
                continue
            with self._state_lock:
                slot = self._pending.pop(msg.get("id"), None)
            if slot is None:
                continue
            if "error" in msg:
                err = msg["error"] if isinstance(msg["error"], dict) else {}
                slot.error = ChildError(err.get("code", -32603), str(err.get("message", msg["error"])))
            else:
                slot.result = msg.get("result", {})
            slot.event.set()
        with self._state_lock:  # the child is gone: fail everything still waiting
            slots, self._pending = list(self._pending.values()), {}
        for slot in slots:
            slot.error = ChildError(-32000, "the browser server exited")
            slot.event.set()

    def _send(self, message: dict[str, Any]) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise ChildError(-32000, "the browser server is not running")
        try:
            with self._write_lock:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise ChildError(-32000, f"cannot write to the browser server: {exc}") from exc

    def _request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        slot = _Pending()
        with self._state_lock:
            self._next_id += 1
            rid = self._next_id
            self._pending[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            if not slot.event.wait(timeout):
                raise ChildError(-32001, f"the browser server did not answer {method} in {timeout:.0f} s")
        except ChildError:
            with self._state_lock:
                self._pending.pop(rid, None)
            raise
        if slot.error is not None:
            raise slot.error
        return slot.result or {}

    def request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_S
    ) -> dict[str, Any]:
        self.ensure_started()
        return self._request(method, params or {}, timeout)

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else {}, 60.0)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = CALL_TIMEOUT_S) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout)

    def close(self) -> None:
        """Close the child's stdin (an MCP server exits on that), then terminate and kill it if it stays."""
        proc = self.proc
        if proc is None:
            return
        with contextlib.suppress(OSError):
            if proc.stdin:
                proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


# --- the log ---


class CallLog:
    def __init__(self, path: str | None, mode: str) -> None:
        self.mode = mode
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8", buffering=1) if path else None  # noqa: SIM115

    def write(self, **fields: Any) -> None:
        if self._file is None:
            return
        entry = {"t": round(time.time(), 3), "mode": self.mode, **fields}
        with self._lock:
            self._file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > LOG_ARG_CHARS:
        return value[:LOG_ARG_CHARS] + f"...[{len(value)} chars]"
    if isinstance(value, dict):
        return {k: shorten(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v) for v in value]
    return value


# --- tool sets ---

SCREEN_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of the browser window (a JPEG). The click tool's coordinates refer to the latest "
            "screenshot: 0,0 is its top-left corner and 1000,1000 its bottom-right corner."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "click",
        "description": (
            "Click at a point of the latest screenshot. x runs from 0 (left edge) to 1000 (right edge) and y from "
            "0 (top edge) to 1000 (bottom edge). Take a screenshot first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": 0, "maximum": 1000, "description": "0 (left) to 1000 (right)"},
                "y": {"type": "number", "minimum": 0, "maximum": 1000, "description": "0 (top) to 1000 (bottom)"},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "name": "type_text",
        "description": "Type text into the element that has focus (click a text field first). It does not press Enter.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to type"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "press_keys",
        "description": (
            'Press keys in order, each one a key or a combination, e.g. ["Enter"], ["Control+A", "Delete"] or '
            '["Tab", "Tab", "Enter"]. Alt+ArrowLeft goes back, Alt+ArrowRight goes forward and F5 reloads the page.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_KEYS,
                    "description": "The keys to press, in order",
                }
            },
            "required": ["keys"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scroll",
        "description": (
            "Scroll the page down or up by whole screens (the PageDown and PageUp keys). The page must have focus, "
            "not a text field: click on the page first if a text field has it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["down", "up"]},
                "amount": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_SCROLL_AMOUNT,
                    "description": "How many screens (default 1)",
                },
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
]
SCREEN_TOOLS = tuple(t["name"] for t in SCREEN_TOOL_DEFS)


def filter_dom_tools(child_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The 13 exposed tools of the child, in ``DOM_TOOLS`` order, with the dropped parameters taken out."""
    by_name = {t["name"]: t for t in child_tools}
    missing = [name for name in DOM_TOOLS if name not in by_name]
    if missing:
        raise ChildError(-32603, f"the browser server lacks the tools {missing}")
    tools = []
    for name in DOM_TOOLS:
        tool = json.loads(json.dumps(by_name[name]))
        schema = tool.get("inputSchema", {})
        for param in DOM_DROPPED_PARAMS.get(name, ()):
            schema.get("properties", {}).pop(param, None)
            if param in schema.get("required", []):
                schema["required"].remove(param)
        tools.append(tool)
    return tools


# --- the server ---


# --- the argument checks shared by the two screen arms ---


def click_args(args: dict[str, Any]) -> tuple[float, float]:
    extra = set(args) - {"x", "y"}
    if extra:
        raise Refused(f"click takes x and y only, not {sorted(extra)}")
    coords = []
    for key in ("x", "y"):
        value = args.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1000:
            raise Refused(f"{key} must be a number from 0 to 1000")
        coords.append(float(value))
    return coords[0], coords[1]


def text_arg(args: dict[str, Any]) -> str:
    text = args.get("text")
    if set(args) - {"text"} or not isinstance(text, str) or not text:
        raise Refused("type_text takes one non-empty string, text")
    return text


def keys_arg(args: dict[str, Any]) -> list[str]:
    """The keys of a press_keys call, every one checked before the first is pressed."""
    keys = args.get("keys")
    if isinstance(keys, str):
        keys = [keys]
    if set(args) - {"keys"} or not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys):
        raise Refused("press_keys takes keys: a list of key names")
    if len(keys) > MAX_KEYS:
        raise Refused(f"press at most {MAX_KEYS} keys at a time")
    for key in keys:
        why = refused_key(key)
        if why:
            raise Refused(f"{key} is not allowed ({why})")
    return keys


def scroll_args(args: dict[str, Any]) -> tuple[str, int]:
    direction = args.get("direction")
    amount = args.get("amount", 1)
    if set(args) - {"direction", "amount"} or direction not in ("down", "up"):
        raise Refused('scroll takes direction ("down" or "up") and an optional amount')
    if isinstance(amount, bool) or not isinstance(amount, int) or not 1 <= amount <= MAX_SCROLL_AMOUNT:
        raise Refused(f"amount must be a whole number from 1 to {MAX_SCROLL_AMOUNT}")
    return direction, amount


class Tap:
    def __init__(self, mode: str, log: CallLog, child: Child | None, chrome_pid: int | None = None) -> None:
        self.mode = mode
        self.log = log
        self.child = child
        self.call_lock = threading.Lock()  # one tool call at a time: the screen tools are sequences
        self._dom_tools: list[dict[str, Any]] | None = None
        self.viewport: tuple[int, int] | None = None  # CSS pixels of the viewport at the latest screenshot
        # S-cua: the Chrome whose window is driven, its window, the latest screenshot's size and the Space it was on
        self.chrome_pid = chrome_pid
        self.window_id: int | None = None
        self.image_size: tuple[int, int] | None = None
        self.space_id: int | None = None
        self.session = f"web-tap-{os.getpid()}"
        self._workdir: str | None = None

    # tools/list
    def tools(self) -> list[dict[str, Any]]:
        if self.mode == "none":
            return []
        if self.mode in ("screen", "cua"):
            return SCREEN_TOOL_DEFS
        assert self.child is not None
        if self._dom_tools is None:
            self._dom_tools = filter_dom_tools(self.child.list_tools())
        return self._dom_tools

    # tools/call
    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        """Run one call; the result is always an MCP tool result (refusals and child errors are tool errors)."""
        start = time.time()
        detail: dict[str, Any] = {}
        status = "ok"
        result: dict[str, Any]
        try:
            if not isinstance(arguments, dict):
                raise Refused("arguments must be an object")
            with self.call_lock:
                result = self._dispatch(name, arguments, detail)
            if result.get("isError"):
                status = "tool_error"
        except Refused as exc:
            status, result = "refused", text_result(f"Refused: {exc}", True)
        except ChildError as exc:
            status, result = "rpc_error", text_result(f"Error: {exc.message}", True)
        except Exception as exc:  # noqa: BLE001 - a bug in the tap is a tool error, not the end of the session
            status, result = "internal_error", text_result(f"Error: {type(exc).__name__}: {exc}", True)
        texts = [b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"]
        images = [b for b in result.get("content", []) if b.get("type") == "image"]
        self.log.write(
            event="call",
            tool=name,
            args=shorten(arguments),
            status=status,
            dur_s=round(time.time() - start, 3),
            chars_out=sum(len(t) for t in texts),
            n_images=len(images),
            image_bytes=sum(len(b.get("data", "")) * 3 // 4 for b in images),
            head=("\n".join(texts))[:LOG_HEAD_CHARS] if texts else "",
            **detail,
        )
        return result

    def _dispatch(self, name: str, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "dom" and name in DOM_TOOLS:
            return self._call_dom(name, args, detail)
        if self.mode == "screen" and name in SCREEN_TOOLS:
            return getattr(self, "_screen_" + name)(args, detail)
        if self.mode == "cua" and name in SCREEN_TOOLS:
            return getattr(self, "_cua_" + name)(args, detail)
        raise Refused(f"there is no tool {name!r}")

    # D arm
    def _call_dom(self, name: str, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        assert self.child is not None
        for param in DOM_DROPPED_PARAMS.get(name, ()):
            if param in args:
                raise Refused(f"{name}.{param} is not available")
        if name in ("navigate_page", "new_page") and "url" in args and not url_allowed(args["url"]):
            raise Refused("only http and https URLs can be opened")
        raw = self.child.call_tool(name, args)
        content: list[dict[str, Any]] = []
        chars_raw = rewritten = dropped = 0
        for block in raw.get("content", []):
            if block.get("type") == "text":
                chars_raw += len(block.get("text", ""))
                text, count = rewrite_urls(block.get("text", ""))
                rewritten += count
                content.append({**block, "text": text})
            else:
                dropped += 1  # none of the 13 tools returns an image; D never sees pixels
        detail.update(chars_raw=chars_raw, urls_rewritten=rewritten, blocks_dropped=dropped)
        result: dict[str, Any] = {"content": content}
        if raw.get("isError"):
            result["isError"] = True
        return result

    # S-cdp arm
    def _child_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert self.child is not None
        return self.child.call_tool(name, args)

    def _viewport_now(self, image_size: tuple[int, int] | None) -> tuple[int, int] | None:
        """The viewport in CSS pixels. The child's script evaluation gives it; the image size is the fallback
        (right when the device pixel ratio is 1, as in headless Chrome)."""
        try:
            result = self._child_call("evaluate_script", {"function": "() => [window.innerWidth, window.innerHeight]"})
            match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", result_text(result))
            if match and not result.get("isError"):
                return int(match.group(1)), int(match.group(2))
        except ChildError:
            pass
        return image_size

    def _screen_screenshot(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        for quality in JPEG_QUALITIES:
            raw = self._child_call("take_screenshot", {"format": "jpeg", "quality": quality})
            if raw.get("isError"):
                return {"content": raw.get("content", []), "isError": True}
            block = first_image(raw)
            if block is None or "jpeg" not in block.get("mimeType", ""):
                return text_result("Error: the browser returned no JPEG screenshot", True)
            data = base64.b64decode(block["data"])
            if len(data) <= MAX_IMAGE_BYTES:
                break
        else:
            return text_result(
                f"Error: the screenshot is over {MAX_IMAGE_BYTES // 1024} KB at JPEG quality {quality}", True
            )
        size = jpeg_size(data)
        viewport = self._viewport_now(size)
        if viewport is not None:
            self.viewport = viewport
        detail.update(quality=quality, bytes=len(data), image_size=size, viewport=viewport)
        return {"content": [{"type": "image", "data": block["data"], "mimeType": "image/jpeg"}]}

    def _screen_click(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        x, y = click_args(args)
        if self.viewport is None:
            raise Refused("take a screenshot first; click coordinates refer to the latest screenshot")
        px, py = map_point(x, y, self.viewport)
        detail.update(norm=[x, y], css=[px, py], viewport=list(self.viewport))
        raw = self._child_call("click_at", {"x": px, "y": py})
        return self._action_result(raw, "Clicked.")

    def _screen_type_text(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        raw = self._child_call("type_text", {"text": text_arg(args)})
        return self._action_result(raw, "Typed.")

    def _screen_press_keys(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        keys = keys_arg(args)
        pressed = []
        last: dict[str, Any] = {}
        for key in keys:
            nav = navigation_shortcut(key)
            if nav:
                last = self._child_call("navigate_page", {"type": nav})
            else:
                last = self._child_call("press_key", {"key": normalize_key(key)})
            if last.get("isError"):
                detail["pressed"] = pressed
                return self._action_result(last, "")
            pressed.append(key)
            self._follow_new_tab(last)
        detail["pressed"] = pressed
        return text_result("Pressed " + ", ".join(pressed) + ".")

    def _screen_scroll(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        direction, amount = scroll_args(args)
        key = "PageDown" if direction == "down" else "PageUp"
        for _ in range(amount):
            raw = self._child_call("press_key", {"key": key})
            if raw.get("isError"):
                return self._action_result(raw, "")
        time.sleep(SCROLL_SETTLE_S)
        return text_result(f"Scrolled {direction} {amount} screen{'s' if amount != 1 else ''}.")

    def _action_result(self, raw: dict[str, Any], ok_text: str) -> dict[str, Any]:
        """A short result for an action: the child's text names URLs and titles, which the screen arm must not see."""
        if raw.get("isError"):
            return {"content": raw.get("content", []) or [{"type": "text", "text": "Error"}], "isError": True}
        self._follow_new_tab(raw)
        return text_result(ok_text)

    def _follow_new_tab(self, raw: dict[str, Any]) -> None:
        """A click that opens a tab: show it, as a window would. The child names new pages under '## Opened pages'."""
        text = result_text(raw)
        if "## Opened pages" not in text:
            return
        ids = re.findall(r"^(\d+):", text.split("## Opened pages", 1)[1], re.MULTILINE)
        if ids:
            try:
                self._child_call("select_page", {"pageId": int(ids[-1]), "bringToFront": True})
            except ChildError:
                pass

    # S-cua arm
    def _cua_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert self.child is not None
        return self.child.call_tool(name, {**args, "session": self.session})

    def _cua_windows(self) -> tuple[list[dict[str, Any]], int | None]:
        """The Chrome's top-level windows, and the Space on screen."""
        raw = self._cua_call("list_windows", {"pid": self.chrome_pid})
        if raw.get("isError"):
            raise Refused("the browser window cannot be listed: " + result_text(raw)[:200])
        data = raw.get("structuredContent") or {}
        windows = [w for w in data.get("windows") or [] if w.get("layer", 0) == 0]
        return windows, data.get("current_space_id")

    def _cua_window(self) -> int:
        """The browser window: the largest top-level window of the Chrome, which must be on the Space on screen."""
        windows, current = self._cua_windows()
        if self.space_id is None:
            self.space_id = current
        if current != self.space_id:
            raise Refused("the browser's desktop is not on screen; the run cannot see the browser")

        def area(w: dict[str, Any]) -> float:
            bounds = w.get("bounds") or {}
            return float(bounds.get("width", 0)) * float(bounds.get("height", 0))

        visible = [w for w in windows if w.get("on_current_space") and area(w) > 0]
        if not visible:
            raise Refused("the browser has no window on the desktop on screen")
        self.window_id = int(max(visible, key=area)["window_id"])
        return self.window_id

    def _cua_target(self) -> dict[str, Any]:
        if self.image_size is None or self.window_id is None:
            raise Refused("take a screenshot first; click coordinates refer to the latest screenshot")
        windows, current = self._cua_windows()
        if current != self.space_id:
            raise Refused("the browser's desktop is not on screen; the run cannot see the browser")
        if not any(int(w.get("window_id", -1)) == self.window_id for w in windows):
            raise Refused("the browser window of the latest screenshot is gone; take a screenshot")
        return {"pid": self.chrome_pid, "window_id": self.window_id}

    def _cua_action(self, raw: dict[str, Any], ok_text: str) -> dict[str, Any]:
        """A short result: cua-driver's text names elements of the page, which the screen arm must not see."""
        if raw.get("isError"):
            return text_result("Error: the action failed", True)
        effect = (raw.get("structuredContent") or {}).get("effect")
        if effect == "refused":
            return text_result("Error: the action was refused", True)
        return text_result(ok_text)

    def _cua_screenshot(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        if args:
            raise Refused("screenshot takes no arguments")
        if self._workdir is None:
            import tempfile

            self._workdir = tempfile.mkdtemp(prefix="web-tap-")
            self._cua_call("set_agent_cursor_enabled", {"enabled": False})
        window = self._cua_window()
        raw = self._cua_call(
            "get_window_state",
            {
                "pid": self.chrome_pid,
                "window_id": window,
                "include_accessibility_tree": False,
                "max_image_dimension": CUA_IMAGE_DIM,
            },
        )
        block = first_image(raw)
        if raw.get("isError") or block is None:
            return text_result("Error: no screenshot of the browser window", True)
        png = base64.b64decode(block["data"])
        for quality in JPEG_QUALITIES:
            data = png_to_jpeg(png, quality, self._workdir)
            if len(data) <= MAX_IMAGE_BYTES:
                break
        else:
            return text_result(
                f"Error: the screenshot is over {MAX_IMAGE_BYTES // 1024} KB at JPEG quality {quality}", True
            )
        self.image_size = jpeg_size(data)
        detail.update(quality=quality, bytes=len(data), image_size=self.image_size, window_id=window)
        return {"content": [{"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": "image/jpeg"}]}

    def _cua_click(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        x, y = click_args(args)
        target = self._cua_target()
        assert self.image_size is not None
        px, py = map_point(x, y, self.image_size)
        detail.update(norm=[x, y], px=[px, py], image_size=list(self.image_size))
        raw = self._cua_call("click", {**target, "x": px, "y": py})
        detail["route"] = (raw.get("structuredContent") or {}).get("route")
        return self._cua_action(raw, "Clicked.")

    def _cua_type_text(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        text = text_arg(args)
        raw = self._cua_call("type_text", {**self._cua_target(), "text": text})
        return self._cua_action(raw, "Typed.")

    def _cua_press_keys(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        keys = keys_arg(args)
        target = self._cua_target()
        pressed = []
        for key in keys:
            combo = cua_key(key)
            if len(combo) == 1:
                raw = self._cua_call("press_key", {**target, "key": combo[0]})
            else:
                raw = self._cua_call("hotkey", {**target, "keys": combo})
            result = self._cua_action(raw, "")
            if result.get("isError"):
                detail["pressed"] = pressed
                return result
            pressed.append(key)
        detail["pressed"] = pressed
        return text_result("Pressed " + ", ".join(pressed) + ".")

    def _cua_scroll(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        direction, amount = scroll_args(args)
        raw = self._cua_call("scroll", {**self._cua_target(), "direction": direction, "by": "page", "amount": amount})
        result = self._cua_action(raw, f"Scrolled {direction} {amount} screen{'s' if amount != 1 else ''}.")
        time.sleep(SCROLL_SETTLE_S)
        return result

    # MCP plumbing
    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """The JSON-RPC response to one message, or None for a notification."""
        method = message.get("method")
        rid = message.get("id")
        if rid is None:
            return None
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        try:
            if method == "initialize":
                wanted = params.get("protocolVersion")
                version = wanted if wanted in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
                self.log.write(event="initialize", client=params.get("clientInfo"), protocol=version)
                result: dict[str, Any] = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                tools = self.tools()
                self.log.write(event="tools_list", tools=[t["name"] for t in tools])
                result = {"tools": tools}
            elif method == "tools/call":
                name = params.get("name")
                if not isinstance(name, str):
                    return {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {"code": -32602, "message": "tools/call needs a name"},
                    }
                result = self.call(name, params.get("arguments", {}))
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
        except ChildError as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": exc.message}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def serve(self, stdin: Any, stdout: Any) -> None:
        out_lock = threading.Lock()

        def reply(message: dict[str, Any] | None) -> None:
            if message is None:
                return
            with out_lock:
                stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
                stdout.flush()

        def work(message: dict[str, Any]) -> None:
            try:
                reply(self.handle(message))
            except Exception as exc:  # noqa: BLE001 - a broken call must not end the server
                self.log.write(event="error", error=f"{type(exc).__name__}: {exc}")
                if message.get("id") is not None:
                    reply({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32603, "message": str(exc)}})

        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                reply({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
                continue
            if not isinstance(message, dict):
                reply({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}})
                continue
            threading.Thread(target=work, args=(message,), daemon=True).start()

    def warm_up(self) -> None:
        """Start the child in the background, so the first call does not wait for node."""
        child = self.child
        if child is None:
            return

        def start() -> None:
            try:
                child.ensure_started()
            except ChildError as exc:
                self.log.write(event="child_start_failed", error=exc.message)

        threading.Thread(target=start, daemon=True).start()

    def close(self) -> None:
        if self.child is not None:
            self.child.close()
        if self._workdir is not None:
            import shutil

            shutil.rmtree(self._workdir, ignore_errors=True)
        self.log.write(event="stop")
        self.log.close()


def build_tap(args: argparse.Namespace, child_argv: list[str]) -> Tap:
    log = CallLog(args.log, args.mode)
    child = None
    if args.mode != "none":
        cmd = child_argv or child_command(args.mode, args.browser_url, args.mcp_bin, args.cua_bin)
        env = {**os.environ, **CHILD_ENV}
        child = Child(cmd, env, args.child_stderr)
    log.write(event="start", pid=os.getpid(), child=(child.cmd if child else None), chrome_pid=args.chrome_pid)
    return Tap(args.mode, log, child, args.chrome_pid)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    child_argv: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, child_argv = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("dom", "screen", "cua", "none"), default=os.environ.get("WEB_TAP_MODE"))
    parser.add_argument(
        "--log", default=os.environ.get("WEB_TAP_LOG"), help="JSONL file for every call (or WEB_TAP_LOG)"
    )
    parser.add_argument("--browser-url", default=os.environ.get("WEB_TAP_BROWSER_URL"), help="http://127.0.0.1:<port>")
    parser.add_argument("--mcp-bin", default=os.environ.get("WEB_TAP_MCP_BIN"), help="the chrome-devtools-mcp binary")
    parser.add_argument("--cua-bin", default=os.environ.get("WEB_TAP_CUA_BIN"), help="the cua-driver binary")
    parser.add_argument(
        "--chrome-pid",
        type=int,
        default=int(os.environ["WEB_TAP_CHROME_PID"]) if os.environ.get("WEB_TAP_CHROME_PID") else None,
        help="the headed Chrome whose window the cua mode drives (or WEB_TAP_CHROME_PID)",
    )
    parser.add_argument(
        "--child-stderr", default=os.environ.get("WEB_TAP_CHILD_STDERR"), help="file for the child's stderr"
    )
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.error("--mode (or WEB_TAP_MODE) is required")
    if args.mode in ("dom", "screen") and not child_argv and not args.browser_url:
        parser.error("--browser-url (or WEB_TAP_BROWSER_URL) is required for the dom and screen modes")
    if args.mode == "cua" and not args.chrome_pid:
        parser.error("--chrome-pid (or WEB_TAP_CHROME_PID) is required for the cua mode")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    tap = build_tap(args, child_argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # the finally below shuts the child down
    tap.warm_up()
    try:
        tap.serve(sys.stdin, sys.stdout)
    except KeyboardInterrupt:
        pass
    finally:
        tap.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
