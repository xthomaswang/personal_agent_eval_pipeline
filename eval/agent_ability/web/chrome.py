"""The bench Chrome of the web search suite: a persistent profile template, and one headless Chrome per run.

    cd eval
    uv run python agent_ability/web/chrome.py init [--force]   # make the profile template once
    uv run python agent_ability/web/chrome.py version          # the installed Chrome's version
    uv run python agent_ability/web/chrome.py preflight        # refuse while a bench Chrome is running

Per run (``bench_chrome``):

1. Clone the template into the run directory with ``cp -c`` (an APFS clone).
2. Pick a free port and start the installed Chrome headless on the clone, with the run flags.
3. Wait for ``/json/version`` and leave one tab, on ``https://www.google.com/?hl=en``, using CDP's HTTP endpoints.
4. The harness runs; its start is t0.
5. Save the tab list. Close Chrome over CDP (``Browser.close``): that is what commits History, because a SIGTERM
   leaves the last ten seconds of visits uncommitted. Then SIGTERM if it is still running, SIGKILL after 15 s.
   Keep ``Default/History`` and delete the clone.

Safety. The user's own Chrome runs on their default profile, and nothing here may touch it:

- A profile is a bench profile only if it carries the marker file ``bench-profile.json``, which ``init`` writes
  and a clone inherits. ``launch`` refuses any other path, and the default Chrome directory in particular.
- A process is signalled only if its command line (read again just before the signal) holds
  ``--user-data-dir=<the bench profile path this module created>``. Nothing is ever signalled by name or by a bare pid.
- Chrome is started by running its binary with an argument list. No ``open``, no AppleScript, no shell.

The binary is ``$WEB_CHROME_BINARY`` if set, else the executable of the app with Chrome's bundle id (Spotlight).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import functools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_ABILITY = Path(__file__).resolve().parent.parent
BENCH_ROOT = AGENT_ABILITY / "runs" / "web"
TEMPLATE_DIR = BENCH_ROOT / "profile-template"
CHROME_BUNDLE_ID = "com.google.Chrome"
CHROME_EXECUTABLE = Path("Contents", "MacOS", "Google Chrome")  # inside the app bundle
START_URL = "https://www.google.com/?hl=en"
MARKER = "bench-profile.json"
MARKER_KIND = "web-bench-chrome-profile"
START_SETTLE_S = 1.5  # after the start page's title shows, before t0
GRACEFUL_TIMEOUT_S = 10.0  # how long Chrome gets to exit after Browser.close, before SIGTERM
TERM_TIMEOUT_S = 15.0  # SIGTERM, then SIGKILL after this long
WINDOW = (1280, 800)

# Answer-key hosts: Hugging Face, Kaggle, OpenAI's blob store and raw GitHub. Chrome cannot resolve them.
BLOCKED_HOSTS = (
    "huggingface.co",
    "hf.co",
    "*.huggingface.co",
    "*.kaggle.com",
    "kaggle.com",
    "openaipublic.blob.core.windows.net",
    "raw.githubusercontent.com",
)

# Files that make Chrome think the profile is in use, or that belong to one running instance.
STALE_FILES = ("SingletonLock", "SingletonSocket", "SingletonCookie", "DevToolsActivePort")

# What a new tab's title is while it has not loaded yet.
_NOT_LOADED_TITLES = ("", "about:blank")


class ChromeError(RuntimeError):
    """A bench Chrome could not be started, or was refused."""


class ChromeVersionChanged(ChromeError):
    """The installed Chrome differs from the version recorded for the profile or the study."""


class UnsafeProfile(ChromeError):
    """A path that is not a bench profile was passed where a bench profile is required."""


class UnsafeSignal(ChromeError):
    """A signal was refused because the process does not use the bench profile path."""


# --- the binary, the version, the user agent and the flags ---


@functools.cache
def chrome_binary() -> Path:
    """The installed Chrome's executable: ``$WEB_CHROME_BINARY``, else the app with Chrome's bundle id.

    Several copies may be indexed; one in the system's Applications folder wins, then the shortest path.
    """
    if os.environ.get("WEB_CHROME_BINARY"):
        return Path(os.environ["WEB_CHROME_BINARY"])
    query = f"kMDItemCFBundleIdentifier == '{CHROME_BUNDLE_ID}'"
    try:
        out = subprocess.run(["mdfind", query], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ChromeError(f"cannot look up {CHROME_BUNDLE_ID}: {exc}") from exc
    apps = [Path(line) for line in out.splitlines() if line.endswith(".app")]
    if not apps:
        raise ChromeError(f"no app with bundle id {CHROME_BUNDLE_ID}; set WEB_CHROME_BINARY")
    app = min(apps, key=lambda a: (a.parent != Path("/Applications"), len(str(a)), str(a)))
    return app / CHROME_EXECUTABLE


def chrome_version(binary: Path | str | None = None) -> str:
    """The installed Chrome's version, e.g. ``156.0.8078.80`` (from ``--version``)."""
    binary = binary or chrome_binary()
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ChromeError(f"cannot run {binary} --version: {exc}") from exc
    match = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
    if not match:
        raise ChromeError(f"cannot read a version from {out.strip()!r}")
    return match.group(1)


def user_agent(version: str) -> str:
    """The user agent of the installed Chrome's normal build: the major version, then ``.0.0.0``."""
    major = version.split(".")[0]
    return f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"


def host_resolver_rules(hosts: Sequence[str] = BLOCKED_HOSTS) -> str:
    return ", ".join(f"MAP {host} ~NOTFOUND" for host in hosts)


def chrome_args(
    binary: Path | str, profile: Path | str, port: int, version: str, *, headless: bool = True
) -> list[str]:
    """The command line of a bench Chrome. The same for every arm; S-cua alone runs with ``headless=False``."""
    width, height = WINDOW
    args = [
        str(binary),
        *(["--headless=new"] if headless else ["--window-position=0,25"]),
        f"--user-data-dir={profile}",
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=127.0.0.1",
        "--no-first-run",
        "--no-default-browser-check",
        "--use-mock-keychain",
        "--disable-background-networking",
        "--disable-features=Translate",
        "--lang=en-US",
        f"--window-size={width},{height}",
        f"--user-agent={user_agent(version)}",
        f"--host-resolver-rules={host_resolver_rules()}",
    ]
    return args


def check_version(installed: str, *recorded: str | None) -> None:
    """Refuse a Chrome whose version differs from any recorded one (None entries are skipped)."""
    for want in recorded:
        if want is not None and want != installed:
            raise ChromeVersionChanged(
                f"Chrome is {installed}, but {want} was recorded; a version change stops a study"
            )


# --- bench profiles and the processes that use them ---


def default_chrome_dirs() -> list[Path]:
    """Where Google Chrome keeps the user's own profile. Nothing here ever uses a path inside it."""
    return [Path.home() / "Library" / "Application Support" / "Google"]


def is_bench_profile(path: Path | str) -> bool:
    """True only for a directory this module made: it holds the marker, and it is not in Chrome's own directory."""
    path = Path(path)
    resolved = path.resolve()
    for chrome_dir in default_chrome_dirs():
        if resolved == chrome_dir.resolve() or chrome_dir.resolve() in resolved.parents:
            return False
    try:
        marker = json.loads((path / MARKER).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(marker, dict) and marker.get("kind") == MARKER_KIND


def assert_bench_profile(path: Path | str) -> Path:
    if not is_bench_profile(path):
        raise UnsafeProfile(f"{path} is not a bench profile (no {MARKER} marker, or it is Chrome's own directory)")
    return Path(path)


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    command: str


def process_table() -> list[Proc]:
    """Every process with its full command line (``ps -ww``; no shell)."""
    out = subprocess.run(
        ["ps", "-axww", "-o", "pid=,ppid=,command="], capture_output=True, text=True, check=True
    ).stdout
    return parse_process_table(out)


def parse_process_table(text: str) -> list[Proc]:
    procs = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\d+)\s+(.*\S)\s*$", line)
        if match:
            procs.append(Proc(int(match.group(1)), int(match.group(2)), match.group(3)))
    return procs


def profile_dir_of(command: str) -> str | None:
    """The ``--user-data-dir`` value of a command line, or None. The value may hold spaces (Electron apps do)."""
    match = re.search(r"--user-data-dir=(.*?)(?=\s--[A-Za-z]|\s*$)", command)
    return match.group(1) if match else None


def command_uses_profile(command: str, profile: Path | str) -> bool:
    """True if the command's ``--user-data-dir`` is exactly this path."""
    value = profile_dir_of(command)
    return value is not None and value == str(profile)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def bench_processes(table: Sequence[Proc], roots: Sequence[Path] = (BENCH_ROOT,)) -> list[Proc]:
    """Chrome processes (with helpers) whose profile is a bench profile: under a root, or holding the marker."""
    found = []
    for proc in table:
        value = profile_dir_of(proc.command)
        if value is None:
            continue
        path = Path(value)
        if any(_under(path, root) for root in roots) or is_bench_profile(path):
            found.append(proc)
    return found


def command_of(pid: int) -> str | None:
    """The current command line of a pid, or None if it is gone."""
    out = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
    ).stdout.strip()
    return out or None


def signal_bench_process(pid: int, profile: Path | str, sig: int) -> bool:
    """Signal a process only if its command line, read now, uses exactly this bench profile. Returns False if gone."""
    assert_bench_profile(profile)
    command = command_of(pid)
    if command is None:
        return False
    if not command_uses_profile(command, profile):
        raise UnsafeSignal(f"refusing to signal pid {pid}: its command line does not use {profile}")
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def preflight(
    roots: Sequence[Path] = (BENCH_ROOT,),
    *,
    binary: Path | str | None = None,
    table: Callable[[], Sequence[Proc]] = process_table,
) -> list[str]:
    """Problems that stop a study: a missing binary, or any Chrome still using a bench profile."""
    problems = []
    try:
        binary = binary or chrome_binary()
    except ChromeError as exc:
        problems.append(str(exc))
    if binary and not Path(binary).exists():
        problems.append(f"Chrome not found at {binary}")
    for proc in bench_processes(table(), roots):
        problems.append(f"a Chrome is still using a bench profile: pid {proc.pid} ({profile_dir_of(proc.command)})")
    return problems


# --- the HTTP endpoints of one Chrome ---

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a system proxy


def http_get(url: str, *, method: str = "GET", timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, method=method)
    with _OPENER.open(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    try:
        return json.loads(body)
    except ValueError:
        return body


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class ChromeInstance:
    """One running bench Chrome."""

    proc: subprocess.Popen[bytes]
    profile: Path
    port: int
    version: str
    args: list[str]
    log_path: Path | None = None
    started_at: float = field(default_factory=time.time)

    @property
    def browser_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def tabs(self) -> list[dict[str, Any]]:
        """The page targets (tabs), from ``/json/list``."""
        targets = http_get(f"{self.browser_url}/json/list")
        return [t for t in targets if t.get("type") == "page"]

    def open_tab(self, url: str) -> dict[str, Any]:
        tab = http_get(f"{self.browser_url}/json/new?{url}", method="PUT")
        if not isinstance(tab, dict) or "id" not in tab:
            raise ChromeError(f"/json/new did not return a tab: {tab!r}")
        return tab

    def close_tab(self, tab_id: str) -> None:
        http_get(f"{self.browser_url}/json/close/{tab_id}")

    def assert_ours(self) -> None:
        """Refuse to act on a process whose command line (read now) does not use this bench profile."""
        command = command_of(self.proc.pid)
        if command is not None and not command_uses_profile(command, self.profile):
            raise UnsafeSignal(f"pid {self.proc.pid} does not use {self.profile}")

    def stop(
        self,
        term_timeout: float = TERM_TIMEOUT_S,
        straggler_grace_s: float = 3.0,
        graceful_timeout: float = GRACEFUL_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Browser.close over CDP, then SIGTERM, then SIGKILL after ``term_timeout``; end any helper left over.

        Only the clean close commits History: after a SIGTERM the visits of the last ten seconds are lost.
        """
        graceful = sigterm = sigkill = False
        if self.proc.poll() is None:
            self.assert_ours()
            if cdp_browser_close(self.port):
                try:
                    self.proc.wait(timeout=graceful_timeout)
                    graceful = True
                except subprocess.TimeoutExpired:
                    pass
        if self.proc.poll() is None:
            sigterm = True
            signal_bench_process(self.proc.pid, self.profile, signal.SIGTERM)
            try:
                self.proc.wait(timeout=term_timeout)
            except subprocess.TimeoutExpired:
                sigkill = True
                signal_bench_process(self.proc.pid, self.profile, signal.SIGKILL)
                self.proc.wait(timeout=10)
        stragglers = 0
        deadline = time.time() + straggler_grace_s
        while True:
            left = [p for p in process_table() if command_uses_profile(p.command, self.profile)]
            if not left:
                break
            if time.time() >= deadline:
                for proc in left:
                    stragglers += signal_bench_process(proc.pid, self.profile, signal.SIGKILL)
                break
            time.sleep(0.2)
        return {
            "exit_code": self.proc.returncode,
            "graceful": graceful,
            "sigterm": sigterm,
            "sigkill": sigkill,
            "stragglers_killed": stragglers,
        }


def cdp_browser_close(port: int, *, timeout_s: float = 5.0) -> bool:
    """Send ``Browser.close`` to the Chrome on 127.0.0.1:port over CDP's WebSocket (stdlib only). True if it was sent."""
    try:
        info = http_get(f"http://127.0.0.1:{port}/json/version", timeout=timeout_s)
        ws = urllib.parse.urlsplit(info["webSocketDebuggerUrl"])
        if ws.hostname not in ("127.0.0.1", "localhost") or ws.port != port:
            return False  # only ever the loopback endpoint of the port this module started
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as sock:
            request = (
                f"GET {ws.path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {base64.b64encode(os.urandom(16)).decode()}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            )
            sock.sendall(request.encode())
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = sock.recv(4096)
                if not chunk:
                    return False
                head += chunk
            if b" 101 " not in head.split(b"\r\n", 1)[0]:
                return False
            payload = json.dumps({"id": 1, "method": "Browser.close"}).encode()
            mask = os.urandom(4)  # a client frame is masked (RFC 6455); this payload is shorter than 126 bytes
            sock.sendall(
                bytes([0x81, 0x80 | len(payload)]) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            )
            sock.settimeout(1.0)
            with contextlib.suppress(OSError):
                sock.recv(4096)  # Chrome answers, then goes away
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def wait_for_devtools(
    port: int, proc: subprocess.Popen[bytes] | None = None, *, timeout_s: float = 30.0, interval_s: float = 0.2
) -> dict[str, Any]:
    """Poll ``/json/version`` until Chrome answers; fail early if the process exits."""
    url = f"http://127.0.0.1:{port}/json/version"
    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ChromeError(f"Chrome exited with code {proc.returncode} before /json/version answered")
        try:
            info = http_get(url, timeout=2.0)
            if isinstance(info, dict) and "Browser" in info:
                return info
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
        time.sleep(interval_s)
    raise ChromeError(f"/json/version on port {port} did not answer in {timeout_s:.0f} s ({last})")


def launch(
    profile: Path | str,
    *,
    binary: Path | str | None = None,
    version: str | None = None,
    port: int | None = None,
    headless: bool = True,
    log_path: Path | None = None,
    startup_timeout_s: float = 30.0,
) -> ChromeInstance:
    """Start Chrome on a bench profile and wait for its DevTools endpoint."""
    profile = assert_bench_profile(profile).resolve()
    binary = binary or chrome_binary()
    version = version or chrome_version(binary)
    port = port or free_port()
    for stale in STALE_FILES:  # a clone or a template may hold the links of a Chrome that is gone
        (profile / stale).unlink(missing_ok=True)
    args = chrome_args(binary, profile, port, version, headless=headless)
    with contextlib.ExitStack() as stack:
        log = stack.enter_context(open(log_path, "wb")) if log_path else subprocess.DEVNULL
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    chrome = ChromeInstance(proc, profile, port, version, args, log_path)
    try:
        wait_for_devtools(port, proc, timeout_s=startup_timeout_s)
    except ChromeError:
        with contextlib.suppress(ChromeError, subprocess.SubprocessError):
            chrome.stop(term_timeout=5.0)
        raise
    return chrome


def _title_is_placeholder(title: str, url: str) -> bool:
    """A tab that is still loading is titled with its URL (Chrome drops the scheme and a leading ``www.``), or nothing."""
    if title in _NOT_LOADED_TITLES:
        return True
    bare = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", url)
    shown = {url.rstrip("/"), bare.rstrip("/"), re.sub(r"^www\.", "", bare).rstrip("/")}
    return title.rstrip("/") in shown


def leave_start_tab(
    chrome: ChromeInstance, url: str = START_URL, *, ready_timeout_s: float = 20.0, settle_s: float = START_SETTLE_S
) -> dict[str, Any]:
    """Leave exactly one tab, on ``url``, using only CDP's HTTP endpoints. Waits until its title shows, then ``settle_s``
    more, so that the page's own scripts (Google adds ``zx=`` to the address) are done before t0."""
    before = chrome.tabs()
    tab = chrome.open_tab(url)
    for old in before:
        chrome.close_tab(old["id"])
    deadline = time.time() + ready_timeout_s
    ready = False
    current = tab
    while time.time() < deadline:
        tabs = chrome.tabs()
        current = next((t for t in tabs if t["id"] == tab["id"]), current)
        if not _title_is_placeholder(current.get("title", ""), current.get("url", "")):
            ready = True
            break
        time.sleep(0.25)
    if ready:
        time.sleep(settle_s)
        current = next((t for t in chrome.tabs() if t["id"] == tab["id"]), current)
    return {"id": tab["id"], "url": current.get("url", url), "title": current.get("title", ""), "ready": ready}


def save_tabs(chrome: ChromeInstance, path: Path) -> list[dict[str, Any]]:
    """Write the tab list (id, title, url) to ``path``. Failure to read it is recorded, not raised."""
    try:
        tabs = [{"id": t.get("id"), "title": t.get("title"), "url": t.get("url")} for t in chrome.tabs()]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        tabs = [{"error": f"{type(exc).__name__}: {exc}"}]
    path.write_text(json.dumps(tabs, indent=2) + "\n")
    return tabs


# --- the profile template and the clones ---


def _write_marker(profile: Path, **fields: Any) -> None:
    (profile / MARKER).write_text(json.dumps({"kind": MARKER_KIND, **fields}, indent=2, sort_keys=True) + "\n")


def read_marker(profile: Path | str) -> dict[str, Any]:
    return json.loads((Path(profile) / MARKER).read_text())


def clone_profile(template: Path, dst: Path) -> str:
    """Clone the template with ``cp -c`` (an APFS clone); fall back to a plain copy. Returns which was used."""
    assert_bench_profile(template)
    if dst.exists():
        raise ChromeError(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    how = "apfs-clone"
    if subprocess.run(["cp", "-c", "-R", str(template), str(dst)], capture_output=True, check=False).returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        subprocess.run(["cp", "-R", str(template), str(dst)], check=True)
        how = "copy"
    for stale in STALE_FILES:
        (dst / stale).unlink(missing_ok=True)
    return how


def _patch_preferences(profile: Path) -> None:
    """English, and notification and location prompts blocked. Done while Chrome is not running."""
    path = profile / "Default" / "Preferences"
    prefs = json.loads(path.read_text()) if path.exists() else {}
    content = prefs.setdefault("profile", {}).setdefault("default_content_setting_values", {})
    content.update({"notifications": 2, "geolocation": 2})  # 2 = block
    prefs.setdefault("intl", {})["accept_languages"] = "en-US,en"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, separators=(",", ":")))


def init_template(
    template: Path = TEMPLATE_DIR,
    *,
    binary: Path | str | None = None,
    force: bool = False,
    start_url: str = START_URL,
    settle_s: float = 3.0,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Make the profile template: run the bench Chrome once on an empty profile, visit Google, patch the preferences.

    The template is made with the same flags as every run. No sign-in, no sync, English, Google as the search engine
    (Chrome's default); the visit leaves Google's cookies. History is removed afterwards, so a clone starts clean.
    """
    template = Path(template)
    binary = binary or chrome_binary()
    problems = preflight(roots or (BENCH_ROOT, template.parent), binary=binary)
    if problems:
        raise ChromeError("; ".join(problems))
    if template.exists():
        if not force:
            raise ChromeError(f"{template} exists; pass --force to make it again")
        if not (is_bench_profile(template) or template.resolve() == TEMPLATE_DIR.resolve()):
            raise UnsafeProfile(f"refusing to delete {template}: not a bench profile")
        shutil.rmtree(template)
    version = chrome_version(binary)
    template.mkdir(parents=True)
    _write_marker(
        template,
        role="template",
        chrome_version=version,
        user_agent=user_agent(version),
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    chrome = launch(template, binary=binary, version=version, log_path=template.parent / f"{template.name}.chrome.log")
    try:
        start = leave_start_tab(chrome, start_url, settle_s=settle_s)  # the settle lets Google write its cookies
    finally:
        chrome.stop()
    for name in ("History", "History-journal", "History-wal"):
        (template / "Default" / name).unlink(missing_ok=True)
    for stale in STALE_FILES:
        (template / stale).unlink(missing_ok=True)
    _patch_preferences(template)
    return {"template": str(template), "chrome_version": version, "start_tab": start}


# --- one run ---


@dataclass
class Run:
    """A bench Chrome during a run. ``finish`` saves the tabs and History, stops Chrome and deletes the clone."""

    chrome_dir: Path
    chrome: ChromeInstance
    version: str
    start_tab: dict[str, Any]
    clone: str
    meta: dict[str, Any] = field(default_factory=dict)
    finished: bool = False

    @property
    def browser_url(self) -> str:
        return self.chrome.browser_url

    @property
    def history_path(self) -> Path:
        return self.chrome_dir / "History"

    def finish(
        self, term_timeout: float = TERM_TIMEOUT_S, graceful_timeout: float = GRACEFUL_TIMEOUT_S
    ) -> dict[str, Any]:
        if self.finished:
            return self.meta
        self.finished = True
        tabs = save_tabs(self.chrome, self.chrome_dir / "tabs.json") if self.chrome.proc.poll() is None else []
        stopped = self.chrome.stop(term_timeout, graceful_timeout=graceful_timeout)
        profile = self.chrome.profile
        for name in ("History", "History-journal"):
            source = profile / "Default" / name
            if source.exists():
                shutil.copy2(source, self.chrome_dir / name)
        if is_bench_profile(profile):  # rmtree only a profile that is ours
            shutil.rmtree(profile, ignore_errors=True)
        self.meta.update(
            stopped_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            exit=stopped,
            n_tabs_at_exit=len(tabs),
            history_kept=self.history_path.exists(),
        )
        (self.chrome_dir / "chrome.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        return self.meta


@contextlib.contextmanager
def bench_chrome(
    chrome_dir: Path,
    *,
    template: Path = TEMPLATE_DIR,
    binary: Path | str | None = None,
    expected_version: str | None = None,
    start_url: str = START_URL,
    headless: bool = True,
    term_timeout: float = TERM_TIMEOUT_S,
    graceful_timeout: float = GRACEFUL_TIMEOUT_S,
    settle_s: float = START_SETTLE_S,
    roots: Sequence[Path] | None = None,
) -> Iterator[Run]:
    """Clone the template, start Chrome on it, leave one tab on Google; on exit keep History and delete the clone.

    ``expected_version`` is the version recorded for the study; a mismatch with the installed Chrome (or with the one
    that made the template) raises ChromeVersionChanged before anything starts.
    """
    chrome_dir = Path(chrome_dir)
    binary = binary or chrome_binary()
    version = chrome_version(binary)
    check_version(version, read_marker(assert_bench_profile(template)).get("chrome_version"), expected_version)
    problems = preflight(roots or (BENCH_ROOT, chrome_dir), binary=binary)
    if problems:
        raise ChromeError("; ".join(problems))
    chrome_dir.mkdir(parents=True, exist_ok=True)
    profile = chrome_dir / "profile"
    clone = clone_profile(template, profile)
    _write_marker(profile, role="clone", chrome_version=version)
    chrome = launch(profile, binary=binary, version=version, headless=headless, log_path=chrome_dir / "chrome.log")
    meta: dict[str, Any] = {
        "chrome_version": version,
        "headless": headless,
        "port": chrome.port,
        "pid": chrome.proc.pid,
        "args": [a.replace(str(chrome.profile), "<profile>") for a in chrome.args[1:]],
        "clone": clone,
        "started_at": dt.datetime.fromtimestamp(chrome.started_at, dt.UTC).isoformat(timespec="seconds"),
    }
    run = Run(chrome_dir, chrome, version, {}, clone, meta)
    try:
        run.start_tab = meta["start_tab"] = leave_start_tab(chrome, start_url, settle_s=settle_s)
        meta["ready_at"] = time.time()
        yield run
    finally:
        run.finish(term_timeout, graceful_timeout)


# --- the command line ---


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="make the profile template (runs/web/profile-template/)")
    init.add_argument("--force", action="store_true", help="make it again if it exists")
    sub.add_parser("version", help="print the installed Chrome's version")
    sub.add_parser("preflight", help="exit 1 while a Chrome uses a bench profile")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(chrome_version())
        return 0
    if args.command == "preflight":
        problems = preflight()
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    print(json.dumps(init_template(force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
