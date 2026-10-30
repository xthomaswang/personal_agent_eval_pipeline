"""The bench Chrome (web/chrome.py) and the History counts (web/history.py).

The unit tests run against a fake Chrome: an executable script that answers /json/version, /json/list,
/json/new and /json/close, takes Browser.close over a WebSocket, and writes a History database when it closes
cleanly. They never start the real Chrome. One opt-in test does (``WEB_REAL_CHROME=1``).

    cd eval
    uv run pytest -q agent_ability/tests/test_web_chrome.py
    WEB_REAL_CHROME=1 uv run pytest -q -s agent_ability/tests/test_web_chrome.py -k real
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_web_tap import Client, text_of
from web import chrome, history, tap

VERSION = "156.0.8078.80"

FAKE_CHROME = r"""#!{python}
import base64, hashlib, http.server, json, os, signal, sqlite3, struct, sys, threading, time

VERSION = os.environ.get("FAKE_CHROME_VERSION", "156.0.8078.80")
if "--version" in sys.argv:
    print("Google Chrome " + VERSION + " ")
    sys.exit(0)
if os.environ.get("FAKE_CHROME_EXIT_EARLY"):
    sys.exit(7)

flags = dict(a.partition("=")[::2] for a in sys.argv[1:])
profile, port = flags["--user-data-dir"], int(flags["--remote-debugging-port"])
os.makedirs(profile + "/Default", exist_ok=True)
with open(profile + "/fake-chrome-args.json", "w") as f:
    json.dump(sys.argv[1:], f)
with open(profile + "/Default/Cookies", "a") as f:
    f.write("cookie\n")
if not os.path.exists(profile + "/Default/Preferences"):
    with open(profile + "/Default/Preferences", "w") as f:
        f.write("{}")
if not os.path.lexists(profile + "/SingletonLock"):
    os.symlink("fake-host-1", profile + "/SingletonLock")  # a real Chrome leaves this behind when it is killed

LOAD_S = float(os.environ.get("FAKE_CHROME_LOAD_S", "0.4"))
tabs = [
    {"id": "T0", "type": "page", "title": "about:blank", "url": "about:blank", "born": 0},
    {"id": "SW", "type": "service_worker", "title": "worker", "url": "https://sw.example/", "born": 0},
]
visits = []


def flush_history():
    path = profile + "/Default/History"
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE urls(id INTEGER PRIMARY KEY, url TEXT, title TEXT, visit_count INTEGER, typed_count INTEGER, "
        "last_visit_time INTEGER, hidden INTEGER);"
        "CREATE TABLE visits(id INTEGER PRIMARY KEY, url INTEGER, visit_time INTEGER, from_visit INTEGER, "
        "transition INTEGER, opener_visit INTEGER);")
    for i, (when, url) in enumerate(visits, 1):
        db.execute("INSERT INTO urls VALUES (?, ?, '', 1, 0, 0, 0)", (i, url))
        db.execute("INSERT INTO visits VALUES (?, ?, ?, 0, ?, 0)", (i, i, int((when + 11644473600) * 1e6), 0x30000006))
    db.commit()
    db.close()


def listing():
    now = time.time()
    out = []
    for t in tabs:
        title = t["title"]
        if t["born"] and now - t["born"] < LOAD_S:
            title = t["url"].split("://", 1)[-1]  # a tab that is still loading is titled with its URL
        out.append({"id": t["id"], "type": t["type"], "url": t["url"], "title": title})
    return out


def on_term(*_):
    os._exit(0)  # a SIGTERM exit does not commit History (as in the real headless Chrome)


if os.environ.get("FAKE_CHROME_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, on_term)


class Handler(http.server.BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Upgrade", "").lower() == "websocket":
            return self.websocket()
        if self.path == "/json/version":
            ws = f"ws://127.0.0.1:{port}/devtools/browser/fake"
            self.send_json({"Browser": "Chrome/" + VERSION, "webSocketDebuggerUrl": ws})
        elif self.path == "/json/list":
            self.send_json(listing())
        elif self.path.startswith("/json/close/"):
            tab_id = self.path.rsplit("/", 1)[1]
            tabs[:] = [t for t in tabs if t["id"] != tab_id]
            self.send_json("Target is closing")
        else:
            self.send_json("not found", 404)  # also GET /json/new, which real Chrome refuses since version 111

    def do_PUT(self):
        if not self.path.startswith("/json/new?"):
            return self.send_json("not found", 404)
        url = self.path.split("?", 1)[1]
        tab = {"id": f"T{len(visits) + 1}", "type": "page", "url": url, "title": "Google", "born": time.time()}
        tabs.append(tab)
        visits.append((time.time(), url))
        self.send_json({k: tab[k] for k in ("id", "type", "url", "title")})

    def websocket(self):
        accept = base64.b64encode(
            hashlib.sha1((self.headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        head = self.rfile.read(2)
        assert head[0] == 0x81 and head[1] & 0x80, "a client frame must be a masked text frame"
        length = head[1] & 0x7F
        mask = self.rfile.read(4)
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(self.rfile.read(length)))
        if json.loads(payload).get("method") == "Browser.close" and not os.environ.get("FAKE_CHROME_IGNORE_CLOSE"):
            reply = b'{"id":1,"result":{}}'
            self.wfile.write(bytes([0x81, len(reply)]) + reply)
            self.wfile.flush()
            flush_history()
            os._exit(0)

    def log_message(self, *args):
        pass


if os.environ.get("FAKE_CHROME_NO_HTTP"):
    threading.Event().wait()
http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake Chrome binary and a bench root; no real Chrome, and no process but the ones the test starts."""
    fake = tmp_path / "fake-chrome"
    fake.write_text(FAKE_CHROME.replace("{python}", sys.executable))
    fake.chmod(0o755)
    root = tmp_path / "bench"
    root.mkdir()
    for name in ("FAKE_CHROME_VERSION", "FAKE_CHROME_IGNORE_TERM", "FAKE_CHROME_IGNORE_CLOSE", "FAKE_CHROME_NO_HTTP"):
        monkeypatch.delenv(name, raising=False)
    started: list[subprocess.Popen[bytes]] = []
    yield SimpleNamespace(fake=fake, root=root, template=root / "profile-template", started=started, tmp=tmp_path)
    for proc in started:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    # nothing the test started may outlive it
    assert chrome.preflight([root], binary=fake) == []


def make_template(env: SimpleNamespace) -> Path:
    chrome.init_template(env.template, binary=env.fake, settle_s=0.0, roots=[env.root])
    return env.template


def run_chrome(env: SimpleNamespace, chrome_dir: Path, template: Path, **kwargs: Any):
    """chrome.bench_chrome on the fake binary and the test's bench root, without the settle time."""
    kwargs.setdefault("settle_s", 0.0)
    return chrome.bench_chrome(chrome_dir, template=template, binary=env.fake, roots=[env.root], **kwargs)


def decoy(env: SimpleNamespace, *args: str) -> subprocess.Popen[bytes]:
    """A harmless process with the given extra arguments in its command line (stands for someone else's Chrome)."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", *args])
    env.started.append(proc)
    return proc


def alive(proc: subprocess.Popen[bytes]) -> bool:
    return proc.poll() is None


def wait_gone(proc: subprocess.Popen[bytes], seconds: float = 5.0) -> bool:
    try:
        proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


# --- the flags, the user agent and the version ---

# The command of the hand test of 2026-10-30 that reached Google results without a CAPTCHA.
EXPECTED_FLAGS = [
    "--headless=new",
    "--user-data-dir=<profile>",
    "--remote-debugging-port=<port>",
    "--remote-debugging-address=127.0.0.1",
    "--no-first-run",
    "--no-default-browser-check",
    "--use-mock-keychain",
    "--disable-background-networking",
    "--disable-features=Translate",
    "--lang=en-US",
    "--window-size=1280,800",
    (
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/156.0.0.0 Safari/537.36"
    ),
    (
        "--host-resolver-rules=MAP huggingface.co ~NOTFOUND, MAP hf.co ~NOTFOUND, MAP *.huggingface.co ~NOTFOUND, "
        "MAP *.kaggle.com ~NOTFOUND, MAP kaggle.com ~NOTFOUND, MAP openaipublic.blob.core.windows.net ~NOTFOUND, "
        "MAP raw.githubusercontent.com ~NOTFOUND"
    ),
]


def test_flags_are_the_hand_tested_headless_set() -> None:
    args = chrome.chrome_args("/bin/chrome", "/p", 9333, VERSION)
    assert args[0] == "/bin/chrome"
    assert args[1:] == [f.replace("<profile>", "/p").replace("<port>", "9333") for f in EXPECTED_FLAGS]


def test_headful_flags_differ_only_in_the_window() -> None:
    headless = chrome.chrome_args("/bin/chrome", "/p", 9333, VERSION)
    headful = chrome.chrome_args("/bin/chrome", "/p", 9333, VERSION, headless=False)
    assert "--headless=new" not in headful and "--window-position=0,25" in headful
    assert [a for a in headless if a != "--headless=new"] == [a for a in headful if a != "--window-position=0,25"]


def test_the_user_agent_follows_the_installed_major_version() -> None:
    assert "Chrome/157.0.0.0 Safari/537.36" in chrome.user_agent("157.2.99.1")
    assert chrome.user_agent(VERSION).startswith("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    (agent,) = [a for a in chrome.chrome_args("/bin/chrome", "/p", 1, "158.0.1.2") if a.startswith("--user-agent=")]
    assert agent.endswith("Chrome/158.0.0.0 Safari/537.36")


def test_no_flag_names_the_users_own_profile_or_a_shell() -> None:
    args = " ".join(chrome.chrome_args("/bin/chrome", "/p", 9333, VERSION))
    assert "Application Support" not in args and "--profile-directory" not in args and "&&" not in args


def test_the_binary_comes_from_the_environment_or_the_bundle_id(monkeypatch) -> None:
    system = Path("/Applications", "Google Chrome.app")  # joined, so no app path is spelled out in this file
    copies = [Path("/Volumes/Backup", "Google Chrome.app"), system, Path("/opt", "Chrome.app")]
    calls = []

    def mdfind(cmd, **_):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "".join(f"{c}\n" for c in copies) + "/opt/readme.txt\n", "")

    monkeypatch.setattr(chrome.subprocess, "run", mdfind)
    try:
        chrome.chrome_binary.cache_clear()
        monkeypatch.setenv("WEB_CHROME_BINARY", "/opt/x/chrome")
        assert chrome.chrome_binary() == Path("/opt/x/chrome") and calls == []
        monkeypatch.delenv("WEB_CHROME_BINARY")
        chrome.chrome_binary.cache_clear()
        assert chrome.chrome_binary() == system / "Contents/MacOS/Google Chrome"  # the system copy wins
        assert calls == [["mdfind", "kMDItemCFBundleIdentifier == 'com.google.Chrome'"]]
        copies.remove(system)
        chrome.chrome_binary.cache_clear()
        assert chrome.chrome_binary() == Path("/opt/Chrome.app/Contents/MacOS/Google Chrome")  # then the shortest
        copies.clear()
        chrome.chrome_binary.cache_clear()
        with pytest.raises(chrome.ChromeError, match="WEB_CHROME_BINARY"):
            chrome.chrome_binary()
        assert chrome.preflight([Path("/elsewhere")], table=list)[0].startswith("no app with bundle id")
    finally:
        chrome.chrome_binary.cache_clear()


def test_the_version_comes_from_the_binary_and_a_change_is_refused(env: SimpleNamespace, monkeypatch) -> None:
    assert chrome.chrome_version(env.fake) == VERSION
    monkeypatch.setenv("FAKE_CHROME_VERSION", "157.0.1.1")
    assert chrome.chrome_version(env.fake) == "157.0.1.1"
    chrome.check_version("157.0.1.1", None, "157.0.1.1")
    with pytest.raises(chrome.ChromeVersionChanged, match="156.0.8078.80 was recorded"):
        chrome.check_version("157.0.1.1", None, VERSION)
    with pytest.raises(chrome.ChromeError):
        chrome.chrome_version(env.tmp / "missing")


# --- which processes are bench processes ---

BENCH = "/work/runs/web"
APPS = "/opt/apps"  # the table's app bundles; any folder will do
TABLE = chrome.parse_process_table(
    f"""
  100     1 {APPS}/Google Chrome.app/Contents/MacOS/Google Chrome
  101   100 {APPS}/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers/Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper --type=renderer --lang=en-US
  200     1 {APPS}/Editor.app/Contents/MacOS/Editor Helper --type=gpu-process --user-data-dir=/example/Library/Application Support/Editor --gpu-preferences=UAAAA
  201     1 {APPS}/Notes.app/Contents/MacOS/Notes Helper --user-data-dir=/example/Library/Application Support/Notes --standard-schemes=app
  300     1 {APPS}/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new --user-data-dir={BENCH}/run1/chrome/profile --remote-debugging-port=9333 --user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) --host-resolver-rules=MAP hf.co ~NOTFOUND
  301   300 {APPS}/Google Chrome.app/Contents/MacOS/Google Chrome Helper --type=renderer --user-data-dir={BENCH}/run1/chrome/profile --lang=en-US
  302     1 {APPS}/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir={BENCH}/../web-other/profile
  303     1 {APPS}/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir=/example/Library/Application Support/Google/Chrome --no-first-run
"""
)


def test_the_process_table_parses_command_lines_with_spaces() -> None:
    assert [p.pid for p in TABLE] == [100, 101, 200, 201, 300, 301, 302, 303]
    assert TABLE[0].command == f"{APPS}/Google Chrome.app/Contents/MacOS/Google Chrome"
    assert chrome.profile_dir_of(TABLE[2].command) == "/example/Library/Application Support/Editor"
    assert chrome.profile_dir_of(TABLE[4].command) == f"{BENCH}/run1/chrome/profile"  # stops at the next flag
    assert chrome.profile_dir_of(TABLE[0].command) is None
    assert chrome.command_uses_profile(TABLE[4].command, f"{BENCH}/run1/chrome/profile")
    assert not chrome.command_uses_profile(TABLE[4].command, f"{BENCH}/run1/chrome/prof")  # no prefix match
    assert not chrome.command_uses_profile(TABLE[4].command, f"{BENCH}/run1/chrome/profile2")


def test_only_chrome_processes_on_a_bench_profile_are_bench_processes() -> None:
    found = chrome.bench_processes(TABLE, [Path(BENCH)])
    assert [p.pid for p in found] == [300, 301]  # not the user's Chrome (100, 101, 303), not other apps (200, 201)
    assert [p.pid for p in chrome.bench_processes(TABLE, [Path("/elsewhere")])] == []


def test_preflight_names_each_running_bench_process(env: SimpleNamespace) -> None:
    problems = chrome.preflight([Path(BENCH)], binary=env.fake, table=lambda: TABLE)
    assert len(problems) == 2 and all("still using a bench profile" in p for p in problems)
    assert "pid 300" in problems[0] and "pid 301" in problems[1]
    assert chrome.preflight([Path("/elsewhere")], binary=env.fake, table=lambda: TABLE) == []
    assert chrome.preflight([Path(BENCH)], binary=env.tmp / "missing", table=list)[0].startswith("Chrome not found")


def test_preflight_sees_a_real_process_on_a_bench_path(env: SimpleNamespace) -> None:
    profile = env.root / "leftover"
    profile.mkdir()
    proc = decoy(env, f"--user-data-dir={profile}")
    time.sleep(0.2)
    problems = chrome.preflight([env.root], binary=env.fake)
    assert len(problems) == 1 and f"pid {proc.pid}" in problems[0]
    proc.kill()
    proc.wait()
    assert chrome.preflight([env.root], binary=env.fake) == []


# --- bench profiles and signals ---


def test_a_bench_profile_has_the_marker_and_is_not_chromes_own_directory(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = env.root / "plain"
    plain.mkdir()
    assert not chrome.is_bench_profile(plain)
    with pytest.raises(chrome.UnsafeProfile):
        chrome.assert_bench_profile(plain)
    assert not chrome.is_bench_profile(env.root / "does-not-exist")

    marked = env.root / "marked"
    marked.mkdir()
    chrome._write_marker(marked, role="template", chrome_version=VERSION)
    assert chrome.is_bench_profile(marked) and chrome.assert_bench_profile(marked) == marked

    # even a marked directory is refused when it lies inside Chrome's own directory
    google = env.tmp / "Library" / "Application Support" / "Google"
    inside = google / "Chrome" / "Profile 1"
    inside.mkdir(parents=True)
    chrome._write_marker(inside, role="clone")
    monkeypatch.setattr(chrome, "default_chrome_dirs", lambda: [google])
    assert not chrome.is_bench_profile(inside) and not chrome.is_bench_profile(google / "Chrome")


def test_the_real_default_chrome_directory_is_never_a_bench_profile() -> None:
    (own,) = chrome.default_chrome_dirs()
    assert own == Path.home() / "Library" / "Application Support" / "Google"
    assert not chrome.is_bench_profile(own / "Chrome")
    assert not chrome.is_bench_profile(own / "Chrome" / "Default")


def test_launch_refuses_a_profile_without_the_marker_before_starting_anything(env: SimpleNamespace) -> None:
    plain = env.root / "plain"
    plain.mkdir()
    with pytest.raises(chrome.UnsafeProfile):
        chrome.launch(plain, binary=env.fake, version=VERSION)
    assert not (plain / "fake-chrome-args.json").exists()  # the binary was never run


def test_a_signal_reaches_only_a_process_that_uses_the_bench_profile(env: SimpleNamespace) -> None:
    profile = make_template(env)
    users_chrome = decoy(env)  # no --user-data-dir at all, like the user's Chrome
    users_dir = decoy(env, "--user-data-dir=/example/Library/Application Support/Google/Chrome")
    sibling = decoy(env, f"--user-data-dir={profile}2")  # a longer path that starts with the bench path
    mine = decoy(env, f"--user-data-dir={profile}")
    time.sleep(0.2)
    for other in (users_chrome, users_dir, sibling):
        with pytest.raises(chrome.UnsafeSignal):
            chrome.signal_bench_process(other.pid, profile, 15)
        assert alive(other)
    assert chrome.signal_bench_process(mine.pid, profile, 15) is True
    assert wait_gone(mine)
    assert chrome.signal_bench_process(mine.pid, profile, 15) is False  # gone: nothing to signal
    for other in (users_chrome, users_dir, sibling):
        assert alive(other)


def test_signals_also_need_a_bench_profile_argument(env: SimpleNamespace) -> None:
    plain = env.root / "plain"
    plain.mkdir()
    proc = decoy(env, f"--user-data-dir={plain}")
    time.sleep(0.2)
    with pytest.raises(chrome.UnsafeProfile):
        chrome.signal_bench_process(proc.pid, plain, 9)
    assert alive(proc)


def test_stopping_refuses_a_process_that_is_not_the_bench_chrome(env: SimpleNamespace) -> None:
    profile = make_template(env)
    stranger = decoy(env)
    time.sleep(0.2)
    instance = chrome.ChromeInstance(stranger, profile.resolve(), 9, VERSION, [])
    with pytest.raises(chrome.UnsafeSignal):
        instance.stop(term_timeout=1, graceful_timeout=1)
    assert alive(stranger)


# --- the template and the clones ---


def test_init_makes_the_template(env: SimpleNamespace) -> None:
    template = make_template(env)
    marker = chrome.read_marker(template)
    assert marker["kind"] == "web-bench-chrome-profile" and marker["role"] == "template"
    assert marker["chrome_version"] == VERSION and "Chrome/156.0.0.0" in marker["user_agent"]
    assert (template / "Default" / "Cookies").read_text() == "cookie\n"
    # a clean template: no History, none of the links of a Chrome that is gone, English and no prompts
    assert not (template / "Default" / "History").exists()
    assert not (template / "SingletonLock").is_symlink() and not (template / "SingletonLock").exists()
    prefs = json.loads((template / "Default" / "Preferences").read_text())
    assert prefs["profile"]["default_content_setting_values"] == {"notifications": 2, "geolocation": 2}
    assert prefs["intl"]["accept_languages"] == "en-US,en"
    args = json.loads((template / "fake-chrome-args.json").read_text())
    assert f"--user-data-dir={template.resolve()}" in args and "--headless=new" in args
    assert (template.parent / "profile-template.chrome.log").read_text() == ""  # Chrome's output goes to a log


def test_init_refuses_an_existing_template_unless_forced(env: SimpleNamespace) -> None:
    template = make_template(env)
    (template / "mark").write_text("old")
    with pytest.raises(chrome.ChromeError, match="--force"):
        chrome.init_template(template, binary=env.fake, settle_s=0.0, roots=[env.root])
    assert (template / "mark").exists()
    chrome.init_template(template, binary=env.fake, settle_s=0.0, roots=[env.root], force=True)
    assert not (template / "mark").exists()  # made again


def test_force_never_deletes_a_directory_that_is_not_a_bench_profile(env: SimpleNamespace) -> None:
    precious = env.root / "precious"
    precious.mkdir()
    (precious / "file").write_text("keep")
    with pytest.raises(chrome.UnsafeProfile):
        chrome.init_template(precious, binary=env.fake, settle_s=0.0, roots=[env.root], force=True)
    assert (precious / "file").read_text() == "keep"


def test_clone_uses_cp_dash_c_and_is_independent_of_the_template(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = make_template(env)
    commands = []
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: commands.append(cmd) or real_run(cmd, *a, **k))
    (template / "SingletonLock").symlink_to("old-host-1")
    how = chrome.clone_profile(template, env.root / "clone")
    assert commands[0][:3] == ["cp", "-c", "-R"] and how in ("apfs-clone", "copy")
    assert (env.root / "clone" / "Default" / "Cookies").read_text() == "cookie\n"
    assert not (env.root / "clone" / "SingletonLock").is_symlink()  # stale links are removed from the clone
    (env.root / "clone" / "Default" / "Cookies").write_text("changed")
    assert (template / "Default" / "Cookies").read_text() == "cookie\n"
    with pytest.raises(chrome.ChromeError, match="exists"):
        chrome.clone_profile(template, env.root / "clone")
    with pytest.raises(chrome.UnsafeProfile):
        chrome.clone_profile(env.root, env.root / "other")  # the root has no marker


# --- one run ---


def test_a_run_clones_launches_leaves_one_tab_and_cleans_up(env: SimpleNamespace) -> None:
    template = make_template(env)
    users_chrome = decoy(env)
    users_dir = decoy(env, "--user-data-dir=/example/Library/Application Support/Google/Chrome")
    chrome_dir = env.root / "run1" / "chrome"
    before = time.time()
    with run_chrome(env, chrome_dir, template) as run:
        t0 = time.time()
        profile = chrome_dir / "profile"
        assert chrome.is_bench_profile(profile) and chrome.read_marker(profile)["role"] == "clone"
        assert (profile / "Default" / "Cookies").read_text().startswith("cookie\n")  # the template's cookies
        assert run.browser_url == f"http://127.0.0.1:{run.chrome.port}" and run.chrome.port > 1024
        assert run.version == VERSION and run.clone in ("apfs-clone", "copy")
        # exactly one tab, on the start page, found only through the HTTP endpoints; the blank tab and the worker are gone
        tabs = run.chrome.tabs()
        assert [(t["url"], t["title"]) for t in tabs] == [("https://www.google.com/?hl=en", "Google")]
        assert run.start_tab["ready"] is True and run.start_tab["url"] == chrome.START_URL
        args = json.loads((profile / "fake-chrome-args.json").read_text())
        assert f"--remote-debugging-port={run.chrome.port}" in args and f"--user-data-dir={profile.resolve()}" in args
        assert chrome.preflight([env.root], binary=env.fake) != []  # while it runs, a second run is refused
        run.chrome.open_tab("https://example.org/page")  # the harness browses
    assert t0 > before

    # after: tabs saved, History kept, the clone deleted, a record written, and the process gone
    saved = json.loads((chrome_dir / "tabs.json").read_text())
    assert [t["url"] for t in saved] == ["https://www.google.com/?hl=en", "https://example.org/page"]
    assert all(set(t) == {"id", "title", "url"} for t in saved)
    assert not (chrome_dir / "profile").exists()
    meta = json.loads((chrome_dir / "chrome.json").read_text())
    assert meta["chrome_version"] == VERSION and meta["headless"] is True and meta["history_kept"] is True
    assert meta["exit"] == {
        "exit_code": 0,
        "graceful": True,
        "sigterm": False,
        "sigkill": False,
        "stragglers_killed": 0,
    }
    assert meta["n_tabs_at_exit"] == 2 and meta["start_tab"]["ready"] is True
    assert "<profile>" in " ".join(meta["args"]) and str(profile) not in " ".join(meta["args"])
    summary = history.summarize(chrome_dir / "History", t0)
    assert summary.urls == ["https://example.org/page"]  # the start page was visited before t0
    assert history.summarize(chrome_dir / "History", before - 5).n_nav == 2
    # the template is untouched, the other Chrome was never signalled, and nothing runs on the bench profile
    assert not (template / "Default" / "History").exists()
    assert alive(users_chrome) and alive(users_dir)
    assert chrome.preflight([env.root], binary=env.fake) == []


@pytest.mark.parametrize(
    ("setting", "exit_fields", "history_kept"),
    [
        ({}, {"graceful": True, "sigterm": False, "sigkill": False}, True),
        ({"FAKE_CHROME_IGNORE_CLOSE": "1"}, {"graceful": False, "sigterm": True, "sigkill": False}, False),
        (
            {"FAKE_CHROME_IGNORE_CLOSE": "1", "FAKE_CHROME_IGNORE_TERM": "1"},
            {"graceful": False, "sigterm": True, "sigkill": True},
            False,
        ),
    ],
    ids=["browser-close", "sigterm-after-ignored-close", "sigkill-after-ignored-term"],
)
def test_chrome_is_closed_gracefully_then_by_sigterm_then_by_sigkill(
    env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    setting: dict[str, str],
    exit_fields: dict[str, bool],
    history_kept: bool,
) -> None:
    template = make_template(env)
    for name, value in setting.items():
        monkeypatch.setenv(name, value)
    chrome_dir = env.root / "run" / "chrome"
    started = time.time()
    with run_chrome(env, chrome_dir, template, graceful_timeout=1.0, term_timeout=1.0):
        pass
    meta = json.loads((chrome_dir / "chrome.json").read_text())
    assert {k: meta["exit"][k] for k in exit_fields} == exit_fields
    assert meta["history_kept"] is history_kept and (chrome_dir / "History").exists() is history_kept
    assert not (chrome_dir / "profile").exists()  # the clone goes, whatever happened
    assert chrome.preflight([env.root], binary=env.fake) == []
    assert time.time() - started < 20


def test_a_version_change_is_refused_before_anything_starts(env: SimpleNamespace, monkeypatch) -> None:
    template = make_template(env)
    chrome_dir = env.root / "run" / "chrome"
    with (
        pytest.raises(chrome.ChromeVersionChanged, match="155.0.1.1 was recorded"),
        run_chrome(env, chrome_dir, template, expected_version="155.0.1.1"),
    ):
        pytest.fail("the run must not start")
    monkeypatch.setenv("FAKE_CHROME_VERSION", "157.0.1.1")  # Chrome updated itself after the template was made
    with (
        pytest.raises(chrome.ChromeVersionChanged, match="156.0.8078.80 was recorded"),
        run_chrome(env, chrome_dir, template),
    ):
        pytest.fail("the run must not start")
    assert not chrome_dir.exists()  # no clone, no process


def test_a_running_bench_chrome_stops_the_next_run_and_is_left_alone(env: SimpleNamespace) -> None:
    template = make_template(env)
    other = env.root / "other"
    chrome.clone_profile(template, other)
    leftover = chrome.launch(other, binary=env.fake, version=VERSION)
    try:
        with (
            pytest.raises(chrome.ChromeError, match="still using a bench profile"),
            run_chrome(env, env.root / "run" / "chrome", template),
        ):
            pytest.fail("the run must not start")
        assert alive(leftover.proc)  # refusing does not mean cleaning up another run's Chrome
        assert not (env.root / "run").exists()
    finally:
        leftover.stop(graceful_timeout=2)


@pytest.mark.parametrize("failure", ["FAKE_CHROME_EXIT_EARLY", "FAKE_CHROME_NO_HTTP"])
def test_a_chrome_that_does_not_start_is_reported_and_cleaned_up(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    template = make_template(env)
    monkeypatch.setenv(failure, "1")
    profile = env.root / "clone"
    chrome.clone_profile(template, profile)
    with pytest.raises(chrome.ChromeError, match="exited with code 7|did not answer"):
        chrome.launch(profile, binary=env.fake, version=VERSION, startup_timeout_s=1.5)
    assert chrome.preflight([env.root], binary=env.fake) == []  # the half-started Chrome was stopped


def test_the_start_tab_waits_for_the_page_to_load(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    template = make_template(env)
    profile = env.root / "clone"
    chrome.clone_profile(template, profile)
    monkeypatch.setenv("FAKE_CHROME_LOAD_S", "1.0")
    instance = chrome.launch(profile, binary=env.fake, version=VERSION)
    try:
        started = time.time()
        tab = chrome.leave_start_tab(instance, settle_s=0.0)
        assert tab["ready"] is True and tab["title"] == "Google" and time.time() - started >= 0.9
        assert len(instance.tabs()) == 1
    finally:
        instance.stop(graceful_timeout=2)


def test_a_start_page_that_never_loads_does_not_stop_the_run(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = make_template(env)
    profile = env.root / "clone"
    chrome.clone_profile(template, profile)
    monkeypatch.setenv("FAKE_CHROME_LOAD_S", "60")
    instance = chrome.launch(profile, binary=env.fake, version=VERSION)
    try:
        tab = chrome.leave_start_tab(instance, ready_timeout_s=1.0, settle_s=0.0)
        assert tab["ready"] is False and tab["url"] == chrome.START_URL  # the record says so; the tab is there
        assert [t["url"] for t in instance.tabs()] == [chrome.START_URL]
    finally:
        instance.stop(graceful_timeout=2)


def test_the_title_of_a_page_that_is_still_loading_is_a_placeholder() -> None:
    assert chrome._title_is_placeholder("", "https://www.google.com/")
    assert chrome._title_is_placeholder("about:blank", "about:blank")
    assert chrome._title_is_placeholder("google.com/?hl=en", "https://www.google.com/?hl=en")  # what Chrome shows
    assert chrome._title_is_placeholder("www.google.com/?hl=en", "https://www.google.com/?hl=en")
    assert chrome._title_is_placeholder("example.com", "https://example.com/")
    assert not chrome._title_is_placeholder("Google", "https://www.google.com/?hl=en")


def test_free_ports_can_be_bound() -> None:
    import socket

    port = chrome.free_port()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


# --- History ---

T0 = 1_800_000_000.0  # the start of a run, epoch seconds
NEG = -0x60000000  # CHAIN_END | SERVER_REDIRECT, as Chrome stores it: a signed 32-bit value


def make_history(path: Path, rows: list[tuple[float, str, int, int]], journal: bool = False) -> None:
    """rows: (epoch seconds, url, transition, opener visit id). Visit ids are 1, 2, 3 ... in the order given."""
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE urls(id INTEGER PRIMARY KEY AUTOINCREMENT, url LONGVARCHAR, title LONGVARCHAR, "
        "visit_count INTEGER DEFAULT 0 NOT NULL, typed_count INTEGER DEFAULT 0 NOT NULL, "
        "last_visit_time INTEGER NOT NULL, hidden INTEGER DEFAULT 0 NOT NULL);"
        "CREATE TABLE visits(id INTEGER PRIMARY KEY AUTOINCREMENT, url INTEGER NOT NULL, visit_time INTEGER NOT NULL, "
        "from_visit INTEGER, transition INTEGER DEFAULT 0 NOT NULL, segment_id INTEGER, "
        "visit_duration INTEGER DEFAULT 0 NOT NULL, opener_visit INTEGER);"
    )
    for i, (when, url, transition, opener) in enumerate(rows, 1):
        db.execute("INSERT INTO urls VALUES (?, ?, '', 1, 0, 0, 0)", (i, url))
        db.execute(
            "INSERT INTO visits VALUES (?, ?, ?, 0, ?, NULL, 0, ?)",
            (i, i, history.webkit_time(when), transition, opener),
        )
    db.commit()
    db.close()


CHAIN = 0x30000000  # CHAIN_START | CHAIN_END: a visit that is a whole chain
SERP = "https://www.google.com/search?q=capital+of+Australia&hl=en"


def test_history_counts_chain_ends_in_the_window(tmp_path: Path) -> None:
    rows = [
        (T0 - 30, "https://www.google.com/?hl=en", CHAIN | 6, 0),  # the start page: before t0
        (T0 + 1, "https://www.google.com/?hl=en&zx=1", CHAIN | 0, 1),  # Google's zx rewrite of it: not a navigation
        (T0 + 2, SERP, 0x10000000 | 7, 0),  # FORM_SUBMIT, start of a chain ...
        (T0 + 2.1, SERP + "&sca=1", 0x40000000 | 0x20000000, 0),  # ... its client-redirect end: the one navigation
        (T0 + 5, "https://www.google.com/goto?url=abc", 0x10000000 | 0, 0),  # a redirect hop, mid-chain
        (T0 + 5.01, "https://en.wikipedia.org/wiki/Canberra", NEG, 0),  # its end (negative in the database)
        (T0 + 6, "https://ads.example.net/frame", CHAIN | 4, 0),  # MANUAL_SUBFRAME: not top-level
        (T0 + 6, "https://ads.example.net/auto", CHAIN | 3, 0),  # AUTO_SUBFRAME
        (T0 + 7, "https://www.google.com/", 0x38000001, 0),  # typed through the API: a navigation, but on Google
        (T0 + 8, "https://www.google.com/?zx=2", CHAIN | 0, 9),  # zx rewrite of the one above
        (T0 + 9, "https://www.google.co.uk/search?q=x", CHAIN | 0, 0),  # another Google domain
        (T0 + 10, "https://www.bbc.co.uk/news", CHAIN | 8, 0),  # a reload of a page
        (T0 + 11, "https://www.google.com/sorry/index?continue=x", CHAIN | 0, 0),  # a CAPTCHA page: nav only
        (T0 + 99, "https://late.example.org/", CHAIN | 0, 0),  # after the run
    ]
    path = tmp_path / "History"
    make_history(path, rows)
    summary = history.summarize(path, T0, T0 + 60)
    assert summary.urls == [
        SERP + "&sca=1",
        "https://en.wikipedia.org/wiki/Canberra",
        "https://www.google.com/",
        "https://www.google.co.uk/search?q=x",
        "https://www.bbc.co.uk/news",
        "https://www.google.com/sorry/index?continue=x",
    ]
    assert (summary.n_nav, summary.n_search, summary.n_pages) == (6, 2, 2)
    assert [v.transition for v in summary.visits] == [0, 0, 1, 0, 8, 0]  # the core types
    assert abs(summary.visits[0].t - (T0 + 2.1)) < 1e-3
    # without an end time the run goes on to the last visit
    assert history.summarize(path, T0).urls[-1] == "https://late.example.org/"
    assert history.summarize(path, T0).n_pages == 3
    # a window that starts later drops the earlier visits
    assert history.summarize(path, T0 + 9).n_nav == 4
    assert history.summarize(path, T0 + 200).to_dict() == {
        "n_nav": 0,
        "n_search": 0,
        "n_pages": 0,
        "urls": [],
        "visits": [],
    }


def test_history_is_read_from_a_copy_and_the_original_is_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "History"
    make_history(path, [(T0 + 1, "https://example.org/", CHAIN, 0)])
    (tmp_path / "History-journal").write_bytes(b"")
    before = (path.read_bytes(), path.stat().st_mtime_ns, sorted(p.name for p in tmp_path.iterdir()))
    assert history.summarize(path, T0).n_nav == 1
    assert (path.read_bytes(), path.stat().st_mtime_ns, sorted(p.name for p in tmp_path.iterdir())) == before
    with pytest.raises(FileNotFoundError):
        history.summarize(tmp_path / "missing", T0)


def test_history_works_without_the_opener_column(tmp_path: Path) -> None:
    path = tmp_path / "History"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE urls(id INTEGER PRIMARY KEY, url TEXT); "
        "CREATE TABLE visits(id INTEGER PRIMARY KEY, url INTEGER, visit_time INTEGER, transition INTEGER);"
    )
    db.execute("INSERT INTO urls VALUES (1, 'https://example.org/')")
    db.execute("INSERT INTO visits VALUES (1, 1, ?, ?)", (history.webkit_time(T0 + 1), CHAIN))
    db.commit()
    db.close()
    assert history.summarize(path, T0).n_pages == 1


@pytest.mark.parametrize(
    ("url", "google", "search"),
    [
        ("https://www.google.com/search?q=a", True, True),
        ("https://google.com/search", True, True),
        ("https://www.google.co.uk/search?q=a", True, True),
        ("https://www.google.com.au/search?q=a", True, True),
        ("https://www.google.com/", True, False),
        ("https://www.google.com/searchbyimage", True, False),
        ("https://accounts.google.com/ServiceLogin", True, False),
        ("https://support.google.com/search", True, False),
        ("https://notgoogle.com/search", False, False),
        ("https://google.com.evil.example/search", False, False),
        ("https://en.wikipedia.org/wiki/Google", False, False),
        ("about:blank", False, False),
    ],
)
def test_google_hosts_and_result_pages(url: str, google: bool, search: bool) -> None:
    host = history.urlsplit(url).hostname or ""
    assert history.is_google_host(host) is google
    assert history.is_search_url(url) is search


def test_the_zx_parameter_is_removed_alone() -> None:
    assert history.without_zx("https://www.google.com/?hl=en&zx=17") == "https://www.google.com/?hl=en"
    assert history.without_zx("https://www.google.com/?zx=17") == "https://www.google.com/"
    assert history.without_zx("https://example.org/?a=1") == "https://example.org/?a=1"
    assert history.has_zx("https://www.google.com/?zx=1") and not history.has_zx("https://www.google.com/?x=1")


def test_the_webkit_clock_round_trips() -> None:
    assert history.webkit_time(0) == 11_644_473_600_000_000
    assert history.epoch_time(history.webkit_time(T0 + 0.25)) == pytest.approx(T0 + 0.25, abs=1e-6)


# --- the real Chrome (opt-in) ---

REAL = os.environ.get("WEB_REAL_CHROME") == "1"
MCP_BIN = Path(chrome.AGENT_ABILITY / "web" / "mcp" / "node_modules" / ".bin" / "chrome-devtools-mcp")


@pytest.mark.skipif(not REAL, reason="set WEB_REAL_CHROME=1: starts the installed Chrome (headless) and visits Google")
def test_real_chrome_google_search_through_the_tap_in_both_modes(tmp_path: Path) -> None:
    """The bench Chrome and the tap against the real thing. CPU only; no window appears and no input is simulated.

    Run with ``-s`` to see the sizes it prints.
    """
    if not MCP_BIN.exists():
        pytest.skip("chrome-devtools-mcp is not installed: cd web/mcp && npm ci")
    root = tmp_path / "bench"
    root.mkdir()
    chrome.init_template(root / "profile-template", roots=[root])
    assert chrome.chrome_version() == chrome.read_marker(root / "profile-template")["chrome_version"]

    def start_tap(mode: str, run: chrome.Run, name: str) -> Client:
        argv = ["--mode", mode, "--log", str(root / f"{name}.jsonl"), "--browser-url", run.browser_url]
        return Client(argv, dict(os.environ))

    def serp_or_skip(run: chrome.Run) -> None:
        if any("/sorry/" in t["url"] for t in run.chrome.tabs()):
            pytest.skip("Google served its CAPTCHA page; try again later")

    # D: the DOM arm
    with chrome.bench_chrome(root / "dom" / "chrome", template=root / "profile-template", roots=[root]) as run:
        t0 = time.time()
        client = start_tap("dom", run, "dom")
        assert [t["name"] for t in client.tools()] == list(tap.DOM_TOOLS)
        home = text_of(client.call("take_snapshot"))
        box = re.search(r'uid=(\S+) combobox "Search"', home)
        assert box, home[:500]
        client.call("fill", uid=box.group(1), value="capital of Australia")
        client.call("press_key", key="Enter")
        client.call("wait_for", text=["Canberra"], timeout=20000)
        serp_or_skip(run)
        serp = text_of(client.call("take_snapshot"))
        assert "Canberra" in serp and 'url="http' not in serp
        assert client.call("take_screenshot")["isError"] is True  # D never gets pixels
        client.close()
    entry = next(
        e for e in map(json.loads, (root / "dom.jsonl").read_text().splitlines()) if e.get("chars_raw", 0) > 10_000
    )
    print(f"\nsnapshot of the results page: {entry['chars_raw']} chars raw -> {entry['chars_out']} chars after the tap")
    assert entry["chars_out"] < 0.6 * entry["chars_raw"] and entry["urls_rewritten"] > 10
    visits = history.summarize(root / "dom" / "chrome" / "History", t0)
    print(f"D history: n_nav={visits.n_nav} n_search={visits.n_search} n_pages={visits.n_pages}")
    assert visits.n_search >= 1 and not (root / "dom" / "chrome" / "profile").exists()

    # S-cdp: the screenshot arm
    with chrome.bench_chrome(root / "screen" / "chrome", template=root / "profile-template", roots=[root]) as run:
        t0 = time.time()
        client = start_tap("screen", run, "screen")
        assert [t["name"] for t in client.tools()] == ["screenshot", "click", "type_text", "press_keys", "scroll"]
        shot = client.call("screenshot")["content"]
        assert [b["type"] for b in shot] == ["image"]
        import base64

        data = base64.b64decode(shot[0]["data"])
        size = tap.jpeg_size(data)
        print(f"first screenshot: {len(data)} bytes, {size[0]}x{size[1]} JPEG")
        assert data[:2] == b"\xff\xd8" and len(data) <= 256 * 1024
        client.call("click", x=500, y=395)  # the search box
        client.call("type_text", text="capital of Australia")
        client.call("press_keys", keys=["Enter"])
        time.sleep(3)
        serp_or_skip(run)
        data2 = base64.b64decode(client.call("screenshot")["content"][0]["data"])
        print(f"results screenshot: {len(data2)} bytes")
        client.call("scroll", direction="down", amount=2)
        client.call("press_keys", keys=["Alt+ArrowLeft"])
        client.close()
    visits = history.summarize(root / "screen" / "chrome" / "History", t0)
    print(f"S history: n_nav={visits.n_nav} n_search={visits.n_search} n_pages={visits.n_pages}")
    assert visits.n_search >= 1 and len(data2) <= 256 * 1024
    assert chrome.preflight([root]) == []  # no Chrome of the bench is left
