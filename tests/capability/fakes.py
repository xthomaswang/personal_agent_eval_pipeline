"""Offline stand-ins for the capability leg's outside world.

- A mock model server (a subprocess, so its pid can die like oMLX's) on an ephemeral port: GET
  /v1/models, and canned OpenAI and Anthropic answers with one tool call; it logs every request body.
- Fake Claude Code, Hermes and localmelo binaries, swapped in through the pins rather than through
  a test hook in the runner: the real bench.harness builders, env-file sourcing, Hermes home copy
  and config rewrite, and localmelo argument insertion all run. Each fake reads its endpoint and
  model the way the real harness does, makes two tool calls through the recorder proxy, records what
  it saw (argv, environment) in its run directory, and then does what its mode says ("memory-hog":
  a command that allocates 0.4 GiB page by page and holds it, then a solution; fake_hog.json records the
  command's pid and exit code).
- A "world": small git repos for hermes-agent (tagged v0.22.2) and the localmelo worktree, an agent
  root with byte copies of the live Claude Code config files and a minimal Hermes home, and pins
  that point at them. personal-agent-bench and harness-smoke are the live repos, read only through
  `git archive` at the SHAs pins.toml names; the task data is the live, read-only checkout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from modeleval import paths

LIVE_AGENT = paths.MLSYS / "agent"
SERVED_ID = "Test-Model-oQ5"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 0}

MOCK_UPSTREAM = r'''
import http.server, json, sys, threading
from pathlib import Path

port_file, log_file = Path(sys.argv[1]), Path(sys.argv[2])
lock = threading.Lock()
COMPLETION = {
    "id": "fake-1", "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "ok",
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}
MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant", "stop_reason": "tool_use", "stop_sequence": None,
    "content": [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "served"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        with lock, log_file.open("a") as f:
            f.write(json.dumps({"path": self.path, "body": body}) + "\n")
        if self.path.startswith("/v1/messages/count_tokens"):
            self._send(200, {"input_tokens": 10})
        elif self.path.startswith("/v1/messages"):
            self._send(200, {**MESSAGE, "model": body.get("model")})
        elif self.path.startswith("/v1/chat/completions"):
            self._send(200, {**COMPLETION, "model": body.get("model")})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
tmp = port_file.with_name(port_file.name + ".tmp")
tmp.write_text(str(server.server_port))
tmp.replace(port_file)
server.serve_forever()
'''

FAKE_AGENT = r'''
import json, os, re, shutil, signal, subprocess, sys, time, urllib.request
from pathlib import Path

HOG = """
import os, sys, time
size = int(float(sys.argv[1]) * 2**30)
block = bytearray(size)
for i in range(0, size, os.sysconf("SC_PAGESIZE")):
    block[i] = 1
time.sleep(float(sys.argv[2]))
"""

CONF = json.loads(Path(CONF_PATH).read_text())
argv = sys.argv[1:]
if argv[:1] == ["--version"]:
    print(f"{KIND} fake 0.0")
    sys.exit(0)
ws = Path.cwd()
slug = ws.name
modes = CONF["modes"].get(KIND, {})
mode = modes.get(slug, modes.get("*", "solve"))
if KIND == "claude_code":
    base, model = os.environ["ANTHROPIC_BASE_URL"], os.environ["ANTHROPIC_MODEL"]
    run_dir = Path(os.environ["CLAUDE_CONFIG_DIR"]).parent
elif KIND == "hermes":
    home = Path(os.environ["HERMES_HOME"])
    text = (home / "config.yaml").read_text()
    base = re.search(r"^\s*api:\s*(\S+)", text, re.M).group(1)
    model = re.search(r"^\s*default:\s*(\S+)", text, re.M).group(1)
    run_dir = home.parent
else:
    base, model = argv[argv.index("--base-url") + 1], argv[argv.index("--chat-model") + 1]
    run_dir = Path(os.environ["LOCALMELO_MEMORY_DIR"]).parent
(run_dir / "fake_seen.json").write_text(json.dumps({"argv": argv, "env": dict(os.environ), "base": base,
                                                    "model": model, "mode": mode, "pid": os.getpid()}))
if mode == "wrong-model":
    model = "Some-Other-Model"


def call():
    root = base.rstrip("/").removesuffix("/v1")
    if KIND == "claude_code":
        url = root + "/v1/messages"
        body = {"model": model, "max_tokens": 32000, "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
                "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]}
    else:
        url = root + "/v1/chat/completions"
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "temperature": 0.3,
                "reasoning_effort": "none",
                "tools": [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}]}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


call()
call()
if mode == "sleep":
    time.sleep(120)
if mode == "kill-server":
    os.kill(int(Path(CONF["server_pid_file"]).read_text()), signal.SIGKILL)
    time.sleep(0.5)
if mode == "memory-hog":  # the agent's command runs away; the runner's memory watchdog should kill it
    hog = subprocess.Popen([sys.executable, "-I", "-c", HOG, "0.4", "60"])
    (run_dir / "fake_hog.json").write_text(json.dumps({"pid": hog.pid, "returncode": hog.wait()}))
if mode in ("solve", "wrong-model", "self-timeout-solved", "memory-hog"):
    if slug == "warmup":
        (ws / "warmup.txt").write_text("ready\n")
    elif slug.startswith("aider__"):
        name = slug.split("__", 1)[1]
        shutil.copyfile(Path(CONF["polyglot"], name, ".meta", "example.py"), ws / (name.replace("-", "_") + ".py"))
if KIND == "claude_code":
    print(json.dumps({"type": "result", "num_turns": 2, "subtype": "success", "is_error": False,
                      "permission_denials": [], "duration_ms": 5}))
elif KIND == "localmelo":
    status = "timeout" if mode.startswith("self-timeout") else "completed"
    print(json.dumps({"task_id": "t1", "status": status, "steps": 2}))
'''

FAKE_KINDS = ("claude_code", "hermes", "localmelo")


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
    }  # fmt: skip
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def _repo(path: Path, files: dict[str, str], tag: str | None = None) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    if tag:
        _git(path, "tag", tag)
    return _git(path, "rev-parse", "HEAD")


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def fake_binary(path: Path, kind: str, conf: Path) -> Path:
    header = f"#!{sys.executable} -I\nKIND = {kind!r}\nCONF_PATH = {str(conf)!r}\n"
    return _executable(path, header + FAKE_AGENT)


def repo_pins() -> dict[str, Any]:
    return tomllib.loads((paths.REPO / "pins.toml").read_text())


def build_world(root: Path) -> dict[str, Any]:
    """Temporary agent root, harness repos, fake binaries and pins; returns handles and pins_raw."""
    live = repo_pins()
    conf = root / "fake-conf.json"
    data_root = Path(live["data"]["root"]).expanduser()
    polyglot = data_root / "aider-polyglot-benchmark" / live["data"]["aider_polyglot"][:12] / "python" / "exercises" / "practice"
    conf.write_text(json.dumps({"modes": {}, "polyglot": str(polyglot), "server_pid_file": str(root / "server.pid")}))

    agent = root / "agent"
    (agent / "configs" / "claude-code").mkdir(parents=True)
    shutil.copyfile(LIVE_AGENT / "configs/claude-code/claude-code.env", agent / "configs/claude-code/claude-code.env")
    (agent / "homes" / "claude-code").mkdir(parents=True)
    shutil.copyfile(LIVE_AGENT / "homes/claude-code/settings.json", agent / "homes/claude-code/settings.json")
    hermes_home = agent / "homes" / "hermes"
    for name in ("memories", "skills", "cron", "tools", "installs", "logs", "sessions", "cache"):
        (hermes_home / name).mkdir(parents=True)
    shutil.copyfile(LIVE_AGENT / "homes/hermes/config.yaml", hermes_home / "config.yaml")
    (hermes_home / ".env").write_text("# no secrets\n")
    (hermes_home / "SOUL.md").write_text("test soul\n")
    (hermes_home / "tools" / "facts.json").write_text('{"packages": {}, "schema": 1}\n')
    (hermes_home / "logs" / "old.log").write_text("not copied\n")

    hermes_repo = root / "hermes-agent"
    hermes_sha = _repo(hermes_repo, {"README.md": "hermes\n"}, tag="v0.22.2")
    hermes_bin = fake_binary(agent / ".venv" / "bin" / "hermes", "hermes", conf)
    melo_repo = root / "localmelo"
    melo_sha = _repo(melo_repo, {".gitignore": ".venv/\n", "README.md": "melo\n"})
    fake_binary(melo_repo / ".venv" / "bin" / "melo", "localmelo", conf)
    claude_bin = fake_binary(root / "claude" / "versions" / "2.1.302", "claude_code", conf)

    ws_root = root / "w1" / "w2" / "w3" / "ws"  # keeps Hermes's 6-level marker window inside tmp
    pins_raw = {
        "epoch": {},
        "snapshot": live["snapshot"],
        "configs": {**live["configs"], "agent_root": str(agent)},
        "localmelo": {"repo": str(melo_repo), "sha": melo_sha},
        "hermes": {"repo": str(hermes_repo), "tag": "v0.22.2", "sha": hermes_sha, "bin": str(hermes_bin)},
        "claude_code": {"version": "2.1.302", "bin": str(claude_bin)},
        "data": live["data"],
        "envs": live["envs"],
        "workspace": {"ws_root": str(ws_root)},
    }
    return {
        "root": root, "conf": conf, "agent": agent, "hermes_repo": hermes_repo, "melo_repo": melo_repo,
        "claude_bin": claude_bin, "ws_root": ws_root, "pins_raw": pins_raw, "polyglot": polyglot,
    }  # fmt: skip


def set_modes(world: dict[str, Any], modes: dict[str, dict[str, str]]) -> None:
    conf = json.loads(world["conf"].read_text())
    conf["modes"] = modes
    world["conf"].write_text(json.dumps(conf))


class MockServer:
    """The mock upstream as a child process (its pid stands in for oMLX's)."""

    def __init__(self, root: Path, pid_file: Path) -> None:
        self.root = root
        self.script = root / "mock_upstream.py"
        self.script.write_text(MOCK_UPSTREAM)
        self.log = root / "upstream.jsonl"
        self.pid_file = pid_file
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0

    def start(self) -> MockServer:
        port_file = self.root / f"port-{time.time_ns()}"
        self.proc = subprocess.Popen([sys.executable, "-I", "-B", str(self.script), str(port_file), str(self.log)])
        deadline = time.monotonic() + 20
        while not port_file.exists():
            if time.monotonic() > deadline or self.proc.poll() is not None:
                raise RuntimeError("mock upstream did not start")
            time.sleep(0.05)
        self.port = int(port_file.read_text())
        self.pid_file.write_text(str(self.proc.pid))
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        if self.proc:
            self.proc.wait()

    def requests(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]
