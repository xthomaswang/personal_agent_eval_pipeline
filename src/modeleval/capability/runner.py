"""The capability runner (DESIGN §6.1-6.4): Claude Code, Hermes and localmelo on the gate and ranking
tasks, through the vendored bench.harness and smoke packages, one (harness, task) at a time.

The orchestrator starts it as

    envs/capability/.venv/bin/python -I -B -m modeleval.capability.runner --spec <run_root>/spec.json

with cwd=REPO and the environment from `allowed_env()` (importable on its own: this module's top level
is stdlib plus modeleval.paths/progress). Exit codes: 0 done; 75 yield (battery pause or stop, hold,
epoch drift; the reason is in <run_root>/yield.json, restart with resume); 76 server lost (relaunch,
then restart with resume); 2 preflight refused; 1 error.

Why the runner owns the task loop instead of calling smoke.runner.run_suite: it gives yield points
between tasks, lets a restart skip pairs that already have a valid verdict, and lets a retry get a
new run directory (bench.harness refuses an existing one). Each task is still exactly
smoke.runner.run_one, so workspaces, prompts and checkers are harness-smoke's. It runs under the memory
watchdog of memwatch.py (DESIGN §6.1), which kills runaway processes the agents or the checker start.

What is changed in the vendored code at run time (no snapshot file is edited):
- every location-derived constant of bench.manifest, bench.harness, smoke.paths and smoke.adapters
  (harnesses.md B5, B6, B10, B11): the snapshot sits at another depth than the live trees, and the
  pipeline runs its own server, homes, environments and workspaces. A selftest asserts each value;
- bench.harness._start_proxy: a copy that adds `--sampling <proxy_sampling>` (the original never
  passes it, so every model got Qwen's preset) and runs the proxy with -B;
- bench.manifest._git: only `rev-parse` and `--no-optional-locks status --porcelain`, no `describe
  --dirty` (which can refresh another session's index); manifest.collect: once per process, since
  it runs system_profiler and the harnesses' --version probes;
- smoke.adapters.agent_env: adds GIT_CEILING_DIRECTORIES=<ws_root>;
- smoke.runner.Settings.run_name: attempt-aware (`<slug>`, then `<slug>__a2`, ...);
- smoke.data._git_state: the same two allowed git commands.

Spec keys beyond docs/CONTRACTS.md, all optional: `launch` (int, recorded per row), `harnesses`
(a subset, default all three), `only` (fnmatch patterns over task ids, for debugging),
`server_log` (oMLX log path, scanned per task for template errors), `selftest` ({"battery": <int>}
forces the battery reading, {"hold_file": <path>} replaces runs/hold; tests only).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from modeleval import paths, progress
from modeleval.capability import checks, export, memwatch
from modeleval.capability import epoch as epochs

EXIT_DONE, EXIT_ERROR, EXIT_PREFLIGHT, EXIT_YIELD, EXIT_SERVER_LOST = 0, 1, 2, 75, 76

HARNESSES = ("claude_code", "hermes", "localmelo")
SETS = ("gate", "ranking")
# DESIGN §6.3: the gate's Aider exercises (harness-smoke's AIDER_CORE + AIDER_EXTENDED) and set sizes
GATE_AIDER = (
    "transpose", "proverb", "pig-latin", "list-ops", "variable-length-quantity", "robot-name",
    "grade-school", "affine-cipher", "scale-generator", "wordy", "phone-number", "react",
)  # fmt: skip
GATE_MCPMARK_COUNT = 10
RANKING_COUNT = 22
CONTEXT_WINDOW = 131072  # DESIGN §2: every model is served with max_context_window 131072
LOCALMELO_REQUEST_TIMEOUT_S = 1800
LOCALMELO_MARGIN_S = 60  # localmelo stops itself this long before the harness kill, so its record is written

STATE_FILE = "state.json"
YIELD_FILE = "yield.json"
SELFTEST_FILE = "selftest.json"
SERVED_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Best effort: oMLX 0.7's TypeError fallback in the batched engine re-renders silently; these are the
# lines it does log when a template fails
TEMPLATE_ERROR_PATTERNS = ("Chat template rendering failed", "Chat template error")

# --- the environment (DESIGN §6.2) ---

ALLOWED_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TERM", "LANG", "HF_HOME", "NVM_DIR",
    "PAB_ROOT", "HARNESS_SMOKE_DATA", "PYTHONDONTWRITEBYTECODE", "GIT_OPTIONAL_LOCKS",
    memwatch.PROC_LIMIT_ENV, memwatch.TREE_LIMIT_ENV,
})  # fmt: skip
ALLOWED_PREFIXES = ("LC_", "BENCH_", "OMLX_")
FORBIDDEN_PATTERNS = ("CLAUDE*", "ANTHROPIC*", "OPENAI*", "*TOKEN*", "*_KEY")
FORCED_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"}


def _forbidden(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name.upper(), pattern) for pattern in FORBIDDEN_PATTERNS)


def forbidden_names(environ: Mapping[str, str]) -> list[str]:
    """Variable names that must never reach the runner: credentials and the driving session's own."""
    return sorted(name for name in environ if _forbidden(name))


def allowed_env(environ: Mapping[str, str], extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The runner's environment: the allowlist of DESIGN §6.2, minus anything credential-shaped
    (BENCH_UPSTREAM_TOKEN_CMD matches BENCH_* and *TOKEN*: dropped), plus `extra`.

    DEVELOPER_DIR is dropped, so manifests record the global Xcode. PYTHONDONTWRITEBYTECODE=1 and
    GIT_OPTIONAL_LOCKS=0 are forced: the proxy and the checkers run without -I and inherit them.
    """
    env = {
        name: value
        for name, value in environ.items()
        if (name in ALLOWED_NAMES or name.startswith(ALLOWED_PREFIXES)) and not _forbidden(name)
    }
    bad = forbidden_names(extra or {})
    if bad:
        raise ValueError(f"refusing to pass {', '.join(bad)} to the capability runner")
    return {**env, **(extra or {}), **FORCED_ENV}


# --- Claude Code config copies (DESIGN §6.1) ---

CLAUDE_MODEL_VARS = (
    "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
)  # fmt: skip
# Must survive in the copy, exactly: they keep the benchmarked Claude Code off the user's account,
# memory and CLAUDE.md files and off the network
CLAUDE_SAFETY = {
    "ANTHROPIC_API_KEY": "",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}
CLAUDE_NONEMPTY = ("CLAUDE_CONFIG_DIR", "ANTHROPIC_AUTH_TOKEN")
_ASSIGNMENT = re.compile(
    r"""^(?P<lead>\s*(?:export\s+)?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>"[^"]*"|'[^']*'|[^\s#;]*)(?P<rest>.*)$"""
)
_BARE = re.compile(r"^[A-Za-z0-9._/:-]+$")


class Refused(RuntimeError):
    """A preflight check failed (exit 2)."""


def claude_values(served_id: str, claude_bin: Path) -> dict[str, str]:
    """The variables the per-run claude-code.env replaces, as shell literals' values."""
    home = str(paths.HOME)
    bin_value = "$HOME" + str(claude_bin)[len(home) :] if str(claude_bin).startswith(home + "/") else str(claude_bin)
    return {
        **dict.fromkeys(CLAUDE_MODEL_VARS, served_id),
        "CLAUDE_BIN": bin_value,
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(CONTEXT_WINDOW),
    }


def patch_claude_env(text: str, values: Mapping[str, str]) -> str:
    """A byte copy of claude-code.env with only the named variables' values replaced, by name.

    A line keeps its `export`, spacing and trailing comment; a value that is already right keeps its
    spelling. Refuses when a named variable has no assignment.
    """
    seen: collections.Counter[str] = collections.Counter()
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        match = _ASSIGNMENT.match(body)
        if match and match["name"] in values:
            name, old = match["name"], match["value"]
            seen[name] += 1
            new = values[name]
            if old.strip("\"'") != new:
                token = new if (_BARE.match(new) and not old.startswith(("'", '"'))) else f'"{new}"'
                body = f"{match['lead']}{name}={token}{match['rest']}"
            line = body + line[len(line.rstrip("\r\n")) :]
        out.append(line)
    missing = [name for name in values if not seen[name]]
    if missing:
        raise Refused(f"claude-code.env has no assignment of {', '.join(missing)}")
    return "".join(out)


def shell_env(env_file: Path) -> dict[str, str]:
    """What bench.harness's launcher shell holds after sourcing env_file: exported variables, plus
    CLAUDE_BIN (a plain shell variable the launcher execs)."""
    script = '. "$1" >/dev/null 2>&1; env -0; printf "__SHELL__CLAUDE_BIN=%s\\0" "$CLAUDE_BIN"'
    proc = subprocess.run(
        ["/usr/bin/env", "-i", f"HOME={paths.HOME}", "PATH=/usr/bin:/bin", "/bin/sh", "-c", script, "sh", str(env_file)],
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise Refused(f"sourcing {paths.tilde(env_file)} failed: {proc.stderr.decode(errors='replace')[:200]}")
    out = {}
    for item in proc.stdout.decode(errors="replace").split("\0"):
        if "=" in item:
            name, value = item.split("=", 1)
            out[name] = value
    return out


def claude_env_problems(original: str, patched: str, env: Mapping[str, str], served_id: str, claude_bin: Path) -> list[str]:
    """Everything wrong with a patched copy: other lines changed, safety variables lost, values off."""
    found = []
    old_lines, new_lines = original.splitlines(), patched.splitlines()
    if len(old_lines) != len(new_lines):
        found.append("the copy has a different number of lines")
    replaced = {*CLAUDE_MODEL_VARS, "CLAUDE_BIN", "CLAUDE_CODE_AUTO_COMPACT_WINDOW"}
    for old, new in zip(old_lines, new_lines, strict=False):
        match = _ASSIGNMENT.match(old)
        if old != new and not (match and match["name"] in replaced):
            found.append(f"an unnamed line changed: {old!r}")
    for name, value in CLAUDE_SAFETY.items():
        if env.get(name) != value:
            found.append(f"{name} must be exported as {value!r}, the copy gives {env.get(name)!r}")
    for name in CLAUDE_NONEMPTY:
        if not env.get(name):
            found.append(f"{name} must be exported and non-empty")
    for name in CLAUDE_MODEL_VARS:
        if env.get(name) != served_id:
            found.append(f"{name} is {env.get(name)!r}, not {served_id!r}")
    if env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") != str(CONTEXT_WINDOW):
        found.append(f"CLAUDE_CODE_AUTO_COMPACT_WINDOW is {env.get('CLAUDE_CODE_AUTO_COMPACT_WINDOW')!r}")
    if env.get("__SHELL__CLAUDE_BIN") != str(claude_bin):
        found.append(f"CLAUDE_BIN is {env.get('__SHELL__CLAUDE_BIN')!r}, not the pinned {paths.tilde(claude_bin)}")
    return found


def patch_claude_settings(text: str, served_id: str) -> str:
    """settings.json with only the value of "model" replaced (bytes otherwise unchanged)."""
    pattern = re.compile(r'("model"\s*:\s*)"(?:[^"\\]|\\.)*"')
    if len(pattern.findall(text)) != 1:
        raise Refused('settings.json must have exactly one "model" entry')
    patched = pattern.sub(lambda m: m.group(1) + json.dumps(served_id), text)
    before, after = json.loads(text), json.loads(patched)
    if after.get("model") != served_id or {**before, "model": served_id} != after:
        raise Refused("settings.json: replacing the model changed something else")
    return patched


def write_claude_config(vendor: Path, dest: Path, served_id: str, claude_bin: Path) -> tuple[Path, Path]:
    """The per-run claude-code.env and Claude Code home (settings.json only), checked; returns both."""
    original = (vendor / epochs.CLAUDE_ENV_COPY).read_text()
    patched = patch_claude_env(original, claude_values(served_id, claude_bin))
    env_file = dest / "claude-code.env"
    home = dest / "claude-home"
    home.mkdir(parents=True, exist_ok=True)
    env_file.write_text(patched)
    problems = claude_env_problems(original, patched, shell_env(env_file), served_id, claude_bin)
    if problems:
        raise Refused("per-run claude-code.env: " + "; ".join(problems))
    settings = (vendor / epochs.CLAUDE_SETTINGS_COPY).read_text()
    (home / "settings.json").write_text(patch_claude_settings(settings, served_id))
    return env_file, home


# --- the workspace root (DESIGN §6.2) ---

# Hermes v0.22.2 agent/coding_context.py _PROJECT_MARKERS: any of them above a workspace makes Hermes
# treat that directory as the project root
PROJECT_MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json", "tsconfig.json", "deno.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile", "composer.json", "mix.exs",
    "pubspec.yaml", "CMakeLists.txt", "Makefile", "Dockerfile", "AGENTS.md", "CLAUDE.md", ".cursorrules",
)  # fmt: skip
MARKER_LEVELS = 6  # Hermes looks 7 levels up from a task workspace, one below ws_root


def ws_root_problems(ws_root: Path) -> list[str]:
    """ws_root must sit in no git repository and below no project marker, or an agent could run git
    or uv against someone's project, and Hermes would pick that project up as context."""
    ws_root.mkdir(parents=True, exist_ok=True)
    found = []
    if epochs._git(ws_root, "rev-parse", "--git-dir", check=False).returncode == 0:
        found.append(f"{paths.tilde(ws_root)} is inside a git repository")
    skip = {paths.HOME.resolve(), Path(tempfile.gettempdir()).resolve()}
    current = ws_root.resolve()
    for directory in (current, *current.parents)[:MARKER_LEVELS]:
        if directory in skip:
            continue
        hits = [m for m in PROJECT_MARKERS if (directory / m).exists()]
        if hits:
            found.append(f"{paths.tilde(directory)} has project markers {', '.join(hits)}")
    return found


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def workspace_lock(ws_root: Path, owner: str) -> Iterator[None]:
    """ws_root/.lock for this process; a stale lock of the same run (dead pid) is taken over."""
    lock = ws_root / ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                held = json.loads(lock.read_text())
            except (OSError, json.JSONDecodeError):
                held = {}
            if held.get("owner") == owner and not _pid_alive(int(held.get("pid") or 0)):
                lock.unlink(missing_ok=True)
                continue
            raise Refused(f"{paths.tilde(lock)} is held by {held or 'an unknown process'}") from None
    else:
        raise Refused(f"cannot take {paths.tilde(lock)}")
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "owner": owner}, f)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


# --- the server and the boundaries ---


def server_problem(pid: int, base_url: str, timeout_s: float = 10.0) -> str | None:
    """None when the server process lives and GET /v1/models answers 200."""
    if not _pid_alive(pid):
        return f"server pid {pid} is gone"
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout_s) as response:
            if response.status != 200:
                return f"GET /v1/models answered {response.status}"
    except (OSError, urllib.error.URLError) as exc:
        return f"GET /v1/models failed ({exc})"
    return None


def _pmset_battery() -> tuple[int, bool] | None:
    try:
        text = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.search(r"(\d+)%", text)
    return (int(found.group(1)), "AC Power" in text) if found else None


def boundary_rule(kind: str, percent: int | None) -> str:
    """The user's rule (DESIGN §5.5) as quiet.boundary states it: pause below 50 % (60 % before a
    harness block or a long unit); stop below 20 % at a capability task."""
    if percent is None:
        return "go"
    if kind == "task" and percent < 20:
        return "stop"
    return "pause" if percent < (60 if kind in ("long-unit", "harness-block") else 50) else "go"


class Boundaries:
    """Battery and hold checks: modeleval.quiet when it imports, else the same rule locally."""

    def __init__(self, selftest: Mapping[str, Any]) -> None:
        self.forced = selftest.get("battery")
        self.hold_file = Path(selftest["hold_file"]) if selftest.get("hold_file") else None
        try:
            from modeleval import quiet

            self._battery, self._boundary, self._hold = quiet.battery, quiet.boundary, quiet.hold_requested
            self.source = "modeleval.quiet"
        except Exception:  # quiet.py is another module's; its absence must not stop the leg
            self._battery = self._boundary = self._hold = None
            self.source = "fallback"

    def battery(self) -> int | None:
        if self.forced is not None:
            return int(self.forced)
        if self._battery is not None:
            with contextlib.suppress(Exception):
                return int(self._battery()[0])
        found = _pmset_battery()
        return found[0] if found else None

    def decision(self, kind: str) -> str:
        if self.forced is None and self._boundary is not None:
            return self._boundary(kind)
        return boundary_rule(kind, self.battery())

    def hold(self) -> bool:
        if self.hold_file is not None:
            return self.hold_file.exists()
        if self._hold is not None:
            return bool(self._hold())
        return paths.HOLD_FILE.exists()


# --- run state ---


class State:
    """Every finished run of this capability leg, in run_root/state.json (rewritten after each)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {"hygiene": []}
        if path.exists():
            doc = json.loads(path.read_text())
            self.records = doc.get("records", [])
            self.meta = {**self.meta, **doc.get("meta", {})}

    def save(self) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps({"records": self.records, "meta": self.meta}, indent=1) + "\n")
        tmp.replace(self.path)

    def pair(self, set_name: str, harness: str, task_id: str) -> list[dict[str, Any]]:
        return [r for r in self.records if (r["set"], r["harness"], r["task"]) == (set_name, harness, task_id)]

    def done(self, set_name: str, harness: str, task_id: str) -> bool:
        """A valid verdict, or invalid twice (excluded after its one re-run)."""
        runs = self.pair(set_name, harness, task_id)
        return any(r["valid"] for r in runs) or sum(r["verdict"] == "invalid" for r in runs) >= 2

    def next_attempt(self, harness: str, task_id: str) -> int:
        return 1 + sum(1 for r in self.records if (r["harness"], r["task"]) == (harness, task_id))

    def partition(self) -> tuple[list[dict], list[dict], list[dict]]:
        """(final row per scored pair, superseded attempts, warm-ups), in run order."""
        last: dict[tuple[str, str, str], dict] = {}
        for r in self.records:
            if r["scored"]:
                last[(r["set"], r["harness"], r["task"])] = r
        finals = {id(r) for r in last.values()}
        rows = [r for r in self.records if id(r) in finals]
        superseded = [r for r in self.records if r["scored"] and id(r) not in finals]
        warmups = [r for r in self.records if not r["scored"]]
        return rows, superseded, warmups


# --- the vendored harness, configured (runs in envs/capability) ---


def _children_of(pid: int) -> list[tuple[int, int]]:
    """(pid, pgid) of pid's live children, leaving out the `ps` that lists them."""
    with subprocess.Popen(["ps", "-A", "-o", "pid=,ppid=,pgid=,stat="], stdout=subprocess.PIPE, text=True) as probe:
        listing, _ = probe.communicate()
    out = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 4 and int(parts[1]) == pid and int(parts[0]) != probe.pid and not parts[3].startswith("Z"):
            out.append((int(parts[0]), int(parts[2])))
    return out


def kill_children() -> None:
    """End every child process group (a harness in its own session, the proxy, a checker), so a
    stopped runner leaves nothing that holds the GPU or the workspace."""
    me, my_group = os.getpid(), os.getpgid(0)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        children = _children_of(me)
        if not children:
            return
        for pid, group in children:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if group == pid and group != my_group:
                    os.killpg(group, sig)
                else:
                    os.kill(pid, sig)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _children_of(me):
            with contextlib.suppress(ChildProcessError):
                os.waitpid(-1, os.WNOHANG)
            time.sleep(0.1)


def proxy_starter(harness: ModuleType, manifest: ModuleType, sampling: str):  # noqa: ANN201
    """bench.harness._start_proxy (harness.py:161-205) plus `--sampling` and -B. The upstream token
    option is left out: the runner's environment carries no token command."""

    def start_proxy(
        run_dir: Path,
        name: str,
        thinking: str,
        max_steps: int | None,
        random_seed: bool = False,
        one_reasoning_field: bool = False,
        log_bodies: bool = False,
    ) -> tuple[subprocess.Popen[str], str]:
        if log_bodies:
            raise ValueError("the pipeline never logs request bodies (DESIGN §8)")
        cmd = [
            sys.executable, "-B", "-m", "bench.proxy",
            "--port", "0",
            "--upstream", harness.UPSTREAM,
            "--harness", name,
            "--run-id", run_dir.name,
            "--run-dir", str(run_dir),
            "--thinking", thinking,
            "--sampling", sampling,
        ]  # fmt: skip
        if max_steps:
            cmd += ["--max-steps", str(max_steps)]
        if random_seed:
            cmd.append("--random-seed")
        if one_reasoning_field:
            cmd.append("--one-reasoning-field")
        log_path = run_dir / "proxy.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, cwd=manifest.BENCH_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            found = re.search(r"listening on (http://\S+)", log_path.read_text())
            if found:
                return proc, found.group(1)
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        proc.terminate()
        raise RuntimeError(f"proxy did not start; see {log_path}")

    return start_proxy


def safe_git(snapshots: Mapping[Path, str]):  # noqa: ANN201
    """bench.manifest._git without `describe --dirty`: rev-parse and a status that takes no lock.
    The vendored snapshot roots report their pinned commit."""

    def _git(repo: Path) -> dict[str, Any]:
        resolved = Path(repo).resolve()
        if resolved in snapshots:
            sha = snapshots[resolved]
            return {"path": str(repo), "present": True, "commit": sha, "describe": sha[:7], "dirty": False, "snapshot": True}
        if not (Path(repo) / ".git").exists():
            return {"path": str(repo), "present": False}
        head, dirty = epochs.repo_state(Path(repo))
        describe = None if head is None else head[:7] + ("-dirty" if dirty else "")
        return {"path": str(repo), "present": True, "commit": head, "describe": describe, "dirty": dirty}

    return _git


def localmelo_args(wall_s: float) -> list[str]:
    """DESIGN §6.1: a long request timeout (slow models), localmelo's own stop a minute before the
    harness kill so its JSON run record is still written, and that record on stdout."""
    return [
        "--request-timeout", str(LOCALMELO_REQUEST_TIMEOUT_S),
        "--timeout", f"{wall_s - LOCALMELO_MARGIN_S:g}",
        "--output-format", "json",
    ]  # fmt: skip


def configure(spec: Mapping[str, Any], epoch: Mapping[str, Any], run_root: Path) -> SimpleNamespace:
    """Import the vendored packages and point every location-derived value at the pipeline's own."""
    vendor = Path(epoch["dir"])
    pins = epoch["pins"]
    pab_root, smoke_root = vendor / epochs.PAB_DIR, vendor / epochs.SMOKE_DIR
    ws_root, base_url = Path(spec["ws_root"]), spec["base_url"]
    model = spec["model"]
    served_id = model["served_id"]
    agent_bin = pins["envs"]["agent_tools"] / ".venv" / "bin"
    scratch = run_root / "scratch"

    os.environ["PAB_ROOT"] = str(pab_root)
    os.environ["HARNESS_SMOKE_DATA"] = str(pins["data"]["root"])
    sys.path[:0] = [str(smoke_root), str(pab_root)]
    from bench import harness, manifest
    from bench.proxy import config as proxy_config
    from bench.proxy import sampling as proxy_sampling
    from smoke import adapters, data, tasks
    from smoke import paths as spaths
    from smoke import runner as smoke_runner

    manifest.MLSYS_ROOT = paths.MLSYS
    manifest.OMLX_BASE_URL = base_url
    manifest.OMLX_REPO = paths.OMLX_REPO
    manifest.OMLX_HOME = paths.BASE
    manifest.BENCH_MODEL = served_id
    manifest.BENCH_MODEL_HF_REPO = ""
    manifest.BENCH_MODEL_PATH = str(Path(model["path"]).expanduser()) if model.get("path") else ""
    manifest.SERVED_DIR = scratch  # no .server-<port>.json: records say verified=false (harnesses.md B10)
    manifest.LOCALMELO_REPO = pins["localmelo"]["repo"]
    manifest.CLAUDE_BIN = pins["claude_code"]["bin"]
    manifest.HERMES_BIN = pins["hermes"]["bin"]
    manifest.HF_HUB = paths.HF_HOME / "hub"
    manifest._git = safe_git({pab_root.resolve(): epoch["snapshot"]["personal_agent_bench"]})

    claude_env, claude_home = write_claude_config(vendor, run_root / "config", served_id, pins["claude_code"]["bin"])
    harness.AGENT_ROOT = pins["agent_root"]
    harness.CLAUDE_ENV = claude_env
    harness.CLAUDE_HOME = claude_home
    harness.HERMES_BIN = pins["hermes"]["bin"]
    harness.HERMES_HOME = vendor / epochs.HERMES_HOME_COPY
    harness.MODEL = served_id
    harness.UPSTREAM = proxy_config.service_root(base_url)
    harness._start_proxy = proxy_starter(harness, manifest, model["proxy_sampling"])

    spaths.PAB_ROOT = pab_root
    spaths.GPU_LOCK = paths.GPU_LOCK
    spaths.SMOKE_VENV_BIN = agent_bin
    spaths.SMOKE_PYTHON = agent_bin / "python"
    spaths.WS_ROOT = ws_root
    spaths.RUNS_ROOT = run_root
    spaths.MCP_DIR = smoke_root / "mcp"  # absent: the MCP-only arm is not part of this protocol
    spaths.FS_SERVER_JS = spaths.MCP_DIR / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js"
    adapters.SHIMS = pins["shims"]
    original_agent_env = adapters.agent_env

    def agent_env(env_file: Path) -> dict[str, str]:
        return {**original_agent_env(env_file), "GIT_CEILING_DIRECTORIES": str(ws_root)}

    adapters.agent_env = agent_env
    data._git_state = epochs.repo_state

    class AttemptSettings(smoke_runner.Settings):
        """run_one's run directory is <harness>/<run_name>: a retry or a repeated warm-up needs a new one."""

        def run_name(self, task: Any, n: int) -> str:
            return task.slug if n <= 1 else f"{task.slug}__a{n}"

    rig = SimpleNamespace(
        harness=harness, manifest=manifest, smoke_runner=smoke_runner, tasks=tasks, data=data, adapters=adapters,
        spaths=spaths, proxy_sampling=proxy_sampling, proxy_config=proxy_config, Settings=AttemptSettings,
        pab_root=pab_root, smoke_root=smoke_root, agent_env=agent_env,
    )  # fmt: skip
    problems = selftest(rig, spec, epoch, run_root)
    if problems:
        raise Refused("selftest: " + "; ".join(problems))

    collected = manifest.collect()  # once: system_profiler and the --version probes are slow

    def collect() -> dict[str, Any]:
        return {**collected, "collected_once_per_runner": True}

    manifest.collect = collect
    rig.env_file = adapters.write_env_file(run_root / "agent-env.sh")
    return rig


def resolved_paths(rig: SimpleNamespace) -> dict[str, str]:
    m, h, s = rig.manifest, rig.harness, rig.spaths
    values = {
        "manifest.BENCH_ROOT": m.BENCH_ROOT, "manifest.MLSYS_ROOT": m.MLSYS_ROOT, "manifest.OMLX_REPO": m.OMLX_REPO,
        "manifest.OMLX_HOME": m.OMLX_HOME, "manifest.SERVED_DIR": m.SERVED_DIR, "manifest.LOCALMELO_REPO": m.LOCALMELO_REPO,
        "manifest.CLAUDE_BIN": m.CLAUDE_BIN, "manifest.HERMES_BIN": m.HERMES_BIN, "manifest.HF_HUB": m.HF_HUB,
        "proxy.config.BENCH_ROOT": rig.proxy_config.BENCH_ROOT,
        "harness.AGENT_ROOT": h.AGENT_ROOT, "harness.CLAUDE_ENV": h.CLAUDE_ENV, "harness.CLAUDE_HOME": h.CLAUDE_HOME,
        "harness.HERMES_BIN": h.HERMES_BIN, "harness.HERMES_HOME": h.HERMES_HOME,
        "smoke.SMOKE_ROOT": s.SMOKE_ROOT, "smoke.PAB_ROOT": s.PAB_ROOT, "smoke.GPU_LOCK": s.GPU_LOCK,
        "smoke.SMOKE_VENV_BIN": s.SMOKE_VENV_BIN, "smoke.SMOKE_PYTHON": s.SMOKE_PYTHON, "smoke.WS_ROOT": s.WS_ROOT,
        "smoke.RUNS_ROOT": s.RUNS_ROOT, "smoke.MCP_DIR": s.MCP_DIR, "smoke.DATA_ROOT": s.DATA_ROOT,
        "smoke.POLYGLOT_DIR": s.POLYGLOT_DIR, "smoke.MCPMARK_DIR": s.MCPMARK_DIR, "smoke.MCPMARK_ENVS": s.MCPMARK_ENVS,
        "adapters.SHIMS": rig.adapters.SHIMS,
    }  # fmt: skip
    return {k: str(v) for k, v in values.items()}


def selftest(rig: SimpleNamespace, spec: Mapping[str, Any], epoch: Mapping[str, Any], run_root: Path) -> list[str]:
    """Assert every resolved path and replaced function (DESIGN §6.1). Returns problems."""
    pins = epoch["pins"]
    m, h, s = rig.manifest, rig.harness, rig.spaths
    agent_bin = pins["envs"]["agent_tools"] / ".venv" / "bin"
    data = pins["data"]
    expect = {
        "bench package": (Path(sys.modules["bench"].__file__).resolve().parent, (rig.pab_root / "bench").resolve()),
        "smoke package": (Path(sys.modules["smoke"].__file__).resolve().parent, (rig.smoke_root / "smoke").resolve()),
        "manifest.BENCH_ROOT": (Path(m.BENCH_ROOT).resolve(), rig.pab_root.resolve()),
        "proxy.config.BENCH_ROOT": (Path(rig.proxy_config.BENCH_ROOT).resolve(), rig.pab_root.resolve()),
        "manifest.MLSYS_ROOT": (m.MLSYS_ROOT, paths.MLSYS),
        "manifest.OMLX_BASE_URL": (m.OMLX_BASE_URL, spec["base_url"]),
        "manifest.OMLX_REPO": (m.OMLX_REPO, paths.OMLX_REPO),
        "manifest.OMLX_HOME": (m.OMLX_HOME, paths.BASE),
        "manifest.BENCH_MODEL": (m.BENCH_MODEL, spec["model"]["served_id"]),
        "manifest.LOCALMELO_REPO": (m.LOCALMELO_REPO, pins["localmelo"]["repo"]),
        "manifest.CLAUDE_BIN": (m.CLAUDE_BIN, pins["claude_code"]["bin"]),
        "manifest.HERMES_BIN": (m.HERMES_BIN, pins["hermes"]["bin"]),
        "harness.HERMES_BIN": (h.HERMES_BIN, pins["hermes"]["bin"]),
        "harness.HERMES_HOME": (h.HERMES_HOME, Path(epoch["dir"]) / epochs.HERMES_HOME_COPY),
        "harness.CLAUDE_ENV": (h.CLAUDE_ENV, run_root / "config" / "claude-code.env"),
        "harness.CLAUDE_HOME": (h.CLAUDE_HOME, run_root / "config" / "claude-home"),
        "harness.MODEL": (h.MODEL, spec["model"]["served_id"]),
        "harness.UPSTREAM": (h.UPSTREAM, rig.proxy_config.service_root(spec["base_url"])),
        "harness.HERMES_SHARED": (list(h.HERMES_SHARED), epoch["hermes_lists"]["shared"]),
        "harness.HERMES_SKIP": (list(h.HERMES_SKIP), epoch["hermes_lists"]["skip"]),
        "smoke.SMOKE_ROOT": (Path(s.SMOKE_ROOT).resolve(), rig.smoke_root.resolve()),
        "smoke.PAB_ROOT": (Path(s.PAB_ROOT).resolve(), rig.pab_root.resolve()),
        "smoke.GPU_LOCK": (s.GPU_LOCK, paths.GPU_LOCK),
        "smoke.SMOKE_PYTHON": (s.SMOKE_PYTHON, agent_bin / "python"),
        "smoke.WS_ROOT": (s.WS_ROOT, Path(spec["ws_root"])),
        "smoke.RUNS_ROOT": (s.RUNS_ROOT, run_root),
        "smoke.DATA_ROOT": (s.DATA_ROOT, data["root"]),
        "smoke.POLYGLOT_SHA": (s.POLYGLOT_SHA, data["aider_polyglot"]),
        "smoke.MCPMARK_SHA": (s.MCPMARK_SHA, data["mcpmark"]),
        "smoke.POLYGLOT_DIR": (s.POLYGLOT_DIR, data["aider_polyglot_dir"]),
        "smoke.MCPMARK_DIR": (s.MCPMARK_DIR, data["mcpmark_dir"]),
        "smoke.MCPMARK_ENVS manifest": (s.MCPMARK_ENVS / "manifest.json", data["mcpmark_envs_manifest"]),
        "adapters.SHIMS": (rig.adapters.SHIMS, pins["shims"]),
        "harness._start_proxy": (h._start_proxy.__qualname__, "proxy_starter.<locals>.start_proxy"),
        "manifest._git": (m._git.__qualname__, "safe_git.<locals>._git"),
        "adapters.agent_env": (rig.adapters.agent_env, rig.agent_env),
    }
    found = [f"{name} is {actual!r}, expected {wanted!r}" for name, (actual, wanted) in expect.items() if actual != wanted]

    live_trees = [pins["pab"]["repo"].resolve(), pins["smoke"]["repo"].resolve()]
    for name, value in resolved_paths(rig).items():
        resolved = Path(value).resolve()
        if any(resolved == tree or tree in resolved.parents for tree in live_trees):
            found.append(f"{name} points into a live peer tree: {paths.tilde(value)}")
    for name in ("pip", "pip3", "uv", "uvx"):
        shim = pins["shims"] / name
        if not (shim.is_file() and os.access(shim, os.X_OK)):
            found.append(f"shim {paths.tilde(shim)} is missing or not executable")
    must_exist = [h.CLAUDE_ENV, Path(h.CLAUDE_HOME) / "settings.json", h.HERMES_HOME, h.HERMES_BIN, m.CLAUDE_BIN,
                  Path(m.LOCALMELO_REPO) / ".venv" / "bin" / "melo"]  # fmt: skip
    found += [f"{paths.tilde(p)} is missing" for p in must_exist if not Path(p).exists()]
    probe = subprocess.run([str(s.SMOKE_PYTHON), "-I", "-B", "-c", "import pytest"], capture_output=True)
    if probe.returncode != 0:
        found.append(f"{paths.tilde(s.SMOKE_PYTHON)} cannot import pytest")
    values = rig.proxy_sampling.SamplingPolicy.parse(spec["model"]["proxy_sampling"]).values
    if values != spec["model"]["sampling"]:
        found.append(f"proxy preset {spec['model']['proxy_sampling']} is {values}, the registry says {spec['model']['sampling']}")
    return found


# --- the task plan ---


def task_sets(rig: SimpleNamespace) -> dict[str, list[Any]]:
    """The gate set (12 Aider + 10 MCPMark easy) and the ranking set (the other 22 Python exercises
    of the pinned data), from smoke.tasks; refuses when the snapshot or the data disagree."""
    gate = rig.tasks.default_suite(extended=True)
    gate_aider = [t.name for t in gate if t.kind == "aider"]
    problems = []
    if sorted(gate_aider) != sorted(GATE_AIDER):
        problems.append(f"the snapshot's gate exercises are {gate_aider}")
    if sum(t.kind == "mcpmark" for t in gate) != GATE_MCPMARK_COUNT:
        problems.append("the snapshot does not have 10 MCPMark easy tasks")
    excluded = set(gate_aider) | set(rig.tasks.AIDER_EXCLUDED)
    ranking = [rig.tasks.aider_task(n) for n in rig.data.exercise_names() if n not in excluded]
    if len(ranking) != RANKING_COUNT:
        problems.append(f"the pinned data has {len(ranking)} ranking exercises, expected {RANKING_COUNT}")
    categories = sorted({t.category for t in gate if t.kind == "mcpmark"})
    problems += rig.data.integrity_problems(categories)
    if problems:
        raise Refused("tasks: " + "; ".join(problems))
    return {"gate": gate, "ranking": ranking}


def plan(rig: SimpleNamespace, spec: Mapping[str, Any]) -> list[tuple[str, str, list[Any]]]:
    """Blocks in run order: set-major (the gate is complete before any ranking task), then
    harness-major (Claude Code, Hermes, localmelo), each block with its own warm-up."""
    sets = task_sets(rig)
    only = spec.get("only")
    blocks = []
    for set_name in spec["sets"]:
        chosen = [t for t in sets[set_name] if not only or any(fnmatch.fnmatchcase(t.id, p) for p in only)]
        for name in spec.get("harnesses") or HARNESSES:
            if chosen:
                blocks.append((set_name, name, chosen))
    if not blocks:
        raise Refused(f"no task matches {only}")
    return blocks


# --- one task ---


def _server_log_size(spec: Mapping[str, Any]) -> int | None:
    log = spec.get("server_log")
    try:
        return Path(log).stat().st_size if log else None
    except OSError:
        return None


def _template_errors(spec: Mapping[str, Any], offset: int | None) -> int:
    if offset is None:
        return 0
    with Path(spec["server_log"]).open("rb") as f:
        f.seek(offset)
        text = f.read().decode(errors="replace")
    return sum(1 for line in text.splitlines() if any(p in line for p in TEMPLATE_ERROR_PATTERNS))


def run_task(
    ctx: SimpleNamespace, set_name: str, harness_name: str, task: Any
) -> dict[str, Any]:
    """smoke.runner.run_one for one (harness, task) under the memory watchdog, then the call checks, the
    server check and the verdict; the record is saved and exported before this returns.

    The row's `flags` stay null for the orchestrator's disturbance flags unless the watchdog killed
    something: then they start as {"memory_kills": [...]}, and the orchestrator adds its flags to them."""
    spec, rig, state = ctx.spec, ctx.rig, ctx.state
    n = state.next_attempt(harness_name, task.id)
    settings = rig.Settings(
        harnesses=[harness_name], tasks=[task], warmup=False, timeout_s=spec["wall_s"], max_steps=spec["max_steps"],
        max_turns=spec["max_turns"], thinking=spec["thinking"], localmelo_args=localmelo_args(spec["wall_s"]),
    )  # fmt: skip
    while (ctx.run_root / harness_name / settings.run_name(task, n)).exists():
        n += 1  # a run that crashed before its record was saved
    progress.emit("TASK-START", set=set_name, harness=harness_name, task=task.id, attempt=n)
    battery = ctx.boundaries.battery()
    log_offset = _server_log_size(spec)
    t_start = time.time()
    with memwatch.Watchdog(os.getpid(), *ctx.memory_limits) as watchdog:  # the agents' code is bounded (§6.1)
        record = rig.smoke_runner.run_one(rig.harness, harness_name, task, settings, ctx.run_root, ctx.ws_root, rig.env_file, n)
    t_end = time.time()
    run_dir = ctx.run_root / record["run_dir"]
    calls_records, unreadable = checks.load_calls(run_dir / "calls.jsonl")
    calls = checks.check_calls(
        calls_records, served_id=spec["model"]["served_id"], sampling=spec["model"]["sampling"],
        thinking=spec["thinking"], unreadable=unreadable,
    )  # fmt: skip
    lost = server_problem(spec["server_pid"], spec["base_url"])
    result = checks.outcome(
        record, calls, max_steps=spec["max_steps"], server_lost=lost is not None,
        template_errors=_template_errors(spec, log_offset),
    )  # fmt: skip
    full = {
        **record, **result,
        "set": set_name, "attempt": n, "launch": spec.get("launch", 1), "calls": calls,
        "turns": checks.native_turns(harness_name, record.get("harness_report") or {}),
        "t_start": round(t_start, 3), "t_end": round(t_end, 3), "battery_start": battery,
        "server_problem": lost, "flags": watchdog.row_flags(),
    }  # fmt: skip
    state.records.append(full)
    state.save()
    write_results(ctx, finished=False)
    export.export_run(run_dir, ctx.results_dir / record["run_dir"])
    progress.emit(
        "TASK-DONE", set=set_name, harness=harness_name, task=task.id, attempt=n, verdict=full["verdict"],
        solved=full["solved"], wall_s=full.get("wall_s"), n_llm=full.get("n_llm"), memory_kills=len(watchdog.kills),
    )  # fmt: skip
    return full


# --- results, yields, hygiene ---


def set_progress(ctx: SimpleNamespace) -> tuple[list[str], dict[str, bool]]:
    """The sets this run has started over all its invocations (the orchestrator runs the gate set
    first and the ranking set in a second invocation, DESIGN §6.3), and which of them are complete:
    every planned pair has a valid verdict or was excluded after its re-run. Kept in state.meta, so
    the gate set stays complete in results.json while the ranking set runs."""
    meta = ctx.state.meta
    complete = dict(meta.get("complete_sets") or {})
    if ctx.blocks is not None:
        for set_name in ctx.spec["sets"]:
            complete[set_name] = all(ctx.state.done(s, h, t.id) for s, h, ts in ctx.blocks if s == set_name for t in ts)
    sets = [x for x in SETS if x in set(meta.get("sets") or []) | set(ctx.spec["sets"])]
    meta["sets"], meta["complete_sets"] = sets, complete
    return sets, complete


def write_results(ctx: SimpleNamespace, *, finished: bool, yielded: dict[str, Any] | None = None) -> None:
    spec, state = ctx.spec, ctx.state
    rows, superseded, warmups = state.partition()
    sets, complete = set_progress(ctx)
    anomalies: collections.Counter[str] = collections.Counter()
    for row in rows:
        anomalies.update(row["anomalies"])
    doc = {
        "schema_version": 1,
        "protocol_version": paths.PROTOCOL_VERSION,
        "model": spec["model"].get("key"),
        "served_id": spec["model"]["served_id"],
        "proxy_sampling": spec["model"]["proxy_sampling"],
        "sampling": spec["model"]["sampling"],
        "thinking": spec["thinking"],
        "epoch": ctx.epoch["id"],
        "snapshot": ctx.epoch["snapshot"],
        "sets": sets,
        "complete_sets": complete,
        "harnesses": list(spec.get("harnesses") or HARNESSES),
        "settings": {
            "wall_s": spec["wall_s"], "max_steps": spec["max_steps"], "max_turns": spec["max_turns"],
            "localmelo_args": localmelo_args(spec["wall_s"]), "context_window": CONTEXT_WINDOW,
            "solved_within_s": checks.SOLVED_WITHIN_S, "order": "set-major, then harness-major",
        },  # fmt: skip
        "ws_root": paths.tilde(ctx.ws_root),
        "env_names": ctx.env_names,
        "quiet_source": ctx.boundaries.source,
        "rows": [export.clean_row(r) for r in rows],
        "superseded": [export.clean_row(r) for r in superseded],
        "warmups": [export.clean_row(r) for r in warmups],
        "anomalies": dict(sorted(anomalies.items())),
        "hygiene": state.meta.get("hygiene", []),
        "updated_at": dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "finished": finished,
        "yield": yielded,
    }
    export.write_results(ctx.results_path, doc)


def _yield(ctx: SimpleNamespace, reason: str, detail: str, code: int = EXIT_YIELD) -> int:
    info = {"reason": reason, "detail": detail, "at": time.time(), "exit_code": code}
    (ctx.run_root / YIELD_FILE).write_text(json.dumps(info, indent=2) + "\n")
    ctx.state.save()
    write_results(ctx, finished=False, yielded=info)
    with contextlib.suppress(OSError, ValueError):   # a dead stderr must not turn a yield into exit 1
        print(f"capability runner: yield ({reason}: {detail})", file=sys.stderr, flush=True)
    return code


def _audit(ctx: SimpleNamespace, operation: str, before: dict[str, Any]) -> None:
    after = epochs.index_stamps(ctx.audit_repos)
    changed = [paths.tilde(p) for p in before if before[p] != after.get(p)]
    if changed:
        ctx.state.meta["hygiene"].append({"operation": operation, "changed": changed, "at": time.time()})


def before_task(ctx: SimpleNamespace, kind: str) -> int | None:
    """The checks at every boundary (DESIGN §6.1): server, battery, hold, live pins. An exit code
    when the runner must stop here, else None."""
    lost = server_problem(ctx.spec["server_pid"], ctx.spec["base_url"])
    if lost:
        return _yield(ctx, "server_lost", lost, EXIT_SERVER_LOST)
    decision = ctx.boundaries.decision(kind)
    if decision != "go":
        reason = "battery" if decision == "pause" else "battery-stop"
        return _yield(ctx, reason, f"{ctx.boundaries.battery()} % at a {kind} boundary")
    if ctx.boundaries.hold():
        return _yield(ctx, "hold", "hold file present")
    stamps = epochs.index_stamps(ctx.audit_repos)
    drift = epochs.verify_live(ctx.epoch)
    _audit(ctx, "verify_live", stamps)
    if drift:
        return _yield(ctx, "drift", "; ".join(drift))
    return None


def run_blocks(ctx: SimpleNamespace) -> int:
    for set_name, harness_name, block_tasks in ctx.blocks:
        pending = [t for t in block_tasks if not ctx.state.done(set_name, harness_name, t.id)]
        if not pending:
            continue
        code = before_task(ctx, "harness-block")
        if code is not None:
            return code
        warm = run_task(ctx, set_name, harness_name, ctx.rig.tasks.WARMUP)
        if warm["verdict"] == "server_lost":
            return _yield(ctx, "server_lost", warm["server_problem"] or "", EXIT_SERVER_LOST)
        if warm["verdict"] == "invalid":
            raise RuntimeError(f"{harness_name} warm-up is invalid ({warm['invalid_reasons']}): {warm['calls']['examples']}")
        for task in pending:
            while not ctx.state.done(set_name, harness_name, task.id):
                code = before_task(ctx, "task")
                if code is not None:
                    return code
                record = run_task(ctx, set_name, harness_name, task)
                if record["verdict"] == "server_lost":
                    return _yield(ctx, "server_lost", record["server_problem"] or "", EXIT_SERVER_LOST)
    return EXIT_DONE


# --- entry point ---


REQUIRED = ("model", "base_url", "server_pid", "sets", "run_root", "results", "ws_root", "epoch", "vendor",
            "wall_s", "max_steps", "max_turns", "thinking", "resume")  # fmt: skip


def spec_problems(spec: Mapping[str, Any]) -> list[str]:
    found = [f"spec has no {key!r}" for key in REQUIRED if key not in spec]
    if found:
        return found
    model = spec["model"]
    for key in ("served_id", "sampling", "proxy_sampling"):
        if key not in model:
            found.append(f"spec model has no {key!r}")
    if not SERVED_ID.match(str(model.get("served_id", ""))):
        found.append(f"served_id {model.get('served_id')!r} is not a plain model id")
    if not spec["sets"] or any(s not in SETS for s in spec["sets"]):
        found.append(f"sets must be a non-empty subset of {SETS}")
    if any(h not in HARNESSES for h in spec.get("harnesses") or ()):
        found.append(f"harnesses must be a subset of {HARNESSES}")
    if spec["thinking"] not in ("off", "on"):
        found.append("thinking must be off or on")
    if not float(spec["wall_s"]) > LOCALMELO_MARGIN_S:
        found.append(f"wall_s must exceed {LOCALMELO_MARGIN_S}")
    if int(spec["max_steps"]) < 1 or int(spec["max_turns"]) < 1:
        found.append("max_steps and max_turns must be positive")
    if Path(spec["vendor"]).name != spec["epoch"]:
        found.append("vendor must be vendor/<epoch>")
    return found


def _refuse(problems: list[str]) -> int:
    for problem in problems:
        with contextlib.suppress(OSError, ValueError):
            print(f"capability runner: preflight: {problem}", file=sys.stderr, flush=True)
    progress.emit("ERROR", stage="preflight", problems=len(problems))
    return EXIT_PREFLIGHT


class _Terminated(BaseException):
    """SIGTERM or SIGINT: stop the current harness, save, exit."""


def _on_signal(signum: int, _frame: Any) -> None:
    raise _Terminated(signum)


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; nothing here writes bytecode
    parser = argparse.ArgumentParser(prog="python -I -B -m modeleval.capability.runner")
    parser.add_argument("--spec", required=True, type=Path)
    spec = json.loads(parser.parse_args(argv).spec.read_text())
    problems = spec_problems(spec)
    if problems:
        return _refuse(problems)
    env_names = sorted(os.environ)
    bad = forbidden_names(os.environ)
    if bad:
        return _refuse([f"the environment carries {', '.join(bad)}; start the runner with allowed_env()"])
    try:
        memory_limits = memwatch.limits(os.environ)
    except ValueError as exc:
        return _refuse([str(exc)])

    run_root = Path(spec["run_root"])
    scratch = run_root / "scratch"
    for name in ("claude-config", "hermes-home"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    # After the assertion: these two are the runner's own, for the manifest's version probes
    os.environ.update(
        {"CLAUDE_CONFIG_DIR": str(scratch / "claude-config"), "HERMES_HOME": str(scratch / "hermes-home"), **FORCED_ENV}
    )
    (run_root / YIELD_FILE).unlink(missing_ok=True)
    state = State(run_root / STATE_FILE)
    if state.records and not spec["resume"]:
        return _refuse([f"{paths.tilde(run_root)} already has runs; restart with resume"])
    if state.records and state.meta.get("epoch", spec["epoch"]) != spec["epoch"]:   # one epoch per run (§7)
        return _refuse([f"{paths.tilde(run_root)} has runs from harness epoch {state.meta['epoch']}, the spec names "
                        f"{spec['epoch']}; start a new run"])
    state.meta["epoch"] = spec["epoch"]
    ws_root = Path(spec["ws_root"])
    try:
        epoch = epochs.load_epoch(spec["epoch"], Path(spec["vendor"]).parent)
    except epochs.EpochError as exc:
        return _refuse([str(exc)])
    problems = [f"vendor snapshot: {p}" for p in epochs.snapshot_problems(epoch)]
    if epoch["pins"]["ws_root"] != ws_root:
        problems.append(f"ws_root {paths.tilde(ws_root)} is not the epoch's {paths.tilde(epoch['pins']['ws_root'])}")
    problems += ws_root_problems(ws_root)
    if problems:
        return _refuse(problems)

    pins = epoch["pins"]
    audit_repos = [pins["pab"]["repo"], pins["smoke"]["repo"], pins["hermes"]["repo"], pins["localmelo"]["repo"],
                   pins["data"]["aider_polyglot_dir"], pins["data"]["mcpmark_dir"], paths.OMLX_REPO]  # fmt: skip
    ctx = SimpleNamespace(
        spec=spec, epoch=epoch, state=state, run_root=run_root, ws_root=ws_root, env_names=env_names,
        results_path=Path(spec["results"]), results_dir=Path(spec["results"]).parent,
        boundaries=Boundaries(spec.get("selftest") or {}), audit_repos=audit_repos, rig=None, blocks=None,
        memory_limits=memory_limits,
    )  # fmt: skip
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        with workspace_lock(ws_root, owner=paths.tilde(run_root)):
            drift = epochs.verify_live(epoch)
            if drift:
                return _yield(ctx, "drift", "; ".join(drift))
            stamps = epochs.index_stamps(audit_repos)
            ctx.rig = configure(spec, epoch, run_root)
            ctx.blocks = plan(ctx.rig, spec)
            _audit(ctx, "configure", stamps)
            (run_root / SELFTEST_FILE).write_text(json.dumps(resolved_paths(ctx.rig), indent=2) + "\n")
            code = run_blocks(ctx)
            if code == EXIT_DONE:
                state.save()
                write_results(ctx, finished=True)
            return code
    except Refused as exc:
        return _refuse([str(exc)])
    except _Terminated as exc:
        with contextlib.suppress(OSError, ValueError):
            print(f"capability runner: stopped by signal {exc.args[0]}", file=sys.stderr, flush=True)
        state.save()
        return EXIT_ERROR
    except Exception as exc:  # recorded, so a restart with resume can continue
        progress.emit("ERROR", stage="capability", error=f"{type(exc).__name__}: {exc}"[:300])
        state.save()
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        kill_children()


if __name__ == "__main__":
    sys.exit(main())
