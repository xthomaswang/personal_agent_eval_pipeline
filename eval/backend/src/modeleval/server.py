"""The oMLX server of one run: settings, launch, readiness, log checks, stop, memory (DESIGN.md §2, §5.2).

The pipeline owns its own oMLX base path (`paths.BASE`, `.modeleval-home`): settings.json and
model_settings.json are rewritten from the protocol values before every launch, with the server
stopped, so nothing depends on the old campaign's `.bench-home` or on what a previous launch
persisted. The launch reproduces the old campaign's `serve.sh build_omlx_oq` exactly (environment,
`omlx serve --base-path BASE --headless`, cwd in the oMLX checkout); the differences are the base
path and the per-run SSD cache directory.

Every child process group goes into the run's pidfile, so that a crash, a signal or `--resume`
can find and stop whatever the orchestrator started.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from modeleval import paths
from modeleval.registry import Model

CONTEXT_WINDOW = 131072
WARMUP_TOKENS = 8
SECRET_ENV = re.compile(r"^(CLAUDE|ANTHROPIC|OPENAI)|(^|_)(TOKEN|SECRET|PASSWORD)(_|$)|_KEY$|^GH_|^GITHUB_")

# The structure of serving/omlx_0.7/.bench-home/settings.json (oMLX 0.7.1rc1, read 2026-10-26),
# without its auth secret and its server_aliases (hostname, LAN address). Values that DESIGN §2
# fixes are set by protocol_settings(); everything else keeps the oMLX default it had there.
SETTINGS_TEMPLATE: dict = {
    "version": "1.0",
    "server": {
        "host": "127.0.0.1", "port": 8703, "log_level": "info", "cors_origins": ["*"],
        "server_aliases": ["localhost", "127.0.0.1"], "sse_keepalive_mode": "chunk",
        "auto_start_on_launch": True, "burst_decode_mode": "balanced", "preserve_mid_system_cache": True,
        "distributed_inference_enabled": False, "max_audio_upload_size": "100MB",
        "max_image_upload_size": "50MB", "max_request_body_size": "512MB", "max_image_side_length": 2048,
        "gpu_keep_warm_interval": 0.5,
    },
    "model": {"model_dirs": [], "model_dir": None, "model_fallback": False, "hide_helper_models": False},
    "memory": {
        "prefill_memory_guard": True, "memory_guard_tier": "balanced", "memory_guard_custom_ceiling_gb": 0.0,
        "soft_threshold": 0.85, "hard_threshold": 0.95, "prefill_safe_zone_ratio": 0.8,
        "prefill_min_chunk_tokens": 32,
    },
    "scheduler": {
        "max_concurrent_requests": 8, "embedding_batch_size": 32, "chunked_prefill": False,
        "prefill_priority": "context", "decode_fairness": True,
    },
    "cache": {
        "enabled": True, "hot_cache_only": False, "gdn_ssd_split_enabled": True, "gdn_snapshot_storage": "auto",
        "gdn_ssd_pending_max_size": "512MB", "gdn_sidecar_precision": "fp32", "ssd_cache_dir": None,
        "ssd_cache_max_size": "auto", "hot_cache_max_size": "0", "hot_cache_write_through": False,
        "ane_compile_cache": False, "initial_cache_blocks": 256,
    },
    "auth": {
        "api_key": None, "secret_key": None, "skip_api_key_verification": False,
        "allow_unauthenticated_inference": False, "sub_keys": [],
    },
    "mcp": {"config_path": None, "expose_tools": True},
    "huggingface": {"endpoint": "", "hf_cache_enabled": False},
    "modelscope": {"endpoint": ""},
    "network": {"http_proxy": "", "https_proxy": "", "no_proxy": "", "ca_bundle": ""},
    "sampling": {
        "max_context_window": 32768, "max_context_window_policy": None, "max_tokens": 32768,
        "temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0,
    },
    "logging": {"log_dir": None, "retention_days": 7},
    "claude_code": {"mode": "cloud", "opus_model": None, "sonnet_model": None, "haiku_model": None},
    "integrations": {
        "codex_model": None, "opencode_model": None, "openclaw_model": None, "hermes_model": None,
        "pi_model": None, "vibe_model": None, "copilot_model": None, "dsh_model": None,
        "openclaw_tools_profile": "coding", "markitdown_enabled": True, "markitdown_expose_model": False,
        "markitdown_max_file_size_mb": 25, "markitdown_max_files_per_request": 5,
        "markitdown_pdf_processing_engine": "markitdown", "web_search_provider": "ddgs",
        "web_search_brave_api_key": "", "web_search_searxng_url": "", "web_search_ddgs_backends": "",
        "web_search_max_results": 3, "web_search_content_mode": "snippet", "web_search_content_truncate": True,
        "web_search_content_max_chars": 20000, "splash_path": None, "splash_kv_format": "int8",
        "splash_max_memory_gb": None, "splash_ssd_cache_gb": None, "splash_ane_prefill": True,
    },
    "ui": {"language": "en", "dashboard_layout": None},
    "usage": {"usage_history": True},
    "idle_timeout": {"idle_timeout_seconds": None},
}

# DESIGN §2, the fixed setting of protocol version 1: the user's production backend
# (~/.omlx-0.7/settings.json) plus a fixed SSD-cache cap. Dotted keys into settings.json.
PROTOCOL_V1_SETTINGS: dict[str, object] = {
    "server.host": "127.0.0.1",
    "server.burst_decode_mode": "balanced",
    "memory.prefill_memory_guard": True,
    "memory.memory_guard_tier": "balanced",
    "memory.soft_threshold": 0.85,
    "memory.hard_threshold": 0.95,
    "scheduler.max_concurrent_requests": 8,
    "scheduler.chunked_prefill": True,
    "scheduler.prefill_priority": "context",
    "scheduler.decode_fairness": True,
    "cache.enabled": True,
    "cache.hot_cache_only": False,
    "cache.hot_cache_max_size": "20GB",
    "cache.ssd_cache_max_size": "128GB",
    "sampling.max_tokens": 32768,
    "sampling.temperature": 1.0,
    "sampling.top_p": 0.95,
    "sampling.top_k": 0,
    "huggingface.hf_cache_enabled": False,
    "idle_timeout.idle_timeout_seconds": None,
}
PROTOCOL_SETTINGS = {
    1: PROTOCOL_V1_SETTINGS,
    2: {**PROTOCOL_V1_SETTINGS, "cache.hot_cache_max_size": "12GB", "memory.memory_guard_tier": "custom",
        "memory.memory_guard_custom_ceiling_gb": 90.0},
}


class ServerError(RuntimeError):
    """The server could not be started as the protocol requires; nothing is measured on it."""


def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_port_free(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while port_listening(port):
        if time.time() > deadline:
            return False
        time.sleep(0.5)
    return True


def _set(d: dict, dotted: str, value: object) -> None:
    *parents, last = dotted.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[last] = value


def protocol_settings(base: Path, port: int, ssd_cache_dir: Path, secret_key: str | None = None,
                      *, version: int | None = None) -> dict:
    """The complete settings.json of a launch: template + protocol values + this run's paths."""
    d = copy.deepcopy(SETTINGS_TEMPLATE)
    for k, v in PROTOCOL_SETTINGS[paths.PROTOCOL_VERSION if version is None else version].items():
        _set(d, k, v)
    empty = str(base / "empty-models")   # models come from OMLX_MODEL_DIR, as in serve.sh
    _set(d, "server.port", port)
    _set(d, "model.model_dirs", [empty])
    _set(d, "model.model_dir", empty)
    _set(d, "cache.ssd_cache_dir", str(ssd_cache_dir))
    # A missing secret makes oMLX generate one and save() the whole file with its env overrides on
    # first start; a fixed one keeps settings.json exactly as written. It never leaves BASE.
    _set(d, "auth.secret_key", secret_key or secrets.token_hex(32))
    return d


def public_settings(settings: dict) -> dict:
    """settings.json as recorded in a manifest: no auth section, no aliases, paths with ~."""
    d = copy.deepcopy(settings)
    d.pop("auth", None)
    d.get("server", {}).pop("server_aliases", None)
    return _tilde_tree(d)


def model_settings(models: list[Model]) -> dict:
    """model_settings.json: 131072 context for every model, mtp_enabled per spec, never max_tokens."""
    out = {}
    for m in models:
        entry: dict = {"max_context_window": CONTEXT_WINDOW}
        if m.spec == "mtp":
            entry["mtp_enabled"] = True
        entry.update(m.model_settings)
        if "max_tokens" in entry:
            raise ServerError(f"{m.key}: a per-model max_tokens is not allowed (DESIGN §2)")
        out[m.served_id] = entry
    return {"version": 1, "models": out}


def _tilde_tree(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _tilde_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tilde_tree(v) for v in obj]
    if isinstance(obj, str) and str(paths.HOME) in obj:
        return obj.replace(str(paths.HOME), "~")
    return obj


def _write_json(path: Path, data: dict, mode: int = 0o600) -> None:
    """Atomic write; refuses a symlink or a hard-linked file (it would edit someone else's copy)."""
    if path.is_symlink() or (path.exists() and path.stat().st_nlink > 1):
        raise ServerError(f"{paths.tilde(path)} is a link; refusing to write through it")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def clean_env(env: dict[str, str]) -> dict[str, str]:
    """The environment minus anything that looks like a credential: the server needs none."""
    return {k: v for k, v in env.items() if not SECRET_ENV.search(k)}


# ---------------------------------------------------------------- pidfile


class Pidfile:
    """runs/<model>/<run-id>/pidfile: one line per child process group: `pgid<TAB>role<TAB>marker`.

    marker is a string the group leader's command line contains (its executable or script), so a
    stale entry is only killed when the PID still belongs to what we started.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def entries(self) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0].isdigit():
                out.append({"pgid": int(parts[0]), "role": parts[1], "marker": parts[2]})
        return out

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(f"{e['pgid']}\t{e['role']}\t{e['marker']}\n" for e in entries), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, pgid: int, role: str, marker: str) -> None:
        with self._lock:
            self._write([e for e in self.entries() if e["pgid"] != pgid] + [{"pgid": pgid, "role": role, "marker": marker}])

    def remove(self, pgid: int) -> None:
        with self._lock:
            self._write([e for e in self.entries() if e["pgid"] != pgid])

    def kill_all(self, timeout: float = 15.0) -> list[str]:
        """Stop every recorded group that still runs what we started (SIGINT for oMLX, then SIGTERM,
        then SIGKILL). Returns one line per group for the log."""
        done = []
        for e in self.entries():
            pgid = e["pgid"]
            if not group_alive(pgid):
                done.append(f"{e['role']} {pgid}: gone")
                self.remove(pgid)
                continue
            if not owned(e):
                done.append(f"{e['role']} {pgid}: PID reused by another program; left alone")
                self.remove(pgid)
                continue
            first = signal.SIGINT if e["role"] == "omlx" else signal.SIGTERM
            ok = kill_group(pgid, first, timeout)
            done.append(f"{e['role']} {pgid}: {'stopped' if ok else 'STILL RUNNING'}")
            if ok:
                self.remove(pgid)
        return done


def owned(entry: dict) -> bool:
    """Whether a live pidfile entry is still what we started. oMLX retitles itself (`omlx-server`),
    so for it a listener on our port or a cwd in the oMLX checkout also counts (as lib.sh pid_owned)."""
    pgid = entry["pgid"]
    if not entry["marker"] or entry["marker"] in pid_command(pgid):
        return True
    if entry["role"] != "omlx":
        return False
    if pgid in listen_pids(paths.PORT):
        return True
    return pid_cwd(pgid) == str(paths.OMLX_REPO)


def listen_pids(port: int) -> set[int]:
    out = _sh("lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp")
    return {int(x[1:]) for x in out.splitlines() if x.startswith("p") and x[1:].isdigit()}


def pid_cwd(pid: int) -> str:
    out = _sh("lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn")
    return next((x[1:] for x in out.splitlines() if x.startswith("n")), "")


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pid_command(pid: int) -> str:
    try:
        return subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def kill_group(pgid: int, first: int = signal.SIGTERM, timeout: float = 15.0) -> bool:
    """Signal a process group, wait, then SIGKILL it. True when nothing of it is left."""
    for sig, wait_s in ((first, timeout), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        deadline = time.time() + wait_s
        while time.time() < deadline:
            _reap(pgid)
            if not group_alive(pgid):
                return True
            time.sleep(0.2)
    return not group_alive(pgid)


def _reap(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


# ---------------------------------------------------------------- server log


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOG_FIELDS = {
    "version": r"Version: (\S+)",
    "base_path": r"^Base path: (.+)$",
    "model_dirs": r"^Model directories: (.+)$",
    "memory_guard": r"^Memory guard: (.+)$",
    "mode": r"^Mode: (.+)$",
    "hot_cache": r"^Hot cache: (\S+) \(in-memory\)",
    "default_max_tokens": r"Default max tokens: (\d+)",
    "memory_enforcer": r"Process memory enforcer started \((.+)\)",
    "metal_cap": r"(Metal cap \(.+)$",
    "speculative": r"Speculative backend selected for .*?: (.+)$",
    "ple_mode": r"PLE mode for .*?: (\w+)",
    "tool_parser": r"tool calling enabled: parser=(\S+)",
}


def parse_server_log(text: str) -> dict:
    """The effective settings a launch reports in its log (DESIGN §2: recorded in the manifest).

    Scheduler values are not printed by oMLX at startup; the manifest records the written
    settings.json next to these.
    """
    text = _ANSI.sub("", text)
    out: dict = {}
    for key, rx in _LOG_FIELDS.items():
        m = re.search(rx, text, re.M)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"^paged SSD cache: (.+) \(max: (.+)\)$", text, re.M)
    if m:
        out["ssd_cache"] = {"dir": m.group(1), "max": m.group(2)}
    m = re.search(r"PagedSSDCacheManager initialized: (.+)$", text, re.M)
    if m:
        out["ssd_cache_manager"] = dict(re.findall(r"(\w+)=([^,]+)", m.group(1)))
    m = re.search(r"Loaded model: (\S+) \((.+)\)$", text, re.M)
    if m:
        out["loaded_model"] = {"id": m.group(1), "detail": m.group(2)}
    out["ple_forced_to_ssd"] = "PLE forced to SSD" in text
    return _tilde_tree(out)


def check_expectations(text: str, expect: list[str]) -> tuple[dict, list[str]]:
    """({regex: first matching line}, [regexes with no match])."""
    lines = _ANSI.sub("", text).splitlines()
    found, missing = {}, []
    for rx in expect:
        hit = next((ln for ln in lines if re.search(rx, ln)), None)
        if hit is None:
            missing.append(rx)
        else:
            found[rx] = str(_tilde_tree(hit.split(" - INFO - ")[-1][:200]))
    return found, missing


# ---------------------------------------------------------------- the server


class Server:
    """One oMLX launch for one model (docs/CONTRACTS.md)."""

    READY_TIMEOUT = 300.0
    WARMUP_TIMEOUT = 1800.0     # the first request loads the model (Flash-Next: ~70 GB)
    STOP_TIMEOUT = 60.0

    def __init__(self, model: Model, run_dir: Path, launch_no: int, *, models: list[Model] | None = None,
                 pidfile: Pidfile | None = None, caffeinate: bool = True):
        self.model = model
        self.models = models or [model]
        self.run_dir = Path(run_dir)
        self.launch_no = launch_no
        self.base, self.port, self.exe = paths.BASE, paths.PORT, paths.OMLX_EXE
        self.models_dir, self.workdir = paths.MODELS_DIR, paths.OMLX_REPO
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.log_path = self.run_dir / "server.log"
        self.ssd_cache = self.run_dir / "ssd-cache"
        self.pidfile = pidfile or Pidfile(self.run_dir / "pidfile")
        self.caffeinate = caffeinate
        self.pid = 0
        self.proc: subprocess.Popen | None = None
        self._caff: subprocess.Popen | None = None
        self.log_offset = 0
        self.effective_settings: dict = {}
        self.written_settings: dict = {}
        self.started_at = self.ready_at = self.stopped_at = 0.0

    # -- settings

    def write_settings(self) -> None:
        """BASE/settings.json + BASE/model_settings.json from DESIGN §2 and the registry."""
        if port_listening(self.port):
            raise ServerError(f"port {self.port} is listening; settings are only written with the server stopped")
        if self.base.is_symlink():
            raise ServerError(f"{paths.tilde(self.base)} is a symlink; the base path must be the pipeline's own")
        for d in (self.base, self.base / "home", self.base / "empty-models", self.ssd_cache):
            d.mkdir(parents=True, exist_ok=True)
        old_secret = None
        sf = self.base / "settings.json"
        if sf.exists():
            try:
                old_secret = json.loads(sf.read_text(encoding="utf-8")).get("auth", {}).get("secret_key")
            except (OSError, ValueError):
                old_secret = None
        settings = protocol_settings(self.base, self.port, self.ssd_cache, old_secret)
        _write_json(sf, settings)
        _write_json(self.base / "model_settings.json", model_settings(self.models), mode=0o644)
        self.written_settings = public_settings(settings)

    def env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """serve.sh build_omlx_oq's environment (HOME isolated so ~/.omlx is never touched)."""
        env = clean_env(dict(os.environ if base_env is None else base_env))
        env.update({
            "HF_HUB_OFFLINE": "1", "HF_HOME": str(paths.HF_HOME), "HOME": str(self.base / "home"),
            "OMLX_BASE_PATH": str(self.base), "OMLX_HOST": "127.0.0.1", "OMLX_PORT": str(self.port),
            "OMLX_MODEL_DIR": str(self.models_dir), "OMLX_HF_CACHE_ENABLED": "false", "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        return env

    def argv(self) -> list[str]:
        return [str(self.exe), "serve", "--base-path", str(self.base), "--headless"]

    # -- start / stop

    def start(self) -> None:
        """Launch, wait for /v1/models, warm up with 8 tokens, then check the log; stop on any failure."""
        if not Path(self.exe).exists():
            raise ServerError(f"missing executable {paths.tilde(self.exe)}")
        self.write_settings()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"# modeleval launch {self.launch_no} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}: "
                      f"{paths.tilde(self.exe)} serve --base-path {paths.tilde(self.base)} --headless\n")
            log.flush()
            self.proc = subprocess.Popen(self.argv(), cwd=self.workdir, env=self.env(), stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.pid = self.proc.pid
        self.started_at = time.time()
        self.pidfile.add(self.pid, "omlx", str(self.exe))
        if self.caffeinate:   # no idle or system sleep while the server lives (serve.sh does the same)
            try:
                self._caff = subprocess.Popen(["caffeinate", "-ims", "-w", str(self.pid)], stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL, start_new_session=True)
                self.pidfile.add(self._caff.pid, "caffeinate", "caffeinate")
            except OSError:
                self._caff = None
        try:
            self._wait_ready()
            self.warm_up()
            self._check_log()
        except BaseException:
            self.stop()
            raise
        self.ready_at = time.time()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def health(self, timeout: float = 10.0) -> str | None:
        """None when the process lives and GET /v1/models answers 200; otherwise why not. The speed
        leg's check between units, so a lost server is relaunched instead of failing the units."""
        if not self.alive():
            return f"oMLX exited (code {self.proc.returncode if self.proc is not None else None})"
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=timeout)
        except httpx.HTTPError as e:
            return f"GET /v1/models failed: {type(e).__name__}"
        return None if r.status_code == 200 else f"GET /v1/models answered HTTP {r.status_code}"

    def _wait_ready(self) -> None:
        deadline = time.time() + self.READY_TIMEOUT
        while time.time() < deadline:
            if not self.alive():
                raise ServerError(f"oMLX exited with code {self.proc.returncode} before it was ready; "
                                  f"see {paths.tilde(self.log_path)}")
            try:
                r = httpx.get(f"{self.base_url}/models", timeout=2.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise ServerError(f"oMLX not ready after {self.READY_TIMEOUT:.0f} s")

    def warm_up(self) -> dict:
        """One 8-token request with the model's sampling and thinking off; it also loads the model."""
        s = self.model.sampling
        body = {"model": self.model.served_id, "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": WARMUP_TOKENS, "temperature": s["temperature"], "top_p": s["top_p"],
                "top_k": s["top_k"], "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.WARMUP_TIMEOUT)
        except httpx.HTTPError as e:
            raise ServerError(f"warm-up failed: {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise ServerError(f"warm-up failed: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def log_text(self) -> str:
        """This launch's part of server.log."""
        try:
            with self.log_path.open("rb") as f:
                f.seek(self.log_offset)
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def _check_log(self) -> None:
        text = self.log_text()
        eff = parse_server_log(text)
        found, missing = check_expectations(text, self.model.server_log_expect)
        eff["log_expect"] = found
        self.effective_settings = eff
        problems = [f"no server-log line matches {rx!r}" for rx in missing]
        mgr = eff.get("ssd_cache_manager", {})
        if mgr.get("auto_size", "False").strip() != "False":
            problems.append(f"SSD cache size is automatic ({mgr})")
        want_dir = paths.tilde(self.ssd_cache.resolve())
        if mgr.get("dir") and mgr["dir"].strip() != want_dir and mgr["dir"].strip() != paths.tilde(self.ssd_cache):
            problems.append(f"SSD cache dir is {mgr['dir']}, expected {want_dir}")
        if re.search(r"Configuration error|Settings file .* is corrupt", text):
            problems.append("oMLX reported a configuration error")
        if problems:
            raise ServerError("; ".join(problems))

    def stop(self) -> None:
        """SIGINT (oMLX's own shutdown), wait, SIGKILL; then wait for the port to be free."""
        had = self.proc is not None
        if self.proc is not None:
            if self.proc.poll() is None:
                kill_group(self.proc.pid, signal.SIGINT, self.STOP_TIMEOUT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self.pidfile.remove(self.proc.pid)
        if self._caff is not None:
            kill_group(self._caff.pid, signal.SIGTERM, 5)
            try:
                self._caff.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            self.pidfile.remove(self._caff.pid)
            self._caff = None
        if had:
            self.stopped_at = time.time()
        self.proc = None
        if had and not wait_port_free(self.port, 30):
            raise ServerError(f"port {self.port} still listening after stop")

    def record(self) -> dict:
        """This launch for manifest.json."""
        return {"launch": self.launch_no, "pid": self.pid, "started": _iso(self.started_at),
                "ready": _iso(self.ready_at), "stopped": _iso(self.stopped_at),
                "effective_settings": self.effective_settings}


def _iso(t: float) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t)) if t else None


# ---------------------------------------------------------------- memory


_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_FP_RE = re.compile(r"\[(\d+)\].*?Footprint:\s*([\d.]+)\s*([KMGT]?B)")
MEMORY_COLUMNS = ("iso", "epoch", "launch", "label", "source", "tree_bytes", "wired_delta", "compressor_delta",
                  "file_backed_delta", "pressure", "swapouts_delta")


def _sh(*cmd: str, timeout: float = 15.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_vm_stat(text: str) -> dict:
    """Bytes of wired, compressor-occupied and file-backed pages, and the cumulative swap-out count."""
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384
    vals = {k.strip().lower(): int(v) for k, v in re.findall(r'^"?([^:\n"]+)"?:\s+(\d+)\.?$', text, re.M)}
    return {"wired": vals.get("pages wired down", 0) * page,
            "compressor": vals.get("pages occupied by compressor", 0) * page,
            "file_backed": vals.get("file-backed pages", 0) * page,
            "swapouts": vals.get("swapouts", 0)}


def system_memory() -> dict:
    mem = parse_vm_stat(_sh("vm_stat"))
    try:
        mem["pressure"] = int(_sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip())
    except ValueError:
        mem["pressure"] = -1
    return mem


def descendants(root: int, ps_text: str) -> list[int]:
    children: dict[int, list[int]] = {}
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        if p not in out:
            out.append(p)
            stack.extend(children.get(p, []))
    return out


def tree_footprint(root: int) -> tuple[str, int]:
    """("footprint" | "rss", bytes) of a PID tree; footprint excludes mmapped file-backed pages."""
    pids = descendants(root, _sh("ps", "-axo", "pid=,ppid="))
    fp = {int(m.group(1)): int(float(m.group(2)) * _UNITS[m.group(3)])
          for m in _FP_RE.finditer(_sh("footprint", "--noCategories", *[a for p in pids for a in ("-p", str(p))]))}
    if fp:
        return "footprint", sum(fp.values())
    rss = 0
    for line in _sh("ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))).splitlines():
        a = line.split()
        if len(a) == 2 and a[1].isdigit():
            rss += int(a[1]) * 1024
    return "rss", rss


class MemorySampler:
    """Background thread, every 2 s: server PID-tree footprint and system-wide deltas (DESIGN §5.2).

    The vm_stat baseline is taken when the sampler is constructed, so construct it before the launch.
    """

    def __init__(self, run_dir: Path, interval: float = 2.0, launch_no: int = 1):
        self.path = Path(run_dir) / "speed" / "memory.tsv"
        self.interval = interval
        self.launch_no = launch_no
        self.baseline = system_memory()
        self.label = "load"
        self.rows: list[dict] = []
        self._pid = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, server_pid: int) -> None:
        self._pid = server_pid
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("\t".join(MEMORY_COLUMNS) + "\n", encoding="utf-8")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="memory-sampler", daemon=True)
        self._thread.start()

    def mark(self, label: str) -> None:
        with self._lock:
            self.label = label

    def sample(self) -> dict:
        source, tree = tree_footprint(self._pid)
        mem = system_memory()
        with self._lock:
            label = self.label
        t = time.time()
        row = {"iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)), "epoch": round(t, 1),
               "launch": self.launch_no, "label": label, "source": source, "tree_bytes": tree,
               "wired_delta": mem["wired"] - self.baseline["wired"],
               "compressor_delta": mem["compressor"] - self.baseline["compressor"],
               "file_backed_delta": mem["file_backed"] - self.baseline["file_backed"],
               "pressure": mem["pressure"], "swapouts_delta": mem["swapouts"] - self.baseline["swapouts"]}
        with self._lock:
            self.rows.append(row)
            with self.path.open("a", encoding="utf-8") as f:
                f.write("\t".join(str(row[c]) for c in MEMORY_COLUMNS) + "\n")
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - sampling must never break a measurement
                pass
            self._stop.wait(max(0.1, self.interval - (time.time() - t0)))

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        try:
            if self._pid:
                self.sample()   # always end with one sample, so short launches are never empty
        except Exception:  # noqa: BLE001
            pass
        return self.summary()

    def summary(self) -> dict:
        """Peaks per label and overall, after-load size, and whether pressure or swap-outs occurred."""
        with self._lock:
            rows = list(self.rows)
        if not rows:
            return {"samples": 0}

        def peak(rs: list[dict]) -> dict:
            return {"tree_bytes": max(r["tree_bytes"] for r in rs),
                    "system_delta_bytes": max(r["wired_delta"] + r["compressor_delta"] + r["file_backed_delta"] for r in rs),
                    "pressure": max(r["pressure"] for r in rs),
                    "swapouts_delta": max(r["swapouts_delta"] for r in rs)}

        labels: dict[str, list[dict]] = {}
        for r in rows:
            labels.setdefault(r["label"], []).append(r)
        after = labels.get("after-load") or labels.get("load") or rows
        return {"samples": len(rows), "launch": self.launch_no, "source": rows[-1]["source"],
                "after_load_tree_bytes": after[0]["tree_bytes"], "peak": peak(rows),
                "by_label": {k: peak(v) for k, v in labels.items()},
                "under_pressure": any(r["pressure"] >= 2 for r in rows),
                "swapped_out": any(r["swapouts_delta"] > 0 for r in rows)}
