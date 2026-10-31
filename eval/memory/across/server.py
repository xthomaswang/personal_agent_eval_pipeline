"""Start and stop one localmelo gateway on a throwaway HOME, and read the machine around it.

A gateway keeps its memory, sessions, token and config under ``$HOME/.cache/localmelo``, so each cell of
the study gets its own HOME and the real one is never read or written. The gateway is started the way
``melo serve`` starts it, with stdin closed and ``BROWSER`` set to a command that exits 0, so it opens no
window; its config is written here, so ``melo serve`` never asks questions. Nothing here talks to a model:
the probes at the end send one-token requests to the model server the caller names.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import client

SECRET_ENV = re.compile(r".*(_KEY|_TOKEN|_SECRET|PASSWORD)$", re.IGNORECASE)
NO_BROWSER = "/usr/bin/true"


class LaunchError(RuntimeError):
    pass


class ColdStartError(RuntimeError):
    """The embedding model could not be brought to "not loaded" before a cell (or its state cannot be read)."""


# --- config ---


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a TOML basic string is a JSON string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} as TOML")


def toml_lines(table: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Scalars of ``table`` first, then each sub-table under its dotted name."""
    lines = [f"{k} = {toml_value(v)}" for k, v in table.items() if not isinstance(v, Mapping)]
    for key, sub in table.items():
        if isinstance(sub, Mapping):
            name = f"{prefix}.{key}" if prefix else key
            lines += ["", f"[{name}]", *toml_lines(sub, name)]
    return lines


def config_document(
    base_url: str, model: str, embedding_model: str | None, extra_body: Mapping[str, Any] | None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The gateway's config.toml as data: an oMLX chat backend, with its embedding model for the mixed arm only.

    ``settings`` are an arm's dotted config keys (``agent.window_messages``), written into their tables. A setting
    that `melo serve -X` refuses goes here: a gateway session takes its window cap from ``[agent] window_messages``."""
    section: dict[str, Any] = {"chat_url": base_url, "chat_model": model}
    if embedding_model:
        section.update(embedding_url=base_url, embedding_model=embedding_model)
    if extra_body:
        section["extra_body"] = dict(extra_body)
    document: dict[str, Any] = {
        "chat_backend": "omlx",
        "embedding_backend": "omlx" if embedding_model else "none",
        "request_timeout_s": 600,
        "vllm": section,
    }
    for key, value in (settings or {}).items():
        *tables, leaf = key.split(".")
        table = document
        for name in tables:
            table = table.setdefault(name, {})
            if not isinstance(table, dict):
                raise ValueError(f"config key {key} runs into a value that is not a table")
        table[leaf] = value
    return document


def write_config(home: Path, document: Mapping[str, Any]) -> Path:
    path = Path(home) / ".cache" / "localmelo" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(toml_lines(document)).lstrip("\n") + "\n", encoding="utf-8")
    return path


# --- process ---


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def serve_command(melo: Path, port: int, x_pairs: Sequence[str]) -> list[str]:
    """``melo serve --port N -X KEY=VALUE ...``"""
    return [str(melo), "serve", "--port", str(port), *(a for pair in x_pairs for a in ("-X", pair))]


def serve_env(home: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a gateway: no credentials, its own HOME and TMPDIR, no browser, no terminal."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if not SECRET_ENV.match(k)}
    for name in ("LOCALMELO_APP_INSTANCE", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        env.pop(name, None)
    tmp = Path(home) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), TMPDIR=str(tmp), BROWSER=NO_BROWSER, TERM="dumb", NO_COLOR="1", PYTHONDONTWRITEBYTECODE="1")
    return env


@dataclass(frozen=True)
class LaunchSpec:
    melo: Path
    home: Path
    port: int
    x_pairs: tuple[str, ...]
    log: Path


class GatewayProcess:
    """A running ``melo serve``. ``stop()`` is safe to call twice."""

    def __init__(self, spec: LaunchSpec, process: subprocess.Popen[bytes]) -> None:
        self.spec = spec
        self.process = process
        self.pid = process.pid
        self.base_url = f"http://127.0.0.1:{spec.port}"

    @property
    def home(self) -> Path:
        return self.spec.home

    def alive(self) -> bool:
        return self.process.poll() is None

    def token(self) -> str:
        return client.read_token(self.spec.home)

    def stop(self, grace: float = 20.0) -> int | None:
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        return self.process.returncode


def log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def wait_ready(proc: Any, timeout: float = 120.0, *, sleep: Callable[[float], None] = time.sleep) -> client.Gateway:
    """A client for the gateway once ``/api/health`` says it is configured and its store is open."""
    deadline = time.monotonic() + timeout
    last = "no answer yet"
    while time.monotonic() < deadline:
        if not proc.alive():
            raise LaunchError(f"melo serve exited with {proc.process.returncode}:\n{log_tail(proc.spec.log)}")
        try:
            gw = client.Gateway(proc.base_url, proc.token(), timeout=10.0)
            health = gw.health()
            if health.get("configured") is True and (health.get("sessions") or {}).get("store") == "ok":
                return gw
            last = f"configured={health.get('configured')!r} errors={health.get('config_errors')!r}"
        except (OSError, client.GatewayError, KeyError, ValueError) as exc:
            last = str(exc)
        sleep(0.25)
    raise LaunchError(f"the gateway was not ready in {timeout:g} s ({last}):\n{log_tail(proc.spec.log)}")


def launch(spec: LaunchSpec) -> GatewayProcess:
    """Start ``melo serve`` on ``spec.home``: stdin closed, output to the log, its own process group."""
    spec.log.parent.mkdir(parents=True, exist_ok=True)
    with spec.log.open("ab") as log:
        process = subprocess.Popen(
            serve_command(spec.melo, spec.port, spec.x_pairs),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            cwd=spec.home, env=serve_env(spec.home), start_new_session=True,
        )
    return GatewayProcess(spec, process)


# --- the machine ---


def out(cmd: Sequence[str], timeout: float = 30.0) -> str:
    """Stripped stdout of a command, or "" when it is missing, fails or is slow."""
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_sha(repo: Path) -> str:
    return out(["git", "-C", str(repo), "rev-parse", "HEAD"])


def git_dirty(repo: Path) -> bool:
    return bool(out(["git", "-C", str(repo), "status", "--porcelain"]))


# oMLX answers its admin routes (`/api/status`, `/v1/models/status`, unload) only with its API key; inference and
# `/v1/models` are open. The key comes from --admin-token-cmd, is kept in memory only and is sent on those routes alone.
_ADMIN_TOKEN: list[str] = []


def set_admin_token(token: str | None) -> None:
    _ADMIN_TOKEN[:] = [token] if token else []


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_ADMIN_TOKEN[0]}"} if _ADMIN_TOKEN else {}


def http_json(url: str, body: Any = None, timeout: float = 30.0, *, admin: bool = False) -> tuple[float, Any]:
    """(milliseconds, parsed JSON) of a GET (or a POST with ``body``); no proxy, no redirect. ``admin`` adds the
    model server's API key, for its admin routes."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers={**headers, **(_admin_headers() if admin else {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter()
    with opener.open(request, timeout=timeout) as response:
        parsed = json.loads(response.read() or b"null")
    return round((time.perf_counter() - t0) * 1000.0, 1), parsed


def model_ids(base_url: str) -> tuple[list[str], Any]:
    """The ids ``GET <base_url>/models`` lists, and the raw answer."""
    _, listing = http_json(base_url.rstrip("/") + "/models", timeout=15.0)
    rows = listing.get("data") if isinstance(listing, dict) else None
    return [str(r["id"]) for r in rows or [] if isinstance(r, dict) and "id" in r], listing


def listener_pid(base_url: str) -> int | None:
    """The process listening on the model server's port, or None."""
    port = urllib.parse.urlsplit(base_url).port if "://" in base_url else None
    text = out(["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]) if port else ""
    return int(text.split()[0]) if text.split() else None


def rss_kb(pid: int | None) -> int | None:
    text = out(["ps", "-o", "rss=", "-p", str(pid)]) if pid else ""
    return int(text) if text.isdigit() else None


def model_server_rss(base_url: str) -> dict[str, Any]:
    pid = listener_pid(base_url)
    return {"pid": pid, "rss_kb": rss_kb(pid), "at": time.time()}


STATUS_KEYS = ("version", "models_loaded", "models_loading", "loaded_models", "model_memory_used", "model_memory_max", "active_requests", "waiting_requests")


def loaded_models(base_url: str) -> dict[str, Any]:
    """Which models the server has in memory now, from oMLX's `GET /api/status` (pool totals and the ids in
    `loaded_models`) and `GET /v1/models/status` (per model: loaded, loading, size). Both sit at the server's
    root, outside `/v1`; either may be missing. `loaded` is the list of ids, or None when neither answers."""
    root = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(base_url))
    out: dict[str, Any] = {"loaded": None}
    try:
        _, status = http_json(root + "/api/status", timeout=10.0, admin=True)
        if isinstance(status, dict):
            out["status"] = {k: status.get(k) for k in STATUS_KEYS}
            if isinstance(status.get("loaded_models"), list):
                out["loaded"] = [str(m) for m in status["loaded_models"]]
    except (OSError, ValueError):
        pass
    try:
        _, listing = http_json(root + "/v1/models/status", timeout=10.0, admin=True)
        rows = listing.get("models") if isinstance(listing, dict) else None
        if isinstance(rows, list):
            out["models"] = [{k: m.get(k) for k in ("id", "loaded", "is_loading", "actual_size", "estimated_size")} for m in rows if isinstance(m, dict)]
            if out["loaded"] is None:
                out["loaded"] = [str(m["id"]) for m in out["models"] if m.get("loaded")]
    except (OSError, ValueError):
        pass
    return out


def model_is_loaded(state: Mapping[str, Any], model_id: str) -> bool | None:
    """Whether ``model_id`` is loaded, from ``loaded_models``: its row in `/v1/models/status` (no row: not loaded),
    else the ids in `/api/status`; None when neither answered."""
    rows = state.get("models")
    if isinstance(rows, list):
        return any(r.get("id") == model_id and bool(r.get("loaded")) for r in rows)
    if isinstance(state.get("loaded"), list):
        return model_id in state["loaded"]
    return None


def unload_model(base_url: str, model_id: str, timeout: float = 30.0) -> int:
    """`POST /v1/models/{id}/unload` at the server root; the HTTP status (200 done, 202 queued behind running
    requests, 400 was not loaded, 404 unknown, 409 still loading), or 0 when the server did not answer."""
    root = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(base_url))
    request = urllib.request.Request(
        f"{root}/v1/models/{urllib.parse.quote(model_id, safe='')}/unload", data=b"", method="POST", headers=_admin_headers()
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except OSError:
        return 0


def ensure_unloaded(
    base_url: str, model_id: str, *, timeout: float = 60.0, poll: float = 0.5,
    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Bring ``model_id`` to "not loaded" so a cell starts cold for it; what happened, for cell.json.

    Asks the server to unload it only if `/v1/models/status` says it is loaded, then waits until the server says it
    is not. Raises ColdStartError if the state cannot be read or the model is still loaded after ``timeout``.
    """
    started = clock()
    loaded = model_is_loaded(loaded_models(base_url), model_id)
    if loaded is None:
        raise ColdStartError(f"cannot tell whether {model_id!r} is loaded: {base_url} answers neither /api/status nor /v1/models/status")
    out: dict[str, Any] = {"model": model_id, "was_loaded": loaded, "unload_status": None, "attempts": 0, "wait_ms": 0.0, "loaded_after": False}
    if not loaded:
        return out
    while clock() - started < timeout:
        if out["attempts"] == 0 or out["unload_status"] in (0, 409):  # not answered, or still loading: ask again
            out["unload_status"] = unload_model(base_url, model_id)
            out["attempts"] += 1
        sleep(poll)
        if model_is_loaded(loaded_models(base_url), model_id) is False:
            out["wait_ms"] = round((clock() - started) * 1000.0, 1)
            return out
    raise ColdStartError(f"{model_id!r} is still loaded {timeout:g} s after the unload request (status {out['unload_status']})")


def model_server_state(base_url: str) -> dict[str, Any]:
    """The model server process's rss and the models it has loaded, at this moment."""
    return {**model_server_rss(base_url), **loaded_models(base_url)}


def warmup(base_url: str, model: str, embedding_model: str | None, timeout: float = 900.0) -> dict[str, Any]:
    """One-token requests to the chat model (and the embedding model), timed: this is where a model is loaded.

    The ms are the load-plus-first-call cost of each model on a server that has not got it in memory yet,
    and a plain call on one that has; the cell records both models' rss around it.
    """
    base = base_url.rstrip("/")
    done: dict[str, Any] = {}
    try:
        done["chat_ms"], _ = http_json(base + "/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "temperature": 0,
        }, timeout)
        if embedding_model:
            done["embed_ms"], _ = http_json(base + "/embeddings", {"model": embedding_model, "input": "ping"}, timeout)
    except (OSError, ValueError) as exc:
        done["error"] = f"{type(exc).__name__}: {exc}"
    return done
