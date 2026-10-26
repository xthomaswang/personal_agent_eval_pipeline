"""Configuration of one proxy instance (one instance per benchmark run)."""

from __future__ import annotations

import datetime as dt
import os
import secrets
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .sampling import DEFAULT_SAMPLING, SamplingPolicy
from .thinking import PASSTHROUGH, ThinkingMode

BENCH_ROOT = Path(__file__).resolve().parents[2]
# What to do with a main call beyond --max-steps: refuse it with HTTP 400
# bench_budget_exhausted (the design's behavior), or forward it and only flag it.
BUDGET_MODES = ("reject", "record")
DEFAULT_UPSTREAM = "http://127.0.0.1:8500"
DEFAULT_PORT = 8600
# An agent call can take many minutes (long prefill plus long generation), and
# oMLX sends a keepalive frame every 10 s, so only a stalled server hits this.
DEFAULT_READ_TIMEOUT = 3600.0
DEFAULT_CONNECT_TIMEOUT = 10.0


def service_root(url: str) -> str:
    """Strip a trailing ``/v1`` (and any query) from a base URL.

    The proxy forwards client paths unchanged (``/v1/chat/completions``,
    ``/v1/messages``), so the upstream must be the server root.
    """
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def upstream_from_env(environ: Mapping[str, str] = os.environ) -> str:
    """Server root of ``OMLX_BASE_URL``, or the oMLX default when unset."""
    raw = environ.get("OMLX_BASE_URL", "").strip()
    return service_root(raw) if raw else DEFAULT_UPSTREAM


def new_run_id() -> str:
    now = dt.datetime.now().astimezone()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def default_run_dir(run_id: str, environ: Mapping[str, str] = os.environ) -> Path:
    base = environ.get("BENCH_RUNS_DIR")
    root = Path(base) if base else BENCH_ROOT / "runs"
    return root / run_id


def upstream_token(command: str) -> str:
    """Run ``command`` once and return its stripped stdout: a bearer token for the upstream.

    The token stays with the caller: it is never printed, recorded, or passed to the harness.
    Errors name the command, never its output.
    """
    try:
        done = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"token command {command!r} failed: {exc}") from exc
    token = done.stdout.strip()
    if done.returncode != 0 or not token or any(c.isspace() for c in token):
        raise ValueError(f"token command {command!r} gave no single-token output (exit {done.returncode})")
    return token

@dataclass(frozen=True)
class ProxyConfig:
    upstream: str
    run_dir: Path
    run_id: str
    harness: str | None = None
    thinking: ThinkingMode = PASSTHROUGH
    sampling: SamplingPolicy = DEFAULT_SAMPLING
    # Step budget: the number of steps (successful main calls, as in n_llm)
    # a run may use; None means no budget.
    max_steps: int | None = None
    budget_mode: str = "reject"
    log_bodies: bool = False
    inject_stream_usage: bool = True
    # Give every OpenAI generation request a fresh random `seed` after the sampling policy. For servers such as
    # rec whose default seed is fixed, so that every unseeded request follows the same sample path.
    random_seed: bool = False
    # Send an assistant message that carries both `reasoning_content` and `reasoning` with only one of them
    # (rec answers 400 "Use one reasoning field per message"; Hermes replays both to local endpoints).
    one_reasoning_field: bool = False
    read_timeout: float = DEFAULT_READ_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    # Print one summary line per call to stderr.
    echo: bool = True
    # Bearer token for an upstream that needs one (`--upstream-token-cmd`). It replaces the client's
    # Authorization header on forwarded calls and is never recorded; repr=False keeps it out of reprs.
    upstream_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream", service_root(self.upstream))
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, not {self.budget_mode!r}")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must not be negative")
