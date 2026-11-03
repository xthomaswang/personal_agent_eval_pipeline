"""Paths, hashing, environment info and the MPS gate.

Every location is set by an environment variable so that tests run in a temp dir:
WSP_ORIGINAL, WSP_LME and WSP_LOCOMO are required (no default); WSP_EDITS, WSP_STATE, WSP_RESULTS, WSP_STAGE_DIR and
HF_HOME have defaults. WSP_GPU_OWNER and WSP_ALLOW_MPS control the MPS gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent  # .../wsp (the uv project)

MAX_DOC_TOKENS = 512
MAX_QUERY_TOKENS = 128
EMBED_DIM = 1024
CPU_MAX_LIMIT = 64
DEFAULT_DEV_LME = 100


class Refused(RuntimeError):
    """A command was refused by one of the harness rules (shown to the user without a traceback)."""


def _required_path(var: str, what: str) -> Path:
    val = os.environ.get(var)
    if not val:
        raise Refused(f"{var} is not set: point it at {what}")
    return Path(val).expanduser()


def original_dir() -> Path:
    return _required_path("WSP_ORIGINAL", "a local copy of the Qwen3-Embedding-0.6B model folder (read-only, never written)")


def edits_dir() -> Path:
    val = os.environ.get("WSP_EDITS")
    return Path(val).expanduser() if val else PKG_ROOT / "models" / "edits"


def state_dir() -> Path:
    return Path(os.environ.get("WSP_STATE", PKG_ROOT / "state"))


def results_dir() -> Path:
    return Path(os.environ.get("WSP_RESULTS", PKG_ROOT / "results"))


def lme_path() -> Path:
    return _required_path("WSP_LME", "longmemeval_s_cleaned.json")


def locomo_path() -> Path:
    return _required_path("WSP_LOCOMO", "locomo10.json")


def stage_dir() -> Path:
    """Folder with stage_set_v1.txt / pilot_set_v1.txt: the sibling eval/memory/public of this package by default."""
    return Path(os.environ.get("WSP_STAGE_DIR", PKG_ROOT.parent / "memory" / "public"))


def hf_home() -> Path | None:
    """HF_HOME when set; otherwise None, which leaves the Hugging Face default cache in place."""
    val = os.environ.get("HF_HOME")
    return Path(val).expanduser() if val else None


# --- hashing ---


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def sha256_file_cached(path: Path) -> str:
    """sha256 of a big file, remembered by (path, size, mtime) in state/hashcache.json."""
    path = Path(path)
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    cache_file = state_dir() / "hashcache.json"
    cache: dict[str, str] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache = {}
    if key not in cache:
        cache[key] = sha256_file(path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    return cache[key]


def sha256_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def sha256_json(obj: object) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --- environment info ---

_ENV_INFO: dict[str, str] | None = None


def env_info() -> dict[str, str]:
    """macOS build and Xcode version, as the machine rules ask for in every recorded result."""
    global _ENV_INFO
    if _ENV_INFO is None:
        info = {}
        for key, cmd in (("sw_vers_build", ["sw_vers", "-buildVersion"]), ("sw_vers_version", ["sw_vers", "-productVersion"])):
            try:
                info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                info[key] = "unknown"
        try:
            out = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True, timeout=20).stdout
            info["xcodebuild"] = " / ".join(out.split("\n")[:2]).strip()
        except (OSError, subprocess.SubprocessError):
            info["xcodebuild"] = "unknown"
        _ENV_INFO = info
    return _ENV_INFO


# --- arm names ---

_ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def check_arm_name(arm: str) -> str:
    if arm == "base" or not _ARM_RE.match(arm):
        raise ValueError(f"bad arm name {arm!r}: use letters, digits, '_', '-', '.', and not 'base'")
    return arm


# --- device gate ---


def gpu_first_line() -> str:
    """First line of the `gpu` command's output (read only, no arguments). Empty when `gpu` is not available."""
    try:
        out = subprocess.run(["gpu"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip().split("\n")[0] if out.strip() else ""


def mps_allowed() -> tuple[bool, str]:
    """MPS is allowed when WSP_ALLOW_MPS=1 (plain opt-in), when a night runner has set NIGHT_DEADLINE for its queued
    job, or when WSP_GPU_OWNER is set and the `gpu` booking tool (if the machine has one) shows that owner's bench slot
    as running now. Otherwise it is refused."""
    if os.environ.get("WSP_ALLOW_MPS") == "1":
        return True, "WSP_ALLOW_MPS=1"
    if os.environ.get("NIGHT_DEADLINE"):
        return True, f"NIGHT_DEADLINE={os.environ['NIGHT_DEADLINE']}"
    owner = os.environ.get("WSP_GPU_OWNER")
    if not owner:
        return False, ""
    line = gpu_first_line()
    if re.search(rf"now:\s*bench by {re.escape(owner)}\b", line):
        return True, line
    return False, line


def check_device(device: str, limit: int | None) -> None:
    """Raise Refused unless this device/limit combination is allowed."""
    if device not in ("cpu", "mps"):
        raise Refused(f"unknown device {device!r} (use cpu or mps)")
    if device == "mps":
        ok, line = mps_allowed()
        if not ok:
            raise Refused(
                "refusing --device mps: set WSP_ALLOW_MPS=1 to opt in, or NIGHT_DEADLINE, or WSP_GPU_OWNER with a "
                f"booking tool that shows `now: bench by <owner>` (WSP_GPU_OWNER={os.environ.get('WSP_GPU_OWNER')!r}, "
                f"first line of `gpu`: {line!r})"
            )
    else:
        if limit is None or limit > CPU_MAX_LIMIT:
            raise Refused(f"--device cpu needs --limit N with N <= {CPU_MAX_LIMIT} (got {limit})")


def lower_priority() -> None:
    """Run CPU jobs at the lowest scheduling priority."""
    try:
        cur = os.nice(0)
        if cur < 19:
            os.nice(19 - cur)
    except OSError:
        pass


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)
