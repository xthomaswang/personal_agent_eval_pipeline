"""Provenance header for every raw JSONL file. Never records serials, UUIDs, or hostnames.
Home and $HF_HOME prefixes are redacted to `~` / `$HF_HOME` so committed files hold no
absolute user paths."""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


def redact(s: str) -> str:
    hf = os.environ.get("HF_HOME")
    if hf:
        s = s.replace(hf, "$HF_HOME")
    return s.replace(str(Path.home()), "~")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def harness_git() -> dict[str, Any]:
    root = str(repo_root())
    head = _run(["git", "-C", root, "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", root, "status", "--porcelain", "--", "harness", "fixtures"])
    return {"commit": head, "dirty": bool(dirty) if dirty is not None else None}


@lru_cache(maxsize=1)
def system_info() -> dict[str, Any]:
    sw = {k: _run(["sw_vers", f"-{k[0].lower()}{k[1:]}"]) for k in ("ProductVersion", "BuildVersion")}
    beta = "/Applications/Xcode-beta.app/Contents/Developer"
    xcode = {"default": _run(["xcodebuild", "-version"])}
    if Path(beta).exists():
        xcode["beta"] = _run(["xcodebuild", "-version"], env={**os.environ, "DEVELOPER_DIR": beta})
    mem = _run(["sysctl", "-n", "hw.memsize"])
    return {
        "sw_vers": sw,
        "xcodebuild": xcode,
        "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(mem) if mem and mem.isdigit() else None,
        "power": (_run(["pmset", "-g", "batt"]) or "").splitlines()[:1],
    }


def header(suite: str, tag: str, framework: dict[str, Any], server: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    fw = {k: (redact(v) if isinstance(v, str) else v) for k, v in framework.items()}
    return {
        "type": "header",
        "suite": suite,
        "tag": tag,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "framework": fw,
        "server": server,
        "harness": dict(harness_git()),
        "system": dict(system_info()),
        "args": {k: (redact(v) if isinstance(v, str) else v) for k, v in args.items()},
    }
