"""Record the environment a benchmark run executed in.

Every run stores this next to its events so numbers stay comparable across
days: macOS build, Xcode, chip, memory and power state, model server commit
and settings, model revision, harness versions, and this repo's commit.

Usage:
    uv run python -m bench.manifest                      # print to stdout
    uv run python -m bench.manifest --out runs/<id>/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 1

HOME = Path.home()
BENCH_ROOT = Path(__file__).resolve().parent.parent
# <mlsys>/experiments/<this repository>/eval: three levels up (two before eval/ existed)
MLSYS_ROOT = BENCH_ROOT.parent.parent.parent
# The active serving setup. .envrc exports the same values; the defaults matter when a run starts
# without direnv (a non-interactive shell), which once recorded the 0.3.9 install for 0.7.1rc1 runs.
OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8500/v1")
OMLX_REPO = Path(os.environ.get("OMLX_REPO", MLSYS_ROOT / "serving" / "omlx-v0.7.1rc1"))
OMLX_HOME = Path(os.environ.get("OMLX_BASE_PATH", HOME / ".omlx-0.7")).expanduser()
BENCH_MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B-oQ5-mtp")
# A stock model comes from the HF cache (BENCH_MODEL_HF_REPO). A model we converted ourselves lives
# in a model directory, with scripts/oq_convert.py's provenance record next to its weights.
BENCH_MODEL_HF_REPO = os.environ.get("BENCH_MODEL_HF_REPO", "")
BENCH_MODEL_PATH = os.environ.get(
    "BENCH_MODEL_PATH", str(MLSYS_ROOT / "models" / "chat" / "Qwen3.8-27B-oQ5-mtp")
)
CONVERSION_RECORD = "pab-conversion.json"
# personal-agent-bench's scripts/serve-omlx.sh records what it started in .server-<port>.json here
SERVED_DIR = Path(os.environ.get("BENCH_SERVED_DIR", MLSYS_ROOT / "experiments" / "personal-agent-bench" / "runs"))
# The localmelo checkout whose .venv runs melo: by default the bench's own worktree, pinned at a
# fixed commit, so the main checkout can change without changing a run. Override to test a branch.
LOCALMELO_REPO = Path(
    os.environ.get("BENCH_LOCALMELO_REPO", MLSYS_ROOT / "agent" / "melo" / "worktrees" / "bench" / "localmelo")
)
CLAUDE_BIN = Path(os.environ.get("CLAUDE_BIN", HOME / ".local" / "share" / "claude" / "versions" / "2.1.302"))
HERMES_BIN = MLSYS_ROOT / "agent" / ".venv" / "bin" / "hermes"
HF_HUB = (
    Path(os.environ.get("HF_HOME", HOME / "Developer" / "ml_source" / "huggingface"))
    / "hub"
)

# The T1 site settles daily check-ins by Los Angeles date; its oracle needs it
CHECKIN_TZ = ZoneInfo("America/Los_Angeles")

# Active `powermode` value reported by `pmset -g`
POWER_MODES = {"0": "automatic", "1": "low_power", "2": "high_power"}


def _run(cmd: list[str], timeout: float = 30.0) -> str | None:
    """Return stripped stdout, or None when the command is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"path": str(repo), "present": False}
    status = _run(["git", "-C", str(repo), "status", "--porcelain"])
    return {
        "path": str(repo),
        "present": True,
        "commit": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "describe": _run(
            ["git", "-C", str(repo), "describe", "--tags", "--always", "--dirty"]
        ),
        "dirty": None if status is None else bool(status),
    }


def _host() -> dict[str, Any]:
    gpu_cores = None
    displays = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=60)
    if displays:
        try:
            entries = json.loads(displays).get("SPDisplaysDataType", [])
        except json.JSONDecodeError:
            entries = []
        gpu_cores = next(
            (int(e["sppci_cores"]) for e in entries if e.get("sppci_cores")), None
        )

    memsize = _run(["sysctl", "-n", "hw.memsize"])
    ncpu = _run(["sysctl", "-n", "hw.ncpu"])
    batt = _run(["pmset", "-g", "batt"]) or ""
    power_source = None
    if "'" in batt:
        power_source = batt.split("'")[1]
    pm = _run(["pmset", "-g"]) or ""
    raw_mode = next(
        (line.split()[1] for line in pm.splitlines() if line.split()[:1] == ["powermode"]),
        None,
    )
    power_mode = POWER_MODES.get(raw_mode, raw_mode)

    return {
        "hostname": socket.gethostname(),
        "model_identifier": _run(["sysctl", "-n", "hw.model"]),
        "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cpu_cores": int(ncpu) if ncpu else None,
        "gpu_cores": gpu_cores,
        "memory_bytes": int(memsize) if memsize else None,
        # Wall-clock time depends on power source and power mode; record both to explain outliers
        "power_source": power_source,
        "power_mode": power_mode,
    }


def _xcode() -> dict[str, Any]:
    return {
        "xcodebuild_version": _run(["xcodebuild", "-version"]),
        "xcode_select_path": _run(["xcode-select", "-p"]),
        "developer_dir_env": os.environ.get("DEVELOPER_DIR"),
    }


def _served(port: int) -> dict[str, Any] | None:
    """What scripts/serve-omlx.sh started on this port, if that process is the one listening now."""
    try:
        info = json.loads((SERVED_DIR / f".server-{port}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    listeners = _run(["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]) or ""
    return info if str(info.get("pid")) in listeners.split() else None


def _model_server() -> dict[str, Any]:
    served = _served(urllib.parse.urlsplit(OMLX_BASE_URL).port or 8500)
    repo = Path(served["repo"]) if served else OMLX_REPO
    home = Path(served["base_path"]) if served else OMLX_HOME
    settings: dict[str, Any] | None = None
    if (home / "settings.json").exists():
        raw = json.loads((home / "settings.json").read_text())
        # Keep only fields that affect results; never copy auth secrets into the record
        settings = {
            "port": raw.get("server", {}).get("port"),
            # The memory guard can pause admission or abort a request; 0.7 ignores model.max_model_memory
            "memory": raw.get("memory"),
            "model_dirs": raw.get("model", {}).get("model_dirs"),
            "cache": raw.get("cache"),
            "sampling": raw.get("sampling"),
            # context_scaling_enabled rescales Anthropic usage numbers that Claude Code sees
            "claude_code": raw.get("claude_code"),
        }
    model_id: str | None = None
    model_settings: dict[str, Any] | None = None
    if (home / "model_settings.json").exists():
        models = json.loads((home / "model_settings.json").read_text()).get("models", {})
        # Per-model overrides (context window, template kwargs, speculative decoding) win over sampling.
        # oMLX 0.7 names HF-cache models "org--name"; the bench model id is then its model_alias.
        model_id, model_settings = next(
            (
                (mid, ms)
                for mid, ms in models.items()
                if BENCH_MODEL in (mid, ms.get("model_alias"))
            ),
            (None, None),
        )
    return {
        "name": "omlx",
        "base_url": OMLX_BASE_URL,
        "base_path": str(home),
        "repo": _git(repo),
        # True: serve-omlx.sh started the process listening on the port, and repo and base_path are
        # what it ran. False: they are the configured values, not checked against the live server.
        "verified": served is not None,
        "pid": served["pid"] if served else None,
        "settings": settings,
        "model_id": model_id,
        "model_settings": model_settings,
    }


def _model() -> dict[str, Any]:
    repo_id = BENCH_MODEL_HF_REPO
    info: dict[str, Any] = {"id": BENCH_MODEL, "hf_repo": repo_id or None}
    if repo_id:
        entry = HF_HUB / ("models--" + repo_id.replace("/", "--"))
        ref = entry / "refs" / "main"
        revision = ref.read_text().strip() if ref.exists() else None
        info["revision"] = revision
        info["snapshot_path"] = (
            str(entry / "snapshots" / revision) if revision else None
        )
    elif BENCH_MODEL_PATH:
        path = Path(BENCH_MODEL_PATH).expanduser()
        record = path / CONVERSION_RECORD
        info["path"] = str(path)
        # Source revision, converter commit and versions, shard hashes; None if it was not converted here
        info["conversion"] = json.loads(record.read_text()) if record.exists() else None
    return info


def _harnesses() -> dict[str, Any]:
    # The pinned binaries the bench launches, not whatever is first on PATH
    return {
        "claude_code": _run([str(CLAUDE_BIN), "--version"]),
        "hermes": _run([str(HERMES_BIN), "--version"]),
        "localmelo": _git(LOCALMELO_REPO),
    }


def collect() -> dict[str, Any]:
    now = dt.datetime.now(dt.UTC)
    local = now.astimezone()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": now.isoformat(),
        "created_at_local": local.isoformat(),
        "checkin_date_la": now.astimezone(CHECKIN_TZ).date().isoformat(),
        "host": _host(),
        "os": {
            "product_version": _run(["sw_vers", "-productVersion"]),
            "build_version": _run(["sw_vers", "-buildVersion"]),
        },
        "xcode": _xcode(),
        "model_server": _model_server(),
        "model": _model(),
        "harnesses": _harnesses(),
        "bench": _git(BENCH_ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the manifest to this path")
    args = parser.parse_args()

    manifest = collect()
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
