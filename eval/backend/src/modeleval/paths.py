"""Paths and constants shared by every module (docs/CONTRACTS.md).

Where things are on this machine comes from config.defaults.toml, overridden key by key by the
gitignored config.toml next to it; nothing else in the pipeline names a machine path.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]  # this backend directory (its own uv project)


def _monorepo(repo: Path) -> Path:
    """The repository the backend lives in: the nearest parent with docs/benchmark (the results contract),
    wherever the backend sits inside it; the parent directory when there is none."""
    return next((p for p in repo.parents if (p / "docs" / "benchmark").is_dir()), repo.parent)


class ConfigError(ValueError):
    """config.defaults.toml / config.toml, or a ${name} reference in pins.toml or models.toml, is wrong."""


CONFIG_FILES = (REPO / "config.defaults.toml", REPO / "config.toml")   # committed defaults, local override
_REF = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def load_config(files: tuple[Path, ...] = CONFIG_FILES) -> dict[str, dict[str, Any]]:
    """The tables of the config files, later files overriding earlier ones key by key."""
    merged: dict[str, dict[str, Any]] = {}
    for f in files:
        if f.is_file():
            try:
                doc = tomllib.loads(f.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(f"{f.name}: {e}") from None
            for table, values in doc.items():
                if isinstance(values, dict):
                    merged.setdefault(table, {}).update(values)
    return merged


CONFIG = load_config()


def expand(value: str | os.PathLike[str], base: Path | None = None) -> Path:
    """A configured path: every ${name} replaced by [paths] name (which may hold references itself), `~`
    expanded, and a relative path taken relative to `base` (default: this backend directory)."""
    text, table = str(value), CONFIG.get("paths", {})
    for _ in range(20):
        try:
            new = _REF.sub(lambda m: str(table[m.group(1)]), text)
        except KeyError as e:
            raise ConfigError(f"{value!r}: no [paths] {e.args[0]} in config.defaults.toml or config.toml") from None
        if new == text:
            break
        text = new
    p = Path(text).expanduser()
    return p if p.is_absolute() else (base or REPO) / p


def _configured(name: str) -> Path:
    if name not in CONFIG.get("paths", {}):
        raise ConfigError(f"config.defaults.toml has no [paths] {name}")
    return expand(CONFIG["paths"][name])


MONOREPO = _monorepo(REPO)
HOME = Path.home()
MLSYS = _configured("mlsys")
OMLX_REPO = _configured("omlx_repo")
OMLX_PY = OMLX_REPO / ".venv/bin/python"
OMLX_EXE = OMLX_REPO / ".venv/bin/omlx"
BASE = _configured("omlx_base")  # pipeline-owned oMLX base path; .bench-home belongs to the old campaign
MODELS = _configured("models")
MODELS_DIR = _configured("models_dir")
PORT = 8703
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
WS_ROOT = _configured("ws_root")
RESULTS = MONOREPO / "docs" / "benchmark"  # verified runs: <machine>/<family>/<quant_dir>/<SUITE>/<run-id>/
SUITE = "backend"                         # this pipeline's suite directory under docs/benchmark
UNVERIFIED = REPO / "results" / "_unverified"   # gitignored: major-disturbed attempts, --unverified runs
RUNS = REPO / "runs"
VENDOR = REPO / "vendor"
FIXTURES = REPO / "fixtures"
SCRIPTS = REPO / "scripts"
GPU_LOCK = _configured("gpu_lock")
HOLD_FILE = RUNS / "hold"
HF_HOME = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else _configured("hf_home")
PROTOCOL_VERSION = 1
LOCK_MODES = ("auto", "on", "off")


def lock_enabled() -> bool:
    """Whether this machine runs the GPU-lock protocol (config [lock] mode; "auto": the lock file's
    directory exists)."""
    mode = str(CONFIG.get("lock", {}).get("mode", "auto"))
    if mode not in LOCK_MODES:
        raise ConfigError(f"[lock] mode must be one of {', '.join(LOCK_MODES)}, got {mode!r}")
    return mode == "on" or (mode == "auto" and GPU_LOCK.parent.is_dir())


def lock_env() -> dict[str, str]:
    """Environment for gpu_lock.sh and quietlib: the configured lock file ($GPU_LOCK_PATH wins, as in the
    scripts), and QUIET_LOCK=off when the protocol is off (quietlib then requires no lock)."""
    env = {"GPU_LOCK_PATH": os.environ.get("GPU_LOCK_PATH") or str(GPU_LOCK)}
    if not lock_enabled():
        env["QUIET_LOCK"] = "off"
    return env

# Ports that belong to other sessions (the agent benchmark): never used here.
FOREIGN_PORTS = (8500,) + tuple(range(8600, 8700))


def status_pathspec(repo: Path | str) -> list[str]:
    """Pathspec for `git status --porcelain` on a peer checkout that leaves out the pipeline's own
    directories inside it (BASE lives in the oMLX checkout). Untracked files still count as dirty
    everywhere else, and the peer's .git/info/exclude stays untouched; [] for any other repo."""
    try:
        rel = BASE.resolve().relative_to(Path(repo).resolve())
    except (OSError, ValueError):
        return []
    return ["--", ".", f":(exclude){rel}"] if str(rel) != "." else []


def _probe(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


@functools.cache
def machine_id() -> str:
    """This Mac as the first level of the results path: <chip>-<GPU cores>c-<RAM>gb, e.g. m5max-40c-128gb.
    Chip from `sysctl machdep.cpu.brand_string` ("Apple M5 Max" -> m5max), GPU cores from the AGX
    accelerator's gpu-core-count (ioreg), RAM from hw.memsize in GiB. $MODELEVAL_MACHINE, then config
    [machine] id, override it; a part that cannot be read is "x"."""
    fixed = os.environ.get("MODELEVAL_MACHINE") or str(CONFIG.get("machine", {}).get("id") or "")
    if fixed:
        if "/" in fixed or fixed in (".", ".."):
            raise ConfigError(f"the machine id is one directory name, got {fixed!r}")
        return fixed
    brand = _probe(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip().lower()
    chip = re.sub(r"[^a-z0-9]", "", brand.removeprefix("apple")) or "x"
    cores = re.search(r'"gpu-core-count"\s*=\s*(\d+)', _probe(["/usr/sbin/ioreg", "-rd1", "-c", "AGXAccelerator"]))
    mem = _probe(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip()
    ram = f"{round(int(mem) / 2**30)}" if mem.isdigit() else "x"
    return f"{chip}-{cores.group(1) if cores else 'x'}c-{ram}gb"


def results_dir(family: str, quant_dir: str, suite: str = SUITE, machine: str | None = None) -> Path:
    """Where a model's verified runs of one suite go on this machine: the only place the results layout of
    docs/benchmark/README.md is spelled out (one <run-id>/ directory per run inside)."""
    return RESULTS / (machine or machine_id()) / family / quant_dir / suite


def run_parts(run_dir: Path | str) -> tuple[str, str, str, str, str]:
    """(machine, family, quant, suite, run_id) of a run directory under results_dir()."""
    *_, machine, family, quant, suite, run_id = Path(run_dir).parts
    return machine, family, quant, suite, run_id


def tilde(p: Path | str) -> str:
    """A path as written to anything that may be committed: the home directory becomes ~."""
    s = str(p)
    h = str(HOME)
    return "~" + s[len(h):] if s == h or s.startswith(h + "/") else s
