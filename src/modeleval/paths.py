"""Paths and constants shared by every module (docs/CONTRACTS.md)."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
MLSYS = HOME / "Developer/mlsys"
OMLX_REPO = MLSYS / "serving/omlx_0.7"
OMLX_PY = OMLX_REPO / ".venv/bin/python"
OMLX_EXE = OMLX_REPO / ".venv/bin/omlx"
BASE = OMLX_REPO / ".modeleval-home"  # pipeline-owned oMLX base path; .bench-home belongs to the old campaign
MODELS_DIR = MLSYS / "models/hf-bench"
PORT = 8703
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
WS_ROOT = MLSYS / "experiments/agent-model-eval-ws"
RESULTS = REPO / "results"
UNVERIFIED = RESULTS / "_unverified"
RUNS = REPO / "runs"
VENDOR = REPO / "vendor"
FIXTURES = REPO / "fixtures"
SCRIPTS = REPO / "scripts"
GPU_LOCK = MLSYS / "experiments/.gpu-lock"
HOLD_FILE = RUNS / "hold"
HF_HOME = Path(os.environ.get("HF_HOME", HOME / "Developer/ml_source/huggingface"))
PROTOCOL_VERSION = 1

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


def tilde(p: Path | str) -> str:
    """A path as written to anything that may be committed: the home directory becomes ~."""
    s = str(p)
    h = str(HOME)
    return "~" + s[len(h):] if s == h or s.startswith(h + "/") else s
