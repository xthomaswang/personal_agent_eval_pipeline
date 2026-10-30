"""Where the suite's code, data, environments and outputs live."""

from __future__ import annotations

import os
from pathlib import Path

SMOKE_ROOT = Path(__file__).resolve().parent.parent
PAB_ROOT = Path(os.environ.get("PAB_ROOT", SMOKE_ROOT.parent / "personal-agent-bench")).resolve()
# The lock shared by every GPU user lives in <mlsys>/experiments, above this repository's eval/ directory
GPU_LOCK = SMOKE_ROOT.parent.parent.parent / ".gpu-lock"
# The holder line personal-agent-bench/scripts/gpu-lock.sh writes (serve-omlx.sh checks the same prefix)
GPU_LOCK_HOLDER = "holder: agent-benchmark"

# Pinned data, fetched by scripts/fetch_data.py (the machine's convention: datasets live in ml_source)
DATA_ROOT = Path(
    os.environ.get("HARNESS_SMOKE_DATA", Path.home() / "Developer" / "ml_source" / "datasets")
).expanduser()
POLYGLOT_SHA = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
MCPMARK_SHA = "cd45b7f57923b9b3985467f5139927575f83141c"
POLYGLOT_DIR = DATA_ROOT / "aider-polyglot-benchmark" / POLYGLOT_SHA[:12]
POLYGLOT_PYTHON = POLYGLOT_DIR / "python" / "exercises" / "practice"
MCPMARK_DIR = DATA_ROOT / "mcpmark" / MCPMARK_SHA[:12]
MCPMARK_EASY = MCPMARK_DIR / "tasks" / "filesystem" / "easy"
MCPMARK_ENVS = DATA_ROOT / "mcpmark" / "filesystem-envs"

# The environment agents see first on PATH, and the one the checkers run tests and verifiers with
SMOKE_VENV_BIN = SMOKE_ROOT / ".venv" / "bin"
SMOKE_PYTHON = SMOKE_VENV_BIN / "python"

# One fixed workspace path per task keeps prompts byte-identical across harnesses and runs
WS_ROOT = SMOKE_ROOT / "ws"
RUNS_ROOT = SMOKE_ROOT / "runs"

# MCP-only arm: the filesystem MCP server, installed per project with nvm's Node
MCP_DIR = SMOKE_ROOT / "mcp"
FS_SERVER_JS = MCP_DIR / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js"
NVM_DIR = Path(os.environ.get("NVM_DIR", Path.home() / ".nvm")).expanduser()


def node_bin() -> Path:
    """The nvm Node named by mcp/.nvmrc (an absolute path: Hermes replaces a bare `node`)."""
    version = (MCP_DIR / ".nvmrc").read_text().strip()
    if not version.startswith("v"):
        version = f"v{version}"
    return NVM_DIR / "versions" / "node" / version / "bin" / "node"
