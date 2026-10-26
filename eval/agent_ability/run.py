"""Run the harness smoke suite. Executed by personal-agent-bench's environment:

    cd ../personal-agent-bench
    uv run python ../harness-smoke/run.py --dry-run          # print the plan and preflight
    uv run python ../harness-smoke/run.py                    # all three harnesses, 15 tasks + warm-ups
    uv run python ../harness-smoke/run.py --mcp-arm          # Claude Code + Hermes, MCPMark tasks, MCP only

`python run.py` puts only this directory on sys.path, so personal-agent-bench (for bench.harness)
is added here: $PAB_ROOT, or ../personal-agent-bench next to this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smoke import paths  # noqa: E402

if not (paths.PAB_ROOT / "bench" / "harness.py").exists():
    raise SystemExit(f"personal-agent-bench not found at {paths.PAB_ROOT}; set PAB_ROOT")
sys.path.insert(1, str(paths.PAB_ROOT))

from smoke import runner  # noqa: E402

if __name__ == "__main__":
    sys.exit(runner.main())
