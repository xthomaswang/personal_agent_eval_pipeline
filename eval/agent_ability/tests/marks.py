"""Skip markers. The data-dependent tests skip when scripts/fetch_data.py has not run, and the
runner tests skip outside personal-agent-bench's environment (they need bench.harness)."""

from __future__ import annotations

import importlib.util

import pytest

from smoke import paths

needs_data = pytest.mark.skipif(
    not (paths.POLYGLOT_PYTHON.is_dir() and paths.MCPMARK_EASY.is_dir() and paths.MCPMARK_ENVS.is_dir()),
    reason="pinned data missing; run: .venv/bin/python -I scripts/fetch_data.py",
)

needs_bench = pytest.mark.skipif(
    importlib.util.find_spec("starlette") is None or importlib.util.find_spec("yaml") is None,
    reason="needs personal-agent-bench's environment: cd ../personal-agent-bench && uv run pytest ../harness-smoke/tests",
)

needs_smoke_venv = pytest.mark.skipif(
    not paths.SMOKE_PYTHON.exists(), reason="smoke environment missing; run: uv sync (in harness-smoke)"
)
