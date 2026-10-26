"""Fixtures for the capability tests: the fake world, the mock server, the live repos' index stamps."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from modeleval import paths
from modeleval.capability import epoch as epochs

from .fakes import MockServer, build_world, repo_pins

CAPABILITY_PY = paths.REPO / "envs" / "capability" / ".venv" / "bin" / "python"
AGENT_TOOLS_PY = paths.REPO / "envs" / "agent-tools" / ".venv" / "bin" / "python"


def _live() -> dict[str, Any]:
    _, pins = epochs.load_pins()
    return pins


def live_repos() -> list[Path]:
    """Every peer repository the pipeline reads: none of their index files may change. The snapshot
    repository is this pipeline's own (other sessions commit to it while tests run), so it is not one."""
    pins = _live()
    return [
        pins["hermes"]["repo"], pins["localmelo"]["repo"],
        pins["data"]["aider_polyglot_dir"], pins["data"]["mcpmark_dir"], paths.OMLX_REPO,
    ]  # fmt: skip


def _missing() -> str | None:
    pins = _live()
    needed = [pins["snapshot"]["repo"], pins["data"]["aider_polyglot_dir"], pins["data"]["mcpmark_dir"],
              pins["data"]["mcpmark_envs_manifest"]]  # fmt: skip
    gone = [str(p) for p in needed if not p.exists()]
    return f"missing: {', '.join(gone)}" if gone else None


needs_live = pytest.mark.skipif(_missing() is not None, reason=_missing() or "")
needs_envs = pytest.mark.skipif(
    not (CAPABILITY_PY.exists() and AGENT_TOOLS_PY.exists()),
    reason="run `uv sync` in envs/capability and envs/agent-tools",
)


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    return build_world(tmp_path)


@pytest.fixture
def mock_server(world: dict[str, Any]) -> Iterator[MockServer]:
    server = MockServer(world["root"], world["root"] / "server.pid").start()
    yield server
    server.stop()


@pytest.fixture
def live_index_stamps() -> Iterator[dict[str, Any]]:
    """The live repos' index mtimes and inodes; asserted unchanged after the test."""
    before = epochs.index_stamps(live_repos())
    yield before
    assert epochs.index_stamps(live_repos()) == before


@pytest.fixture
def pinned_sha() -> str:
    return repo_pins()["snapshot"]["sha"]
