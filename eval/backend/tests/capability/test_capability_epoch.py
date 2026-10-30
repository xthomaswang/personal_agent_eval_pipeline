"""Epoch creation from a git-archive snapshot of the live peer repos (read only), its content hash, and
the live re-checks (DESIGN §6.5)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from modeleval.capability import epoch as epochs

from .conftest import needs_envs, needs_live
from .fakes import _git as fake_git

pytestmark = [needs_live, needs_envs]


def _create(world: dict[str, Any], sha: str, vendor: Path, raw: dict[str, Any] | None = None) -> str:
    return epochs.create_epoch(sha, pins_raw=raw or world["pins_raw"], vendor_root=vendor)


def test_create_snapshot_and_hash(world: dict[str, Any], pinned_sha: str, tmp_path: Path,
                                  live_index_stamps: dict) -> None:  # fmt: skip
    vendor = tmp_path / "vendor"
    epoch_id = _create(world, pinned_sha, vendor)
    root = vendor / epoch_id
    for rel in ("personal-agent-bench/bench/harness.py", "personal-agent-bench/bench/proxy/app.py",
                "harness-smoke/smoke/runner.py", "harness-smoke/shims/pip", epochs.CLAUDE_ENV_COPY,
                epochs.CLAUDE_SETTINGS_COPY, f"{epochs.HERMES_HOME_COPY}/config.yaml"):  # fmt: skip
        assert (root / rel).is_file(), rel
    hermes = root / epochs.HERMES_HOME_COPY
    assert (hermes / "tools").is_symlink() and (hermes / "tools").resolve() == (world["agent"] / "homes/hermes/tools").resolve()
    assert not (hermes / "logs").exists() and not (hermes / "cache").exists()
    record = json.loads((root / "epoch.json").read_text())
    live = epochs.load_pins()[1]
    trees = record["components"]["trees"]
    snap = live["snapshot"]
    for part in epochs.SNAPSHOT_PARTS:
        assert trees[part] == epochs._rev(snap["repo"], f"{pinned_sha}:{snap[part]}")
    assert record["snapshot"]["sha"].startswith(pinned_sha[:7]) and record["snapshot"]["bench"] == snap["bench"]
    epoch = epochs.load_epoch(epoch_id, vendor)
    assert epochs.snapshot_problems(epoch) == []
    assert epochs.verify_live(epoch) == []
    # Idempotent: the same inputs give the same id and leave no staging directory
    assert _create(world, pinned_sha, vendor) == epoch_id
    assert [p.name for p in vendor.iterdir()] == [epoch_id]


def test_the_id_follows_content(world: dict[str, Any], pinned_sha: str, tmp_path: Path) -> None:
    first = _create(world, pinned_sha, tmp_path / "v1")
    # Another ws_root is another prompt text, so another epoch
    raw = json.loads(json.dumps(world["pins_raw"]))
    raw["workspace"]["ws_root"] = str(tmp_path / "other-ws")
    assert _create(world, pinned_sha, tmp_path / "v2", raw) != first
    # A config edit is a new epoch too
    env = world["agent"] / "configs/claude-code/claude-code.env"
    env.write_text(env.read_text() + "# edited\n")
    assert _create(world, pinned_sha, tmp_path / "v3") != first


def test_live_drift_is_reported(world: dict[str, Any], pinned_sha: str, tmp_path: Path) -> None:
    epoch = epochs.load_epoch(_create(world, pinned_sha, tmp_path / "vendor"), tmp_path / "vendor")
    (world["hermes_repo"] / "local-edit.txt").write_text("x\n")
    assert any("hermes-agent" in p and "uncommitted" in p for p in epochs.verify_live(epoch))
    (world["hermes_repo"] / "local-edit.txt").unlink()
    fake_git(world["melo_repo"], "commit", "-q", "--allow-empty", "-m", "moved")
    assert any("localmelo" in p and "pinned" in p for p in epochs.verify_live(epoch))
    fake_git(world["melo_repo"], "reset", "-q", "--hard", "HEAD~1")
    assert epochs.verify_live(epoch) == []
    st = world["claude_bin"].stat()
    os.utime(world["claude_bin"], ns=(0, 0))
    assert any("changed since the epoch" in p for p in epochs.verify_live(epoch))
    os.utime(world["claude_bin"], ns=(st.st_atime_ns, st.st_mtime_ns))
    assert epochs.verify_live(epoch) == []
    (world["agent"] / "homes/hermes/tools/facts.json").write_text("{}\n")
    assert epochs.verify_live(epoch) == ["hermes_tools_facts changed since the epoch was created"]


def test_snapshot_edits_are_detected(world: dict[str, Any], pinned_sha: str, tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    epoch = epochs.load_epoch(_create(world, pinned_sha, vendor), vendor)
    root = epoch["dir"]
    (root / "personal-agent-bench/bench/harness.py").write_text("tampered\n")
    pycache = root / "harness-smoke/smoke/__pycache__"
    pycache.mkdir()
    (pycache / "runner.cpython-313.pyc").write_bytes(b"x")
    problems = epochs.snapshot_problems(epoch)
    assert "personal-agent-bench/bench/harness.py: changed" in problems
    assert "harness-smoke/smoke/__pycache__/runner.cpython-313.pyc: not part of the snapshot" in problems


@pytest.mark.parametrize("break_it", ["memories", "env"])
def test_hermes_home_must_be_clean(world: dict[str, Any], pinned_sha: str, tmp_path: Path, break_it: str) -> None:
    home = world["agent"] / "homes/hermes"
    if break_it == "memories":
        (home / "memories" / "MEMORY.md").write_text("remembered\n")
    else:
        (home / ".env").write_text("OPENAI_API_KEY=sk-x\n")
    with pytest.raises(epochs.EpochError):
        _create(world, pinned_sha, tmp_path / "vendor")
    assert not any((tmp_path / "vendor").iterdir())  # the staging directory is gone


def test_creation_refuses_drifted_pins(world: dict[str, Any], pinned_sha: str, tmp_path: Path) -> None:
    raw = json.loads(json.dumps(world["pins_raw"]))
    raw["hermes"]["sha"] = "0" * 40
    with pytest.raises(epochs.EpochError, match="hermes"):
        _create(world, pinned_sha, tmp_path / "vendor", raw)
    raw = json.loads(json.dumps(world["pins_raw"]))
    raw["snapshot"]["smoke"] = "no/such/dir"            # a snapshot directory that is not at the pinned commit
    with pytest.raises(epochs.EpochError, match="no directory no/such/dir"):
        _create(world, pinned_sha, tmp_path / "vendor2", raw)


def test_only_allowed_git_commands_run_on_peer_repos(tmp_path: Path) -> None:
    for args in (("describe", "--tags", "--dirty"), ("status", "--porcelain"), ("fetch",), ("worktree", "add", "x")):
        with pytest.raises(ValueError, match="not an allowed"):
            epochs._git(tmp_path, *args)


def test_ensure_epoch_matches_the_pinned_id(world: dict[str, Any], pinned_sha: str, tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    epoch_id = _create(world, pinned_sha, vendor)
    pins_file = tmp_path / "pins.toml"

    def write_pins(pinned: str) -> None:
        raw = {**world["pins_raw"], "epoch": {"id": pinned}}
        lines = []
        for table, values in raw.items():
            lines.append(f"[{table}]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in values.items() if not isinstance(v, dict)]
            for key, value in values.items():
                if isinstance(value, dict):
                    lines.append(f"[{table}.{key}]")
                    lines += [f"{k} = {json.dumps(v)}" for k, v in value.items()]
        pins_file.write_text("\n".join(lines) + "\n")

    write_pins(epoch_id)
    assert epochs.ensure_epoch(pins_file, vendor) == epoch_id
    shutil.rmtree(vendor / epoch_id)
    assert epochs.ensure_epoch(pins_file, vendor) == epoch_id  # recreated from the pinned commits
    write_pins("000000000000")
    with pytest.raises(epochs.EpochError, match="hash to"):
        epochs.ensure_epoch(pins_file, vendor)


def test_the_repo_pins_describe_this_machine() -> None:
    raw, pins = epochs.load_pins()
    assert pins["claude_code"]["bin"].name == pins["claude_code"]["version"] == "2.1.312"
    assert pins["hermes"]["sha"].startswith("d923a5eb48") and pins["localmelo"]["sha"].startswith("6921912")
    assert pins["data"]["aider_polyglot"].startswith("7e0611e") and pins["data"]["mcpmark"].startswith("cd45b7f")
    assert pins["epoch_id"] and len(pins["epoch_id"]) == epochs.ID_LENGTH
