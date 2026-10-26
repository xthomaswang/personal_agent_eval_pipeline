"""The launch adapters: the agents' shell environment, the MCP-only arm, and the installed MCP server."""

from __future__ import annotations

import json
import os
import select
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from marks import needs_bench

from smoke import adapters, paths

REAL_HERMES_CONFIG = paths.SMOKE_ROOT.parent.parent / "agent" / "homes" / "hermes" / "config.yaml"


@needs_bench
def test_claude_code_gets_only_the_fs_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import harness

    template = tmp_path / "claude-template"
    template.mkdir()
    (template / "settings.json").write_text("{}")
    monkeypatch.setattr(harness, "CLAUDE_HOME", template)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ws = tmp_path / "ws"

    build = adapters.claude_code_mcp_builder(harness._claude_code, ws)
    cmd, env = build("do it", run_dir, "http://127.0.0.1:43210", 40)

    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__fs__*"
    assert "--strict-mcp-config" in cmd
    config = json.loads(cmd[cmd.index("--mcp-config") + 1])
    assert config == {
        "mcpServers": {"fs": {"type": "stdio", "command": str(paths.node_bin()), "args": [str(paths.FS_SERVER_JS), str(ws)]}}
    }
    assert env["ENABLE_TOOL_SEARCH"] == "false"
    assert env["PAB_BASE_URL"] == "http://127.0.0.1:43210"
    assert cmd[cmd.index("-p") + 1] == "do it"


@needs_bench
def test_hermes_config_copy_enables_only_fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml
    from bench import harness

    template = tmp_path / "hermes-template"
    template.mkdir()
    if REAL_HERMES_CONFIG.exists():  # the real config, so the YAML round trip is tested on its structure
        shutil.copyfile(REAL_HERMES_CONFIG, template / "config.yaml")
    else:
        (template / "config.yaml").write_text(
            "providers:\n  omlx:\n    api: http://127.0.0.1:8500/v1\nmcp_servers:\n  other:\n    command: /bin/false\n"
        )
    monkeypatch.setattr(harness, "HERMES_HOME", template)
    monkeypatch.setattr(harness, "UPSTREAM", "http://127.0.0.1:8500")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ws = tmp_path / "ws"
    original = (template / "config.yaml").read_text()

    build = adapters.hermes_mcp_builder(harness._hermes, ws)
    cmd, env = build("do it", run_dir, "http://127.0.0.1:43210", 40)

    config = yaml.safe_load((Path(env["HERMES_HOME"]) / "config.yaml").read_text())
    servers = config["mcp_servers"]
    assert servers["fs"] == {"command": str(paths.node_bin()), "args": [str(paths.FS_SERVER_JS), str(ws)], "enabled": True}
    assert all(entry.get("enabled") is False for name, entry in servers.items() if name != "fs")
    assert config["tools"]["tool_search"]["enabled"] == "off"  # the string, not YAML's boolean false
    assert "http://127.0.0.1:43210/v1" in json.dumps(config)
    assert "8500" not in json.dumps(config.get("providers"))
    assert cmd[cmd.index("-t") + 1] == "fs"
    assert cmd[cmd.index("--query-file") + 1] == str(run_dir / "prompt.md")
    assert "--yolo" not in cmd
    assert (template / "config.yaml").read_text() == original  # the home template is never touched


@needs_bench
def test_builders_are_restored_after_a_run(tmp_path: Path) -> None:
    from bench import harness

    before = dict(harness.BUILDERS)
    with pytest.raises(RuntimeError), adapters.overrides(harness, tmp_path, mcp=True, env_file=tmp_path / "env.sh"):
        assert harness.BUILDERS["claude_code"] is not before["claude_code"]
        assert harness.BUILDERS["localmelo"] is before["localmelo"]
        raise RuntimeError("the run failed")
    assert harness.BUILDERS == before


def test_a_missing_flag_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="--tools"):
        adapters._set_flag(["claude", "-p", "x"], "--tools", "")


class _Client:
    """A minimal MCP stdio client: newline-delimited JSON-RPC."""

    def __init__(self, cmd: list[str]) -> None:
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.next_id = 1

    def send(self, method: str, params: dict[str, Any] | None = None, notify: bool = False) -> dict[str, Any] | None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            message["id"] = self.next_id
            self.next_id += 1
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()
        if notify:
            return None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0.5)
            if not ready:
                continue
            line = self.proc.stdout.readline()
            if not line:
                break
            reply = json.loads(line)
            if reply.get("id") == message["id"]:
                return reply
        raise TimeoutError(f"no reply to {method}")

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()


@pytest.mark.skipif(not paths.FS_SERVER_JS.exists() or not paths.node_bin().exists(), reason="MCP server not installed")
def test_installed_server_lists_its_tools_and_only_the_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    client = _Client([str(paths.node_bin()), str(paths.FS_SERVER_JS), str(ws)])
    try:
        init = client.send(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "harness-smoke", "version": "1"}},
        )
        assert init and init["result"]["serverInfo"]["name"]
        client.send("notifications/initialized", notify=True)
        tools = client.send("tools/list")
        names = {tool["name"] for tool in tools["result"]["tools"]}  # type: ignore[index]
        assert {"list_allowed_directories", "write_file", "list_directory", "move_file"} <= names

        allowed = client.send("tools/call", {"name": "list_allowed_directories", "arguments": {}})
        text = allowed["result"]["content"][0]["text"]  # type: ignore[index]
        assert os.path.realpath(ws) in text

        target = ws / "hello.txt"
        client.send("tools/call", {"name": "write_file", "arguments": {"path": str(target), "content": "hi"}})
        assert target.read_text() == "hi"
        outside = client.send("tools/call", {"name": "write_file", "arguments": {"path": str(tmp_path / "x.txt"), "content": "no"}})
        assert outside["result"].get("isError") is True  # type: ignore[index]
        assert not (tmp_path / "x.txt").exists()
    finally:
        client.close()


def test_env_file_puts_shims_and_the_smoke_venv_first(tmp_path: Path) -> None:
    env_file = adapters.write_env_file(tmp_path / "agent-env.sh")

    assert f'export PATH="{adapters.SHIMS}:{paths.SMOKE_VENV_BIN}:$PATH"' in env_file.read_text()
    env = adapters.agent_env(env_file)
    assert env["PATH"].split(os.pathsep)[:2] == [str(adapters.SHIMS), str(paths.SMOKE_VENV_BIN)]
    assert env["CLAUDE_ENV_FILE"] == str(env_file)


@needs_bench
def test_hermes_sources_the_env_file_in_both_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml
    from bench import harness

    template = tmp_path / "hermes-template"
    template.mkdir()
    (template / "config.yaml").write_text("providers:\n  omlx:\n    api: http://127.0.0.1:8500/v1\n")
    monkeypatch.setattr(harness, "HERMES_HOME", template)
    monkeypatch.setattr(harness, "UPSTREAM", "http://127.0.0.1:8500")
    env_file = tmp_path / "agent-env.sh"
    for mcp in (False, True):
        run_dir = tmp_path / f"run-{mcp}"
        run_dir.mkdir()
        with adapters.overrides(harness, tmp_path / "ws", mcp=mcp, env_file=env_file):
            cmd, env = harness.BUILDERS["hermes"]("do it", run_dir, "http://127.0.0.1:43210", 40)
        config = yaml.safe_load((Path(env["HERMES_HOME"]) / "config.yaml").read_text())
        assert config["terminal"]["shell_init_files"] == [str(env_file)]
        assert cmd[cmd.index("-t") + 1] == ("fs" if mcp else harness.HERMES_TOOLSETS)


@needs_bench
def test_builtin_arm_leaves_claude_code_untouched(tmp_path: Path) -> None:
    from bench import harness

    original = harness.BUILDERS["claude_code"]
    with adapters.overrides(harness, tmp_path, mcp=False, env_file=tmp_path / "env.sh"):
        assert harness.BUILDERS["claude_code"] is original
        assert harness.BUILDERS["hermes"] is not original


HERMES_PYTHON = paths.SMOKE_ROOT.parent.parent / "agent" / ".venv" / "bin" / "python"
HERMES_REPO = paths.SMOKE_ROOT.parent.parent / "agent" / "hermes-agent"
PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
from tools.environments.local import LocalEnvironment
env = LocalEnvironment(cwd=sys.argv[2], timeout=30)
print(env.execute("command -v python3; command -v python; command -v pytest; command -v pip3")["output"])
env.cleanup()
"""


@pytest.mark.skipif(not HERMES_PYTHON.exists() or not HERMES_REPO.is_dir(), reason="Hermes is not installed")
def test_hermes_terminal_finds_the_smoke_python(tmp_path: Path) -> None:
    """Offline, through Hermes's own terminal code (no model): with the env file in
    terminal.shell_init_files, its login-shell snapshot resolves python3 to the smoke venv."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    env_file = adapters.write_env_file(tmp_path / "agent-env.sh")
    (home / "config.yaml").write_text(f"terminal:\n  shell_init_files: [{json.dumps(str(env_file))}]\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    env = {
        "HOME": os.environ["HOME"],
        "USER": os.environ.get("USER", ""),
        "SHELL": "/bin/zsh",
        "TMPDIR": str(tmp_path),
        "HERMES_HOME": str(home),
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "PATH": adapters.agent_env(env_file)["PATH"],
    }
    proc = subprocess.run(
        [str(HERMES_PYTHON), "-c", PROBE, str(HERMES_REPO), str(ws)], env=env, capture_output=True, text=True, timeout=120
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("/")]
    assert lines[:4] == [
        str(paths.SMOKE_VENV_BIN / "python3"),
        str(paths.SMOKE_VENV_BIN / "python"),
        str(paths.SMOKE_VENV_BIN / "pytest"),
        str(adapters.SHIMS / "pip3"),
    ], proc.stdout + proc.stderr


CLAUDE_CODE_PROBE = 'for c in python3 python pytest pip3; do printf "%s=%s\\n" "$c" "$(command -v "$c")"; done'


def _claude_code_bash(snapshot: Path, env_file: Path | None, env: dict[str, str]) -> dict[str, str]:
    """Run CLAUDE_CODE_PROBE the way Claude Code 2.1.302's Bash tool runs a command (read from the binary):
    `$SHELL -c -l "source <snapshot> 2>/dev/null || true && <CLAUDE_ENV_FILE contents>\\n: && eval <command>"`."""
    parts = [f"source {shlex.quote(str(snapshot))} 2>/dev/null || true"]
    if env_file is not None:
        parts.append(f"{env_file.read_text().strip()}\n:")
    parts.append(f"eval {shlex.quote(CLAUDE_CODE_PROBE)}")
    proc = subprocess.run(
        ["/bin/zsh", "-c", "-l", " && ".join(parts)], env=env, capture_output=True, text=True, timeout=60, check=True
    )
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


@pytest.mark.skipif(not Path("/bin/zsh").exists(), reason="needs /bin/zsh")
def test_claude_code_bash_prefix_finds_the_smoke_python(tmp_path: Path) -> None:
    """Shell mechanics only: the command layout is emulated, and the snapshot is stood in for by the PATH
    a login shell builds from the agent's inherited PATH. That Claude Code splices CLAUDE_ENV_FILE in at
    all (it skips it only for remote-tools calls) was read from the binary, not exercised here."""
    env_file = adapters.write_env_file(tmp_path / "agent-env.sh")
    env = {**os.environ, **adapters.agent_env(env_file)}
    login_path = subprocess.run(
        ["/bin/zsh", "-l", "-c", 'printf %s "$PATH"'], env=env, capture_output=True, text=True, timeout=60, check=True
    ).stdout
    snapshot = tmp_path / "snapshot.sh"
    snapshot.write_text(f"export PATH={shlex.quote(login_path)}\n")

    without = _claude_code_bash(snapshot, None, env)
    found = _claude_code_bash(snapshot, env_file, env)

    # Without the env file, path_helper's order wins wherever /usr/bin has the command
    assert without["python3"] != str(paths.SMOKE_VENV_BIN / "python3")
    assert found == {
        "python3": str(paths.SMOKE_VENV_BIN / "python3"),
        "python": str(paths.SMOKE_VENV_BIN / "python"),
        "pytest": str(paths.SMOKE_VENV_BIN / "pytest"),
        "pip3": str(adapters.SHIMS / "pip3"),
    }
