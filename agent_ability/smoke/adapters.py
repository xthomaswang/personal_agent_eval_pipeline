"""What the suite changes in bench.harness's harness launch, and why.

1. The agents' shell environment. bench.harness puts extra_env's PATH (the smoke .venv first) into
   each harness's environment, but Claude Code and Hermes run tool commands in macOS login shells,
   whose path_helper moves /usr/bin ahead of inherited PATH entries. Measured offline with Hermes's
   own terminal code: `python3` became /usr/bin/python3 (3.9, no pytest) and Hermes even put its
   own venv's `python` (3.14) first. So each run writes agent-env.sh, which re-prepends the shims
   and the smoke .venv, and both harnesses source it through their documented hooks: Claude Code's
   CLAUDE_ENV_FILE (sourced before each Bash command) and Hermes's terminal.shell_init_files
   (sourced into its login-shell snapshot). localmelo runs commands with /bin/sh -c, which keeps
   the inherited PATH. The shims make `pip`/`pip3` fail, so an agent cannot install into the
   user's system Python while it tries to run the tests.
   Claude Code 2.1.302, as read from its binary: the Bash tool runs `$SHELL -c -l "source
   <snapshot> 2>/dev/null || true && <CLAUDE_ENV_FILE contents>\n: && eval <command>"`, so the
   file's PATH comes after the snapshot's and wins. The only case that leaves the file out is a
   remote-tools call whose constraints set scrubCredentialEnv; `claude -p` makes no such calls,
   and CLAUDE_CODE_SUBPROCESS_ENV_SCRUB does not gate it.
2. The MCP-only arm (--mcp-arm): Claude Code and Hermes with the filesystem MCP server as their
   only tools.
   - Claude Code: `--tools ""`, `--allowedTools "mcp__fs__*"`, `--mcp-config` naming one stdio
     server `fs`; ENABLE_TOOL_SEARCH=false keeps the MCP tools loaded upfront.
   - Hermes: an enabled `mcp_servers.fs` entry in the per-run config.yaml copy, every other server
     disabled, `tools.tool_search.enabled: off`, and `-t fs`. Hermes's spawn filter matches -t names
     against mcp_servers keys, and `fs` is registered as an alias of the `mcp-fs` toolset, so
     `-t mcp-fs` would start no server. Without the tool_search setting, Hermes would replace the fs
     tools with its tool_search/tool_describe/tool_call bridge.
   Both launch the server as absolute node + index.js paths: Hermes replaces a bare `node`/`npx`.

bench.harness has no hooks for either, so for the length of one run BUILDERS[name] is swapped for
a wrapper around the original builder. The wrapper keeps everything the original sets up (home
copy, base URL, budgets) and the harness name stays the same, so bench.harness's native step-count
parsing still applies.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from smoke import paths

SERVER = "fs"
SERVER_PACKAGE = "@modelcontextprotocol/server-filesystem@2025.12.18"
MCP_HARNESSES = ("claude_code", "hermes")
SHIMS = paths.SMOKE_ROOT / "shims"

Builder = Callable[[str, Path, str, int], tuple[list[str], dict[str, str]]]


# --- The agents' shell environment ---


def write_env_file(path: Path) -> Path:
    path.write_text(
        "# Written by harness-smoke. Agent shell commands source this file (Claude Code: CLAUDE_ENV_FILE;\n"
        "# Hermes: terminal.shell_init_files) because macOS login shells reorder an inherited PATH.\n"
        f'export PATH="{SHIMS}:{paths.SMOKE_VENV_BIN}:$PATH"\n'
    )
    return path


def agent_env(env_file: Path) -> dict[str, str]:
    """bench.harness's extra_env: the same PATH and hooks for every harness."""
    return {
        "PATH": os.pathsep.join([str(SHIMS), str(paths.SMOKE_VENV_BIN), os.environ.get("PATH", "")]),
        "CLAUDE_ENV_FILE": str(env_file),
    }


def _edit_hermes_config(env: dict[str, str], edit: Callable[[dict[str, Any]], None]) -> None:
    import yaml  # personal-agent-bench's environment provides PyYAML

    home = env.get("HERMES_HOME")
    if not home:
        raise RuntimeError("expected HERMES_HOME from the Hermes builder")
    config_path = Path(home) / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    edit(config)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def hermes_env_builder(original: Builder, env_file: Path) -> Builder:
    def build(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        cmd, env = original(prompt, run_dir, base_url, max_turns)

        def edit(config: dict[str, Any]) -> None:
            terminal = config.get("terminal") or {}
            terminal["shell_init_files"] = [str(env_file)]
            config["terminal"] = terminal

        _edit_hermes_config(env, edit)
        return cmd, env

    return build


# --- The MCP-only arm ---


def server_spec(ws: Path) -> dict[str, Any]:
    return {"command": str(paths.node_bin()), "args": [str(paths.FS_SERVER_JS), str(ws)]}


def _set_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    """Replace the value that follows `flag`; fail loudly if the builder stopped passing it."""
    if flag not in cmd:
        raise RuntimeError(f"expected {flag} in the harness command line: {cmd}")
    out = list(cmd)
    out[out.index(flag) + 1] = value
    return out


def claude_code_mcp_builder(original: Builder, ws: Path) -> Builder:
    def build(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        cmd, env = original(prompt, run_dir, base_url, max_turns)
        config = {"mcpServers": {SERVER: {"type": "stdio", **server_spec(ws)}}}
        cmd = _set_flag(cmd, "--tools", "")
        cmd = _set_flag(cmd, "--allowedTools", f"mcp__{SERVER}__*")
        cmd = _set_flag(cmd, "--mcp-config", json.dumps(config))
        if "--strict-mcp-config" not in cmd:
            raise RuntimeError("expected --strict-mcp-config in the Claude Code command line")
        return cmd, {**env, "ENABLE_TOOL_SEARCH": "false"}

    return build


def hermes_mcp_builder(original: Builder, ws: Path) -> Builder:
    def build(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        cmd, env = original(prompt, run_dir, base_url, max_turns)

        def edit(config: dict[str, Any]) -> None:
            servers = config.get("mcp_servers") or {}
            for entry in servers.values():
                if isinstance(entry, dict):
                    entry["enabled"] = False
            servers[SERVER] = {**server_spec(ws), "enabled": True}
            config["mcp_servers"] = servers
            # Hermes hides mcp-* toolsets behind its tool_search bridge by default ("auto"); "off" puts the
            # fs tools themselves in every request, as ENABLE_TOOL_SEARCH=false does for Claude Code
            tools = config.get("tools") or {}
            tools["tool_search"] = {**(tools.get("tool_search") or {}), "enabled": "off"}
            config["tools"] = tools

        _edit_hermes_config(env, edit)
        return _set_flag(cmd, "-t", SERVER), env

    return build


@contextmanager
def overrides(harness: ModuleType, ws: Path, *, mcp: bool, env_file: Path) -> Iterator[None]:
    """Swap in the wrapped builders for one run, then restore every original."""
    saved = dict(harness.BUILDERS)
    try:
        hermes = hermes_env_builder(saved["hermes"], env_file)
        claude_code = saved["claude_code"]
        if mcp:
            claude_code = claude_code_mcp_builder(claude_code, ws)
            hermes = hermes_mcp_builder(hermes, ws)
        harness.BUILDERS["claude_code"] = claude_code
        harness.BUILDERS["hermes"] = hermes
        yield
    finally:
        harness.BUILDERS.clear()
        harness.BUILDERS.update(saved)


def mcp_problems() -> list[str]:
    """What is missing for the MCP arm to run."""
    found = []
    node = paths.node_bin()
    if not node.exists():
        found.append(f"node {node} is missing (nvm install, per mcp/.nvmrc)")
    if not paths.FS_SERVER_JS.exists():
        found.append(
            f"{paths.FS_SERVER_JS} is missing; install it with "
            f"npm install --prefix mcp --ignore-scripts --no-audit --no-fund {SERVER_PACKAGE}"
        )
    return found
