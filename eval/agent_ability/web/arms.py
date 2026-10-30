"""The arms of the web suite, per harness: the MCP config of the tap, the prompt, and what each harness is told.

Every arm reaches the web through one MCP server, named ``webtap`` (a name that no Hermes toolset has):
``web/tap.py`` (see its docstring). Only the tap's mode differs:

- ``D``: ``--mode dom --browser-url <the run's Chrome>``. chrome-devtools-mcp's text tools.
- ``S`` (S-cdp): ``--mode screen --browser-url <the run's Chrome>``. Screenshots and coordinate input through CDP.
- ``S-cua``: ``--mode cua --chrome-pid <the run's headed Chrome>``. The same five tools over cua-driver, with OS-level
  input; the window must be on the Space on screen (``--cua-space``).
- ``C``: ``--mode none``. No tools, and no Chrome.

Every arm logs its calls to ``<run_dir>/tap.jsonl``. The tap runs under ``sys.executable`` (it uses the standard library
only). Its environment is set in the MCP config of each harness, because localmelo and Hermes hand an MCP server a
reduced environment (what its config lists, plus a short list of safe variables). ``PATH`` holds the directory of the
Node binary named by ``mcp/.nvmrc`` (found under nvm at run time) and ``/usr/bin:/bin``; chrome-devtools-mcp's usage
statistics and update checks are off. S-cua adds the real ``HOME``, because cua-driver finds its daemon through it
and bench.harness gives localmelo an empty HOME of its own.

Per harness:

- Claude Code: ``--tools ""``, ``--allowedTools "mcp__webtap__*"``, ``--mcp-config`` with the tap, and
  ``ENABLE_TOOL_SEARCH=false`` so the tools are loaded upfront. It also gets a per-run copy of claude-code.env with
  ``CLAUDE_BIN`` and the model variables replaced, and a ``"model"`` in the copied home's settings.json (see below).
- Hermes: an enabled ``mcp_servers.webtap`` entry in the per-run config.yaml copy, every other server disabled,
  ``tools.tool_search.enabled: off`` and ``-t webtap``. The screenshot arms also set ``agent.image_input_mode: native``
  and ``model.supports_vision: true``, which make Hermes hand a tool's screenshot to the model as an image.
- localmelo: flags for ``bench.harness.run(harness_args=...)``, which puts them before the ``--`` that precedes the
  prompt: ``--mcp <mcp.json> --mcp-only -o json``, and ``-X agent.tool_images=on`` for the screenshot arms, which sends
  the pictures a tool returns to the model. A localmelo-only A/B adds its own flags after these (``--localmelo-args``;
  ``check_localmelo_extra`` refuses the ones the arms set) and may pin another commit (``--localmelo-sha``).

The model id reaches the three harnesses through ``bench.harness.MODEL`` (Hermes's config and localmelo's
``--chat-model``; run.py sets it for the study) and, for Claude Code, through the per-run env and settings copies.

The builders of bench.harness, and its ``CLAUDE_ENV``, are swapped for one run and restored afterwards (``overrides``),
as smoke.adapters does.
The localmelo checkout is pinned like e3's: ``localmelo_problems`` refuses another commit or a dirty tree.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from smoke import adapters
from web import tap as tap_module

HERE = Path(__file__).resolve().parent
AGENT_ABILITY = HERE.parent
TAP = HERE / "tap.py"
MCP_DIR = HERE / "mcp"
CHROME_DEVTOOLS_MCP = MCP_DIR / "node_modules" / ".bin" / "chrome-devtools-mcp"

# Hermes has toolsets named web and browser (toolsets.py), and `-t <name>` of an MCP server that shares a name would
# also select the built-in tools. Nothing in Hermes is called webtap.
SERVER = "webtap"
ARMS = ("D", "S", "C", "S-cua")
DEFAULT_ARMS = ("D", "S", "C")  # S-cua runs only when it is named
HARNESSES = ("claude_code", "hermes", "localmelo")
TAP_MODES = {"D": "dom", "S": "screen", "C": "none", "S-cua": "cua"}
NEEDS_CHROME = {"D": True, "S": True, "C": False, "S-cua": True}
NEEDS_NODE = frozenset({"D", "S"})  # the arms whose tap starts chrome-devtools-mcp
HEADED = frozenset({"S-cua"})  # Chrome with a window, on the Space on screen
IMAGE_ARMS = frozenset({"S", "S-cua"})  # the arms whose tool results are pictures

# The model every harness is told to use, and the Claude Code binary the capability reference run pins
DEFAULT_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
CLAUDE_BIN = AGENT_ABILITY.parent / "backend" / "vendor" / "claude-code" / "2.1.312"

# The tap's environment: node's directory is added at run time. chrome-devtools-mcp reads these two itself.
SYSTEM_PATH = ("/usr/bin", "/bin")
TAP_ENV = {"CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1", "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1"}

# localmelo: the bench's own clone, pinned like E3's
LOCALMELO = AGENT_ABILITY / "e3" / "vendor" / "localmelo-a9fb42b"
LOCALMELO_SHA = "a9fb42b6633f6da8a887b3a65bdb092f8f2c4605"
LOCALMELO_COMMON_ARGS = ("--mcp-only", "-o", "json")  # --mcp <file> goes in front of these
LOCALMELO_IMAGE_ARGS = ("-X", "agent.tool_images=on")  # the screenshot arms
# What the arms and bench.harness set for localmelo, in every spelling its parser takes (melo's __main__ lists them):
# the MCP config, the output format, the model and the endpoint. --localmelo-args may not repeat them.
LOCALMELO_OWN_LONG = (
    "--mcp", "--mcp-config", "--mcp-only", "--output-format", "--output",
    "--chat-model", "--model", "--base-url", "--url",
)  # fmt: skip
LOCALMELO_OWN_SHORT = ("-o", "-m", "-u")

Builder = adapters.Builder


# --- prompts (DESIGN.md, section 5) ---

OPEN_PARAGRAPH = (
    "A Google Chrome window is open on Google Search. Use the browser tools to search the web and\n"
    "answer the question below, checking the answer on the pages you find. Only read: do not sign in,\n"
    "post, buy or change settings."
)
CLOSED_PARAGRAPH = "Answer the question below from your own knowledge. You have no tools."
ANSWER_RULES = (
    "End your reply with one line: FINAL ANSWER: <answer>\n"
    "Your final answer should be a number OR as few words as possible OR a comma separated list of\n"
    "numbers and/or strings. If you are asked for a number, don't use comma to write your number\n"
    "neither use units such as $ or percent sign unless specified otherwise. Write a date as day,\n"
    "month name and year. If you cannot find the answer, write FINAL ANSWER: unknown"
)


def prompt(arm: str, problem: str) -> str:
    """The task text of an arm. D, S and S-cua share one text, so a difference between them is not the prompt's."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    opening = CLOSED_PARAGRAPH if arm == "C" else OPEN_PARAGRAPH
    return f"{opening}\n\nQuestion: {problem}\n\n{ANSWER_RULES}"


# --- the tap as an MCP server ---


def nvm_dir() -> Path:
    return Path(os.environ.get("NVM_DIR") or Path.home() / ".nvm").expanduser()


def node_version() -> str:
    """The Node version of web/mcp/.nvmrc, with its leading ``v``."""
    version = (MCP_DIR / ".nvmrc").read_text().strip()
    return version if version.startswith("v") else f"v{version}"


def node_dir() -> Path:
    """The directory of the nvm Node that mcp/.nvmrc names. The tap's child is a ``#!/usr/bin/env node`` script,
    and the harnesses give an MCP server no nvm shell, so this goes into the tap's PATH."""
    return nvm_dir() / "versions" / "node" / node_version() / "bin"


def tap_path() -> str:
    return os.pathsep.join([str(node_dir()), *SYSTEM_PATH])


def tap_env(arm: str | None = None) -> dict[str, str]:
    """The tap's environment. S-cua also gets the real HOME: cua-driver finds its daemon through it, and bench.harness
    gives localmelo an empty HOME of its own (the daemon then reads as not running)."""
    env = {"PATH": tap_path(), **TAP_ENV}
    if arm == "S-cua":
        env["HOME"] = str(Path.home())
    return env


def tap_args(
    arm: str,
    run_dir: Path,
    browser_url: str | None,
    *,
    chrome_pid: int | None = None,
    cua_bin: str | Path | None = None,
) -> list[str]:
    """The tap's command line after the interpreter."""
    if arm not in TAP_MODES:
        raise ValueError(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    args = [str(TAP), "--mode", TAP_MODES[arm], "--log", str(run_dir / "tap.jsonl")]
    if arm == "S-cua":
        if not chrome_pid:
            raise ValueError("arm S-cua needs the pid of the run's headed Chrome (chrome_pid)")
        args += ["--chrome-pid", str(chrome_pid)]
        if cua_bin:
            args += ["--cua-bin", str(cua_bin)]
    elif NEEDS_CHROME[arm]:
        if not browser_url:
            raise ValueError(f"arm {arm} needs the run's Chrome (browser_url)")
        args += ["--browser-url", browser_url]
    return args


def server_spec(
    arm: str,
    run_dir: Path,
    browser_url: str | None,
    *,
    chrome_pid: int | None = None,
    cua_bin: str | Path | None = None,
) -> dict[str, Any]:
    """The ``webtap`` server of an arm: command, args and env, as every harness's MCP config takes them."""
    args = tap_args(arm, run_dir, browser_url, chrome_pid=chrome_pid, cua_bin=cua_bin)
    return {"command": sys.executable, "args": args, "env": tap_env(arm)}


def mcp_config(arm: str, run_dir: Path, browser_url: str | None, **browser: Any) -> dict[str, Any]:
    """An ``mcpServers`` document with the tap as its one server (Claude Code's ``--mcp-config`` and localmelo's file)."""
    return {"mcpServers": {SERVER: {"type": "stdio", **server_spec(arm, run_dir, browser_url, **browser)}}}


def check_localmelo_extra(flags: Sequence[str]) -> tuple[str, ...]:
    """The extra localmelo flags of a study, or a ``ValueError`` naming one that collides with what the arms set.

    A long flag collides when it is one of ``LOCALMELO_OWN_LONG``, with or without ``=value``, or an abbreviation of one
    (argparse takes ``--chat`` for ``--chat-model``); a short one when it starts with ``-o``, ``-m`` or ``-u``
    (``-ojson`` is ``-o json``). A bare ``--`` would end the options before the prompt that bench.harness puts after
    its own.
    """
    for flag in flags:
        if flag == "--":
            raise ValueError("localmelo extra flags cannot contain `--`: bench.harness puts it before the task text")
        own = None
        if flag.startswith("--"):
            name = flag.split("=", 1)[0]
            own = next((o for o in LOCALMELO_OWN_LONG if o.startswith(name)), None)
        elif flag[:2] in LOCALMELO_OWN_SHORT:
            own = flag[:2]
        if own:
            raise ValueError(f"localmelo extra flag {flag!r} collides with {own}, which the arms or bench.harness set")
    return tuple(flags)


def parse_localmelo_extra(text: str) -> tuple[str, ...]:
    """``--localmelo-args``: shell-quoted flags, split like a shell would and checked."""
    return check_localmelo_extra(shlex.split(text))


def localmelo_args(arm: str, mcp_json: Path, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """localmelo's flags for ``harness.run(harness_args=...)``: the arm's own, then ``extra`` (``--localmelo-args``)."""
    args = ("--mcp", str(mcp_json), *LOCALMELO_COMMON_ARGS)
    if arm in IMAGE_ARMS:
        args += LOCALMELO_IMAGE_ARGS
    return args + tuple(extra)


@dataclass(frozen=True)
class Plan:
    """What one run of a harness on an arm is given."""

    harness: str
    arm: str
    server: dict[str, Any]
    harness_args: tuple[str, ...] = ()  # localmelo's flags; empty for the other harnesses
    mcp_json: Path | None = None  # localmelo's config file


# --- Claude Code's per-run configuration ---
#
# bench.harness._claude_code sources agent/configs/claude-code/claude-code.env, which pins a Claude Code binary that an
# auto-update can delete and the model of an earlier study, and it copies a home whose settings.json names that model
# too. The capability runner solves this with a per-run copy of both; the functions below are the minimum of
# eval/backend/src/modeleval/capability/runner.py (claude_values, patch_claude_env, shell_env, claude_env_problems,
# patch_claude_settings) that this study needs, ported because modeleval cannot be imported from this environment.
# The auto-compact window is left as the file has it.

CLAUDE_MODEL_VARS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)
# Must survive in the copy, exactly: they keep the benchmarked Claude Code off the user's account, memory and
# CLAUDE.md files and off the network
CLAUDE_SAFETY = {
    "ANTHROPIC_API_KEY": "",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}
CLAUDE_NONEMPTY = ("CLAUDE_CONFIG_DIR", "ANTHROPIC_AUTH_TOKEN")
_ASSIGNMENT = re.compile(
    r"""^(?P<lead>\s*(?:export\s+)?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>"[^"]*"|'[^']*'|[^\s#;]*)(?P<rest>.*)$"""
)
_BARE = re.compile(r"^[A-Za-z0-9._/:-]+$")


class Refused(RuntimeError):
    """A per-run configuration check failed."""


def claude_values(model: str, claude_bin: Path) -> dict[str, str]:
    """The variables the per-run claude-code.env replaces."""
    return {**dict.fromkeys(CLAUDE_MODEL_VARS, model), "CLAUDE_BIN": str(claude_bin)}


def patch_claude_env(text: str, values: Mapping[str, str]) -> str:
    """A copy of claude-code.env with only the named variables' values replaced, by name.

    A line keeps its ``export``, spacing and trailing comment; a value that is already right keeps its spelling.
    Refuses when a named variable has no assignment.
    """
    seen: dict[str, int] = dict.fromkeys(values, 0)
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        match = _ASSIGNMENT.match(body)
        if match and match["name"] in values:
            name, old = match["name"], match["value"]
            seen[name] += 1
            new = values[name]
            if old.strip("\"'") != new:
                token = new if (_BARE.match(new) and not old.startswith(("'", '"'))) else f'"{new}"'
                body = f"{match['lead']}{name}={token}{match['rest']}"
            line = body + line[len(line.rstrip("\r\n")) :]
        out.append(line)
    missing = [name for name, count in seen.items() if not count]
    if missing:
        raise Refused(f"claude-code.env has no assignment of {', '.join(missing)}")
    return "".join(out)


def shell_env(env_file: Path) -> dict[str, str]:
    """What bench.harness's launcher shell holds after sourcing env_file: the exported variables, plus CLAUDE_BIN (a plain
    shell variable that the launcher execs) as ``__SHELL__CLAUDE_BIN``."""
    script = '. "$1" >/dev/null 2>&1; env -0; printf "__SHELL__CLAUDE_BIN=%s\\0" "$CLAUDE_BIN"'
    proc = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            f"HOME={Path.home()}",
            "PATH=/usr/bin:/bin",
            "/bin/sh",
            "-c",
            script,
            "sh",
            str(env_file),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise Refused(f"sourcing {env_file.name} failed: {proc.stderr.decode(errors='replace')[:200]}")
    out = {}
    for item in proc.stdout.decode(errors="replace").split("\0"):
        if "=" in item:
            name, value = item.split("=", 1)
            out[name] = value
    return out


def claude_env_problems(original: str, patched: str, env: Mapping[str, str], model: str, claude_bin: Path) -> list[str]:
    """Everything wrong with a patched copy: other lines changed, safety variables lost, values off."""
    found = []
    old_lines, new_lines = original.splitlines(), patched.splitlines()
    if len(old_lines) != len(new_lines):
        found.append("the copy has a different number of lines")
    replaced = {*CLAUDE_MODEL_VARS, "CLAUDE_BIN"}
    for old, new in zip(old_lines, new_lines, strict=False):
        match = _ASSIGNMENT.match(old)
        if old != new and not (match and match["name"] in replaced):
            found.append(f"an unnamed line changed: {old!r}")
    for name, value in CLAUDE_SAFETY.items():
        given = env.get(name)
        if given != value:
            shown = "a non-empty value" if name == "ANTHROPIC_API_KEY" and given else repr(given)  # never echo a key
            found.append(f"{name} must be exported as {value!r}, the copy gives {shown}")
    for name in CLAUDE_NONEMPTY:
        if not env.get(name):
            found.append(f"{name} must be exported and non-empty")
    for name in CLAUDE_MODEL_VARS:
        if env.get(name) != model:
            found.append(f"{name} is {env.get(name)!r}, not {model!r}")
    if env.get("__SHELL__CLAUDE_BIN") != str(claude_bin):
        found.append(f"CLAUDE_BIN is {env.get('__SHELL__CLAUDE_BIN')!r}, not {str(claude_bin)!r}")
    return found


def patch_claude_settings(text: str, model: str) -> str:
    """settings.json with only the value of ``"model"`` replaced."""
    pattern = re.compile(r'("model"\s*:\s*)"(?:[^"\\]|\\.)*"')
    if len(pattern.findall(text)) != 1:
        raise Refused('settings.json must have exactly one "model" entry')
    patched = pattern.sub(lambda m: m.group(1) + json.dumps(model), text)
    before, after = json.loads(text), json.loads(patched)
    if after.get("model") != model or {**before, "model": model} != after:
        raise Refused("settings.json: replacing the model changed something else")
    return patched


def write_claude_env(template: Path, dest: Path, model: str, claude_bin: Path) -> Path:
    """The per-run claude-code.env, checked by sourcing it the way the launcher does. Raises ``Refused``."""
    original = template.read_text()
    patched = patch_claude_env(original, claude_values(model, claude_bin))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(patched)
    problems = claude_env_problems(original, patched, shell_env(dest), model, claude_bin)
    if problems:
        raise Refused("per-run claude-code.env: " + "; ".join(problems))
    return dest


def claude_problems(env_template: Path, home_template: Path, model: str, claude_bin: Path) -> list[str]:
    """What would make Claude Code run the wrong binary or model, or lose a safety variable, as messages."""
    found = []
    if not (claude_bin.is_file() and os.access(claude_bin, os.X_OK)):
        found.append(f"Claude Code {claude_bin} is missing or not executable (--claude-bin)")
    try:
        with tempfile.TemporaryDirectory(prefix="web-claude-env-") as tmp:
            write_claude_env(env_template, Path(tmp) / "claude-code.env", model, claude_bin)
        patch_claude_settings((home_template / "settings.json").read_text(), model)
    except (Refused, OSError, ValueError, subprocess.SubprocessError) as exc:
        found.append(str(exc))
    return found


# --- cua-driver (S-cua) ---


class CuaError(RuntimeError):
    """cua-driver could not be asked."""


def resolve_cua_bin() -> str:
    """The cua-driver binary, found as the tap finds it (``$WEB_TAP_CUA_BIN``, else the app with its bundle id)."""
    return os.environ.get("WEB_TAP_CUA_BIN") or tap_module.default_cua_bin()


def cua_status_problem(cua_bin: str | Path) -> str | None:
    """None when ``cua-driver status`` reports the daemon running."""
    try:
        proc = subprocess.run([str(cua_bin), "status"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"cannot run {cua_bin} status: {exc}"
    if proc.returncode != 0 or "daemon is running" not in proc.stdout.lower():
        return f"the cua-driver daemon is not running ({(proc.stdout or proc.stderr).strip()[:100]!r})"
    return None


def cua_current_space(cua_bin: str | Path) -> int:
    """The Space on screen, from ``cua-driver call list_windows '{}'`` (read-only; only ``current_space_id`` is kept)."""
    try:
        proc = subprocess.run(
            [str(cua_bin), "call", "list_windows", "{}"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CuaError(f"cannot run {cua_bin} call list_windows: {exc}") from exc
    if proc.returncode != 0:
        raise CuaError(f"list_windows failed: {(proc.stderr or proc.stdout).strip()[:100]!r}")
    try:
        data = json.loads(proc.stdout)
    except ValueError as exc:
        raise CuaError("list_windows gave no JSON") from exc
    if isinstance(data, dict) and isinstance(data.get("structuredContent"), dict):
        data = data["structuredContent"]
    space = data.get("current_space_id") if isinstance(data, dict) else None
    if not isinstance(space, int) or isinstance(space, bool):
        raise CuaError("list_windows gave no current_space_id")
    return space


# --- the builders ---


def claude_code_builder(original: Builder, server: dict[str, Any], model: str) -> Builder:
    def build(prompt_text: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        cmd, env = original(prompt_text, run_dir, base_url, max_turns)
        config = {"mcpServers": {SERVER: {"type": "stdio", **server}}}
        cmd = adapters._set_flag(cmd, "--tools", "")
        cmd = adapters._set_flag(cmd, "--allowedTools", f"mcp__{SERVER}__*")
        cmd = adapters._set_flag(cmd, "--mcp-config", json.dumps(config))
        if "--strict-mcp-config" not in cmd:
            raise RuntimeError("expected --strict-mcp-config in the Claude Code command line")
        if "PAB_CONFIG_DIR" not in env:
            raise RuntimeError("expected PAB_CONFIG_DIR from the Claude Code builder")
        settings = Path(env["PAB_CONFIG_DIR"]) / "settings.json"  # the copy of the home, made by the original builder
        settings.write_text(patch_claude_settings(settings.read_text(), model))
        return cmd, {**env, "ENABLE_TOOL_SEARCH": "false"}

    return build


def hermes_builder(original: Builder, server: dict[str, Any], *, images: bool) -> Builder:
    def build(prompt_text: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        cmd, env = original(prompt_text, run_dir, base_url, max_turns)

        def edit(config: dict[str, Any]) -> None:
            servers = config.get("mcp_servers") or {}
            for entry in servers.values():
                if isinstance(entry, dict):
                    entry["enabled"] = False
            servers[SERVER] = {**server, "enabled": True}
            config["mcp_servers"] = servers
            # "off" puts the server's tools in every request; the default would hide them behind a search bridge
            tools = config.get("tools") or {}
            tools["tool_search"] = {**(tools.get("tool_search") or {}), "enabled": "off"}
            config["tools"] = tools
            if images:
                agent = config.get("agent") or {}
                agent["image_input_mode"] = "native"
                config["agent"] = agent
                model = config.get("model") or {}
                model["supports_vision"] = True
                config["model"] = model

        adapters._edit_hermes_config(env, edit)
        return adapters._set_flag(cmd, "-t", SERVER), env

    return build


@contextmanager
def overrides(
    harness_module: ModuleType,
    harness: str,
    arm: str,
    *,
    run_dir: Path,
    browser_url: str | None = None,
    chrome_pid: int | None = None,
    cua_bin: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    claude_bin: Path = CLAUDE_BIN,
    localmelo_extra: Sequence[str] = (),
) -> Iterator[Plan]:
    """Swap in the arm's builder for one run, then restore every original. Yields the run's ``Plan``.

    Claude Code and Hermes get a wrapped builder; Claude Code also gets ``CLAUDE_ENV`` pointed at a per-run copy of
    claude-code.env (``<run_dir>/claude-code.env``, with the binary and the model replaced and checked) and a home
    whose settings.json names ``model``. localmelo's builder stays as it is: its flags are the plan's ``harness_args``
    (the arm's, then ``localmelo_extra``; no other harness gets them), and its MCP config is written to
    ``<run_dir>/mcp.json``. The model id itself reaches Hermes and localmelo through ``harness_module.MODEL``, which
    run.py sets for the study.
    """
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness {harness!r}; choose from {', '.join(HARNESSES)}")
    server = server_spec(arm, run_dir, browser_url, chrome_pid=chrome_pid, cua_bin=cua_bin)
    saved = dict(harness_module.BUILDERS)
    saved_env = getattr(harness_module, "CLAUDE_ENV", None)
    try:
        if harness == "claude_code":
            env_file = write_claude_env(Path(saved_env), run_dir / "claude-code.env", model, claude_bin)
            harness_module.CLAUDE_ENV = env_file
            harness_module.BUILDERS["claude_code"] = claude_code_builder(saved["claude_code"], server, model)
            plan = Plan(harness, arm, server)
        elif harness == "hermes":
            harness_module.BUILDERS["hermes"] = hermes_builder(saved["hermes"], server, images=arm in IMAGE_ARMS)
            plan = Plan(harness, arm, server)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            mcp_json = run_dir / "mcp.json"
            config = mcp_config(arm, run_dir, browser_url, chrome_pid=chrome_pid, cua_bin=cua_bin)
            mcp_json.write_text(json.dumps(config, indent=2) + "\n")
            plan = Plan(harness, arm, server, localmelo_args(arm, mcp_json, localmelo_extra), mcp_json)
        yield plan
    finally:
        harness_module.BUILDERS.clear()
        harness_module.BUILDERS.update(saved)
        if saved_env is not None:
            harness_module.CLAUDE_ENV = saved_env


# --- what must be in place ---


def _git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def localmelo_sha(localmelo: Path) -> str:
    return _git(localmelo, "rev-parse", "HEAD")


def localmelo_problems(localmelo: Path, expect_sha: str = LOCALMELO_SHA) -> list[str]:
    """A localmelo checkout other than ``expect_sha`` (the pinned commit, or a ``--localmelo-sha``), or one with
    uncommitted changes, makes the study invalid."""
    if not re.fullmatch(r"[0-9a-f]{7,40}", expect_sha):
        return [f"--localmelo-sha {expect_sha!r} is not 7 to 40 lowercase hex digits"]
    sha = localmelo_sha(localmelo)
    if not sha.startswith(expect_sha):
        return [f"localmelo at {localmelo} is {sha or 'not a git checkout'}, not {expect_sha}"]
    if _git(localmelo, "status", "--porcelain"):
        return [f"localmelo at {localmelo} has uncommitted changes"]
    return []


def mcp_problems(arm_names: Sequence[str]) -> list[str]:
    """What is missing for the arms: the tap, and for D and S Node and the pinned chrome-devtools-mcp."""
    found = []
    if any(a in NEEDS_NODE for a in arm_names):
        node = node_dir() / "node"
        if not node.exists():
            found.append(f"node {node} is missing (nvm install {node_version()}, per web/mcp/.nvmrc)")
        if not CHROME_DEVTOOLS_MCP.exists():
            found.append(f"{CHROME_DEVTOOLS_MCP} is missing; run npm ci --ignore-scripts in {MCP_DIR}")
    if not TAP.exists():
        found.append(f"{TAP} is missing")
    return found


def mcp_versions() -> dict[str, str | None]:
    """chrome-devtools-mcp's pin (web/mcp/package.json) and the version installed in node_modules."""
    pinned = json.loads((MCP_DIR / "package.json").read_text()).get("dependencies", {}).get("chrome-devtools-mcp")
    installed = MCP_DIR / "node_modules" / "chrome-devtools-mcp" / "package.json"
    try:
        have = json.loads(installed.read_text()).get("version")
    except (OSError, ValueError):
        have = None
    return {"pinned": pinned, "installed": have}


def harness_arg_templates(arm_names: Sequence[str], extra: Sequence[str] = ()) -> dict[str, list[str]]:
    """localmelo's flags per arm with the config file left as ``<mcp.json>``: the shape a plan or a checkpoint records."""
    return {arm: list(localmelo_args(arm, Path("<mcp.json>"), extra)) for arm in arm_names}
