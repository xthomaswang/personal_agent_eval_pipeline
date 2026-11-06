"""The arms of the forms suite, and how a localmelo run is set up for one.

An arm is data (``Arm``): what it adds to the local backend's ``extra_body``, to config.toml, and to melo's flags. A new
arm, such as a per-step thinking key, is one more entry in ``ARMS``, or one in the JSON file of ``--arms-file``:

    {"T-step": {"description": "...", "settings": {"agent.some_key": "value"}, "flags": ["-X", "agent.other=1"]}}

**How T-off reaches the model.** ``-X`` cannot set it: melo's settings catalog leaves out the backend sections and every
table (support/settings.py, ``catalog``), so ``melo -X vllm.extra_body=...`` is an unknown setting. A config.toml is not
enough either while ``--base-url`` is on the command line: that path builds its provider with no ``extra_body``
(``__main__.py``, ``_build_direct_mode_agent``). So every arm of this suite runs from a config.toml in the run's HOME,
with no ``--base-url`` and no ``--chat-model``: ``melo_builder`` takes them off bench.harness's command line and writes
the same address (the run's recorder proxy) and model into the file, for chat and embeddings. Both arms take this path,
and only ``extra_body`` differs. The proxy runs with ``--thinking passthrough``: its default would rewrite both arms to
thinking off.

**How melo drives the browser.** melo's own browser tools cannot do this suite. A direct run does not offer them (only
the terminal chat and the console do), and they refuse 127.0.0.1 with no setting to allow it
(support/browser/addresses.py, ``require_public``). So the runs use the web suite's D arm: chrome-devtools-mcp through a
tap (forms/tap.py), on a headless Chrome that run.py starts, with ``--mcp-only``. ``--mcp-only`` also leaves melo with
no built-in tool at all.

**Nobody answers.** A direct run calls ``Agent.run_task`` with no approval policy: no gate and no ``TOOL_ASK``, so no
approval or take-over card can be raised, and a tool that needs approval runs as any other (executor.py: ``None`` asks
no one). Its stdin is closed. A reply that asks a question ends the run like any other reply, and nothing is
submitted. If a later build does wait for someone, melo's ``-t`` and then bench.harness's wall cap end the run.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from forms import tap as forms_tap
from memory.across import server as melo_config
from smoke import adapters
from web import arms as web_arms

HERE = Path(__file__).resolve().parent
AGENT_ABILITY = HERE.parent
TAP = HERE / "tap.py"

HARNESS = "localmelo"
SERVER = web_arms.SERVER
TOOL_PREFIX = f"mcp__{SERVER}__"
DEFAULT_MODEL = "Qwen3.8-Flash-Next-IQ3XXS-splash"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"
LOCALMELO = AGENT_ABILITY / "e3" / "vendor" / "localmelo-f234b37"
LOCALMELO_SHA = "f234b37b0a08b0decc3f8cfc68f70f2bedf60897"
MELO_TIMEOUT_MARGIN_S = 30  # melo's own -t ends the run this much before the wall cap, so it leaves its record
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    extra_body: Mapping[str, Any] = field(default_factory=dict)  # [vllm] extra_body of the run's config.toml
    settings: Mapping[str, Any] = field(default_factory=dict)  # dotted keys of the run's config.toml
    flags: tuple[str, ...] = ()  # more flags for melo, after the suite's own

    def identity(self) -> dict[str, Any]:
        return {"extra_body": dict(self.extra_body), "settings": dict(self.settings), "flags": list(self.flags)}


ARMS: dict[str, Arm] = {
    "T-as-is": Arm("T-as-is", "no change: thinking as the server's default leaves it"),
    "T-off": Arm("T-off", "thinking off in every chat request", extra_body=THINKING_OFF),
}
DEFAULT_ARMS = tuple(ARMS)


def load_arms(path: Path | None) -> dict[str, Arm]:
    """``ARMS``, with the arms of a JSON file after them. A file may add arms but not change these two."""
    if path is None:
        return dict(ARMS)
    extra = json.loads(path.read_text())
    if not isinstance(extra, dict):
        raise ValueError(f"{path} must hold a JSON object of arms")
    arms = dict(ARMS)
    for name, spec in extra.items():
        if name in arms:
            raise ValueError(f"{path}: arm {name} is built in")
        unknown = set(spec) - {"description", "extra_body", "settings", "flags"}
        if unknown:
            raise ValueError(f"{path}: arm {name} has unknown keys {sorted(unknown)}")
        arms[name] = Arm(
            name,
            str(spec.get("description", "")),
            dict(spec.get("extra_body") or {}),
            dict(spec.get("settings") or {}),
            tuple(str(f) for f in spec.get("flags") or ()),
        )
    return arms


def check_extra(flags: Sequence[str]) -> tuple[str, ...]:
    """Extra melo flags, or a ``ValueError`` for one that would change what the suite sets."""
    web_arms.check_localmelo_extra(flags)
    for flag in flags:
        if flag.split("=", 1)[0] == "--timeout" or flag[:2] == "-t":
            raise ValueError(f"localmelo extra flag {flag!r} collides with the wall cap's -t")
    return tuple(flags)


def melo_flags(
    arm: Arm, mcp_json: Path, max_calls: int, wall_cap_s: float, extra: Sequence[str] = ()
) -> tuple[str, ...]:
    """melo's flags for ``harness.run(harness_args=...)``.

    ``max_calls`` is the proxy's budget: melo stops one short of it, so it ends on its own budget (web/arms.py).
    """
    return (
        "--mcp", str(mcp_json), "--mcp-only", "-o", "json",
        "-X", "agent.tool_images=on",
        "-X", f"agent.max_calls={max_calls - 1}",
        "-t", f"{max(wall_cap_s - MELO_TIMEOUT_MARGIN_S, 1):g}",
        *arm.flags, *extra,
    )  # fmt: skip


def flag_template(arm: Arm, max_calls: int, wall_cap_s: float, extra: Sequence[str] = ()) -> list[str]:
    """The flags with the config file left as ``<mcp.json>``: the shape a plan or a checkpoint records."""
    return list(melo_flags(arm, Path("<mcp.json>"), max_calls, wall_cap_s, extra))


def mcp_config(
    run_dir: Path, browser_url: str, workspace: Path, site_url: str, password: str | None = None
) -> dict[str, Any]:
    """The ``mcpServers`` document of a run: the forms tap, on the run's Chrome. *password* is the fixture account's."""
    env = {**web_arms.tap_env(), forms_tap.UPLOAD_ROOT_ENV: str(workspace), forms_tap.SITE_ENV: site_url}
    if password:
        env[forms_tap.PASSWORD_ENV] = password
    args = [str(TAP), "--mode", "dom", "--log", str(run_dir / "tap.jsonl"), "--browser-url", browser_url]
    return {"mcpServers": {SERVER: {"type": "stdio", "command": sys.executable, "args": args, "env": env}}}


def config_document(arm: Arm, proxy_url: str, model: str, embedding_model: str | None) -> dict[str, Any]:
    """The run's config.toml as data: an OpenAI-compatible local backend at the recorder proxy, as the memory bench
    writes it (its request timeout of 600 s also keeps a long thinking step from ending as a provider error)."""
    return melo_config.config_document(proxy_url, model, embedding_model, arm.extra_body, arm.settings)


def without_flag(cmd: list[str], flag: str) -> list[str]:
    """``cmd`` without ``flag`` and its value; a ``RuntimeError`` when it is not there, as bench.harness changed."""
    if flag not in cmd:
        raise RuntimeError(f"expected {flag} in the localmelo command line")
    at = cmd.index(flag)
    return [*cmd[:at], *cmd[at + 2 :]]


def melo_builder(original: adapters.Builder, arm: Arm, model: str, embedding_model: str | None) -> adapters.Builder:
    def build(prompt_text: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        cmd, env = original(prompt_text, run_dir, base_url, max_turns)
        cmd = without_flag(without_flag(cmd, "--base-url"), "--chat-model")
        melo_config.write_config(Path(env["HOME"]), config_document(arm, f"{base_url}/v1", model, embedding_model))
        return cmd, env

    return build


@dataclass(frozen=True)
class Plan:
    """What one run is given."""

    harness_args: tuple[str, ...]
    mcp_json: Path


@contextmanager
def overrides(
    harness_module: ModuleType,
    arm: Arm,
    *,
    run_dir: Path,
    browser_url: str,
    workspace: Path,
    site_url: str,
    model: str,
    embedding_model: str | None,
    max_calls: int,
    wall_cap_s: float,
    extra: Sequence[str] = (),
    password: str | None = None,
) -> Iterator[Plan]:
    """Swap in the arm's localmelo builder for one run and restore it after; writes ``<run_dir>/mcp.json``."""
    saved = dict(harness_module.BUILDERS)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        mcp_json = run_dir / "mcp.json"
        config = mcp_config(run_dir, browser_url, workspace, site_url, password)
        mcp_json.write_text(json.dumps(config, indent=2) + "\n")
        harness_module.BUILDERS[HARNESS] = melo_builder(saved[HARNESS], arm, model, embedding_model)
        yield Plan(melo_flags(arm, mcp_json, max_calls, wall_cap_s, extra), mcp_json)
    finally:
        harness_module.BUILDERS.clear()
        harness_module.BUILDERS.update(saved)
