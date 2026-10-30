"""The arms of the web suite (web/arms.py): the tap's command and environment, the prompts, and what each harness is
told. No harness, no model, no Chrome: builders run against copies of the harness homes in tmp_path."""

from __future__ import annotations

import json
import queue
import stat
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from marks import needs_bench
from web import arms

URL = "http://127.0.0.1:43999"
QUESTION = "Which planet has the most moons?"
MODEL = "Some-Model-oQ4e"
OLD_MODEL = "Old-Model"

# A claude-code.env of the shape the bench uses, with the safety variables, an old binary and an old model
CLAUDE_ENV_TEXT = """\
# synthetic claude-code.env
AGENT_ROOT="$HOME/agent"
export CLAUDE_CONFIG_DIR="$AGENT_ROOT/homes/claude-code"
CLAUDE_BIN="$HOME/.local/share/claude/versions/2.1.302"
export ANTHROPIC_BASE_URL="http://127.0.0.1:8500"
export ANTHROPIC_AUTH_TOKEN="placeholder"
export ANTHROPIC_API_KEY=""           # cleared so a real key is never inherited
export ANTHROPIC_MODEL="Old-Model"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Old-Model"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Old-Model"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Old-Model"
export CLAUDE_CODE_SUBAGENT_MODEL="Old-Model"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1   # no telemetry
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=131072
"""
CLAUDE_SETTINGS_TEXT = (
    '{\n  "$schema": "https://example.org/schema.json",\n  "model": "Old-Model",\n  "includeCoAuthoredBy": false\n}\n'
)


@pytest.fixture
def nvm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NVM_DIR", str(tmp_path / "nvm"))
    return tmp_path / "nvm"


# --- the prompts ---


def test_d_and_s_prompts_are_byte_identical() -> None:
    assert arms.prompt("D", QUESTION).encode() == arms.prompt("S", QUESTION).encode()


def test_the_open_prompt_is_the_design_text() -> None:
    assert arms.prompt("D", QUESTION) == (
        "A Google Chrome window is open on Google Search. Use the browser tools to search the web and\n"
        "answer the question below, checking the answer on the pages you find. Only read: do not sign in,\n"
        "post, buy or change settings.\n"
        "\n"
        f"Question: {QUESTION}\n"
        "\n"
        "End your reply with one line: FINAL ANSWER: <answer>\n"
        "Your final answer should be a number OR as few words as possible OR a comma separated list of\n"
        "numbers and/or strings. If you are asked for a number, don't use comma to write your number\n"
        "neither use units such as $ or percent sign unless specified otherwise. Write a date as day,\n"
        "month name and year. If you cannot find the answer, write FINAL ANSWER: unknown"
    )


def test_the_closed_book_prompt_changes_the_first_paragraph_only() -> None:
    closed, open_ = arms.prompt("C", QUESTION), arms.prompt("D", QUESTION)
    assert closed.startswith("Answer the question below from your own knowledge. You have no tools.\n\n")
    assert "Chrome" not in closed
    assert closed.split("\n\n", 1)[1] == open_.split("\n\n", 1)[1]


def test_an_unknown_arm_has_no_prompt() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        arms.prompt("X", QUESTION)


# --- node and the tap's environment ---


def test_node_is_found_under_nvm_by_the_nvmrc_version(nvm: Path) -> None:
    version = (arms.MCP_DIR / ".nvmrc").read_text().strip()
    assert version == "v24.21.0"
    assert arms.node_dir() == nvm / "versions" / "node" / "v24.21.0" / "bin"
    assert arms.tap_path() == f"{nvm}/versions/node/v24.21.0/bin:/usr/bin:/bin"


def test_a_version_without_its_v_gets_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".nvmrc").write_text("24.21.0\n")
    monkeypatch.setattr(arms, "MCP_DIR", tmp_path)
    assert arms.node_version() == "v24.21.0"


def test_without_nvm_dir_the_home_nvm_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVM_DIR", raising=False)
    assert arms.nvm_dir() == Path.home() / ".nvm"


def test_the_tap_env_has_node_on_the_path_and_no_telemetry(nvm: Path) -> None:
    env = arms.tap_env()
    assert env == {
        "PATH": arms.tap_path(),
        "CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1",
        "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
    }
    assert str(arms.node_dir()) in env["PATH"].split(":")
    assert env["PATH"].split(":")[-2:] == ["/usr/bin", "/bin"]


def test_no_path_in_the_module_is_a_home_path() -> None:
    source = Path(arms.__file__).read_text()
    assert str(Path.home()) not in source
    assert "/Users/" not in source


# --- the tap as a server ---


@pytest.mark.parametrize(("arm", "mode"), [("D", "dom"), ("S", "screen"), ("C", "none")])
def test_the_tap_command_line_of_each_arm(arm: str, mode: str, tmp_path: Path) -> None:
    args = arms.tap_args(arm, tmp_path, URL)
    assert args[:5] == [str(arms.TAP), "--mode", mode, "--log", str(tmp_path / "tap.jsonl")]
    assert args[5:] == (["--browser-url", URL] if arm != "C" else [])


def test_c_needs_no_browser_and_d_and_s_do(tmp_path: Path) -> None:
    assert "--browser-url" not in arms.tap_args("C", tmp_path, None)
    for arm in ("D", "S"):
        with pytest.raises(ValueError, match="Chrome"):
            arms.tap_args(arm, tmp_path, None)
    assert arms.NEEDS_CHROME == {"D": True, "S": True, "C": False, "S-cua": True}


def test_the_server_runs_under_this_interpreter(nvm: Path, tmp_path: Path) -> None:
    spec = arms.server_spec("D", tmp_path, URL)
    assert spec["command"] == sys.executable
    assert spec["args"] == arms.tap_args("D", tmp_path, URL)
    assert spec["env"] == arms.tap_env()
    assert arms.mcp_config("D", tmp_path, URL) == {"mcpServers": {"webtap": {"type": "stdio", **spec}}}


def test_the_closed_book_tap_starts_with_the_arm_environment_and_offers_no_tool(nvm: Path, tmp_path: Path) -> None:
    spec = arms.server_spec("C", tmp_path, None)
    proc = subprocess.Popen(
        [spec["command"], *spec["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=spec["env"],
    )
    lines: queue.Queue[str] = queue.Queue()
    assert proc.stdin is not None and proc.stdout is not None
    stdout = proc.stdout
    threading.Thread(target=lambda: [lines.put(line) for line in stdout], daemon=True).start()

    def ask(rid: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        proc.stdin.flush()
        return json.loads(lines.get(timeout=30))

    try:
        init = ask(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {}})
        assert init["result"]["serverInfo"]["name"] == "web"  # the tap's own name; the harnesses' key is arms.SERVER
        assert ask(2, "tools/list", {})["result"]["tools"] == []
    finally:
        proc.stdin.close()
        proc.wait(timeout=20)
    events = [json.loads(line) for line in (tmp_path / "tap.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "start" and events[0]["mode"] == "none"


# --- Claude Code ---


@pytest.fixture
def claude_files(tmp_path: Path) -> SimpleNamespace:
    """A synthetic claude-code.env and home template, and an executable that stands for the pinned binary."""
    env = tmp_path / "templates" / "claude-code.env"
    env.parent.mkdir()
    env.write_text(CLAUDE_ENV_TEXT)
    home = tmp_path / "templates" / "claude-home"
    home.mkdir()
    (home / "settings.json").write_text(CLAUDE_SETTINGS_TEXT)
    binary = tmp_path / "bin" / "2.1.312"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\necho 2.1.312\n")
    binary.chmod(0o755)
    return SimpleNamespace(env=env, home=home, binary=binary)


def sourced(env_file: Path) -> dict[str, str]:
    return arms.shell_env(env_file)


@needs_bench
def test_claude_code_gets_only_the_web_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvm: Path, claude_files: SimpleNamespace
) -> None:
    from bench import harness

    monkeypatch.setattr(harness, "CLAUDE_HOME", claude_files.home)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    server = arms.server_spec("S", run_dir, URL)

    build = arms.claude_code_builder(harness._claude_code, server, MODEL)
    cmd, env = build("do it", run_dir, "http://127.0.0.1:43210", 40)

    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__webtap__*"
    assert "--strict-mcp-config" in cmd
    config = json.loads(cmd[cmd.index("--mcp-config") + 1])
    assert config == {"mcpServers": {"webtap": {"type": "stdio", **server}}}
    assert env["ENABLE_TOOL_SEARCH"] == "false"
    assert env["PAB_BASE_URL"] == "http://127.0.0.1:43210"
    assert cmd[cmd.index("-p") + 1] == "do it"
    assert cmd[cmd.index("--max-turns") + 1] == "40"
    # the copied home names the study's model; the template is untouched
    assert json.loads((Path(env["PAB_CONFIG_DIR"]) / "settings.json").read_text())["model"] == MODEL
    assert (claude_files.home / "settings.json").read_text() == CLAUDE_SETTINGS_TEXT


def test_a_claude_code_command_without_its_flags_fails_loudly(tmp_path: Path) -> None:
    def original(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        return ["claude", "-p", prompt], {}

    build = arms.claude_code_builder(original, {"command": "x", "args": [], "env": {}}, MODEL)
    with pytest.raises(RuntimeError, match="--tools"):
        build("p", tmp_path, "http://x", 1)


def test_the_defaults_are_the_flash_model_and_the_reference_runs_binary() -> None:
    assert arms.DEFAULT_MODEL == "Qwen3.8-Flash-Next-oQ4e-mtp"
    assert arms.CLAUDE_BIN.parts[-4:] == ("backend", "vendor", "claude-code", "2.1.312")


def test_the_env_copy_replaces_the_binary_and_the_model_variables_only() -> None:
    values = arms.claude_values(MODEL, Path("/opt/claude/2.1.312"))
    assert values == {**dict.fromkeys(arms.CLAUDE_MODEL_VARS, MODEL), "CLAUDE_BIN": "/opt/claude/2.1.312"}
    patched = arms.patch_claude_env(CLAUDE_ENV_TEXT, values)
    changed = [(a, b) for a, b in zip(CLAUDE_ENV_TEXT.splitlines(), patched.splitlines(), strict=True) if a != b]
    assert len(changed) == 6  # CLAUDE_BIN and the five model variables
    assert 'CLAUDE_BIN="/opt/claude/2.1.312"' in patched and f'export ANTHROPIC_MODEL="{MODEL}"' in patched
    assert 'export ANTHROPIC_API_KEY=""           # cleared so a real key is never inherited' in patched
    assert "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1   # no telemetry" in patched  # comments and spacing stay
    assert arms.patch_claude_env(patched, values) == patched  # a value that is already right is left alone


def test_the_env_copy_refuses_a_template_without_a_variable() -> None:
    text = CLAUDE_ENV_TEXT.replace('export CLAUDE_CODE_SUBAGENT_MODEL="Old-Model"\n', "")
    with pytest.raises(arms.Refused, match="CLAUDE_CODE_SUBAGENT_MODEL"):
        arms.patch_claude_env(text, arms.claude_values(MODEL, Path("/opt/c")))


def test_sourcing_the_env_copy_gives_the_study_model_and_the_binary(
    tmp_path: Path, claude_files: SimpleNamespace
) -> None:
    written = arms.write_claude_env(claude_files.env, tmp_path / "run" / "claude-code.env", MODEL, claude_files.binary)
    env = sourced(written)
    assert [env[name] for name in arms.CLAUDE_MODEL_VARS] == [MODEL] * 5
    assert env["__SHELL__CLAUDE_BIN"] == str(claude_files.binary)
    assert env["ANTHROPIC_API_KEY"] == "" and env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert claude_files.env.read_text() == CLAUDE_ENV_TEXT  # the template is never touched


@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ("export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1\n", "CLAUDE_CODE_DISABLE_CLAUDE_MDS must be exported as '1'"),
        ("export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1\n", "CLAUDE_CODE_DISABLE_AUTO_MEMORY must be exported as '1'"),
        (
            "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1   # no telemetry\n",
            "NONESSENTIAL_TRAFFIC must be exported",
        ),
        (
            'export ANTHROPIC_API_KEY=""           # cleared so a real key is never inherited\n',
            "ANTHROPIC_API_KEY must",
        ),
        (
            'export CLAUDE_CONFIG_DIR="$AGENT_ROOT/homes/claude-code"\n',
            "CLAUDE_CONFIG_DIR must be exported and non-empty",
        ),
        ('export ANTHROPIC_AUTH_TOKEN="placeholder"\n', "ANTHROPIC_AUTH_TOKEN must be exported and non-empty"),
    ],
)
def test_a_template_that_loses_a_safety_variable_is_refused(
    drop: str, expected: str, tmp_path: Path, claude_files: SimpleNamespace
) -> None:
    assert drop in CLAUDE_ENV_TEXT
    claude_files.env.write_text(CLAUDE_ENV_TEXT.replace(drop, ""))
    with pytest.raises(arms.Refused, match=expected):
        arms.write_claude_env(claude_files.env, tmp_path / "copy.env", MODEL, claude_files.binary)
    problems = arms.claude_problems(claude_files.env, claude_files.home, MODEL, claude_files.binary)
    assert len(problems) == 1 and expected in problems[0]


def test_a_template_that_sets_a_safety_variable_wrongly_is_refused(
    tmp_path: Path, claude_files: SimpleNamespace
) -> None:
    claude_files.env.write_text(
        CLAUDE_ENV_TEXT.replace('export ANTHROPIC_API_KEY=""', 'export ANTHROPIC_API_KEY="sk-live"')
    )
    problems = arms.claude_problems(claude_files.env, claude_files.home, MODEL, claude_files.binary)
    assert len(problems) == 1 and "ANTHROPIC_API_KEY must be exported as ''" in problems[0]
    assert "sk-live" not in problems[0]  # a key is never echoed


def test_claude_problems_checks_the_binary_and_the_settings(claude_files: SimpleNamespace) -> None:
    assert arms.claude_problems(claude_files.env, claude_files.home, MODEL, claude_files.binary) == []
    gone = claude_files.binary.with_name("2.1.999")
    assert "missing or not executable" in arms.claude_problems(claude_files.env, claude_files.home, MODEL, gone)[0]
    claude_files.binary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert "not executable" in arms.claude_problems(claude_files.env, claude_files.home, MODEL, claude_files.binary)[0]
    claude_files.binary.chmod(0o755)
    (claude_files.home / "settings.json").write_text('{"includeCoAuthoredBy": false}')
    problems = arms.claude_problems(claude_files.env, claude_files.home, MODEL, claude_files.binary)
    assert problems == ['settings.json must have exactly one "model" entry']


def test_settings_json_changes_its_model_and_nothing_else() -> None:
    patched = arms.patch_claude_settings(CLAUDE_SETTINGS_TEXT, MODEL)
    assert json.loads(patched) == {**json.loads(CLAUDE_SETTINGS_TEXT), "model": MODEL}
    assert patched == CLAUDE_SETTINGS_TEXT.replace(OLD_MODEL, MODEL)  # the bytes around it stay
    for bad in ('{"a": 1}', '{"model": "a", "x": {"model": "b"}}'):
        with pytest.raises(arms.Refused, match='exactly one "model"'):
            arms.patch_claude_settings(bad, MODEL)


@needs_bench
def test_claude_code_overrides_swap_the_env_file_and_restore_it(
    tmp_path: Path, nvm: Path, claude_files: SimpleNamespace
) -> None:
    from bench import harness

    original_env = harness.CLAUDE_ENV
    before = dict(harness.BUILDERS)
    run_dir = tmp_path / "run"
    with arms.overrides(
        harness, "claude_code", "D", run_dir=run_dir, browser_url=URL, model=MODEL, claude_bin=claude_files.binary
    ) as plan:
        # bench.harness would now source the per-run copy
        assert harness.CLAUDE_ENV == run_dir / "claude-code.env"
        env = sourced(harness.CLAUDE_ENV)
        assert env["ANTHROPIC_MODEL"] == MODEL and env["__SHELL__CLAUDE_BIN"] == str(claude_files.binary)
        assert plan.harness_args == ()
    assert harness.CLAUDE_ENV == original_env and harness.BUILDERS == before
    with (
        pytest.raises(RuntimeError),
        arms.overrides(
            harness,
            "claude_code",
            "D",
            run_dir=tmp_path / "run2",
            browser_url=URL,
            model=MODEL,
            claude_bin=claude_files.binary,
        ),
    ):
        raise RuntimeError("the run failed")
    assert harness.CLAUDE_ENV == original_env and harness.BUILDERS == before


@needs_bench
def test_other_harnesses_leave_the_claude_env_alone(tmp_path: Path, nvm: Path) -> None:
    from bench import harness

    original_env = harness.CLAUDE_ENV
    for name in ("hermes", "localmelo"):
        with arms.overrides(harness, name, "D", run_dir=tmp_path / name, browser_url=URL):
            assert harness.CLAUDE_ENV == original_env


@needs_bench
def test_a_claude_env_with_a_lost_safety_variable_stops_the_override(
    tmp_path: Path, nvm: Path, claude_files: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench import harness

    claude_files.env.write_text(CLAUDE_ENV_TEXT.replace("export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1\n", ""))
    monkeypatch.setattr(harness, "CLAUDE_ENV", claude_files.env)
    before = dict(harness.BUILDERS)
    with (
        pytest.raises(arms.Refused, match="CLAUDE_CODE_DISABLE_CLAUDE_MDS"),
        arms.overrides(
            harness,
            "claude_code",
            "D",
            run_dir=tmp_path / "run",
            browser_url=URL,
            model=MODEL,
            claude_bin=claude_files.binary,
        ),
    ):
        pass
    assert harness.CLAUDE_ENV == claude_files.env and harness.BUILDERS == before


# --- Hermes ---

HERMES_TEMPLATE = """\
model:
  provider: local
  default: some-model
providers:
  local:
    api: http://127.0.0.1:8500/v1
agent:
  max_turns: null
mcp_servers:
  other:
    command: /bin/false
"""


def hermes_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str) -> tuple[list[str], dict[str, Any]]:
    from bench import harness

    template = tmp_path / "hermes-template"
    template.mkdir(exist_ok=True)
    (template / "config.yaml").write_text(HERMES_TEMPLATE)
    monkeypatch.setattr(harness, "HERMES_HOME", template)
    monkeypatch.setattr(harness, "UPSTREAM", "http://127.0.0.1:8500")
    run_dir = tmp_path / f"run-{arm}"
    run_dir.mkdir()
    server = arms.server_spec(
        arm, run_dir, URL if arm not in ("C", "S-cua") else None, chrome_pid=4242 if arm == "S-cua" else None
    )
    cmd, env = arms.hermes_builder(harness._hermes, server, images=arm in arms.IMAGE_ARMS)(
        "do it", run_dir, "http://127.0.0.1:43210", 40
    )
    config = yaml.safe_load((Path(env["HERMES_HOME"]) / "config.yaml").read_text())
    assert (template / "config.yaml").read_text() == HERMES_TEMPLATE  # the home template is never touched
    return cmd, config


@needs_bench
@pytest.mark.parametrize("arm", ["D", "S", "C"])
def test_hermes_config_copy_enables_only_the_web_server(
    arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvm: Path
) -> None:
    cmd, config = hermes_config(tmp_path, monkeypatch, arm)
    servers = config["mcp_servers"]
    run_dir = tmp_path / f"run-{arm}"
    assert servers["webtap"] == {**arms.server_spec(arm, run_dir, URL if arm != "C" else None), "enabled": True}
    assert servers["other"]["enabled"] is False
    assert config["tools"]["tool_search"]["enabled"] == "off"  # the string, not YAML's boolean false
    assert "http://127.0.0.1:43210/v1" in json.dumps(config)
    assert cmd[cmd.index("-t") + 1] == "webtap"
    assert "--yolo" not in cmd


@needs_bench
def test_only_the_screenshot_arms_get_native_images_in_hermes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvm: Path
) -> None:
    for arm in ("D", "C"):
        _, config = hermes_config(tmp_path, monkeypatch, arm)
        assert config["agent"].get("image_input_mode") != "native"
        assert "supports_vision" not in config["model"]
    for arm in ("S", "S-cua"):
        _, config = hermes_config(tmp_path, monkeypatch, arm)
        assert config["agent"]["image_input_mode"] == "native"
        assert config["model"]["supports_vision"] is True
        assert config["model"]["provider"] == "local"  # the rest of the model block stays


@needs_bench
def test_the_server_name_is_no_hermes_toolset() -> None:
    """`-t <name>` of an MCP server that shares a name with a built-in toolset (web, browser) also selects the built-in
    tools. The static toolsets are read from the Hermes checkout when it is there."""
    import re

    from bench import harness

    toolsets = harness.AGENT_ROOT / "hermes-agent" / "toolsets.py"
    if not toolsets.exists():
        pytest.skip("no Hermes checkout")
    names = set(re.findall(r'^\s{4}"([A-Za-z0-9_-]+)":\s*_ts\(', toolsets.read_text(), re.MULTILINE))
    assert {"web", "browser"} <= names  # the pattern finds them
    assert arms.SERVER not in names and arms.SERVER == "webtap"


# --- localmelo ---


@pytest.mark.parametrize("arm", ["D", "C"])
def test_localmelo_flags_of_d_and_c(arm: str) -> None:
    assert arms.localmelo_args(arm, Path("m.json")) == ("--mcp", "m.json", "--mcp-only", "-o", "json")


def test_localmelo_s_sends_tool_images_to_the_model() -> None:
    assert arms.localmelo_args("S", Path("m.json")) == (
        "--mcp", "m.json", "--mcp-only", "-o", "json", "-X", "agent.tool_images=on",
    )  # fmt: skip


def test_the_flag_templates_for_a_checkpoint() -> None:
    assert arms.harness_arg_templates(("D", "S")) == {
        "D": ["--mcp", "<mcp.json>", "--mcp-only", "-o", "json"],
        "S": ["--mcp", "<mcp.json>", "--mcp-only", "-o", "json", "-X", "agent.tool_images=on"],
    }


@needs_bench
@pytest.mark.parametrize("arm", ["D", "S", "C"])
def test_localmelo_flags_land_before_the_double_dash(arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import harness

    monkeypatch.setattr(harness.manifest, "LOCALMELO_REPO", tmp_path / "melo-clone")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd, _ = harness._localmelo("the task", run_dir, "http://127.0.0.1:43210", 40)
    args = arms.localmelo_args(arm, run_dir / "mcp.json")
    full = harness._with_harness_args(cmd, "the task", args)
    assert full[-2:] == ["--", "the task"]
    assert full[full.index("--") - len(args) : full.index("--")] == list(args)
    assert "--mcp-only" in full[: full.index("--")]


# --- overrides: swapped for one run, restored afterwards ---


@needs_bench
def test_claude_code_and_hermes_get_wrapped_builders_and_localmelo_does_not(tmp_path: Path, nvm: Path) -> None:
    from bench import harness

    before = dict(harness.BUILDERS)
    with arms.overrides(harness, "claude_code", "D", run_dir=tmp_path / "a", browser_url=URL) as plan:
        assert harness.BUILDERS["claude_code"] is not before["claude_code"]
        assert harness.BUILDERS["hermes"] is before["hermes"]
        assert harness.BUILDERS["localmelo"] is before["localmelo"]
        assert plan.harness_args == () and plan.mcp_json is None
    assert harness.BUILDERS == before
    with arms.overrides(harness, "hermes", "S", run_dir=tmp_path / "b", browser_url=URL) as plan:
        assert harness.BUILDERS["hermes"] is not before["hermes"]
        assert harness.BUILDERS["claude_code"] is before["claude_code"]
        assert plan.harness_args == ()
    assert harness.BUILDERS == before
    assert all(harness.BUILDERS[k] is before[k] for k in before)


@needs_bench
def test_localmelo_gets_its_flags_and_a_config_file_and_its_builder_stays(tmp_path: Path, nvm: Path) -> None:
    from bench import harness

    before = dict(harness.BUILDERS)
    run_dir = tmp_path / "run"
    with arms.overrides(harness, "localmelo", "S", run_dir=run_dir, browser_url=URL) as plan:
        assert dict(harness.BUILDERS) == before
        assert plan.mcp_json == run_dir / "mcp.json"
        assert plan.harness_args == arms.localmelo_args("S", run_dir / "mcp.json")
        assert json.loads(plan.mcp_json.read_text()) == arms.mcp_config("S", run_dir, URL)
        assert plan.server == arms.server_spec("S", run_dir, URL)
    assert dict(harness.BUILDERS) == before


@needs_bench
@pytest.mark.parametrize("name", arms.HARNESSES)
def test_builders_are_restored_when_the_run_fails(name: str, tmp_path: Path, nvm: Path) -> None:
    from bench import harness

    before = dict(harness.BUILDERS)
    with pytest.raises(RuntimeError), arms.overrides(harness, name, "D", run_dir=tmp_path, browser_url=URL):
        raise RuntimeError("the run failed")
    assert harness.BUILDERS == before
    assert all(harness.BUILDERS[k] is before[k] for k in before)


def test_an_unknown_harness_is_refused(tmp_path: Path) -> None:
    holder = type("H", (), {"BUILDERS": {}})
    with (
        pytest.raises(ValueError, match="unknown harness"),
        arms.overrides(holder, "some_agent", "D", run_dir=tmp_path, browser_url=URL),
    ):
        pass


# --- what must be in place ---


def test_localmelo_is_pinned_like_e3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = arms.LOCALMELO_SHA
    assert sha.startswith("a9fb42b") and len(sha) == 40
    assert arms.LOCALMELO.name == "localmelo-a9fb42b"
    answers: dict[tuple[str, ...], str] = {("rev-parse", "HEAD"): sha, ("status", "--porcelain"): ""}
    monkeypatch.setattr(arms, "_git", lambda repo, *args: answers.get(args, ""))
    assert arms.localmelo_problems(tmp_path) == []

    answers[("status", "--porcelain")] = " M melo/agent/agent.py"
    assert "uncommitted changes" in arms.localmelo_problems(tmp_path)[0]

    answers[("rev-parse", "HEAD")] = "ce4a3a8" + "0" * 33
    problem = arms.localmelo_problems(tmp_path)[0]
    assert "ce4a3a8" in problem and "not a9fb42b" in problem

    answers[("rev-parse", "HEAD")] = ""
    assert "not a git checkout" in arms.localmelo_problems(tmp_path)[0]


def test_missing_node_and_the_missing_child_are_reported_for_d_and_s_only(
    nvm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problems = arms.mcp_problems(("D", "S", "C"))
    assert any("node" in p and "v24.21.0" in p for p in problems)
    assert arms.mcp_problems(("C",)) == []
    (nvm / "versions" / "node" / "v24.21.0" / "bin").mkdir(parents=True)
    (arms.node_dir() / "node").write_text("")
    assert not any("node " in p for p in arms.mcp_problems(("D",)))


def test_the_pinned_chrome_devtools_mcp_version_is_read_from_package_json() -> None:
    versions = arms.mcp_versions()
    assert (
        versions["pinned"]
        == json.loads((arms.MCP_DIR / "package.json").read_text())["dependencies"]["chrome-devtools-mcp"]
    )
    assert versions["pinned"] == "1.11.0"


# --- S-cua ---


def test_the_arms_and_their_defaults() -> None:
    assert arms.ARMS == ("D", "S", "C", "S-cua")
    assert arms.DEFAULT_ARMS == ("D", "S", "C")  # S-cua runs only when it is named
    assert arms.TAP_MODES["S-cua"] == "cua" and arms.NEEDS_CHROME["S-cua"] is True
    assert arms.HEADED == {"S-cua"} and arms.NEEDS_NODE == {"D", "S"} and arms.IMAGE_ARMS == {"S", "S-cua"}


def test_s_cua_uses_the_prompt_of_d_and_s() -> None:
    assert (
        arms.prompt("S-cua", QUESTION).encode()
        == arms.prompt("D", QUESTION).encode()
        == arms.prompt("S", QUESTION).encode()
    )


def test_the_s_cua_tap_gets_the_chrome_pid_and_the_cua_driver_binary(tmp_path: Path) -> None:
    args = arms.tap_args("S-cua", tmp_path, None, chrome_pid=4242, cua_bin="cua-driver")
    assert args == [
        str(arms.TAP),
        "--mode",
        "cua",
        "--log",
        str(tmp_path / "tap.jsonl"),
        "--chrome-pid",
        "4242",
        "--cua-bin",
        "cua-driver",
    ]
    assert arms.tap_args("S-cua", tmp_path, None, chrome_pid=7)[-2:] == [
        "--chrome-pid",
        "7",
    ]  # the tap finds the binary itself
    with pytest.raises(ValueError, match="pid"):
        arms.tap_args("S-cua", tmp_path, URL)  # a browser URL is no substitute
    assert "--chrome-pid" not in arms.tap_args("S", tmp_path, URL, chrome_pid=4242)  # the other arms ignore it
    assert "--browser-url" not in arms.tap_args("S-cua", tmp_path, URL, chrome_pid=4242)


def test_only_s_cua_gets_the_real_home_in_the_tap_env(nvm: Path, tmp_path: Path) -> None:
    """cua-driver finds its daemon through HOME, and bench.harness gives localmelo an empty one (measured with
    `cua-driver status` under an empty HOME: the daemon reads as not running)."""
    assert arms.tap_env("S-cua") == {**arms.tap_env(), "HOME": str(Path.home())}
    for arm in ("D", "S", "C", None):
        assert "HOME" not in arms.tap_env(arm)
    spec = arms.server_spec("S-cua", tmp_path, None, chrome_pid=1)
    assert spec["env"]["HOME"] == str(Path.home())
    assert "HOME" not in arms.server_spec("D", tmp_path, URL)["env"]


def test_s_cua_needs_no_node(nvm: Path) -> None:
    assert arms.mcp_problems(("S-cua",)) == []
    assert any("node" in p for p in arms.mcp_problems(("S-cua", "D")))
    assert arms.mcp_config("S-cua", Path("r"), None, chrome_pid=1)["mcpServers"]["webtap"]["args"][2] == "cua"


@needs_bench
def test_s_cua_gets_images_in_localmelo_and_a_config_with_the_pid(tmp_path: Path, nvm: Path) -> None:
    from bench import harness

    assert arms.localmelo_args("S-cua", Path("m.json")) == arms.localmelo_args("S", Path("m.json"))
    with arms.overrides(
        harness, "localmelo", "S-cua", run_dir=tmp_path, browser_url=URL, chrome_pid=4242, cua_bin="cua-driver"
    ) as plan:
        assert plan.harness_args[-2:] == ("-X", "agent.tool_images=on")
        args = json.loads(plan.mcp_json.read_text())["mcpServers"]["webtap"]["args"]
    assert args[1:3] == ["--mode", "cua"] and args[-4:] == ["--chrome-pid", "4242", "--cua-bin", "cua-driver"]


FAKE_CUA = """#!{python}
import json, os, sys
mode = os.environ.get("FAKE_CUA", "ok")
if sys.argv[1:] == ["status"]:
    print("Cua Driver daemon is running" if mode != "down" else "Cua Driver daemon is not running")
    sys.exit(0 if mode != "down" else 1)
if sys.argv[1:3] == ["call", "list_windows"]:
    assert sys.argv[3] == "{{}}"
    if mode == "fail":
        print("no permission", file=sys.stderr)
        sys.exit(1)
    if mode == "garbage":
        print("not json")
    elif mode == "nospace":
        print(json.dumps({{"windows": []}}))
    elif mode == "wrapped":
        print(json.dumps({{"structuredContent": {{"current_space_id": 7, "windows": []}}}}))
    else:
        print(json.dumps({{"current_space_id": int(os.environ.get("FAKE_SPACE", "400")), "windows": [{{"window_id": 1}}]}}))
    sys.exit(0)
sys.exit(2)
"""


@pytest.fixture
def fake_cua(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "fake-cua-driver"
    path.write_text(FAKE_CUA.format(python=sys.executable))
    path.chmod(0o755)
    for name in ("FAKE_CUA", "FAKE_SPACE"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_the_cua_daemon_must_be_running(fake_cua: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert arms.cua_status_problem(fake_cua) is None
    monkeypatch.setenv("FAKE_CUA", "down")
    assert "not running" in arms.cua_status_problem(fake_cua)
    assert "cannot run" in arms.cua_status_problem(tmp_path / "missing")


def test_the_space_on_screen_is_read_from_list_windows(fake_cua: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert arms.cua_current_space(fake_cua) == 400
    monkeypatch.setenv("FAKE_SPACE", "99")
    assert arms.cua_current_space(fake_cua) == 99
    monkeypatch.setenv("FAKE_CUA", "wrapped")
    assert arms.cua_current_space(fake_cua) == 7
    for mode, message in (("fail", "list_windows failed"), ("garbage", "no JSON"), ("nospace", "no current_space_id")):
        monkeypatch.setenv("FAKE_CUA", mode)
        with pytest.raises(arms.CuaError, match=message):
            arms.cua_current_space(fake_cua)


def test_the_cua_binary_is_found_as_the_tap_finds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_TAP_CUA_BIN", raising=False)
    monkeypatch.setattr(arms.tap_module, "default_cua_bin", lambda: "found-by-bundle-id")
    assert arms.resolve_cua_bin() == "found-by-bundle-id"
    monkeypatch.setenv("WEB_TAP_CUA_BIN", "from-env")
    assert arms.resolve_cua_bin() == "from-env"
    monkeypatch.delenv("WEB_TAP_CUA_BIN")

    def nothing() -> str:
        raise ValueError("no app with bundle id")

    monkeypatch.setattr(arms.tap_module, "default_cua_bin", nothing)
    with pytest.raises(ValueError, match="bundle id"):
        arms.resolve_cua_bin()
