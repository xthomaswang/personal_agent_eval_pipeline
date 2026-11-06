"""The arms of the forms suite (forms/arms.py): what each run is given. No melo starts."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from forms import arms
from forms import tap as forms_tap


def original_builder(prompt: str, run_dir: Path, base_url: str, _max_turns: int) -> tuple[list[str], dict[str, str]]:
    """bench.harness's localmelo command line, with its HOME."""
    home = run_dir / "localmelo-home"
    home.mkdir(parents=True, exist_ok=True)
    return ["melo", "--base-url", f"{base_url}/v1", "--chat-model", "Bench-Model", "--", prompt], {"HOME": str(home)}


def built(
    tmp_path: Path, arm: arms.Arm, embedding: str | None = arms.EMBEDDING_MODEL
) -> tuple[list[str], dict[str, Any]]:
    builder = arms.melo_builder(original_builder, arm, "The-Model", embedding)
    cmd, env = builder("Sign in.", tmp_path, "http://127.0.0.1:43999", 60)
    return cmd, tomllib.loads((Path(env["HOME"]) / ".cache" / "localmelo" / "config.toml").read_text())


def test_the_arms_differ_in_extra_body_only() -> None:
    as_is, off = arms.ARMS["T-as-is"], arms.ARMS["T-off"]
    assert dict(as_is.extra_body) == {} and off.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert as_is.flags == off.flags == () and dict(as_is.settings) == dict(off.settings) == {}
    assert arms.DEFAULT_ARMS == ("T-as-is", "T-off")


def test_a_run_has_no_base_url_or_model_on_its_command_line(tmp_path: Path) -> None:
    cmd, _ = built(tmp_path, arms.ARMS["T-as-is"])
    assert cmd == ["melo", "--", "Sign in."]


def test_an_unknown_command_line_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="--base-url"):
        arms.without_flag(["melo", "--", "x"], "--base-url")


def test_the_config_points_chat_and_embeddings_at_the_proxy(tmp_path: Path) -> None:
    _, config = built(tmp_path, arms.ARMS["T-as-is"])
    assert config["chat_backend"] == "omlx" and config["embedding_backend"] == "omlx"
    assert config["request_timeout_s"] == 600
    assert config["vllm"] == {
        "chat_url": "http://127.0.0.1:43999/v1",
        "chat_model": "The-Model",
        "embedding_url": "http://127.0.0.1:43999/v1",
        "embedding_model": arms.EMBEDDING_MODEL,
    }


def test_t_off_alone_turns_thinking_off(tmp_path: Path) -> None:
    _, as_is = built(tmp_path / "a", arms.ARMS["T-as-is"])
    _, off = built(tmp_path / "b", arms.ARMS["T-off"])
    assert off["vllm"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "extra_body" not in as_is["vllm"]
    assert {**off["vllm"], "extra_body": None} == {**as_is["vllm"], "extra_body": None}


def test_no_embedding_model_means_no_embeddings(tmp_path: Path) -> None:
    _, config = built(tmp_path, arms.ARMS["T-off"], None)
    assert config["embedding_backend"] == "none" and "embedding_url" not in config["vllm"]


def test_an_arm_can_be_settings_and_flags(tmp_path: Path) -> None:
    arm = arms.Arm("T-step", "per step", settings={"agent.step_thinking": "first"}, flags=("-X", "agent.other=1"))
    _, config = built(tmp_path, arm)
    assert config["agent"] == {"step_thinking": "first"}
    assert arms.melo_flags(arm, Path("m.json"), 50, 900.0)[-2:] == ("-X", "agent.other=1")


def test_the_flags_keep_melo_under_the_budget_and_the_wall_cap() -> None:
    flags = arms.melo_flags(arms.ARMS["T-as-is"], Path("/r/mcp.json"), 50, 900.0, ("-v",))
    assert flags == (
        "--mcp", "/r/mcp.json", "--mcp-only", "-o", "json", "-X", "agent.tool_images=on", "-X", "agent.max_calls=49",
        "-t", "870", "-v",
    )  # fmt: skip
    assert arms.flag_template(arms.ARMS["T-off"], 50, 900.0)[1] == "<mcp.json>"


@pytest.mark.parametrize("flag", ["--mcp=x", "--mcp-only", "-o", "--url", "-m", "-t", "-t5", "--timeout=5", "--"])
def test_extra_flags_may_not_repeat_the_suites(flag: str) -> None:
    with pytest.raises(ValueError):
        arms.check_extra([flag])


def test_extra_flags_pass_otherwise() -> None:
    assert arms.check_extra(["-X", "agent.reflection=off"]) == ("-X", "agent.reflection=off")


def test_more_arms_come_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({"T-step": {"description": "d", "settings": {"agent.x": 1}, "flags": ["-X", "a.b=1"]}}))
    known = arms.load_arms(path)
    assert list(known) == ["T-as-is", "T-off", "T-step"] and known["T-step"].flags == ("-X", "a.b=1")
    assert known["T-step"].identity() == {"extra_body": {}, "settings": {"agent.x": 1}, "flags": ["-X", "a.b=1"]}
    assert arms.load_arms(None) == arms.ARMS


@pytest.mark.parametrize(
    ("text", "message"),
    [('{"T-off": {}}', "built in"), ('{"X": {"nope": 1}}', "unknown keys"), ("[]", "JSON object")],
)
def test_a_bad_arms_file_is_refused(text: str, message: str, tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        arms.load_arms(path)


def test_the_mcp_config_is_the_forms_tap_on_the_runs_chrome(tmp_path: Path) -> None:
    config = arms.mcp_config(tmp_path, "http://127.0.0.1:9333", tmp_path / "ws", "http://127.0.0.1:50000")
    server = config["mcpServers"][arms.SERVER]
    assert server["command"] == sys.executable and server["type"] == "stdio"
    assert server["args"] == [
        str(arms.TAP), "--mode", "dom", "--log", str(tmp_path / "tap.jsonl"), "--browser-url", "http://127.0.0.1:9333",
    ]  # fmt: skip
    assert server["env"][forms_tap.UPLOAD_ROOT_ENV] == str(tmp_path / "ws")
    assert server["env"][forms_tap.SITE_ENV] == "http://127.0.0.1:50000"
    assert "node" in server["env"]["PATH"] and "HOME" not in server["env"]


def test_the_builder_is_swapped_for_one_run_and_restored(tmp_path: Path) -> None:
    harness = SimpleNamespace(BUILDERS={"localmelo": original_builder, "hermes": print})
    options: dict[str, Any] = {
        "run_dir": tmp_path / "run", "browser_url": "http://127.0.0.1:9", "workspace": tmp_path / "ws",
        "site_url": "http://127.0.0.1:8", "model": "M", "embedding_model": None, "max_calls": 50, "wall_cap_s": 900.0,
    }  # fmt: skip
    with arms.overrides(harness, arms.ARMS["T-off"], **options) as plan:
        assert harness.BUILDERS["localmelo"] is not original_builder and harness.BUILDERS["hermes"] is print
        assert plan.mcp_json == tmp_path / "run" / "mcp.json" and plan.harness_args[1] == str(plan.mcp_json)
        assert json.loads(plan.mcp_json.read_text())["mcpServers"][arms.SERVER]["args"][0] == str(arms.TAP)
    assert harness.BUILDERS == {"localmelo": original_builder, "hermes": print}
    with pytest.raises(KeyError), arms.overrides(harness, arms.ARMS["T-off"], **options):
        raise KeyError("a run that fails")
    assert harness.BUILDERS["localmelo"] is original_builder
