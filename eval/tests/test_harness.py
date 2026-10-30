"""bench.harness: per-run harness homes, command lines, and the run loop (no model, no oMLX)."""

from __future__ import annotations

import json
import shlex
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

from bench import harness


@pytest.fixture
def no_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.manifest, "collect", lambda: {"stub": True})


def test_hermes_home_is_copied_and_pointed_at_the_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = tmp_path / "template"
    (template / "tools").mkdir(parents=True)
    (template / "logs").mkdir()
    (template / "config.yaml").write_text("providers:\n  omlx:\n    api: http://127.0.0.1:8500/v1\n")
    monkeypatch.setattr(harness, "HERMES_HOME", template)
    monkeypatch.setattr(harness, "UPSTREAM", "http://127.0.0.1:8500")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cmd, env = harness._hermes("do it", run_dir, "http://127.0.0.1:43210", 30)

    home = Path(env["HERMES_HOME"])
    assert home == run_dir / "hermes-home"
    assert "http://127.0.0.1:43210/v1" in (home / "config.yaml").read_text()
    assert "8500" not in (home / "config.yaml").read_text()
    assert (home / "tools").is_symlink()
    assert not (home / "logs").exists()
    assert (template / "config.yaml").read_text().count("8500") == 1
    assert cmd[cmd.index("-t") + 1] == harness.HERMES_TOOLSETS
    assert cmd[cmd.index("--max-turns") + 1] == "30"
    assert cmd[cmd.index("--query-file") + 1] == str(run_dir / "prompt.md")
    assert "--yolo" not in cmd


def test_hermes_config_names_the_proxy_and_bench_model_whatever_the_template_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = (
        "# template for Qwen-x on oMLX\n"
        "model:\n  provider: local\n  default: Qwen-x\n"
        "providers:\n  local:\n    api: http://127.0.0.1:8500/v1\n    default_model: Qwen-x\n"
        "    models:\n      Qwen-x:\n        context_length: 131072\n"
    )
    monkeypatch.setattr(harness, "UPSTREAM", "http://127.0.0.1:11500")  # another engine, e.g. rec
    monkeypatch.setattr(harness, "MODEL", "flint")

    text = harness._hermes_config(template, "http://127.0.0.1:43210")

    config = yaml.safe_load(text)
    assert config["model"]["default"] == "flint"
    assert config["providers"]["local"] == {
        "api": "http://127.0.0.1:43210/v1", "default_model": "flint", "models": {"flint": {"context_length": 131072}},
    }  # fmt: skip
    assert text.startswith("# template for flint")
    with pytest.raises(KeyError):
        harness._hermes_config("model:\n  provider: other\n", "http://127.0.0.1:43210")


def test_claude_code_overrides_base_url_and_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = tmp_path / "claude-template"
    template.mkdir()
    (template / "settings.json").write_text("{}")
    monkeypatch.setattr(harness, "CLAUDE_HOME", template)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cmd, env = harness._claude_code("do it", run_dir, "http://127.0.0.1:43210", 20)

    assert env == {"PAB_BASE_URL": "http://127.0.0.1:43210", "PAB_CONFIG_DIR": str(run_dir / "claude-home")}
    assert (run_dir / "claude-home" / "settings.json").exists()
    assert cmd[cmd.index("--tools") + 1] == harness.CLAUDE_TOOLS
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("-p") + 1] == "do it"


def test_run_records_exit_wall_time_and_harness_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_manifest: None
) -> None:
    script = "import json, os; print(json.dumps({'type': 'result', 'num_turns': 3, 'cwd': os.getcwd()}))"
    monkeypatch.setitem(harness.BUILDERS, "fake", lambda *a: ([sys.executable, "-c", script], {}))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    summary = harness.run("fake", "task", workspace, tmp_path / "runs" / "r1", timeout_s=60)

    assert summary["exit_code"] == 0
    assert summary["timed_out"] is False
    assert summary["harness_report"] == {"result_event": {"type": "result", "num_turns": 3}}
    assert json.loads((tmp_path / "runs" / "r1" / "stdout.log").read_text())["cwd"] == str(workspace.resolve())
    assert json.loads((tmp_path / "runs" / "r1" / "summary.json").read_text())["run_id"] == "r1"
    assert "listening on" in (tmp_path / "runs" / "r1" / "proxy.log").read_text()


def test_run_passes_the_token_command_and_rec_options_to_the_proxy_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_manifest: None
) -> None:
    printer = "print('tok-' + 'xyz')"  # the token never appears in the command itself
    monkeypatch.setenv(harness.UPSTREAM_TOKEN_CMD_ENV, f"{shlex.quote(sys.executable)} -c {shlex.quote(printer)}")
    script = f"import os; print({harness.UPSTREAM_TOKEN_CMD_ENV!r} in os.environ)"
    monkeypatch.setitem(harness.BUILDERS, "fake", lambda *a: ([sys.executable, "-c", script], {}))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_dir = tmp_path / "runs" / "r3"

    harness.run("fake", "task", workspace, run_dir, timeout_s=60, random_seed=True, one_reasoning_field=True)

    assert (run_dir / "stdout.log").read_text().strip() == "False"
    proxy_log = (run_dir / "proxy.log").read_text()
    assert "upstream auth: bearer token from" in proxy_log
    assert "sampling=inject:qwen3.8-gen-config+random-seed" in proxy_log
    assert "one_reasoning_field=on" in proxy_log
    assert not any("tok-xyz" in f.read_text(errors="replace") for f in run_dir.rglob("*") if f.is_file())


def test_localmelo_persists_memory_per_run(tmp_path: Path) -> None:
    cmd, env = harness._localmelo("-do it", tmp_path, "http://127.0.0.1:43210", 30)

    assert cmd[0].endswith("/.venv/bin/melo")
    assert cmd[-2:] == ["--", "-do it"]  # a task text that starts with "-" is not read as an option
    assert harness._with_harness_args(cmd, "-do it", ["--x", "1"])[-4:] == ["--x", "1", "--", "-do it"]
    assert cmd[cmd.index("--base-url") + 1] == "http://127.0.0.1:43210/v1"
    assert env == {"LOCALMELO_PERSIST_MEMORY": "1", "LOCALMELO_MEMORY_DIR": str(tmp_path / "mem")}


def _checkout(root: Path, main_py: str | None) -> Path:
    """A localmelo checkout as the bench sees it: a root __main__.py, or none at all when main_py is None."""
    root.mkdir(parents=True)
    if main_py is not None:
        (root / "__main__.py").write_text(main_py)
    return root


def test_localmelo_opts_out_of_session_recording_when_the_checkout_knows_the_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path / "localmelo", 'parser.add_argument("--no-save-session", action="store_true")\n')
    monkeypatch.setattr(harness.manifest, "LOCALMELO_REPO", checkout)

    cmd, _ = harness._localmelo("-do it", tmp_path, "http://127.0.0.1:43210", 30)

    assert cmd[-5:] == ["--chat-model", harness.MODEL, "--no-save-session", "--", "-do it"]


@pytest.mark.parametrize(
    "main_py",
    [pytest.param("parser.add_argument('--chat-model')\n", id="older-checkout"), pytest.param(None, id="no-main")],
)
def test_localmelo_older_checkout_gets_no_session_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main_py: str | None
) -> None:
    checkout = _checkout(tmp_path / "localmelo", main_py)
    monkeypatch.setattr(harness.manifest, "LOCALMELO_REPO", checkout)

    cmd, _ = harness._localmelo("-do it", tmp_path, "http://127.0.0.1:43210", 30)

    assert "--no-save-session" not in cmd
    assert cmd[-2:] == ["--", "-do it"]


@pytest.mark.parametrize("prompt", ["", "  \n"])
def test_localmelo_refuses_a_blank_task(tmp_path: Path, prompt: str) -> None:
    with pytest.raises(ValueError, match="non-blank"):
        harness._localmelo(prompt, tmp_path, "http://127.0.0.1:43210", 30)


def test_run_puts_harness_args_before_the_prompt_and_logs_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_manifest: None
) -> None:
    script = "import json, sys; print(json.dumps(sys.argv[1:]))"
    monkeypatch.setitem(
        harness.BUILDERS, "localmelo", lambda prompt, *a: ([sys.executable, "-c", script, "--base-url", "u", prompt], {})
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_dir = tmp_path / "runs" / "r4"

    summary = harness.run(
        "localmelo", "the task", workspace, run_dir, timeout_s=60, log_bodies=True,
        harness_args=["--tool-exposure", "all", "--output-format", "json"],
    )  # fmt: skip

    argv = json.loads((run_dir / "stdout.log").read_text())
    assert argv == ["--base-url", "u", "--tool-exposure", "all", "--output-format", "json", "the task"]
    assert summary["harness_args"] == ["--tool-exposure", "all", "--output-format", "json"]
    assert "log_bodies=on" in (run_dir / "proxy.log").read_text()


def test_harness_args_are_refused_for_other_harnesses(tmp_path: Path, no_manifest: None) -> None:
    with pytest.raises(ValueError, match="localmelo only"):
        harness.run("hermes", "task", tmp_path, tmp_path / "runs" / "r5", harness_args=["--x"])
    assert not (tmp_path / "runs").exists()
    # A command without a trailing prompt gets the flags at the end
    assert harness._with_harness_args(["melo", "--y"], "task", ["--x", "1"]) == ["melo", "--y", "--x", "1"]


def test_harness_reports_native_step_counts(tmp_path: Path) -> None:
    def write_events(events: list[dict[str, object]]) -> None:
        (tmp_path / "stdout.log").write_text("".join(json.dumps(e) + "\n" for e in events) + "not json\n")

    write_events([{"type": "result", "num_turns": 4, "subtype": "success", "is_error": False,
                   "permission_denials": [{"tool": "Bash"}], "duration_ms": 1200}])  # fmt: skip
    assert harness._harness_report("claude_code", tmp_path) == {
        "num_turns": 4, "subtype": "success", "is_error": False, "permission_denials": 1, "duration_ms": 1200,
    }  # fmt: skip

    write_events([{"type": "tool_use", "name": "terminal"}, {"type": "tool_use", "name": "file"},
                  {"type": "result", "exit_code": 0}])  # fmt: skip
    (tmp_path / "hermes-home").mkdir()
    with sqlite3.connect(tmp_path / "hermes-home" / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT, api_call_count INTEGER)")
        conn.executemany("INSERT INTO sessions VALUES (?, ?)", [("a", 3), ("b", 1)])
    assert harness._harness_report("hermes", tmp_path) == {"tool_use_events": 2, "api_call_count": 4, "exit_code": 0}

    (tmp_path / "mem").mkdir()
    with sqlite3.connect(tmp_path / "mem" / "history.db") as conn:
        conn.execute("CREATE TABLE steps (id INTEGER)")
        conn.executemany("INSERT INTO steps VALUES (?)", [(1,), (2,), (3,)])
    assert harness._harness_report("localmelo", tmp_path) == {  # no tasks table, text output
        "steps": 3, "task_status": None, "run_record": None,
    }  # fmt: skip
    with sqlite3.connect(tmp_path / "mem" / "history.db") as conn:
        conn.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY, query TEXT, status TEXT, result TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('t1', 'do it', 'failed', 'best effort')")
    assert harness._harness_report("localmelo", tmp_path) == {"steps": 3, "task_status": "failed", "run_record": None}

    # --output-format json: one RunResult line on stdout; the answer stays in stdout.log only
    write_events([{"task_id": "t1", "status": "stuck", "result": "long answer", "error": "", "steps": 4,
                   "n_llm": 4, "usage": {"prompt_tokens": 900}, "config": {"tool_exposure": "all"},
                   "localmelo_version": "0.1.0"}])  # fmt: skip
    record = harness._harness_report("localmelo", tmp_path)["run_record"]
    assert record == {"task_id": "t1", "status": "stuck", "error": "", "steps": 4, "n_llm": 4,
                      "usage": {"prompt_tokens": 900}, "config": {"tool_exposure": "all"},
                      "localmelo_version": "0.1.0"}  # fmt: skip


def test_run_kills_a_harness_that_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_manifest: None) -> None:
    monkeypatch.setitem(
        harness.BUILDERS, "fake", lambda *a: ([sys.executable, "-c", "import time; time.sleep(60)"], {})
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    summary = harness.run("fake", "task", workspace, tmp_path / "runs" / "r2", timeout_s=1)

    assert summary["timed_out"] is True
    assert summary["wall_s"] < 15
