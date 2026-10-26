"""The runner end to end, through bench.harness and its real recorder proxy, with fake harnesses.

As in personal-agent-bench's tests/test_harness.py, BUILDERS entries are monkeypatched. Each fake
harness is a small script that makes model calls through the proxy (answered by a local fake
upstream, so no model and no oMLX) and then solves, ignores, tampers with, sleeps through or
crashes on its task.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from marks import needs_bench, needs_data, needs_smoke_venv

from smoke import paths, runner, watch

pytestmark = [needs_bench, needs_data, needs_smoke_venv]

TESTS_DIR = Path(__file__).resolve().parent

FAKE_AGENT = r'''
import json, os, shutil, sys, time, urllib.request
from pathlib import Path

base_url, mode, slug, run_dir, tests_dir, polyglot = sys.argv[1:7]
seen = {"argv": sys.argv[1:], "claude_env_file": os.environ.get("CLAUDE_ENV_FILE"),
        "path_head": os.environ["PATH"].split(os.pathsep)[:2]}
Path(run_dir, "fake_seen.json").write_text(json.dumps(seen))
if mode == "crash":
    sys.exit(3)

def call(with_tools):
    body = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
    if with_tools:
        body["tools"] = [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}]
    request = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()

call(True); call(True); call(False)
ws = Path.cwd()
if mode == "solve":
    if slug == "warmup":
        (ws / "warmup.txt").write_text("ready\n")
    elif slug.startswith("aider__"):
        name = slug.split("__", 1)[1]
        stub = name.replace("-", "_") + ".py"
        shutil.copyfile(Path(polyglot, name, ".meta", "example.py"), ws / stub)
    else:
        sys.path.insert(0, tests_dir)
        from mcpmark_oracles import ORACLES
        ORACLES[slug.replace("__", "/")](ws)
elif mode == "tamper":
    for test in ws.glob("*_test.py"):
        test.write_text("def test_nothing():\n    assert True\n")
elif mode == "sleep":
    time.sleep(60)
elif mode == "versions" and slug == "aider__transpose":  # wrong, then the reference, then broken
    solution = ws / "transpose.py"
    solution.write_text("def transpose(text):\n    return text\n")
    time.sleep(0.3)
    call(True)
    shutil.copyfile(Path(polyglot, "transpose", ".meta", "example.py"), solution)
    time.sleep(0.3)
    call(True); call(False); call(True)
    solution.write_text("def transpose(text):\n    return ''\n")
    time.sleep(0.3)
print(json.dumps({"type": "result", "num_turns": 2, "subtype": "success", "is_error": False,
                  "permission_denials": [], "duration_ms": 5}))
'''

COMPLETION = {
    "id": "fake-1",
    "object": "chat.completion",
    "model": "fake",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server's method name)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """bench.harness with a fake upstream, a stub manifest and workspaces under tmp_path."""
    from bench import harness as module

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(module, "UPSTREAM", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(module.manifest, "collect", lambda: {"stub": True})
    monkeypatch.setattr(paths, "WS_ROOT", tmp_path / "ws")
    yield module
    server.shutdown()


def _fake(script: Path, mode: str, extra: list[str] | None = None, hermes_home: bool = False):  # noqa: ANN202
    def build(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        env: dict[str, str] = {}
        if hermes_home:  # what bench.harness._hermes leaves behind, for the MCP arm to edit
            home = run_dir / "hermes-home"
            home.mkdir()
            (home / "config.yaml").write_text("mcp_servers:\n  other:\n    command: /bin/false\n")
            env["HERMES_HOME"] = str(home)
        args = [base_url, mode, run_dir.name, str(run_dir), str(TESTS_DIR), str(paths.POLYGLOT_PYTHON)]
        return [sys.executable, str(script), *args, *(extra or [])], env

    return build


def _results(runs: Path) -> dict:
    (run_root,) = runs.iterdir()
    return json.loads((run_root / "results.json").read_text())


def test_suite_runs_end_to_end(tmp_path: Path, harness, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    monkeypatch.setitem(harness.BUILDERS, "claude_code", _fake(script, "solve"))
    monkeypatch.setitem(harness.BUILDERS, "hermes", _fake(script, "noop", hermes_home=True))
    monkeypatch.setitem(harness.BUILDERS, "localmelo", _fake(script, "sleep"))
    runs = tmp_path / "runs"

    code = runner.main(
        ["--task", "mcpmark/file_property/largest_rename", "--task", "aider/transpose", "--timeout", "4",
         "--skip-preflight", "--runs-dir", str(runs)],
        harness=harness,
    )  # fmt: skip

    assert code == 0
    doc = _results(runs)
    records = doc["records"]
    assert [(r["harness"], r["task"]) for r in records] == [
        (h, t)
        for h in ("claude_code", "hermes", "localmelo")
        for t in ("warmup", "mcpmark/file_property/largest_rename", "aider/transpose")
    ]
    by = {(r["harness"], r["task"]): r for r in records}
    for task in ("warmup", "mcpmark/file_property/largest_rename", "aider/transpose"):
        assert by[("claude_code", task)]["verdict"] == "pass"
        assert by[("hermes", task)]["verdict"] == "fail"
        assert by[("localmelo", task)]["verdict"] == "timeout"
        # Same bytes for every harness: the workspace path in the prompt is fixed per task
        assert len({by[(h, task)]["prompt_sha256"] for h in ("claude_code", "hermes", "localmelo")}) == 1
    solved = by[("claude_code", "aider/transpose")]
    assert (solved["n_llm"], solved["n_tool"], solved["n_llm_aux"]) == (2, 2, 1)
    assert solved["native_steps"] == "turns=2"
    assert solved["check"]["tests_modified"] is False
    assert by[("claude_code", "warmup")]["scored"] is False
    assert doc["finished"] is True
    assert doc["totals"]["claude_code"] == {"scored": 2, "pass": 2, "fail": 0, "timeout": 0, "harness_error": 0,
                                           "wall_s": doc["totals"]["claude_code"]["wall_s"]}  # fmt: skip
    # Workspaces moved into their run directories; nothing is left in ws/, lock included
    run_root = Path(doc["run_root"])
    assert (run_root / "claude_code" / "aider__transpose" / "workspace" / "transpose.py").exists()
    assert not any((tmp_path / "ws").iterdir())
    # Every harness got the same PATH head and the env file its shell hooks source
    env_file = run_root / "agent-env.sh"
    assert f'{paths.SMOKE_VENV_BIN}:$PATH' in env_file.read_text()
    for name in ("claude_code", "hermes", "localmelo"):
        seen = json.loads((run_root / name / "aider__transpose" / "fake_seen.json").read_text())
        assert seen["claude_env_file"] == str(env_file)
        assert seen["path_head"][1] == str(paths.SMOKE_VENV_BIN)
    hermes_config = (run_root / "hermes" / "aider__transpose" / "hermes-home" / "config.yaml").read_text()
    assert str(env_file) in hermes_config
    markdown = (run_root / "results.md").read_text()
    assert "| claude_code | aider/transpose | pass |" in markdown
    assert "Prompts byte-identical across harnesses: yes" in markdown
    system = json.loads((run_root / "results.json").read_text())["system"]
    assert system["macos_build"] and f"(build {system['macos_build']})" in markdown


def test_tampering_and_crashes_get_their_own_verdicts(tmp_path: Path, harness, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    monkeypatch.setitem(harness.BUILDERS, "claude_code", _fake(script, "tamper"))
    monkeypatch.setitem(harness.BUILDERS, "hermes", _fake(script, "crash", hermes_home=True))
    runs = tmp_path / "runs"

    runner.main(
        ["--harness", "claude_code,hermes", "--task", "aider/transpose", "--no-warmup", "--timeout", "10",
         "--skip-preflight", "--runs-dir", str(runs)],
        harness=harness,
    )  # fmt: skip

    tampered, crashed = _results(runs)["records"]
    assert tampered["verdict"] == "fail"
    assert tampered["check"]["tests_modified"] is True
    assert crashed["verdict"] == "harness_error"
    assert crashed["exit_code"] == 3 and crashed["n_llm"] == 0


def test_mcp_arm_reaches_the_harness_command_line(tmp_path: Path, harness, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    cc_flags = ["--tools", "Bash", "--allowedTools", "Bash", "--strict-mcp-config", "--mcp-config", "{}"]
    monkeypatch.setitem(harness.BUILDERS, "claude_code", _fake(script, "solve", cc_flags))
    monkeypatch.setitem(harness.BUILDERS, "hermes", _fake(script, "solve", ["-t", "terminal,file"], hermes_home=True))
    runs = tmp_path / "runs"

    runner.main(
        ["--mcp-arm", "--task", "mcpmark/papers/papers_counting", "--no-warmup", "--timeout", "10",
         "--skip-preflight", "--runs-dir", str(runs)],
        harness=harness,
    )  # fmt: skip

    doc = _results(runs)
    assert doc["arm"] == "mcp" and doc["not_runnable"] == ["localmelo (no MCP client)"]
    assert [r["verdict"] for r in doc["records"]] == ["pass", "pass"]
    run_root = Path(doc["run_root"])
    ws = paths.WS_ROOT / "mcpmark__papers__papers_counting"
    cc = json.loads((run_root / "claude_code" / "mcpmark__papers__papers_counting" / "fake_seen.json").read_text())["argv"]
    assert cc[cc.index("--tools") + 1] == "" and cc[cc.index("--allowedTools") + 1] == "mcp__fs__*"
    assert json.loads(cc[cc.index("--mcp-config") + 1])["mcpServers"]["fs"]["args"] == [str(paths.FS_SERVER_JS), str(ws)]
    hermes_run = run_root / "hermes" / "mcpmark__papers__papers_counting"
    hermes_argv = json.loads((hermes_run / "fake_seen.json").read_text())["argv"]
    assert hermes_argv[hermes_argv.index("-t") + 1] == "fs"
    assert f"- {ws}" in (hermes_run / "hermes-home" / "config.yaml").read_text()
    # The prompt names the workspace but offers no shell
    assert (hermes_run / "prompt.md").read_text().endswith(f"Work in {ws}.\n")


def test_dry_run_plans_without_touching_anything(tmp_path: Path, harness, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: ANN001
    runs = tmp_path / "runs"
    assert runner.main(["--dry-run", "--runs-dir", str(runs)], harness=harness) == 0
    out = capsys.readouterr().out
    assert "48 runs (45 scored)" in out
    assert not runs.exists() and not (tmp_path / "ws").exists()

    assert runner.main(["--dry-run", "--mcp-arm", "--runs-dir", str(runs)], harness=harness) == 0
    out = capsys.readouterr().out
    assert "22 runs (20 scored)" in out and "localmelo" in out and "aider/" not in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--mcp-arm", "--task", "aider/transpose"],
        ["--mcp-arm", "--harness", "localmelo"],
        ["--harness", "codex"],
        ["--harness", "hermes", "--localmelo-args=--tool-exposure all"],
        ["--repeat", "0"],
    ],
)
def test_invalid_selections_are_refused(argv: list[str], harness) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit):
        runner.main([*argv, "--dry-run"], harness=harness)


def test_localmelo_flags_body_logging_and_passes(tmp_path: Path, harness, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    monkeypatch.setitem(harness.BUILDERS, "localmelo", _fake(script, "noop"))
    runs = tmp_path / "runs"

    code = runner.main(
        ["--harness", "localmelo", "--task", "aider/transpose", "--task", "aider/proverb", "--repeat", "2",
         "--localmelo-args", "--tool-exposure all --builtins file_read,file_write", "--log-bodies",
         "--timeout", "10", "--skip-preflight", "--runs-dir", str(runs)],
        harness=harness,
    )  # fmt: skip

    assert code == 0
    doc = _results(runs)
    run_root = Path(doc["run_root"])
    records = doc["records"]
    # The warm-up once, then pass 1 over both tasks (in the suite's order), then pass 2 in the same order
    order = [r["task"] for r in records[1:3]]
    assert sorted(order) == ["aider/proverb", "aider/transpose"]
    assert [(r["task"], r["repeat"]) for r in records] == [("warmup", 1)] + [(t, n) for n in (1, 2) for t in order]
    assert [r["run_dir"] for r in records] == ["localmelo/warmup"] + [
        f"localmelo/{t.replace('/', '__')}__r{n}" for n in (1, 2) for t in order
    ]
    assert doc["settings"]["localmelo_args"] == ["--tool-exposure", "all", "--builtins", "file_read,file_write"]
    assert doc["settings"]["log_bodies"] is True and doc["settings"]["repeat"] == 2
    assert doc["totals"]["localmelo"]["scored"] == 4
    for record in records:
        run_dir = run_root / record["run_dir"]
        # The fake's command line has no trailing prompt, so the flags come last
        argv = json.loads((run_dir / "fake_seen.json").read_text())["argv"]
        assert argv[-4:] == ["--tool-exposure", "all", "--builtins", "file_read,file_write"]
        assert json.loads((run_dir / "summary.json").read_text())["harness_args"] == argv[-4:]
        assert "log_bodies=on" in (run_dir / "proxy.log").read_text()
        assert any((run_dir / "bodies").iterdir())
    markdown = (run_root / "results.md").read_text()
    assert "| localmelo | aider/transpose #2 |" in markdown
    assert "request and response bodies logged" in markdown and "2 passes over the tasks" in markdown
    assert "Prompts byte-identical across harnesses: yes" in markdown


def test_dry_run_counts_passes(tmp_path: Path, harness, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: ANN001
    argv = ["--dry-run", "--harness", "localmelo", "--task", "aider/transpose", "--repeat", "3"]
    assert runner.main([*argv, "--runs-dir", str(tmp_path / "runs")], harness=harness) == 0
    out = capsys.readouterr().out
    assert "4 runs (3 scored)" in out and "aider/transpose #3" in out and "3 passes over the tasks" in out


def test_rec_options_reach_the_proxy(tmp_path: Path, harness, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    monkeypatch.setitem(harness.BUILDERS, "localmelo", _fake(script, "noop"))
    runs = tmp_path / "runs"

    code = runner.main(
        ["--harness", "localmelo", "--task", "aider/transpose", "--no-warmup", "--random-seed",
         "--one-reasoning-field", "--skip-preflight", "--runs-dir", str(runs)],
        harness=harness,
    )  # fmt: skip

    assert code == 0
    doc = _results(runs)
    assert doc["settings"]["random_seed"] is True and doc["settings"]["one_reasoning_field"] is True
    proxy_log = (Path(doc["run_root"]) / "localmelo" / "aider__transpose" / "proxy.log").read_text()
    assert "sampling=inject:qwen3.8-gen-config+random-seed" in proxy_log
    assert "one_reasoning_field=on" in proxy_log


def test_watch_solution_counts_requests_after_the_first_pass(
    tmp_path: Path, harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]  # noqa: ANN001
) -> None:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    monkeypatch.setitem(harness.BUILDERS, "claude_code", _fake(script, "versions"))
    monkeypatch.setitem(harness.BUILDERS, "localmelo", _fake(script, "sleep"))
    monkeypatch.setattr(watch, "POLL_S", 0.05)
    runs = tmp_path / "runs"

    code = runner.main(
        ["--harness", "claude_code,localmelo", "--task", "aider/transpose", "--task", "mcpmark/file_property/largest_rename",
         "--no-warmup", "--timeout", "3", "--watch-solution", "--skip-preflight", "--runs-dir", str(runs)],
        harness=harness,
    )  # fmt: skip

    assert code == 0
    assert runner.WATCH_NOTE in capsys.readouterr().out
    doc = _results(runs)
    assert doc["settings"]["watch_solution"] is True
    run_root = Path(doc["run_root"])
    by = {(r["harness"], r["task"]): r for r in doc["records"]}
    solved = by[("claude_code", "aider/transpose")]
    assert solved["verdict"] == "fail"  # the last version is broken
    assert solved["check"]["extra_files"] == []  # the watcher never wrote into the workspace
    watched = solved["watch"]
    assert [c["passed"] for c in watched["checks"]] == [False, False, True, False]
    assert watched["versions"] == 4 and watched["first_pass_version"] == 2
    # After the reference landed: two calls with tools and one without (the three before it do not count)
    assert (watched["planning_after_first_pass"], watched["aux_after_first_pass"]) == (2, 1)
    assert watched["final_version_passes"] is False
    run_dir = run_root / solved["run_dir"]
    assert json.loads((run_dir / "summary.json").read_text())["watch"] == watched
    versions = [json.loads(line) for line in (run_dir / "solution_versions" / "versions.jsonl").read_text().splitlines()]
    assert [v["n"] for v in versions] == [0, 1, 2, 3] and {v["file"] for v in versions} == {"transpose.py"}
    assert all(v["t"] <= v["detected_t"] for v in versions)
    assert [v["t"] for v in versions] == sorted(v["t"] for v in versions)
    copies = run_dir / "solution_versions"
    assert (copies / "000-transpose.py").read_bytes() == (paths.POLYGLOT_PYTHON / "transpose" / "transpose.py").read_bytes()
    assert (copies / "002-transpose.py").read_bytes() == (paths.POLYGLOT_PYTHON / "transpose" / ".meta" / "example.py").read_bytes()
    # Killed at the budget: the watcher stops with the harness and keeps the stub only
    killed = by[("localmelo", "aider/transpose")]
    assert killed["verdict"] == "timeout"
    assert killed["watch"]["versions"] == 1 and killed["watch"]["first_pass_version"] is None
    assert killed["watch"]["planning_after_first_pass"] is None and killed["watch"]["final_version_passes"] is False
    # MCPMark tasks run unwatched
    for name in ("claude_code", "localmelo"):
        record = by[(name, "mcpmark/file_property/largest_rename")]
        assert record["watch"] is None and not (run_root / record["run_dir"] / "solution_versions").exists()


def test_preflight_requires_this_sessions_gpu_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / ".gpu-lock"
    monkeypatch.setattr(paths, "GPU_LOCK", lock)
    assert "is absent" in (runner.lock_problem() or "")

    lock.write_text("holder: local_ai-benchmark main agent\npurpose: timing\n")
    assert "held by another session (holder: local_ai-benchmark main agent)" in (runner.lock_problem() or "")

    lock.write_text("holder: agent-benchmark session (personal-agent-bench)\npurpose: smoke\n")
    assert runner.lock_problem() is None


class _TokenUpstream(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server's method name)
        self.send_response(200 if self.headers.get("Authorization") == "Bearer tok-abc" else 401)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def test_preflight_sends_the_upstream_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import harness as module

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TokenUpstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(module, "UPSTREAM", f"http://127.0.0.1:{server.server_port}")
    settings = runner.settings_from(runner.parse_args(["--harness", "localmelo", "--task", "aider/transpose"]))

    def server_problems(token_cmd: str | None) -> list[str]:
        if token_cmd is None:
            monkeypatch.delenv(module.UPSTREAM_TOKEN_CMD_ENV, raising=False)
        else:
            monkeypatch.setenv(module.UPSTREAM_TOKEN_CMD_ENV, token_cmd)
        return [p for p in runner.preflight(settings, module) if "model server" in p or "token command" in p]

    try:
        assert "401" in " ".join(server_problems(None))
        assert server_problems("printf tok-abc") == []
        assert "token command 'false'" in " ".join(server_problems("false"))
    finally:
        server.shutdown()
