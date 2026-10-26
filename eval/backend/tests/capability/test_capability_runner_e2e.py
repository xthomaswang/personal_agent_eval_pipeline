"""The capability runner end to end, as the orchestrator starts it: envs/capability's Python 3.13 with
-I -B, the allowlisted environment, a temporary epoch (git archive of the live personal-agent-bench
and harness-smoke, read only), the real recorder proxy, fake harness binaries and a mock server.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from modeleval import paths
from modeleval.capability import epoch as epochs
from modeleval.capability import memwatch, runner

from .conftest import CAPABILITY_PY, needs_envs, needs_live
from .fakes import SAMPLING, SERVED_ID, MockServer, set_modes

pytestmark = [needs_live, needs_envs]

SECRETS = {
    "OPENAI_API_KEY": "sk-test-secret",
    "ANTHROPIC_API_KEY": "sk-ant-test-secret",
    "GH_TOKEN": "ghp-test-secret",
    "CLAUDE_CODE_SESSION_ID": "session-of-the-driver",
    "DEVELOPER_DIR": "/Applications/Xcode-beta.app/Contents/Developer",
}
MCP = "mcpmark/file_property/largest_rename"
TASKS = [MCP, "aider/transpose", "aider/proverb"]  # the gate order: MCPMark first


def _spec(world: dict[str, Any], vendor: Path, epoch_id: str, server: MockServer, name: str, **kw: Any) -> Path:
    root = world["root"]
    spec = {
        "model": {
            "key": "test-model", "display": "Test", "served_id": SERVED_ID, "path": str(root / "model"),
            "sampling": SAMPLING, "proxy_sampling": "inject:omlx-default", "spec": "none", "vision": False,
        },
        "base_url": server.base_url, "server_pid": server.proc.pid, "sets": ["gate"],
        "run_root": str(root / "runs" / name / "capability"),
        "results": str(root / "results" / name / "capability" / "results.json"),
        "ws_root": str(world["ws_root"]), "epoch": epoch_id, "vendor": str(vendor / epoch_id),
        "wall_s": 120, "max_steps": 30, "max_turns": 40, "thinking": "off", "resume": True,
        "only": TASKS, "launch": 1,
        "selftest": {"battery": 100, "hold_file": str(root / "no-hold")},
        **kw,
    }  # fmt: skip
    path = root / f"spec-{name}.json"
    path.write_text(json.dumps(spec))
    return path


def _run(spec: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = env if env is not None else runner.allowed_env({**os.environ, **SECRETS})
    cmd = [str(CAPABILITY_PY), "-I", "-B", "-m", "modeleval.capability.runner", "--spec", str(spec)]
    return subprocess.run(cmd, cwd=paths.REPO, env=env, capture_output=True, text=True, timeout=900)


def _results(spec: Path) -> dict[str, Any]:
    return json.loads(Path(json.loads(spec.read_text())["results"]).read_text())


@pytest.fixture
def epoch_id(world: dict[str, Any], pinned_sha: str) -> str:
    return epochs.create_epoch(pinned_sha, pins_raw=world["pins_raw"], vendor_root=world["root"] / "vendor")


def _seen(run_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    return json.loads((run_root / row["run_dir"] / "fake_seen.json").read_text())


def test_runner_end_to_end_and_resume(world, epoch_id, mock_server, live_index_stamps) -> None:  # noqa: ANN001
    set_modes(world, {
        "claude_code": {"*": "solve"},
        "hermes": {"aider__transpose": "noop"},
        "localmelo": {"aider__transpose": "self-timeout", "aider__proverb": "self-timeout-solved"},
    })  # fmt: skip
    vendor = world["root"] / "vendor"
    spec = _spec(world, vendor, epoch_id, mock_server, "r1")
    proc = _run(spec)
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = proc.stdout.splitlines()
    assert sum(" TASK-START " in line for line in lines) == 12  # 3 warm-ups + 3 harnesses x 3 tasks
    assert sum(" TASK-DONE " in line for line in lines) == 12

    doc = _results(spec)
    run_root = Path(json.loads(spec.read_text())["run_root"])
    assert doc["finished"] is True and doc["epoch"] == epoch_id and doc["yield"] is None
    by = {(r["harness"], r["task"]): r for r in doc["rows"]}
    assert [(r["harness"], r["task"]) for r in doc["rows"]] == [(h, t) for h in runner.HARNESSES for t in TASKS]
    expect = {
        ("claude_code", "aider/transpose"): ("pass", True),
        ("claude_code", "aider/proverb"): ("pass", True),
        ("hermes", "aider/transpose"): ("fail", False),
        ("hermes", "aider/proverb"): ("pass", True),
        ("localmelo", "aider/transpose"): ("timeout", False),  # localmelo's own --timeout
        ("localmelo", "aider/proverb"): ("timeout", True),  # timed out, but the checker passes: solved
        **{(h, MCP): ("fail", False) for h in runner.HARNESSES},  # verify.py ran on the untouched folder
    }
    assert {k: (r["verdict"], r["solved"]) for k, r in by.items()} == expect
    for row in doc["rows"]:
        assert row["valid"] and row["set"] == "gate" and row["attempt"] == 1 and row["launch"] == 1
        assert row["calls"]["model_mismatch"] == row["calls"]["sampling_mismatch"] == row["calls"]["thinking_mismatch"] == 0
        assert row["calls"]["generation_calls"] == 2 and row["n_llm"] == 2 and row["model_s"] > 0
        assert row["t_end"] >= row["t_start"] and row["prompt_sha256"]
    assert by[("localmelo", "aider/transpose")]["localmelo_status"] == "timeout"
    assert {by[(h, MCP)]["check"]["kind"] for h in runner.HARNESSES} == {"mcpmark"}
    assert by[("hermes", "aider/proverb")]["check"]["kind"] == "aider"
    assert [w["verdict"] for w in doc["warmups"]] == ["pass", "pass", "pass"]
    assert doc["anomalies"] == {} and doc["superseded"] == []
    assert not any(name in doc["env_names"] for name in SECRETS)

    # The fakes saw the pipeline's environment, not the driver's
    shims, agent_bin = world["pins_raw"]["envs"]["shims"], "envs/agent-tools/.venv/bin"
    for row in doc["rows"]:
        seen = _seen(run_root, row)
        env = seen["env"]
        assert not set(SECRETS.values()) & set(env.values()), row["harness"]
        assert not {"OPENAI_API_KEY", "GH_TOKEN", "CLAUDE_CODE_SESSION_ID", "DEVELOPER_DIR"} & set(env), row["harness"]
        if row["harness"] != "claude_code":  # Claude Code's own env file exports ANTHROPIC_API_KEY=""
            assert "ANTHROPIC_API_KEY" not in env
        assert env["GIT_CEILING_DIRECTORIES"] == str(world["ws_root"])
        head = env["PATH"].split(os.pathsep)[:2]
        assert head == [str(paths.REPO / shims), str(paths.REPO / agent_bin)]
        assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["GIT_OPTIONAL_LOCKS"] == "0"
        assert seen["model"] == SERVED_ID
    claude = _seen(run_root, by[("claude_code", "aider/transpose")])["env"]
    assert claude["ANTHROPIC_MODEL"] == claude["ANTHROPIC_DEFAULT_SONNET_MODEL"] == SERVED_ID
    assert claude["ANTHROPIC_API_KEY"] == "" and claude["ANTHROPIC_AUTH_TOKEN"] == "omlx"
    assert claude["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == claude["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert claude["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "131072"
    claude_home = run_root / by[("claude_code", "aider/transpose")]["run_dir"] / "claude-home"
    assert claude["CLAUDE_CONFIG_DIR"] == str(claude_home)
    assert json.loads((claude_home / "settings.json").read_text())["model"] == SERVED_ID
    hermes = _seen(run_root, by[("hermes", "aider/proverb")])["env"]
    assert hermes["HERMES_HOME"] == str(run_root / by[("hermes", "aider/proverb")]["run_dir"] / "hermes-home")
    assert hermes["HERMES_DISABLE_LAZY_INSTALLS"] == "1"
    melo = _seen(run_root, by[("localmelo", "aider/transpose")])["argv"]
    assert melo[4:11] == ["--request-timeout", "1800", "--timeout", "60", "--output-format", "json", "--"]
    assert melo[:4] == ["--base-url", melo[1], "--chat-model", SERVED_ID]

    # The proxy forwarded the model's own sampling and thinking off, and the server got them
    upstream = [r for r in mock_server.requests() if r["path"] in ("/v1/messages", "/v1/chat/completions")]
    assert len(upstream) == 24
    for request in upstream:
        body = request["body"]
        assert {k: body.get(k) for k in SAMPLING} == SAMPLING and body["model"] == SERVED_ID
        assert body["chat_template_kwargs"]["enable_thinking"] is False and "reasoning_effort" not in body
    for row in doc["rows"]:
        assert "sampling=inject:omlx-default" in (run_root / row["run_dir"] / "proxy.log").read_text()

    # The allowlisted export, next to results.json
    results_dir = Path(json.loads(spec.read_text())["results"]).parent
    exported = sorted(p.relative_to(results_dir).as_posix() for p in results_dir.rglob("*") if p.is_file())
    assert all(runner.export.is_allowlisted(rel) for rel in exported)
    assert "claude_code/aider__transpose/calls.jsonl" in exported and "hermes/aider__proverb/check.txt" in exported
    assert "localmelo/mcpmark__file_property__largest_rename/check.txt" in exported
    assert not any(p.endswith(("prompt.md", "manifest.json", "stdout.log")) for p in exported)

    # Nothing compiled into the snapshot, nothing left in the workspaces, the lock released
    assert epochs.snapshot_problems(epochs.load_epoch(epoch_id, vendor)) == []
    assert sorted(p.name for p in world["ws_root"].iterdir()) == []
    selftest = json.loads((run_root / "selftest.json").read_text())
    assert selftest["smoke.SMOKE_PYTHON"] == str(paths.REPO / agent_bin / "python")

    # Resume: every pair has a valid verdict, so nothing runs again
    dirs = sorted(p for p in run_root.glob("*/*") if p.is_dir())
    again = _run(spec)
    assert again.returncode == 0, again.stderr[-2000:]
    assert " TASK-START " not in again.stdout
    assert sorted(p for p in run_root.glob("*/*") if p.is_dir()) == dirs
    assert _results(spec)["rows"] == doc["rows"]


def test_server_lost_then_resume_with_retries(world, epoch_id, mock_server, live_index_stamps) -> None:  # noqa: ANN001
    set_modes(world, {"claude_code": {"aider__transpose": "kill-server"}, "hermes": {"aider__transpose": "wrong-model"}})
    vendor = world["root"] / "vendor"
    spec = _spec(world, vendor, epoch_id, mock_server, "r2", only=["aider/transpose"], harnesses=["claude_code", "hermes"])
    proc = _run(spec)
    assert proc.returncode == runner.EXIT_SERVER_LOST, proc.stderr[-3000:]
    doc = _results(spec)
    (row,) = doc["rows"]
    assert (row["harness"], row["verdict"], row["valid"], row["solved"]) == ("claude_code", "server_lost", False, False)
    run_root = Path(json.loads(spec.read_text())["run_root"])
    assert json.loads((run_root / "yield.json").read_text())["reason"] == "server_lost"
    assert not (world["ws_root"] / ".lock").exists()

    # The orchestrator relaunches the server and restarts with resume
    mock_server.stop()
    relaunched = MockServer(world["root"], world["root"] / "server.pid").start()
    try:
        set_modes(world, {"hermes": {"aider__transpose": "wrong-model"}})
        spec2 = _spec(world, vendor, epoch_id, relaunched, "r2", only=["aider/transpose"],
                      harnesses=["claude_code", "hermes"], launch=2)  # fmt: skip
        proc = _run(spec2)
    finally:
        relaunched.stop()
    assert proc.returncode == 0, proc.stderr[-3000:]
    doc = _results(spec2)
    by = {r["harness"]: r for r in doc["rows"]}
    assert (by["claude_code"]["verdict"], by["claude_code"]["attempt"], by["claude_code"]["launch"]) == ("pass", 2, 2)
    assert by["claude_code"]["run_dir"] == "claude_code/aider__transpose__a2"
    # A pipeline mismatch is excluded and re-run once, then stays excluded
    assert (by["hermes"]["verdict"], by["hermes"]["invalid_reasons"], by["hermes"]["attempt"]) == ("invalid", ["model"], 2)
    superseded = [(r["harness"], r["verdict"], r["run_dir"]) for r in doc["superseded"]]
    assert superseded == [
        ("claude_code", "server_lost", "claude_code/aider__transpose"),
        ("hermes", "invalid", "hermes/aider__transpose"),
    ]
    # Each restart warms up again, in a new run directory
    assert [w["run_dir"] for w in doc["warmups"]] == ["claude_code/warmup", "claude_code/warmup__a2", "hermes/warmup"]
    assert (run_root / "hermes" / "aider__transpose__a2" / "calls.jsonl").exists()


@pytest.mark.parametrize("case", ["battery", "hold", "drift", "dead-server"])
def test_yields_before_any_task(world, epoch_id, mock_server, case: str) -> None:  # noqa: ANN001
    vendor = world["root"] / "vendor"
    selftest = {"battery": 100, "hold_file": str(world["root"] / "hold")}
    kw: dict[str, Any] = {}
    if case == "battery":
        selftest["battery"] = 45
    elif case == "hold":
        (world["root"] / "hold").write_text("")
    elif case == "drift":
        (world["hermes_repo"] / "dirty.txt").write_text("x\n")
    else:
        dead = subprocess.Popen(["/usr/bin/true"])
        dead.wait()
        kw["server_pid"] = dead.pid
    spec = _spec(world, vendor, epoch_id, mock_server, f"y-{case}", selftest=selftest, **kw)
    proc = _run(spec)
    expected = runner.EXIT_SERVER_LOST if case == "dead-server" else runner.EXIT_YIELD
    assert proc.returncode == expected, proc.stderr[-3000:]
    run_root = Path(json.loads(spec.read_text())["run_root"])
    reason = json.loads((run_root / "yield.json").read_text())["reason"]
    assert reason == {"battery": "battery", "hold": "hold", "drift": "drift", "dead-server": "server_lost"}[case]
    assert " TASK-START " not in proc.stdout
    doc = _results(spec)
    assert doc["yield"]["reason"] == reason and doc["rows"] == [] and doc["finished"] is False
    assert not (world["ws_root"] / ".lock").exists()


def test_preflight_refusals(world, epoch_id, mock_server) -> None:  # noqa: ANN001
    vendor = world["root"] / "vendor"
    spec = _spec(world, vendor, epoch_id, mock_server, "p1")
    # The driver's credentials must never reach the runner
    leaky = {**runner.allowed_env(os.environ), "OPENAI_API_KEY": "sk-x"}
    proc = _run(spec, env=leaky)
    assert proc.returncode == runner.EXIT_PREFLIGHT and "OPENAI_API_KEY" in proc.stderr
    # A model whose registry sampling differs from its proxy preset
    bad = json.loads(spec.read_text())
    bad["model"]["sampling"] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    spec.write_text(json.dumps(bad))
    proc = _run(spec)
    assert proc.returncode == runner.EXIT_PREFLIGHT and "proxy preset" in proc.stderr
    # A memory limit that is not a number of GiB
    proc = _run(_spec(world, vendor, epoch_id, mock_server, "p3"),
                env=runner.allowed_env({**os.environ, memwatch.PROC_LIMIT_ENV: "lots"}))  # fmt: skip
    assert proc.returncode == runner.EXIT_PREFLIGHT and memwatch.PROC_LIMIT_ENV in proc.stderr
    # A project marker above the workspace root
    (world["ws_root"].parent / "pyproject.toml").write_text("[project]\n")
    spec = _spec(world, vendor, epoch_id, mock_server, "p2")
    proc = _run(spec)
    assert proc.returncode == runner.EXIT_PREFLIGHT and "pyproject.toml" in proc.stderr


def test_a_stopped_runner_leaves_no_harness_behind(world, epoch_id, mock_server) -> None:  # noqa: ANN001
    set_modes(world, {"localmelo": {"aider__transpose": "sleep"}})
    spec = _spec(world, world["root"] / "vendor", epoch_id, mock_server, "s1", only=["aider/transpose"], harnesses=["localmelo"])
    cmd = [str(CAPABILITY_PY), "-I", "-B", "-m", "modeleval.capability.runner", "--spec", str(spec)]
    proc = subprocess.Popen(cmd, cwd=paths.REPO, env=runner.allowed_env(os.environ), stdout=subprocess.PIPE, text=True)
    seen = Path(json.loads(spec.read_text())["run_root"]) / "localmelo" / "aider__transpose" / "fake_seen.json"
    deadline = time.monotonic() + 120
    while not seen.exists() and time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.1)
    assert seen.exists(), "the sleeping fake never started"
    harness_pid = json.loads(seen.read_text())["pid"]
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == runner.EXIT_ERROR
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(harness_pid, 0)
    assert not (world["ws_root"] / ".lock").exists()


def test_memory_watchdog_kills_a_runaway_command(world, epoch_id, mock_server, live_index_stamps) -> None:  # noqa: ANN001
    """DESIGN §6.1: a command the agent runs that grows past the per-process limit (0.2 GiB here) is
    killed; the harness, the proxy and the task go on, the verdict is the checker's, and the kill is in
    the row's flags and on its TASK-DONE line. Other tasks get no kill and keep null flags."""
    set_modes(world, {"claude_code": {"aider__transpose": "memory-hog"}})
    spec = _spec(world, world["root"] / "vendor", epoch_id, mock_server, "m1",
                 only=["aider/transpose", "aider/proverb"], harnesses=["claude_code"])  # fmt: skip
    env = runner.allowed_env({**os.environ, memwatch.PROC_LIMIT_ENV: "0.2", memwatch.TREE_LIMIT_ENV: "0.3"})
    proc = _run(spec, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    doc = _results(spec)
    run_root = Path(json.loads(spec.read_text())["run_root"])
    by = {r["task"]: r for r in doc["rows"]}
    row = by["aider/transpose"]
    hog = json.loads((run_root / row["run_dir"] / "fake_hog.json").read_text())
    assert hog["returncode"] == -9
    assert (row["verdict"], row["solved"], row["valid"], row["n_llm"]) == ("pass", True, True, 2)
    (kill,) = row["flags"]["memory_kills"]
    assert kill["pid"] == hog["pid"] and kill["limit"] == "process" and kill["footprint_gib"] >= 0.2
    assert " -I -c import os, sys, time" in kill["cmdline"] and len(kill["cmdline"]) <= 120   # cut: path-dependent
    assert str(Path.home()) not in json.dumps(doc)
    assert by["aider/proverb"]["flags"] is None and [w["flags"] for w in doc["warmups"]] == [None]
    done = {line.split(" task=")[1].split()[0]: line for line in proc.stdout.splitlines() if " TASK-DONE " in line}
    assert {task: line.split("memory_kills=")[1].split()[0] for task, line in done.items()} == {
        "warmup": "0", "aider/transpose": "1", "aider/proverb": "0",
    }
    assert "memory watchdog: killed pid" in proc.stderr

