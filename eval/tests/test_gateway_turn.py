"""bench.gateway_turn and the harness's ``localmelo_gateway`` path: one task through a gateway chat (a fake gateway, no model)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

from bench import gateway_turn, harness
from memory.across import client
from memory.tests.across_fake import FakeGateway, FakeModelServer

EVAL = Path(__file__).resolve().parents[1]


def fake_melo(folder: Path, **gateway: Any) -> Path:
    """A ``melo`` whose ``serve`` is the fake gateway built with ``gateway``; it leaves serve.json (argv, cwd, pid) in its HOME."""
    folder.mkdir(parents=True, exist_ok=True)
    melo = folder / "melo"
    melo.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, threading\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(EVAL)!r})\n"
        "from memory.tests.across_fake import FakeGateway\n"
        "args = sys.argv[1:]\n"
        "if args[:1] != ['serve']:\n"
        "    sys.exit(2)\n"
        "home = Path(os.environ['HOME'])\n"
        "pairs = [args[i + 1] for i, a in enumerate(args) if a == '-X']\n"
        f"gateway = FakeGateway(home, pairs, port=int(args[args.index('--port') + 1]), **{gateway!r})\n"
        "(home / 'serve.json').write_text(json.dumps({'argv': args, 'cwd': os.getcwd(), 'pid': os.getpid()}))\n"
        "done = threading.Event()\n"
        "signal.signal(signal.SIGTERM, lambda *_: done.set())\n"
        "done.wait()\n"
        "gateway.stop()\n"
    )
    melo.chmod(0o755)
    return melo


def command(melo: Path, tmp_path: Path, prompt: str, *more: str) -> list[str]:
    (tmp_path / "prompt.md").write_text(prompt, encoding="utf-8")
    (tmp_path / "ws").mkdir(exist_ok=True)
    return [
        sys.executable, "-m", "bench.gateway_turn",
        "--melo", str(melo),
        "--home", str(tmp_path / "home"),
        "--run-dir", str(tmp_path / "run"),
        "--base-url", "http://127.0.0.1:9/v1",
        "--model", "flint",
        "--prompt-file", str(tmp_path / "prompt.md"),
        *more,
    ]  # fmt: skip


def turn(melo: Path, tmp_path: Path, prompt: str, *more: str) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run the command in the workspace, as the harness does; (the finished process, gateway_turn.json)."""
    done = subprocess.run(command(melo, tmp_path, prompt, *more), cwd=tmp_path / "ws", capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(EVAL)}, timeout=120, check=False)  # fmt: skip
    return done, json.loads((tmp_path / "run" / "gateway_turn.json").read_text())


def gone(pid: int, within: float = 5.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def served(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "home" / "serve.json").read_text())


def test_a_completed_turn_leaves_its_events_its_record_and_the_reply(tmp_path: Path) -> None:
    melo = fake_melo(tmp_path / "bin", tool_words={"list": "file_list"})
    done, record = turn(melo, tmp_path, "Please list the files of this folder.", "-X", "memory.across_chats=false",
                        "--set", "agent.window_messages=32", "--set", "ui.mode=developer")  # fmt: skip

    assert done.returncode == 0, done.stderr
    assert done.stdout == "Noted.\n"
    assert record["status"] == "completed" and record["gateway_exit"] == 0 and record["approvals_allowed"] == []
    assert (record["handed_off"], record["tasks"], record["tasks_timed_out"]) == (False, [], False)
    assert record["workspace"] == str((tmp_path / "ws").resolve()) and record["ttfe_ms"] is not None
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert record["events"] == len(events)
    types = [e["type"] for e in events]
    assert types[0] == "turn.started" and types[-1] == "turn.finished" and "tool.started" in types
    assert all(isinstance(e["ts"], float) and e["source"] == "chat" for e in events)
    info = served(tmp_path)
    assert info["argv"][:3] == ["serve", "--port", info["argv"][2]] and info["argv"][3:] == ["-X", "memory.across_chats=false"]
    assert info["cwd"] == str((tmp_path / "home").resolve())  # the gateway's own folder; the task's is the chat's workspace
    assert gone(info["pid"])
    config = tomllib.loads((tmp_path / "home" / ".cache" / "localmelo" / "config.toml").read_text())
    assert config["vllm"] == {"chat_url": "http://127.0.0.1:9/v1", "chat_model": "flint"}
    assert config["agent"] == {"window_messages": 32} and config["ui"] == {"mode": "developer"}
    assert config["embedding_backend"] == "none"
    assert "gateway.log" in os.listdir(tmp_path / "run")


def test_an_approval_the_turn_asks_for_is_allowed_and_counted(tmp_path: Path) -> None:
    # the fake fails a turn whose write was allowed (it stands for a memory test's refused write): the exit says so
    melo = fake_melo(tmp_path / "bin", approval_texts=("save",))
    done, record = turn(melo, tmp_path, "Please save this to notes.txt.")

    assert record["approvals_allowed"] == ["file_write"]
    assert record["status"] == "failed" and done.returncode == 1
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    resolved = next(e for e in events if e["type"] == "approval.resolved")
    assert resolved["data"]["outcome"] not in ("denied", "timeout")


def test_a_turn_over_its_time_is_asked_to_stop_and_exits_124(tmp_path: Path) -> None:
    melo = fake_melo(tmp_path / "bin", hang_texts=("wait",), hang_seconds=30.0)
    done, record = turn(melo, tmp_path, "Please wait for me.", "--timeout", "1")

    assert done.returncode == 124
    assert record["status"] == "timeout" and record["cancelled"] is True
    assert gone(served(tmp_path)["pid"])
    # the turn came back without its events: what the gateway stored of it is kept, so the run shows how far it got
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert record["events"] == len(events) >= 1 and {e["source"] for e in events} == {"chat"}
    assert events[0]["type"] == "turn.started"


def test_a_term_signal_stops_the_gateway_before_the_exit(tmp_path: Path) -> None:
    melo = fake_melo(tmp_path / "bin", hang_texts=("wait",), hang_seconds=60.0)
    proc = subprocess.Popen(command(melo, tmp_path, "Please wait for me."), cwd=tmp_path / "ws", stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": str(EVAL)}, start_new_session=True)  # fmt: skip
    try:
        end = time.monotonic() + 30
        while not (tmp_path / "home" / "serve.json").exists() and time.monotonic() < end:
            time.sleep(0.05)
        time.sleep(0.5)  # the turn is running by now
        os.killpg(proc.pid, signal.SIGTERM)  # as the harness stops a run over its limit: the gateway is in another group
        assert proc.wait(timeout=30) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
    record = json.loads((tmp_path / "run" / "gateway_turn.json").read_text())
    assert record["status"] == "stopped" and record["gateway_exit"] == 0
    assert gone(served(tmp_path)["pid"])
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert record["events"] == len(events) >= 1 and events[0]["type"] == "turn.started"  # read before the gateway stopped


def test_a_gateway_that_does_not_start_is_an_error_record(tmp_path: Path) -> None:
    melo = tmp_path / "melo"
    melo.write_text("#!/bin/sh\necho 'melo: error: no such thing' >&2\nexit 2\n")
    melo.chmod(0o755)
    done, record = turn(melo, tmp_path, "Hello.", "--ready-timeout", "10")

    assert done.returncode == 1
    assert record["status"] == "error" and "LaunchError" in record["error"]
    assert "gateway_turn:" in done.stderr
    assert not (tmp_path / "run" / "events.jsonl").exists()


def test_a_blank_task_is_refused_before_anything_starts(tmp_path: Path) -> None:
    melo = fake_melo(tmp_path / "bin")
    done = subprocess.run(command(melo, tmp_path, "  \n"), cwd=tmp_path / "ws", capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(EVAL)}, timeout=60, check=False)  # fmt: skip
    assert done.returncode != 0 and "blank" in done.stderr
    assert not (tmp_path / "home" / "serve.json").exists()


def test_allow_all_allows_any_tool_once(tmp_path: Path) -> None:
    gateway = FakeGateway(tmp_path / "home", approval_texts=("save",), approval_tool="shell_exec")
    try:
        gw = client.Gateway(gateway.base_url, client.read_token(tmp_path / "home"))
        result = client.run_turn(gw, "Please save it.", "k1", deny_approvals=False, allow_all=True, timeout_s=20)
        other = client.run_turn(gw, "Please save it.", "k2", deny_approvals=False, allow_tools=("file_write",), timeout_s=20)
    finally:
        gateway.stop()
    assert [a["tool"] for a in result.allowed] == ["shell_exec"] and not result.denied
    assert other.allowed == []  # a tool that is not named is left to the gateway without allow_all


class TaskGateway:
    """What follow_tasks asks of a gateway: the task list (a script of states, one row per poll), approvals, stops."""

    def __init__(self, polls: list[list[dict[str, Any]]]) -> None:
        self.polls, self.n, self.stops, self.answered = polls, 0, [], []

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        if (method, path) == ("GET", "/api/tasks"):
            rows = self.polls[min(self.n, len(self.polls) - 1)]
            self.n += 1
            return 200, {"tasks": rows}
        if method == "PATCH":
            self.stops.append((path, body))
            return 200, {}
        return 404, None

    def allow_once(self, request_id: str) -> bool:
        self.answered.append(request_id)
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def follow(gw: Any, budget: float, **kw: Any) -> tuple[dict[str, Any], Clock]:
    clock = Clock()
    return gateway_turn.follow_tasks(gw, budget, clock=clock, sleep=clock.sleep, **kw), clock


def test_a_run_without_tasks_does_not_wait() -> None:
    class NoTasks:  # a build without the route
        def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
            return 404, {"error": "not found"}

    got, clock = follow(NoTasks(), 100.0)
    assert got == {"tasks": [], "approvals_allowed": [], "timed_out": False} and clock.now == 0.0
    got, clock = follow(NoTasks(), 100.0, expect_task=True)  # a turn that handed off: the task gets a moment to show
    assert got["tasks"] == [] and clock.now == gateway_turn.QUIET_S


def test_the_work_a_chat_gave_to_tasks_is_waited_for_and_its_cards_allowed() -> None:
    card = {"request_id": "ap-1", "tool_name": "shell_exec"}
    gw = TaskGateway([
        [{"task_id": "t1", "state": "running", "pending_approvals": []}],
        [{"task_id": "t1", "state": "waiting", "pending_approvals": [card]}],
        [{"task_id": "t1", "state": "waiting", "pending_approvals": [card]}],  # still listed: not answered twice
        [{"task_id": "t1", "state": "done"}],
        [{"task_id": "t1", "state": "done"}, {"task_id": "t2", "state": "running"}],  # an ended task's successor goes on with it
        [{"task_id": "t1", "state": "done"}, {"task_id": "t2", "state": "failed"}],
    ])  # fmt: skip
    got, clock = follow(gw, 100.0)

    assert got["tasks"] == [{"task_id": "t1", "state": "done"}, {"task_id": "t2", "state": "failed"}]
    assert got["approvals_allowed"] == ["shell_exec"] and gw.answered == ["ap-1"]
    assert got["timed_out"] is False and gw.stops == []
    assert clock.now >= 5 + gateway_turn.QUIET_S  # it waited for the quiet time after the last task ended


def test_tasks_over_the_time_are_asked_to_stop() -> None:
    gw = TaskGateway([[{"task_id": "t1", "state": "running"}]] * 12 + [[{"task_id": "t1", "state": "cancelled"}]])
    got, clock = follow(gw, 10.0)

    assert got["timed_out"] is True and got["tasks"] == [{"task_id": "t1", "state": "cancelled"}]
    assert gw.stops and all(stop == ("/api/tasks/t1", {"state": "stop"}) for stop in gw.stops)
    assert clock.now < 10.0 + gateway_turn.STOP_WAIT_S

    stuck = TaskGateway([[{"task_id": "t1", "state": "running"}]])
    got, clock = follow(stuck, 0.0)
    assert got["timed_out"] is True and got["tasks"] == [{"task_id": "t1", "state": "running"}]
    assert clock.now >= gateway_turn.STOP_WAIT_S  # a task that does not stop is not waited for longer than that


def test_a_task_that_waits_for_a_resume_is_not_waited_for() -> None:
    gw = TaskGateway([[{"task_id": "t1", "state": "running"}], [{"task_id": "t1", "state": "interrupted"}]])
    got, clock = follow(gw, 600.0)

    assert got["timed_out"] is False and got["tasks"] == [{"task_id": "t1", "state": "interrupted"}]
    assert not gw.stops and clock.now < 60.0  # it does not change while this gateway runs: no stop, no wait to the budget


# --- the harness ---


def test_the_builder_runs_the_gateway_turn_with_the_checkout_under_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.manifest, "LOCALMELO_REPO", tmp_path / "localmelo")
    monkeypatch.setattr(harness, "MODEL", "flint")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cmd, env = harness._localmelo_gateway("do it", run_dir, "http://127.0.0.1:43210", 30)

    assert cmd[:3] == [sys.executable, "-m", "bench.gateway_turn"]
    assert cmd[cmd.index("--melo") + 1] == str(tmp_path / "localmelo" / ".venv" / "bin" / "melo")
    assert cmd[cmd.index("--home") + 1] == str(run_dir / "localmelo-home")
    assert cmd[cmd.index("--base-url") + 1] == "http://127.0.0.1:43210/v1"
    assert cmd[cmd.index("--model") + 1] == "flint"
    assert cmd[cmd.index("--prompt-file") + 1] == str(run_dir / "prompt.md")
    assert env["PYTHONPATH"] == str(EVAL)
    with pytest.raises(ValueError, match="non-blank"):
        harness._localmelo_gateway(" \n", run_dir, "http://127.0.0.1:43210", 30)


def test_a_run_through_the_gateway_is_recorded_as_shipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway -> recording proxy -> model server, end to end with fakes: the proxy only records, the block has the stream's figures."""
    model = FakeModelServer(["flint"])
    repo = tmp_path / "localmelo"
    fake_melo(repo / ".venv" / "bin", model_calls=True, tool_words={"list": "file_list"})
    monkeypatch.setattr(harness.manifest, "collect", lambda: {"stub": True})
    monkeypatch.setattr(harness.manifest, "LOCALMELO_REPO", repo)
    monkeypatch.setattr(harness, "MODEL", "flint")
    monkeypatch.setattr(harness, "UPSTREAM", model.base_url.removesuffix("/v1"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_dir = tmp_path / "runs" / "g1"
    try:
        summary = harness.run("localmelo_gateway", "Please list the files of this folder.", workspace, run_dir, timeout_s=120,
                              thinking="off", harness_args=["-X", "memory.across_chats=false"])  # fmt: skip
    finally:
        model.stop()

    assert summary["exit_code"] == 0 and summary["timed_out"] is False
    turn_report = summary["harness_report"]["turn"]
    assert turn_report["status"] == "completed" and turn_report["approvals_allowed"] == [] and turn_report["gateway_exit"] == 0
    block = summary["metrics"]
    assert block["model_calls"] == 1 and block["tool_calls"] == 1 and block["repeat_calls"] == 0
    assert block["failed_tools"] == 0 and block["approvals_asked"] == 0 and block["first_sign_s"] is not None
    proxy_log = (run_dir / "proxy.log").read_text()
    assert "thinking=passthrough" in proxy_log and "sampling=passthrough" in proxy_log  # whatever the caller's --thinking says
    sent = [b for b in model.bodies if "messages" in b]
    assert len(sent) == 1
    assert "chat_template_kwargs" not in sent[0] and "temperature" not in sent[0]  # nothing set on the way: as shipped
    info = json.loads((run_dir / "localmelo-home" / "serve.json").read_text())
    assert info["argv"][3:] == ["-X", "memory.across_chats=false"]
    assert gone(info["pid"])
    assert (run_dir / "stdout.log").read_text() == "Noted.\n"


def test_the_gateway_turn_gets_the_runs_limit_less_its_stop_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def builder(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> tuple[list[str], dict[str, str]]:
        return [sys.executable, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))"], {}

    monkeypatch.setattr(harness.manifest, "collect", lambda: {"stub": True})
    monkeypatch.setitem(harness.BUILDERS, "localmelo_gateway", builder)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for name, limit in (("a", 600.0), ("b", 50.0)):
        harness.run("localmelo_gateway", "task", workspace, tmp_path / "runs" / name, timeout_s=limit)
        seen.append(json.loads((tmp_path / "runs" / name / "stdout.log").read_text()))
    assert seen == [["--timeout", "510"], ["--timeout", "30"]]  # 600 less harness.GATEWAY_STOP_S; never under 30
    # no gateway_turn.json: the report says the turn left none
    assert json.loads((tmp_path / "runs" / "a" / "summary.json").read_text())["harness_report"] == {"turn": None}
