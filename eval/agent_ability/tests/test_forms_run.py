"""The forms driver (forms/run.py): the schedule, one run's record, the night contract. Fakes only: a fake harness
``run`` plays the agent over http against the real fixture site, with a fake Chrome. No model and no Chrome start."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from forms import arms, fixture, tasks
from forms import run as forms_run
from forms import tap as forms_tap
from test_forms_site import Browser, solve
from web import chrome
from web import run as web_run

CHROME_VERSION = "156.0.8078.80"
COMMAND = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new --user-data-dir=<profile>"
FAKE_ENV = {"sw_vers": "fake", "xcodebuild_version": "fake", "chrome_devtools_mcp": {"pinned": "1.11.0"}}


@dataclass
class Scenario:
    """What a fake run does."""

    submit: bool = True  # the right submissions
    reply: str | None = None  # None: the task's own
    status: str | None = "completed"  # melo's; None leaves no result line
    timed_out: bool = False
    raises: BaseException | None = None
    n_calls: int = 4
    n_aux: int = 1
    duration_s: float = 2.0
    reasoning_tokens: int | None = 30
    reasoning_chars: int = 120
    enable_thinking: bool | None = None  # what the requests carry; None: as the arm's config.toml says
    tap_calls: tuple[dict[str, Any], ...] = ({"tool": "click", "status": "ok"},)
    n_tool: int | None = None  # what melo counted; None: the tap's
    wall_s: float = 10.0


def parse_run_id(run_id: str) -> dict[str, Any]:
    _, task, repeat, arm = run_id.split("__")
    return {"task": task, "repeat": int(repeat[1:]), "arm": arm}


class Rig:
    """The fakes of one study, and what they saw."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.out = tmp_path / "study"
        self.original = self.build_original
        self.harness = SimpleNamespace(
            manifest=SimpleNamespace(
                LOCALMELO_REPO=Path("melo"), OMLX_BASE_URL="http://127.0.0.1:8500/v1", BENCH_MODEL="Old",
                CLAUDE_BIN=Path("claude"),
            ),
            UPSTREAM="http://127.0.0.1:8500",
            MODEL="Old",
            UPSTREAM_TOKEN_CMD_ENV="BENCH_UPSTREAM_TOKEN_CMD",
            BUILDERS={"localmelo": self.original},
            run=self.run,
        )  # fmt: skip
        self.script: Any = lambda run_id: Scenario()
        self.calls: list[SimpleNamespace] = []
        self.chromes: list[tuple[Path, str, str | None]] = []
        self.chrome_error: Exception | None = None
        self.after_run: Any = None

    def build_original(
        self, prompt: str, run_dir: Path, base_url: str, _turns: int
    ) -> tuple[list[str], dict[str, str]]:
        home = run_dir / "localmelo-home"
        home.mkdir(parents=True, exist_ok=True)
        return ["melo", "--base-url", f"{base_url}/v1", "--chat-model", "m", "--", prompt], {"HOME": str(home)}

    @property
    def executed(self) -> list[str]:
        return [c.run_id for c in self.calls]

    def run(self, name: str, prompt: str, workspace: Path, run_dir: Path, **kw: Any) -> dict[str, Any]:
        run_id = run_dir.parent.name
        scenario = self.script(run_id)
        run_dir.mkdir(parents=True)
        cmd, env = self.harness.BUILDERS["localmelo"](prompt, run_dir, "http://127.0.0.1:43999", kw["max_turns"])
        config = (Path(env["HOME"]) / ".cache" / "localmelo" / "config.toml").read_text()
        server = json.loads((run_dir.parent / "mcp.json").read_text())["mcpServers"][arms.SERVER]
        self.calls.append(
            SimpleNamespace(
                run_id=run_id, name=name, prompt=prompt, workspace=workspace, run_dir=run_dir, kw=kw, cmd=cmd,
                config=config, server=server, model=self.harness.MODEL,
            )
        )  # fmt: skip
        if scenario.raises:
            raise scenario.raises
        task = parse_run_id(run_id)["task"]
        site = SimpleNamespace(url=lambda path: server["env"][forms_tap.SITE_ENV] + path)
        reply = solve(task, Browser(site)) if scenario.submit else "I could not do it."
        reply = scenario.reply if scenario.reply is not None else reply
        off = "enable_thinking = false" in config if scenario.enable_thinking is None else not scenario.enable_thinking
        kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if off else {}
        rows = [
            {
                "endpoint": "chat_completions", "path": "/v1/chat/completions", "status": 200, "error": None,
                "stream_complete": True, "role": "main" if i < scenario.n_calls else "aux",
                "duration_s": scenario.duration_s, "thinking_forwarded": kwargs,
                "output_reasoning_chars": scenario.reasoning_chars,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "usage_raw": {"completion_tokens_details": {"reasoning_tokens": scenario.reasoning_tokens}},
                "request_tool_names": [f"{arms.TOOL_PREFIX}click"],
            }
            for i in range(scenario.n_calls + scenario.n_aux)
        ]  # fmt: skip
        rows.append({"endpoint": "other", "path": "/v1/embeddings", "status": 200, "duration_s": 0.5})
        (run_dir / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        record = None
        if scenario.status:
            record = {
                "task_id": "t1", "status": scenario.status, "result": reply, "error": "",
                "n_tool": len(scenario.tap_calls) if scenario.n_tool is None else scenario.n_tool,
            }  # fmt: skip
        (run_dir / "stdout.log").write_text(json.dumps(record) + "\n" if record else "")
        events = [{"event": "call", "dur_s": 1, **c} for c in scenario.tap_calls]
        (run_dir.parent / "tap.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        if self.after_run:
            self.after_run(run_id)
        return {
            "run_id": run_id, "harness": name, "started_at": time.time(), "exit_code": 0,
            "timed_out": scenario.timed_out, "wall_s": scenario.wall_s, "steps": {},
            "harness_report": {
                "run_record": None if record is None else {k: v for k, v in record.items() if k != "result"}
            },
        }  # fmt: skip

    @contextmanager
    def forms_chrome(self, chrome_dir: Path, start_url: str, *, expected_version: str | None = None) -> Iterator[Any]:
        self.chromes.append((chrome_dir, start_url, expected_version))
        if self.chrome_error:
            raise self.chrome_error
        chrome_dir.mkdir(parents=True)
        yield SimpleNamespace(
            browser_url="http://127.0.0.1:9333",
            meta={"headless": True, "command": COMMAND, "chrome_version": CHROME_VERSION},
        )

    def go(self, **kw: Any) -> int:
        kw.setdefault("chosen", [tasks.BY_ID["signin"], tasks.BY_ID["search"]])
        kw.setdefault("repeats", 1)
        return forms_run.run_study(self.out, harness=self.harness, localmelo=self.tmp / "melo", **kw)

    def state(self) -> dict[str, Any]:
        return json.loads((self.out / "forms.json").read_text())

    def record(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.out / "runs" / run_id / "record.json").read_text())

    def only(self) -> dict[str, Any]:
        (record,) = self.state()["runs"].values()
        return record

    def go_one(self, scenario: Scenario, task: str = "signin", arm: str = "T-as-is") -> dict[str, Any]:
        self.script = lambda run_id: scenario
        assert self.go(chosen=[tasks.BY_ID[task]], arm_names=(arm,)) == 0
        return self.only()


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    rig = Rig(tmp_path)
    monkeypatch.setattr(forms_run, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(forms_run, "forms_chrome", rig.forms_chrome)
    monkeypatch.setattr(forms_run.chrome, "chrome_version", lambda *_: CHROME_VERSION)
    monkeypatch.setattr(forms_run.runner, "lock_problem", lambda: None)
    monkeypatch.setattr(web_run, "battery_problem", lambda *_: None)
    monkeypatch.setattr(web_run, "system_snapshot", lambda *a, **k: FAKE_ENV)
    return rig


# --- the schedule ---


def test_the_default_study_is_eight_tasks_two_arms_three_repeats() -> None:
    planned = forms_run.schedule(tasks.TASKS)
    assert len(planned) == 48 and len({p["run_id"] for p in planned}) == 48
    assert Counter((p["task"], p["arm"]) for p in planned) == {
        (t.id, a): 3 for t in tasks.TASKS for a in arms.DEFAULT_ARMS
    }
    assert planned[0]["run_id"] == "001__signin__r1__T-off"  # task 0, repeat 1: the rotation starts at arm 1


def test_the_order_is_repeat_major_with_the_arms_back_to_back_and_rotated() -> None:
    planned = forms_run.schedule(tasks.TASKS, 2)
    pairs = [planned[i : i + 2] for i in range(0, len(planned), 2)]
    assert [(p[0]["repeat"], p[0]["task"]) for p in pairs] == [(r, t.id) for r in (1, 2) for t in tasks.TASKS]
    firsts = [p[0]["arm"] for p in pairs]
    assert Counter(firsts) == {"T-as-is": 8, "T-off": 8}
    for repeat in (1, 2):  # the same as web/run.py's rotation
        for i, task in enumerate(tasks.TASKS):
            group = [p["arm"] for p in planned if p["repeat"] == repeat and p["task"] == task.id]
            assert group == web_run.rotated(list(arms.DEFAULT_ARMS), i + repeat)


def test_a_third_arm_is_only_one_more_cell() -> None:
    planned = forms_run.schedule(tasks.TASKS[:2], 1, ("T-as-is", "T-off", "T-step"))
    assert len(planned) == 6 and [p["position"] for p in planned[:3]] == [1, 2, 3]


def test_the_schedule_validates() -> None:
    with pytest.raises(ValueError, match="repeats"):
        forms_run.schedule(tasks.TASKS, 0)
    with pytest.raises(ValueError, match="empty"):
        forms_run.schedule([], 1)


def test_tasks_and_arms_are_selected_by_name() -> None:
    assert forms_run.select_tasks("all") == list(tasks.TASKS)
    assert [t.id for t in forms_run.select_tasks("search, signin,search")] == ["search", "signin"]
    with pytest.raises(ValueError, match="unknown task nope"):
        forms_run.select_tasks("nope")
    assert forms_run.select_arms(None, arms.ARMS) == ["T-as-is", "T-off"]
    with pytest.raises(ValueError, match="unknown arm T-on"):
        forms_run.select_arms("T-on", arms.ARMS)


# --- one run ---


def test_a_run_that_submits_the_right_fields_passes_on_the_sites_records(rig: Rig) -> None:
    record = rig.go_one(Scenario())
    assert record["verdict"] == "pass" and record["passed"] is True and record["end_reason"] == "completed"
    assert record["submissions"] == {"taskmere.signin:accepted": 1}
    assert rig.record(record["run_id"]) == record
    logged = (rig.out / "runs" / record["run_id"] / "site" / "submissions.jsonl").read_text()
    assert json.loads(logged)["fields"] == {"username": "rowan.ellis", "ok": True}


def test_saying_so_passes_nothing(rig: Rig) -> None:
    record = rig.go_one(Scenario(submit=False, reply="I signed in."))
    assert record["verdict"] == "fail" and record["check"]["reason"] == "nothing was submitted"


def test_the_search_task_passes_on_its_answer(rig: Rig) -> None:
    assert rig.go_one(Scenario(), task="search")["verdict"] == "pass"


def test_a_wrong_answer_fails_the_search(rig: Rig) -> None:
    record = rig.go_one(Scenario(reply="FINAL ANSWER: HC-19382-P"), task="search")
    assert record["verdict"] == "fail" and "HC-19382-P" in record["check"]["reason"]


def test_the_metrics_come_from_the_proxy_and_the_tap(rig: Rig) -> None:
    taps = (
        {"tool": "click", "status": "ok"},
        {"tool": "fill", "status": "tool_error"},
        {"tool": "x", "status": "refused"},
    )
    record = rig.go_one(Scenario(tap_calls=taps, n_tool=5, wall_s=20.0, n_calls=4, n_aux=1, duration_s=3.0))
    assert record["n_calls"] == 5 and record["n_main"] == 4 and record["n_aux"] == 1
    assert record["model_s"] == 15.0 and record["model_share"] == 0.75 and record["longest_call_s"] == 3.0
    assert record["n_embedding_calls"] == 1 and record["embedding_s"] == 0.5
    assert record["reasoning_tokens"] == 150 and record["reasoning_chars"] == 600
    assert record["n_tool_tap"] == 3 and record["n_tool_melo"] == 5
    assert record["refused_by_tap"] == 1 and record["refused_by_melo"] == 2 and record["refused_calls"] == 3
    assert record["tool_errors"] == 1 and record["tools"] == {"click": 1, "fill": 1, "x": 1}
    assert record["foreign_tools"] == []


def test_no_reasoning_tokens_in_the_usage_means_chars_are_all_there_is(rig: Rig) -> None:
    record = rig.go_one(Scenario(reasoning_tokens=None))
    assert record["reasoning_tokens"] is None and record["reasoning_chars"] == 600


def test_failed_calls_count_in_the_time_and_not_as_steps(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    rows = [
        {"endpoint": "chat_completions", "status": 200, "role": "main", "duration_s": 1.0},
        {"endpoint": "chat_completions", "status": 500, "role": "main", "duration_s": 4.0, "error": "boom"},
    ]
    calls.write_text("".join(json.dumps(r) + "\n" for r in rows))
    metrics = forms_run.model_metrics(calls, arms.ARMS["T-as-is"])
    assert metrics["n_calls"] == 1 and metrics["n_failed_calls"] == 1 and metrics["model_s"] == 5.0


def test_the_arm_reaches_the_config_and_the_record_proves_it(rig: Rig) -> None:
    rig.go(chosen=[tasks.BY_ID["signin"]])
    configs = {parse_run_id(c.run_id)["arm"]: c.config for c in rig.calls}
    assert "enable_thinking = false" in configs["T-off"] and "enable_thinking" not in configs["T-as-is"]
    runs = {r["arm"]: r for r in rig.state()["runs"].values()}
    assert runs["T-off"]["arm_applied"] is True and runs["T-off"]["thinking_off_calls"] == 5
    assert runs["T-as-is"]["arm_applied"] is True and runs["T-as-is"]["thinking_off_calls"] == 0


def test_an_arm_that_did_not_reach_the_server_is_seen(rig: Rig) -> None:
    assert rig.go_one(Scenario(enable_thinking=True), arm="T-off")["arm_applied"] is False


def test_the_budget_the_cap_and_passthrough_reach_the_harness(rig: Rig) -> None:
    rig.go(
        chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",), wall_cap_s=300.0, max_calls=20, model="M", log_bodies=True
    )
    (call,) = rig.calls
    assert call.kw["timeout_s"] == 300.0 and call.kw["max_steps"] == 20 and call.kw["thinking"] == "passthrough"
    assert call.kw["log_bodies"] is True and call.model == "M" and call.name == "localmelo"
    flags = call.kw["harness_args"]
    assert flags[flags.index("-t") + 1] == "270" and "agent.max_calls=19" in flags and "--mcp-only" in flags
    assert call.cmd[:2] == ["melo", "--"] and 'chat_model = "M"' in call.config


def test_each_run_gets_its_site_its_chrome_and_its_workspace(rig: Rig) -> None:
    rig.go(chosen=[tasks.BY_ID["job_upload"]], arm_names=("T-off",), repeats=2)
    assert len(rig.chromes) == 2 and rig.chromes[0][1] != rig.chromes[1][1]  # a new port each time
    assert rig.chromes[0][1].endswith("/careers/jobs/PM-3310") and rig.chromes[0][2] == CHROME_VERSION
    for call in rig.calls:
        assert (call.workspace / tasks.RESUME_NAME).read_bytes() == tasks.resume_bytes()
        assert str(call.workspace / tasks.RESUME_NAME) in call.prompt
        assert call.server["env"][forms_tap.UPLOAD_ROOT_ENV] == str(call.workspace)


def test_the_record_keeps_the_command_line_of_a_headless_chrome(rig: Rig) -> None:
    record = rig.go_one(Scenario())
    assert record["headless"] is True and "--headless=new" in record["chrome_command"].split()
    assert record["melo_chrome_profile"] is False


def test_a_melo_chrome_profile_in_the_run_home_is_recorded(rig: Rig) -> None:
    def make_profile(run_id: str) -> None:
        home = rig.out / "runs" / run_id / "localmelo" / "localmelo-home"
        (home / ".cache" / "localmelo" / "browser" / "chrome").mkdir(parents=True)

    rig.after_run = make_profile
    assert rig.go_one(Scenario())["melo_chrome_profile"] is True


# --- how a run ends ---


@pytest.mark.parametrize(
    ("scenario", "reason", "verdict"),
    [
        (Scenario(), "completed", "pass"),
        (Scenario(submit=False, status="budget_exhausted"), "budget_exhausted", "fail"),
        (Scenario(submit=False, status="timeout"), "timeout", "fail"),
        (Scenario(submit=False, status="stuck"), "stuck", "fail"),
        (Scenario(submit=False, status="provider_error"), "provider_error", "fail"),
        (Scenario(submit=False, timed_out=True, status=None), "wall_cap", "fail"),
        (Scenario(submit=False, status=None), "no_record", "fail"),
        (Scenario(submit=False, reply="Which account should I use?"), "asked_person", "fail"),
    ],
)
def test_the_reason_a_run_ended_is_recorded(rig: Rig, scenario: Scenario, reason: str, verdict: str) -> None:
    record = rig.go_one(scenario)
    assert record["end_reason"] == reason and record["verdict"] == verdict and record["passed"] is (verdict == "pass")


def test_a_question_after_the_work_is_done_still_passes(rig: Rig) -> None:
    record = rig.go_one(Scenario(reply="Done. Anything else?"))
    assert record["verdict"] == "pass" and record["end_reason"] == "completed"


def test_a_run_that_hangs_past_the_wall_cap_is_a_failure_and_the_study_goes_on(rig: Rig) -> None:
    rig.script = lambda run_id: (
        Scenario(submit=False, timed_out=True, status=None) if "signin" in run_id else Scenario()
    )
    assert rig.go(arm_names=("T-off",)) == 0
    assert [(r["task"], r["verdict"], r["end_reason"]) for r in rig.state()["runs"].values()] == [
        ("signin", "fail", "wall_cap"),
        ("search", "pass", "completed"),
    ]


def test_a_chrome_that_will_not_run_stops_the_study_and_keeps_no_record(rig: Rig) -> None:
    rig.chrome_error = chrome.ChromeError("the launched Chrome is not headless: x")
    assert rig.go() == 3
    assert rig.state()["runs"] == {} and rig.calls == []
    assert "not headless" in (rig.out / "forms.log").read_text()


def test_a_harness_that_raises_after_a_model_call_is_recorded(rig: Rig) -> None:
    def raising(run_id: str) -> None:
        raise RuntimeError("proxy gone")

    rig.after_run = raising
    record = rig.go_one(Scenario())
    assert record["verdict"] == "harness_error" and record["end_reason"] == "harness_error"
    assert "proxy gone" in record["error"]


# --- no model ---


def test_a_run_without_a_model_call_is_dropped_and_run_once_more(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    pauses: list[float] = []
    seen: Counter[str] = Counter()

    def script(run_id: str) -> Scenario:
        seen[run_id] += 1
        return Scenario(n_calls=0, n_aux=0, status="provider_error") if seen[run_id] == 1 else Scenario()

    rig.script = script
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",), sleep=pauses.append) == 0
    assert rig.only()["verdict"] == "pass" and sum(pauses) == forms_run.NO_MODEL_PAUSE_S
    run_id = rig.only()["run_id"]
    assert (rig.out / "runs" / f"{run_id}.dropped-1").is_dir() and (rig.out / "runs" / run_id).is_dir()


def test_two_runs_without_a_model_call_stop_the_study_and_a_resume_runs_them(rig: Rig) -> None:
    rig.script = lambda run_id: Scenario(n_calls=0, n_aux=0, status="provider_error")
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",), sleep=lambda _: None) == 3
    assert rig.state()["runs"] == {} and rig.state()["complete"] is False
    rig.script = lambda run_id: Scenario()
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",)) == 0
    assert rig.only()["verdict"] == "pass"


# --- the night contract ---


def test_a_study_runs_everything_once_and_a_rerun_replays_nothing(rig: Rig) -> None:
    assert rig.go() == 0
    assert len(rig.executed) == 4 and rig.state()["complete"] is True
    assert rig.go() == 0
    assert len(rig.executed) == 4


def test_the_stop_file_stops_between_runs_with_exit_3_and_a_rerun_resumes(rig: Rig) -> None:
    rig.after_run = lambda run_id: (rig.out / "STOP").touch() if len(rig.calls) == 2 else None
    assert rig.go() == 3
    assert len(rig.executed) == 2 and rig.state()["complete"] is False
    assert "STOP file found" in (rig.out / "forms.log").read_text()
    (rig.out / "STOP").unlink()
    rig.after_run = None
    assert rig.go() == 0
    assert len(rig.executed) == 4 and len(set(rig.executed)) == 4


def test_a_stop_file_before_the_first_run_is_no_reason_to_refuse_the_directory(rig: Rig) -> None:
    rig.out.mkdir()
    (rig.out / "STOP").touch()
    assert rig.go() == 3 and rig.executed == []


def test_past_the_deadline_nothing_starts(rig: Rig) -> None:
    assert rig.go(deadline=time.time() - 1) == 3
    assert rig.executed == [] and "deadline" in (rig.out / "forms.log").read_text()


def test_a_run_that_could_pass_the_deadline_does_not_start(rig: Rig) -> None:
    assert rig.go(deadline=time.time() + 100, wall_cap_s=900.0) == 3
    assert rig.executed == []


def test_a_far_deadline_changes_nothing_and_is_not_part_of_the_study(rig: Rig) -> None:
    assert rig.go(deadline=time.time() + 100_000) == 0
    assert rig.go(deadline=time.time() + 200_000) == 0  # the next night's deadline may differ


def test_a_lost_gpu_lock_stops_the_study(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forms_run.runner, "lock_problem", lambda: "held by another session")
    assert rig.go() == 3 and rig.executed == []
    assert rig.go(gpu_lock=False) == 0


def test_a_cut_off_run_is_dropped_and_run_again(rig: Rig) -> None:
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",), repeats=2) == 0
    state = rig.state()
    first = next(iter(state["runs"]))
    (rig.out / "runs" / first / "record.json").unlink()
    state["runs"][first] = {"state": "started"}
    state["complete"] = False
    (rig.out / "forms.json").write_text(json.dumps(state))
    (rig.out / "runs" / first / "leftover").touch()
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",), repeats=2) == 0
    assert rig.executed.count(first) == 2 and rig.state()["runs"][first]["state"] == "finished"
    assert (rig.out / "runs" / f"{first}.dropped-1" / "leftover").exists()
    assert "found cut off" in (rig.out / "forms.log").read_text()


def test_a_started_run_with_a_record_is_adopted(rig: Rig) -> None:
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",)) == 0
    state = rig.state()
    run_id = next(iter(state["runs"]))
    state["runs"][run_id] = {"state": "started"}
    (rig.out / "forms.json").write_text(json.dumps(state))
    assert rig.go(chosen=[tasks.BY_ID["signin"]], arm_names=("T-off",)) == 0
    assert len(rig.executed) == 1 and rig.only()["verdict"] == "pass"


def test_sigterm_during_a_run_drops_it_and_exits_3(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[Path] = []
    monkeypatch.setattr(forms_run, "kill_leftovers", killed.append)
    rig.after_run = lambda run_id: os.kill(os.getpid(), signal.SIGTERM)
    before = signal.getsignal(signal.SIGTERM)
    assert rig.go() == 3
    assert signal.getsignal(signal.SIGTERM) is before
    assert rig.state()["runs"] == {} and len(killed) == 1
    run_id = rig.executed[0]
    assert (rig.out / "runs" / f"{run_id}.dropped-1").is_dir()
    assert not (rig.out / "ws" / ".lock").exists()
    rig.after_run = None
    assert rig.go() == 0 and rig.state()["complete"] is True


def test_a_stale_lock_does_not_refuse_the_rerun(rig: Rig) -> None:
    assert rig.go() == 0
    (rig.out / "ws").mkdir()
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    (rig.out / "ws" / ".lock").write_text(f"{dead.pid}\n")
    assert rig.go() == 0


def test_a_live_lock_still_refuses(rig: Rig) -> None:
    assert rig.go() == 0
    (rig.out / "ws").mkdir()
    (rig.out / "ws" / ".lock").write_text(f"{os.getpid()}\n")
    with pytest.raises(SystemExit, match="another run"):
        rig.go()


def test_a_changed_study_needs_a_fresh_directory(rig: Rig) -> None:
    assert rig.go() == 0
    with pytest.raises(ValueError, match="changed"):
        rig.go(wall_cap_s=100.0)
    with pytest.raises(ValueError, match="changed"):
        rig.go(model="Other")


def test_a_full_directory_without_a_checkpoint_is_refused(rig: Rig) -> None:
    rig.out.mkdir()
    (rig.out / "something").touch()
    with pytest.raises(ValueError, match="fresh directory"):
        rig.go()


def test_a_refused_preflight_starts_nothing(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forms_run, "preflight", lambda *a, **k: ["no melo"])
    assert rig.go() == 2 and rig.executed == [] and not rig.out.exists()


def test_a_chrome_update_between_nights_refuses_the_resume(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    assert rig.go() == 0
    monkeypatch.setattr(forms_run.chrome, "chrome_version", lambda *_: "157.0.0.1")
    assert rig.go() == 2


def test_the_end_writes_the_report_and_copies_it_for_the_night(rig: Rig, tmp_path: Path) -> None:
    night = tmp_path / "night"
    night.mkdir()
    assert rig.go(night_dir=str(night)) == 0
    assert "## Per arm" in (rig.out / "report.md").read_text()
    assert (night / "forms-study.md").read_text() == (rig.out / "report.md").read_text()


def test_a_report_that_fails_changes_nothing_about_the_exit(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forms_run.report, "write", lambda out: {}["boom"])
    assert rig.go() == 0
    assert "the report failed" in (rig.out / "forms.log").read_text()


def test_the_builder_is_restored_and_the_study_settings_too(rig: Rig) -> None:
    assert rig.go(base_url="http://127.0.0.1:43000/v1", model="M") == 0
    assert rig.harness.BUILDERS["localmelo"] == rig.original
    assert rig.harness.UPSTREAM == "http://127.0.0.1:8500" and rig.harness.MODEL == "Old"
    assert json.loads((rig.out / "environment.json").read_text())["base_url"] == "http://127.0.0.1:43000/v1"


# --- what is around a run ---


def test_stop_reason_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forms_run.runner, "lock_problem", lambda: "no lock")
    monkeypatch.setattr(web_run, "battery_problem", lambda *_: "on battery at 10%")
    options = {"deadline": None, "wall_cap_s": 900.0}
    assert forms_run.stop_reason(tmp_path, **options) == "no lock"
    assert forms_run.stop_reason(tmp_path, gpu_lock=False, **options) == "on battery at 10%"
    assert "deadline" in forms_run.stop_reason(tmp_path, deadline=time.time(), wall_cap_s=900.0)
    (tmp_path / "STOP").touch()
    assert forms_run.stop_reason(tmp_path, **options) == "STOP file found"


def test_only_the_runs_own_melo_is_killed(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    code = "import time; time.sleep(60)"
    ours = subprocess.Popen([sys.executable, "-c", code, "--mcp", str(run_root / "mcp.json")], start_new_session=True)
    other = subprocess.Popen(
        [sys.executable, "-c", code, "--mcp", str(tmp_path / "other" / "mcp.json")], start_new_session=True
    )
    try:
        time.sleep(0.2)
        assert forms_run.kill_leftovers(run_root) == [ours.pid]
        assert ours.wait(timeout=10) == -signal.SIGTERM
        assert other.poll() is None
    finally:
        other.kill()
        ours.kill()


def test_the_fixture_preflight_reads_every_start_page() -> None:
    assert forms_run.fixture_problems() == []


def test_the_real_preflight_leaves_the_model_alone_in_a_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no(*_: Any) -> str:
        raise AssertionError("a dry run must not call the model server")

    monkeypatch.setattr(web_run, "model_problem", no)
    monkeypatch.setattr(forms_run.web_arms, "localmelo_problems", lambda *_: [])
    monkeypatch.setattr(forms_run.web_arms, "mcp_problems", lambda *_: [])
    monkeypatch.setattr(forms_run.chrome, "preflight", lambda *_: [])
    monkeypatch.setattr(forms_run.chrome, "chrome_version", lambda *_: CHROME_VERSION)
    monkeypatch.setattr(forms_run.runner, "_binaries", lambda _: {"localmelo": Path(sys.executable)})
    harness = SimpleNamespace()
    assert forms_run.preflight(harness, tmp_path, tmp_path, gpu_lock=False, check_model=False) == []
    monkeypatch.setattr(web_run, "model_problem", lambda _: "the model server is not reachable")
    assert forms_run.preflight(harness, tmp_path, tmp_path, gpu_lock=False) == ["the model server is not reachable"]


# --- the command line ---


@pytest.fixture
def dry(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> Rig:
    monkeypatch.setattr(forms_run, "preflight", lambda *a, **k: [])
    return rig


def test_the_dry_run_prints_the_plan_and_starts_nothing(dry: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    code = forms_run.main(["--run-dir", str(dry.out), "--dry-run"], harness=dry.harness)
    text = capsys.readouterr().out
    assert code == 0 and dry.executed == [] and not dry.out.exists()
    assert "48 runs = 8 tasks x 3 repeats x 2 arms" in text and "at most 720 GPU minutes" in text
    assert "not contacted in a dry run" in text and "Preflight: ok" in text
    assert "Arm T-off" in text and '"enable_thinking": false' in text
    assert "--mcp <mcp.json> --mcp-only -o json" in text
    assert " 48. picture_grid  r3 " in text


def test_the_dry_run_reports_a_problem_with_exit_2(
    dry: Rig, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forms_run, "preflight", lambda *a, **k: ["the lock is held"])
    assert forms_run.main(["--run-dir", str(dry.out), "--dry-run"], harness=dry.harness) == 2
    assert "PROBLEMS" in capsys.readouterr().out


def test_the_night_variables_are_the_defaults(dry: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(forms_run, "run_study", lambda out, **kw: seen.update(kw, out=out) or 0)
    monkeypatch.setenv("NIGHT_URL", "http://127.0.0.1:8500/v1")
    monkeypatch.setenv("NIGHT_DEADLINE", "1793990000")
    monkeypatch.setenv("NIGHT_DIR", "/night/20261106")
    assert forms_run.main(["--run-dir", str(dry.out)], harness=dry.harness) == 0
    assert seen["base_url"] == "http://127.0.0.1:8500/v1" and seen["deadline"] == 1793990000.0
    assert seen["night_dir"] == "/night/20261106" and seen["repeats"] == 3 and seen["gpu_lock"] is True
    assert seen["model"] == "Qwen3.8-Flash-Next-IQ3XXS-splash" and seen["embedding_model"] == "Qwen3-Embedding-0.6B"
    assert seen["expect_sha"] == arms.LOCALMELO_SHA and seen["localmelo"] == arms.LOCALMELO.resolve()


def test_no_embedding_model_can_be_asked_for(dry: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(forms_run, "run_study", lambda out, **kw: seen.update(kw) or 0)
    forms_run.main(["--run-dir", str(dry.out), "--embedding-model", "none"], harness=dry.harness)
    assert seen["embedding_model"] is None


def test_a_third_arm_comes_in_by_the_arms_file(dry: Rig, tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({"T-step": {"settings": {"agent.thinking_steps": "first"}}}))
    rig_args = [
        "--run-dir",
        str(dry.out),
        "--arms-file",
        str(path),
        "--arms",
        "T-as-is,T-off,T-step",
        "--tasks",
        "signin",
    ]
    rig_args += ["--repeats", "1"]
    assert forms_run.main(rig_args, harness=dry.harness) == 0
    assert len(dry.executed) == 3
    step = next(c for c in dry.calls if c.run_id.endswith("T-step"))
    assert "thinking_steps" in step.config


def test_bad_arguments_end_with_a_message(dry: Rig) -> None:
    with pytest.raises(SystemExit, match="unknown task"):
        forms_run.main(["--run-dir", str(dry.out), "--tasks", "nope"], harness=dry.harness)
    with pytest.raises(SystemExit, match="collides"):
        forms_run.main(["--run-dir", str(dry.out), "--localmelo-args=--mcp x"], harness=dry.harness)


def test_the_fake_agent_really_goes_through_the_site(rig: Rig) -> None:
    """The fakes above lean on this: a run's site is a real server, gone after the run."""
    seen: list[str] = []
    rig.after_run = lambda run_id: seen.append(next(iter(rig.calls)).server["env"][forms_tap.SITE_ENV])
    rig.go_one(Scenario())
    with pytest.raises(OSError):
        chrome.http_get(seen[0] + "/forum/", timeout=1.0)


def test_runs_do_not_share_a_thread_with_the_server_after_stopping() -> None:
    before = threading.active_count()
    site = fixture.Site()
    site.stop()
    assert threading.active_count() <= before
