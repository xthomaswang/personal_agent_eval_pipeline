"""The web study runner (web/run.py): the schedule, one run's verdict and records, resume, the CAPTCHA pause and the
guards. Fakes only: a fake harness ``run``, a fake ``bench_chrome`` and a synthetic History. No harness, no model
server and no Chrome starts."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from marks import needs_bench
from test_web_arms import CLAUDE_ENV_TEXT, CLAUDE_SETTINGS_TEXT, HERMES_TEMPLATE
from test_web_chrome import CHAIN, make_history
from web import arms, chrome, data
from web import run as web_run

PILOT = data.load()
EASY, MEDIUM = 3503, 1934
SERP = "https://www.google.com/search?q=south+africa+state+funeral&hl=en"
WIKI = "https://en.wikipedia.org/wiki/Marais_Viljoen"
SORRY = "https://www.google.com/sorry/index?continue=https://www.google.com/search%3Fq%3Dx"
CHROME_VERSION = "156.0.8078.80"
ALL_HARNESSES = ("claude_code", "hermes", "localmelo")
MODEL = "Some-Model-oQ4e"
URL = "http://127.0.0.1:43999"
EXTRA = ("-X", "agent.reflection_gate=patient", "-X", "agent.answer_on_stop=on")
OTHER_SHA = "5fa22ee" + "1" * 33


def gold(question: int) -> str:
    return PILOT.by_id(question).answer


@dataclass
class Scenario:
    """What a fake run does. ``reply`` None means the gold answer of the run's question."""

    reply: str | None = None
    visits: list[tuple[float, str]] = field(default_factory=lambda: [(1.0, SERP), (3.0, WIKI)])
    tabs: list[dict[str, str]] | None = None
    timed_out: bool = False
    n_llm: int = 3
    n_aux: int = 0
    raises: Exception | None = None
    tools: tuple[str, ...] = ("mcp__webtap__navigate_page", "mcp__webtap__take_snapshot")
    usage: tuple[int, int, int] = (1000, 600, 50)  # prompt, cached, completion tokens per call
    tap_calls: tuple[dict[str, Any], ...] = ()


def parse_run_id(run_id: str) -> dict[str, Any]:
    parts = run_id.split("__")
    return {
        "question": int(parts[1][1:]),
        "repeat": int(parts[2][1:]),
        "harness": parts[3],
        "arm": parts[4],
        "retry": len(parts) > 5,
    }


class Rig:
    """The fakes of one study, and what they saw."""

    def __init__(self, tmp_path: Path) -> None:
        from bench import harness as real

        self.tmp = tmp_path
        self.out = tmp_path / "study"
        self.original = {name: (lambda *a, _n=name: ([_n], {})) for name in ("claude_code", "hermes", "localmelo")}
        # what bench.harness would source and copy for Claude Code, and the binary the study pins
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "claude-code.env").write_text(CLAUDE_ENV_TEXT)
        (templates / "claude-home").mkdir()
        (templates / "claude-home" / "settings.json").write_text(CLAUDE_SETTINGS_TEXT)
        self.claude_env = templates / "claude-code.env"
        self.claude_bin = tmp_path / "bin" / "claude-2.1.312"
        self.claude_bin.parent.mkdir()
        self.claude_bin.write_text("#!/bin/sh\necho 2.1.312\n")
        self.claude_bin.chmod(0o755)
        self.harness = SimpleNamespace(
            manifest=SimpleNamespace(
                LOCALMELO_REPO=Path("original-melo"),
                OMLX_BASE_URL="http://127.0.0.1:8500/v1",
                BENCH_MODEL="Old-Model",
                CLAUDE_BIN=Path("original-claude"),
            ),
            UPSTREAM="http://127.0.0.1:8500",
            MODEL="Old-Model",
            CLAUDE_ENV=self.claude_env,
            CLAUDE_HOME=templates / "claude-home",
            UPSTREAM_TOKEN_CMD_ENV=real.UPSTREAM_TOKEN_CMD_ENV,
            BUILDERS=dict(self.original),
            run=self.run,
        )
        self.script: Any = lambda run_id: Scenario()
        self.calls: list[SimpleNamespace] = []
        self.chrome_calls: list[tuple[Path, str | None]] = []
        self.headless: list[bool] = []
        self.taps: list[list[str]] = []  # the tap's command line of each run, as arms.py built it
        self.chrome_error: Exception | None = None
        self.t0: dict[str, float] = {}
        self.sleeps: list[float] = []
        self.chrome_version = CHROME_VERSION

    @property
    def executed(self) -> list[str]:
        return [c.run_id for c in self.calls]

    def scenario(self, run_id: str) -> Scenario:
        found = self.script(run_id)
        question = parse_run_id(run_id)["question"]
        return (
            found
            if found.reply is not None
            else Scenario(**{**found.__dict__, "reply": f"FINAL ANSWER: {gold(question)}"})
        )

    # the fake bench.harness.run
    def run(self, name: str, prompt: str, workspace: Path, run_dir: Path, **kw: Any) -> dict[str, Any]:
        run_id = run_dir.parent.name
        swapped = name in ("claude_code", "hermes") and self.harness.BUILDERS[name] is not self.original[name]
        self.calls.append(
            SimpleNamespace(
                run_id=run_id,
                name=name,
                prompt=prompt,
                workspace=workspace,
                run_dir=run_dir,
                kw=kw,
                swapped=swapped,
                claude_env=self.harness.CLAUDE_ENV,
                model=self.harness.MODEL,
                tap_args=list(self.taps[-1]) if self.taps else None,
            )
        )
        scenario = self.scenario(run_id)
        if scenario.raises:
            raise scenario.raises
        run_dir.mkdir(parents=True)
        t0 = time.time()
        self.t0[run_id] = t0
        reply = scenario.reply or ""
        lines: list[dict[str, Any]]
        if name == "claude_code":
            lines = [{"type": "system"}, {"type": "result", "subtype": "success", "result": reply}]
        elif name == "hermes":
            lines = [{"type": "system", "subtype": "init"}, {"type": "result", "exit_code": 0, "text": reply}]
        else:
            lines = [{"task_id": "t1", "status": "completed", "result": reply}]
        (run_dir / "stdout.log").write_text("".join(json.dumps(line) + "\n" for line in lines))
        prompt_tokens, cached, completion = scenario.usage
        role = "aux" if parse_run_id(run_id)["arm"] == "C" else "main"
        calls = [
            {
                "endpoint": "chat_completions", "status": 200, "error": None, "stream_complete": True, "role": role,
                "usage": {"prompt_tokens": prompt_tokens, "cached_tokens": cached, "completion_tokens": completion},
                "request_tool_names": list(scenario.tools),
            }
            for _ in range(scenario.n_llm + scenario.n_aux)
        ]  # fmt: skip
        (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
        tap_log = run_dir.parent / "tap.jsonl"
        if parse_run_id(run_id)["arm"] != "C":
            events = [
                {"event": "start"},
                *({"event": "call", "status": "ok", "dur_s": 1, **c} for c in scenario.tap_calls),
            ]
            tap_log.write_text("".join(json.dumps(e) + "\n" for e in events))
        return {
            "run_id": run_id, "harness": name, "started_at": t0, "exit_code": 0, "timed_out": scenario.timed_out,
            "wall_s": 5.0,
            "steps": {
                "n_llm": scenario.n_llm if role == "main" else 0,
                "n_llm_aux": scenario.n_aux if role == "main" else scenario.n_llm + scenario.n_aux,
                "n_llm_err": 0, "n_tool": max(scenario.n_llm - 1, 0), "images_new_sum": 0,
            },
            "harness_report": {},
        }  # fmt: skip

    # the fake chrome.bench_chrome: History is written when "Chrome" closes, as the real one commits it
    @contextmanager
    def bench_chrome(
        self, chrome_dir: Path, *, expected_version: str | None = None, headless: bool = True, **_: Any
    ) -> Iterator[SimpleNamespace]:
        self.chrome_calls.append((chrome_dir, expected_version))
        self.headless.append(headless)
        if self.chrome_error:
            raise self.chrome_error
        chrome_dir.mkdir(parents=True)
        try:
            yield SimpleNamespace(
                browser_url=f"http://127.0.0.1:{9000 + len(self.chrome_calls)}",
                chrome=SimpleNamespace(proc=SimpleNamespace(pid=5000 + len(self.chrome_calls))),
            )
        finally:
            run_id = chrome_dir.parent.name
            scenario = self.scenario(run_id)
            t0 = self.t0.get(run_id, time.time())
            rows = [(t0 - 30, "https://www.google.com/?hl=en", CHAIN | 6, 0)]  # the start page, before t0
            rows += [(t0 + offset, url, CHAIN | 0, 0) for offset, url in scenario.visits]
            make_history(chrome_dir / "History", rows)
            tabs = scenario.tabs if scenario.tabs is not None else [
                {"id": "1", "title": "Google", "url": scenario.visits[-1][1] if scenario.visits else "about:blank"}
            ]  # fmt: skip
            (chrome_dir / "tabs.json").write_text(json.dumps(tabs))

    def context(self, *, chrome_version: str | None = CHROME_VERSION, **kw: Any) -> web_run.Context:
        kw.setdefault("model", MODEL)
        kw.setdefault("claude_bin", self.claude_bin)
        kw.setdefault("sleep", self.sleeps.append)
        return web_run.Context(
            self.out, self.harness, {r.id: r for r in PILOT.rows}, self.tmp / "cwd", chrome_version, **kw
        )

    def item(
        self, question: int = EASY, harness: str = "claude_code", arm: str = "D", repeat: int = 1, index: int = 1
    ) -> dict[str, Any]:
        return {
            "index": index, "question": question, "tier": PILOT.by_id(question).tier, "repeat": repeat,
            "harness": harness, "arm": arm, "position": 1, "attempt": 1,
            "run_id": f"{index:03d}__q{question}__r{repeat}__{harness}__{arm}",
        }  # fmt: skip

    def go(self, **kwargs: Any) -> int:
        kwargs.setdefault("rows", [PILOT.by_id(EASY), PILOT.by_id(MEDIUM)])
        kwargs.setdefault("repeats", 1)
        kwargs.setdefault("harnesses", ("claude_code", "hermes"))
        kwargs.setdefault("arm_names", ("D", "C"))
        kwargs.setdefault("captcha_pause_s", 0.0)
        kwargs.setdefault("model", MODEL)
        kwargs.setdefault("claude_bin", self.claude_bin)
        return web_run.run_study(
            self.out, localmelo=self.tmp / "melo", harness=self.harness, sleep=self.sleeps.append, **kwargs
        )

    def state(self) -> dict[str, Any]:
        return json.loads((self.out / "web.json").read_text())

    def record(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.out / "runs" / run_id / "record.json").read_text())


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    rig = Rig(tmp_path)
    monkeypatch.setattr(web_run, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(web_run, "installed_chrome_version", lambda: rig.chrome_version)
    monkeypatch.setattr(
        web_run,
        "system_snapshot",
        lambda localmelo, chrome_version, **kw: {
            "sw_vers": "fake",
            "chrome_version": chrome_version,
            "model": kw["model"],
        },
    )
    monkeypatch.setattr(
        web_run, "stop_reason", lambda out, *, gpu_lock=True: "STOP file found" if (out / "STOP").exists() else None
    )
    monkeypatch.setattr(web_run, "bench_chrome", rig.bench_chrome)
    real_tap_args = arms.tap_args

    def recording(*a: Any, **k: Any) -> list[str]:
        rig.taps.append(real_tap_args(*a, **k))
        return rig.taps[-1]

    monkeypatch.setattr(arms, "tap_args", recording)
    return rig


# --- the schedule ---


def test_stage_zero_is_two_questions_by_nine_cells() -> None:
    rows = [PILOT.by_id(EASY), PILOT.by_id(MEDIUM)]
    planned = web_run.schedule(rows, 1, ALL_HARNESSES, ("D", "S", "C"))
    assert len(planned) == 18
    assert [p["index"] for p in planned] == list(range(1, 19))
    assert len({p["run_id"] for p in planned}) == 18
    assert (
        planned[0]["run_id"] == "001__q3503__r1__claude_code__S"
    )  # question 0, repeat 1: the rotation starts at cell 1
    assert Counter((p["harness"], p["arm"]) for p in planned) == {(h, a): 2 for h in ALL_HARNESSES for a in "DSC"}


def test_the_order_is_repeat_major_with_all_cells_back_to_back() -> None:
    rows = list(PILOT.rows)[:4]
    planned = web_run.schedule(rows, 2, ("claude_code", "hermes"), ("D", "S", "C"))
    assert len(planned) == 2 * 4 * 6
    groups = [planned[i : i + 6] for i in range(0, len(planned), 6)]
    assert [(g[0]["repeat"], g[0]["question"]) for g in groups] == [(r, q.id) for r in (1, 2) for q in rows]
    for group in groups:
        assert {(p["repeat"], p["question"]) for p in group} == {(group[0]["repeat"], group[0]["question"])}
        assert len({(p["harness"], p["arm"]) for p in group}) == 6
        assert [p["position"] for p in group] == [1, 2, 3, 4, 5, 6]


def test_the_cells_rotate_by_question_index_plus_repeat() -> None:
    harnesses, arm_names = ("claude_code", "hermes"), ("D", "S", "C")
    cells = [(h, a) for h in harnesses for a in arm_names]
    rows = list(PILOT.rows)
    planned = web_run.schedule(rows, 2, harnesses, arm_names)
    firsts = []
    for gi in range(len(rows) * 2):
        group = planned[gi * 6 : gi * 6 + 6]
        repeat, question_index = divmod(gi, len(rows))
        offset = (question_index + repeat + 1) % 6
        assert [(p["harness"], p["arm"]) for p in group] == cells[offset:] + cells[:offset]
        firsts.append((group[0]["harness"], group[0]["arm"]))
    assert set(firsts) == set(cells)  # no cell is always first
    assert max(Counter(firsts).values()) - min(Counter(firsts).values()) <= 1


def test_the_schedule_validates() -> None:
    rows = [PILOT.by_id(EASY)]
    with pytest.raises(ValueError, match="repeats"):
        web_run.schedule(rows, 0)
    with pytest.raises(ValueError, match="unknown arm"):
        web_run.schedule(rows, 1, ("claude_code",), ("D", "X"))
    with pytest.raises(ValueError, match="unknown harness"):
        web_run.schedule(rows, 1, ("some_agent",), ("D",))
    with pytest.raises(ValueError, match="empty"):
        web_run.schedule([], 1)


def test_the_retry_of_a_run_is_the_same_cell_in_a_directory_of_its_own() -> None:
    item = web_run.schedule([PILOT.by_id(EASY)], 1, ("hermes",), ("D",))[0]
    again = web_run.retry_of(item)
    assert (
        again["run_id"] == item["run_id"] + "__retry" and again["attempt"] == 2 and again["retry_of"] == item["run_id"]
    )
    assert {k: again[k] for k in ("question", "harness", "arm", "repeat")} == {
        k: item[k] for k in ("question", "harness", "arm", "repeat")
    }


def test_questions_are_ids_tiers_or_all() -> None:
    assert [r.id for r in web_run.select_questions("3503,1934", PILOT)] == [3503, 1934]
    assert [r.id for r in web_run.select_questions("1934, 3503", PILOT)] == [1934, 3503]
    assert len(web_run.select_questions("easy", PILOT)) == 12 and len(web_run.select_questions("medium", PILOT)) == 12
    assert len(web_run.select_questions("all", PILOT)) == 24 == len(web_run.select_questions(None, PILOT))
    assert [r.id for r in web_run.select_questions("3503,easy", PILOT)][:2] == [3503, 1337]  # no repeats
    assert len(web_run.select_questions("3503,easy", PILOT)) == 12
    with pytest.raises(ValueError, match="unknown question"):
        web_run.select_questions("12", PILOT)
    with pytest.raises(ValueError, match="unknown question"):
        web_run.select_questions("hard", PILOT)


def test_harness_names_take_aliases() -> None:
    assert web_run.parse_names("cc,melo,hermes", arms.HARNESSES, "harness") == ["claude_code", "localmelo", "hermes"]
    assert web_run.parse_names(None, arms.ARMS, "arm", arms.DEFAULT_ARMS) == ["D", "S", "C"]  # S-cua only when named
    assert web_run.parse_names("D,S-cua", arms.ARMS, "arm", arms.DEFAULT_ARMS) == ["D", "S-cua"]
    with pytest.raises(ValueError, match="unknown arm"):
        web_run.parse_names("D,Z", arms.ARMS, "arm")


def test_budget_and_caps_are_the_designs() -> None:
    assert web_run.BUDGET == {"max_steps": 30, "max_turns": 40, "thinking": "off"}
    assert web_run.WALL_CAP_S == {"D": 1800.0, "S": 1800.0, "C": 600.0, "S-cua": 1800.0}
    assert web_run.CAPTCHA_PAUSE_S == 1800.0


# --- the final reply of each harness ---


def test_claude_code_reply_is_the_result_event(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "thinking"}, {"type": "tool_use"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "FINAL ANSWER: from the message"}]}},
        {"type": "result", "subtype": "success", "result": "FINAL ANSWER: from the result"},
    ]
    path.write_text("not json\n" + "".join(json.dumps(e) + "\n" for e in events))
    assert web_run.final_text("claude_code", path) == "FINAL ANSWER: from the result"
    path.write_text("".join(json.dumps(e) + "\n" for e in events[:2]))  # killed before the result event
    assert web_run.final_text("claude_code", path) == "FINAL ANSWER: from the message"


def test_hermes_reply_is_the_result_text_or_the_deltas_after_the_last_tool(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    events = [
        {"type": "text", "text": "Let me look. "},
        {"type": "tool_use", "name": "mcp__webtap__take_snapshot"},
        {"type": "tool_result", "name": "mcp__webtap__take_snapshot", "output": "..."},
        {"type": "text", "text": "FINAL ANSWER: "},
        {"type": "text", "text": "Delhi"},
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert web_run.final_text("hermes", path) == "FINAL ANSWER: Delhi"
    path.write_text(
        path.read_text() + json.dumps({"type": "result", "exit_code": 0, "text": "FINAL ANSWER: Agra"}) + "\n"
    )
    assert web_run.final_text("hermes", path) == "FINAL ANSWER: Agra"


def test_localmelo_reply_is_the_result_of_its_json_line(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text(
        json.dumps({"task_id": "t", "status": "completed", "result": "FINAL ANSWER: 1889", "n_llm": 3}) + "\n"
    )
    assert web_run.final_text("localmelo", path) == "FINAL ANSWER: 1889"
    path.write_text("")
    assert web_run.final_text("localmelo", path) == ""
    with pytest.raises(ValueError, match="unknown harness"):
        web_run.final_text("some_agent", path)


# --- one run: Chrome, the harness, the verdict ---


def run_one(rig: Rig, scenario: Scenario, **item_kw: Any) -> dict[str, Any]:
    rig.script = lambda run_id: scenario
    return web_run.execute(rig.item(**item_kw), rig.context())


def test_a_correct_answer_with_a_page_visit_passes(rig: Rig) -> None:
    record = run_one(rig, Scenario())
    assert record["verdict"] == "pass" and record["passed"] is True
    assert record["answer"] == "Marais Viljoen" and record["answer_status"] == "answer"
    assert (record["n_nav"], record["n_search"], record["n_pages"]) == (2, 1, 1)
    assert record["urls"] == [SERP, WIKI]  # the start page was before t0
    assert record["gold_url_visited"] is True and record["zero_nav"] is False and record["serp_only"] is False
    assert record["captcha"] is False and record["answer_key_suspect"] is False
    assert record["state"] == "finished" and record["tier"] == "easy" and record["kind"] == "string"
    assert record == rig.record(record["run_id"])  # record.json is what execute returns


def test_cost_fields_come_from_the_proxy_and_the_tap(rig: Rig) -> None:
    calls = (
        {"tool": "take_snapshot", "chars_out": 900, "n_images": 0},
        {"tool": "click", "chars_out": 120, "n_images": 1},
    )
    record = run_one(rig, Scenario(n_llm=4, usage=(1000, 600, 50), tap_calls=calls))
    assert record["n_llm"] == 4 and record["n_llm_aux"] == 0 and record["n_tool"] == 3
    assert record["prompt_tokens"] == 4000 and record["uncached_tokens"] == 1600 and record["output_tokens"] == 200
    assert record["wall_s"] == 5.0 and record["wall_cap_s"] == 1800.0
    assert record["tap"]["n_calls"] == 2 and record["tap"]["n_images"] == 1 and record["tap"]["max_chars_out"] == 900
    assert record["tap"]["by_tool"] == {"take_snapshot": 1, "click": 1}
    assert record["foreign_tools"] == [] and record["search_tool_leak"] is False
    assert len(record["prompt_sha256"]) == 64 and record["mcp_output_truncated"] is None


@pytest.mark.parametrize(
    ("scenario", "verdict"),
    [
        (Scenario(reply="FINAL ANSWER: Pieter Botha"), "fail"),
        (Scenario(reply="FINAL ANSWER: unknown"), "abstain"),
        (Scenario(reply="I could not find it."), "no_answer"),
        (Scenario(reply="FINAL ANSWER: Marais Viljoen", timed_out=True), "timeout"),
        (Scenario(n_llm=0), "harness_error"),
        (Scenario(raises=RuntimeError("proxy did not start")), "harness_error"),
    ],
)
def test_verdicts(rig: Rig, scenario: Scenario, verdict: str) -> None:
    record = run_one(rig, scenario)
    assert record["verdict"] == verdict
    assert record["passed"] is (verdict == "pass")
    if scenario.raises:
        assert record["error"] == "RuntimeError: proxy did not start"
        assert (rig.out / "runs" / record["run_id"] / "claude_code").is_dir()


def test_a_wrong_answer_names_nothing_matched(rig: Rig) -> None:
    record = run_one(rig, Scenario(reply="FINAL ANSWER: Pieter Botha"))
    assert record["answer"] == "Pieter Botha" and record["matched"] is None


@pytest.mark.parametrize("where", ["history", "tab"])
def test_a_captcha_page_makes_the_run_env_blocked(rig: Rig, where: str) -> None:
    if where == "history":
        scenario = Scenario(visits=[(1.0, SERP), (2.0, SORRY)])
    else:  # not a top-level visit of the window, but the tab it ended on
        scenario = Scenario(tabs=[{"id": "1", "title": "https://www.google.com/sorry/index", "url": SORRY}])
    record = run_one(rig, scenario)
    assert record["verdict"] == "env_blocked" and record["captcha"] is True
    assert record["answer"] == "Marais Viljoen"  # still recorded; the report leaves the run out


@pytest.mark.parametrize(
    ("urls", "blocked"),
    [
        (["https://www.google.com/sorry/index?continue=x"], True),
        (["https://www.google.co.uk/sorry/index"], True),
        (["https://www.google.com/search?q=sorry+about+that"], False),
        (["https://example.com/sorry/"], False),
        ([], False),
    ],
)
def test_captcha_detection_looks_for_googles_sorry_page(urls: list[str], blocked: bool) -> None:
    assert web_run.captcha_seen(urls, []) is blocked
    assert web_run.captcha_seen([], [{"url": u} for u in urls]) is blocked


def test_zero_navigation_and_serp_only_answers(rig: Rig) -> None:
    record = run_one(rig, Scenario(visits=[]))
    assert (record["n_nav"], record["zero_nav"], record["serp_only"], record["gold_url_visited"]) == (
        0,
        True,
        False,
        False,
    )
    record = run_one(rig, Scenario(visits=[(1.0, SERP)]), index=2)
    assert (record["n_nav"], record["zero_nav"], record["serp_only"]) == (1, False, True)
    record = run_one(rig, Scenario(visits=[], reply="I do not know."), index=3)
    assert record["zero_nav"] is False  # no answer was given


def test_the_answer_key_flag_looks_at_visited_urls_and_tab_titles(rig: Rig) -> None:
    key = "https://www.google.com/search?q=simpleqa+verified"
    assert run_one(rig, Scenario(visits=[(1.0, key)]))["answer_key_suspect"] is True
    titled = Scenario(
        tabs=[{"id": "1", "title": "google/simpleqa-verified - Hugging Face", "url": "https://example.org/"}]
    )
    assert run_one(rig, titled, index=2)["answer_key_suspect"] is True
    assert run_one(rig, Scenario(), index=3)["answer_key_suspect"] is False
    assert web_run.answer_key_suspect(["https://x.org/Simple_QA"], []) is True


def test_gold_hosts_are_matched_without_www(rig: Rig) -> None:
    row = PILOT.by_id(EASY)
    assert web_run.gold_url_visited(["https://www.gov.za/news/x"], row) is True
    assert web_run.gold_url_visited(["https://www.google.com/search?q=gov.za"], row) is False


def test_the_visits_of_the_window_only_count(rig: Rig) -> None:
    record = run_one(
        rig,
        Scenario(visits=[(-5.0, "https://before.example.org/"), (1.0, SERP), (500.0, "https://after.example.org/")]),
    )
    assert record["urls"] == [SERP]


def test_foreign_tools_and_a_search_tool_leak_are_flagged(rig: Rig) -> None:
    leaky = Scenario(tools=("mcp__webtap__take_snapshot", "web_search", "memory"))
    record = run_one(rig, leaky, harness="hermes")
    assert record["foreign_tools"] == ["memory", "web_search"] and record["search_tool_leak"] is True


def test_a_run_without_a_readable_history_has_null_navigation(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*a: Any, **k: Any) -> Any:
        raise web_run.sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(web_run.history, "summarize", broken)
    record = run_one(rig, Scenario())
    assert record["n_nav"] is None and record["zero_nav"] is None and "DatabaseError" in record["nav_error"]
    assert record["verdict"] == "pass"


# --- the arms in a run ---


def test_d_and_s_get_byte_identical_prompts_and_c_a_different_one(rig: Rig) -> None:
    for harness in ALL_HARNESSES:
        for arm in ("D", "S", "C"):
            run_one(rig, Scenario(), harness=harness, arm=arm, index=len(rig.calls) + 1)
    by = {(c.name, c.run_id.split("__")[-1]): c.prompt for c in rig.calls}
    texts = {by[(h, "D")] for h in ALL_HARNESSES} | {by[(h, "S")] for h in ALL_HARNESSES}
    assert len(texts) == 1  # one text for D and S on every harness
    assert texts.pop() == arms.prompt("D", PILOT.by_id(EASY).problem)
    assert {by[(h, "C")] for h in ALL_HARNESSES} == {arms.prompt("C", PILOT.by_id(EASY).problem)}
    assert arms.prompt("C", "x") != arms.prompt("D", "x")


def test_c_launches_no_chrome_and_has_no_navigation(rig: Rig) -> None:
    record = run_one(rig, Scenario(), arm="C")
    assert rig.chrome_calls == []
    assert not (rig.out / "runs" / record["run_id"] / "chrome").exists()
    assert (record["n_nav"], record["n_search"], record["n_pages"]) == (None, None, None)
    assert (record["gold_url_visited"], record["zero_nav"], record["serp_only"], record["answer_key_suspect"]) == (
        None,
    ) * 4
    assert record["captcha"] is False and record["wall_cap_s"] == 600.0 and record["tap"] is None
    assert rig.calls[0].kw["timeout_s"] == 600.0


def test_c_counts_its_tool_less_requests_as_llm_calls(rig: Rig) -> None:
    record = run_one(rig, Scenario(n_llm=1), arm="C")
    assert record["n_llm"] == 1 and record["n_llm_aux"] == 1 and record["verdict"] == "pass"


def test_d_and_s_launch_one_chrome_each_on_the_study_version_and_get_its_address(rig: Rig) -> None:
    run_one(rig, Scenario(), arm="D")
    run_one(rig, Scenario(), arm="S", index=2)
    assert [(d.parent.name, v) for d, v in rig.chrome_calls] == [
        ("001__q3503__r1__claude_code__D", CHROME_VERSION),
        ("002__q3503__r1__claude_code__S", CHROME_VERSION),
    ]
    assert all(d.name == "chrome" for d, _ in rig.chrome_calls)


def test_the_budget_and_the_cap_reach_the_harness(rig: Rig) -> None:
    run_one(rig, Scenario(), arm="D")
    run_one(rig, Scenario(), arm="S", index=2)
    for call in rig.calls:
        assert call.kw["timeout_s"] == 1800.0
        assert (call.kw["max_steps"], call.kw["max_turns"], call.kw["thinking"]) == (30, 40, "off")
        assert call.run_dir == rig.out / "runs" / call.run_id / "claude_code"


def test_the_builders_are_swapped_during_the_run_and_restored_after(rig: Rig) -> None:
    before = dict(rig.harness.BUILDERS)
    for harness in ("claude_code", "hermes"):
        run_one(rig, Scenario(), harness=harness, index=len(rig.calls) + 1)
    assert [c.swapped for c in rig.calls] == [True, True]
    assert rig.harness.BUILDERS == before


def test_localmelo_gets_its_flags_and_its_config_file(rig: Rig) -> None:
    run_one(rig, Scenario(), harness="localmelo", arm="S")
    run_one(rig, Scenario(), harness="localmelo", arm="D", index=2)
    run_one(rig, Scenario(), harness="claude_code", arm="D", index=3)
    s, d, cc = (c.kw["harness_args"] for c in rig.calls)
    assert s[-2:] == ("-X", "agent.tool_images=on") and "agent.tool_images=on" not in d and cc == ()
    assert d[0] == "--mcp" and d[2:] == ("--mcp-only", "-o", "json", "-X", "agent.max_calls=29")
    config = json.loads(Path(d[1]).read_text())
    web_args = config["mcpServers"]["webtap"]["args"]
    assert web_args[1:3] == ["--mode", "dom"]
    assert web_args[3:5] == ["--log", str(rig.out / "runs" / "002__q3503__r1__localmelo__D" / "tap.jsonl")]


def test_extra_localmelo_flags_reach_every_localmelo_arm_and_no_other_harness(rig: Rig) -> None:
    cells = [("localmelo", "D"), ("localmelo", "S"), ("localmelo", "C"), ("claude_code", "D"), ("hermes", "S")]
    rig.script = lambda run_id: Scenario()
    for index, (harness, arm) in enumerate(cells, 1):
        web_run.execute(rig.item(harness=harness, arm=arm, index=index), rig.context(localmelo_extra=EXTRA))
    d, s, c, claude, hermes = (call.kw["harness_args"] for call in rig.calls)
    assert d[-4:] == s[-4:] == c[-4:] == EXTRA
    assert s[-6:-4] == ("-X", "agent.tool_images=on")  # the arm's own flags stay in front
    assert d[0] == "--mcp" and d[2:5] == ("--mcp-only", "-o", "json")
    assert claude == hermes == ()


def test_a_chrome_that_cannot_run_stops_the_study_and_leaves_no_record(rig: Rig) -> None:
    rig.chrome_error = chrome.ChromeVersionChanged("Chrome is 157.0.1.2, but 156.0.8078.80 was recorded")
    with pytest.raises(web_run.StopStudy, match="157.0.1.2"):
        run_one(rig, Scenario())
    assert rig.calls == []


# --- the study: resume, stop, refusals ---


def test_a_study_runs_everything_once_and_a_resume_replays_nothing(rig: Rig) -> None:
    assert rig.go() == 0
    planned = web_run.schedule([PILOT.by_id(EASY), PILOT.by_id(MEDIUM)], 1, ("claude_code", "hermes"), ("D", "C"))
    assert rig.executed == [p["run_id"] for p in planned]  # the schedule's own order
    saved = rig.state()
    assert saved["complete"] is True and set(saved["runs"]) == {p["run_id"] for p in planned}
    assert saved["chrome_version"] == CHROME_VERSION and saved["config"]["protocol"] == "web-v1"
    assert all(r["state"] == "finished" and r["verdict"] == "pass" for r in saved["runs"].values())
    assert (rig.out / "schedule.json").exists()
    env = json.loads((rig.out / "environment.json").read_text())
    assert env["start"] == env["end"] == {"sw_vers": "fake", "chrome_version": CHROME_VERSION, "model": MODEL}
    assert env["base_url"] == "http://127.0.0.1:8500/v1"

    rig.calls.clear()
    assert rig.go() == 0
    assert rig.executed == []
    assert not (rig.out / "ws").exists()  # the lock and the empty working directory are gone


def test_the_cells_run_in_the_rotated_order_over_the_whole_study(rig: Rig) -> None:
    rows = [PILOT.by_id(EASY), PILOT.by_id(MEDIUM)]
    assert rig.go(rows=rows, arm_names=("D", "S", "C")) == 0
    planned = web_run.schedule(rows, 1, ("claude_code", "hermes"), ("D", "S", "C"))
    assert rig.executed == [p["run_id"] for p in planned]
    cells = [(c.name, c.run_id.rsplit("__", 1)[1]) for c in rig.calls]
    first, second = cells[:6], cells[6:]
    assert sorted(first) == sorted(second)  # the same six cells for each question
    assert first != second  # in another order


def test_an_interrupted_run_is_recorded_not_replayed(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = web_run.execute

    def dies_on_third(item: dict[str, Any], ctx: web_run.Context) -> dict[str, Any]:
        if item["index"] == 3:
            raise RuntimeError("killed")
        return real(item, ctx)

    monkeypatch.setattr(web_run, "execute", dies_on_third)
    with pytest.raises(RuntimeError, match="killed"):
        rig.go()
    third = web_run.schedule([PILOT.by_id(EASY), PILOT.by_id(MEDIUM)], 1, ("claude_code", "hermes"), ("D", "C"))[2][
        "run_id"
    ]
    assert rig.state()["runs"][third]["state"] == "started"
    assert not (rig.out / "ws" / ".lock").exists()

    monkeypatch.setattr(web_run, "execute", real)
    rig.calls.clear()
    assert rig.go() == 0
    assert len(rig.executed) == 5 and third not in rig.executed  # runs 4 to 8; the third is not replayed
    runs = rig.state()["runs"]
    assert (
        runs[third]["state"] == "interrupted"
        and runs[third]["verdict"] == "interrupted"
        and runs[third]["passed"] is False
    )
    assert rig.state()["complete"] is True


def test_a_started_run_with_a_record_is_adopted(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = web_run.execute

    def writes_then_dies(item: dict[str, Any], ctx: web_run.Context) -> dict[str, Any]:
        real(item, ctx)
        raise RuntimeError("died after record.json")

    monkeypatch.setattr(web_run, "execute", writes_then_dies)
    with pytest.raises(RuntimeError):
        rig.go()
    first = web_run.schedule([PILOT.by_id(EASY), PILOT.by_id(MEDIUM)], 1, ("claude_code", "hermes"), ("D", "C"))[0][
        "run_id"
    ]
    monkeypatch.setattr(web_run, "execute", real)
    rig.calls.clear()
    rig.go()
    assert first not in rig.executed
    assert rig.state()["runs"][first]["state"] == "finished"


def test_a_changed_config_needs_a_fresh_directory(rig: Rig, tmp_path: Path) -> None:
    rig.go()
    with pytest.raises(ValueError, match="changed"):
        rig.go(repeats=2)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "file").write_text("x")
    rig.out = stray
    with pytest.raises(ValueError, match="not empty"):
        rig.go()


def test_stop_file_stops_before_the_next_run(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = web_run.execute

    def stops_after_first(item: dict[str, Any], ctx: web_run.Context) -> dict[str, Any]:
        record = real(item, ctx)
        (ctx.out / "STOP").write_text("")
        return record

    monkeypatch.setattr(web_run, "execute", stops_after_first)
    assert rig.go() == 3
    assert len(rig.executed) == 1 and rig.state()["complete"] is False
    assert "STOP file found" in (rig.out / "web.log").read_text()
    (rig.out / "STOP").unlink()
    monkeypatch.setattr(web_run, "execute", real)
    rig.calls.clear()
    assert rig.go() == 0
    assert len(rig.executed) == 7  # the other seven runs


def test_a_chrome_version_change_between_sessions_refuses_the_resume(rig: Rig) -> None:
    rig.go(rows=[PILOT.by_id(EASY)])
    rig.chrome_version = "157.0.8100.1"
    rig.calls.clear()
    assert rig.go(rows=[PILOT.by_id(EASY)]) == 2
    assert rig.executed == []


def test_an_expected_chrome_version_is_enforced(rig: Rig) -> None:
    assert rig.go(expected_chrome_version="155.0.1.1") == 2
    assert rig.executed == []


def test_a_chrome_version_change_during_the_study_stops_it_and_the_run_is_retried_on_resume(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [PILOT.by_id(EASY), PILOT.by_id(MEDIUM)]
    original = rig.bench_chrome
    starts = {"n": 0}

    @contextmanager
    def updates_before_the_second(chrome_dir: Path, **kw: Any) -> Iterator[SimpleNamespace]:
        starts["n"] += 1
        if starts["n"] == 2:  # Chrome updated itself between two runs
            raise chrome.ChromeVersionChanged("Chrome is 157.0.1.2, but 156.0.8078.80 was recorded")
        with original(chrome_dir, **kw) as run:
            yield run

    monkeypatch.setattr(web_run, "bench_chrome", updates_before_the_second)
    assert rig.go(rows=rows, harnesses=("claude_code",), arm_names=("D",)) == 3
    assert len(rig.state()["runs"]) == 1  # the second run left nothing behind
    assert "157.0.1.2" in (rig.out / "web.log").read_text()
    monkeypatch.setattr(web_run, "bench_chrome", original)
    rig.calls.clear()
    assert rig.go(rows=rows, harnesses=("claude_code",), arm_names=("D",)) == 0
    assert rig.executed == [study_ids()[1]]


def test_a_study_of_c_alone_never_asks_for_chrome(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_chrome() -> str:
        raise AssertionError("Chrome's version was read")

    monkeypatch.setattr(web_run, "installed_chrome_version", no_chrome)
    assert rig.go(arm_names=("C",)) == 0
    assert rig.chrome_calls == [] and rig.state()["chrome_version"] is None


def test_a_refused_preflight_starts_nothing(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(web_run, "preflight", lambda *a, **k: ["the model server is not reachable"])
    assert rig.go() == 2
    assert "not reachable" in capsys.readouterr().err
    assert rig.executed == [] and not rig.out.exists()


# --- the CAPTCHA ---


def blocked_first(*blocked: str) -> Any:
    """A script in which the named run ids hit the CAPTCHA page and every other run passes."""

    def script(run_id: str) -> Scenario:
        return Scenario(visits=[(1.0, SERP), (2.0, SORRY)]) if run_id in blocked else Scenario()

    return script


def study_ids() -> list[str]:
    return [
        p["run_id"] for p in web_run.schedule([PILOT.by_id(EASY), PILOT.by_id(MEDIUM)], 1, ("claude_code",), ("D",))
    ]


def test_a_blocked_run_is_paused_then_run_once_more(rig: Rig) -> None:
    first, second = study_ids()
    rig.script = blocked_first(first)
    assert rig.go(harnesses=("claude_code",), arm_names=("D",), captcha_pause_s=90.0) == 0
    assert rig.executed == [first, first + "__retry", second]  # the retry follows at once, before the next question
    assert sum(rig.sleeps) == pytest.approx(90.0, abs=1.0) and max(rig.sleeps) <= web_run.PAUSE_CHUNK_S
    runs = rig.state()["runs"]
    assert runs[first]["verdict"] == "env_blocked" and runs[first + "__retry"]["verdict"] == "pass"
    assert runs[first + "__retry"]["attempt"] == 2 and runs[first + "__retry"]["retry_of"] == first
    assert rig.state()["complete"] is True
    assert (rig.out / "runs" / first / "chrome").is_dir() and (
        rig.out / "runs" / (first + "__retry") / "chrome"
    ).is_dir()


def test_the_pause_is_configurable_and_zero_waits_for_nothing(rig: Rig) -> None:
    rig.script = blocked_first(study_ids()[0])
    assert rig.go(harnesses=("claude_code",), arm_names=("D",), captcha_pause_s=0.0) == 0
    assert rig.sleeps == [] and len(rig.executed) == 3


def test_two_blocked_runs_in_a_row_stop_the_study(rig: Rig) -> None:
    first, second = study_ids()
    rig.script = blocked_first(first, first + "__retry")
    assert rig.go(harnesses=("claude_code",), arm_names=("D",)) == 3
    assert rig.executed == [first, first + "__retry"]  # no third attempt, and the next question is not started
    assert rig.state()["complete"] is False and second not in rig.state()["runs"]
    assert "two runs in a row" in (rig.out / "web.log").read_text()


def test_blocked_runs_with_a_good_run_between_do_not_stop_the_study(rig: Rig) -> None:
    first, second = study_ids()
    rig.script = blocked_first(first, second)
    assert rig.go(harnesses=("claude_code",), arm_names=("D",)) == 0
    assert rig.executed == [first, first + "__retry", second, second + "__retry"]
    assert [rig.state()["runs"][r]["verdict"] for r in rig.executed] == ["env_blocked", "pass", "env_blocked", "pass"]


def test_a_second_blocked_attempt_is_final_and_two_in_a_row_wins_over_a_third_try(rig: Rig) -> None:
    first, _ = study_ids()
    rig.script = blocked_first(first, first + "__retry")
    rig.go(harnesses=("claude_code",), arm_names=("D",))
    rig.calls.clear()
    # resumed by the operator: the blocked pair stays as it is, and the next question runs
    assert rig.go(harnesses=("claude_code",), arm_names=("D",)) == 0
    assert len(rig.executed) == 1 and rig.executed[0].startswith("002__")


def test_a_stop_file_during_the_pause_stops_and_the_resume_runs_the_retry(rig: Rig) -> None:
    first, second = study_ids()
    rig.script = blocked_first(first)

    def sleep_then_stop(seconds: float) -> None:
        rig.sleeps.append(seconds)
        (rig.out / "STOP").write_text("")

    code = web_run.run_study(
        rig.out, rows=[PILOT.by_id(EASY), PILOT.by_id(MEDIUM)], harnesses=("claude_code",), arm_names=("D",),
        localmelo=rig.tmp / "melo", harness=rig.harness, sleep=sleep_then_stop, captcha_pause_s=90.0,
        model=MODEL, claude_bin=rig.claude_bin,
    )  # fmt: skip
    assert code == 3 and rig.executed == [first]
    assert "STOP file found during the pause" in (rig.out / "web.log").read_text()
    (rig.out / "STOP").unlink()
    rig.calls.clear()
    rig.script = lambda run_id: Scenario()
    assert rig.go(harnesses=("claude_code",), arm_names=("D",), captcha_pause_s=0.0) == 0
    assert rig.executed == [first + "__retry", second]


def test_the_pause_counts_the_time_since_the_blocked_run_finished(rig: Rig) -> None:
    first, _ = study_ids()
    rig.script = blocked_first(first)

    def stop_in_pause(seconds: float) -> None:
        (rig.out / "STOP").write_text("")

    assert web_run.run_study(
        rig.out, rows=[PILOT.by_id(EASY), PILOT.by_id(MEDIUM)], harnesses=("claude_code",), arm_names=("D",),
        localmelo=rig.tmp / "melo", harness=rig.harness, sleep=stop_in_pause, captcha_pause_s=90.0,
        model=MODEL, claude_bin=rig.claude_bin,
    ) == 3  # fmt: skip
    (rig.out / "STOP").unlink()
    record = rig.state()["runs"][first]
    assert time.time() - record["finished_at"] < 90.0  # so a resume waits for the rest of the 90 s only
    rig.sleeps.clear()
    rig.script = lambda run_id: Scenario()
    rig.go(harnesses=("claude_code",), arm_names=("D",), captcha_pause_s=90.0)
    assert 0 < sum(rig.sleeps) <= 90.0


def test_pause_stops_at_the_stop_file(tmp_path: Path) -> None:
    slept: list[float] = []
    assert web_run._pause(tmp_path, 100.0, slept.append) is None
    assert slept == [30.0, 30.0, 30.0, 10.0]
    (tmp_path / "STOP").write_text("")
    assert web_run._pause(tmp_path, 100.0, slept.append) == "STOP file found"
    assert web_run._pause(tmp_path, 0.0, slept.append) is None


# --- bench settings, guards, the plan ---


def settings_state(harness: Any) -> tuple[Any, ...]:
    m = harness.manifest
    return (m.LOCALMELO_REPO, m.OMLX_BASE_URL, harness.UPSTREAM, m.BENCH_MODEL, harness.MODEL, m.CLAUDE_BIN)


def test_bench_settings_point_the_harness_at_the_clone_the_endpoint_the_model_and_the_binary(
    rig: Rig, tmp_path: Path
) -> None:
    manifest = rig.harness.manifest
    before = settings_state(rig.harness)
    with web_run.bench_settings(rig.harness, tmp_path / "melo", "http://127.0.0.1:9412/v1", MODEL, rig.claude_bin):
        assert manifest.LOCALMELO_REPO == tmp_path / "melo"
        assert manifest.OMLX_BASE_URL == "http://127.0.0.1:9412/v1"
        assert rig.harness.UPSTREAM == "http://127.0.0.1:9412"
        assert manifest.BENCH_MODEL == rig.harness.MODEL == MODEL  # the manifests and the builders agree
        assert manifest.CLAUDE_BIN == rig.claude_bin
    assert settings_state(rig.harness) == before
    with (
        pytest.raises(RuntimeError),
        web_run.bench_settings(rig.harness, tmp_path / "melo", "http://127.0.0.1:9412/v1/", MODEL, rig.claude_bin),
    ):
        assert rig.harness.UPSTREAM == "http://127.0.0.1:9412"
        raise RuntimeError("boom")
    assert settings_state(rig.harness) == before
    with web_run.bench_settings(rig.harness, tmp_path / "melo", None, MODEL, rig.claude_bin):  # the endpoint stays
        assert (manifest.OMLX_BASE_URL, rig.harness.UPSTREAM) == before[1:3]


def test_run_study_points_the_harness_at_the_endpoint_only_while_it_runs(rig: Rig) -> None:
    seen: list[str] = []
    original = rig.run

    def spy(*a: Any, **k: Any) -> dict[str, Any]:
        seen.append(rig.harness.UPSTREAM)
        return original(*a, **k)

    rig.harness.run = spy
    assert (
        rig.go(base_url="http://127.0.0.1:9412/v1", rows=[PILOT.by_id(EASY)], harnesses=("hermes",), arm_names=("C",))
        == 0
    )
    assert seen == ["http://127.0.0.1:9412"] and rig.harness.UPSTREAM == "http://127.0.0.1:8500"
    assert json.loads((rig.out / "environment.json").read_text())["base_url"] == "http://127.0.0.1:9412/v1"


@pytest.mark.parametrize(
    ("text", "low"),
    [
        ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t29%; discharging; 4:00 remaining", True),
        ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t30%; discharging; 4:00 remaining", False),
        ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t10%; charging; 1:00 remaining", False),
        ("", False),
    ],
)
def test_battery_problem(text: str, low: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_run, "_out", lambda cmd, timeout=30.0: text)
    assert (web_run.battery_problem() is not None) is low


def test_stop_reason_checks_the_stop_file_the_lock_and_the_battery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_run, "battery_problem", lambda: None)
    lock: list[str | None] = [None]
    monkeypatch.setattr(web_run.runner, "lock_problem", lambda: lock[0])
    assert web_run.stop_reason(tmp_path) is None
    lock[0] = "the lock is held by another session"
    assert web_run.stop_reason(tmp_path) == lock[0]
    assert web_run.stop_reason(tmp_path, gpu_lock=False) is None  # a stub model does not use the shared GPU
    (tmp_path / "STOP").write_text("")
    assert web_run.stop_reason(tmp_path) == "STOP file found"
    (tmp_path / "STOP").unlink()
    lock[0] = None
    monkeypatch.setattr(web_run, "battery_problem", lambda: "on battery at 12% (below 30%)")
    assert "on battery" in web_run.stop_reason(tmp_path)


def test_decide_orders_its_reasons() -> None:
    summary = {"timed_out": False}
    base = {"blocked": False, "error": None, "summary": summary, "n_llm_total": 3, "graded": "pass"}
    assert web_run.decide(**base) == "pass"
    assert web_run.decide(**{**base, "graded": "abstain"}) == "abstain"
    assert web_run.decide(**{**base, "n_llm_total": 0}) == "harness_error"
    assert web_run.decide(**{**base, "summary": {"timed_out": True}}) == "timeout"
    assert web_run.decide(**{**base, "error": "RuntimeError: x"}) == "harness_error"
    assert web_run.decide(**{**base, "summary": None}) == "harness_error"
    assert web_run.decide(**{**base, "blocked": True, "error": "x", "summary": {"timed_out": True}}) == "env_blocked"


def dry_run_fakes(monkeypatch: pytest.MonkeyPatch, problems: list[str]) -> None:
    monkeypatch.setattr(web_run, "preflight", lambda *a, **k: problems)
    monkeypatch.setattr(web_run, "installed_chrome_version", lambda: CHROME_VERSION)
    monkeypatch.setattr(web_run, "battery_problem", lambda: None)
    monkeypatch.setattr(
        web_run,
        "system_snapshot",
        lambda localmelo, chrome_version, **kw: {
            "sw_vers": "macOS",
            "xcodebuild_version": "Xcode",
            "chrome_version": chrome_version,
            "claude_code_version": "2.1.312 (Claude Code)" if kw.get("claude_bin") else None,
            "chrome_devtools_mcp": {"pinned": "1.11.0"},
            "localmelo_sha": "a9fb42b" + "0" * 33,
        },
    )


def test_the_dry_run_prints_the_stage_zero_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path)
    before = settings_state(rig.harness)
    dry_run_fakes(monkeypatch, ["Chrome not found"])
    code = web_run.main(
        [
            "--run-dir", str(tmp_path / "stage0"), "--dry-run", "--questions", "3503,1934",
            "--harnesses", "claude_code,hermes,localmelo", "--arms", "D,S,C", "--base-url", "http://127.0.0.1:9412/v1",
            "--no-gpu-lock",
        ],
        harness=rig.harness,
    )  # fmt: skip
    shown = capsys.readouterr().out
    assert code == 2  # the preflight problem
    assert "Web plan: 18 runs = 2 questions x 1 repeats x 3 harnesses x 3 arms" in shown
    assert f"Model endpoint: http://127.0.0.1:9412/v1; model {arms.DEFAULT_MODEL};" in shown
    assert f"Claude Code: {arms.CLAUDE_BIN}" in shown and "Claude Code 2.1.312 (Claude Code) |" in shown
    assert "Arm C: tap --mode none; no Chrome" in shown and "Arm D: tap --mode dom; headless Chrome" in shown
    assert "S-cua" not in shown  # not among the default arms
    assert "localmelo S: --mcp <mcp.json> --mcp-only -o json -X agent.max_calls=29 -X agent.tool_images=on" in shown
    assert shown.count("q3503") == 9 and shown.count("q1934") == 9
    assert "Chrome not found" in shown and f"Chrome {CHROME_VERSION}" in shown
    assert not (tmp_path / "stage0").exists()  # nothing is created by a dry run
    assert settings_state(rig.harness) == before  # and the harness is as it was


def test_the_s_cua_dry_run_reports_the_space_on_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path)
    dry_run_fakes(monkeypatch, [])
    monkeypatch.setattr(arms, "resolve_cua_bin", lambda: "cua-driver")
    monkeypatch.setattr(arms, "cua_current_space", lambda binary: 99)
    argv = ["--run-dir", str(tmp_path / "x"), "--dry-run", "--questions", "3503", "--harnesses", "localmelo"]
    assert web_run.main([*argv, "--arms", "S-cua", "--cua-space", "400"], harness=rig.harness) == 0
    shown = capsys.readouterr().out
    assert "Arm S-cua: tap --mode cua; headed Chrome on a profile clone, on the Space on screen" in shown
    assert "S-cua: Space on screen 99, wanted 400; a run waits up to 1800 s for it" in shown
    assert "localmelo S-cua: --mcp <mcp.json> --mcp-only -o json -X agent.max_calls=29 -X agent.tool_images=on" in shown
    assert "Claude Code:" not in shown  # localmelo only
    assert web_run.main([*argv, "--arms", "S-cua", "--cua-space", "99"], harness=rig.harness) == 0
    assert "wanted 99\n" in capsys.readouterr().out


def test_the_command_line_reaches_the_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(web_run, "run_study", lambda out, **kw: seen.update(out=out, **kw) or 0)
    code = web_run.main(
        [
            "--run-dir", str(tmp_path / "x"), "--questions", "easy", "--harnesses", "cc,melo", "--arms", "D,C",
            "--repeats", "2", "--base-url", "http://127.0.0.1:9412/v1", "--expected-chrome-version", CHROME_VERSION,
            "--captcha-pause-s", "5", "--no-gpu-lock", "--model", MODEL, "--claude-bin", str(rig.claude_bin),
            "--cua-space", "400",
        ],
        harness=rig.harness,
    )  # fmt: skip
    assert code == 0
    assert [r.id for r in seen["rows"]] == [r.id for r in PILOT.tier("easy")]
    assert (seen["harnesses"], seen["arm_names"], seen["repeats"]) == (["claude_code", "localmelo"], ["D", "C"], 2)
    assert (seen["base_url"], seen["expected_chrome_version"], seen["captcha_pause_s"], seen["gpu_lock"]) == (
        "http://127.0.0.1:9412/v1", CHROME_VERSION, 5.0, False,
    )  # fmt: skip
    assert (seen["model"], seen["claude_bin"], seen["cua_space"]) == (MODEL, rig.claude_bin, 400)

    defaults = web_run.parse_args(["--run-dir", "x"])
    assert (defaults.repeats, defaults.captcha_pause_s, defaults.no_gpu_lock, defaults.base_url) == (
        1,
        1800.0,
        False,
        None,
    )
    assert (defaults.model, defaults.claude_bin, defaults.cua_space) == (arms.DEFAULT_MODEL, arms.CLAUDE_BIN, None)
    assert (
        defaults.harnesses == "claude_code,hermes,localmelo"
        and defaults.arms == "D,S,C"
        and defaults.questions == "all"
    )
    seen.clear()
    web_run.main(["--run-dir", str(tmp_path / "y")], harness=rig.harness)
    assert seen["arm_names"] == ["D", "S", "C"] and seen["model"] == arms.DEFAULT_MODEL and seen["cua_space"] is None


def test_the_localmelo_flags_and_sha_reach_the_study_and_the_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(web_run, "run_study", lambda out, **kw: seen.update(out=out, **kw) or 0)
    flags = "-X agent.reflection_gate=patient -X agent.answer_on_stop=on"
    argv = ["--run-dir", str(tmp_path / "x"), "--questions", "easy", "--harnesses", "localmelo", "--arms", "D,S"]
    melo = tmp_path / "melo"
    assert web_run.main(argv, harness=rig.harness) == 0
    assert seen["expect_sha"] == arms.LOCALMELO_SHA and seen["localmelo_extra"] == ()  # the defaults
    given = ["--localmelo", str(melo), "--localmelo-sha", OTHER_SHA[:7], "--localmelo-args", flags]
    assert web_run.main([*argv, *given], harness=rig.harness) == 0
    assert (seen["expect_sha"], seen["localmelo_extra"], seen["localmelo"]) == (OTHER_SHA[:7], EXTRA, melo)

    checked: dict[str, Any] = {}
    dry_run_fakes(monkeypatch, [])
    monkeypatch.setattr(web_run, "preflight", lambda *a, **k: checked.update(k) or [])
    assert web_run.main([*argv, "--dry-run", *given], harness=rig.harness) == 0
    shown = capsys.readouterr().out
    assert checked["expect_sha"] == OTHER_SHA[:7]
    assert f"localmelo D: --mcp <mcp.json> --mcp-only -o json -X agent.max_calls=29 {flags}\n" in shown
    assert f"localmelo S: --mcp <mcp.json> --mcp-only -o json -X agent.max_calls=29 -X agent.tool_images=on {flags}\n" in shown
    assert web_run.parse_args(["--run-dir", "x", "--localmelo-args=-v"]).localmelo_args == "-v"


@pytest.mark.parametrize(
    "extra",
    [
        ["--localmelo-args=--mcp=x.json"], ["--localmelo-args=-X agent.a=b -o text"], ["--localmelo-args=-ojson"],
        ["--localmelo-args=-m Other-Model"], ["--localmelo-args=-u http://127.0.0.1:1/v1"],
        ["--localmelo-args=-X a=b -- the task"], ["--localmelo-args=-X 'agent.note=open"],
        ["--localmelo-sha", OTHER_SHA[:7]],  # the flags and the sha need localmelo among the harnesses
    ],
)  # fmt: skip
def test_bad_localmelo_flags_exit_with_a_message(extra: list[str], tmp_path: Path) -> None:
    harnesses = ["--harnesses", "claude_code,hermes" if extra[0] == "--localmelo-sha" else "localmelo"]
    for mode in ([], ["--dry-run"]):
        with pytest.raises(SystemExit) as stop:
            web_run.main(["--run-dir", str(tmp_path), *mode, *harnesses, *extra])
        assert isinstance(stop.value.code, str)  # our message, not argparse's usage error (exit code 2)


def test_localmelo_options_without_localmelo_are_a_mistake_but_the_defaults_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(web_run, "run_study", lambda out, **kw: 0)
    argv = ["--run-dir", str(tmp_path / "x"), "--harnesses", "hermes", "--arms", "C"]
    assert web_run.main(argv, harness=rig.harness) == 0
    explicit = [*argv, "--localmelo-args", "", "--localmelo-sha", arms.LOCALMELO_SHA]
    assert web_run.main(explicit, harness=rig.harness) == 0


def test_bad_names_on_the_command_line_exit_with_a_message(tmp_path: Path) -> None:
    for argv in (["--questions", "9999"], ["--harnesses", "some_agent"], ["--arms", "Z"], ["--repeats", "0"]):
        with pytest.raises(SystemExit) as stop:
            web_run.main(["--run-dir", str(tmp_path), "--dry-run", *argv])
        assert stop.value.code != 0


def test_the_environment_records_the_versions_of_the_pieces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_run, "_out", lambda cmd, timeout=30.0: " ".join(cmd))
    monkeypatch.setattr(arms, "localmelo_sha", lambda localmelo: "a9fb42b" + "0" * 33)
    from bench import manifest

    monkeypatch.setattr(manifest, "_model_server", lambda: {"name": "stub"})
    snapshot = web_run.system_snapshot(
        tmp_path, CHROME_VERSION, model=MODEL, claude_bin=Path("claude-bin"), cua_bin="cua-bin"
    )
    assert snapshot["sw_vers"] == "sw_vers" and snapshot["xcodebuild_version"] == "xcodebuild -version"
    assert (
        snapshot["chrome_version"] == CHROME_VERSION and snapshot["node"] == "v24.21.0" and snapshot["model"] == MODEL
    )
    assert snapshot["chrome_devtools_mcp"]["pinned"] == "1.11.0"
    assert snapshot["claude_code_version"] == "claude-bin --version"  # the pinned binary's own --version
    assert snapshot["cua_driver_version"] == "cua-bin --version"
    assert snapshot["localmelo_sha"].startswith("a9fb42b") and snapshot["model_server"] == {"name": "stub"}
    bare = web_run.system_snapshot(None, None)
    assert bare["localmelo_sha"] is None and bare["claude_code_version"] is None and bare["cua_driver_version"] is None


def test_the_study_records_the_claude_code_version_in_the_environment(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench import manifest

    monkeypatch.undo()  # the real snapshot, with only its outside calls faked
    monkeypatch.setattr(web_run, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(web_run, "stop_reason", lambda out, *, gpu_lock=True: None)
    monkeypatch.setattr(web_run, "bench_chrome", rig.bench_chrome)
    monkeypatch.setattr(web_run, "installed_chrome_version", lambda: CHROME_VERSION)
    monkeypatch.setattr(manifest, "_model_server", dict)
    monkeypatch.setattr(
        web_run, "_out", lambda cmd, timeout=30.0: "2.1.312 (Claude Code)" if cmd[-1] == "--version" else ""
    )
    assert rig.go(rows=[PILOT.by_id(EASY)], harnesses=("claude_code",), arm_names=("C",)) == 0
    env = json.loads((rig.out / "environment.json").read_text())
    assert env["start"]["claude_code_version"] == env["end"]["claude_code_version"] == "2.1.312 (Claude Code)"
    assert env["start"]["model"] == MODEL


def test_preflight_collects_what_is_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(
        web_run.runner, "_binaries", lambda harness: {"claude_code": tmp_path / "gone", "hermes": tmp_path}
    )
    monkeypatch.setattr(web_run.runner, "lock_problem", lambda: "the lock is held by another session")
    monkeypatch.setattr(web_run, "model_problem", lambda harness: "the model server is not reachable")
    monkeypatch.setattr(arms, "mcp_problems", lambda names: ["node is missing"])
    monkeypatch.setattr(
        arms, "localmelo_problems", lambda path, sha=arms.LOCALMELO_SHA: ["localmelo has uncommitted changes"]
    )
    monkeypatch.setattr(web_run.chrome, "preflight", lambda: ["a Chrome is still using a bench profile"])
    monkeypatch.setattr(web_run.chrome, "is_bench_profile", lambda path: False)
    (rig.harness.CLAUDE_HOME / "settings.json").write_text('{"includeCoAuthoredBy": false}')  # no "model" to replace

    problems = web_run.preflight(
        rig.harness,
        ("claude_code", "hermes", "localmelo"),
        ("D", "C"),
        tmp_path,
        model=MODEL,
        claude_bin=tmp_path / "gone",
    )
    for expected in (
        "gone is missing or not executable",
        'settings.json must have exactly one "model" entry',
        "lock is held",
        "not reachable",
        "node is missing",
        "uncommitted changes",
        "still using a bench profile",
        "not a profile template",
    ):
        assert any(expected in p for p in problems), expected
    assert not any("hermes:" in p for p in problems)
    assert not any(p.startswith("claude_code:") for p in problems)  # arms.claude_problems speaks for the binary

    c_only = web_run.preflight(rig.harness, ("hermes",), ("C",), tmp_path, gpu_lock=False)
    assert c_only == ["the model server is not reachable", "node is missing"]


def test_preflight_refuses_a_localmelo_checkout_at_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(web_run.runner, "_binaries", lambda harness: {})
    monkeypatch.setattr(web_run, "model_problem", lambda harness: None)
    monkeypatch.setattr(arms, "mcp_problems", lambda names: [])
    answers = {("rev-parse", "HEAD"): OTHER_SHA, ("status", "--porcelain"): ""}
    monkeypatch.setattr(arms, "_git", lambda repo, *args: answers.get(args, ""))

    def check(**kw: Any) -> list[str]:
        return web_run.preflight(rig.harness, ("localmelo",), ("C",), tmp_path, gpu_lock=False, **kw)

    problems = check()  # the pinned commit is the default
    assert len(problems) == 1 and "5fa22ee" in problems[0] and f"not {arms.LOCALMELO_SHA}" in problems[0]
    assert check(expect_sha=OTHER_SHA) == [] and check(expect_sha=OTHER_SHA[:7]) == []
    assert "not a9fb42b" in check(expect_sha="a9fb42b")[0]
    # other harnesses do not look at the localmelo checkout
    assert web_run.preflight(rig.harness, ("hermes",), ("C",), tmp_path, gpu_lock=False) == []


def test_preflight_refuses_a_claude_env_that_lost_a_safety_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(web_run.runner, "_binaries", lambda harness: {})
    monkeypatch.setattr(web_run, "model_problem", lambda harness: None)
    ok = web_run.preflight(
        rig.harness, ("claude_code",), ("C",), tmp_path, gpu_lock=False, model=MODEL, claude_bin=rig.claude_bin
    )
    assert ok == []  # the synthetic template, sourced as the launcher does, has the binary and the model right
    rig.claude_env.write_text(CLAUDE_ENV_TEXT.replace("export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1\n", ""))
    bad = web_run.preflight(
        rig.harness, ("claude_code",), ("C",), tmp_path, gpu_lock=False, model=MODEL, claude_bin=rig.claude_bin
    )
    assert len(bad) == 1 and "CLAUDE_CODE_DISABLE_AUTO_MEMORY must be exported as '1'" in bad[0]
    # a harness that is not Claude Code does not read the file
    assert web_run.preflight(rig.harness, ("hermes",), ("C",), tmp_path, gpu_lock=False, model=MODEL) == []


def test_the_model_endpoint_is_checked_at_v1_models(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = SimpleNamespace(UPSTREAM="http://127.0.0.1:1", UPSTREAM_TOKEN_CMD_ENV="WEB_TEST_NO_SUCH_TOKEN_CMD")
    problem = web_run.model_problem(harness)
    assert problem is not None and "not reachable at http://127.0.0.1:1/v1/models" in problem


# --- the model and the Claude Code binary, on all three harnesses ---


@needs_bench
def test_the_model_reaches_all_three_harnesses_and_the_claude_code_binary_is_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real builders of bench.harness, on synthetic homes: nothing is started."""
    import yaml
    from bench import harness

    rig = Rig(tmp_path)
    hermes_home = tmp_path / "hermes-template"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(HERMES_TEMPLATE)
    monkeypatch.setattr(harness, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(harness, "CLAUDE_ENV", rig.claude_env)
    monkeypatch.setattr(harness, "CLAUDE_HOME", rig.harness.CLAUDE_HOME)
    run_dirs = {name: tmp_path / f"run-{name}" for name in ("claude_code", "hermes", "localmelo")}
    for run_dir in run_dirs.values():
        run_dir.mkdir()
    before = (harness.MODEL, harness.manifest.BENCH_MODEL, harness.manifest.CLAUDE_BIN, harness.CLAUDE_ENV)

    with web_run.bench_settings(harness, tmp_path / "melo", None, MODEL, rig.claude_bin):
        # Hermes: the config copy names the model (and not the template's)
        _, env = harness.BUILDERS["hermes"]("task", run_dirs["hermes"], "http://127.0.0.1:43210", 40)
        config = yaml.safe_load((Path(env["HERMES_HOME"]) / "config.yaml").read_text())
        assert config["model"]["default"] == MODEL and "some-model" not in json.dumps(config)
        # localmelo: -m
        cmd, _ = harness.BUILDERS["localmelo"]("task", run_dirs["localmelo"], "http://127.0.0.1:43210", 40)
        assert cmd[cmd.index("--chat-model") + 1] == MODEL
        assert cmd[0] == str(tmp_path / "melo" / ".venv" / "bin" / "melo")  # the pinned clone
        # Claude Code: the per-run env file the launcher sources, and the copied home
        with arms.overrides(
            harness, "claude_code", "D", run_dir=run_dirs["claude_code"], browser_url=URL, model=MODEL,
            claude_bin=rig.claude_bin,
        ):  # fmt: skip
            cmd, env = harness.BUILDERS["claude_code"]("task", run_dirs["claude_code"], "http://127.0.0.1:43210", 40)
            assert str(harness.CLAUDE_ENV) in cmd and harness.CLAUDE_ENV != rig.claude_env
            sourced = arms.shell_env(harness.CLAUDE_ENV)
            assert [sourced[v] for v in arms.CLAUDE_MODEL_VARS] == [MODEL] * 5
            assert sourced["__SHELL__CLAUDE_BIN"] == str(rig.claude_bin)  # not the template's deleted version
            assert json.loads((Path(env["PAB_CONFIG_DIR"]) / "settings.json").read_text())["model"] == MODEL
        assert harness.CLAUDE_ENV == rig.claude_env
        assert harness.manifest.BENCH_MODEL == MODEL  # what each run's manifest records
    assert (harness.MODEL, harness.manifest.BENCH_MODEL, harness.manifest.CLAUDE_BIN, harness.CLAUDE_ENV) == before


def test_each_claude_code_run_has_its_own_env_copy_and_the_study_model_is_set_while_it_runs(rig: Rig) -> None:
    assert rig.go(harnesses=("claude_code", "hermes"), arm_names=("C",), rows=[PILOT.by_id(EASY)]) == 0
    by_name = {c.name: c for c in rig.calls}
    assert by_name["claude_code"].claude_env == by_name["claude_code"].run_dir.parent / "claude-code.env"
    assert by_name["hermes"].claude_env == rig.claude_env  # only Claude Code's run swaps it
    assert {c.model for c in rig.calls} == {MODEL}
    assert rig.harness.MODEL == "Old-Model" and rig.harness.CLAUDE_ENV == rig.claude_env  # restored
    assert rig.harness.manifest.CLAUDE_BIN == Path("original-claude")
    assert rig.state()["config"]["model"] == MODEL


def test_a_different_model_is_a_different_study(rig: Rig) -> None:
    rig.go()
    with pytest.raises(ValueError, match="changed"):
        rig.go(model="Another-Model")


def localmelo_study(rig: Rig, **kw: Any) -> int:
    return rig.go(harnesses=("localmelo",), arm_names=("D", "C"), **kw)


def test_the_extra_flags_and_the_expected_sha_are_part_of_the_study(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(web_run, "preflight", lambda *a, **k: seen.append(k) or [])
    assert localmelo_study(rig, localmelo_extra=EXTRA, expect_sha=OTHER_SHA[:7]) == 0
    assert seen[0]["expect_sha"] == OTHER_SHA[:7]  # preflight checks the checkout against it
    config = rig.state()["config"]
    assert config["localmelo_sha"] == OTHER_SHA[:7]
    assert config["localmelo_flags"]["D"][-4:] == list(EXTRA) and config["localmelo_flags"]["C"][-4:] == list(EXTRA)
    assert all(call.kw["harness_args"][-4:] == EXTRA for call in rig.calls) and len(rig.calls) == 4

    rig.calls.clear()
    assert localmelo_study(rig, localmelo_extra=EXTRA, expect_sha=OTHER_SHA[:7]) == 0  # the same study resumes
    assert rig.executed == []
    for other in ({"localmelo_extra": ()}, {"localmelo_extra": EXTRA[:2]}, {"expect_sha": arms.LOCALMELO_SHA}):
        with pytest.raises(ValueError, match="changed"):
            localmelo_study(rig, **{"localmelo_extra": EXTRA, "expect_sha": OTHER_SHA[:7], **other})


def test_the_identity_changes_with_the_extra_flags_and_the_sha() -> None:
    rows = [PILOT.by_id(EASY)]
    base = web_run.identity(PILOT, rows, 1, ALL_HARNESSES, ("D", "S"), MODEL)
    assert base["localmelo_sha"] == arms.LOCALMELO_SHA
    assert base["localmelo_flags"] == arms.harness_arg_templates(("D", "S"))  # unchanged for a study without extras
    flagged = web_run.identity(PILOT, rows, 1, ALL_HARNESSES, ("D", "S"), MODEL, EXTRA)
    assert flagged != base and flagged["localmelo_flags"]["S"][-4:] == list(EXTRA)
    assert {k for k in flagged if flagged[k] != base[k]} == {"localmelo_flags"}
    assert flagged != web_run.identity(PILOT, rows, 1, ALL_HARNESSES, ("D", "S"), MODEL, EXTRA[:2])
    moved = web_run.identity(PILOT, rows, 1, ALL_HARNESSES, ("D", "S"), MODEL, localmelo_sha=OTHER_SHA)
    assert moved != base and {k for k in moved if moved[k] != base[k]} == {"localmelo_sha"}
    assert moved["localmelo_sha"] == OTHER_SHA
    no_melo = web_run.identity(PILOT, rows, 1, ("hermes",), ("D",), MODEL, EXTRA, OTHER_SHA)
    assert no_melo["localmelo_flags"] is None and no_melo["localmelo_sha"] is None


# --- S-cua ---

WINDOW_GONE = {
    "tool": "click",
    "status": "refused",
    "head": "Refused: the browser's desktop is not on screen; the run cannot see the browser",
}
NO_WINDOW = {
    "tool": "screenshot",
    "status": "refused",
    "head": "Refused: the browser has no window on the desktop on screen",
}
OTHER_REFUSAL = {
    "tool": "click",
    "status": "refused",
    "head": "Refused: take a screenshot first; click coordinates refer to the latest screenshot",
}


def cua_context(rig: Rig, **kw: Any) -> web_run.Context:
    return rig.context(cua_bin="cua-driver", cua_space=400, **{k: v for k, v in kw.items()})


def run_cua(rig: Rig, scenario: Scenario, **item_kw: Any) -> dict[str, Any]:
    rig.script = lambda run_id: scenario
    item_kw.setdefault("arm", "S-cua")
    return web_run.execute(rig.item(**item_kw), cua_context(rig))


def test_an_s_cua_run_gets_a_headed_chrome_and_hands_the_tap_its_pid(rig: Rig) -> None:
    record = run_cua(rig, Scenario(), harness="localmelo")
    assert rig.headless == [False]  # headed, on the Space on screen
    tap = rig.taps[-1]
    assert tap[1:3] == ["--mode", "cua"]
    assert tap[tap.index("--chrome-pid") + 1] == "5001"  # Run.chrome.proc.pid of the fake
    assert tap[tap.index("--cua-bin") + 1] == "cua-driver" and "--browser-url" not in tap
    assert record["verdict"] == "pass" and record["blocked_by"] is None and record["cua_space"] == 400
    assert record["space_lost"] == 0 and record["wall_cap_s"] == 1800.0 and record["n_nav"] == 2
    assert rig.calls[0].kw["harness_args"][-2:] == ("-X", "agent.tool_images=on")  # tool images, as for S


def test_s_cua_has_the_prompt_of_d_and_s_and_the_other_arms_stay_headless(rig: Rig) -> None:
    for arm in ("D", "S", "S-cua", "C"):
        run_cua(rig, Scenario(), arm=arm, index=len(rig.calls) + 1)
    assert len({c.prompt for c in rig.calls[:3]}) == 1 and rig.calls[3].prompt != rig.calls[0].prompt
    assert rig.headless == [True, True, False]  # D, S, S-cua; C has no Chrome
    assert "--chrome-pid" not in rig.taps[0] and "--chrome-pid" in rig.taps[2]
    records = [rig.record(c.run_id) for c in rig.calls]
    assert [r["cua_space"] for r in records] == [None, None, 400, None]
    assert [r["space_lost"] for r in records] == [None, None, 0, None]


@pytest.mark.parametrize("refusal", [WINDOW_GONE, NO_WINDOW])
def test_an_s_cua_run_that_lost_its_space_is_env_blocked_not_failed(rig: Rig, refusal: dict[str, Any]) -> None:
    record = run_cua(rig, Scenario(reply="FINAL ANSWER: wrong", tap_calls=(refusal, refusal)))
    assert record["verdict"] == "env_blocked" and record["blocked_by"] == "space"
    assert record["captcha"] is False and record["space_lost"] == 2 and record["passed"] is False


def test_other_refusals_and_other_arms_are_not_a_lost_space(rig: Rig) -> None:
    ordinary = run_cua(rig, Scenario(reply="FINAL ANSWER: wrong", tap_calls=(OTHER_REFUSAL,)))
    assert ordinary["verdict"] == "fail" and ordinary["space_lost"] == 0
    # the same words in an S run (CDP: no Space) mean nothing
    screen = run_cua(rig, Scenario(tap_calls=(WINDOW_GONE,)), arm="S", index=2)
    assert screen["verdict"] == "pass" and screen["blocked_by"] is None and screen["space_lost"] is None
    only_ok = run_cua(rig, Scenario(tap_calls=({"tool": "click", "status": "ok", "head": "not on screen"},)), index=3)
    assert only_ok["space_lost"] == 0


def test_a_captcha_in_an_s_cua_run_is_still_a_captcha(rig: Rig) -> None:
    record = run_cua(rig, Scenario(visits=[(1.0, SERP), (2.0, SORRY)], tap_calls=(WINDOW_GONE,)))
    assert record["blocked_by"] == "captcha" and record["captcha"] is True and record["verdict"] == "env_blocked"


# the Space on screen, read before each S-cua run


class Spaces:
    """The Space cua-driver reports on each call: a list consumed in order, then its last value for ever."""

    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self, cua_bin: str) -> int:
        self.calls += 1
        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def spaces(monkeypatch: pytest.MonkeyPatch) -> Spaces:
    holder = Spaces(400)
    monkeypatch.setattr(arms, "resolve_cua_bin", lambda: "cua-driver")
    monkeypatch.setattr(arms, "cua_current_space", lambda binary: holder(binary))
    return holder


def test_the_right_space_needs_no_wait(rig: Rig, spaces: Spaces) -> None:
    assert web_run.wait_for_space(cua_context(rig)) is None
    assert rig.sleeps == [] and spaces.calls == 1


def test_a_wrong_space_is_rechecked_every_minute(rig: Rig, spaces: Spaces) -> None:
    spaces.values = [1, 1, 400]
    assert web_run.wait_for_space(cua_context(rig)) is None
    assert rig.sleeps == [60.0, 60.0] and spaces.calls == 3


def test_a_wrong_space_for_thirty_minutes_stops_with_the_reason(rig: Rig, spaces: Spaces) -> None:
    spaces.values = [1]
    reason = web_run.wait_for_space(cua_context(rig))
    assert reason == "the Space on screen is 1, not 400, after 1800 s"
    assert rig.sleeps == [60.0] * 30 and spaces.calls == 31  # a check every minute, and one at the end


def test_an_unreadable_space_counts_as_a_wrong_one(rig: Rig, spaces: Spaces) -> None:
    spaces.values = [arms.CuaError("list_windows failed: 'no permission'"), 400]
    assert web_run.wait_for_space(cua_context(rig)) is None and rig.sleeps == [60.0]
    spaces.values = [arms.CuaError("list_windows failed")]
    assert "not 400" in web_run.wait_for_space(cua_context(rig)) and "list_windows failed" in web_run.wait_for_space(
        cua_context(rig)
    )


def test_the_stop_file_ends_the_space_wait(rig: Rig, spaces: Spaces) -> None:
    spaces.values = [1]
    rig.out.mkdir()
    (rig.out / "STOP").write_text("")
    assert web_run.wait_for_space(cua_context(rig)) == "STOP file found" and rig.sleeps == []


def test_a_study_waits_for_the_space_before_each_s_cua_run_and_only_those(rig: Rig, spaces: Spaces) -> None:
    spaces.values = [1, 400]  # wrong at the first look, then right (and the later runs find it right)
    assert rig.go(harnesses=("localmelo",), arm_names=("D", "S-cua"), cua_space=400, rows=[PILOT.by_id(EASY)]) == 0
    assert len(rig.executed) == 2 and spaces.calls == 2  # the D run asks nothing; the S-cua run looked twice
    assert rig.sleeps == [60.0]
    assert {rig.state()["runs"][r]["arm"] for r in rig.executed} == {"D", "S-cua"}


def test_a_study_that_never_finds_its_space_stops_and_records_nothing(rig: Rig, spaces: Spaces) -> None:
    spaces.values = [1]
    code = rig.go(harnesses=("localmelo",), arm_names=("S-cua",), cua_space=400, rows=[PILOT.by_id(EASY)])
    assert code == 3 and rig.executed == [] and rig.state()["runs"] == {}
    assert sum(rig.sleeps) == 1800.0
    assert "the Space on screen is 1, not 400" in (rig.out / "web.log").read_text()
    spaces.values = [400]  # the operator fixes the Space and resumes
    rig.sleeps.clear()
    assert rig.go(harnesses=("localmelo",), arm_names=("S-cua",), cua_space=400, rows=[PILOT.by_id(EASY)]) == 0
    assert len(rig.executed) == 1 and rig.sleeps == []


def test_a_lost_space_run_is_run_again_without_the_captcha_pause(rig: Rig, spaces: Spaces) -> None:
    first = "001__q3503__r1__localmelo__S-cua"
    rig.script = lambda run_id: Scenario(tap_calls=(WINDOW_GONE,)) if run_id == first else Scenario()
    assert (
        rig.go(
            harnesses=("localmelo",),
            arm_names=("S-cua",),
            cua_space=400,
            captcha_pause_s=90.0,
            rows=[PILOT.by_id(EASY)],
        )
        == 0
    )
    assert rig.executed == [first, first + "__retry"] and rig.sleeps == []  # no 90 s pause
    runs = rig.state()["runs"]
    assert runs[first]["blocked_by"] == "space" and runs[first + "__retry"]["verdict"] == "pass"


def test_two_lost_spaces_in_a_row_stop_the_study(rig: Rig, spaces: Spaces) -> None:
    first = "001__q3503__r1__localmelo__S-cua"
    rig.script = lambda run_id: Scenario(tap_calls=(WINDOW_GONE,))
    code = rig.go(
        harnesses=("localmelo",), arm_names=("S-cua",), cua_space=400, rows=[PILOT.by_id(EASY), PILOT.by_id(MEDIUM)]
    )
    assert code == 3 and rig.executed == [first, first + "__retry"]


def test_s_cua_needs_its_space_number(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="--cua-space"):
        rig.go(arm_names=("D", "S-cua"))
    argv = ["--run-dir", str(rig.out), "--questions", "3503", "--arms", "S-cua", "--harnesses", "localmelo"]
    assert web_run.main(argv, harness=rig.harness) == 2
    assert "--cua-space" in capsys.readouterr().err
    assert not rig.out.exists()


def test_preflight_wants_the_cua_space_and_a_running_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(web_run.runner, "_binaries", lambda harness: {})
    monkeypatch.setattr(web_run, "model_problem", lambda harness: None)
    monkeypatch.setattr(arms, "mcp_problems", lambda names: [])
    monkeypatch.setattr(arms, "localmelo_problems", lambda path, sha=arms.LOCALMELO_SHA: [])
    monkeypatch.setattr(web_run.chrome, "preflight", list)
    monkeypatch.setattr(web_run.chrome, "is_bench_profile", lambda path: False)
    monkeypatch.setattr(arms, "resolve_cua_bin", lambda: "cua-driver")
    monkeypatch.setattr(arms, "cua_status_problem", lambda binary: None)

    def check(space: int | None) -> list[str]:
        return web_run.preflight(rig.harness, ("localmelo",), ("S-cua",), tmp_path, gpu_lock=False, cua_space=space)

    assert check(400) == [f"{web_run.chrome.TEMPLATE_DIR} is not a profile template; run web/chrome.py init"]
    assert "S-cua needs --cua-space" in check(None)[0]
    monkeypatch.setattr(arms, "cua_status_problem", lambda binary: "the cua-driver daemon is not running ('')")
    assert any("daemon is not running" in p for p in check(400))

    def nobody() -> str:
        raise ValueError("no app with bundle id com.trycua.driver; pass --cua-bin")

    monkeypatch.setattr(arms, "resolve_cua_bin", nobody)
    assert any("no app with bundle id" in p for p in check(400))
    # without S-cua among the arms neither is looked at
    assert not any(
        "cua" in p.lower() for p in web_run.preflight(rig.harness, ("localmelo",), ("C",), tmp_path, gpu_lock=False)
    )


def test_s_cua_is_not_a_default_arm() -> None:
    rows = [PILOT.by_id(EASY)]
    assert {p["arm"] for p in web_run.schedule(rows)} == {"D", "S", "C"}
    planned = web_run.schedule(rows, 1, ("localmelo",), ("S-cua",))
    assert [p["run_id"] for p in planned] == ["001__q3503__r1__localmelo__S-cua"]
    assert parse_run_id(planned[0]["run_id"])["arm"] == "S-cua"
    assert web_run.WALL_CAP_S["S-cua"] == 1800.0
