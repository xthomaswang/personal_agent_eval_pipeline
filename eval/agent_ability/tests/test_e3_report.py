"""The E3 report on small fake run directories: calls.jsonl with usage_raw, bodies with and without tools,
and localmelo's JSON result line. No model, no oMLX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from e3 import report as e3_report

PROMPT = "The current attempt has ended without a final answer. Reflect on what happened and decide.\n\n"
CLONE_SOURCE = (
    "REFLECTION_PROMPT = (\n"
    '    "The current attempt has ended without a final answer. "\n'
    '    "Reflect on what happened and decide.\\n\\n"\n'
    ")\n"
)
DECISION = '{"progress_made": false, "recommended_action": "continue", "summary": "s"}'
TOOLS = [{"type": "function", "function": {"name": "bash"}}]


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    source = tmp_path / "localmelo" / "melo" / "agent"
    source.mkdir(parents=True)
    (source / "_chat.py").write_text(CLONE_SOURCE)
    return tmp_path / "localmelo"


def sse(text: str) -> bytes:
    chunks = [text[i : i + 7] for i in range(0, len(text), 7)]
    events = [{"choices": [{"delta": {"content": c}}]} for c in chunks]
    events.append({"choices": [], "usage": {"prompt_tokens": 1}})
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def completion(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()


def plan_body() -> dict[str, Any]:
    return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}], "tools": TOOLS}


def reflect_body(*, tools: bool = False, in_system: bool = True) -> dict[str, Any]:
    """tools=False: the default mode's request (the text in the system message); tools=True: the fixed-prefix
    request, whose last user message starts with the text."""
    if tools:
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
                    {"role": "user", "content": PROMPT + "Original task: t"}]  # fmt: skip
        return {"messages": messages, "tools": TOOLS}
    system = f"sys\n\n{PROMPT}" if in_system else "sys"
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "go"}]}


def write_run(run_dir: Path, calls: list[dict[str, Any]]) -> None:
    """calls: {body, prompt, cached, reply (bytes), sse (bool), status}"""
    (run_dir / "bodies").mkdir(parents=True)
    rows = []
    for i, call in enumerate(calls):
        row: dict[str, Any] = {
            "call_index": i, "endpoint": "chat_completions", "status": call.get("status", 200), "error": None,
            "stream_complete": True, "mono_start": 100.0 + i, "role": "main" if call["body"].get("tools") else "aux",
            "usage_raw": {"prompt_tokens": call["prompt"], "prompt_tokens_details": {"cached_tokens": call["cached"]}},
        }  # fmt: skip
        bodies = {}
        (run_dir / "bodies" / f"{i:06d}-request.json").write_text(json.dumps(call["body"]))
        bodies["request"] = f"bodies/{i:06d}-request.json"
        if "reply" in call:
            ext = "sse" if call.get("sse") else "json"
            (run_dir / "bodies" / f"{i:06d}-response.{ext}").write_bytes(call["reply"])
            bodies["response"] = f"bodies/{i:06d}-response.{ext}"
        row["bodies"] = bodies
        rows.append(row)
    # The proxy writes a call when it completes, so the file is not in start order
    rows.reverse()
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def plan(prompt: int, cached: int) -> dict[str, Any]:
    return {"body": plan_body(), "prompt": prompt, "cached": cached, "reply": completion("ok")}


# --- parsing ---


def test_reflection_prompt_is_read_from_the_clone_source(clone: Path) -> None:
    assert e3_report.reflection_prompt(clone) == PROMPT
    (clone / "melo" / "agent" / "_chat.py").write_text("OTHER = 1\n")
    with pytest.raises(ValueError):
        e3_report.reflection_prompt(clone)


@pytest.mark.parametrize(
    ("text", "readable"),
    [
        (DECISION, True),
        (f"```json\n{DECISION}\n```", True),
        (f"I think so. {DECISION} Done.", True),
        ('{"summary": "x", "next": {"recommended_action": "stop"}}', True),  # found by scanning, even nested
        ('{"summary": "no action"}', False),
        ("I would stop here.", False),
        ('{"recommended_action": "continue"', False),
        ("", False),
    ],
)
def test_has_decision(text: str, readable: bool) -> None:
    assert e3_report.has_decision(text) is readable


def test_reply_text_reads_json_and_sse() -> None:
    assert e3_report.reply_text(completion(DECISION), "x.json") == DECISION
    assert e3_report.reply_text(sse(DECISION), "x.sse") == DECISION
    assert e3_report.reply_text(b"not json", "x.json") == ""


def test_reflection_request_classification() -> None:
    assert e3_report.is_reflection_request(reflect_body(), PROMPT) is True
    assert e3_report.is_reflection_request(reflect_body(tools=True), PROMPT) is True
    assert e3_report.is_reflection_request(plan_body(), PROMPT) is False
    assert e3_report.is_reflection_request(reflect_body(in_system=False), PROMPT) is None
    list_content = {"messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT + "x"}]}], "tools": TOOLS}
    assert e3_report.is_reflection_request(list_content, PROMPT) is True


# --- one run ---


def test_cache_efficiency_leaves_out_the_first_call_and_failed_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    calls = [plan(1000, 0), plan(2000, 1800), plan(3000, 2700), {**plan(9999, 9999), "status": 500}]
    write_run(run_dir, calls)
    m = e3_report.run_metrics(run_dir, PROMPT)
    assert m["model_calls"] == 3
    assert m["prompt_tokens"] == 6000
    assert m["uncached_tokens"] == 6000 - 4500
    assert m["cache_efficiency"] == pytest.approx(4500 / 5000)
    assert (m["later_prompt_tokens"], m["later_cached_tokens"]) == (5000, 4500)


def test_a_single_call_run_has_no_efficiency(tmp_path: Path) -> None:
    write_run(tmp_path / "run", [plan(1000, 0)])
    assert e3_report.run_metrics(tmp_path / "run", PROMPT)["cache_efficiency"] is None
    assert e3_report.run_metrics(tmp_path / "missing", PROMPT)["model_calls"] == 0


def test_reflections_unreadable_replies_and_retries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    calls = [
        plan(1000, 0),
        {"body": reflect_body(), "prompt": 500, "cached": 0, "reply": completion(f"```json\n{DECISION}\n```")},  # readable
        plan(1100, 900),
        {"body": reflect_body(), "prompt": 600, "cached": 0, "reply": completion("Let me think about it.")},  # unreadable
        {"body": reflect_body(), "prompt": 600, "cached": 600, "reply": completion(DECISION)},  # its retry
        {"body": reflect_body(tools=True), "prompt": 1200, "cached": 1100, "reply": sse("no json here"), "sse": True},
        {"body": reflect_body(), "prompt": 700, "cached": 0, "reply": completion("nope")},  # its retry, also unreadable
        plan(1300, 1200),
        {"body": reflect_body(tools=True), "prompt": 1200, "cached": 1100, "reply": sse(DECISION), "sse": True},
        {"body": reflect_body(in_system=False), "prompt": 300, "cached": 0, "reply": completion("a tool-less planning call")},
    ]
    write_run(run_dir, calls)
    m = e3_report.run_metrics(run_dir, PROMPT)
    assert m["reflection_requests"] == 6
    assert m["reflections"] == 4  # readable; unreadable; unreadable (SSE, tools); readable (SSE, tools) after a planning call
    assert m["unreadable_reflections"] == 2
    assert m["toolless_other"] == 1


def test_a_reflection_without_a_stored_reply_is_unreadable(tmp_path: Path) -> None:
    write_run(tmp_path / "run", [plan(1000, 0), {"body": reflect_body(), "prompt": 500, "cached": 0}])
    m = e3_report.run_metrics(tmp_path / "run", PROMPT)
    assert (m["reflections"], m["unreadable_reflections"]) == (1, 1)


# --- a study ---


def make_study(tmp_path: Path, specs: list[dict[str, Any]], arms: tuple[str, ...], exercises: tuple[str, ...], repeats: int = 2) -> Path:
    """specs: arm, exercise, repeat, passed, efficiency (0..1) of its second call, reflections: list of 'ok'/'bad',
    attempts, n_aux, line (put attempts and n_aux only in summary.json)."""
    out = tmp_path / "e3"
    runs: dict[str, Any] = {}
    for index, spec in enumerate(specs, 1):
        run_id = f"{index:03d}__{spec['exercise']}__r{spec['repeat']}__{spec['arm']}"
        rel = f"runs/{run_id}/localmelo/aider__{spec['exercise']}"
        run_dir = out / rel
        calls = [plan(1000, 0), plan(1000, int(1000 * spec["efficiency"]))]
        for kind in spec.get("reflections", []):
            reply = DECISION if kind == "ok" else "unclear"
            cached = int(1000 * spec["efficiency"])
            calls.append({"body": reflect_body(), "prompt": 1000, "cached": cached, "reply": completion(reply)})
            calls.append(plan(1000, cached))  # a planning call between reflections: no retry chain
        write_run(run_dir, calls)
        line = {"task_id": "t", "status": "done", "attempts": spec.get("attempts", 1), "n_aux": spec.get("n_aux", 0)}
        (run_dir / "summary.json").write_text(json.dumps({"harness_report": {"run_record": line}}))
        record = {
            "run_id": run_id, "index": index, "arm": spec["arm"], "exercise": spec["exercise"], "repeat": spec["repeat"],
            "state": spec.get("state", "finished"), "verdict": "pass" if spec["passed"] else "fail",
            "passed": spec["passed"], "wall_s": spec.get("wall_s", 60.0), "run_dir": rel,
        }  # fmt: skip
        if not spec.get("line"):
            record.update(attempts=spec.get("attempts", 1), n_aux=spec.get("n_aux", 0))
        runs[run_id] = record
    config = {"protocol": "e3-v1", "arms": {a: ["--output-format", "json"] for a in arms}, "exercises": list(exercises),
              "repeats": repeats, "localmelo_sha": "ce4a3a8" + "0" * 33, "model": "m"}  # fmt: skip
    out.mkdir(exist_ok=True)
    (out / "e3.json").write_text(json.dumps({"config": config, "runs": runs, "complete": True}))
    return out


def spec(arm: str, exercise: str, repeat: int, passed: bool, efficiency: float, **kw: Any) -> dict[str, Any]:
    return {"arm": arm, "exercise": exercise, "repeat": repeat, "passed": passed, "efficiency": efficiency, **kw}


def study_specs(rows: dict[str, tuple[list[bool], list[bool], float, list[str]]]) -> list[dict[str, Any]]:
    """rows: arm -> (passes of task A, passes of task B, efficiency, reflections of each run)."""
    out = []
    for arm, (a, b, eff, reflections) in rows.items():
        for exercise, passes in (("A", a), ("B", b)):
            for repeat, passed in enumerate(passes, 1):
                out.append(spec(arm, exercise, repeat, passed, eff, reflections=reflections))
    return out


def build(tmp_path: Path, clone: Path, rows: dict[str, tuple[list[bool], list[bool], float, list[str]]]) -> dict[str, Any]:
    out = make_study(tmp_path, study_specs(rows), tuple(rows), ("A", "B"))
    return e3_report.build_report(out, clone)


def test_the_decision_picks_the_most_efficient_arm_that_loses_nothing(tmp_path: Path, clone: Path) -> None:
    rep = build(
        tmp_path,
        clone,
        {
            "S0": ([True, True], [True, False], 0.50, ["ok"]),
            "C1": ([True, False], [True, True], 0.95, ["ok"]),  # most efficient, but loses A (S0: 2/2, C1: 1/2)
            "B1": ([True, True], [False, False], 0.80, ["ok"]),  # loses nothing: B is 1/2 for S0
            "B2": ([True, True], [True, True], 0.70, ["ok"]),
        },
    )
    d = rep["decision"]
    assert d["arms"]["C1"]["losses"] == ["A"] and not d["arms"]["C1"]["eligible"]
    assert d["arms"]["B1"]["losses"] == [] and d["arms"]["B1"]["eligible"]
    assert d["winner"] == "B1"
    pairs = {(p["task"], p["arm"]): p for p in d["discordant"]}
    assert set(pairs) == {("A", "C1"), ("B", "C1"), ("B", "B1"), ("B", "B2")}
    assert pairs[("A", "C1")] == {"task": "A", "arm": "C1", "s0_passes": 2, "arm_passes": 1, "loss": True}
    assert pairs[("B", "C1")]["loss"] is False


def test_the_unreadable_rate_may_exceed_s0s_by_two_points_at_most(tmp_path: Path) -> None:
    # 25 reflections per run, 4 runs per arm: S0 has 2 unreadable (2%), B1 4 (4%, +2 points), C1 6 (6%, +4 points)
    runs = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    specs = []
    for arm, eff, bad in (("S0", 0.50, (1, 1, 0, 0)), ("B1", 0.90, (1, 1, 1, 1)), ("C1", 0.95, (2, 2, 1, 1))):
        for (exercise, repeat), n_bad in zip(runs, bad, strict=True):
            reflections = ["bad"] * n_bad + ["ok"] * (25 - n_bad)
            specs.append(spec(arm, exercise, repeat, True, eff, reflections=reflections))
    clone = tmp_path / "localmelo"
    (clone / "melo" / "agent").mkdir(parents=True)
    (clone / "melo" / "agent" / "_chat.py").write_text(CLONE_SOURCE)
    rep = e3_report.build_report(make_study(tmp_path, specs, ("S0", "B1", "C1"), ("A", "B")), clone)
    arms, d = rep["arms"], rep["decision"]["arms"]
    assert [arms[a]["reflections"] for a in ("S0", "B1", "C1")] == [100, 100, 100]
    assert [arms[a]["unreadable_reflections"] for a in ("S0", "B1", "C1")] == [2, 4, 6]
    assert arms["B1"]["unreadable_rate"] == pytest.approx(0.04)
    assert d["B1"]["unreadable_rate_ok"] is True
    assert d["C1"]["unreadable_rate_ok"] is False and d["C1"]["eligible"] is False
    assert rep["decision"]["winner"] == "B1"  # C1 is more efficient, but its unreadable rate is 4 points above S0's


def test_a_tie_goes_to_the_simpler_arm_and_s0_wins_when_nothing_beats_it(tmp_path: Path, clone: Path) -> None:
    same = ([True, True], [True, True])
    rep = build(tmp_path, clone, {"S0": (*same, 0.50, []), "B1": (*same, 0.90, []), "C1": (*same, 0.90, [])})
    assert rep["decision"]["winner"] == "C1"
    rep = build(tmp_path / "second", clone, {"S0": (*same, 0.90, []), "C1": (*same, 0.60, [])})
    assert rep["decision"]["winner"] == "S0"
    assert rep["decision"]["discordant"] == []


def test_per_arm_numbers(tmp_path: Path, clone: Path) -> None:
    specs = [
        spec("S0", "A", 1, True, 0.9, attempts=2, n_aux=3, wall_s=100.0),
        spec("S0", "A", 2, False, 0.5, attempts=1, n_aux=1, wall_s=200.0, line=True),  # from summary.json
        spec("S0", "B", 1, True, 0.7),
        spec("S0", "B", 2, True, 0.7, state="interrupted"),
    ]
    out = make_study(tmp_path, specs, ("S0",), ("A", "B"))
    rep = e3_report.build_report(out, clone)
    arm = rep["arms"]["S0"]
    assert (arm["runs"], arm["interrupted"], arm["n_pass"]) == (4, 1, 2)  # an interrupted run is not a pass
    assert arm["per_task"]["A"]["passes"] == [True, False] and arm["per_task"]["B"]["passes"] == [True, False]
    assert arm["cache_efficiency"] == pytest.approx((900 + 500 + 700 + 700) / 4000)
    assert arm["cache_efficiency_median"] == pytest.approx(0.7)
    assert arm["attempts"] == 2 + 1 + 1 + 1 and arm["n_aux"] == 3 + 1 + 0 + 0
    assert arm["per_task"]["A"]["wall_s"] == pytest.approx(150.0)
    assert arm["per_task"]["A"]["steps"] == 2 and arm["per_task"]["A"]["prompt_tokens"] == 2000
    assert arm["per_task"]["A"]["uncached_tokens"] == pytest.approx(((2000 - 900) + (2000 - 500)) / 2)
    assert rep["runs_recorded"] == 4 and rep["runs_planned"] == 4


def test_main_writes_the_markdown_and_json_reports(tmp_path: Path, clone: Path, capsys: pytest.CaptureFixture[str]) -> None:
    same = ([True, True], [True, False])
    out = make_study(tmp_path, study_specs({"S0": (*same, 0.5, ["ok"]), "C1": (*same, 0.9, ["bad"])}), ("S0", "C1"), ("A", "B"))
    assert e3_report.main([str(out), "--localmelo", str(clone)]) == 0
    data = json.loads((out / "report.json").read_text())
    assert data["decision"]["winner"] in ("S0", "C1")
    text = (out / "report.md").read_text()
    for heading in ("## Cache efficiency and cost", "## Reflection health", "## Passes per task", "## Decision rule", "## Discordant"):
        assert heading in text
    assert "| C1 | `none` |" in text
    assert e3_report.main([str(tmp_path / "nowhere")]) == 2
