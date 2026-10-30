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


def usage_of(call: dict[str, Any]) -> Any:
    """The call's usage_raw: its "usage" as given, else prompt and cached tokens, plus completion_tokens when "completion" is set."""
    if "usage" in call:
        return call["usage"]
    usage: dict[str, Any] = {"prompt_tokens": call["prompt"], "prompt_tokens_details": {"cached_tokens": call["cached"]}}
    if "completion" in call:
        usage["completion_tokens"] = call["completion"]
    return usage


def write_run(run_dir: Path, calls: list[dict[str, Any]]) -> None:
    """calls: {body, prompt, cached, completion (optional), usage (optional usage_raw that replaces the built one),
    reply (bytes), sse (bool), status}"""
    (run_dir / "bodies").mkdir(parents=True)
    rows = []
    for i, call in enumerate(calls):
        row: dict[str, Any] = {
            "call_index": i, "endpoint": "chat_completions", "status": call.get("status", 200), "error": None,
            "stream_complete": True, "mono_start": 100.0 + i, "role": "main" if call["body"].get("tools") else "aux",
            "usage_raw": usage_of(call),
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


def plan(prompt: int, cached: int, output: int | None = None) -> dict[str, Any]:
    """A planning call; output is its completion_tokens, left out of usage_raw when None."""
    call = {"body": plan_body(), "prompt": prompt, "cached": cached, "reply": completion("ok")}
    return call if output is None else {**call, "completion": output}


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
    assert m["output_tokens"] == 0  # none of these calls has completion_tokens


def test_a_single_call_run_has_no_efficiency(tmp_path: Path) -> None:
    write_run(tmp_path / "run", [plan(1000, 0)])
    assert e3_report.run_metrics(tmp_path / "run", PROMPT)["cache_efficiency"] is None
    assert e3_report.run_metrics(tmp_path / "missing", PROMPT)["model_calls"] == 0
    assert e3_report.run_metrics(tmp_path / "missing", PROMPT)["output_tokens"] == 0


def test_output_tokens_sum_completion_tokens_over_the_calls_the_prompt_tokens_count(tmp_path: Path) -> None:
    calls = [
        plan(1000, 0, 50),
        plan(2000, 1800, 70),
        plan(3000, 2700),  # usage_raw without completion_tokens adds 0
        {**plan(500, 0), "usage": {"prompt_tokens": 500, "completion_tokens": None}},  # so does a null one
        {**plan(9999, 9999, 999), "status": 500},  # a failed call is not counted
        {**plan(0, 0), "usage": {"completion_tokens": 400}},  # no prompt_tokens: not one of the counted calls
        {**plan(0, 0), "usage": None},  # no usage at all
    ]
    write_run(tmp_path / "run", calls)
    m = e3_report.run_metrics(tmp_path / "run", PROMPT)
    assert m["output_tokens"] == 50 + 70
    # The existing figures are those of the same four calls
    assert (m["model_calls"], m["prompt_tokens"], m["uncached_tokens"]) == (4, 6500, 6500 - 4500)


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
    attempts, n_aux, line (put attempts and n_aux only in summary.json), out (the run's output tokens: the
    completion_tokens of its first call; its other calls have none)."""
    out = tmp_path / "e3"
    runs: dict[str, Any] = {}
    for index, spec in enumerate(specs, 1):
        run_id = f"{index:03d}__{spec['exercise']}__r{spec['repeat']}__{spec['arm']}"
        rel = f"runs/{run_id}/localmelo/aider__{spec['exercise']}"
        run_dir = out / rel
        calls = [plan(1000, 0, spec.get("out")), plan(1000, int(1000 * spec["efficiency"]))]
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


# --- output tokens and the paired passes against S0 ---

# (arm, task) -> per repeat: (passed, output tokens, cache efficiency, wall s, reflections). A run's uncached tokens
# are 1500 (efficiency 0.5, no reflection), 2000 (0.0), 1250 (0.75), 2500 (0.5, one reflection) or 1750 (0.75, one);
# its steps are 2, plus 2 per reflection.
PAIRED_RUNS = {
    ("S0", "A"): [(True, 100, 0.5, 10, 0), (False, 900, 0.0, 90, 0)],
    ("S0", "B"): [(True, 5000, 0.5, 500, 0), (True, 5000, 0.5, 500, 0)],  # SH never passes B
    ("S0", "C"): [(True, 200, 0.5, 20, 1), (True, 400, 0.5, 40, 0)],
    ("S0", "D"): [(False, 7000, 0.5, 700, 0), (False, 7000, 0.5, 700, 0)],  # SH passes D
    ("SH", "A"): [(True, 50, 0.75, 8, 0), (True, 70, 0.75, 12, 0)],
    ("SH", "B"): [(False, 8000, 0.5, 800, 0), (False, 8000, 0.5, 800, 0)],
    ("SH", "C"): [(True, 100, 0.75, 20, 1), (False, 3020, 0.0, 300, 0)],
    ("SH", "D"): [(True, 9000, 0.5, 900, 0), (True, 9000, 0.5, 900, 0)],
    **{("L", task): [(False, 1000, 0.5, 100, 0)] * 2 for task in "ABCD"},  # L passes nothing
}
PAIRED_HEADING = "## Paired passes against S0"
PER_TASK_HEADING = "## Per task (mean over finished repeats): prompt tokens / uncached tokens / output tokens / steps / wall s"


def paired_study(tmp_path: Path, arms: tuple[str, ...] = ("S0", "SH", "L")) -> Path:
    specs = [
        spec(arm, task, repeat, passed, efficiency, out=output, wall_s=wall, reflections=["ok"] * reflections)
        for (arm, task), repeats in PAIRED_RUNS.items()
        if arm in arms
        for repeat, (passed, output, efficiency, wall, reflections) in enumerate(repeats, 1)
    ]
    return make_study(tmp_path, specs, arms, ("A", "B", "C", "D"))


def md_table(text: str, heading: str) -> dict[str, dict[str, str]]:
    """The first table under a heading, as {first cell: {column: cell}}."""
    rows: list[list[str]] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("#"):
            if in_section:
                break
            in_section = line.startswith(heading)
        elif in_section and line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
    header, _separator, *body = rows
    return {row[0]: dict(zip(header, row, strict=True)) for row in body}


def test_output_tokens_are_per_run_per_arm_and_per_task(tmp_path: Path, clone: Path) -> None:
    rep = e3_report.build_report(paired_study(tmp_path), clone)
    by_id = {r["run_id"]: r for r in rep["runs"]}
    assert by_id["001__A__r1__S0"]["output_tokens"] == 100 and by_id["002__A__r2__S0"]["output_tokens"] == 900
    assert by_id["001__A__r1__S0"]["model_calls"] == 2  # the existing keys are still there
    assert [rep["arms"][a]["output_tokens_per_run"] for a in ("S0", "SH", "L")] == [3200, 4655, 1000]
    assert rep["arms"]["S0"]["per_task"]["A"]["output_tokens"] == 500  # the mean of the finished runs, passing or not
    assert rep["arms"]["SH"]["per_task"]["C"]["output_tokens"] == 1560


def test_an_interrupted_run_is_left_out_of_the_output_token_means(tmp_path: Path, clone: Path) -> None:
    specs = [spec("S0", "A", 1, True, 0.5, out=100), spec("S0", "A", 2, True, 0.5, out=900, state="interrupted")]
    rep = e3_report.build_report(make_study(tmp_path, specs, ("S0",), ("A",)), clone)
    assert rep["arms"]["S0"]["output_tokens_per_run"] == 100
    assert rep["arms"]["S0"]["per_task"]["A"]["output_tokens"] == 100
    assert rep["arms"]["S0"]["per_task"]["A"]["passing"]["output_tokens"] == 100


def test_the_tables_show_output_tokens(tmp_path: Path, clone: Path) -> None:
    text = e3_report.render_markdown(e3_report.build_report(paired_study(tmp_path), clone))
    arm_rows = md_table(text, "## Cache efficiency and cost")
    header = list(arm_rows["S0"])
    assert header[header.index("Uncached tokens") + 1] == "Output tokens"
    assert [arm_rows[a]["Output tokens"] for a in ("S0", "SH", "L")] == ["3,200", "4,655", "1,000"]
    assert PER_TASK_HEADING in text
    per_task = md_table(text, PER_TASK_HEADING)
    assert per_task["A"]["S0"] == "2,000 / 1,750 / 500 / 2.0 / 50"  # prompt / uncached / output / steps / wall s
    assert per_task["C"]["SH"] == "3,000 / 1,875 / 1,560 / 3.0 / 160"


def test_paired_tasks_are_those_both_arms_passed_at_least_once(tmp_path: Path, clone: Path) -> None:
    paired = e3_report.build_report(paired_study(tmp_path), clone)["paired"]
    assert paired["baseline"] == "S0" and list(paired["arms"]) == ["SH", "L"]
    # B is passed by S0 only and D by SH only; A and C are passed by both, A in one repeat of S0
    assert paired["arms"]["SH"]["tasks"] == ["A", "C"] and paired["arms"]["SH"]["n_tasks"] == 2
    assert paired["arms"]["L"]["tasks"] == [] and paired["arms"]["L"]["n_tasks"] == 0


def test_paired_figures_are_means_over_paired_tasks_of_the_means_over_passing_runs(tmp_path: Path, clone: Path) -> None:
    rep = e3_report.build_report(paired_study(tmp_path), clone)
    # S0, A: its one passing run (the failing run's 900 output tokens are left out); S0, C: the mean of two passing runs
    s0_tasks = {t: rep["arms"]["S0"]["per_task"][t]["passing"] for t in "ABCD"}
    assert s0_tasks["A"] == pytest.approx({"output_tokens": 100, "uncached_tokens": 1500, "steps": 2, "wall_s": 10})
    assert s0_tasks["C"] == pytest.approx({"output_tokens": 300, "uncached_tokens": 2000, "steps": 3, "wall_s": 30})
    assert s0_tasks["D"] == dict.fromkeys(e3_report.PAIRED_METRICS)  # no passing run: no means
    # SH, C: its one passing run (the failing run's 3020 output tokens are left out)
    sh_c = rep["arms"]["SH"]["per_task"]["C"]["passing"]
    assert sh_c == pytest.approx({"output_tokens": 100, "uncached_tokens": 1750, "steps": 4, "wall_s": 20})
    sh = rep["paired"]["arms"]["SH"]
    # Over tasks A and C; pooling the passing runs instead would give S0 233 output tokens, not 200
    assert sh["s0"] == pytest.approx({"output_tokens": 200, "uncached_tokens": 1750, "steps": 2.5, "wall_s": 20})
    assert sh["arm"] == pytest.approx({"output_tokens": 80, "uncached_tokens": 1500, "steps": 3, "wall_s": 15})


def test_paired_ratios_are_the_arms_mean_over_s0s(tmp_path: Path, clone: Path) -> None:
    paired = e3_report.build_report(paired_study(tmp_path), clone)["paired"]["arms"]
    assert paired["SH"]["output_ratio"] == pytest.approx(80 / 200)
    assert paired["SH"]["uncached_ratio"] == pytest.approx(1500 / 1750)
    # An arm with no paired task has no means and no ratios
    assert paired["L"]["s0"] == dict.fromkeys(e3_report.PAIRED_METRICS)
    assert paired["L"]["arm"] == dict.fromkeys(e3_report.PAIRED_METRICS)
    assert paired["L"]["output_ratio"] is None and paired["L"]["uncached_ratio"] is None


def test_the_paired_section_follows_the_arm_table_and_states_its_definition_once(tmp_path: Path, clone: Path) -> None:
    text = e3_report.render_markdown(e3_report.build_report(paired_study(tmp_path), clone))
    assert text.index("## Cache efficiency and cost") < text.index(PAIRED_HEADING) < text.index("## Reflection health")
    rows = md_table(text, PAIRED_HEADING)
    assert list(rows) == ["SH", "L"]
    assert rows["SH"]["Paired tasks"] == "2"
    assert rows["SH"]["Output tokens (S0 / arm)"] == "200 / 80" and rows["SH"]["Output ratio"] == "0.40"
    assert rows["SH"]["Uncached tokens (S0 / arm)"] == "1,750 / 1,500" and rows["SH"]["Uncached ratio"] == "0.86"
    assert rows["SH"]["Steps (S0 / arm)"] == "2.5 / 3.0" and rows["SH"]["Wall s (S0 / arm)"] == "20 / 15"
    assert rows["L"]["Paired tasks"] == "0"
    assert rows["L"]["Output tokens (S0 / arm)"] == "n/a / n/a" and rows["L"]["Output ratio"] == "n/a"
    assert rows["L"]["Uncached ratio"] == "n/a"
    # Apart from its heading and table, the section is one sentence, directly under the table
    section = text.split(PAIRED_HEADING + "\n", 1)[1].split("\n## ", 1)[0]
    prose = [line for line in section.splitlines() if line and not line.startswith("|")]
    assert len(prose) == 1 and section.strip().splitlines()[-1] == prose[0]
    assert prose[0].startswith("Paired tasks are those that S0 and the arm both passed at least once")
    assert prose[0].endswith(".") and "." not in prose[0][:-1]


@pytest.mark.parametrize(
    ("arms", "shown"),
    [(("SH", "L"), False), (("S0",), False), (("S0", "SH"), True), (("L", "S0", "SH"), True)],
)
def test_the_paired_section_needs_s0_and_another_arm(tmp_path: Path, clone: Path, arms: tuple[str, ...], shown: bool) -> None:
    rep = e3_report.build_report(paired_study(tmp_path, arms), clone)
    text = e3_report.render_markdown(rep)
    assert (rep["paired"] is not None) is shown and (PAIRED_HEADING in text) is shown
    if shown:
        assert list(rep["paired"]["arms"]) == [a for a in arms if a != "S0"]
    # The rest of the report is there either way
    assert "## Cache efficiency and cost" in text and "## Reflection health" in text and "## Decision rule" in text


def test_main_writes_the_markdown_and_json_reports(tmp_path: Path, clone: Path, capsys: pytest.CaptureFixture[str]) -> None:
    same = ([True, True], [True, False])
    out = make_study(tmp_path, study_specs({"S0": (*same, 0.5, ["ok"]), "C1": (*same, 0.9, ["bad"])}), ("S0", "C1"), ("A", "B"))
    assert e3_report.main([str(out), "--localmelo", str(clone)]) == 0
    data = json.loads((out / "report.json").read_text())
    assert data["decision"]["winner"] in ("S0", "C1")
    assert set(data["paired"]["arms"]["C1"]) == {"tasks", "n_tasks", "s0", "arm", "output_ratio", "uncached_ratio"}
    assert "output_tokens" in data["runs"][0] and "output_tokens_per_run" in data["arms"]["S0"]
    text = (out / "report.md").read_text()
    for heading in ("## Cache efficiency and cost", PAIRED_HEADING, "## Reflection health", "## Passes per task", "## Decision rule", "## Discordant"):
        assert heading in text
    assert "| C1 | `none` |" in text
    assert e3_report.main([str(tmp_path / "nowhere")]) == 2
