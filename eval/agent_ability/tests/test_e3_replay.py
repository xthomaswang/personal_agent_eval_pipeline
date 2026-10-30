"""The reflection replay on a small fake study (e3.json and runs with request, forwarded and response bodies),
with an injected poster in place of the model server. No model, no oMLX."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from e3 import replay

PROMPT = "The current attempt has ended without a final answer. Reflect on what happened and decide.\n\n"
CLONE_SOURCE = (
    "REFLECTION_PROMPT = (\n"
    '    "The current attempt has ended without a final answer. "\n'
    '    "Reflect on what happened and decide.\\n\\n"\n'
    ")\n"
)
DECISION = '{"progress_made": false, "recommended_action": "continue", "summary": "s"}'
TOOLS = [{"type": "function", "function": {"name": "bash"}}]
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    source = tmp_path / "localmelo" / "melo" / "agent"
    source.mkdir(parents=True)
    (source / "_chat.py").write_text(CLONE_SOURCE)
    return tmp_path / "localmelo"


def completion(text: str | None, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}


def plan_body() -> dict[str, Any]:
    return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}], "tools": TOOLS}


def reflect_body(tag: str) -> dict[str, Any]:
    """A fixed-prefix reflection request: its last user message starts with the prompt."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
                {"role": "user", "content": f"{PROMPT}Original task: {tag}"}]  # fmt: skip
    return {"messages": messages, "tools": TOOLS}


def write_run(run_dir: Path, calls: list[tuple[dict[str, Any], str]]) -> None:
    """calls: (request body, reply text); the forwarded body is the request plus the bench's sampling."""
    (run_dir / "bodies").mkdir(parents=True)
    rows = []
    for i, (body, reply) in enumerate(calls):
        bodies = {}
        for kind, value in (("request", body), ("forwarded", {**body, **SAMPLING}), ("response", completion(reply))):
            (run_dir / "bodies" / f"{i:06d}-{kind}.json").write_text(json.dumps(value))
            bodies[kind] = f"bodies/{i:06d}-{kind}.json"
        rows.append(
            {"call_index": i, "endpoint": "chat_completions", "status": 200, "error": None, "stream_complete": True,
             "mono_start": 100.0 + i, "bodies": bodies}  # fmt: skip
        )
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)))


def make_study(root: Path) -> Path:
    """S0 run 1: a reflection read badly, its retry, a later good one. V64 run 2 (interrupted): one reflection.
    V3 run 3 (another arm) and S0 run 4 (still running) are not replayed."""
    study = root / "study"
    runs = {
        "001__a__r1__S0": ("S0", "finished", [(plan_body(), "ok"), (reflect_body("1"), "I would stop."),
                                              (reflect_body("1r"), DECISION), (plan_body(), "ok"),
                                              (reflect_body("2"), DECISION)]),
        "002__a__r1__V64": ("V64", "interrupted", [(plan_body(), "ok"), (reflect_body("3"), DECISION)]),
        "003__a__r1__V3": ("V3", "finished", [(reflect_body("4"), DECISION)]),
        "004__b__r1__S0": ("S0", "running", [(reflect_body("5"), DECISION)]),
    }  # fmt: skip
    records = {}
    for index, (run_id, (arm, state, calls)) in enumerate(runs.items(), 1):
        run_dir = f"runs/{run_id}/localmelo/aider__a"
        write_run(study / run_dir, calls)
        records[run_id] = {"index": index, "run_id": run_id, "arm": arm, "exercise": "a", "repeat": 1,
                           "state": state, "run_dir": run_dir}  # fmt: skip
    (study / "e3.json").write_text(json.dumps({"config": {}, "runs": records, "complete": False}))
    return study


def poster(replies: list[Any]) -> tuple[replay.Post, list[dict[str, Any]]]:
    """A post function answering with `replies` in order (an exception is raised), and the bodies it got."""
    sent: list[dict[str, Any]] = []

    def post(base_url: str, body: dict[str, Any]) -> dict[str, Any]:
        sent.append(body)
        reply = replies[len(sent) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    return post, sent


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (completion(DECISION), "readable"),
        (completion(f"```json\n{DECISION}\n```"), "readable"),
        (completion("", [{"function": {"name": "bash", "arguments": "{}"}}]), "tool_call"),
        (completion('{"summary": "s", "recommended_action": "continue"]}'), "invalid"),
        (completion("I would stop here."), "no_json"),
        (completion(None), "no_json"),
        ({}, "no_json"),
    ],
)
def test_classify(reply: dict[str, Any], expected: str) -> None:
    assert replay.classify(reply) == expected


def test_reflections_are_chain_starts_of_the_chosen_arms_with_forwarded_bodies(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    found = replay.reflections(study, ("S0", "V64"), PROMPT)
    assert [(f["run_id"], f["body"], f["recorded"]) for f in found] == [
        ("001__a__r1__S0", "bodies/000001-forwarded.json", "unreadable"),  # its retry (000002) is not replayed
        ("001__a__r1__S0", "bodies/000004-forwarded.json", "readable"),
        ("002__a__r1__V64", "bodies/000001-forwarded.json", "readable"),
    ]
    assert found[0]["forwarded"] == {**reflect_body("1"), **SAMPLING}
    assert all(f["study"] == "study" and f["exercise"] == "a" for f in found)


def test_replay_is_repeat_major_and_resumes(tmp_path: Path) -> None:
    items = replay.reflections(make_study(tmp_path), ("S0", "V64"), PROMPT)
    out = tmp_path / "out" / "replies.jsonl"
    out.parent.mkdir()
    post, sent = poster([completion(DECISION)] * 4 + [completion("", [{"function": {"name": "bash"}}])] * 2)
    assert replay.replay(items, 2, out, "http://server/v1", post) is None
    assert sent == [i["forwarded"] for i in items] * 2
    rows = replay.load(out)
    assert [(r["run_id"], r["repeat"]) for r in rows] == [(i["run_id"], rep) for rep in (1, 2) for i in items]
    assert [r["class"] for r in rows] == ["readable"] * 4 + ["tool_call"] * 2
    assert rows[-1]["excerpt"].startswith("<tool call instead of text, finish stop: bash(")
    assert rows[0]["completion_tokens"] == 9 and rows[0]["recorded"] == "unreadable"

    post, sent = poster([])
    assert replay.replay(items, 2, out, "http://server/v1", post) is None
    assert sent == []
    post, sent = poster([completion(DECISION)] * 3)
    assert replay.replay(items, 3, out, "http://server/v1", post) is None
    assert len(sent) == 3 and len(replay.load(out)) == 9


def test_a_reply_that_is_not_readable_is_kept_in_full(tmp_path: Path) -> None:
    items = replay.reflections(make_study(tmp_path), ("S0", "V64"), PROMPT)
    out = tmp_path / "out" / "replies.jsonl"
    out.parent.mkdir()
    # It names the decision but never closes its JSON object: invalid, and longer than the 200-character excerpt.
    truncated = "Here is my reflection: " + "the last attempt failed again. " * 8 + '{"recommended_action": "continue"'
    # The excerpt keeps only 80 characters of the arguments; the reply keeps all of them.
    arguments = json.dumps({"command": "ls " + "-l " * 80})
    tool_calls = [{"function": {"name": "bash", "arguments": arguments}}]
    post, _ = poster([completion(DECISION), completion(truncated), completion("", tool_calls)])
    assert replay.replay(items, 1, out, "http://server/v1", post) is None
    rows = replay.load(out)
    assert [r["class"] for r in rows] == ["readable", "invalid", "tool_call"]
    assert rows[0]["reply"] is None
    assert len(truncated) > 200 and rows[1]["excerpt"] == truncated[:200]
    assert rows[1]["reply"] == truncated
    assert f"[tool_call] bash({arguments})" in rows[2]["reply"]


def test_full_replies_adds_the_content_and_tool_calls_only_when_asked(tmp_path: Path) -> None:
    items = replay.reflections(make_study(tmp_path), ("S0", "V64"), PROMPT)[:2]
    arguments = json.dumps({"command": "ls " + "-l " * 80})
    tool_calls = [{"function": {"name": "bash", "arguments": arguments}}]
    for full in (False, True):
        out = tmp_path / ("full" if full else "plain") / "replies.jsonl"
        out.parent.mkdir()
        post, _ = poster([completion(DECISION), completion("", tool_calls)])
        assert replay.replay(items, 1, out, "http://server/v1", post, full_replies=full) is None
        readable, tool = replay.load(out)
        if full:
            assert readable["content"] == DECISION and readable["tool_calls"] == []
            assert tool["content"] == "" and tool["tool_calls"] == [{"name": "bash", "arguments": arguments}]
        else:
            assert all("content" not in row and "tool_calls" not in row for row in (readable, tool))


def test_stop_file_and_failed_requests_stop_it_and_failures_are_retried(tmp_path: Path) -> None:
    items = replay.reflections(make_study(tmp_path), ("S0", "V64"), PROMPT)
    out = tmp_path / "out" / "replies.jsonl"
    out.parent.mkdir()
    (out.parent / "STOP").touch()
    post, sent = poster([])
    assert replay.replay(items, 1, out, "http://server/v1", post) == "STOP file found"
    assert sent == []
    (out.parent / "STOP").unlink()

    refused = urllib.error.URLError("connection refused")
    post, sent = poster([completion(DECISION), refused, refused, refused])
    stopped = replay.replay(items, 2, out, "http://server/v1", post)
    assert stopped is not None and stopped.startswith("3 failed requests in a row")
    assert [r["class"] for r in replay.load(out)] == ["readable", "error", "error", "error"]
    assert [r["reply"] for r in replay.load(out)] == [None] * 4

    post, sent = poster([completion("I would stop.")] * 5)
    assert replay.replay(items, 2, out, "http://server/v1", post) is None
    assert len(sent) == 5   # the three failures again, then the two not tried yet
    rows = replay.latest(replay.load(out))
    assert len(rows) == 6 and not any(r["class"] == "error" for r in rows)


def test_summary_counts_the_latest_reply_of_each_reflection(tmp_path: Path) -> None:
    def row(arm: str, run_id: str, repeat: int, cls: str) -> dict[str, Any]:
        return {"study": "s", "run_id": run_id, "arm": arm, "body": "b", "repeat": repeat, "class": cls}

    rows = [
        row("S0", "1", 1, "readable"), row("S0", "1", 2, "error"), row("S0", "1", 2, "invalid"),
        row("S0", "2", 1, "tool_call"), row("V64", "3", 1, "no_json"), row("V64", "3", 2, "error"),
    ]  # fmt: skip
    table = replay.summary(rows, ("S0", "V64", "B1")).splitlines()
    assert table[2] == "| S0 | 2 | 3 | 1 | 1 | 1 | 0 | 0 | 66.7% |"
    assert table[3] == "| V64 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 100.0% |"
    assert table[4] == "| B1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |"


def test_main_replays_and_prints_the_summary(tmp_path: Path, clone: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study = make_study(tmp_path)
    out = tmp_path / "out" / "replies.jsonl"
    post, sent = poster([completion(DECISION)] * 3)
    argv = [str(study), "--out", str(out), "--arms", "S0,V64,B1", "--repeats", "1", "--localmelo", str(clone),
            "--base-url", "http://server/v1"]  # fmt: skip
    assert replay.main(argv, post=post) == 0
    assert len(sent) == 3
    printed = capsys.readouterr()
    assert "| S0 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0.0% |" in printed.out
    assert "| V64 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.0% |" in printed.out
    assert printed.err == ""

    argv[4] = "B1"
    assert replay.main(argv, post=poster([])[0]) == 0
    assert "no reflections of B1" in capsys.readouterr().err


def test_main_full_replies_flag_keeps_the_content(tmp_path: Path, clone: Path) -> None:
    study = make_study(tmp_path)
    out = tmp_path / "out" / "replies.jsonl"
    post, _ = poster([completion(DECISION)] * 3)
    argv = [str(study), "--out", str(out), "--arms", "S0,V64", "--repeats", "1", "--localmelo", str(clone),
            "--base-url", "http://server/v1", "--full-replies"]  # fmt: skip
    assert replay.main(argv, post=post) == 0
    assert [(r["content"], r["tool_calls"]) for r in replay.load(out)] == [(DECISION, [])] * 3
