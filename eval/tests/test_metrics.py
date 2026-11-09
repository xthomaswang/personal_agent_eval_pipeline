"""The fixed metrics block (bench/metrics.py): the same keys from every kind of run, None for what a run cannot give."""

from __future__ import annotations

from typing import Any

from bench import metrics
from bench.proxy.observe import tool_call_sha256


def call(
    index: int,
    prompt: int,
    cached: int,
    tools: list[tuple[str, str]] | None = None,
    role: str = "main",
    **more: Any,
) -> dict[str, Any]:
    tools = tools or []
    return {"call_index": index, "role": role, "status": 200, "error": None, "stream_complete": True,
            "ts_start": 100.0 + index * 10, "ts_end": 105.0 + index * 10,
            "usage": {"prompt_tokens": prompt, "completion_tokens": 20, "cached_tokens": cached},
            "usage_raw": {"completion_tokens_details": {"reasoning_tokens": 7}}, "output_reasoning_chars": 30,
            "tool_calls": len(tools), "tool_names": [n for n, _ in tools],
            "tool_call_sha256": [tool_call_sha256(n, a) for n, a in tools], **more}  # fmt: skip


def event(ts: float, kind: str, **data: Any) -> dict[str, Any]:
    return {"ts": ts, "type": kind, "data": data}


FETCH_1 = ("web_fetch", '{"url": "https://example.org/doc", "part": 1}')
FETCH_2 = ("web_fetch", '{"url": "https://example.org/doc", "part": 2}')


def test_a_direct_run_has_no_stream_figures() -> None:
    calls = [call(1, 1000, 0, [FETCH_1]), call(2, 3000, 900, [FETCH_2, FETCH_1]), call(3, 200, 0, role="aux"),
             call(4, 2000, 1800, [("file_write", '{"path": "a.md"}')])]  # fmt: skip
    b = metrics.block(calls, wall_s=61.26)
    assert tuple(b) == metrics.KEYS
    assert (b["wall_s"], b["model_calls"], b["failed_model_calls"]) == (61.3, 4, 0)
    assert (b["tool_calls"], b["repeat_calls"]) == (4, 1)  # part 1 was fetched twice
    assert (b["prompt_first"], b["prompt_median"], b["prompt_max"]) == (
        1000,
        2000,
        3000,
    )  # main calls only
    assert (
        b["cached_share"] == 0.45
        and b["reasoning_tokens"] == 28
        and b["reasoning_chars"] == 120
    )
    assert (
        b["first_sign_s"] is None
        and b["failed_tools"] is None
        and b["approvals_asked"] is None
    )  # not 0: not countable
    assert (
        metrics.block(calls)["wall_s"] == 35.0
    )  # without the caller's time: the span of the model calls


def test_old_records_without_hashes_give_no_repeat_count() -> None:
    old = call(1, 1000, 0, [FETCH_1, FETCH_1])
    del old["tool_call_sha256"]
    assert metrics.block([old])["repeat_calls"] is None
    assert (
        metrics.block([call(1, 1000, 0)])["repeat_calls"] == 0
    )  # no tool call at all: nothing repeated


def test_failed_model_calls() -> None:
    calls = [call(1, 10, 0), call(2, 10, 0, status=500), call(3, 10, 0, error={"type": "x"}), call(4, 10, 0, stream_complete=False),
             {"call_index": 5, "role": None, "status": 404}]  # fmt: skip
    b = metrics.block(calls)
    assert (b["model_calls"], b["failed_model_calls"]) == (
        4,
        3,
    )  # the fifth row is not a generation request


def test_a_gateway_run_counts_from_its_events() -> None:
    events = [
        event(50.0, "turn.started"), event(50.2, "step.started"), event(52.1, "assistant.message", content=""),
        event(52.2, "tool.started", name="web_fetch", arguments={"url": "u", "part": 1}), event(53.0, "tool.finished", name="web_fetch", status="success"),
        event(60.0, "tool.started", name="web_fetch", arguments={"part": 1, "url": "u"}), event(61.0, "tool.finished", name="web_fetch", status="error"),
        event(62.0, "approval.requested", tool_name="file_write"), event(63.0, "tool.started", name="file_write", arguments={"path": "a"}),
        event(64.0, "tool.finished", name="file_write", status="success"), event(70.0, "turn.finished"),
    ]  # fmt: skip
    b = metrics.block([call(1, 500, 0), call(2, 900, 450)], events)
    assert (b["wall_s"], b["first_sign_s"]) == (
        20.0,
        2.1,
    )  # a step line is not a sign; the reply at 52.1 is
    assert (
        b["tool_calls"],
        b["repeat_calls"],
        b["failed_tools"],
        b["approvals_asked"],
    ) == (3, 1, 1, 1)
    assert b["cached_share"] == round(450 / 1400, 3)
    assert (
        metrics.block([], [])["tool_calls"] == 0
        and metrics.block([], [])["first_sign_s"] is None
    )  # an empty stream is still a stream


def test_nothing_recorded() -> None:
    b = metrics.block([])
    assert (
        b["model_calls"] == 0
        and b["wall_s"] is None
        and b["prompt_max"] is None
        and b["cached_share"] is None
    )
    assert metrics.line(b) == "0 model calls (0 failed) · 0 tool calls · 0 repeated"


def test_the_line() -> None:
    b = metrics.block([call(1, 1000, 250, [FETCH_1])], wall_s=12)
    assert metrics.line(b) == (
        "wall 12 s · 1 model calls (0 failed) · 1 tool calls · 0 repeated · "
        "prompt 1000 / 1000 / 1000 (first / median / max) · cached 25% · 7 thinking tokens"
    )
