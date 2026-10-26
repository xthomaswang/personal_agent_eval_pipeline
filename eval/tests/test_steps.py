"""Roles, tool-name hashes, the step budget, ttft_valid and the n_llm helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import httpx
import pytest

from bench.proxy.request_info import tool_names_sha256, tools_raw_sha256
from bench.proxy.steps import counts_as_step, is_ok, load_calls, new_image_long_sides, step_counts
from tests.support import MODEL, FakeUpstream, ProxyHandle

TOOLS = [
    {"name": "Read", "input_schema": {"type": "object"}},
    {"name": "Bash", "input_schema": {"type": "object"}},
]
OPENAI_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def messages(tools: list | None = None, **extra: object) -> dict:
    body: dict = {"model": MODEL, "max_tokens": 100, "messages": [{"role": "user", "content": "go"}], **extra}
    if tools is not None:
        body["tools"] = tools
    return body


def post(proxy: ProxyHandle, path: str, body: dict, scenario: str = "", run_id: str = "") -> httpx.Response:
    headers = {}
    if scenario:
        headers["x-fake-scenario"] = scenario
    if run_id:
        headers["X-Bench-Run-Id"] = run_id
    return httpx.post(proxy.url + path, json=body, headers=headers, timeout=10)


# -- role and tool names -----------------------------------------------------------


def test_tool_names_hash_is_sha256_of_the_sorted_names() -> None:
    expected = hashlib.sha256(b'["Bash","Read"]').hexdigest()
    assert tool_names_sha256(["Read", "Bash"]) == tool_names_sha256(["Bash", "Read"]) == expected
    assert tool_names_sha256([]) == hashlib.sha256(b"[]").hexdigest()


def test_raw_tools_hash_sees_order_and_schema_changes() -> None:
    read = {"name": "Read", "input_schema": {"type": "object"}}
    bash = {"name": "Bash", "input_schema": {"type": "object"}}
    edited = {"name": "Bash", "input_schema": {"type": "object", "required": ["command"]}}
    assert tools_raw_sha256([read, bash]) != tools_raw_sha256([bash, read])
    assert tools_raw_sha256([read, bash]) != tools_raw_sha256([read, edited])
    assert tools_raw_sha256([read, bash]) == tools_raw_sha256([dict(read), dict(bash)])
    assert tools_raw_sha256(None) is None


def test_role_and_tool_names(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    post(proxy, "/v1/messages", messages(TOOLS))
    post(proxy, "/v1/messages", messages([]))  # empty tools list: aux
    post(proxy, "/v1/messages", messages())  # e.g. title generation: aux
    post(proxy, "/v1/chat/completions", messages(OPENAI_TOOLS))
    post(proxy, "/v1/messages/count_tokens", messages(TOOLS))  # carries tools, still never main
    httpx.head(proxy.url + "/api/hello", timeout=10)
    main, empty, no_tools, openai, counted, hello = proxy.records()

    assert main["role"] == "main" and main["main_index"] == 1
    assert main["request_tool_names"] == ["Bash", "Read"]
    assert main["request_tools_sha256"] == tool_names_sha256(["Bash", "Read"])
    assert main["request_tools_raw_sha256"] == tools_raw_sha256(TOOLS)
    assert hello["request_tools_raw_sha256"] is None
    assert (empty["role"], empty["main_index"], empty["request_tool_names"]) == ("aux", None, [])
    assert empty["request_tools_sha256"] == tool_names_sha256([])
    assert no_tools["role"] == "aux"
    assert (openai["role"], openai["main_index"], openai["request_tool_names"]) == ("main", 2, ["get_weather"])
    assert counted["role"] is None and counted["main_index"] is None
    assert counted["request_tool_names"] == ["Bash", "Read"]
    assert hello["role"] is None and hello["request_tool_names"] is None


# -- step budget ---------------------------------------------------------------------


def test_budget_refuses_main_calls_beyond_max_steps(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(max_steps=2)
    assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 200
    assert post(proxy, "/v1/messages", messages()).status_code == 200  # aux calls are not steps
    assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 200
    refused = post(proxy, "/v1/messages", messages(TOOLS))
    assert refused.status_code == 400
    assert refused.json()["type"] == "error"
    assert refused.json()["error"]["message"].startswith("bench proxy: bench_budget_exhausted")
    openai = post(proxy, "/v1/chat/completions", messages(OPENAI_TOOLS))
    assert openai.status_code == 400 and openai.json()["error"]["code"] == "bench_budget_exhausted"
    assert len(upstream.requests) == 3  # the refused calls never reached the model

    records = proxy.records()
    assert [r["main_index"] for r in records] == [1, None, 2, 3, 4]
    assert [r["budget_exhausted"] for r in records] == [False, False, False, True, True]
    assert all(r["max_steps"] == 2 for r in records)
    last = records[-1]
    assert last["handled_locally"] is True and last["status"] == 400
    assert last["error"]["type"] == "bench_budget_exhausted" and last["error"]["source"] == "proxy"
    assert step_counts(records) == {
        "n_llm": 2,
        "n_llm_aux": 1,
        "n_llm_err": 2,
        "n_llm_post_submit": 0,
        "n_tool": 2,
        "images_new_sum": 0,
    }


def test_failed_main_calls_give_their_step_back(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(max_steps=1)
    assert post(proxy, "/v1/messages", messages(TOOLS), scenario="error-503").status_code == 503
    assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 200
    assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 400
    assert [r["budget_exhausted"] for r in proxy.records()] == [False, False, True]


def test_budget_record_mode_only_flags(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(max_steps=1, budget_mode="record")
    for _ in range(3):
        assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 200
    records = proxy.records()
    assert [r["budget_exhausted"] for r in records] == [False, True, True]
    assert all(not r["handled_locally"] for r in records)
    assert step_counts(records)["n_llm"] == 3


def test_budget_is_per_run_id(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(max_steps=1)
    assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 200
    assert post(proxy, "/v1/messages", messages(TOOLS), run_id="warmup").status_code == 200
    assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 400
    assert [r["main_index"] for r in proxy.records()] == [1, 1, 2]


def test_no_budget_by_default(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    for _ in range(3):
        assert post(proxy, "/v1/messages", messages(TOOLS)).status_code == 200
    assert all(r["max_steps"] is None and r["budget_exhausted"] is False for r in proxy.records())


# -- ttft_valid ----------------------------------------------------------------------


@pytest.mark.parametrize("path,tools", [("/v1/chat/completions", OPENAI_TOOLS), ("/v1/messages", TOOLS)])
def test_ttft_valid(upstream: FakeUpstream, proxy: ProxyHandle, path: str, tools: list) -> None:
    with httpx.stream("POST", proxy.url + path, json=messages(tools, stream=True), timeout=10) as r:
        r.read()  # thinking streams first: a real first token
    with httpx.stream(
        "POST", proxy.url + path, json=messages(tools, stream=True), headers={"x-fake-scenario": "tool-only"}, timeout=10
    ) as r:
        r.read()  # only a tool call, sent after generation ended
    post(proxy, path, messages(tools))  # not streamed: no first-token time at all
    streamed, tool_only, plain = proxy.records()
    assert streamed["ttft_valid"] is True
    assert tool_only["ttft_valid"] is False and tool_only["tool_calls"] == 1 and tool_only["ttft_s"] is not None
    assert plain["ttft_valid"] is None and plain["ttft_s"] is None


# -- helpers ----------------------------------------------------------------------------


def rec(start: float, role: str | None = "main", status: int = 200, endpoint: str = "messages", **extra: object) -> dict:
    return {
        "endpoint": endpoint,
        "role": role,
        "status": status,
        "error": None if status == 200 else {"source": "upstream"},
        "stream_complete": True,
        "mono_start": start,
        "tool_calls": 1 if role == "main" else 0,
        "images_new": 0,
        **extra,
    }


def test_step_counts_window_and_partition() -> None:
    records = [
        rec(5.0),  # before t0
        rec(10.0, images_new=2, images_new_long_side=[1280, 1280]),
        rec(11.0, tool_calls=3),
        rec(12.0, role="aux"),
        rec(13.0, status=503),
        rec(14.0, stream_complete=False),  # stream cut off
        rec(15.0, role=None, endpoint="count_tokens"),  # never counted
        rec(16.0, endpoint="other", role=None),
        rec(25.0),  # after t_end (post submit)
        rec(26.0, role="aux"),
    ]
    assert step_counts(records, t0=10.0, t_end=20.0) == {
        "n_llm": 2,
        "n_llm_aux": 1,
        "n_llm_err": 2,
        "n_llm_post_submit": 1,
        "n_tool": 4,
        "images_new_sum": 2,
    }
    assert step_counts(records)["n_llm"] == 4
    assert new_image_long_sides(records, t0=10.0, t_end=20.0) == [1280, 1280]
    assert is_ok(rec(1.0)) and not is_ok(rec(1.0, stream_complete=False))
    assert counts_as_step(rec(1.0)) and not counts_as_step(rec(1.0, role="aux"))
    # a complete non-streaming call has stream_complete=None
    assert counts_as_step(rec(1.0, stream_complete=None))


def test_load_calls_reads_a_run_directory(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    post(proxy, "/v1/messages", messages(TOOLS))
    post(proxy, "/v1/messages", messages())
    records = load_calls(proxy.run_dir)
    assert records == load_calls(proxy.calls_path) and len(records) == 2
    counts = step_counts(records)
    assert (counts["n_llm"], counts["n_llm_aux"], counts["n_tool"]) == (1, 1, 1)
