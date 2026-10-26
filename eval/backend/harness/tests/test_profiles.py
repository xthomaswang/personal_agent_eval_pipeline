import asyncio
import json

import pytest

from agentbench.cli import main
from agentbench.profiles import PROFILES, get_profile

from conftest import cli_args, start_mock

HERMES_BODY = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 8,
               "seed": 7, "reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True},
               "stream_options": {"include_usage": False, "continuous_usage_stats": True}}


@pytest.mark.parametrize("name", ["llamacpp", "omlx", "vllm", "lithos", "mlxlm", "generic"])
def test_stream_options_always_exactly_include_usage(name):
    body, _ = get_profile(name).shape_openai(HERMES_BODY)
    assert body["stream_options"] == {"include_usage": True}


def test_mlxlm_omits_model_and_seed():
    body, notes = get_profile("mlxlm").shape_openai(HERMES_BODY)
    assert "model" not in body and "seed" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}  # still sent; works on mlx-lm
    assert any("omitted model" in n for n in notes) and any("seed" in n for n in notes)
    ab, _ = get_profile("mlxlm").shape_anthropic({"model": "x", "max_tokens": 4, "messages": []})
    assert "model" not in ab


def test_lithos_strips_thinking_fields():
    p = get_profile("lithos")
    body, notes = p.shape_openai(HERMES_BODY)
    assert "chat_template_kwargs" not in body and "reasoning_effort" not in body and body["model"] == "qwen"
    assert len([n for n in notes if "stripped" in n]) == 2
    body, _ = p.shape_openai(dict(HERMES_BODY, reasoning_effort="none"))
    assert body["reasoning_effort"] == "none"
    for th, kept in (("adaptive", False), ("enabled", False), ("disabled", True)):
        ab, _ = p.shape_anthropic({"model": "m", "max_tokens": 4, "messages": [], "thinking": {"type": th}})
        assert ("thinking" in ab) is kept
    assert p.thinking_blocked(True) and not p.thinking_blocked(False)
    assert p.anthropic_thinking_blocked("adaptive") and not p.anthropic_thinking_blocked(None)
    assert p.concurrency == "reject-429" and p.cached_source == "server-log"


@pytest.mark.parametrize("name", ["llamacpp", "omlx", "vllm"])
def test_pass_through_profiles_keep_hermes_fields(name):
    body, notes = get_profile(name).shape_openai(HERMES_BODY)
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["reasoning_effort"] == "medium"
    assert body["model"] == "qwen" and body["seed"] == 7
    assert all("stripped" not in n for n in notes)


def test_profile_ports_and_dirs():
    assert {p.port for p in PROFILES.values() if p.port} == {8700, 8701, 8702, 8703, 8704, 8705}
    assert get_profile("vllm").framework == "vllm-metal" and get_profile("mlxlm").framework == "mlx-lm"
    assert get_profile("mlc").framework == "mlc-llm" and get_profile("mlc").tool_call_mode == "structured_output"


@pytest.mark.parametrize("profile,behavior", [("mlxlm", "mlxlm"), ("lithos", "lithos"), ("vllm", "vllm"),
                                              ("llamacpp", "llamacpp"), ("omlx", "omlx")])
def test_profile_shaping_end_to_end(built, tmp_path, profile, behavior):
    """The gate sends real Hermes / Claude Code shapes; the mock records what arrived."""
    root, _ = built
    srv, resp = start_mock(root, behavior, log_path=str(tmp_path / "server.log"))
    try:
        extra = ["--server-log", str(tmp_path / "server.log")] if profile == "lithos" else []
        assert main(["gate"] + cli_args(srv.url, root, tmp_path, profile, "fw", *extra)) == 0
    finally:
        srv.stop()
    chat = [r for r in resp.requests if r["path"].startswith("/v1/chat")]
    msgs = [r for r in resp.requests if r["path"].startswith("/v1/messages")]
    assert chat and all(r["stream_options"] == {"include_usage": True} for r in chat)
    assert resp.stats["seed_requests"] == 0
    if profile == "mlxlm":
        assert all(r["model"] == "<absent>" for r in resp.requests) and resp.stats["model_reloads"] == 0
    else:
        assert all(r["model"] == "mock" for r in chat)
    if profile == "lithos":
        assert all(r["chat_template_kwargs"] is None for r in chat)
        assert all(r["reasoning_effort"] in (None, "none") for r in chat)
        assert all(r["thinking"] in (None, {"type": "disabled"}) for r in msgs)
    else:
        assert any(r["chat_template_kwargs"] == {"enable_thinking": False} and r["reasoning_effort"] == "none" for r in chat)
        assert any(r["thinking"] == {"type": "adaptive"} for r in msgs) or profile == "mlxlm"
    summary = [json.loads(x) for x in next((tmp_path / "fw" / "raw").glob("gate-*.jsonl")).read_text().splitlines()][-1]
    items = summary["items"]
    expect = {"mlxlm": {"G1": "PASS", "G2": "FAIL", "G3": "FAIL", "G4": "PASS"},
              "lithos": {"G1": "PASS", "G2": "FAIL", "G3": "FAIL", "G4": "FAIL"},
              "vllm": {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS"},
              "llamacpp": {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS"},
              "omlx": {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS"}}[profile]
    assert {k: items[k]["result"] for k in expect} == expect
    if profile == "lithos":
        assert "thinking cannot be enabled" in items["G4"]["detail"]
        assert any(p["probe"] == "G3.text-adaptive" and p["outcome"].startswith("N/A") for p in summary["probes"])
    if profile == "mlxlm":
        assert "404" in items["G2"]["detail"] and "top-level effort ignored" in items["G4"]["detail"]
    if profile in ("vllm", "mlxlm"):
        assert "in reasoning" in items["G4"]["detail"]  # reasoning arrived in the `reasoning` field
    if profile in ("omlx", "mlxlm"):
        assert "one delta" in items["G1"]["detail"]


def test_reasoning_fields_and_llamacpp_timings(built, tmp_path):
    from agentbench.client import ServerConfig, StreamClient
    from agentbench.mock_server import MockServer
    from agentbench.mock_smart import SmartResponder
    from agentbench.suites.common import Outcome

    async def one(responder):
        srv = MockServer(port=0, responder=responder).start()
        try:
            async with StreamClient(ServerConfig(base_url=srv.url, model="m")) as c:
                body = c.openai_body([{"role": "user", "content": "hi"}], 16,
                                     extra={"chat_template_kwargs": {"enable_thinking": True}})
                return await c.chat(body)
        finally:
            srv.stop()

    for field, expect in (("reasoning", ["reasoning"]), ("reasoning_content", ["reasoning_content"]),
                          ("inline", ["inline_think"])):
        rec = asyncio.run(one(SmartResponder("generic", reasoning_field=field)))
        assert rec.reasoning and rec.reasoning_fields == expect and "<think>" not in rec.content
    # llama.cpp: cached tokens from timings.cache_n when usage has no prompt_tokens_details
    resp = SmartResponder("llamacpp", cached_report="timings_only")
    rec = asyncio.run(one(resp))
    m = Outcome(rec, [], {}).metrics()
    assert m["cached_source"] == "timings.cache_n" and rec.timings_raw is not None


def test_lithos_sends_all_text_content_as_strings():
    p = get_profile("lithos")
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "[run 1] "}, {"type": "text", "text": "screen text"}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}},
                                         {"type": "text", "text": "what is this?"}]}]
    body, notes = p.shape_openai(dict(HERMES_BODY, messages=msgs))
    assert body["messages"][0]["content"] == "sys"
    assert body["messages"][1]["content"] == "[run 1] screen text"
    assert body["messages"][2]["content"] == "ok"
    assert isinstance(body["messages"][3]["content"], list)  # image probes stay as they are
    assert any("2 all-text content list(s)" in n for n in notes)
    assert msgs[1]["content"][0]["type"] == "text"  # the caller's body is not modified
    body, _ = get_profile("mlxlm").shape_openai(dict(HERMES_BODY, messages=msgs))
    assert isinstance(body["messages"][1]["content"], list)  # mlx-lm takes text parts


@pytest.mark.parametrize("name,forced", [("lithos", True), ("mlxlm", True), ("mlc", True),
                                         ("llamacpp", False), ("omlx", False), ("vllm", False)])
def test_text_only_is_forced_for_servers_that_reject_images(name, forced):
    # Each suite runs in its own process, so the gate's auto switch cannot carry over; the profile must say it.
    assert bool(get_profile(name).force_text_only) is forced
