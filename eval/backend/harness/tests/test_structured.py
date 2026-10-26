import json

import jsonschema
import pytest

from agentbench.cli import main
from agentbench.fixtures.agent_tools import TOOLS
from agentbench.profiles import get_profile
from agentbench.structured import agent_step_schema, parse_step, to_structured_request

from conftest import cli_args, start_mock


def _body(fx, msgs, nonce=None):
    return {"model": "m", "stream": True, "max_tokens": 64, "messages": msgs, "tools": fx.tools(nonce=nonce),
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}


def test_mlc_shaping_inlines_tools_and_flattens_history(fx):
    case = next(c for c in fx.m7["cases"] if c["id"] == "m7-16-commit-message")
    body, notes = get_profile("mlc").shape_openai(_body(fx, [fx.agent_system()] + case["messages"], nonce="123456789012"))
    assert "tools" not in body and "chat_template_kwargs" not in body and "reasoning_effort" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user", "user"]
    sys = body["messages"][0]["content"]
    assert sys.startswith("[run 123456789012]\n# Tools") and fx.system_prompt in sys
    assert "[run " not in sys[20:]  # the nonce moved out of the tool description
    assert json.loads(body["messages"][2]["content"]) == {"tool": "git_status", "arguments": {}}
    assert body["messages"][3]["content"].startswith("Tool result (git_status):\n")
    rf = body["response_format"]
    assert rf["type"] == "json_object" and isinstance(rf["schema"], str)
    assert any("flattened" in n for n in notes) and body["stream_options"] == {"include_usage": True}


def test_step_schema_validates_calls_final_and_parallel():
    schema = agent_step_schema(TOOLS)
    assert "anyOf" in schema and "oneOf" not in json.dumps(schema) and "description" not in json.dumps(schema["$defs"])
    v = jsonschema.Draft202012Validator(schema)
    good = [{"tool": "fs_read_file", "arguments": {"path": "README.md"}}, {"final": "done"},
            {"tool_calls": [{"tool": "fs_read_file", "arguments": {"path": "a"}},
                            {"tool": "git_status", "arguments": {}}]}]
    bad = [{"tool": "fs_read_file", "arguments": {"bogus": 1}}, {"tool": "no_such_tool", "arguments": {}},
           {"final": "x", "tool": "fs_read_file"}, {"tool_calls": [{"tool": "git_status", "arguments": {}}]}]
    assert all(v.is_valid(x) for x in good) and not any(v.is_valid(x) for x in bad)
    assert "oneOf" in agent_step_schema(TOOLS, union="oneOf")


@pytest.mark.parametrize("style,expect", [("mlc", "json_object"), ("openai", "json_schema"), ("none", None)])
def test_response_format_styles(fx, style, expect):
    from dataclasses import replace
    body, _ = replace(get_profile("mlc"), structured_format=style).shape_openai(
        _body(fx, [fx.agent_system(), {"role": "user", "content": "hi"}]))
    assert (body.get("response_format") or {}).get("type") == expect
    if style == "openai":
        assert body["response_format"]["json_schema"]["strict"] is True


def test_parse_step_variants():
    calls, final, err = parse_step('```json\n{"tool": "fs_read_file", "arguments": {"path": "x"}}\n```')
    assert not err and calls[0]["name"] == "fs_read_file" and calls[0]["arguments_parsed"] == {"path": "x"}
    calls, final, err = parse_step('{"tool_calls": [{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {}}]}')
    assert [c["name"] for c in calls] == ["a", "b"] and [c["index"] for c in calls] == [0, 1]
    assert parse_step('{"final": "hello"}') == ([], "hello", None)
    assert parse_step("not json")[2].startswith("content is not JSON")
    assert parse_step('{"tool": "a", "arguments": "x"}')[0][0]["arguments_parsed"] is None


def test_mlc_profile_end_to_end(built, tmp_path):
    root, _ = built
    srv, resp = start_mock(root, "mlc")
    try:
        rc = main(["run-all"] + cli_args(srv.url, root, tmp_path, "mlc", "mlc-llm", "--suites", "gate,m1,m4,m7"))
    finally:
        srv.stop()
    assert rc == 0
    chat = [r for r in resp.requests if r["path"].startswith("/v1/chat")]
    assert chat and not any(r["has_tools"] for r in chat) and not any("tool" in r["roles"] for r in chat)
    assert all(r["chat_template_kwargs"] is None and r["reasoning_effort"] is None for r in chat)
    raw = tmp_path / "mlc-llm" / "raw"
    gate = [json.loads(x) for x in next(raw.glob("gate-*.jsonl")).read_text().splitlines()]
    items = gate[-1]["items"]
    assert items["G1"]["result"] == "PASS via structured output (non-standard)"
    assert [items[k]["result"] for k in ("G2", "G3", "G4")] == ["FAIL", "FAIL", "FAIL"]
    assert "no per-request thinking control" in items["G4"]["detail"]
    assert gate[0]["mode"]["tool_call_mode"] == "structured_output" and gate[0]["mode"]["structured_format"] == "mlc"
    for suite in ("m1", "m4", "m7"):
        lines = [json.loads(x) for x in next(raw.glob(f"{suite}-*.jsonl")).read_text().splitlines()]
        assert lines[0]["mode"]["text_only"] and "nightly" in lines[0]["mode"]["text_only_reason"]
        reqs = [x for x in lines if x["type"] == "request"]
        assert reqs and all(x["tool_call_mode"] == "structured_output" and x["metrics"]["ok"] for x in reqs)
        assert all(x["prompt"]["n_images"] == 0 for x in reqs)
    m1 = [json.loads(x) for x in next(raw.glob("m1-*.jsonl")).read_text().splitlines()]
    warm = [x["metrics"] for x in m1 if x.get("phase") == "warm"]
    assert all(m["cached_tokens"] == 0 and m["cached_source"].startswith("usage.extra") for m in warm)
    m7 = [json.loads(x) for x in next(raw.glob("m7-*.jsonl")).read_text().splitlines()]
    assert m7[-1]["passed"] == m7[-1]["scored"] == 5 and m7[-1]["na"] == 1
    calls = [x for x in m7 if x.get("type") == "request" and x["cell"].get("case") == "m7-18-parallel"][0]["calls"]
    assert [c["name"] for c in calls] == ["fs_read_file", "fs_read_file"]
    assert "structured output" in (tmp_path / "mlc-llm" / "summary.md").read_text()


def test_mlc_openai_style_rejected_then_instruction_only_probe(built, tmp_path):
    root, _ = built
    srv, resp = start_mock(root, "mlc")
    try:
        assert main(["gate"] + cli_args(srv.url, root, tmp_path, "mlc", "mlc-llm", "--structured-format", "openai")) == 0
    finally:
        srv.stop()
    gate = [json.loads(x) for x in next((tmp_path / "mlc-llm" / "raw").glob("gate-*.jsonl")).read_text().splitlines()]
    g1 = gate[-1]["items"]["G1"]
    assert g1["result"] == "FAIL" and "422" in g1["detail"]
    assert any(s["probe"] == "G1.no-grammar" and s["ok"] for s in g1["sub"])
