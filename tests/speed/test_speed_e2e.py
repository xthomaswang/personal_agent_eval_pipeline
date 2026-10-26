"""The functional block and every timed unit end to end against agentbench's mock server (ephemeral
port), with /v1/tokenize and /v1/detokenize served from a fake renderer whose token counts also feed
the chat usage, so G0's checks pass or fail for real."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from speedkit import NANBEIGE, make_model

from modeleval import speed

BOS = 1_200_000  # outside the code-point range the fake tokenizer uses
CLICK = {"name": "computer_click", "arguments": {"x": 287, "y": 361, "button": "left", "clicks": 1}}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def fake_render(messages: list[dict], tools: list[dict] | None, kwargs: dict) -> str:
    """A Qwen-like chat template: tools first in the system turn, Qwen3-Coder XML tool calls."""
    out = ""
    for m in messages:
        role, body = m["role"], _text(m.get("content"))
        if role == "system":
            out += "<|im_start|>system\n" + (f"# Tools\n{json.dumps(tools)}\n\n" if tools else "") + body + "<|im_end|>\n"
        elif role == "assistant":
            out += "<|im_start|>assistant\n" + body
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                out += f"\n\n<tool_call>\n<function={fn['name']}>\n" + "".join(
                    f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>"
            out += "<|im_end|>\n"
        elif role == "tool":
            out += f"<|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n"
        else:
            out += f"<|im_start|>user\n{body}<|im_end|>\n"
    think = kwargs.get("enable_thinking", True)
    return out + "<|im_start|>assistant\n<think>\n" + ("" if think else "\n</think>\n\n")


def fake_ids(messages: list[dict], tools: list[dict] | None, kwargs: dict, double_bos: bool = False) -> list[int]:
    return ([BOS, BOS] if double_bos else []) + [ord(c) for c in fake_render(messages, tools, kwargs)]


def make_responder(fixtures_root: Path, **knobs: Any):
    from agentbench.mock_smart import SmartResponder

    class Tokenizing(SmartResponder):
        """SmartResponder plus oMLX's /v1/tokenize and /v1/detokenize; chat usage uses the same counts."""
        double_bos = False
        usage_offset = 0
        tokenize_bodies: list[dict] = []

        def ids(self, body: dict) -> list[int]:
            return fake_ids(body["messages"], body.get("tools"), body.get("chat_template_kwargs") or {}, self.double_bos)

        async def __call__(self, method, path, headers, body):
            self._body = None
            if method == "POST" and path.startswith("/v1/tokenize"):
                self.tokenize_bodies.append(body)
                ids = self.ids(body)
                return "json", 200, {"count": len(ids), "max_model_len": 131072, "tokens": ids}
            if method == "POST" and path.startswith("/v1/detokenize"):
                return "json", 200, {"prompt": "".join(chr(i) for i in body["tokens"] if i < 0x110000)}
            return await super().__call__(method, path, headers, body)

        def _openai(self, body):
            self._body = body
            return super()._openai(body)

        def _prompt_cache(self, segments):
            total, cached = super()._prompt_cache(segments)
            body, self._body = getattr(self, "_body", None), None
            if body is None or not total:
                return total, cached
            n = len(self.ids(body)) + self.usage_offset
            return n, min(n - 1, int(cached * n / total))

    r = Tokenizing.from_fixtures(fixtures_root, "generic", default_call=CLICK)
    r.tokenize_bodies = []
    for k, v in knobs.items():
        setattr(r, k, v)
    return r


@pytest.fixture
def harness_ready():
    from agentbench.suites.common import RunConfig
    from agentbench.suites.opts import SuiteOpts

    need = {"vision", "cold_nonce", "anthropic_sampling", "tokenizer_dir"} - {f for f in RunConfig.__dataclass_fields__}
    need |= {"think_effort", "warmup_max_tokens", "reuse_nonce"} - {f for f in SuiteOpts.__dataclass_fields__}
    if need:
        pytest.skip(f"harness lacks {sorted(need)}")


@pytest.fixture
def server(fixtures_root, monkeypatch, request, harness_ready):
    from agentbench.mock_server import MockServer

    knobs = getattr(request, "param", {})
    resp = make_responder(fixtures_root, **knobs.get("responder", {}))
    srv = MockServer(port=0, responder=resp).start()
    monkeypatch.setattr(speed, "FIXTURES_DIR", fixtures_root)
    offset = knobs.get("render_offset", 0)

    def render(tokenizer_dir, messages, tools, kwargs):
        ids = fake_ids(messages, tools, kwargs)
        if offset:
            ids[offset] += 1
        return {"ids": ids, "transformers": "fake"}

    monkeypatch.setattr(speed, "independent_render", render)
    yield srv, resp
    srv.stop()


def ctx_for(tmp_path: Path, srv, **model_kw) -> speed.SpeedContext:
    return speed.SpeedContext(make_model(**model_kw), srv.url + "/v1", 0, tmp_path / "results" / "m" / "run1",
                              tmp_path / "no-tokenizer", tmp_path / "results" / "_unverified" / "m" / "run1")


async def test_functional_block_text_only_model(server, tmp_path):
    srv, resp = server
    ctx = ctx_for(tmp_path, srv, **NANBEIGE)
    fr = await speed.run_functional(ctx)
    assert fr.ok and fr.stop_reason is None, fr.items
    assert [fr.items[g]["result"] for g in ("G0", "G1", "G3", "G4")] == ["PASS"] * 4
    assert "G2" not in fr.items  # vision: none
    assert "N/A" in fr.items["G3"]["detail"]
    g0 = {c["check"]: c["ok"] for c in fr.items["G0"]["checks"]}
    assert g0["independent render"] and g0["G1 usage"] and g0["tool-call turn rendered"]
    assert fr.m7["scored_cases"] == 22 and fr.m7["trials"] == 110 and fr.m7["na"] == ["m7-22-image-read",
                                                                                         "m7-23-image-click"]
    assert [p.name for p in fr.files] == ["gate-1.jsonl", "m7-1.jsonl"]
    assert all(p.parent == speed.speed_dir(ctx.run_dir) for p in fr.files)
    # the three tokenize requests: thinking off, add_special_tokens unset
    assert len(resp.tokenize_bodies) == 3
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} and "add_special_tokens" not in b
               for b in resp.tokenize_bodies)
    assert [len(b.get("tools") or []) for b in resp.tokenize_bodies] == [0, 44, 44]
    # sampling in every generation request, OpenAI and Anthropic alike; no image was sent
    gen = [r for r in resp.requests if r["path"].startswith("/v1/")]
    assert gen and all(r["sampling"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 0} for r in gen)
    assert any(r["path"].startswith("/v1/messages") for r in gen)
    assert all(r["n_images"] == 0 for r in gen)
    doc = speed.load_units(ctx.run_dir)
    f = speed.functional_result(doc)
    assert doc["functional"]["passed"] and f["ok"] and f["files"] == ["speed/gate-1.jsonl", "speed/m7-1.jsonl"]
    assert doc["model"] == "nanbeige" and doc["model_info"]["vision"] is False and doc["system"]["macos_build"]
    assert not (speed.speed_dir(ctx.run_dir) / "_agentbench").exists()


@pytest.mark.parametrize("server", [{"responder": {"double_bos": True}}], indirect=True)
async def test_g0_doubled_bos_stops_the_run(server, tmp_path):
    srv, resp = server
    fr = await speed.run_functional(ctx_for(tmp_path, srv, **NANBEIGE))
    assert not fr.ok and fr.stop_reason == "G0 failed"
    assert fr.items["G0"]["result"] == "FAIL" and "single BOS" in fr.items["G0"]["detail"]
    assert set(fr.items) == {"G0"} and fr.m7 is None
    assert not any(r["path"].startswith("/v1/chat") for r in resp.requests)  # nothing generated after the stop


@pytest.mark.parametrize("server", [{"render_offset": 40}], indirect=True)
async def test_g0_independent_render_mismatch(server, tmp_path):
    srv, _ = server
    fr = await speed.run_functional(ctx_for(tmp_path, srv))
    assert not fr.ok and "first difference at index 40" in fr.items["G0"]["detail"]


@pytest.mark.parametrize("server", [{"responder": {"usage_offset": 1}}], indirect=True)
async def test_g0_usage_mismatch_stops_after_g1(server, tmp_path):
    srv, _ = server
    fr = await speed.run_functional(ctx_for(tmp_path, srv))
    assert not fr.ok and fr.stop_reason == "G0 failed (G1 usage)"
    assert fr.items["G1"]["result"] == "PASS" and "G3" not in fr.items


async def test_missing_prompt_fixtures_stop_before_any_request(server, tmp_path, fixtures_root, monkeypatch):
    import shutil

    srv, resp = server
    root = tmp_path / "fx"
    shutil.copytree(fixtures_root, root)
    (root / "generated/prompts/m3-off-110000.json").unlink()
    monkeypatch.setattr(speed, "FIXTURES_DIR", root)
    fr = await speed.run_functional(ctx_for(tmp_path, srv))
    assert not fr.ok and "m3-off-110000" in fr.stop_reason and not resp.requests


async def test_every_timed_unit(server, tmp_path):
    srv, resp = server
    ctx = ctx_for(tmp_path, srv)  # spec mtp: includes the thinking cell
    results = {}
    for unit in speed.timed_units(ctx.model):
        n0 = len(resp.requests)
        res = await speed.run_unit(ctx, unit, 1)
        results[unit] = (res, resp.requests[n0:])
        assert res.valid, (unit, res.note)
        assert res.raw_file == speed.speed_dir(ctx.run_dir) / f"{unit}-1.jsonl" and res.raw_file.exists()
        assert res.t_start <= res.t_end
    g5 = results["g5+m3-64k"][0].metrics
    assert g5["g5"]["ok"] and g5["g5"]["cached_tokens"] == 0 and g5["m3"]["warmup"] == "reused"
    assert g5["m3"]["trials"] == 3 and g5["m3"]["decode_tps"]["median"] > 0
    assert results["m3-16k"][0].metrics["m3"]["warmup"] == "normal"
    # warm-ups decode 16 tokens; the thinking cell sends enable_thinking only
    think = results["m3-think-16k"][1]
    assert think[0]["max_tokens"] == 16 and all(r["chat_template_kwargs"] == {"enable_thinking": True}
                                                and r["reasoning_effort"] is None for r in think)
    assert results["m3-think-16k"][0].metrics["m3"]["cap"] == 1000
    off = results["m3-32k"][1]
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} and r["reasoning_effort"] is None for r in off)
    m1 = results["m1"][0].metrics["m1"]
    assert m1["warm_ttft_s"]["n"] == 5 and m1["cold_ttft_s"]["n"] == 1 and m1["cold_cached_tokens"] == 0
    m4 = results["m4-1"][0].metrics["m4"]
    assert m4["steps"] == 21 and m4["headline_s"] and m4["step0_cached_tokens"] == 0
    # cold rule system+tool: the M4 run's nonce starts the system message and the first tool
    step0 = next(r for r in results["m4-1"][1] if r["has_tools"])
    assert step0["system"].startswith("[run ")
    doc = speed.load_units(ctx.run_dir)
    assert sorted(doc["units"]) == sorted(speed.timed_units(ctx.model))
    assert doc["units"]["m4-1"]["attempts"][0]["raw_file"] == "speed/m4-1-1.jsonl"
    raw = [json.loads(x) for x in (speed.speed_dir(ctx.run_dir) / "m4-1-1.jsonl").read_text().splitlines()]
    assert raw[0]["type"] == "header" and raw[0]["mode"]["cold_nonce"] == "system+tool"
    assert raw[0]["mode"]["text_only"] and raw[0]["mode"]["thinking_fields"] == "kwargs"
    summ = next(x for x in raw if x.get("type") == "summary")
    assert summ["headline_s"][0] == pytest.approx(m4["headline_s"])  # units.json agrees with the harness


async def test_unit_errors_are_recorded_not_raised(server, tmp_path):
    import dataclasses
    import socket

    srv, _ = server
    with socket.socket() as s:  # an ephemeral port nobody listens on: the server is gone
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    ctx = dataclasses.replace(ctx_for(tmp_path, srv), base_url=f"http://127.0.0.1:{dead}/v1")
    res = await speed.run_unit(ctx, "m3-16k", 2)
    assert not res.valid and res.raw_file.name == "m3-16k-2.jsonl"
    with pytest.raises(ValueError):
        await speed.run_unit(ctx_for(tmp_path, srv, **NANBEIGE), "m3-think-16k", 1)
