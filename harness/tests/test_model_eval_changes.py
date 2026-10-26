"""The agent-model-eval changes against qwen3.8-27b_20261017-2258_benchmark@c15a0a8 (harness/CHANGES.md),
driven in-process the way modeleval's speed leg drives them. Mock servers on ephemeral ports only."""

import json
import shutil
import time
from pathlib import Path

import pytest

from agentbench.fixtures.build import COMMITTED, EXTRA_PROMPTS, ZERO12
from agentbench.fixtures.store import Fixtures
from agentbench.profiles import get_profile
from agentbench.suites import run_suite, validate
from agentbench.suites.common import RunConfig, Target
from agentbench.suites.gate import NO_VISION, run_probes
from agentbench.suites.opts import SuiteOpts

from conftest import start_mock

SHORT_CALL = {"name": "fs_read_file", "arguments": {"path": "docs/setup.md"}}
LONG_CALL = {"name": "fs_write_file", "arguments": {"path": "notes/release.md", "content": "release checklist " * 30}}


def cfg_for(url: str, root: Path, out: Path, **kw) -> RunConfig:
    return RunConfig(profile=get_profile("omlx"), base_url=url, model="mock", framework="omlx", out_root=out,
                     fixtures_dir=root, **{"tag": "t", **kw})


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def requests(lines: list[dict], **match) -> list[dict]:
    return [x for x in lines if x.get("type") == "request" and all(x.get(k) == v for k, v in match.items())]


@pytest.fixture
def mock(built):
    servers = []

    def make(behavior: str = "generic", **overrides):
        srv, resp = start_mock(built[0], behavior, **overrides)
        servers.append(srv)
        return srv, resp

    yield make
    for s in servers:
        s.stop()


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> Path:
    """A word-level tokenizer whose template lives in tokenizer_config.json and uses bos_token, as
    Nanbeige's does (no chat_template.jinja, no preprocessor_config.json)."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    d = tmp_path_factory.mktemp("tiny-tokenizer")
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "<s>": 1, "hello": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    template = ("{{ bos_token }}{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                "{% if tools %}tools: {{ tools | length }}\n{% endif %}{% if add_generation_prompt %}assistant:{% endif %}")
    (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": {"content": "<s>"}, "chat_template": template}))
    return d


# ---------------------------------------------------------------- 1. gate probes on their own, vision
async def test_gate_probes_run_alone_and_vision_false_passes_g3_on_text_and_tool_use(built, fx, mock, tmp_path):
    srv, resp = mock("generic", images=False)  # a text-only model: every image request is rejected
    cfg = cfg_for(srv.url, built[0], tmp_path)
    path, s = await run_probes(cfg, [("g1", {}), ("g3", {"vision": False}), ("g4", {})], fx=fx)
    assert {k: v["result"] for k, v in s["items"].items()} == {"G1": "PASS", "G3": "PASS", "G4": "PASS"}
    g3 = s["items"]["G3"]
    assert not g3["vision"] and "image N/A" in g3["detail"] and not g3["dropped"]
    na = {x["probe"]: x for x in g3["sub"] if x["ok"] is None}
    assert set(na) == {"G3.image", "G3.tool_result-image", "G3.tool_result-control"}
    assert all(x["outcome"] == NO_VISION for x in na.values())
    assert all(r["n_images"] == 0 for r in resp.requests)
    assert path.parent == tmp_path / "omlx" / "raw" and path.name.startswith("gate-t-")
    assert not (tmp_path / "omlx" / "capabilities.md").exists() and not path.with_suffix(".md").exists()
    assert s["t_start"] <= s["t_end"] and read(path)[-1]["t_end"] == s["t_end"]
    # with vision, the same model fails G3 on the image parts
    _, s2 = await run_probes(cfg, [("g3", {})], fx=fx)
    assert s2["items"]["G3"]["result"] == "FAIL" and "image FAIL" in s2["items"]["G3"]["detail"]


def test_gate_run_respects_runconfig_vision(built, mock, tmp_path):
    from agentbench.cli import main
    from conftest import cli_args

    srv, resp = mock("generic", images=False)
    assert main(["gate"] + cli_args(srv.url, built[0], tmp_path, "omlx", "omlx", "--no-vision")) == 0
    lines = read(next((tmp_path / "omlx" / "raw").glob("gate-*.jsonl")))
    assert lines[0]["mode"]["vision"] is False and lines[0]["mode"]["text_only_reason"] == "--no-vision"
    items = lines[-1]["items"]
    assert {k: v["result"] for k, v in items.items()} == {
        "G1": "PASS", "G2": "N/A", "G2b": "N/A", "G3": "PASS", "G4": "PASS", "G5": "INFO"}
    assert items["G2"]["rejected"] is False  # run-all's automatic text-only switch stays off
    assert all(r["n_images"] == 0 for r in resp.requests)
    assert "| G2 |" in (tmp_path / "omlx" / "capabilities.md").read_text()


async def test_unknown_probe_or_g5_prompt_is_refused(built, fx, mock, tmp_path):
    srv, _ = mock()
    cfg = cfg_for(srv.url, built[0], tmp_path)
    with pytest.raises(ValueError):
        await run_probes(cfg, [("g9", {})], fx=fx)
    assert not (tmp_path / "omlx").exists()
    with pytest.raises(ValueError):
        await run_probes(cfg, [("g5", {"prompt": "helper1"})], fx=fx)


async def test_g5_with_the_m3_prompt_primes_the_m3_cell(built, fx, mock, tmp_path):
    srv, resp = mock()
    cfg = cfg_for(srv.url, built[0], tmp_path, quick=True)  # quick: the 2,000-token prompts
    nonce = "123456789012"
    _, s = await run_probes(cfg, [("g5", {"prompt": "m3-off", "nonce": nonce})], fx=fx)
    g5 = s["items"]["G5"]
    assert g5["prompt"] == "m3-off-2000" and g5["nonce"] == nonce and g5["metrics"]["ok"]
    assert [p["probe"] for p in s["probes"]] == ["G5.m3-off-2000"]
    g5_req = resp.requests[-1]
    opts = SuiteOpts(spec_label="off", contexts=[2000], caps=[50], trials=2, no_thinking_bucket=True,
                     reuse_nonce=nonce)
    path, m3 = await run_suite(cfg, opts, "m3", fx)
    lines = read(path)
    assert not requests(lines, phase="warmup")
    trials = requests(lines, phase="trial")
    assert [t["trial"] for t in trials] == [0, 1] and trials[0]["warmup"] == "reused"
    assert all(t["nonce"] == nonce for t in trials)
    cell = m3["cells"]["spec=off thinking=off ctx=2000 cap=50"]
    assert cell["warmup"]["mode"] == "reused" and cell["warmup"]["trial0_cached_tokens"] >= 0.9 * cell["warmup"]["trial0_prompt_tokens"]
    # same request path as M3: identical body apart from max_tokens
    m3_req = resp.requests[-2]
    for k in ("keys", "chat_template_kwargs", "reasoning_effort", "system", "has_tools", "sampling"):
        assert g5_req[k] == m3_req[k], k
    # a nonce nobody primed: trial 0 misses, becomes the warm-up, and the trials follow it
    path, m3 = await run_suite(cfg, SuiteOpts(spec_label="off", contexts=[2000], caps=[50], trials=2,
                                              no_thinking_bucket=True, reuse_nonce="987654321098"), "m3", fx)
    lines = read(path)
    warm = requests(lines, phase="warmup")
    assert len(warm) == 1 and warm[0]["warmup"] == "fallback" and warm[0]["record"]["usage_raw"]
    assert [t["trial"] for t in requests(lines, phase="trial")] == [0, 1]
    w = m3["cells"]["spec=off thinking=off ctx=2000 cap=50"]["warmup"]
    assert w["mode"] == "fallback" and w["trial0_cached_tokens"] < 0.9 * w["trial0_prompt_tokens"]


async def test_g5_default_is_the_cold_prompt(built, fx, mock, tmp_path):
    srv, resp = mock()
    _, s = await run_probes(cfg_for(srv.url, built[0], tmp_path, quick=True), [("g5", {})], fx=fx)
    assert s["items"]["G5"]["prompt"] == "cold-2000" and s["probes"][0]["probe"] == "G5.cold-2000"
    assert resp.requests[-1]["reasoning_effort"] == "none"


# ------------------------------------------------------------------------------ 2. cold-nonce rule
def test_system_and_tool_nonce_keep_exact_prefixes_within_a_run(fx, qwen):
    from agentbench.suites.m1 import trial_messages
    from agentbench.suites.m4 import step_messages

    cfg = RunConfig(profile=get_profile("generic"), base_url="http://127.0.0.1:1", cold_nonce="system+tool")
    nonce = "246813579024"
    tools = fx.tools(nonce=nonce)
    prev = None
    for k in range(0, 21):
        msgs = fx.resolve(step_messages(fx, cfg, k, system_nonce=nonce), text_only=True)
        text = qwen.render(msgs, tools)
        assert msgs[0]["content"].startswith(f"[run {nonce}]\n") and text.count(f"[run {nonce}]") == 2
        if prev is not None:
            assert text.startswith(prev), k
        prev = text
    a, _ = trial_messages(fx, cfg, 0, fx.m1["m1_images"], system_nonce=nonce)
    b, _ = trial_messages(fx, cfg, 0, fx.m1["m1_images"], system_nonce=ZERO12)
    assert a[0]["content"].startswith(f"[run {nonce}]\n")
    assert qwen.count(a, tools) == qwen.count(b, fx.tools(nonce=ZERO12))  # digits: one token each


async def test_system_first_template_needs_the_system_nonce(built, fx, mock, tmp_path):
    srv, _ = mock("generic", system_first=True)  # the cache sees the system message before the tools
    opts = SuiteOpts(trials=1, cold_trials=1)
    _, s = await run_suite(cfg_for(srv.url, built[0], tmp_path, cold_nonce="tool"), opts, "m1", fx)
    assert s["cold_nonce"] == "tool" and s["cold_checks"][0]["cached_tokens"] > 256  # the system prompt hit
    path, s = await run_suite(cfg_for(srv.url, built[0], tmp_path, cold_nonce="system+tool"), opts, "m1", fx)
    assert s["cold_checks"][0]["cached_tokens"] < 256 and s["warm"]["cached_tokens"]["median"] > 256
    lines = read(path)
    assert lines[0]["mode"]["cold_nonce"] == "system+tool"
    cold = requests(lines, phase="cold")[0]
    assert cold["nonce_at"] == "system+tool" and cold["record"]["error"] is None
    path, s = await run_suite(cfg_for(srv.url, built[0], tmp_path, cold_nonce="system+tool"),
                              SuiteOpts(steps=3, runs=2), "m4", fx)
    assert [c["cached_tokens"] < 256 for c in s["cold_checks"]] == [True, True]
    for r in (0, 1):
        by = {x["step"]: x["metrics"] for x in requests(read(path), run=r)}
        assert all(by[k]["cached_tokens"] >= by[k - 1]["prompt_tokens"] - 2 for k in (2, 3))


def test_unknown_cold_nonce_mode_is_refused(built, fx):
    with pytest.raises(SystemExit):
        Target(cfg_for("http://127.0.0.1:1", built[0], Path("/nonexistent"), cold_nonce="system"), fx)


# --------------------------------------------------------------------------------- 3. M3 110K, warm-up
def test_110k_prompt_is_built_outside_the_manifest(built, fx, qwen):
    root, m = built
    assert EXTRA_PROMPTS == [("m3-off", 110_000)] and m["extra_prompts"] == {"m3-off-110000": 110_000}
    rel = "generated/prompts/m3-off-110000.json"
    assert (root / rel).exists() and rel not in m["files"] and "110000" not in m["sizes"]["m3_off"]
    assert "extra_prompts" not in json.loads((root / "manifest.json").read_text())
    assert fx.prompt("m3-off-110000")["outside_manifest"]
    assert qwen.count(fx.prompt_messages("m3-off-110000", nonce=None), None) == 110_000
    assert qwen.count(fx.prompt_messages("m3-off-110000", nonce="555555555555"), None) == 110_000


def test_on_the_fly_prompts_are_cached_under_generated(built, tmp_path, qwen):
    for rel in COMMITTED + ["manifest.json"]:
        shutil.copy(built[0] / rel, tmp_path / rel)
    (tmp_path / "generated").mkdir()
    a = Fixtures(tmp_path).prompt("m3-off-3000")
    path = tmp_path / "generated" / "prompts" / "m3-off-3000.json"
    assert path.exists() and a["outside_manifest"] and a["tokens"] == 3000
    assert Fixtures(tmp_path).prompt("m3-off-3000") == a == json.loads(path.read_text())
    assert not list(path.parent.glob("*.tmp"))


async def test_m3_single_cell_warmup_cap(built, fx, mock, tmp_path):
    srv, resp = mock()
    cfg = cfg_for(srv.url, built[0], tmp_path)
    one_cell = dict(spec_label="off", contexts=[2000], caps=[150], trials=1, no_thinking_bucket=True)
    await run_suite(cfg, SuiteOpts(**one_cell, warmup_max_tokens=16), "m3", fx)
    assert [r["max_tokens"] for r in resp.requests[-2:]] == [16, 150]
    await run_suite(cfg, SuiteOpts(**one_cell), "m3", fx)
    assert [r["max_tokens"] for r in resp.requests[-2:]] == [150, 150]  # c15a0a8: min(caps)


def test_reuse_nonce_validation():
    base = dict(spec_label="off", contexts=[64000])
    assert validate(["m3"], SuiteOpts(**base, reuse_nonce="123456789012")) == []
    assert validate(["m3"], SuiteOpts(spec_label="off", contexts=[16000, 32000], reuse_nonce="123456789012"))
    assert validate(["m3"], SuiteOpts(**base, reuse_nonce="123456789012", bust_cache=True))
    assert validate(["m3"], SuiteOpts(**base, reuse_nonce="12ab"))
    assert validate(["m3"], SuiteOpts(**base, warmup_max_tokens=0))


# ------------------------------------------------------------------------------ 4. thinking effort
async def test_think_effort_none_sends_enable_thinking_only(built, fx, mock, tmp_path):
    srv, resp = mock()
    cfg = cfg_for(srv.url, built[0], tmp_path)
    think_cell = dict(spec_label="on", contexts=[], think_contexts=[2000], think_caps=[100], trials=1)
    _, s = await run_suite(cfg, SuiteOpts(**think_cell, think_effort=None), "m3", fx)
    sent = resp.requests[-2:]
    assert all(r["chat_template_kwargs"] == {"enable_thinking": True} and "reasoning_effort" not in r["keys"]
               for r in sent)
    assert s["think_effort"] is None and list(s["cells"]) == ["spec=on thinking=on ctx=2000 cap=100"]
    await run_suite(cfg, SuiteOpts(**think_cell), "m3", fx)  # c15a0a8: xhigh, top level and in the kwargs
    assert all(r["reasoning_effort"] == "xhigh" and r["chat_template_kwargs"] == {
        "enable_thinking": True, "reasoning_effort": "xhigh"} for r in resp.requests[-2:])


def test_thinking_fields_omit_effort_per_style(built, fx):
    def fields(style: str, **kw):
        cfg = cfg_for("http://127.0.0.1:1", built[0], Path("/nonexistent"), thinking_fields=style)
        return Target(cfg, fx).thinking_fields(True, None, True, **kw)

    assert fields("hermes", omit_effort=True) == fields("kwargs", omit_effort=True) == {
        "chat_template_kwargs": {"enable_thinking": True}}
    assert fields("effort", omit_effort=True) == fields("none", omit_effort=True) == {}
    assert fields("hermes") == {"reasoning_effort": "medium",
                                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}


# --------------------------------------------------------------------------- 5. Anthropic sampling
async def test_anthropic_sampling_is_merged_only_when_asked(built, fx, mock, tmp_path):
    srv, resp = mock()
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    extra = {**sampling, "chat_template_kwargs": {"enable_thinking": False}}
    await run_probes(cfg_for(srv.url, built[0], tmp_path, extra_body=extra), [("g3", {"vision": False})], fx=fx)
    off = [r for r in resp.requests if r["path"].startswith("/v1/messages")]
    assert off and all(r["sampling"] == {} for r in off)
    n = len(resp.requests)
    await run_probes(cfg_for(srv.url, built[0], tmp_path, extra_body=extra, anthropic_sampling=True),
                     [("g3", {"vision": False})], fx=fx)
    on = resp.requests[n:]
    assert on and all(r["path"].startswith("/v1/messages") and r["sampling"] == sampling for r in on)
    assert all("chat_template_kwargs" not in r["keys"] and r["model"] == "mock" for r in on)


# ---------------------------------------------------------------------------- 6. model tokenizer
def test_model_tokenizer_loader(tiny_tokenizer, tmp_path):
    from agentbench.tokenizer import load_model_prompt

    p, err = load_model_prompt(tiny_tokenizer)
    assert err is None and p.pre == {}
    msgs = [{"role": "user", "content": "hello hello world"}]
    assert p.render(msgs) == "<s>user: hello hello world\nassistant:"
    assert p.count_text("hello world") == 2 and p.count(msgs) == p.count_text(p.render(msgs))
    # chat_template.jinja wins over tokenizer_config.json, as in transformers
    d = tmp_path / "jinja"
    shutil.copytree(tiny_tokenizer, d)
    (d / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    assert load_model_prompt(d)[0].render(msgs) == "hello hello world"
    for broken in (tmp_path / "missing", tmp_path):  # no directory; no tokenizer.json
        p, err = load_model_prompt(broken)
        assert p is None and "tokenizer.json" in err
    d2 = tmp_path / "no-template"
    d2.mkdir()
    shutil.copy(tiny_tokenizer / "tokenizer.json", d2)
    p, err = load_model_prompt(d2)
    assert p is None and "chat_template" in err


def test_qwen_dir_as_model_tokenizer_matches_the_default(qwen):
    from agentbench.tokenizer import load_model_prompt

    p, err = load_model_prompt(qwen.snap)
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Read docs/setup.md."}]
    assert err is None and p.render(msgs, [], enable_thinking=False) == qwen.render(msgs, [], enable_thinking=False)


def test_nanbeige_tokenizer_loads_if_present():
    d = Path.home() / "Developer/mlsys/models/oq/Nanbeige4.3-33B-oQ5"
    if not (d / "tokenizer.json").exists():
        pytest.skip("Nanbeige4.3-33B-oQ5 tokenizer not present")
    from agentbench.tokenizer import load_model_prompt

    p, err = load_model_prompt(d)  # read only: tokenizer.json and tokenizer_config.json
    tools = [{"type": "function", "function": {"name": "fs_read_file", "description": "Read a file",
                                               "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    text = p.render([{"role": "system", "content": "SYSTEM-FIRST"}, {"role": "user", "content": "hi"}], tools)
    assert err is None and text.index("SYSTEM-FIRST") < text.index("fs_read_file")  # why system+tool exists
    assert p.tok.encode(text, add_special_tokens=False).ids[:2] != [p.tok.token_to_id("<|im_start|>")] * 2


async def test_fallback_counts_use_the_model_tokenizer_or_stay_empty(built, fx, mock, tmp_path, tiny_tokenizer, qwen):
    srv, _ = mock("generic", report_usage=False)
    opts = SuiteOpts(spec_label="off", contexts=[2000], caps=[50], trials=1, no_thinking_bucket=True)

    async def trial(**kw) -> tuple[dict, dict]:
        path, _ = await run_suite(cfg_for(srv.url, built[0], tmp_path, **kw), opts, "m3", fx)
        lines = read(path)
        return lines[0], requests(lines, phase="trial")[0]

    hdr, t = await trial(tokenizer_dir=tiny_tokenizer)
    from agentbench.tokenizer import load_model_prompt
    tiny = load_model_prompt(tiny_tokenizer)[0]
    m = t["metrics"]
    assert hdr["mode"]["local_tokenizer"]["loaded"] and m["completion_tokens_source"] == "tokenizer"
    assert m["completion_tokens"] == tiny.count_text(t["record"]["content"]) != qwen.count_text(t["record"]["content"])
    assert t["prompt"]["local_tokens"] == tiny.count(fx.prompt_messages("m3-off-2000", None), None)
    hdr, t = await trial(tokenizer_dir=tmp_path / "missing")
    m = t["metrics"]
    assert hdr["mode"]["local_tokenizer"]["loaded"] is False and hdr["mode"]["local_tokenizer"]["error"]
    assert m["completion_tokens"] is None and m["completion_tokens_source"] is None and m["decode_tps"] is None
    assert "local_tokens" not in t["prompt"] and "unavailable" in t["prompt"]["local_tokens_error"]
    hdr, t = await trial()  # c15a0a8: Qwen3.8
    assert hdr["mode"]["local_tokenizer"]["dir"] is None
    assert t["metrics"]["completion_tokens"] == qwen.count_text(t["record"]["content"])


# ------------------------------------------------------------------------------- 7. M4 headline
async def test_m4_cut_steps_are_latency_samples_and_feed_the_headline(built, fx, mock, tmp_path):
    srv, resp = mock("generic", default_call=SHORT_CALL, cut_tool_calls=True)
    resp.answers[fx.m4["steps"][1]["caption"]] = [LONG_CALL]  # step 2 is cut by the 20-token cap
    path, s = await run_suite(cfg_for(srv.url, built[0], tmp_path), SuiteOpts(steps=3, runs=1, m4_max_tokens=20), "m4", fx)
    lines = requests(read(path))
    by = {x["step"]: x for x in lines}
    cut = by[2]
    assert not cut["metrics"]["ok"] and "incomplete_tool_call" in cut["metrics"]["error"]
    derived = {k: cut["derived"][k] for k in ("latency_sample", "cap_hit", "decode_s", "completion_tokens",
                                             "completion_tokens_source")}
    assert derived == {"latency_sample": True, "cap_hit": "incomplete_tool_call",
                       "decode_s": cut["metrics"]["decode_s"], "completion_tokens": cut["metrics"]["completion_tokens"],
                       "completion_tokens_source": "tokenizer"}
    # the mock opens every step with the call and reports no server timing: kept, but marked
    assert cut["derived"]["direct_tool_call"] and cut["derived"]["ttft_source"] == "client-incl-decode"
    assert cut["metrics"]["ttft_s"] is not None and cut["metrics"]["prompt_tokens"] is None
    assert all(by[k]["derived"]["cap_hit"] is None and by[k]["metrics"]["completion_tokens_source"] == "usage"
               for k in (0, 1, 3))
    r = s["per_run"][0]
    assert r["complete"] and r["n_steps"] == 4 and r["latency_samples"] == 4 and r["failed_steps"] == []
    assert r["cap_hits"] == s["cap_hits"] == 1 and r["cap_hits_by_kind"] == {"length": 0, "incomplete_tool_call": 1}
    assert r["ttft_sum_s"] == pytest.approx(sum(x["metrics"]["ttft_s"] for x in lines))
    usage = [by[k]["metrics"] for k in (0, 1, 3)]
    assert r["usage_steps"] == 3 and r["usage_completion_tokens"] == sum(m["completion_tokens"] for m in usage)
    assert r["usage_decode_s"] == pytest.approx(sum(m["decode_s"] for m in usage))
    rate = r["usage_completion_tokens"] / r["usage_decode_s"]
    assert r["decode_rate_tps"] == pytest.approx(rate)
    assert r["headline_s"] == pytest.approx(r["ttft_sum_s"] + 4 * 100 / rate)
    assert s["ttft_sum_s"] == [r["ttft_sum_s"]] and s["headline_s"] == [r["headline_s"]]
    assert s["errors"] == 1 and s["invalid_steps"] == 0 and s["per_step"]["2"]["n"] == 1  # the cut step counts
    assert s["per_step"]["2"]["cap_hits"] == 1 and s["fixed_tokens_per_step"] == 100
    assert s["t_start"] <= cut["record"]["t_send_wall"] <= s["t_end"]


async def test_m4_length_cap_hits_and_a_failed_step(built, fx, mock, tmp_path):
    srv, resp = mock()  # text replies, cut by finish_reason=length
    path, s = await run_suite(cfg_for(srv.url, built[0], tmp_path), SuiteOpts(steps=2, runs=1, m4_max_tokens=10), "m4", fx)
    r = s["per_run"][0]
    assert r["cap_hits_by_kind"] == {"length": 3, "incomplete_tool_call": 0} and r["headline_s"] is not None
    # a step that fails before its first token is not a latency sample: no TTFT sum, no headline
    from agentbench.suites.m4 import run_stats

    metrics = [x["metrics"] | {"step": x["step"], "cap_hit": None, "latency_sample": True} for x in requests(read(path))]
    metrics[1] |= {"ok": False, "ttft_s": None, "latency_sample": False}
    st = run_stats(metrics)
    assert not st["complete"] and st["failed_steps"] == [1] and st["ttft_sum_s"] is None and st["headline_s"] is None
    # an ok stream that ended before any token (EOS at once) has no TTFT either; it must not crash the sum
    from agentbench.metrics import latency_sample

    empty = dict(requests(read(path))[2]["metrics"], ttft_s=None)
    assert not latency_sample(empty)
    metrics = [x["metrics"] | {"step": x["step"], "cap_hit": None} for x in requests(read(path))]
    metrics[2] = empty | {"step": 2, "cap_hit": None}
    for m in metrics:
        m["latency_sample"] = latency_sample(m)
    st = run_stats(metrics)
    assert st["failed_steps"] == [2] and st["ttft_sum_s"] is None and st["headline_s"] is None
    # the summary tables still render the new M4 lines
    from agentbench.summarize import summarize

    md = summarize(tmp_path / "omlx", "t")
    assert "M4" in md and "| 2 |" in md and "total e2e per run" in md


def test_cap_helpers():
    from agentbench.metrics import cap_hit, cap_truncated, latency_sample
    from agentbench.summarize import cap_truncated as reexported

    err = "stream_error: {'message': 'Model output contains an unrecoverable tool call.', 'code': 'incomplete_tool_call'}"
    cut = {"ok": False, "error": err, "ttft_s": 3.1, "e2e_s": 8.5, "finish_reason": None}
    early = dict(cut, ttft_s=None, e2e_s=0.28)  # M1, 2026-10-18: failed before its first token
    assert cap_truncated(cut) and not cap_truncated(early) and reexported is cap_truncated
    assert latency_sample(cut) and not latency_sample(early)
    assert cap_hit(cut) == "incomplete_tool_call" and cap_hit(early) is None
    assert cap_hit({"ok": True, "finish_reason": "length", "error": None}) == "length"
    assert cap_hit({"ok": True, "finish_reason": "tool_calls", "error": None}) is None


# ------------------------------------------------------------------------- 8. single units in-process
async def test_run_suite_runs_single_units_with_their_window(built, fx, mock, tmp_path):
    srv, _ = mock()
    cfg = cfg_for(srv.url, built[0], tmp_path, tag="m3-16k")
    t0 = time.time()
    path, s = await run_suite(cfg, SuiteOpts(spec_label="off", contexts=[2000], caps=[150], no_thinking_bucket=True),
                              "m3", fx)
    lines = read(path)
    assert path.parent == tmp_path / "omlx" / "raw" and path.name.startswith("m3-m3-16k-")
    assert len(requests(lines, phase="warmup")) == 1 and len(requests(lines, phase="trial")) == 3
    assert list(s["cells"]) == ["spec=off thinking=off ctx=2000 cap=150"]
    assert t0 <= s["t_start"] <= min(x["record"]["t_send_wall"] for x in requests(lines)) and s["t_end"] <= time.time()
    assert lines[-1]["type"] == "summary" and lines[-1]["t_start"] == s["t_start"]
    _, s = await run_suite(cfg, SuiteOpts(spec_label="on", contexts=[], think_contexts=[2000], think_caps=[1000],
                                          trials=1, warmup_max_tokens=16), "m3", fx)
    assert list(s["cells"]) == ["spec=on thinking=on ctx=2000 cap=1000"] and s["t_start"] < s["t_end"]
    cfg.unverified = True
    path, s = await run_suite(cfg, SuiteOpts(steps=2, runs=1), "m4", fx)
    assert path.parent == tmp_path / "omlx" / "raw" / "_unverified"
    assert s["runs"] == 1 and len(s["per_run"]) == 1 and s["t_start"] < s["t_end"]
    with pytest.raises(ValueError):
        await run_suite(cfg, SuiteOpts(), "m3", fx)  # no spec label


def test_cli_plumbs_the_new_options(built, mock, tmp_path, tiny_tokenizer):
    from agentbench.cli import main
    from conftest import cli_args

    srv, resp = mock()
    args = ["--cold-nonce", "system+tool", "--tokenizer-dir", str(tiny_tokenizer), "--anthropic-sampling",
            "--extra-body", '{"top_k": 20}', "--spec-label", "off", "--contexts", "2000", "--caps", "50",
            "--no-thinking-bucket", "--warmup-max-tokens", "16", "--trials", "1"]
    assert main(["m3"] + cli_args(srv.url, built[0], tmp_path, "omlx", "omlx", *args)) == 0
    lines = read(next((tmp_path / "omlx" / "raw").glob("m3-*.jsonl")))
    mode = lines[0]["mode"]
    assert mode["cold_nonce"] == "system+tool" and mode["anthropic_sampling"] and mode["vision"]
    assert mode["local_tokenizer"]["loaded"] and mode["local_tokenizer"]["kind"] == "model"
    assert [r["max_tokens"] for r in resp.requests[-2:]] == [16, 50]
    assert main(["m3"] + cli_args(srv.url, built[0], tmp_path, "omlx", "omlx", "--spec-label", "off", "--contexts",
                                  "2000", "--caps", "50", "--no-thinking-bucket", "--trials", "1",
                                  "--reuse-nonce", "123456789012")) == 0
    assert resp.requests[-1]["system"].startswith("[run 123456789012]")
    assert main(["m3"] + cli_args(srv.url, built[0], tmp_path, "omlx", "omlx", "--spec-label", "on", "--contexts", "",
                                  "--think-contexts", "2000", "--think-caps", "100", "--trials", "1",
                                  "--no-think-effort")) == 0
    assert resp.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in resp.requests[-1]["keys"]
    assert main(["m3"] + cli_args(srv.url, built[0], tmp_path, "omlx", "omlx", "--spec-label", "off", "--contexts",
                                  "2000,16000", "--reuse-nonce", "123456789012")) == 2  # validated first


async def test_m4_direct_tool_calls_use_the_servers_ttft_and_generation_time(built, fx, mock, tmp_path):
    """oMLX streams a tool call as one delta after generating all of it: the client's first visible
    chunk then comes after the call's decode. With the server's timing in usage, TTFT and R use it."""
    srv, _ = mock("generic", default_call=SHORT_CALL, tool_call_style="single", server_timing=True,
                  token_delay=0.01)
    path, s = await run_suite(cfg_for(srv.url, built[0], tmp_path), SuiteOpts(steps=2, runs=1, m4_max_tokens=60), "m4", fx)
    lines = requests(read(path))
    for x in lines:
        m = x["metrics"]
        assert m["first_kind"] == "tool_call" and m["n_token_chunks"] == 1 and m["decode_s"] == 0
        assert m["ttft_source"] == "server" and m["ttft_eff_s"] == m["server_ttft_s"] < m["ttft_s"]
        assert x["derived"]["decode_time_s"] == m["server_generation_s"] > 0
    r = s["per_run"][0]
    assert r["direct_calls"] == [0, 1, 2] and r["ttft_includes_decode"] == []
    assert r["ttft_sum_s"] == pytest.approx(sum(x["metrics"]["server_ttft_s"] for x in lines))
    assert r["usage_steps"] == 3 and r["decode_rate_tps"] == pytest.approx(
        sum(x["metrics"]["completion_tokens"] for x in lines) / sum(x["metrics"]["server_generation_s"] for x in lines))
    assert r["headline_s"] is not None


async def test_m4_direct_tool_calls_without_server_timing_are_flagged(built, fx, mock, tmp_path):
    srv, _ = mock("generic", default_call=SHORT_CALL, tool_call_style="single", token_delay=0.01)
    path, s = await run_suite(cfg_for(srv.url, built[0], tmp_path), SuiteOpts(steps=1, runs=1, m4_max_tokens=60), "m4", fx)
    r = s["per_run"][0]
    assert r["ttft_includes_decode"] == [0, 1] and r["decode_rate_tps"] is None and r["headline_s"] is None
    assert all(x["metrics"]["ttft_source"] == "client-incl-decode" for x in requests(read(path)))
