import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from agentbench.fixtures.build import COMMITTED, ZERO12, check, nonce_line
from agentbench.fixtures.store import with_nonce_at_start


def test_agent_prefix_size_and_tools(built, fx, qwen):
    root, m = built
    s = m["sizes"]
    assert 14_500 <= s["prefix_tokens"] <= 15_500
    assert 35 <= len(fx.tools_openai) <= 45 and len(fx.tools_anthropic) == len(fx.tools_openai)
    for o, a in zip(fx.tools_openai, fx.tools_anthropic):
        assert o["function"]["name"] == a["name"] and o["function"]["parameters"] == a["input_schema"]
    full = qwen.render([fx.agent_system(), {"role": "user", "content": "x"}], fx.tools_openai)
    # tools render before the system prompt, so a tools nonce must go into the first tool's description
    assert full.index("<tools>") < full.index(fx.system_prompt[:60])
    assert qwen.count_text(full[: full.index("<|im_end|>") + len("<|im_end|>\n")]) == s["prefix_tokens"]


def test_tool_results_about_250_tokens(fx, qwen):
    pool = fx.m1["tool_results"]
    assert len(pool) >= 20 and len(set(pool)) == len(pool)
    for text in pool:
        assert 240 <= qwen.count_text(text.replace("{{REQUEST_ID}}", ZERO12)) <= 260
        assert text.startswith("request_id: {{REQUEST_ID}}")
    for st in fx.m4["steps"]:
        assert 240 <= qwen.count_text(st["tool"]["content"]) <= 260


def test_cold_and_m3_prompts_exact(fx, qwen):
    for name, target, thinking in [("cold-16000", 16000, False), ("cold-32000", 32000, False), ("cold-64000", 64000, False),
                                   ("m3-off-64000", 64000, False), ("m3-on-32000", 32000, True), ("helper1", 2000, False)]:
        msgs = fx.prompt_messages(name, nonce=None)
        assert msgs[0]["content"].startswith(nonce_line(ZERO12))
        assert qwen.count(msgs, None, enable_thinking=thinking) == target, name
        busted = fx.prompt_messages(name, nonce="739201846350")
        assert busted[0]["content"].startswith("[run 739201846350]")
        assert qwen.count(busted, None, enable_thinking=thinking) == target  # digits: one token each


def test_cold_prompt_is_varied(fx, qwen):
    ids = qwen.tok.encode(fx.prompt("cold-64000")["messages"][1]["content"], add_special_tokens=False).ids
    grams = [tuple(ids[i:i + 8]) for i in range(len(ids) - 7)]
    assert len(set(grams)) / len(grams) > 0.9


def test_nonce_placement_and_count_invariance(fx, qwen):
    msgs = [fx.agent_system(), {"role": "user", "content": fx.m1["m1_task"]}]
    t1, t2 = fx.tools(nonce="000000000000"), fx.tools(nonce="987654321098")
    assert t2[0]["function"]["description"].startswith("[run 987654321098] ")
    assert fx.tools(nonce="1" * 12, api="anthropic")[0]["description"].startswith("[run 111111111111] ")
    assert qwen.count(msgs, t1) == qwen.count(msgs, t2)
    rendered = qwen.render(msgs, t2)
    assert rendered.index("[run 987654321098]") < rendered.index(fx.system_prompt[:40])
    plain = with_nonce_at_start([{"role": "user", "content": "hello"}], "123456789012")
    assert plain[0]["content"].startswith("[run 123456789012]\n")


def test_images_distinct_and_text_equivalents(built, fx, qwen):
    root, m = built
    shas = set()
    for iid, meta in fx.images.items():
        img = Image.open(root / meta["png"])
        assert img.size == (1280, 1024)
        shas.add(hashlib.sha256((root / meta["png"]).read_bytes()).hexdigest())
        assert qwen.count_text(fx.image_text(iid)) == m["sizes"]["image_tokens"] == 1282
    assert len(shas) == len(fx.images) >= 40
    code = m["g2_code"]
    assert re.fullmatch(r"[A-Z0-9]{3}-[A-Z0-9]{3}", code) and fx.images["g2-verify"]["code"] == code


def test_text_only_has_same_token_count_as_image(fx, qwen):
    from agentbench.suites.m1 import trial_messages
    from agentbench.suites.common import RunConfig
    from agentbench.profiles import get_profile

    cfg = RunConfig(profile=get_profile("generic"), base_url="http://127.0.0.1:1")
    img_msgs, _ = trial_messages(fx, cfg, 0, fx.m1["m1_images"])
    cfg.text_only = True
    txt_msgs, _ = trial_messages(fx, cfg, 0, fx.m1["m1_images"])
    assert qwen.count(img_msgs, fx.tools_openai) == qwen.count(txt_msgs, fx.tools_openai)


def test_m1_warm_prompt_is_prefix_of_trial(fx, qwen):
    from agentbench.profiles import get_profile
    from agentbench.suites.common import RunConfig
    from agentbench.suites.m1 import trial_messages, warm_messages

    warm = warm_messages(fx)
    assert warm[-1]["role"] == "user"
    cfg = RunConfig(profile=get_profile("generic"), base_url="http://127.0.0.1:1")
    for i in range(3):
        trial, _ = trial_messages(fx, cfg, i, fx.m1["m1_images"])
        assert qwen.render(trial, fx.tools_openai).startswith(qwen.render(warm, fx.tools_openai))
    n_suffix = qwen.count(trial, fx.tools_openai) - qwen.count(warm, fx.tools_openai)
    assert 1_400 <= n_suffix <= 1_800  # one tool result plus one screenshot, about 1.5K new tokens


def test_m4_trajectory_shape_and_exact_prefixes(built, fx, qwen):
    from agentbench.profiles import get_profile
    from agentbench.suites.common import RunConfig
    from agentbench.suites.m4 import step_messages

    steps = fx.m4["steps"]
    assert len(steps) == 20 and len({s["image"] for s in steps}) == 20
    sizes = built[1]["sizes"]["m4"]["assistant_tokens"]
    assert 120 <= sizes["min"] and sizes["max"] <= 180 and 135 <= sizes["median"] <= 165
    for s in steps:
        a = s["assistant"]
        assert "reasoning_content" not in a and isinstance(a["tool_calls"][0]["function"]["arguments"], str)
        assert s["tool"]["tool_call_id"] == a["tool_calls"][0]["id"]
    cfg = RunConfig(profile=get_profile("generic"), base_url="http://127.0.0.1:1")
    prev = None
    for k in range(0, 21):
        text = qwen.render(fx.resolve(step_messages(fx, cfg, k), text_only=False), fx.tools_openai)
        if prev is not None:
            assert text.startswith(prev), k
        prev = text
    assert built[1]["sizes"]["m4"]["final_prompt_tokens"] < 60_000


def test_m7_suite_covers_categories_and_references_pass(fx):
    from agentbench.m7check import check_case

    cases = fx.m7["cases"]
    assert len(cases) >= 20
    cats = " ".join(c["category"] for c in cases)
    for need in ["single call", "multiple args", "nested objects", "arrays", "enums", "numbers", "booleans",
                 "Unicode / Chinese", "quotes and newlines", "long string", "two parallel calls",
                 "choosing between similar tools", "no call needed", "call after an image"]:
        assert need in cats, need
    for c in cases:
        calls = [{"name": x["name"], "arguments": json.dumps(x["arguments"]), "arguments_parsed": x["arguments"]}
                 for x in c["reference"]]
        assert check_case(c, calls, "Answer." if not calls else "", [], fx.schemas())["pass"], c["id"]


def test_manifest_hashes_and_no_absolute_paths(built):
    root, m = built
    for rel, meta in m["files"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == meta["sha256"]
    for rel in COMMITTED + ["manifest.json"]:
        assert "/Users/" not in (root / rel).read_text(encoding="utf-8"), rel
    assert all(not rel.startswith("generated/") for rel in COMMITTED)


def test_build_is_deterministic(built):
    root, _ = built
    assert check(root, log=lambda s: None) == []


def test_committed_fixtures_match_the_generator(built):
    repo = Path(__file__).resolve().parents[2] / "fixtures" / "manifest.json"
    if not repo.exists():
        return
    committed = json.loads(repo.read_text())
    for rel, meta in built[1]["files"].items():
        assert committed["files"].get(rel, {}).get("sha256") == meta["sha256"], f"stale committed fixture: {rel}"
