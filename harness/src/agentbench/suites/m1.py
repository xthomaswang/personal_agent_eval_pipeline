"""M1: cached-prefix TTFT.

Warm-up = agent prefix (system + tools) + task, ending with a user message (mlx-lm only
checkpoints after a user message). Each trial = prefix + task + canned assistant tool
call + a unique tool result (fresh request_id digits) + a unique screenshot, so the
warm-up prompt is an exact token prefix of every trial. The cold comparison sends the
same shape with a nonce in the first tool's description, which busts every prefix cache.
TTFT follows METHODOLOGY (first content / reasoning / tool_calls chunk); `first_kind`
and e2e are recorded because some servers emit a tool call only when it is complete.
"""

from __future__ import annotations

from typing import Any

from ..fixtures.store import Fixtures, new_digits
from ..metrics import mmm
from .common import RunConfig, Target, Writer
from .opts import SuiteOpts, pick


def trial_messages(fx: Fixtures, cfg: RunConfig, i: int, images: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    img = images[i % len(images)]
    rid = new_digits()
    result = fx.tool_result(i, rid)
    msgs = [fx.agent_system(), {"role": "user", "content": fx.m1["m1_task"]}, fx.m1["m1_assistant"]]
    if cfg.screenshot_placement == "tool":
        msgs.append({"role": "tool", "tool_call_id": "call_m1_click",
                     "content": [{"type": "text", "text": result}, {"type": "image_ref", "image": img}]})
    else:
        msgs += [{"role": "tool", "tool_call_id": "call_m1_click", "content": result},
                 fx.screenshot_message(img, fx.m1["m1_caption"])]
    return fx.resolve(msgs, text_only=cfg.text_only), {"image": img, "request_id": rid}


def warm_messages(fx: Fixtures) -> list[dict[str, Any]]:
    return [fx.agent_system(), {"role": "user", "content": fx.m1["m1_task"]}]


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    fx = t.fx
    trials = pick(o.trials, cfg.quick, 2, 5)
    cold = pick(o.cold_trials, cfg.quick, 1, 3)
    tools = fx.tools()
    images = fx.m1["m1_images"]
    warm = warm_messages(fx)
    oc = await t.chat(warm, tools=tools, max_tokens=1)
    w.write(oc.line("m1", t.count_tokens, phase="warmup", cell={"phase": "warmup"}))
    res: dict[str, list[dict[str, Any]]] = {"warm": [], "cold": []}
    for i in range(trials):
        if o.rewarm and i:
            oc = await t.chat(warm, tools=tools, max_tokens=1)
            w.write(oc.line("m1", t.count_tokens, phase="rewarm", cell={"phase": "rewarm"}))
        msgs, meta = trial_messages(fx, cfg, i, images)
        oc = await t.chat(msgs, tools=tools, max_tokens=o.m1_max_tokens)
        line = oc.line("m1", t.count_tokens, phase="warm", trial=i, cell={"phase": "warm"}, **meta)
        w.write(line)
        res["warm"].append(line["metrics"])
    for j in range(cold):
        nonce = new_digits()
        msgs, meta = trial_messages(fx, cfg, trials + j, images)
        oc = await t.chat(msgs, tools=fx.tools(nonce=nonce), max_tokens=o.m1_max_tokens)
        line = oc.line("m1", t.count_tokens, phase="cold", trial=j, cell={"phase": "cold"}, nonce=nonce, **meta)
        w.write(line)
        res["cold"].append(line["metrics"])
    summary: dict[str, Any] = {"type": "summary", "suite": "m1"}
    for ph, ms in res.items():
        ok = [m for m in ms if m["ok"]]
        summary[ph] = {"ttft_s": mmm([m["ttft_s"] for m in ok]), "e2e_s": mmm([m["e2e_s"] for m in ok]),
                       "cached_tokens": mmm([m["cached_tokens"] for m in ok]),
                       "prompt_tokens": mmm([m["prompt_tokens"] for m in ok]),
                       "first_kind": sorted({m["first_kind"] for m in ok if m["first_kind"]}), "errors": len(ms) - len(ok)}
    wm, cm = summary["warm"]["ttft_s"]["median"], summary["cold"]["ttft_s"]["median"]
    summary["cold_over_warm_ttft"] = (cm / wm) if wm and cm else None
    w.write(summary)
    print(f"M1 warm TTFT median {wm} s, cold {cm} s, cached tokens (warm median) {summary['warm']['cached_tokens']['median']}")
    return summary
