"""M4: 20-step agent trajectory with canned history.

Each run starts with prefix + task (step 0), then step k appends canned assistant tool
call k, tool result k (text) and screenshot k, and asks for the next step (~150 output
tokens, discarded). Step k's prompt is an exact prefix of step k+1's, so a server that
reuses screenshot-bearing history keeps TTFT flat; otherwise it grows with k. Each run
gets a fresh nonce in the first tool's description (`--no-bust` to disable), so runs do
not hit each other's cache entries. Per step: TTFT, e2e, cached and prompt tokens.
"""

from __future__ import annotations

from typing import Any

from ..fixtures.store import Fixtures, new_digits
from ..metrics import mmm
from .common import RunConfig, Target, Writer
from .opts import SuiteOpts, pick


def step_messages(fx: Fixtures, cfg: RunConfig, k: int) -> list[dict[str, Any]]:
    """Prompt of step k (0 = prefix + task), with image refs unresolved."""
    msgs = [fx.agent_system(), {"role": "user", "content": fx.m4["task"]}]
    for st in fx.m4["steps"][:k]:
        tool = dict(st["tool"])
        if cfg.screenshot_placement == "tool":
            tool["content"] = [{"type": "text", "text": tool["content"]}, {"type": "image_ref", "image": st["image"]}]
            msgs += [st["assistant"], tool]
        else:
            msgs += [st["assistant"], tool, fx.screenshot_message(st["image"], st["caption"])]
    return msgs


async def trajectory(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts, run: int, steps: int, suite: str = "m4",
                     deadline: float | None = None) -> list[dict[str, Any]]:
    import time

    fx = t.fx
    nonce = None if o.no_bust else new_digits()
    tools = fx.tools(nonce=nonce)
    out = []
    for k in range(0, steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        msgs = fx.resolve(step_messages(fx, cfg, k), text_only=cfg.text_only)
        oc = await t.chat(msgs, tools=tools, max_tokens=o.m4_max_tokens)
        line = oc.line(suite, t.count_tokens, phase="start" if k == 0 else "step", run=run, step=k,
                       cell={"step": k}, nonce=nonce)
        w.write(line)
        out.append(line["metrics"] | {"step": k})
    return out


def slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    steps = min(pick(o.steps, cfg.quick, 3, 20), len(t.fx.m4["steps"]))
    runs = pick(o.runs, cfg.quick, 1, 3)
    all_runs = [await trajectory(cfg, t, w, o, r, steps) for r in range(runs)]
    per_step = {}
    for k in range(0, steps + 1):
        ms = [m for run in all_runs for m in run if m["step"] == k and m["ok"]]
        per_step[str(k)] = {"ttft_s": mmm([m["ttft_s"] for m in ms]), "e2e_s": mmm([m["e2e_s"] for m in ms]),
                            "cached_tokens": mmm([m["cached_tokens"] for m in ms]),
                            "prompt_tokens": mmm([m["prompt_tokens"] or m["prompt_tokens_local"] for m in ms]),
                            "n": len(ms)}
    ks = [k for k in range(1, steps + 1) if per_step[str(k)]["ttft_s"]["median"] is not None]
    summary = {"type": "summary", "suite": "m4", "steps": steps, "runs": runs, "per_step": per_step,
               "ttft_slope_s_per_step": slope([float(k) for k in ks], [per_step[str(k)]["ttft_s"]["median"] for k in ks]),
               "run_total_e2e_s": [sum(m["e2e_s"] or 0 for m in run) for run in all_runs],
               "errors": sum(1 for run in all_runs for m in run if not m["ok"])}
    w.write(summary)
    print(f"M4 {steps} steps x {runs} runs: TTFT slope {summary['ttft_slope_s_per_step']} s/step, "
          f"run totals {['%.1f' % x for x in summary['run_total_e2e_s']]} s")
    return summary
