"""M4: 20-step agent trajectory with canned history.

Each run starts with prefix + task (step 0), then step k appends canned assistant tool
call k, tool result k (text) and screenshot k, and asks for the next step (~150 output
tokens, discarded). Step k's prompt is an exact prefix of step k+1's, so a server that
reuses screenshot-bearing history keeps TTFT flat; otherwise it grows with k. Each run
gets a fresh nonce in the first tool's description (`--no-bust` to disable), so runs do
not hit each other's cache entries; with `RunConfig.cold_nonce="system+tool"` the same nonce
also starts the system message of every step of the run (templates that render the system
prompt first), so the steps of a run still share their prefix. Per step: TTFT, e2e, cached
and prompt tokens.

agent-model-eval additions (harness/CHANGES.md): every step line has `derived` with its latency
validity, cap hit (`length` or a cap-truncated `incomplete_tool_call` stream, which is still a
latency sample) and decode time; per run, the summary's `per_run` gives the inputs of the
headline `ttft_sum_s + n_steps * FIXED_TOKENS / R` (R = completion tokens / decode time over
the steps with server usage) and step 0's cached tokens (`cold_checks`). Runs with cut steps
are kept.
"""

from __future__ import annotations

from typing import Any

from ..fixtures.store import Fixtures, new_digits
from ..metrics import cap_hit, decode_time, direct_tool_call, latency_sample, mmm, ttft
from .common import RunConfig, Target, Writer
from .opts import SuiteOpts, pick

FIXED_TOKENS = 100  # headline output length per step (DESIGN §5.1), counted at the run's own decode rate


def step_messages(fx: Fixtures, cfg: RunConfig, k: int, system_nonce: str | None = None) -> list[dict[str, Any]]:
    """Prompt of step k (0 = prefix + task), with image refs unresolved."""
    msgs = [fx.agent_system(system_nonce), {"role": "user", "content": fx.m4["task"]}]
    for st in fx.m4["steps"][:k]:
        tool = dict(st["tool"])
        if cfg.screenshot_placement == "tool":
            tool["content"] = [{"type": "text", "text": tool["content"]}, {"type": "image_ref", "image": st["image"]}]
            msgs += [st["assistant"], tool]
        else:
            msgs += [st["assistant"], tool, fx.screenshot_message(st["image"], st["caption"])]
    return msgs


def step_derived(m: dict[str, Any]) -> dict[str, Any]:
    """Per-step facts the M4 headline needs, from one request's metrics."""
    t, src = ttft(m)
    return {"latency_sample": latency_sample(m), "cap_hit": cap_hit(m), "decode_s": m["decode_s"],
            "completion_tokens": m["completion_tokens"], "completion_tokens_source": m["completion_tokens_source"],
            "ttft_s": t, "ttft_source": src, "decode_time_s": decode_time(m), "direct_tool_call": direct_tool_call(m)}


async def trajectory(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts, run: int, steps: int, suite: str = "m4",
                     deadline: float | None = None) -> list[dict[str, Any]]:
    import time

    fx = t.fx
    nonce = None if o.no_bust else new_digits()
    tools = fx.tools(nonce=nonce)
    system_nonce = nonce if cfg.cold_nonce == "system+tool" else None
    out = []
    for k in range(0, steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        msgs = fx.resolve(step_messages(fx, cfg, k, system_nonce), text_only=cfg.text_only)
        oc = await t.chat(msgs, tools=tools, max_tokens=o.m4_max_tokens)
        line = oc.line(suite, t.count_tokens, phase="start" if k == 0 else "step", run=run, step=k,
                       cell={"step": k}, nonce=nonce, **({"nonce_at": cfg.cold_nonce} if nonce else {}))
        line["derived"] = step_derived(line["metrics"])
        w.write(line)
        out.append(line["metrics"] | {"step": k, "cap_hit": line["derived"]["cap_hit"],
                                      "latency_sample": line["derived"]["latency_sample"]})
    return out


def run_stats(run: list[dict[str, Any]], fixed_tokens: int = FIXED_TOKENS) -> dict[str, Any]:
    """One run's headline inputs. Cut steps (cap-truncated tool calls) count: TTFT and e2e are
    valid, they have no usage. R uses only steps with server usage, whose token counts come from
    the model's own tokenizer on the server. A step that opens with a tool call (`direct_calls`)
    uses the server's TTFT and generation time: oMLX streams such a call as one delta at its end,
    so the client's TTFT would include its decode and its client-side decode time is ~0
    (`ttft_includes_decode` lists the steps where the server reported no timing)."""
    valid = [m for m in run if m["latency_sample"]]
    usage = [m for m in valid if m["completion_tokens_source"] == "usage" and m["completion_tokens"]
             and decode_time(m)]
    tokens, decode_s = sum(m["completion_tokens"] for m in usage), sum(decode_time(m) for m in usage)
    rate = tokens / decode_s if tokens and decode_s else None
    complete = len(valid) == len(run) and bool(run)
    ttft_sum = sum(ttft(m)[0] for m in valid) if complete else None
    counted = [m["completion_tokens"] for m in valid if m["completion_tokens"] is not None]
    start = next((m for m in run if m["step"] == 0), None)
    return {"n_steps": len(run), "latency_samples": len(valid), "complete": complete,
            "failed_steps": [m["step"] for m in run if not m["latency_sample"]],
            "ttft_sum_s": ttft_sum, "e2e_total_s": sum(m["e2e_s"] or 0 for m in run),
            "cap_hits": sum(1 for m in run if m["cap_hit"]),
            "cap_hits_by_kind": {k: sum(1 for m in run if m["cap_hit"] == k) for k in ("length", "incomplete_tool_call")},
            "mean_completion_tokens": sum(counted) / len(counted) if counted else None,
            "usage_steps": len(usage), "usage_completion_tokens": tokens, "usage_decode_s": decode_s,
            "decode_rate_tps": rate, "fixed_tokens_per_step": fixed_tokens,
            "headline_s": (ttft_sum + len(run) * fixed_tokens / rate) if ttft_sum is not None and rate else None,
            "start_cached_tokens": start["cached_tokens"] if start else None,
            "start_prompt_tokens": start["prompt_tokens"] if start else None,
            "direct_calls": [m["step"] for m in run if direct_tool_call(m)],
            "ttft_includes_decode": [m["step"] for m in valid if ttft(m)[1] == "client-incl-decode"]}


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
        ms = [m for run in all_runs for m in run if m["step"] == k and m["latency_sample"]]
        per_step[str(k)] = {"ttft_s": mmm([ttft(m)[0] for m in ms]), "e2e_s": mmm([m["e2e_s"] for m in ms]),
                            "cached_tokens": mmm([m["cached_tokens"] for m in ms]),
                            "prompt_tokens": mmm([m["prompt_tokens"] or m["prompt_tokens_local"] for m in ms]),
                            "n": len(ms), "cap_hits": sum(1 for run in all_runs for m in run
                                                          if m["step"] == k and m["cap_hit"])}
    ks = [k for k in range(1, steps + 1) if per_step[str(k)]["ttft_s"]["median"] is not None]
    stats = [run_stats(r) for r in all_runs]
    summary = {"type": "summary", "suite": "m4", "steps": steps, "runs": runs, "per_step": per_step,
               "ttft_slope_s_per_step": slope([float(k) for k in ks], [per_step[str(k)]["ttft_s"]["median"] for k in ks]),
               "run_total_e2e_s": [s["e2e_total_s"] for s in stats],
               "errors": sum(1 for run in all_runs for m in run if not m["ok"]),  # includes cut steps (c15a0a8)
               "invalid_steps": sum(1 for run in all_runs for m in run if not m["latency_sample"]),
               "cold_nonce": cfg.cold_nonce, "per_run": stats,
               "cap_hits": sum(s["cap_hits"] for s in stats), "step_requests": sum(s["n_steps"] for s in stats),
               "ttft_sum_s": [s["ttft_sum_s"] for s in stats], "headline_s": [s["headline_s"] for s in stats],
               "decode_rate_tps": [s["decode_rate_tps"] for s in stats], "fixed_tokens_per_step": FIXED_TOKENS,
               "headline_formula": "ttft_sum_s + n_steps * fixed_tokens_per_step / decode_rate_tps",
               "cold_checks": [{"run": i, "cached_tokens": s["start_cached_tokens"],
                                "prompt_tokens": s["start_prompt_tokens"]} for i, s in enumerate(stats)]}
    w.write(summary)
    print(f"M4 {steps} steps x {runs} runs: TTFT slope {summary['ttft_slope_s_per_step']} s/step, "
          f"run totals {['%.1f' % x for x in summary['run_total_e2e_s']]} s, cap hits {summary['cap_hits']}/"
          f"{summary['step_requests']}")
    return summary
