"""M3: short structured decode at long context, labelled spec on/off.

Outputs of 50 / 150 / 300 tokens with thinking off at 16K / 32K / 64K, and 1000 / 2000
tokens with thinking on at 16K / 32K. The prompt asks for a 150-element JSON array, so
the output reliably exceeds every cap; cells whose finish_reason is not `length` are
flagged. The JSON is plain `content` and no `tools` are sent, because servers that hold
back tool-call markup until a call is complete would make chunk timing meaningless.

Thinking on uses effort `xhigh` (Qwen3.8's published computer-use scores use high
effort), sent as top-level `reasoning_effort` and inside `chat_template_kwargs` (mlx-lm
ignores the top-level field), so every server renders the same template. With
`think_effort=None` only `chat_template_kwargs.enable_thinking=true` is sent and every template
renders its own default effort (agent-model-eval: one setting across model families).
By default the prompt cache is allowed (the warm-up per context primes it; decode speed
is what is measured); `--bust-cache` adds a fresh nonce per trial.

agent-model-eval additions (harness/CHANGES.md): `warmup_max_tokens` caps the warm-up (default:
the smallest cap); `reuse_nonce` sends the thinking-off cell with a given nonce and no warm-up,
for a prefix the caller already primed (G5's 64K prompt). If trial 0 then reports a cached
prefix (>= REUSE_MIN_SHARE of its prompt), the trials continue; otherwise that request is
relabelled as the warm-up and the trials start after it. The cell's `warmup` says which.
The 110,000-token thinking-off prompt is outside the fixture manifest (fixtures.build.EXTRA_PROMPTS).
"""

from __future__ import annotations

from typing import Any

from ..fixtures.store import new_digits
from ..metrics import mmm
from .common import RunConfig, Target, Writer
from .opts import SuiteOpts, pick

REUSE_MIN_SHARE = 0.9  # trial 0 reused the prefix if the server reports at least this share as cached


def _reused(m: dict[str, Any]) -> bool:
    prompt = m["prompt_tokens"] or m["prompt_tokens_local"]
    return bool(m["ok"] and m["cached_tokens"] is not None and prompt and m["cached_tokens"] >= REUSE_MIN_SHARE * prompt)


async def _bucket(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts, thinking: bool, contexts: list[int],
                  caps: list[int], trials: int, spec: str, cells: dict[str, Any]) -> None:
    fx = t.fx
    kind = "m3-on" if thinking else "m3-off"
    reuse = None if thinking else o.reuse_nonce
    warm_cap = o.warmup_max_tokens if o.warmup_max_tokens is not None else min(caps)
    for ctx in contexts:
        blocked = t.profile.thinking_blocked(thinking)
        if blocked:
            for cap in caps:
                key = f"spec={spec} thinking={'on' if thinking else 'off'} ctx={ctx} cap={cap}"
                cells[key] = {"na": blocked}
                w.write({"type": "skip", "suite": "m3", "cell": {"spec": spec, "thinking": thinking, "ctx": ctx, "cap": cap},
                         "reason": blocked})
            continue
        name = f"{kind}-{ctx}"
        kw = dict(thinking=thinking, effort=o.think_effort if thinking else None, effort_in_kwargs=thinking,
                  omit_effort=thinking and o.think_effort is None)

        def nonce_for() -> str | None:
            return reuse or (new_digits() if o.bust_cache else None)

        warm: dict[str, Any] = {"mode": "normal", "max_tokens": warm_cap}
        if reuse is None:
            nonce = nonce_for()
            oc = await t.chat(fx.prompt_messages(name, nonce=nonce), max_tokens=warm_cap, **kw)
            w.write(oc.line("m3", t.count_tokens, phase="warmup", warmup="normal", nonce=nonce,
                            cell={"spec": spec, "thinking": thinking, "ctx": ctx, "cap": warm_cap, "phase": "warmup"}))
        check_reuse = reuse is not None
        for cap in caps:
            ms = []
            i = 0
            while i < trials:
                nonce = nonce_for()
                oc = await t.chat(fx.prompt_messages(name, nonce=nonce), max_tokens=cap, **kw)
                cell = {"spec": spec, "thinking": thinking, "ctx": ctx, "cap": cap}
                if check_reuse:
                    check_reuse = False
                    m = oc.metrics()
                    hit = _reused(m)
                    warm = {"mode": "reused" if hit else "fallback", "max_tokens": None if hit else cap,
                            "trial0_cached_tokens": m["cached_tokens"],
                            "trial0_prompt_tokens": m["prompt_tokens"] or m["prompt_tokens_local"]}
                    if not hit:  # this request primed the prefix: it is the warm-up, and the trials follow it
                        w.write(oc.line("m3", t.count_tokens, phase="warmup", warmup="fallback", nonce=nonce,
                                        cell=dict(cell, phase="warmup")))
                        continue
                line = oc.line("m3", t.count_tokens, phase="trial", trial=i, cell=cell, nonce=nonce,
                               **({"warmup": warm["mode"]} if i == 0 and reuse else {}))
                m = line["metrics"]
                line["derived"] = {"hit_cap": bool(m["ok"] and (m["finish_reason"] == "length"
                                                               or (m["completion_tokens"] or 0) >= cap))}
                w.write(line)
                ms.append((m, line["derived"]["hit_cap"]))
                i += 1
            ok = [m for m, _ in ms if m["ok"]]
            key = f"spec={spec} thinking={'on' if thinking else 'off'} ctx={ctx} cap={cap}"
            cells[key] = {"decode_tps": mmm([m["decode_tps"] for m in ok]), "ttft_s": mmm([m["ttft_s"] for m in ok]),
                          "e2e_s": mmm([m["e2e_s"] for m in ok]), "completion_tokens": mmm([m["completion_tokens"] for m in ok]),
                          "hit_cap": sum(1 for _, h in ms if h), "n": len(ms), "errors": len(ms) - len(ok),
                          "prompt_tokens": mmm([m["prompt_tokens"] for m in ok]), "warmup": warm}
            print(f"M3 {key}: decode median {cells[key]['decode_tps']['median']} tok/s, "
                  f"hit cap {cells[key]['hit_cap']}/{len(ms)}")


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    if o.spec_label not in ("on", "off"):
        raise SystemExit("m3 needs --spec-label on|off (how the server was launched)")
    trials = pick(o.trials, cfg.quick, 1, 3)
    cells: dict[str, Any] = {}
    await _bucket(cfg, t, w, o, False, pick(o.contexts, cfg.quick, [2000], [16000, 32000, 64000]),
                  pick(o.caps, cfg.quick, [50], [50, 150, 300]), trials, o.spec_label, cells)
    if not o.no_thinking_bucket:
        await _bucket(cfg, t, w, o, True, pick(o.think_contexts, cfg.quick, [2000], [16000, 32000]),
                      pick(o.think_caps, cfg.quick, [100], [1000, 2000]), trials, o.spec_label, cells)
    summary = {"type": "summary", "suite": "m3", "spec": o.spec_label, "cells": cells,
               "think_effort": o.think_effort, "reuse_nonce": o.reuse_nonce}
    w.write(summary)
    return summary
