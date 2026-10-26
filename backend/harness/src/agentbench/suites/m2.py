"""M2: cold prefill throughput at 16K / 32K / 64K (64K = 64,000 tokens).

Each request has a fresh digits-only nonce at the very start of the prompt (no tools are
sent), max_tokens=1. Throughput = prompt_tokens / TTFT, using the server's prompt_tokens
when reported and the local count otherwise. If no token-bearing chunk arrives (some
servers skip a whitespace-only first token), e2e is used and flagged.
"""

from __future__ import annotations

from typing import Any

from ..fixtures.store import new_digits
from ..metrics import mmm
from .common import RunConfig, Target, Writer
from .opts import SuiteOpts, pick


def prefill(m: dict[str, Any]) -> tuple[float | None, str | None]:
    if m["ttft_s"] is not None:
        return m["ttft_s"], "ttft"
    if m["ok"] and m["e2e_s"] is not None:
        return m["e2e_s"], "e2e"
    return None, None


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    fx = t.fx
    sizes = pick(o.sizes, cfg.quick, [2000], [16000, 32000, 64000])
    trials = pick(o.trials, cfg.quick, 1, 3)
    oc = await t.chat(fx.prompt_messages(f"cold-{min(sizes)}", nonce=new_digits()), max_tokens=1)
    w.write(oc.line("m2", t.count_tokens, phase="warmup", cell={"size": min(sizes), "phase": "warmup"}))
    cells: dict[str, Any] = {}
    for size in sizes:
        tps, pre = [], []
        for i in range(trials):
            nonce = new_digits()
            oc = await t.chat(fx.prompt_messages(f"cold-{size}", nonce=nonce), max_tokens=1)
            line = oc.line("m2", t.count_tokens, phase="trial", trial=i, cell={"size": size}, nonce=nonce)
            m = line["metrics"]
            p_s, src = prefill(m)
            n = m["prompt_tokens"] or m["prompt_tokens_local"]
            line["derived"] = {"prefill_s": p_s, "prefill_source": src, "prompt_tokens_used": n,
                               "throughput_tps": (n / p_s) if n and p_s else None}
            w.write(line)
            if line["derived"]["throughput_tps"]:
                tps.append(line["derived"]["throughput_tps"])
                pre.append(p_s)
        cells[str(size)] = {"throughput_tps": mmm(tps), "prefill_s": mmm(pre)}
        print(f"M2 {size}: prefill median {cells[str(size)]['prefill_s']['median']} s, "
              f"throughput median {cells[str(size)]['throughput_tps']['median']} tok/s")
    summary = {"type": "summary", "suite": "m2", "cells": cells}
    w.write(summary)
    return summary
