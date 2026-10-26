"""M5: TTFT of a main agent step while 1-2 auxiliary helper requests run concurrently.

The main step has the M1 trial shape (warm prefix + unique tool result + screenshot).
Helpers are independent ~2K-token summarization requests with fresh nonces and
`--aux-max-tokens` (256) outputs. Helpers start first and the main step follows after
`--aux-lead` seconds (0 = all at once), so the main step arrives while helpers prefill or
decode. The prefix is re-warmed before every trial (untimed) so that eviction by the
previous trial's helpers does not masquerade as contention (`--no-rewarm` to keep it).
Concurrent requests are never retried: a 429 (lithos-metal) is recorded per role.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..fixtures.store import new_digits
from ..metrics import mmm
from .common import RunConfig, Target, Writer
from .m1 import trial_messages, warm_messages
from .opts import SuiteOpts, pick


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    fx = t.fx
    counts = [0] + [n for n in pick(o.aux, cfg.quick, [1, 2], [1, 2]) if n > 0]
    trials = pick(o.trials, cfg.quick, 1, 3)
    tools = fx.tools()
    warm = warm_messages(fx)
    images = fx.m1["m5_images"]
    oc = await t.chat(warm, tools=tools, max_tokens=1)
    w.write(oc.line("m5", t.count_tokens, phase="warmup", cell={"phase": "warmup"}))
    cells: dict[str, Any] = {}
    idx = 0
    for n in counts:
        mains, auxs = [], []
        for i in range(trials):
            if not o.no_rewarm:
                oc = await t.chat(warm, tools=tools, max_tokens=1)
                w.write(oc.line("m5", t.count_tokens, phase="rewarm", cell={"phase": "rewarm", "n_aux": n}))
            msgs, meta = trial_messages(fx, cfg, idx, images)
            idx += 1

            async def main():
                if n and o.aux_lead > 0:
                    await asyncio.sleep(o.aux_lead)
                return await t.chat(msgs, tools=tools, max_tokens=o.m1_max_tokens, retry_429=(n == 0),
                                    count_local=False)

            async def aux(j: int):
                nonce = new_digits()
                return nonce, await t.chat(fx.prompt_messages(f"helper{1 + j % 2}", nonce=nonce),
                                           max_tokens=o.aux_max_tokens, retry_429=False, count_local=False)

            results = await asyncio.gather(main(), *(aux(j) for j in range(n)))
            mo = results[0]
            line = mo.line("m5", t.count_tokens, phase="trial", role="main", trial=i, n_aux=n,
                           cell={"n_aux": n}, **meta)
            w.write(line)
            mains.append(line["metrics"])
            for j, (nonce, ao) in enumerate(results[1:]):
                aline = ao.line("m5", t.count_tokens, phase="trial", role=f"aux{j + 1}", trial=i, n_aux=n,
                                cell={"n_aux": n, "role": "aux"}, nonce=nonce)
                w.write(aline)
                auxs.append(aline["metrics"])
        ok = [m for m in mains if m["ok"]]
        cells[str(n)] = {"main_ttft_s": mmm([m["ttft_s"] for m in ok]), "main_e2e_s": mmm([m["e2e_s"] for m in ok]),
                         "main_cached_tokens": mmm([m["cached_tokens"] for m in ok]),
                         "main_429": sum(1 for m in mains if m["http_status"] == 429),
                         "main_errors": len(mains) - len(ok),
                         "aux_ttft_s": mmm([m["ttft_s"] for m in auxs if m["ok"]]),
                         "aux_429": sum(1 for m in auxs if m["http_status"] == 429),
                         "aux_errors": sum(1 for m in auxs if not m["ok"]), "n": len(mains)}
    solo = cells["0"]["main_ttft_s"]["median"]
    for n, c in cells.items():
        med = c["main_ttft_s"]["median"]
        c["ttft_vs_solo"] = (med / solo) if med and solo else None
        print(f"M5 n_aux={n}: main TTFT median {med} s (x{c['ttft_vs_solo']}), main 429 {c['main_429']}, aux 429 {c['aux_429']}")
    summary = {"type": "summary", "suite": "m5", "cells": cells, "aux_lead_s": o.aux_lead}
    w.write(summary)
    return summary
