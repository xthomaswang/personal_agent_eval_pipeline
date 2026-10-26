"""M6: memory and stability.

1. Footprint at 64K context: a 64,000-token cold prompt while the server's PID tree is
   sampled every second (`footprint`, RSS fallback, plus system `vm_stat`).
2. Sustained loop: M4 trajectories (fresh nonce per run) back to back for `--minutes`,
   with memory sampled every `--mem-interval` seconds. Reports memory start / peak / end
   and growth, latency drift (same step index, first vs last complete run), and errors.
Without `--server-pid` only system memory is sampled.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

from ..fixtures.store import new_digits
from ..memory import MemorySampler, parse_vm_stat
from ..metrics import mmm
from .common import RunConfig, Target, Writer, eprint
from .m4 import trajectory
from .opts import SuiteOpts, pick


class _SystemSampler:
    """vm_stat-only sampler when no server PID is given."""

    def __init__(self, sink, interval: float) -> None:
        self.sink, self.interval, self.samples, self._task = sink, interval, [], None

    async def _loop(self) -> None:
        while True:
            p = await asyncio.create_subprocess_exec("vm_stat", stdout=asyncio.subprocess.PIPE)
            out, _ = await p.communicate()
            s = {"type": "memory", "t_wall": time.time(), "root_pid": None, "tree_bytes": None,
                 "system": parse_vm_stat(out.decode())}
            self.samples.append(s)
            self.sink(s)
            await asyncio.sleep(self.interval)

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        p = await asyncio.create_subprocess_exec("vm_stat", stdout=asyncio.subprocess.PIPE)
        out, _ = await p.communicate()
        s = {"type": "memory", "t_wall": time.time(), "root_pid": None, "tree_bytes": None,
             "system": parse_vm_stat(out.decode())}
        self.samples.append(s)
        self.sink(s)

    def peak_tree_bytes(self):
        return None


def _sampler(cfg: RunConfig, w: Writer, interval: float):
    if cfg.server_pid:
        return MemorySampler(cfg.server_pid, sink=w.write, interval=interval)
    return _SystemSampler(w.write, interval)


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    if o.minutes is None:
        raise SystemExit("m6 needs --minutes N (and --server-pid PID for the server's footprint)")
    if not cfg.server_pid:
        eprint("m6: no --server-pid given; only system memory (vm_stat) is sampled")
    size = 2000 if cfg.quick else 64000
    async with _sampler(cfg, w, 1.0) as ms:
        oc = await t.chat(t.fx.prompt_messages(f"cold-{size}", nonce=new_digits()), max_tokens=8)
    w.write(oc.line("m6", t.count_tokens, phase="footprint", cell={"phase": f"footprint-{size}"}))
    footprint = {"size": size, "ok": oc.rec.ok, "peak_tree_bytes": ms.peak_tree_bytes(),
                 "peak_system_used_bytes": max((s["system"]["used_bytes"] for s in ms.samples), default=None)}
    steps = min(pick(o.steps, cfg.quick, 3, 20), len(t.fx.m4["steps"]))
    deadline = time.monotonic() + o.minutes * 60
    runs: list[list[dict[str, Any]]] = []
    async with _sampler(cfg, w, o.mem_interval) as loop_ms:
        r = 0
        while time.monotonic() < deadline:
            runs.append(await trajectory(cfg, t, w, o, r, steps, suite="m6", deadline=deadline))
            r += 1
    samples = [s for s in loop_ms.samples if s.get("tree_bytes")]
    sysu = [s["system"]["used_bytes"] for s in loop_ms.samples]
    mem = {"start_tree_bytes": samples[0]["tree_bytes"] if samples else None,
           "end_tree_bytes": samples[-1]["tree_bytes"] if samples else None,
           "peak_tree_bytes": max((s["tree_bytes"] for s in samples), default=None),
           "start_system_used_bytes": sysu[0] if sysu else None, "end_system_used_bytes": sysu[-1] if sysu else None}
    if mem["start_tree_bytes"] and mem["end_tree_bytes"]:
        mem["growth_tree_bytes"] = mem["end_tree_bytes"] - mem["start_tree_bytes"]
    full = [run for run in runs if len(run) == steps + 1]
    drift = None
    if len(full) >= 2:
        ratios = [b["e2e_s"] / a["e2e_s"] for a, b in zip(full[0], full[-1]) if a["ok"] and b["ok"] and a["e2e_s"]]
        drift = {"e2e_last_over_first_median": statistics.median(ratios) if ratios else None,
                 "first_run_total_e2e_s": sum(m["e2e_s"] or 0 for m in full[0]),
                 "last_run_total_e2e_s": sum(m["e2e_s"] or 0 for m in full[-1])}
    allm = [m for run in runs for m in run]
    summary = {"type": "summary", "suite": "m6", "minutes": o.minutes, "footprint": footprint, "memory": mem,
               "drift": drift, "runs": len(runs), "complete_runs": len(full), "requests": len(allm),
               "errors": sum(1 for m in allm if not m["ok"]),
               "e2e_s": mmm([m["e2e_s"] for m in allm if m["ok"]]), "ttft_s": mmm([m["ttft_s"] for m in allm if m["ok"]])}
    w.write(summary)
    print(f"M6 {o.minutes} min: {len(allm)} requests, {summary['errors']} errors, peak tree "
          f"{(mem['peak_tree_bytes'] or 0) / 2**30:.2f} GiB, drift {drift}")
    return summary
