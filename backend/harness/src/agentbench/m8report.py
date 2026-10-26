"""`agentbench m8-report`: the M8 interference slowdown per framework and scenario (METHODOLOGY.md, section 4).

Reads the M8 raw files, `<framework>/raw/<suite>-<tag>-m8-<block>-<ts>.jsonl` (verified files only;
raw/_unverified/ is read only with --unverified, for mechanics checks). Each scenario block is
compared with the baseline blocks of the same framework, tag, suite and server launch (the header's
`interference.run`) on either side of it, the nearest by block index (normally the adjacent ones).
The baseline value is their mean, or one side alone when the other is missing; a comparison that is
not bracketed by both adjacent baselines is marked *.

Slowdown is the extra time the same work takes under the scenario, so the three columns read alike:
  M1  warm TTFT median:                   TTFT_S / TTFT_B - 1
  M3  decode tok/s median (16K, cap 150): tps_B / tps_S - 1   (time per token)
  M4  run total (median over runs):       total_S / total_B - 1
A scenario with several blocks reports the median over its blocks. The "baseline drift" row is the
noise floor: the median absolute change between consecutive baseline blocks, in the same terms.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import mmm
from .summarize import _ok, load, reqs, table

SUITES = ("m1", "m3", "m4")
ORDER = ("cpu1", "cpuhalf", "cpuall", "gpu", "mem", "disk", "power")
M3_CELL = (16000, 150)  # (context, cap) that run_m8.sh measures
NAME = re.compile(r"^(m1|m3|m4)-.+-m8-[A-Za-z0-9]+-\d{8}-\d{6}(-\d+)?\.jsonl$")
TAG = re.compile(r"^(?P<tag>.+)-m8-(?P<block>[A-Za-z0-9]+)$")
HEAD = {"m1": "M1 warm TTFT", "m3": "M3 decode (time per token)", "m4": "M4 total"}
UNIT = {"m1": "s", "m3": "tok/s", "m4": "s"}


@dataclass
class Block:
    fw: str
    tag: str
    scenario: str
    suite: str
    run: str
    index: int | None
    t: float
    value: float | None
    path: Path

    @property
    def pos(self) -> float:
        return float(self.index) if self.index is not None else self.t


def value(suite: str, run: dict[str, Any]) -> float | None:
    """The block's number for one suite, from the same samples summary.md uses (`_ok`: successful
    requests plus cap-truncated streams)."""
    if suite == "m1":
        return mmm([m["ttft_s"] for m in _ok(reqs([run], phase="warm"))])["median"]
    if suite == "m3":
        ls = [x for x in reqs([run], phase="trial") if not x["cell"].get("thinking")]
        cells = sorted({(x["cell"]["ctx"], x["cell"]["cap"]) for x in ls})
        cell = M3_CELL if M3_CELL in cells else (cells[0] if len(cells) == 1 else None)
        if cell is None:
            return None
        sel = [x for x in ls if (x["cell"]["ctx"], x["cell"]["cap"]) == cell]
        return mmm([m["decode_tps"] for m in _ok(sel)])["median"]
    # m4: the suite's own run totals (as in the published M4 totals), for complete runs only: a run
    # with a failed step would look faster than it was
    summ = next((ln for ln in run["lines"] if ln.get("type") == "summary"), None)
    if not summ:
        return None
    steps = [x for x in reqs([run]) if x.get("phase") in ("start", "step")]
    good = []
    for r, total in enumerate(summ.get("run_total_e2e_s") or []):
        sel = [x for x in steps if x.get("run", 0) == r]
        if sel and len(_ok(sel)) == len(sel) == summ.get("steps", len(sel) - 1) + 1:
            good.append(total)
    return mmm(good)["median"]


def slowdown(suite: str, base: float | None, scen: float | None) -> float | None:
    if not base or not scen:
        return None
    return base / scen - 1 if suite == "m3" else scen / base - 1


def collect(root: Path, frameworks: list[str] | None = None, tag: str | None = None,
            unverified: bool = False) -> list[Block]:
    dirs = [root / f for f in frameworks] if frameworks else sorted(p.parent for p in root.glob("*/raw"))
    out = []
    for fw_dir in dirs:
        raw = fw_dir / "raw" / ("_unverified" if unverified else "")
        for f in sorted(raw.glob("m[134]-*-m8-*.jsonl")):
            if not NAME.match(f.name):
                continue
            try:
                run = load(f)
            except (ValueError, json.JSONDecodeError):
                continue
            h = run["header"]
            m = TAG.match(h.get("tag", ""))
            if not m or h.get("suite") not in SUITES or (tag and m["tag"] != tag):
                continue
            inter = h.get("interference") or {}
            try:
                t = _dt.datetime.fromisoformat(h["created"]).timestamp()
            except (KeyError, ValueError):
                continue
            out.append(Block(fw=fw_dir.name, tag=m["tag"], scenario=str(inter.get("scenario") or m["block"]),
                             suite=h["suite"], run=str(inter.get("run") or f"pid {h.get('args', {}).get('server_pid')}"),
                             index=inter.get("block"), t=t, value=value(h["suite"], run), path=f))
    return out


def groups(blocks: list[Block]) -> dict[tuple[str, str, str, str], list[Block]]:
    """(framework, tag, suite, run) -> blocks in run order."""
    g: dict[tuple[str, str, str, str], list[Block]] = defaultdict(list)
    for b in blocks:
        g[(b.fw, b.tag, b.suite, b.run)].append(b)
    for bs in g.values():
        bs.sort(key=lambda b: (b.pos, b.t))
    return g


def compare(blocks: list[Block]) -> list[dict[str, Any]]:
    """One row per scenario block: the bracketing baselines, the baseline value, the slowdown."""
    rows = []
    for (fw, tag, suite, run), bs in groups(blocks).items():
        bases = [b for b in bs if b.scenario == "base" and b.value is not None]
        for b in bs:
            if b.scenario == "base":
                continue
            before = max((x for x in bases if x.pos < b.pos), key=lambda x: x.pos, default=None)
            after = min((x for x in bases if x.pos > b.pos), key=lambda x: x.pos, default=None)
            vals = [x.value for x in (before, after) if x is not None]
            base = sum(vals) / len(vals) if vals else None
            # bracketed: a baseline on each side and no other block between (a quarantined
            # baseline makes the nearest one further away)
            between = [x for x in bs if x is not b and before is not None and after is not None
                       and before.pos < x.pos < after.pos]
            rows.append({"fw": fw, "tag": tag, "suite": suite, "run": run, "scenario": b.scenario, "block": b.index,
                         "before": before.value if before else None, "after": after.value if after else None,
                         "base": base, "value": b.value, "slowdown": slowdown(suite, base, b.value),
                         "bracketed": before is not None and after is not None and not between, "file": b.path.name})
    return rows


def drift(blocks: list[Block]) -> dict[tuple[str, str, str], float]:
    """(framework, tag, suite) -> median |change| between consecutive baseline blocks."""
    per: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (fw, tag, suite, _run), bs in groups(blocks).items():
        bases = [b for b in bs if b.scenario == "base" and b.value is not None]
        for x, y in zip(bases, bases[1:]):
            d = slowdown(suite, x.value, y.value)
            if d is not None:
                per[(fw, tag, suite)].append(abs(d))
    return {k: statistics.median(v) for k, v in per.items()}


def pct(x: float | None, signed: bool = True) -> str:
    return "-" if x is None else (f"{x * 100:+.1f} %" if signed else f"±{x * 100:.1f} %")


def num(x: float | None, suite: str) -> str:
    return "-" if x is None else f"{x:.{1 if suite == 'm3' else 2}f} {UNIT[suite]}"


def scen_key(s: str) -> tuple[int, str]:
    return (ORDER.index(s), s) if s in ORDER else (len(ORDER), s)


def report(root: Path, frameworks: list[str] | None = None, tag: str | None = None, details: bool = False,
           unverified: bool = False) -> str:
    blocks = collect(root, frameworks, tag, unverified)
    if not blocks:
        return "_no M8 result files found_\n"
    rows, noise = compare(blocks), drift(blocks)
    agg: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["slowdown"] is not None:
            agg[(r["fw"], r["tag"], r["scenario"])][r["suite"]].append(r)
    base_vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for b in blocks:
        if b.scenario == "base" and b.value is not None:
            base_vals[(b.fw, b.tag, b.suite)].append(b.value)

    def cell(rs: list[dict[str, Any]]) -> str:
        if not rs:
            return "-"
        out = pct(statistics.median(r["slowdown"] for r in rs))
        if not all(r["bracketed"] for r in rs):
            out += "*"
        return out + (f" (n={len(rs)})" if len(rs) > 1 else "")

    out_rows = []
    for fw, tg in sorted({(b.fw, b.tag) for b in blocks}):
        out_rows.append([fw, tg, "baseline (median)"] + [num(mmm(base_vals[(fw, tg, s)])["median"], s) for s in SUITES])
        out_rows.append([fw, tg, "baseline drift (B to B)"] + [pct(noise.get((fw, tg, s)), signed=False) for s in SUITES])
        for sc in sorted({r["scenario"] for r in rows if (r["fw"], r["tag"]) == (fw, tg)}, key=scen_key):
            out_rows.append([fw, tg, sc] + [cell(agg[(fw, tg, sc)].get(s, [])) for s in SUITES])
    md = ["## M8 interference: slowdown against the clean baseline", "",
          "Extra time for the same work under each scenario, against the mean of the clean baseline blocks "
          "just before and after it on the same server launch (METHODOLOGY.md, section 4). M3 is time per "
          "token, i.e. the inverse of decode tok/s at 16K context, cap 150. * = not bracketed by both adjacent "
          "baselines. Baseline drift is the median change between consecutive baselines: the noise floor.", "",
          table(["Framework", "Tag", "Scenario"] + [HEAD[s] for s in SUITES], out_rows)]
    if details:
        det = [[r["fw"], r["tag"], r["scenario"], str(r["block"] if r["block"] is not None else "-"), r["suite"],
                num(r["before"], r["suite"]), num(r["value"], r["suite"]), num(r["after"], r["suite"]),
                pct(r["slowdown"]) + ("" if r["bracketed"] else "*"), f"`{r['file']}`"]
               for r in sorted(rows, key=lambda r: (r["fw"], r["tag"], r["run"], r["block"] or 0, r["suite"]))]
        md += ["", "### Per block", "",
               table(["Framework", "Tag", "Scenario", "Block", "Suite", "Baseline before", "Scenario", "Baseline after",
                      "Slowdown", "Source"], det)]
    return "\n".join(md) + "\n"
