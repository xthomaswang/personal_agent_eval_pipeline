"""M7: tool-call correctness, ~20 streamed prompts on top of the agent prefix.

Success per case: expected call(s) present with the right name, schema-valid JSON
arguments, matching argument values, no spurious call, no markup leak (see m7check).
Image cases are N/A in --text-only mode.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..m7check import check_case
from .common import RunConfig, Target, Writer
from .opts import SuiteOpts, pick


async def run(cfg: RunConfig, t: Target, w: Writer, o: SuiteOpts) -> dict[str, Any]:
    fx = t.fx
    cases = fx.m7["cases"]
    if o.cases:
        cases = [c for c in cases if c["id"] in o.cases or any(c["id"].startswith(x) for x in o.cases)]
    elif cfg.quick:
        cases = [c for c in cases if c["id"] in fx.m7["quick"]]
    trials = pick(o.trials, cfg.quick, 1, 1)
    tools, schemas = fx.tools(), fx.schemas()
    oc = await t.chat([fx.agent_system(), {"role": "user", "content": "Hello."}], tools=tools, max_tokens=1)
    w.write(oc.line("m7", t.count_tokens, phase="warmup", cell={"phase": "warmup"}))
    results = []
    for c in cases:
        cell = {"case": c["id"], "category": c["category"]}
        if c.get("image") and cfg.text_only:
            reason = "N/A: image case in --text-only mode"
            w.write({"type": "skip", "suite": "m7", "cell": cell, "reason": reason})
            results.append({"case": c["id"], "category": c["category"], "pass": None, "reasons": [reason]})
            continue
        msgs = [fx.agent_system()] + fx.resolve(c["messages"], text_only=cfg.text_only)
        for i in range(trials):
            oc = await t.chat(msgs, tools=tools, max_tokens=o.m7_max_tokens)
            if oc.rec.ok:
                res = check_case(c, oc.rec.tool_calls, oc.rec.content, oc.rec.markup_leak(), schemas)
            else:
                res = {"pass": False, "reasons": [oc.metrics()["error"] or "request failed"]}
            line = oc.line("m7", t.count_tokens, phase="trial", trial=i, cell=cell, judgement=res,
                           calls=[{"name": x.get("name"), "arguments": x.get("arguments")} for x in oc.rec.tool_calls])
            w.write(line)
            results.append({"case": c["id"], "category": c["category"], **res})
    scored = [r for r in results if r["pass"] is not None]
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r["pass"])
    summary = {"type": "summary", "suite": "m7", "passed": sum(1 for r in scored if r["pass"]), "scored": len(scored),
               "na": len(results) - len(scored),
               "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()},
               "failures": [{"case": r["case"], "reasons": r["reasons"]} for r in scored if not r["pass"]],
               "results": results}
    w.write(summary)
    print(f"M7: {summary['passed']}/{summary['scored']} passed ({summary['na']} N/A)")
    for f in summary["failures"]:
        print(f"  FAIL {f['case']}: {'; '.join(f['reasons'])[:200]}")
    return summary
