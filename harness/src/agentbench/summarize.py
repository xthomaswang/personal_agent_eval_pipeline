"""`agentbench summarize`: markdown tables (median, min, max) per suite and cell.

Reads `<framework>/raw/*.jsonl`. By default the latest file per (tag, suite) is used;
`--all-runs` pools every file. `--write` upserts one section per tag into
`<framework>/summary.md` between `<!-- agentbench:begin summary <tag> -->` markers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .metrics import mmm

ORDER = ["gate", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]


def load(path: Path) -> dict[str, Any]:
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path}: no provenance header")
    return {"path": path, "header": lines[0], "lines": lines[1:]}


def select(files: Iterable[Path], tag: str | None = None, all_runs: bool = False) -> dict[str, dict[str, list[dict]]]:
    """{tag: {suite: [run, ...]}}"""
    by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(files):
        try:
            run = load(f)
        except (ValueError, json.JSONDecodeError):
            continue
        h = run["header"]
        if "tag" not in h or "suite" not in h:  # not a suite run, e.g. Core AI's ab-* direct-call files
            continue
        if tag and h["tag"] != tag:
            continue
        by[h["tag"]][h["suite"]].append(run)
    if not all_runs:
        for t in by:
            for s in by[t]:
                by[t][s] = [max(by[t][s], key=lambda r: (r["header"]["created"], r["path"].name))]
    return by


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def cell(xs: list[Any], nd: int = 3) -> str:
    m = mmm(xs)
    if not m["n"]:
        return "-"
    if m["n"] == 1:
        return _f(m["median"], nd)
    return f"{_f(m['median'], nd)} ({_f(m['min'], nd)}-{_f(m['max'], nd)})"


def med(xs: list[Any], nd: int = 0) -> str:
    m = mmm(xs)
    return _f(m["median"], nd) if m["n"] else "-"


def reqs(runs: list[dict], **match: Any) -> list[dict]:
    out = []
    for r in runs:
        for ln in r["lines"]:
            if ln.get("type") != "request":
                continue
            if all(ln.get(k) == v for k, v in match.items()):
                out.append(ln)
    return out


def table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


CAP_TRUNCATED = (
    "incomplete_tool_call",  # oMLX: the token cap cut a tool call; the stream ends with this error code
    "incomplete tool call; increase the output budget",  # lithos-metal: the same case, by message
)


def cap_truncated(m: dict) -> bool:
    """The stream produced tokens and ended at the token cap inside a tool call, which the server
    flagged as a stream error (oMLX and lithos-metal). llama.cpp reports the same case as
    finish_reason=length."""
    err = m.get("error") or ""
    return (not m["ok"] and m.get("ttft_s") is not None and m.get("e2e_s") is not None
            and any(c in err for c in CAP_TRUNCATED))


def _ok(lines: list[dict]) -> list[dict]:
    """Latency samples: successful requests plus cap-truncated streams (their TTFT and e2e are
    complete measurements; they carry no usage, so no cached-token count)."""
    return [ln["metrics"] for ln in lines if ln["metrics"]["ok"] or cap_truncated(ln["metrics"])]


def truncation_note(lines: list[dict]) -> str:
    errs = [ln["metrics"]["error"] for ln in lines if cap_truncated(ln["metrics"])]
    if not errs:
        return ""
    flag = ("an `incomplete_tool_call` stream error" if all("incomplete_tool_call" in e for e in errs)
            else 'a stream error ("incomplete tool call")')
    return (f"\n\n{len(errs)} of {len(lines)} requests ended at the token cap inside a tool call, and the server "
            f"flagged it as {flag}. They count as latency samples. They have no "
            "usage chunk, so they report no cached tokens.")


# ------------------------------------------------------------------ suites
def s_gate(runs: list[dict]) -> str:
    summ = next((ln for ln in runs[-1]["lines"] if ln.get("type") == "summary"), None)
    if not summ:
        return "_no gate summary in this file_"
    rows = [[gid, it["requirement"], f"**{it['result']}**", it["detail"].replace("|", "\\|")]
            for gid, it in summ["items"].items()]
    return table(["ID", "Requirement", "Result", "Detail"], rows)


def s_m1(runs: list[dict]) -> str:
    rows = []
    for ph in ("warm", "cold"):
        ls = reqs(runs, phase=ph)
        ms = _ok(ls)
        rows.append([ph, str(len(ms)), cell([m["ttft_s"] for m in ms]), cell([m["e2e_s"] for m in ms]),
                     med([m["cached_tokens"] for m in ms]), med([m["prompt_tokens"] or m["prompt_tokens_local"] for m in ms]),
                     ", ".join(sorted({m["first_kind"] for m in ms if m["first_kind"]})) or "-", str(len(ls) - len(ms))])
    w = mmm([m["ttft_s"] for m in _ok(reqs(runs, phase="warm"))])["median"]
    c = mmm([m["ttft_s"] for m in _ok(reqs(runs, phase="cold"))])["median"]
    note = f"\n\nCold / warm TTFT (medians): {_f(c / w, 2) if w and c else '-'}x."
    return table(["Phase", "n", "TTFT s", "e2e s", "cached tok (med)", "prompt tok (med)", "first chunk", "errors"],
                 rows) + note


def s_m2(runs: list[dict]) -> str:
    sizes = sorted({ln["cell"]["size"] for ln in reqs(runs, phase="trial")})
    rows = []
    for size in sizes:
        ls = reqs(runs, phase="trial")
        ls = [x for x in ls if x["cell"]["size"] == size]
        d = [x["derived"] for x in ls if x.get("derived", {}).get("throughput_tps")]
        ms = _ok(ls)
        rows.append([f"{size:,}", str(len(d)), cell([x["prefill_s"] for x in d]), cell([x["throughput_tps"] for x in d], 0),
                     med([m["prompt_tokens"] for m in ms]), med([m["cached_tokens"] for m in ms]),
                     ", ".join(sorted({x["prefill_source"] for x in d})) or "-", str(len(ls) - len(ms))])
    return table(["Prompt tokens (target)", "n", "prefill s (TTFT)", "throughput tok/s", "prompt tok (server)",
                  "cached tok", "timing", "errors"], rows)


def s_m3(runs: list[dict]) -> str:
    ls = reqs(runs, phase="trial")
    keys = sorted({(x["cell"]["spec"], x["cell"]["thinking"], x["cell"]["ctx"], x["cell"]["cap"]) for x in ls},
                  key=lambda k: (k[0], k[1], k[2], k[3]))
    rows = []
    for spec, th, ctx, cap in keys:
        sel = [x for x in ls if (x["cell"]["spec"], x["cell"]["thinking"], x["cell"]["ctx"], x["cell"]["cap"]) == (spec, th, ctx, cap)]
        ms = _ok(sel)
        hit = sum(1 for x in sel if x.get("derived", {}).get("hit_cap"))
        rows.append([spec, "on" if th else "off", f"{ctx:,}", str(cap), str(len(ms)), cell([m["decode_tps"] for m in ms], 1),
                     cell([m["ttft_s"] for m in ms]), cell([m["e2e_s"] for m in ms]), med([m["completion_tokens"] for m in ms]),
                     f"{hit}/{len(sel)}", med([m["gap_p50_ms"] for m in ms], 1), med([m["gap_p99_ms"] for m in ms], 1),
                     med([m["tokens_per_chunk"] for m in ms], 2)])
    skips = [ln for r in runs for ln in r["lines"] if ln.get("type") == "skip"]
    for s in skips:
        c = s["cell"]
        rows.append([c["spec"], "on" if c["thinking"] else "off", f"{c['ctx']:,}", str(c["cap"]), "0",
                     "N/A: " + s["reason"].replace("|", "\\|")] + ["-"] * 7)
    return table(["spec", "thinking", "context", "cap", "n", "decode tok/s", "TTFT s", "e2e s", "out tok (med)",
                  "hit cap", "gap p50 ms", "gap p99 ms", "tok/chunk"], rows)


def s_m4(runs: list[dict]) -> str:
    ls = [x for x in reqs(runs) if x.get("phase") in ("start", "step")]
    steps = sorted({x["step"] for x in ls})
    rows = []
    for k in steps:
        sel = [x for x in ls if x["step"] == k]
        ms = _ok(sel)
        prompt = [m["prompt_tokens"] or m["prompt_tokens_local"] for m in ms]
        cached = [m["cached_tokens"] for m in ms]
        share = [c / p for c, p in zip(cached, prompt) if c is not None and p]
        rows.append([str(k) + (" (start)" if k == 0 else ""), str(len(ms)), cell([m["ttft_s"] for m in ms]),
                     cell([m["e2e_s"] for m in ms]), med(cached), med(prompt), med(share, 2), str(len(sel) - len(ms))])
    summ = next((ln for r in runs for ln in r["lines"] if ln.get("type") == "summary"), {})
    tot = summ.get("run_total_e2e_s")
    # slope recomputed from the request lines, so cap-truncated steps count (the suite's own value skips them);
    # also without step 1, a near-full re-prefill when the prefix cache does not reuse the step-0 prompt (oMLX)
    from .suites.m4 import slope
    meds = {k: mmm([m["ttft_s"] for m in _ok([x for x in ls if x["step"] == k])])["median"] for k in steps if k >= 1}

    def sl(first: int) -> str:
        ks = [k for k in sorted(meds) if k >= first and meds[k] is not None]
        return _f(slope([float(k) for k in ks], [meds[k] for k in ks]), 4)

    last = max(steps) if steps else 0
    note = (f"\n\nTTFT slope over steps 1-{last}: {sl(1)} s/step"
            + (f" (steps 2-{last}: {sl(2)} s/step)" if last >= 2 else "")
            + f"; total e2e per run: {', '.join(_f(x, 1) for x in tot) if tot else '-'} s.")
    return table(["Step", "n", "TTFT s", "e2e s", "cached tok (med)", "prompt tok (med)", "cached share", "errors"],
                 rows) + note + truncation_note(ls)


def s_m5(runs: list[dict]) -> str:
    ls = reqs(runs, phase="trial")
    counts = sorted({x["n_aux"] for x in ls})
    solo = mmm([m["ttft_s"] for m in _ok([x for x in ls if x["n_aux"] == 0 and x["role"] == "main"])])["median"]
    rows = []
    for n in counts:
        main = [x for x in ls if x["n_aux"] == n and x["role"] == "main"]
        aux = [x for x in ls if x["n_aux"] == n and x["role"] != "main"]
        ms = _ok(main)
        m_med = mmm([m["ttft_s"] for m in ms])["median"]
        rows.append([str(n), str(len(ms)), cell([m["ttft_s"] for m in ms]), _f(m_med / solo, 2) if m_med and solo else "-",
                     cell([m["e2e_s"] for m in ms]), med([m["cached_tokens"] for m in ms]),
                     str(sum(1 for x in main if x["metrics"]["http_status"] == 429)),
                     str(sum(1 for x in aux if x["metrics"]["http_status"] == 429)),
                     cell([m["ttft_s"] for m in _ok(aux)])])
    return table(["aux requests", "n", "main TTFT s", "vs solo", "main e2e s", "main cached tok", "main 429",
                  "aux 429", "aux TTFT s"], rows)


def s_m6(runs: list[dict]) -> str:
    summ = next((ln for ln in runs[-1]["lines"] if ln.get("type") == "summary"), None)
    if not summ:
        return "_no M6 summary in this file_"

    def gib(x: Any) -> str:
        return "-" if not x else f"{x / 2**30:.2f}"

    fp, mem, dr = summ["footprint"], summ["memory"], summ.get("drift") or {}
    rows = [["footprint at {:,} tokens: server tree peak GiB".format(fp["size"]), gib(fp.get("peak_tree_bytes"))],
            ["footprint: system used peak GiB", gib(fp.get("peak_system_used_bytes"))],
            ["loop minutes / runs (complete) / requests", f"{summ['minutes']} / {summ['runs']} ({summ['complete_runs']}) / {summ['requests']}"],
            ["errors", str(summ["errors"])],
            ["server tree GiB start / peak / end", f"{gib(mem.get('start_tree_bytes'))} / {gib(mem.get('peak_tree_bytes'))} / {gib(mem.get('end_tree_bytes'))}"],
            ["server tree growth GiB", gib(mem.get("growth_tree_bytes"))],
            ["system used GiB start / end", f"{gib(mem.get('start_system_used_bytes'))} / {gib(mem.get('end_system_used_bytes'))}"],
            ["e2e last run / first run (median per step)", _f(dr.get("e2e_last_over_first_median"), 2)],
            ["e2e s over all requests", f"{_f(summ['e2e_s']['median'])} ({_f(summ['e2e_s']['min'])}-{_f(summ['e2e_s']['max'])})"]]
    return table(["Metric", "Value"], rows)


def s_m7(runs: list[dict]) -> str:
    ls = reqs(runs, phase="trial")
    rows = []
    for x in ls:
        j = x.get("judgement", {})
        rows.append([x["cell"]["case"], x["cell"]["category"], "pass" if j.get("pass") else "**FAIL**",
                     "; ".join(j.get("reasons", []))[:240].replace("|", "\\|") or "-"])
    for r in runs:
        for s in r["lines"]:
            if s.get("type") == "skip" and s.get("suite") == "m7":
                rows.append([s["cell"]["case"], s["cell"]["category"], "N/A", s["reason"]])
    passed = sum(1 for x in ls if x.get("judgement", {}).get("pass"))
    return f"Passed {passed}/{len(ls)}.\n\n" + table(["Case", "Category", "Result", "Reasons"], rows)


SUITE_FN = {"gate": s_gate, "m1": s_m1, "m2": s_m2, "m3": s_m3, "m4": s_m4, "m5": s_m5, "m6": s_m6, "m7": s_m7}
TITLES = {"gate": "Gate G1-G5", "m1": "M1 cached-prefix TTFT", "m2": "M2 cold prefill", "m3": "M3 short structured decode",
          "m4": "M4 20-step trajectory (per step)", "m5": "M5 concurrency", "m6": "M6 memory and stability",
          "m7": "M7 tool-call correctness"}


def render(by_suite: dict[str, list[dict]], tag: str) -> str:
    out = [f"## Results, tag `{tag}`"]
    for s in ORDER:
        runs = by_suite.get(s)
        if not runs:
            continue
        h = runs[-1]["header"]
        mode = h.get("mode", {})
        flags = [k for k in ("quick", "text_only") if mode.get(k)]
        if mode.get("tool_call_mode") == "structured_output":
            flags.append("tool calls via structured output")
        src = ", ".join(f"`raw/{r['path'].name}`" for r in runs)
        out.append(f"### {TITLES[s]}{' (' + ', '.join(flags) + ')' if flags else ''}\n\nSource: {src}.\n\n"
                   + SUITE_FN[s](runs))
    return "\n\n".join(out) + "\n"


def summarize(fw_dir: Path, tag: str | None = None, files: list[Path] | None = None, all_runs: bool = False,
              write: bool = False) -> str:
    from .suites.common import upsert_section

    from .suites.common import remove_section

    full = files is None  # a full regeneration from raw/*.jsonl (run-all passes only its own files)
    files = files or sorted((fw_dir / "raw").glob("*.jsonl"))
    by = select(files, tag, all_runs)
    parts = []
    for t in sorted(by):
        md = render(by[t], t)
        parts.append(md)
        if write:
            upsert_section(fw_dir / "summary.md", f"summary {t}", md, f"{fw_dir.name}: results")
    if write and full:  # capabilities.md shows only verified gate runs: the latest one per tag, or nothing
        for t in ([tag] if tag else sorted(by)):
            gates = (by.get(t) or {}).get("gate") or []
            frag = gates[-1]["path"].with_suffix(".md") if gates else None
            if frag is not None and frag.exists() and "_unverified" not in frag.parts:
                upsert_section(fw_dir / "capabilities.md", f"gate {t}", frag.read_text(encoding="utf-8"),
                               f"{fw_dir.name}: capabilities")
            else:
                remove_section(fw_dir / "capabilities.md", f"gate {t}")
    return "\n".join(parts) if parts else "_no result files found_\n"
