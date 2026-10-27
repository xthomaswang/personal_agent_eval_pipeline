"""Build the results site, docs/index.html (served by GitHub Pages), from docs/benchmark/**/summary.json.

    python3 scripts/build_docs.py           # write docs/index.html
    python3 scripts/build_docs.py --check   # exit 1 if docs/index.html is out of date (for CI and pre-commit)

Standard library only. The format of summary.json is in docs/benchmark/README.md. The page is
deterministic: the same results give the same bytes, so --check can compare.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "docs" / "benchmark"
PAGE = ROOT / "docs" / "index.html"
REPO_URL = "https://github.com/xthomaswang/personal_agent_eval_pipeline"   # a fork changes it here
SUITES = {"backend": "Backend", "agent_ability": "Agent ability", "memory": "Memory"}
REQUIRED = ("schema", "suite", "protocol", "title", "model", "family", "quant", "run_id", "date", "system", "metrics")
LEVELS = ("machine", "family", "quant", "suite", "run_id")
e = html.escape


# ---- results -------------------------------------------------------------------------------------------

def well_typed(row: dict) -> bool:
    return (all(isinstance(row[k], str) for k in ("protocol", "title", "model", "family", "quant", "run_id", "date"))
            and isinstance(row["system"], dict) and isinstance(row.get("reference", False), bool)
            and isinstance(row.get("notes", ""), str) and isinstance(row.get("details", ""), str)
            and isinstance(row["metrics"], list)
            and all(isinstance(m, dict) and isinstance(m.get("name"), str) and isinstance(m.get("value"), str)
                    for m in row["metrics"]))


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        sys.exit(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")


def exists_exactly(rel: str) -> bool:
    """Whether a file exists with exactly this case. macOS volumes ignore case, GitHub links do not."""
    here = ROOT
    for part in rel.split("/"):
        if not here.is_dir() or part not in os.listdir(here):
            return False
        here = here / part
    return here.is_file()


def load() -> list[dict]:
    """Every row: the summary.json directly inside a <run-id>/ directory. A run's own files may hold
    more summary.json files deeper down; one that looks like a row, or any shallower one, is a path
    mistake."""
    for path in BENCH.rglob("summary.json"):
        depth = len(path.relative_to(BENCH).parts)
        misplaced = depth < len(LEVELS) + 1
        if depth > len(LEVELS) + 1:
            data = read_json(path)
            misplaced = isinstance(data, dict) and {"schema", "protocol", "suite"} <= data.keys()
        if misplaced:
            sys.exit(f"{path.relative_to(ROOT)}: a row belongs in <machine>/<family>/<quant>/<suite>/<run-id>/")
    rows = []
    for path in sorted(BENCH.glob("*/*/*/*/*/summary.json")):
        where = path.relative_to(ROOT)
        row = read_json(path)
        if not isinstance(row, dict):
            sys.exit(f"{where}: not a JSON object")
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            sys.exit(f"{where}: missing {', '.join(missing)}")
        if row["schema"] != 1 or not isinstance(row["suite"], str) or row["suite"] not in SUITES:
            sys.exit(f"{where}: schema must be 1 and suite one of {', '.join(SUITES)}")
        if not well_typed(row):
            sys.exit(f"{where}: a field has the wrong type (see docs/benchmark/README.md)")
        found = dict(zip(LEVELS, path.parent.relative_to(BENCH).parts))
        want = {**{k: row.get(k) for k in LEVELS}, "machine": row["system"].get("machine")}
        if found != want:
            sys.exit(f"{where}: the path does not match system.machine/family/quant/suite/run_id")
        row["_dir"] = path.parent.relative_to(ROOT).as_posix()
        if row.get("details"):
            target = posixpath.normpath(f"{row['_dir']}/{row['details']}")
            if not target.startswith("docs/benchmark/") or not exists_exactly(target):
                sys.exit(f"{where}: details {row['details']!r} is not a file under docs/benchmark")
            row["_details"] = target
        rows.append(row)
    return rows


def tables(rows: list[dict]) -> list[dict]:
    """One table per (machine, suite, protocol); every row of a table has the same title and metrics. Results
    from different machines never share a table: speed and timeouts depend on the machine."""
    out: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        machine = row["system"]["machine"]
        key = (machine, row["suite"], row["protocol"])
        names = [m["name"] for m in row["metrics"]]
        table = out.setdefault(key, {"machine": machine, "suite": row["suite"], "protocol": row["protocol"],
                                     "title": row["title"], "metrics": names, "rows": []})
        if row["title"] != table["title"] or names != table["metrics"]:
            sys.exit(f"{row['_dir']}: title or metric names differ from the other rows of protocol {row['protocol']}")
        table["rows"].append(row)
    for table in out.values():
        # The reference model's rows all carry reference: true; its newest run is the one compared with.
        refs = [r for r in table["rows"] if r.get("reference")]
        table["ref"] = max(refs, key=lambda r: (r["date"], r["run_id"]), default=None)
        table["rows"].sort(key=lambda r: (r is not table["ref"], r["family"], r["quant"], r["date"]))
    order = list(SUITES)
    return sorted(out.values(), key=lambda t: (t["machine"], order.index(t["suite"]), t["protocol"]))


_FRACTION = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")
_NUMBER = re.compile(r"^\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*([^\d\s(,./][^(]*?)?\s*$")


def fraction(value: str) -> tuple[int, int] | None:
    m = _FRACTION.match(value)
    return (int(m.group(1)), int(m.group(2))) if m and int(m.group(2)) > 0 else None


def number(value: str) -> tuple[float, str] | None:
    m = _NUMBER.match(value)
    return (float(m.group(1).replace(",", "")), (m.group(2) or "").strip()) if m else None


def cell(value: str, ref: str | None) -> str:
    """A value, with a bar for a pass count (a/b) and, next to the reference row, its ratio to it."""
    text = f"<span class=v>{e(value)}</span>"
    frac = fraction(value)
    if frac:
        pct = 100 * frac[0] / frac[1]
        return f"<td class=num>{text}<span class=bar><i style=\"width:{pct:.1f}%\"></i></span></td>"
    num, base = number(value), number(ref) if ref else None
    if num and base and base[0] and num[1] == base[1]:
        return f"<td class=num>{text}<span class=ratio>{num[0] / base[0]:.2f}&times; ref</span></td>"
    return f"<td class=num>{text}</td>"


SYSTEM_LABELS = {"macos_build": "macOS", "xcode_build": "Xcode"}


def results_html(ts: list[dict], repo: str) -> str:
    if not ts:
        return "<p class=muted>No results yet.</p>"
    parts = []
    machine = suite = None
    for t in ts:
        if t["machine"] != machine:
            machine, suite = t["machine"], None
            parts.append(f"<h3 class=machine>Machine <code>{e(machine)}</code></h3>")
        if t["suite"] != suite:
            suite = t["suite"]
            parts.append(f"<h4 class=suite-head><span class=\"dot {suite}\"></span>{e(SUITES[suite])}</h4>")
        ref = t["ref"]
        head = "".join(f"<th scope=col>{e(m)}</th>" for m in t["metrics"])
        body = []
        for r in t["rows"]:
            tag = " <span class=tag>reference</span>" if r is ref else ""
            refs = [m["value"] for m in ref["metrics"]] if ref and r is not ref else [None] * len(r["metrics"])
            cells = "".join(cell(m["value"], rv) for m, rv in zip(r["metrics"], refs))
            run = f"<a href=\"{e(repo)}/tree/main/{e(urllib.parse.quote(r['_dir']))}\">{e(r['run_id'])}</a>"
            if r.get("_details"):
                run += f" · <a href=\"{e(repo)}/blob/main/{e(urllib.parse.quote(r['_details']))}\">details</a>"
            body.append(f"<tr><th scope=row>{e(r['model'])}{tag}</th>{cells}<td>{e(r['date'])}</td><td>{run}</td></tr>")
        # Notes go below the table, where they wrap; a note every row shares is shown once.
        noted = [r for r in t["rows"] if r.get("notes")]
        if len(noted) == len(t["rows"]) > 1 and len({r["notes"] for r in noted}) == 1:
            notes = f"<p class=muted>{e(noted[0]['notes'])}</p>"
        else:
            models = [r["model"] for r in t["rows"]]
            label = lambda r: f"{r['model']} ({r['run_id']})" if models.count(r["model"]) > 1 else r["model"]
            notes = "".join(f"<p class=muted>{e(label(r))}: {e(r['notes'])}</p>" for r in noted)
        systems = []
        for r in t["rows"]:
            s = ", ".join(f"{SYSTEM_LABELS.get(k, k.replace('_', ' '))} {v}" for k, v in r["system"].items()
                          if v and k != "machine")
            if s not in systems:
                systems.append(s)
        parts.append(
            f"<figure class=result><figcaption><span class=title>{e(t['title'])}</span>"
            f"<code class=proto>{e(t['protocol'])}</code></figcaption>"
            f"<div class=scroll><table><thead><tr><th scope=col>Model</th>{head}<th scope=col>Date</th>"
            f"<th scope=col>Run</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"
            f"{notes}<p class=muted>{e('; '.join(systems))}</p></figure>")
    return "\n".join(parts)


# ---- the architecture diagram ------------------------------------------------------------------------

def box(x: int, y: int, w: int, h: int, cls: str, title: str, lines: tuple[str, ...] = ()) -> str:
    out = [f"<g class=\"node {cls}\"><rect x={x} y={y} width={w} height={h} rx=10></rect>",
           f"<text x={x + w / 2:g} y={y + 24} class=t>{e(title)}</text>"]
    for i, line in enumerate(lines):
        out.append(f"<text x={x + w / 2:g} y={y + 44 + 17 * i} class=s>{e(line)}</text>")
    return "".join(out) + "</g>"


def arrow(points: list[tuple[int, int]], label: str = "", at: tuple[int, int] | None = None, cls: str = "",
          label_cls: str = "") -> str:
    path = " ".join(f"{x},{y}" for x, y in points)
    out = f"<polyline class=\"edge {cls}\" points=\"{path}\" marker-end=\"url(#arrowhead)\"></polyline>"
    if label and at:
        out += f"<text x={at[0]} y={at[1]} class=\"el {label_cls}\">{e(label)}</text>"
    return out


DESC = ("The backend runs the speed leg and, as its capability leg, the agent_ability tasks. Those tasks, with "
        "the same prompts, go to three agent harnesses (Claude Code, Hermes Agent, localmelo), which use their "
        "built-in tools; the memory suite will drive the harnesses one session a day (planned) with the pab MCP "
        "server. Every model call from a harness goes through the recorder proxy to the oMLX server, and so do the "
        "memory suite's T4 reader calls, while the backend's speed probes go straight to the server. The backend's "
        "modeleval publish writes one summary.json row per suite and run to docs/benchmark, and "
        "scripts/build_docs.py builds this page from them.")

DEFS = ("<defs><marker id=arrowhead viewBox=\"0 0 10 10\" refX=9 refY=5 markerWidth=7 markerHeight=7 "
        "orient=auto-start-reverse><path d=\"M0,0 L10,5 L0,10 z\" class=ah></path></marker></defs>")


def diagram_wide() -> str:
    parts = [
        "<rect x=10 y=52 width=250 height=330 rx=14 class=group></rect>",
        "<text x=24 y=74 class=gl>eval/ suites</text>",
        box(26, 86, 218, 84, "c-backend", "backend", ("one command per model:", "speed leg + capability leg")),
        box(26, 186, 218, 70, "c-ability", "agent_ability", ("MCPMark + Aider tasks,", "objective checkers")),
        box(26, 272, 218, 96, "c-memory", "memory", ("T4: a 10-day timeline,", "pab MCP server,", "fixed reader + oracle")),
        box(350, 52, 210, 82, "c-tools", "Tools", ("built-in; MCP: pab (T4, planned),", "filesystem (optional arm)")),
        box(350, 166, 210, 112, "c-harness", "Agent harnesses",
            ("Claude Code · Hermes Agent", "· localmelo", "driven by bench/harness")),
        box(620, 166, 200, 112, "c-proxy", "Recorder proxy", ("bench/proxy: counts steps", "and model time; sets", "sampling and thinking")),
        box(875, 166, 195, 112, "c-server", "oMLX server", ("one model, fixed settings,", "on an Apple-silicon Mac")),
        box(350, 410, 470, 62, "c-results", "docs/benchmark/<machine>/<family>/<quant>/…",
            ("one summary.json row per suite and run",)),
        box(875, 410, 195, 62, "c-page", "This page", ("scripts/build_docs.py",)),
        arrow([(135, 86), (135, 30), (972, 30), (972, 162)], "speed probes", (560, 24)),
        arrow([(135, 170), (135, 182)], "", None),
        arrow([(244, 221), (346, 221)], "same prompts", (304, 212)),
        arrow([(244, 300), (310, 300), (310, 252), (346, 252)], "a session a day (planned)", (330, 318), "planned"),
        arrow([(455, 166), (455, 138)], "tool calls", (497, 156)),
        arrow([(560, 222), (616, 222)], "every call", (588, 212)),
        arrow([(820, 222), (871, 222)], "", None),
        arrow([(244, 350), (720, 350), (720, 282)], "reader calls (T4)", (500, 342)),
        arrow([(26, 128), (18, 128), (18, 441), (346, 441)], "modeleval publish", (190, 432)),
        arrow([(820, 441), (871, 441)], "", None),
    ]
    return (f"<svg class=\"diagram wide\" viewBox=\"0 0 1080 490\" role=img aria-labelledby=dgw "
            f"aria-describedby=dgw-d><title id=dgw>How the pipeline fits together</title>"
            f"<desc id=dgw-d>{e(DESC)}</desc>{''.join(parts)}</svg>")


def diagram_narrow() -> str:
    parts = [
        "<rect x=30 y=8 width=300 height=276 rx=14 class=group></rect>",
        "<text x=44 y=30 class=gl>eval/ suites</text>",
        box(44, 42, 272, 72, "c-backend", "backend", ("one command per model:", "speed leg + capability leg")),
        box(44, 124, 272, 70, "c-ability", "agent_ability", ("MCPMark + Aider, checkers",)),
        box(44, 204, 272, 70, "c-memory", "memory", ("T4 timeline, pab, reader, oracle",)),
        arrow([(180, 284), (180, 318)], "tasks; T4 sessions planned", (190, 305), "", "l"),
        box(44, 322, 272, 88, "c-harness", "Agent harnesses",
            ("Claude Code · Hermes · localmelo", "built-in tools; MCP: pab (T4, planned),",
             "filesystem (optional arm)")),
        arrow([(180, 410), (180, 444)], "every model call", (190, 431), "", "l"),
        box(44, 448, 272, 72, "c-proxy", "Recorder proxy", ("counts steps and time,", "sets sampling and thinking")),
        arrow([(180, 520), (180, 554)], "", None),
        box(44, 558, 272, 72, "c-server", "oMLX server", ("one model, fixed settings",)),
        arrow([(316, 78), (346, 78), (346, 594), (320, 594)], "", None),
        "<text x=334 y=340 class=el transform=\"rotate(90 334 340)\">speed probes</text>",
        box(44, 672, 272, 72, "c-results", "docs/benchmark/…/summary.json", ("one row per suite and run",)),
        arrow([(44, 78), (14, 78), (14, 708), (40, 708)], "", None),
        "<text x=24 y=480 class=el transform=\"rotate(-90 24 480)\">modeleval publish</text>",
        arrow([(180, 744), (180, 778)], "", None),
        box(44, 782, 272, 60, "c-page", "This page", ("scripts/build_docs.py",)),
    ]
    return (f"<svg class=\"diagram narrow\" viewBox=\"0 0 360 852\" role=img aria-labelledby=dgn "
            f"aria-describedby=dgn-d><title id=dgn>How the pipeline fits together</title>"
            f"<desc id=dgn-d>{e(DESC)}</desc>{''.join(parts)}</svg>")


# ---- the page ------------------------------------------------------------------------------------------

def page(ts: list[dict]) -> str:
    repo = REPO_URL
    n_rows = sum(len(t["rows"]) for t in ts)
    models = sorted({r["model"] for t in ts for r in t["rows"]})
    machines = sorted({t["machine"] for t in ts})
    latest = max((r["date"] for t in ts for r in t["rows"]), default="")
    stats = (f"<p class=stats><b>{len(models)}</b> model builds · <b>{n_rows}</b> results · "
             f"<b>{len(machines)}</b> machine{'s' if len(machines) != 1 else ''}"
             + (f" · latest {e(latest)}" if latest else "") + "</p>")
    diagram = (f"<svg class=defs width=0 height=0 aria-hidden=true focusable=false>{DEFS}</svg>"
               f"<div class=dwide>{diagram_wide()}</div>{diagram_narrow()}")
    parts = {"REPO": e(repo), "STATS": stats, "RESULTS": results_html(ts, repo), "DIAGRAM": diagram}
    # One pass, so text from a summary.json is never scanned for placeholders.
    return HEAD + re.sub(r"\{\{(\w+)\}\}", lambda m: parts[m.group(1)], BODY)


HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal agent eval</title>
<meta name="description" content="Local models as the backend of a personal agent on Apple-silicon Macs: serving speed, agent ability through three harnesses, and memory.">
<style>
:root {
  color-scheme: light;
  --bg: #ffffff; --bg2: #f6f8fa; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --card: #ffffff;
  --link: #0969da; --tag: #ddf4ff; --tagfg: #0550ae; --bar: #2da44e; --barbg: #e8ecef; --shadow: 0 1px 2px rgba(31,35,40,.06);
  --c-backend: #0969da; --c-ability: #8250df; --c-memory: #bf3989; --c-harness: #6e40c9; --c-tools: #57606a;
  --c-proxy: #9a6700; --c-server: #1a7f37; --c-results: #57606a; --c-page: #0969da;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0d1117; --bg2: #151b23; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --card: #151b23;
    --link: #4493f8; --tag: #12263f; --tagfg: #79c0ff; --bar: #3fb950; --barbg: #262c36; --shadow: none;
    --c-backend: #4493f8; --c-ability: #ab7df8; --c-memory: #f778ba; --c-harness: #a371f7; --c-tools: #9198a1;
    --c-proxy: #d29922; --c-server: #3fb950; --c-results: #9198a1; --c-page: #4493f8;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d1117; --bg2: #151b23; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --card: #151b23;
  --link: #4493f8; --tag: #12263f; --tagfg: #79c0ff; --bar: #3fb950; --barbg: #262c36; --shadow: none;
  --c-backend: #4493f8; --c-ability: #ab7df8; --c-memory: #f778ba; --c-harness: #a371f7; --c-tools: #9198a1;
  --c-proxy: #d29922; --c-server: #3fb950; --c-results: #9198a1; --c-page: #4493f8;
}
* { box-sizing: border-box; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
a { color: var(--link); text-decoration: none; }
a:hover, a:focus-visible { text-decoration: underline; }
.intro a, footer a { text-decoration: underline; text-underline-offset: 2px; }
code { font: 0.86em ui-monospace, SFMono-Regular, Menlo, monospace; }
p code, dd code, li code { overflow-wrap: anywhere; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 16px; }
header.top { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line); }
header.top .wrap { display: flex; align-items: center; gap: 18px; height: 52px; overflow-x: auto; white-space: nowrap; }
header.top .brand { font-weight: 650; color: var(--fg); margin-right: auto; }
header.top nav a { color: var(--muted); font-size: 14px; margin-left: 14px; }
header.top nav a:hover { color: var(--fg); }
.hero { padding: 48px 0 24px; }
.hero h1 { font-size: clamp(30px, 5vw, 44px); line-height: 1.15; margin: 0 0 12px; letter-spacing: -0.02em; }
.hero .lead { font-size: 18px; color: var(--muted); max-width: 760px; margin: 0; }
.stats { color: var(--muted); font-size: 14px; margin: 16px 0 0; }
.stats b { color: var(--fg); }
.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 28px 0 8px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px; box-shadow: var(--shadow); }
.card h2 { margin: 0 0 6px; font-size: 17px; display: flex; align-items: center; gap: 8px; }
.card p { margin: 0; color: var(--muted); font-size: 15px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex: none; }
.dot.backend { background: var(--c-backend); } .dot.agent_ability { background: var(--c-ability); }
.dot.memory { background: var(--c-memory); }
section { padding: 28px 0; border-top: 1px solid var(--line); scroll-margin-top: 52px; }
section > .wrap > h2 { font-size: 26px; margin: 8px 0 8px; letter-spacing: -0.01em; }
.intro { color: var(--muted); max-width: 800px; margin: 0 0 8px; }
h3.machine { font-size: 18px; margin: 28px 0 4px; }
h3.machine code { font-size: 15px; background: var(--bg2); padding: 2px 6px; border-radius: 6px; }
h4.suite-head { display: flex; align-items: center; gap: 8px; font-size: 16px; margin: 22px 0 6px; }
figure.result { margin: 10px 0 22px; }
figcaption { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px; margin-bottom: 8px; font-weight: 600; }
figcaption .proto { font-weight: 400; color: var(--muted); background: var(--bg2); padding: 1px 6px; border-radius: 6px; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 8px 10px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { background: var(--bg2); font-weight: 600; color: var(--muted); font-size: 13px; white-space: normal; min-width: 72px; vertical-align: bottom; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: 0; }
tbody th { font-weight: 600; }
td.num .v { font-variant-numeric: tabular-nums; }
.bar { display: block; height: 4px; margin-top: 5px; min-width: 56px; background: var(--barbg); border-radius: 2px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--bar); }
.ratio { display: block; font-size: 12px; color: var(--muted); }
tbody th, thead th:first-child { position: sticky; left: 0; z-index: 1; }
tbody th { background: var(--bg); }
.tag { font-size: 11px; font-weight: 600; background: var(--tag); color: var(--tagfg); border-radius: 999px; padding: 1px 7px; margin-left: 4px; }
.muted { color: var(--muted); font-size: 13px; margin: 6px 0 0; }
.diagram { width: 100%; height: auto; margin: 12px 0 4px; }
.diagram.narrow { display: none; max-width: 420px; }
.dwide { overflow-x: auto; }
svg.defs { position: absolute; }
.diagram .group { fill: var(--bg2); stroke: var(--line); }
.diagram .gl { fill: var(--muted); font-size: 12px; font-weight: 600; }
.diagram .node rect { fill: var(--card); stroke-width: 1.6; }
.diagram .node .t { font-size: 15px; font-weight: 650; text-anchor: middle; fill: var(--fg); }
.diagram .node .s { font-size: 12.5px; text-anchor: middle; fill: var(--muted); }
.diagram .edge { fill: none; stroke: var(--muted); stroke-width: 1.4; }
.ah { fill: var(--muted); }
.diagram .el { font-size: 11.5px; fill: var(--muted); text-anchor: middle; }
.diagram .el.l { text-anchor: start; }
.diagram .edge.planned { stroke-dasharray: 5 4; }
.c-backend rect { stroke: var(--c-backend); } .c-ability rect { stroke: var(--c-ability); }
.c-memory rect { stroke: var(--c-memory); } .c-harness rect { stroke: var(--c-harness); }
.c-tools rect { stroke: var(--c-tools); } .c-proxy rect { stroke: var(--c-proxy); }
.c-server rect { stroke: var(--c-server); } .c-results rect { stroke: var(--c-results); }
.c-page rect { stroke: var(--c-page); }
.points { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 28px; margin: 18px 0 0; padding: 0; list-style: none; }
.points li { padding-left: 14px; border-left: 3px solid var(--line); }
.points b { display: block; margin-bottom: 2px; }
.points span { color: var(--muted); font-size: 15px; }
.suite { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 8px 28px; padding: 18px 0; border-bottom: 1px solid var(--line); }
.suite:last-child { border-bottom: 0; }
.suite h3 { margin: 0; font-size: 18px; display: flex; align-items: center; gap: 8px; }
.suite .where { color: var(--muted); font-size: 13px; margin-top: 4px; }
.suite dl { margin: 0; display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 6px 16px; }
.suite dt { color: var(--muted); font-size: 14px; }
.suite dd { margin: 0; font-size: 15px; }
pre { background: var(--bg2); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; overflow-x: auto;
      font: 13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
pre .c { color: var(--muted); }
ul.todo { color: var(--muted); max-width: 800px; margin: 0 0 8px; padding-left: 20px; }
ul.todo li { margin: 4px 0; }
footer { border-top: 1px solid var(--line); padding: 24px 0 40px; color: var(--muted); font-size: 14px; }
@media (max-width: 860px) {
  .cards { grid-template-columns: 1fr; }
  .points { grid-template-columns: 1fr; }
  .suite { grid-template-columns: 1fr; }
  .suite dl { grid-template-columns: 1fr; gap: 2px 0; }
  .suite dd { margin-bottom: 8px; }
}
@media (min-width: 641px) and (max-width: 960px) {
  .diagram.wide { min-width: 900px; }
}
@media (max-width: 640px) {
  .dwide, .diagram.wide { display: none; }
  .diagram.narrow { display: block; margin: 12px auto 4px; }
  header.top nav a { margin-left: 10px; }
  .hero { padding-top: 32px; }
  tbody th { white-space: normal; min-width: 136px; }
}
@media (max-width: 520px) {
  header.top .brand { display: none; }
  header.top nav a { margin-left: 0; margin-right: 14px; }
}
</style>
</head>
"""

BODY = """<body>
<header class=top><div class=wrap>
<a class=brand href="#">Personal agent eval</a>
<nav><a href="#results">Results</a><a href="#how">How it works</a><a href="#suites">Suites</a><a href="#run">Run it</a><a href="{{REPO}}">GitHub</a></nav>
</div></header>

<main>
<div class=hero><div class=wrap>
<h1>Local models as a personal agent's backend</h1>
<p class=lead>How well does a local model serve a personal agent on a Mac? Three suites measure it under fixed,
versioned settings (the memory suite is still being built), so a new model or quantization lands next to the ones
already measured, and anyone with an Apple-silicon Mac can rerun the backend's speed and capability legs.</p>
{{STATS}}
<div class=cards>
<div class=card><h2><span class="dot backend"></span>Backend</h2><p>What an agent waits for: time to first token with a
cached or a cold prompt, decode speed from 16K to 110K tokens of context, a 20-step agent trajectory, and the
server's memory. Plus a functional gate, and tool-call accuracy (shown with agent ability).</p></div>
<div class=card><h2><span class="dot agent_ability"></span>Agent ability</h2><p>Whether agents built on the model get
work done. The same tasks, with byte-identical prompts, run through Claude Code, Hermes Agent and localmelo, and
objective checkers score them.</p></div>
<div class=card><h2><span class="dot memory"></span>Memory</h2><p>Whether an agent keeps a correct, current and compact
memory. A synthetic 10-day timeline is fed to the agent one session a day, and a fixed reader then answers 60
questions from the memory store alone. The reader and the scoring are built; the session runner is next.</p></div>
</div>
</div></div>

<section id=results><div class=wrap>
<h2>Results</h2>
<p class=intro>Rows share a table only when they come from the same machine and the same protocol. A protocol fixes
the tasks, budgets, sampling rule (each model's recommended sampling), thinking setting and server settings; a
change to any of them starts a new table.
A bar shows a pass count; "&times; ref" compares a value with the reference model's in the same table.</p>
{{RESULTS}}
</div></section>

<section id=how><div class=wrap>
<h2>How it works</h2>
<p class=intro>Each suite drives a model through the same pieces. The arrows follow the model calls and, at the
bottom, the results.</p>
{{DIAGRAM}}
<ul class=points>
<li><b>One recorder proxy for every harness</b><span>Every model call of an agent harness, and every T4 reader call,
goes through bench/proxy. It counts agent steps and model time the same way for Claude Code, Hermes Agent and
localmelo, sets the thinking setting, and gives the harnesses' requests the model's recommended sampling. The
backend's speed probes call the server directly, with the same sampling, so their timings carry no proxy
overhead.</span></li>
<li><b>Fixed, versioned protocols</b><span>Tasks, budgets, the sampling rule, thinking (off in the main arm) and
server settings are fixed per protocol. Every result names its protocol and machine, and no table mixes
them.</span></li>
<li><b>Quiet machine for timed work</b><span>The backend's timed units wait until other processes' CPU and GPU load is
below fixed thresholds and the battery is high enough. A unit with a major disturbance is repeated, up to 3
attempts; a minor one is kept and flagged.</span></li>
<li><b>Recorded results</b><span>Each result records the macOS and Xcode builds, the server version and the machine.
A backend run publishes its manifest and the speed leg's raw records (timings, machine samples, and the model's
output on synthetic prompts); the capability leg publishes only allowlisted files (per-call metrics and checker
output, no request or response bodies). Agent transcripts, harness homes and server logs stay on the
machine.</span></li>
</ul>
</div></section>

<section id=suites><div class=wrap>
<h2>Suites</h2>
<div class=suite><div><h3><span class="dot backend"></span>Backend</h3><div class=where><a href="{{REPO}}/tree/main/eval/backend">eval/backend</a></div></div>
<dl>
<dt>One run</dt><dd>Once the checkpoint is prepared (download, convert, register), one command per model starts the server with the fixed settings and runs the speed leg, then the capability leg on the same server launch. <code>modeleval publish</code> writes the rows.</dd>
<dt>Speed leg</dt><dd>Time to first token on a cached ~15K-token agent prefix and on a cold 64K-token prompt; decode speed at 16K, 32K, 64K and 110K tokens of context; a 20-step agent trajectory; peak server memory. Before the timed units, an untimed functional gate and 22 tool-call cases (shown in the Agent ability table).</dd>
<dt>Capability leg</dt><dd>The agent_ability tasks through all three harnesses, one run each, with a budget of 30 agent steps (successful model calls that carry tools), 40 harness turns and 3600 s per task. A gate set first (12 Aider exercises and 10 MCPMark tasks); the ranking set (the other 22 Aider Python exercises) runs unless the gate finds the model worse than the reference. The decision rules are fixed in advance: the gate calls a model worse only when it loses at least 5 more tasks than it wins, consistently in at least two harnesses; the ranking set calls it better or worse only at sign-test p &lt; 0.05 over tasks.</dd>
</dl></div>
<div class=suite><div><h3><span class="dot agent_ability"></span>Agent ability</h3><div class=where><a href="{{REPO}}/tree/main/eval/agent_ability">eval/agent_ability</a></div></div>
<dl>
<dt>Tasks</dt><dd>MCPMark filesystem tasks (easy) and Aider polyglot Python exercises, at pinned commits.</dd>
<dt>Harnesses</dt><dd>Claude Code, Hermes Agent and localmelo, each run headless on the same prompts, through the recorder proxy.</dd>
<dt>Scoring</dt><dd>The benchmark's own checkers, run outside the agent's workspace: MCPMark's verifiers, and the Aider unit tests restored to their original form.</dd>
</dl></div>
<div class=suite><div><h3><span class="dot memory"></span>Memory</h3><div class=where><a href="{{REPO}}/tree/main/eval/memory">eval/memory</a></div></div>
<dl>
<dt>T4 timeline</dt><dd>86 synthetic messages, emails and calendar entries over 10 days, with updates, relative dates, facts that only combine across days, and distractors.</dd>
<dt>Probes</dt><dd>60 questions, 12 each: extraction, cross-session, temporal, knowledge update, abstention. A fixed reader answers them from the memory store alone, with thinking off and temperature 0.</dd>
<dt>Store checks</dt><dd>Fabricated sources, source precision, duplicates and the store's size in tokens.</dd>
<dt>Status</dt><dd>Fixtures, the pab MCP server, the reader and the oracle are built; the reader answers 48/48 answerable and 12/12 abstention probes from the ideal store. The runner that drives the harnesses through the 10 days comes next.</dd>
</dl></div>
</div></section>

<section id=run><div class=wrap>
<h2>Run it on your Mac</h2>
<p class=intro>You need an Apple-silicon Mac, <a href="https://docs.astral.sh/uv/">uv</a>, and an
<a href="https://github.com/jundot/omlx">oMLX</a> 0.7.1rc1 checkout (commit 7a5a3d75) with its own .venv, the
protocol's server. The capability leg also needs the three harnesses, at the versions in
<a href="{{REPO}}/blob/main/eval/backend/pins.toml">eval/backend/pins.toml</a>, and the pinned task data
(<a href="{{REPO}}/tree/main/eval/agent_ability#data">eval/agent_ability</a>).
<a href="{{REPO}}/tree/main/eval/backend#on-another-mac">eval/backend/README.md</a> has the details.</p>
<pre><code>git clone {{REPO}}.git
cd personal_agent_eval_pipeline/eval/backend
<span class=c># once per Mac: put the paths that differ from config.defaults.toml in config.toml</span>
uv sync
uv sync --project envs/capability &amp;&amp; uv sync --project envs/agent-tools
uv run modeleval selftest       <span class=c># offline; prints the resolved paths, a FAIL names what is missing</span>
<span class=c># once per model: add it to models.toml, then prepare its checkpoint</span>
uv run modeleval download &lt;model&gt; &amp;&amp; uv run modeleval convert &lt;model&gt;   <span class=c># it has a prepare entry</span>
uv run modeleval register &lt;model&gt;   <span class=c># or: its checkpoint is already at its models.toml path</span>
(cd harness &amp;&amp; uv run agentbench fixtures build)   <span class=c># once: the speed prompts (Qwen3.8 tokenizer)</span>
uv run modeleval plan &lt;model&gt;
uv run modeleval run &lt;model&gt;    <span class=c># speed leg + capability leg</span>
uv run modeleval publish        <span class=c># rows under docs/benchmark/&lt;machine&gt;/...</span>
cd ../.. &amp;&amp; python3 scripts/build_docs.py   <span class=c># rebuild this page</span></code></pre>
<p class=intro>On a new machine, run the reference model (Qwen3.8-27B oQ5 + MTP) first, so that every other model
has something to be compared with. Three steps are not in these commands:</p>
<ul class=todo>
<li>The reference model has no prepare entry: download <code>Qwen/Qwen3.8-27B</code> at the revision in models.toml,
convert it at oQ level 5 with <code>eval/backend/scripts/oq_convert.py</code> (its docstring has the command), then
register it. The fixture build reads its tokenizer: set <code>AGENTBENCH_TOKENIZER_DIR</code> to the checkpoint
unless it is under <code>~/Developer/mlsys/models/oq/</code>.</li>
<li>The capability leg's harness epoch includes the workspace path and the agent configs under
<code>[paths] agent_root</code>, so on another Mac <code>run</code> refuses with the epoch id your inputs hash to.
Check the difference, then write that id into <code>eval/backend/pins.toml</code> (its header explains epochs).</li>
<li><code>publish</code> writes rows only under a known protocol: a different macOS or Xcode build, or a different
harness epoch, first needs its own protocol id in <code>eval/backend/src/modeleval/publish.py</code>; otherwise the
run is skipped.</li>
</ul>
</div></section>
</main>

<footer><div class=wrap>Generated by <code>scripts/build_docs.py</code> from <code>docs/benchmark</code>.
Source and documentation on <a href="{{REPO}}">GitHub</a>.</div></footer>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/index.html is out of date")
    args = ap.parse_args()
    new = page(tables(load()))
    if args.check:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != new:
            print("stale: docs/index.html (run python3 scripts/build_docs.py)")
            return 1
        return 0
    PAGE.write_text(new, encoding="utf-8")
    print(f"docs/index.html: {sum(1 for _ in BENCH.glob('*/*/*/*/*/summary.json'))} results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
