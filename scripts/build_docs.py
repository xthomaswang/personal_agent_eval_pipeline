"""Build the results page and the README's results block from docs/benchmark/**/summary.json.

    python3 scripts/build_docs.py           # write docs/index.html and the README block
    python3 scripts/build_docs.py --check   # exit 1 if either is out of date (for CI and pre-commit)

Standard library only. The format of summary.json is in docs/benchmark/README.md.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "docs" / "benchmark"
PAGE = ROOT / "docs" / "index.html"
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
SUITES = {"backend": "Backend", "agent_ability": "Agent ability", "memory": "Memory"}
REQUIRED = ("schema", "suite", "protocol", "title", "model", "family", "quant", "run_id", "date", "system", "metrics")
LEVELS = ("machine", "family", "quant", "suite", "run_id")


def load() -> list[dict]:
    rows = []
    for path in sorted(BENCH.glob("*/*/*/*/*/summary.json")):
        row = json.loads(path.read_text())
        where = path.relative_to(ROOT)
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            sys.exit(f"{where}: missing {', '.join(missing)}")
        if row["schema"] != 1 or row["suite"] not in SUITES:
            sys.exit(f"{where}: schema must be 1 and suite one of {', '.join(SUITES)}")
        found = dict(zip(LEVELS, path.parent.relative_to(BENCH).parts))
        want = {**{k: row.get(k) for k in LEVELS}, "machine": row["system"].get("machine")}
        if found != want:
            sys.exit(f"{where}: the path does not match system.machine/family/quant/suite/run_id")
        row["_dir"] = path.parent.relative_to(ROOT / "docs").as_posix()
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
        table["rows"].sort(key=lambda r: (not r.get("reference", False), r["family"], r["quant"], r["date"]))
    order = list(SUITES)
    return sorted(out.values(), key=lambda t: (t["machine"], order.index(t["suite"]), t["protocol"]))


def systems(table: dict) -> str:
    seen = []
    for row in table["rows"]:
        s = row["system"]
        text = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in s.items() if v and k != "machine")
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def page(ts: list[dict]) -> str:
    e = html.escape
    parts = [] if ts else ["<p class=note>No results yet.</p>"]
    machine = suite = None
    for t in ts:
        if t["machine"] != machine:
            machine, suite = t["machine"], None
            parts.append(f"<h2>Machine <code>{e(machine)}</code></h2>")
        if t["suite"] != suite:
            suite = t["suite"]
            parts.append(f"<h3>{e(SUITES[suite])}</h3>")
        head = "".join(f"<th>{e(m)}</th>" for m in t["metrics"])
        body = []
        for r in t["rows"]:
            ref = " <span class=tag>reference</span>" if r.get("reference") else ""
            cells = "".join(f"<td>{e(str(m['value']))}</td>" for m in r["metrics"])
            link = r.get("details") or "summary.json"
            body.append(f"<tr><td>{e(r['model'])}{ref}</td>{cells}<td>{e(r['date'])}</td>"
                        f"<td><a href=\"{e(r['_dir'])}/{e(link)}\">{e(r['run_id'])}</a></td></tr>")
            if r.get("notes"):
                body.append(f"<tr class=notes><td></td><td colspan={len(t['metrics']) + 2}>{e(r['notes'])}</td></tr>")
        parts.append(f"<h4>{e(t['title'])} <code>{e(t['protocol'])}</code></h4>"
                     f"<div class=scroll><table><thead><tr><th>Model</th>{head}<th>Date</th><th>Run</th></tr></thead>"
                     f"<tbody>{''.join(body)}</tbody></table></div><p class=note>{e(systems(t))}</p>")
    return TEMPLATE.replace("{{BODY}}", "\n".join(parts))


def markdown(ts: list[dict]) -> str:
    lines = [START, ""]
    for t in ts:
        lines += [f"**{SUITES[t['suite']]}, `{t['machine']}`: {t['title']}** (`{t['protocol']}`)", "",
                  "| Model | " + " | ".join(t["metrics"]) + " | Date |",
                  "|---|" + "---|" * (len(t["metrics"]) + 1)]
        for r in t["rows"]:
            ref = " (reference)" if r.get("reference") else ""
            vals = " | ".join(str(m["value"]) for m in r["metrics"])
            lines.append(f"| [{r['model']}{ref}](docs/{r['_dir']}/) | {vals} | {r['date']} |")
        lines.append("")
    if not ts:
        lines += ["No results yet.", ""]
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/index.html or the README block is stale")
    args = ap.parse_args()
    ts = tables(load())
    new_page = page(ts)
    readme = README.read_text()
    if START not in readme or END not in readme:
        sys.exit(f"README.md needs the markers {START} and {END}")
    head, rest = readme.split(START, 1)
    new_readme = head + markdown(ts) + rest.split(END, 1)[1]
    stale = [p for p, new in ((PAGE, new_page), (README, new_readme)) if not p.exists() or p.read_text() != new]
    if args.check:
        for p in stale:
            print(f"stale: {p.relative_to(ROOT)} (run python3 scripts/build_docs.py)")
        return 1 if stale else 0
    PAGE.write_text(new_page)
    README.write_text(new_readme)
    print(f"{sum(len(t['rows']) for t in ts)} rows in {len(ts)} tables")
    return 0


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal agent eval</title>
<style>
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --head: #f6f8fa; --tag: #ddf4ff; --link: #0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --head: #151b23; --tag: #12263f; --link: #4493f8; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
a { color: var(--link); }
h1 { margin: 0 0 4px; font-size: 26px; }
h2 { margin-top: 36px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
h3 { font-size: 18px; margin: 24px 0 8px; }
h4 { font-size: 15px; margin: 18px 0 8px; }
code { font-size: 13px; color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 14px; }
th, td { border: 1px solid var(--line); padding: 5px 10px; text-align: left; white-space: nowrap; }
th { background: var(--head); }
tr.notes td { border-top: 0; color: var(--muted); font-size: 13px; white-space: normal; }
.tag { background: var(--tag); border-radius: 4px; padding: 0 5px; font-size: 12px; }
.note { color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<h1>Personal agent eval</h1>
<p class=note>Local models as the backend of a personal agent on one Mac (M5 Max, 128 GB): serving speed, agent ability
through three harnesses, and memory. Generated from <code>docs/benchmark</code> by <code>scripts/build_docs.py</code>;
how it works is in the repository README.</p>
{{BODY}}
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
