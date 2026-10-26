"""Second look at a harness-smoke run: is each harness actually working, beyond pass/fail?

Usage: python3 -I scripts/analyze_run.py runs/<ts>

Per run: the verdict, the proxy's counts next to the harness's own count, budget hits, upstream
errors, refused commands (Claude Code permission_denials; Hermes tool results that read like a
refusal), signs that the agent's shell found the wrong Python, and pip attempts.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

WRONG_PYTHON = re.compile(r"/usr/bin/python3|No module named pytest|python3\.9|Python 3\.9")
PIP = re.compile(r"\bpip3?\s+install\b|-m\s+pip\b")
# Hermes's one-shot approval policy (approvals.single_query_mode: deny) answers a flagged command with this
REFUSAL = re.compile(r"BLOCKED: Command flagged as dangerous \(([^)]*)\)")


def events(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


def tool_commands(harness: str, run: Path) -> tuple[list[str], list[str]]:
    """(tool inputs as text, tool outputs as text) from the harness's own log."""
    inputs, outputs = [], []
    evs = events(run / "stdout.log")
    if harness == "claude_code":
        for e in evs:
            content = (e.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use":
                    inputs.append(f"{c.get('name')}: {json.dumps(c.get('input'))}")
                elif c.get("type") == "tool_result":
                    outputs.append(json.dumps(c.get("content")))
    elif harness == "hermes":
        for e in evs:
            if e.get("type") == "tool_use":
                inputs.append(f"{e.get('name')}: {e.get('input')}")
            elif e.get("type") == "tool_result":
                outputs.append(str(e.get("output")))
    elif harness == "localmelo":
        db = run / "mem" / "history.db"
        if db.exists():
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    cols = [r[1] for r in conn.execute("PRAGMA table_info(steps)")]
                    for row in conn.execute("SELECT * FROM steps"):
                        text = json.dumps(dict(zip(cols, row)), default=str)
                        inputs.append(text)
                        outputs.append(text)
            except sqlite3.Error as exc:
                inputs.append(f"history.db unreadable: {exc}")
    return inputs, outputs


def native_count(harness: str, report: dict) -> int | None:
    key = {"claude_code": "num_turns", "hermes": "api_call_count", "localmelo": "steps"}.get(harness)
    return report.get(key) if key else None


def calls_info(run: Path) -> dict:
    info = {"budget_hits": 0, "upstream_errors": 0, "stop_reasons": {}, "tools_used": {}}
    path = run / "calls.jsonl"
    if not path.exists():
        return info
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("role") != "main":
            continue
        error = record.get("error") or {}
        if error.get("type") == "bench_budget_exhausted":
            info["budget_hits"] += 1
        elif error:
            info["upstream_errors"] += 1
        reason = record.get("stop_reason")
        info["stop_reasons"][reason] = info["stop_reasons"].get(reason, 0) + 1
        for name in record.get("tool_names") or []:
            info["tools_used"][name] = info["tools_used"].get(name, 0) + 1
    return info


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    doc = json.loads((root / "results.json").read_text())
    rows = []
    for r in doc["records"]:
        run = root / r["run_dir"]
        inputs, outputs = tool_commands(r["harness"], run)
        report = r.get("harness_report") or {}
        native = native_count(r["harness"], report)
        info = calls_info(run)
        stdout = (run / "stdout.log").read_text(errors="replace") if (run / "stdout.log").exists() else ""
        flags = []
        if r["harness"] == "claude_code" and report.get("permission_denials"):
            flags.append(f"denials={report['permission_denials']}")
        refusals = [m.group(1) for o in outputs for m in [REFUSAL.search(o)] if m] if r["harness"] == "hermes" else []
        if refusals:
            flags.append(f"blocked={len(refusals)} ({', '.join(sorted(set(refusals)))})")
        if (r.get("check") or {}).get("tests_modified"):
            flags.append("tests modified")
        if r["kind"] == "aider" and WRONG_PYTHON.search(stdout + "\n".join(outputs)):
            flags.append("wrong-python?")
        pips = [i for i in inputs if PIP.search(i)]
        if pips:
            flags.append(f"pip attempts={len(pips)}")
        if info["budget_hits"]:
            flags.append("budget hit")
        if info["upstream_errors"]:
            flags.append(f"upstream errors={info['upstream_errors']}")
        # localmelo's steps table has one row per tool call, so it pairs with n_tool, not n_llm
        proxy_key = "n_tool" if r["harness"] == "localmelo" else "n_llm"
        if native is not None and r[proxy_key] is not None and native != r[proxy_key]:
            flags.append(f"native {native} != {proxy_key} {r[proxy_key]}")
        rows.append({**r, "native": native, "flags": flags, "tools_used": info["tools_used"],
                     "stop_reasons": info["stop_reasons"], "refusal_samples": refusals[:2], "pip_samples": pips[:2]})

    print(f"# {root.name}: {len(rows)} runs; finished={doc.get('finished')}")
    for r in rows:
        tag = "" if r["scored"] else " (unscored)"
        print(f"{r['harness']:11s} {r['task']:44s}{tag:11s} {r['verdict']:13s} wall={r['wall_s']} "
              f"n_llm={r['n_llm']} n_tool={r['n_tool']} native={r['native']} {'; '.join(r['flags'])}")
    print()
    for harness in dict.fromkeys(r["harness"] for r in rows):
        scored = [r for r in rows if r["harness"] == harness and r["scored"]]
        if not scored:
            continue
        verdicts = {v: sum(1 for r in scored if r["verdict"] == v) for v in ("pass", "fail", "timeout", "harness_error")}
        walls = [r["wall_s"] for r in scored if r["wall_s"] is not None]
        llms = [r["n_llm"] for r in scored if r["n_llm"] is not None]
        by_kind = {}
        for r in scored:
            k = by_kind.setdefault(r["kind"], [0, 0])
            k[0] += r["verdict"] == "pass"
            k[1] += 1
        tools = {}
        for r in scored:
            for name, n in r["tools_used"].items():
                tools[name] = tools.get(name, 0) + n
        print(f"{harness}: {verdicts}; by kind {{{', '.join(f'{k}: {p}/{n}' for k, (p, n) in by_kind.items())}}}; "
              f"median wall {statistics.median(walls) if walls else None} s; median n_llm {statistics.median(llms) if llms else None}; "
              f"flagged runs {sum(1 for r in scored if r['flags'])}")
        print(f"  tools: {dict(sorted(tools.items(), key=lambda kv: -kv[1]))}")
        for r in scored:
            for sample in r["refusal_samples"] + r["pip_samples"]:
                print(f"  {r['task']}: {sample[:220]}")


if __name__ == "__main__":
    main()
