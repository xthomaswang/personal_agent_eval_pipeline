"""results.json and results.md for one suite run, rewritten after every task so partial runs keep."""

from __future__ import annotations

import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

VERDICTS = ("pass", "fail", "timeout", "harness_error")


def native_steps(harness: str, report: dict[str, Any]) -> str:
    """Each harness's own step count, as bench.harness parsed it from the harness's logs."""
    if harness == "claude_code":
        return f"turns={report.get('num_turns')}"
    if harness == "hermes":
        return f"api_calls={report.get('api_call_count')} tool_use={report.get('tool_use_events')}"
    if harness == "localmelo":
        return f"steps={report.get('steps')}"
    return json.dumps(report, sort_keys=True) if report else ""


def totals(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record["scored"]:
            continue
        row = out.setdefault(record["harness"], {"scored": 0, **dict.fromkeys(VERDICTS, 0), "wall_s": 0.0})
        row["scored"] += 1
        row[record["verdict"]] += 1
        row["wall_s"] = round(row["wall_s"] + (record["wall_s"] or 0.0), 3)
    return out


def _notes(record: dict[str, Any]) -> str:
    notes = []
    check = record["check"]
    if record["error"]:
        notes.append(record["error"])
    if check.get("tests_modified"):
        notes.append("tests modified")
    if check.get("missing_solution_files"):
        notes.append("solution missing")
    if record["exit_code"] not in (0, None):
        notes.append(f"exit {record['exit_code']}")
    if record["timed_out"] and check.get("passed"):
        notes.append("check passed after the kill")
    if not check.get("passed") and check.get("detail"):
        notes.append(f"check: {check['detail']}")
    return "; ".join(notes).replace("|", "/")[:160]


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def write(run_root: Path, header: dict[str, Any], records: list[dict[str, Any]]) -> None:
    doc = {**header, "records": records, "totals": totals(records)}
    (run_root / "results.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    settings = header["settings"]
    passes = settings.get("repeat") or 1
    localmelo_args = shlex.join(settings.get("localmelo_args") or [])
    lines = [
        f"# Harness smoke results: {run_root.name}",
        "",
        f"- Arm: {'MCP-only (filesystem MCP server)' if header['arm'] == 'mcp' else 'built-in tools'}; "
        f"thinking {settings['thinking']}{'; a random seed per request' if settings.get('random_seed') else ''}"
        f"{'; one reasoning field per replayed message' if settings.get('one_reasoning_field') else ''}"
        f"{'; request and response bodies logged' if settings.get('log_bodies') else ''}"
        f"{f'; localmelo flags `{localmelo_args}`' if localmelo_args else ''}"
        f"{f'; {passes} passes over the tasks' if passes > 1 else ''}; "
        f"budgets {settings['max_steps']} proxy steps, "
        f"{settings['max_turns']} harness turns, {settings['timeout_s']:g} s per task",
        f"- Data: polyglot-benchmark@{header['data']['polyglot_sha'][:7]}, mcpmark@{header['data']['mcpmark_sha'][:7]}, "
        f"start zips per manifest.json (sha256 {header['data']['zip_manifest_sha256'][:12]})",
        f"- Started {header['started_at']}; updated {header['updated_at']}; finished: {'yes' if header['finished'] else 'no'}",
    ]
    system = header.get("system") or {}
    if system:
        xcode = ", ".join((system.get("xcodebuild_version") or "unknown").splitlines())
        lines.append(f"- System: macOS {system.get('macos')} (build {system.get('macos_build')}); {xcode}")
    if header.get("not_runnable"):
        lines.append(f"- Not runnable in this arm: {', '.join(header['not_runnable'])}")
    lines += [
        "",
        "| Harness | Task | Verdict | wall_s | n_llm | n_tool | n_llm_aux | Native steps | Notes |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in records:
        task = r["task"] if r["scored"] else f"{r['task']} (unscored)"
        if passes > 1 and r["scored"]:
            task += f" #{r.get('repeat', 1)}"
        lines.append(
            f"| {r['harness']} | {task} | {r['verdict']} | {_cell(r['wall_s'])} | {_cell(r['n_llm'])} | "
            f"{_cell(r['n_tool'])} | {_cell(r['n_llm_aux'])} | {r['native_steps']} | {_notes(r)} |"
        )
    lines += [
        "",
        "## Totals (scored tasks)",
        "",
        "| Harness | Scored | Pass | Fail | Timeout | Harness error | wall (min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for harness, row in totals(records).items():
        lines.append(
            f"| {harness} | {row['scored']} | {row['pass']} | {row['fail']} | {row['timeout']} | "
            f"{row['harness_error']} | {row['wall_s'] / 60:.1f} |"
        )
    hashes: dict[str, set[str]] = {}
    for r in records:
        hashes.setdefault(r["task"], set()).add(r["prompt_sha256"])
    differing = sorted(task for task, seen in hashes.items() if len(seen) > 1)
    lines += [
        "",
        "Prompts byte-identical across harnesses: "
        + ("yes" if not differing else f"NO, they differ for {', '.join(differing)}"),
    ]
    (run_root / "results.md").write_text("\n".join(lines) + "\n")


def prompts_identical(records: list[dict[str, Any]]) -> bool:
    by_task = Counter((r["task"], r["prompt_sha256"]) for r in records)
    return len({task for task, _ in by_task}) == len(by_task)
