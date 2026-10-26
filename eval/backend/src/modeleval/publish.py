"""`modeleval publish`: the results page's rows (docs/benchmark/README.md) for this pipeline's runs.

Every run this pipeline wrote under docs/benchmark/<machine>/<family>/<quant>/backend/<run-id>/ (a
manifest.json of ours; paths.results_dir) gets up to two summary.json rows, with the numbers RESULTS.md
shows (report.py's helpers):

    .../<quant>/backend/<run-id>/summary.json         suite backend, protocol speed-<n>
    .../<quant>/agent_ability/<run-id>/summary.json   suite agent_ability, protocol capability-<n>

A protocol id names one fixed setting (docs/benchmark/README.md): the protocol version plus the macOS and
Xcode builds for speed (DESIGN §2, §7) and plus the harness epoch for capability (§6.5). A run whose setting
has no id below is skipped with a message: a new setting gets a new id (and so a new table), never an old
one. Afterwards `python3 scripts/build_docs.py` at the repository root rebuilds the page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modeleval import paths, report, stats

DASH = "—"              # not run
PARTIAL = " (partial)"  # a set or harness block that stopped before its last task
WRITE_UP = "README.md"  # a run's write-up, next to its manifest.json
CAP_SUITE = "agent_ability"
GIB = 2**30

SPEED_PROTOCOLS: dict[str, dict[str, Any]] = {
    "speed-1": {"protocol_version": 1, "macos_build": "26B5101f", "xcode_build": "27A9275",
                "title": "Speed on oMLX 0.7.1rc1 (M5 Max): text-only, agent sampling, thinking off"},
}
CAP_PROTOCOLS: dict[str, dict[str, Any]] = {
    "capability-1": {"protocol_version": 1, "epoch": "7582825160ad",
                     "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and "
                              "Aider 22 (ranking set); thinking off"},
}
SPEED_METRICS = ("64K cold TTFT", "Cached-prefix TTFT", "Decode 16K", "Decode 32K", "Decode 64K", "Decode 110K",
                 "20-step agent trajectory", "Peak server memory")
CAP_METRICS = ("Claude Code (gate)", "Hermes (gate)", "localmelo (gate)", "Gate set", "Ranking set",
               "M7 tool-call cases")
DECODE_UNITS = ("m3-16k", "m3-32k", "g5+m3-64k", "m3-110k")


def speed_protocol(run: report.Run) -> str | None:
    mac, xc = run.builds
    return next((pid for pid, p in SPEED_PROTOCOLS.items() if (p["protocol_version"], p["macos_build"],
                                                                p["xcode_build"]) == (run.protocol, mac, xc)), None)


def cap_protocol(run: report.Run) -> str | None:
    return next((pid for pid, p in CAP_PROTOCOLS.items()
                 if (p["protocol_version"], p["epoch"]) == (run.protocol, run.epoch)), None)


# ---------------------------------------------------------------- values


def _num(v: Any, nd: int, unit: str) -> str:
    return f"{v:.{nd}f} {unit}"


def _cell(run: report.Run, unit: str, path: tuple[str, ...], nd: int, suffix: str) -> str:
    v, state = report._metric(run, unit, *path)
    if v is None:
        return DASH if state == "missing" else state   # "disturbed x3", "invalid"
    return _num(v, nd, suffix)


def speed_values(run: report.Run) -> list[str]:
    """The SPEED_METRICS cells: the kept attempts' medians, as RESULTS.md prints them."""
    if not (report.functional(run) or {}).get("ok"):
        return [DASH] * len(SPEED_METRICS)
    m4 = report.m4_summary(run)["headline_s"]
    peak = report.memory(run).get("peak")
    return [_cell(run, "g5+m3-64k", ("g5", "ttft_s"), 1, "s"),
            _cell(run, "m1", ("m1", "warm_ttft_s", "median"), 2, "s"),
            *(_cell(run, u, ("m3", "decode_tps", "median"), 1, "tok/s") for u in DECODE_UNITS),
            DASH if m4 is None else _num(m4, 1, "s"),
            DASH if peak is None else _num(peak / GIB, 1, "GiB")]


def speed_notes(run: report.Run) -> str:
    f = report.functional(run) or {}
    if not f.get("ok"):
        return f"Functional gate failed ({f.get('stop_reason') or 'see RESULTS.md'}); nothing timed ran."
    notes = []
    units = [u for u in report.speed.timed_units(_spec(run)) if report.minor_kept(run, u)]
    if units:
        notes.append(f"Kept with a minor disturbance flag (DESIGN §5.4): {', '.join(units)}.")
    if any(report._decode_mark(run, f"m4-{i}", "m4") for i in (1, 2, 3)):
        notes.append("Some trajectory steps opened with a tool call, so their TTFT includes its decode.")
    if not report.speed_complete(run):
        notes.append("Speed leg incomplete.")
    return " ".join(notes)


def _spec(run: report.Run) -> Any:
    from types import SimpleNamespace
    return SimpleNamespace(spec=run.model.get("spec", "mtp"))


def _set_cell(run: report.Run, counts: dict[str, dict[str, Any]], set_name: str) -> str:
    runs = sum(d["runs"] for d in counts.values())
    if not runs:
        return DASH
    solved = sum(d["solved"] for d in counts.values())
    return f"{solved}/{runs}" + ("" if report.cap_complete(run, set_name) else PARTIAL)


def cap_values(run: report.Run) -> list[str]:
    """The CAP_METRICS cells: solved runs per harness (gate set), per set, and the M7 case gate."""
    gate = stats.harness_counts(run.records, "gate")
    ranking = stats.harness_counts(run.records, "ranking")
    cells = []
    for h in report.HARNESSES:
        d = gate.get(h)
        cells.append(DASH if not d else f"{d['solved']}/{d['runs']}"
                     + ("" if d["runs"] >= report.SET_TASKS["gate"] else PARTIAL))
    m7 = (report.functional(run) or {}).get("m7") or {}
    m7_cell = f"{m7['passed_cases']}/{m7['scored_cases']}" if m7.get("scored_cases") else DASH
    return cells + [_set_cell(run, gate, "gate"), _set_cell(run, ranking, "ranking"), m7_cell]


def cap_notes(run: report.Run) -> str:
    gate = stats.harness_counts(run.records, "gate")
    ranking = stats.harness_counts(run.records, "ranking")
    notes = []
    if not report.cap_complete(run, "gate"):
        done = sum(d["runs"] for d in gate.values())
        status = str(run.manifest.get("status") or "stopped")
        notes.append(f"Capability leg {'stopped (interrupted)' if status == 'interrupted' else status} after "
                     f"{done} of {report.SET_TASKS['gate'] * len(report.HARNESSES)} gate-set runs.")
    missing = [report.HARNESS_NAMES[h] for h in report.HARNESSES if h not in gate]
    decision = run.manifest.get("gate_decision") or {}
    if not ranking and decision.get("action") == "skip":
        notes.append(f"Ranking set skipped (gate decision: {decision.get('decision')}).")
    elif not ranking:
        missing.append("the ranking set")
    if missing:
        notes.append(f"Not run: {', '.join(missing)}.")
    return " ".join(notes)


# ---------------------------------------------------------------- rows


def _system(run: report.Run) -> dict[str, str]:
    mac, xc = run.builds
    machine = run.machine
    server = run.manifest.get("server") or {}
    version = next((ln.get("effective_settings", {}).get("version") for ln in run.manifest.get("launches") or []
                    if isinstance(ln, dict) and ln.get("effective_settings", {}).get("version")), None)
    sha = str((server.get("omlx") or {}).get("sha") or "")[:8]
    text = " ".join(x for x in (f"oMLX {version}" if version else "oMLX", f"({sha})" if sha else "") if x)
    return {"machine": machine, "macos_build": str(mac or ""), "xcode_build": str(xc or ""), "server": text}


def _base(run: report.Run, family: str, quant: str, suite: str) -> dict[str, Any]:
    row = {"schema": 1, "suite": suite, "protocol": "", "title": "", "model": run.name, "family": family,
           "quant": quant, "run_id": run.run_id, "date": str(run.manifest.get("started") or run.run_id)[:10],
           "system": _system(run), "metrics": []}
    if run.is_reference:
        row["reference"] = True
    return row


def rows(run: report.Run) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    """((path, summary) per row, messages) for one run directory (paths.results_dir)."""
    _, family, quant, _, run_id = paths.run_parts(run.run_dir)
    cap_dir = run.run_dir.parent.parent / CAP_SUITE / run_id   # the same model's agent_ability run directory
    out, msgs = [], []
    write_up = (run.run_dir / WRITE_UP).is_file()
    if run.units is not None:
        pid = speed_protocol(run)
        if pid is None:
            mac, xc = run.builds
            msgs.append(f"{family}/{quant}/{run_id}: no speed protocol id for protocol {run.protocol}, macOS {mac}, "
                        f"Xcode {xc} (add one to publish.SPEED_PROTOCOLS): speed row skipped")
        else:
            row = _base(run, family, quant, paths.SUITE)
            row.update(protocol=pid, title=SPEED_PROTOCOLS[pid]["title"],
                       metrics=[{"name": n, "value": v} for n, v in zip(SPEED_METRICS, speed_values(run))])
            if notes := speed_notes(run):
                row["notes"] = notes
            row["details"] = WRITE_UP if write_up else "speed/units.json"
            out.append((run.run_dir / "summary.json", row))
    if run.records:
        pid = cap_protocol(run)
        if pid is None:
            msgs.append(f"{family}/{quant}/{run_id}: no capability protocol id for protocol {run.protocol}, harness "
                        f"epoch {run.epoch} (add one to publish.CAP_PROTOCOLS): capability row skipped")
        else:
            row = _base(run, family, quant, CAP_SUITE)
            row.update(protocol=pid, title=CAP_PROTOCOLS[pid]["title"],
                       metrics=[{"name": n, "value": v} for n, v in zip(CAP_METRICS, cap_values(run))])
            if notes := cap_notes(run):
                row["notes"] = notes
            back = f"../../{paths.SUITE}/{run_id}/"
            row["details"] = back + (WRITE_UP if write_up else "capability/results.json")
            out.append((cap_dir / "summary.json", row))
    return out, msgs


def ours(run: report.Run) -> bool:
    """A verified, finished run this pipeline wrote (not another pipeline's summary-only directory)."""
    m = run.manifest
    return bool(m.get("protocol_version")) and not m.get("unverified") and m.get("status") != "running"


def publish(root: Path | None = None, model: str | None = None, run_id: str | None = None) -> list[str]:
    """Write the summary.json rows of every matching run; returns one line per file written or skipped."""
    root = Path(root or paths.RESULTS)
    lines = []
    found = [r for r in report.discover(root) if ours(r)
             and (model is None or model in (r.model_key, r.model.get("served_id")))
             and (run_id is None or r.run_id == run_id)]
    if not found:
        return [f"no run of this pipeline under {paths.tilde(root)}"
                + (f" for {model}" if model else "") + (f" {run_id}" if run_id else "")]
    for run in found:
        written, msgs = rows(run)
        lines += msgs
        for path, row in written:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            lines.append(f"wrote {paths.tilde(path)}")
    return lines
