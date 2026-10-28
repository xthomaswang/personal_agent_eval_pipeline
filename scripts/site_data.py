"""Enrich validated public result rows for the benchmark website. Standard library only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import statistics
from pathlib import Path

HARNESSES = ("claude_code", "hermes", "localmelo")
HISTORICAL = {"aider12-v0", "quant-fidelity-v0"}
UI_METRICS = {
    "Click, screenshot": ("replica:vision", "click"),
    "Click, AX list": ("replica:ax", "click"),
    "Fill, screenshot": ("replica:vision", "fill"),
    "Fill, AX list": ("replica:ax", "fill"),
    "Read, screenshot": ("replica:vision", "read_strict"),
    "Read, AX list": ("replica:ax", "read_strict"),
    "Read, original capture": ("capture:vision", "read_strict"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def median_field(stats, records: list[dict], name: str) -> float | None:
    values = [stats.rec_float(record, name) for record in records]
    valid = [value for value in values if value is not None]
    return round(statistics.median(valid), 1) if valid else None


def capability_data(root: Path, rows: list[dict]) -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location("site_capability_stats", root / "eval/backend/src/modeleval/stats.py")
    require(bool(spec and spec.loader), "cannot load capability statistics")
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    tasks: dict[str, list[str]] = {"gate": [], "ranking": []}
    for row in rows:
        if not re.fullmatch(r"capability-\d+", row["protocol"]):
            continue
        source = (root / "docs/benchmark" / row["system"]["machine"] / row["family"] / row["quant"]
                  / "backend" / row["run_id"] / "capability/results.json")
        results = json.loads(source.read_text(encoding="utf-8"))
        row["_harness_source"] = source.relative_to(root).as_posix()
        row["_harnesses"] = {}
        for harness in HARNESSES:
            counts = {}
            for set_name in ("gate", "ranking"):
                records = [r for r in stats.final_records(results["rows"], set_name)
                           if r.get("harness") == harness and r.get("scored") is True]
                counts[set_name] = {
                    "passed": sum(stats.rec_solved(r) for r in records), "runs": len(records), "expected": 22,
                    "modelTimeMedian": median_field(stats, records, "model_s"),
                    "modelCallsMedian": median_field(stats, records, "n_llm"),
                }
            row["_harnesses"][harness] = counts
        for set_name, metric_name in (("gate", "Gate set"), ("ranking", "Ranking set")):
            passed = sum(h[set_name]["passed"] for h in row["_harnesses"].values())
            runs = sum(h[set_name]["runs"] for h in row["_harnesses"].values())
            published = next(m["value"] for m in row["metrics"] if m["name"] == metric_name)
            require((runs == 0 and published == "—") or published.split()[0] == f"{passed}/{runs}",
                    f"{source.relative_to(root)}: {metric_name} disagrees with the summary")
            if results.get("complete_sets", {}).get(set_name):
                found = sorted({r["task"] for r in results["rows"] if r["set"] == set_name and r.get("scored")})
                require(len(found) == 22, f"{source}: {set_name} must contain 22 unique tasks")
                require(not tasks[set_name] or tasks[set_name] == found, f"{source}: task lists differ across capability protocols")
                tasks[set_name] = found
    require(set(tasks["gate"]).isdisjoint(tasks["ranking"]), "gate and ranking tasks overlap")
    return tasks


def ui_data(root: Path, rows: list[dict]) -> None:
    for row in rows:
        if row["protocol"] != "ui-real-1":
            continue
        source = (root / row["_details"]).parent / "results.json"
        results = json.loads(source.read_text(encoding="utf-8"))
        require(results.get("protocol") == row["protocol"] and results.get("run_id") == row["run_id"],
                f"{source}: UI protocol or run does not match the summary")
        audited = results.get("gold", {}).get("audited", {}).get(row["model"])
        require(isinstance(audited, dict), f"{source}: audited counts are missing for {row['model']}")
        samples = results.get("samples")
        require(isinstance(samples, int) and samples > 0, f"{source}: invalid sample count")
        for arm, counts in audited.items():
            for metric_name, pair in counts.items():
                require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair)
                        and 0 <= pair[0] <= pair[1] and pair[1] > 0,
                        f"{source}: invalid {arm}/{metric_name} counts")
        for metric in row["metrics"]:
            arm, name = UI_METRICS[metric["name"]]
            pair = audited[arm][name]
            require(metric["value"] == f"{pair[0]}/{pair[1]}", f"{source}: {metric['name']} disagrees with the summary")
        counts = {}
        for kind, key in (("click", "click"), ("fill", "fill"), ("read", "read_strict")):
            total = audited["replica:vision"][key][1]
            require(total % samples == 0, f"{source}: item-samples do not divide evenly")
            counts[kind] = total // samples
        pages = re.search(r"\b(\d+) pages\b", row["title"])
        row["_ui"] = audited
        row["_ui_source"] = source.relative_to(root).as_posix()
        row["_ui_meta"] = {"pages": int(pages.group(1)) if pages else None,
                           "items": counts, "totalItems": sum(counts.values()), "samples": samples}


def kind(row: dict) -> str:
    if row["protocol"] in HISTORICAL:
        return "archive"
    if row["protocol"].startswith("ui-"):
        return "ui"
    return {"backend": "speed", "agent_ability": "ability", "memory": "memory"}[row["suite"]]


def machine_label(machine: str) -> str:
    match = re.fullmatch(r"(m\d+)(pro|max|ultra)?-(\d+)c-(\d+)gb", machine)
    if not match:
        return machine
    chip, variant, cores, memory = match.groups()
    return f"{chip.upper()}{' ' + variant.title() if variant else ''} · {cores}-core GPU · {memory} GB"


def build_payload(root: Path, rows: list[dict], repo_url: str) -> dict:
    tasks = capability_data(root, rows)
    ui_data(root, rows)
    groups = {}
    machines = {}
    for row in rows:
        machine = row["system"]["machine"]
        machines[machine] = {"id": machine, "label": machine_label(machine)}
        key = (machine, row["suite"], row["protocol"])
        group = groups.setdefault(key, {"machine": machine, "suite": row["suite"], "id": row["protocol"],
                                        "kind": kind(row), "title": row["title"], "latest": row["date"],
                                        "metrics": [m["name"] for m in row["metrics"]]})
        group["latest"] = max(group["latest"], row["date"])
    payload = {
        "repoUrl": repo_url,
        "latest": max((r["date"] for r in rows), default=None),
        "buildCount": len({(r["family"], r["quant"]) for r in rows}),
        "resultCount": len(rows), "machineCount": len(machines),
        "harnessCount": len({h for r in rows for h in r.get("_harnesses", {})}),
        "machines": [machines[key] for key in sorted(machines)],
        "protocols": [groups[key] for key in sorted(groups)], "rows": rows, "taskSelection": tasks,
    }
    payload["sourceDigest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return payload
