"""The report of a forms study: each arm's pass rate and cost, and the arms against the first, paired by task.

    cd eval
    uv run python agent_ability/forms/report.py <dir>   # writes <dir>/report.md and report.json

Reads <dir>/forms.json (the records run.py keeps); a partial study reports what it has. Nothing here needs a model, a
server or Chrome.

Per arm: pass, steps (successful model calls), wall, the model's share of the wall (summed proxy durations over wall),
thinking per step, refused tool calls, and how the runs ended.

**Thinking per step** is in the best unit every run has, and the report says which: ``reasoning_tokens`` (the usage of
the responses, ``completion_tokens_details``), else ``reasoning_chars`` (the length of the reasoning text in the
responses, as the proxy measures it), else ``melo_thinking_chars`` (the thinking melo kept in its history.db, for a
server that leaves it in the reply's text).

**The paired difference.** Runs pair up by task and repeat: the two arms of one pair ran back to back. A task's
difference is the mean over its pairs (the other arm minus the first), and the study's is the mean over tasks, so no
task weighs more for having more pairs. The 95% interval is a percentile bootstrap, resampling tasks and then the pairs
inside each, with a fixed seed. With eight tasks and three repeats it describes this task set, not forms at large.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

STATE_FILE = "forms.json"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 7
LEVEL = 0.95
UNITS = {
    "reasoning_tokens": "reasoning tokens, from the responses' usage",
    "reasoning_chars": "characters of reasoning text in the responses",
    "melo_thinking_chars": "characters of thinking in melo's history.db",
}


def load(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = json.loads((out / STATE_FILE).read_text())
    records = [r for r in state.get("runs", {}).values() if r.get("state") == "finished"]
    return state, sorted(records, key=lambda r: r["index"])


def thinking_unit(records: Sequence[Mapping[str, Any]]) -> str:
    """The best unit that every run has; see the module's docstring."""
    if records and all(r.get("reasoning_tokens") is not None for r in records):
        return "reasoning_tokens"
    if any(r.get("reasoning_chars") for r in records) or not any(r.get("melo_thinking_chars") for r in records):
        return "reasoning_chars"
    return "melo_thinking_chars"


def thinking_per_step(record: Mapping[str, Any], unit: str) -> float | None:
    value, steps = record.get(unit), record.get("n_calls")
    return value / steps if value is not None and steps else None


def metrics(unit: str) -> dict[str, Callable[[Mapping[str, Any]], float | None]]:
    """What the paired table compares, by name."""
    return {
        "pass": lambda r: 1.0 if r.get("passed") else 0.0,
        "steps": lambda r: r.get("n_calls"),
        "wall_s": lambda r: r.get("wall_s"),
        "model_share": lambda r: r.get("model_share"),
        "thinking_per_step": lambda r: thinking_per_step(r, unit),
        "refused_calls": lambda r: r.get("refused_calls"),
    }


def mean(values: Iterable[float | None]) -> float | None:
    found = [v for v in values if v is not None]
    return sum(found) / len(found) if found else None


def median(values: Iterable[float | None]) -> float | None:
    found = [v for v in values if v is not None]
    return statistics.median(found) if found else None


def arm_summary(records: Sequence[Mapping[str, Any]], unit: str) -> dict[str, Any]:
    named = metrics(unit)
    n_pass = sum(1 for r in records if r.get("passed"))
    return {
        "runs": len(records),
        "passed": n_pass,
        "pass_rate": n_pass / len(records) if records else None,
        "steps_mean": mean(named["steps"](r) for r in records),
        "steps_median": median(named["steps"](r) for r in records),
        "wall_s_mean": mean(named["wall_s"](r) for r in records),
        "wall_s_median": median(named["wall_s"](r) for r in records),
        "model_share_mean": mean(named["model_share"](r) for r in records),
        "thinking_per_step_mean": mean(named["thinking_per_step"](r) for r in records),
        "longest_call_s": max((r.get("longest_call_s") or 0.0 for r in records), default=None),
        "refused_calls": sum(r.get("refused_calls") or 0 for r in records),
        "tool_errors": sum(r.get("tool_errors") or 0 for r in records),
        "end_reasons": dict(Counter(str(r.get("end_reason")) for r in records)),
    }


def by_task(records: Sequence[Mapping[str, Any]], arm_names: Sequence[str], unit: str) -> dict[str, dict[str, Any]]:
    tasks = list(dict.fromkeys(r["task"] for r in records))
    return {
        task: {
            arm: arm_summary([r for r in records if r["task"] == task and r["arm"] == arm], unit) for arm in arm_names
        }
        for task in tasks
    }


def differences(
    records: Sequence[Mapping[str, Any]], base: str, other: str, value: Callable[[Mapping[str, Any]], float | None]
) -> dict[str, list[float]]:
    """Per task, ``other`` minus ``base`` for each repeat that has both with a value."""
    runs = {(r["task"], r["repeat"], r["arm"]): r for r in records}
    found: dict[str, list[float]] = {}
    for task, repeat in dict.fromkeys((r["task"], r["repeat"]) for r in records):
        a, b = runs.get((task, repeat, base)), runs.get((task, repeat, other))
        if a is None or b is None:
            continue
        x, y = value(a), value(b)
        if x is not None and y is not None:
            found.setdefault(task, []).append(y - x)
    return found


def bootstrap_interval(
    groups: Mapping[str, Sequence[float]],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = LEVEL,
) -> tuple[float, float] | None:
    """A percentile interval of the mean over groups: resample the groups, then the values inside each."""
    names = [name for name, values in groups.items() if values]
    if not names:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        total = 0.0
        for name in rng.choices(names, k=len(names)):
            values = groups[name]
            total += sum(rng.choices(values, k=len(values))) / len(values)
        means.append(total / len(names))
    means.sort()
    tail = (1 - level) / 2
    return means[int(tail * samples)], means[min(int((1 - tail) * samples), samples - 1)]


def paired(records: Sequence[Mapping[str, Any]], base: str, other: str, unit: str) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for name, value in metrics(unit).items():
        groups = differences(records, base, other, value)
        task_means = {task: sum(v) / len(v) for task, v in groups.items()}
        table[name] = {
            "pairs": sum(len(v) for v in groups.values()),
            "mean_difference": mean(task_means.values()),
            "interval": bootstrap_interval(groups),
            "by_task": task_means,
        }
    return table


def flags(records: Sequence[Mapping[str, Any]], arm_names: Sequence[str], unit: str) -> list[str]:
    """What would make the numbers mean something else."""
    found = []
    not_headless = sum(1 for r in records if r.get("headless") is not True)
    if not_headless:
        found.append(f"{not_headless} runs have no proof of a headless Chrome")
    not_applied = sum(1 for r in records if r.get("arm_applied") is False)
    if not_applied:
        found.append(f"{not_applied} runs sent a request without their arm's chat_template_kwargs")
    own = sum(1 for r in records if r.get("melo_chrome_profile"))
    if own:
        found.append(f"melo started a Chrome of its own in {own} runs")
    foreign = sorted({t for r in records for t in r.get("foreign_tools") or []})
    if foreign:
        found.append(f"tools other than the tap's were offered: {', '.join(foreign)}")
    for arm in arm_names:
        rows = [r for r in records if r["arm"] == arm]
        asked_off = all(r.get("thinking_off_calls") == r.get("n_calls") for r in rows)
        thought = any(r.get(unit) for r in rows)
        if rows and thought and asked_off:
            found.append(f"arm {arm} asked for no thinking and still shows some ({unit})")
        if rows and not thought and not asked_off:
            found.append(f"arm {arm} shows no thinking at all ({unit})")
    return found


def build_report(out: Path) -> dict[str, Any]:
    state, records = load(out)
    config = state.get("config", {})
    arm_names = [a for a in config.get("arms", {}) if any(r["arm"] == a for r in records)]
    unit = thinking_unit(records)
    environment_path = out / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    return {
        "config": config,
        "complete": state.get("complete", False),
        "environment": {
            k: (environment.get("start") or {}).get(k) for k in ("sw_vers", "xcodebuild_version", "chrome_version")
        },
        "n_runs": len(records),
        "thinking_unit": unit,
        "arms": {arm: arm_summary([r for r in records if r["arm"] == arm], unit) for arm in arm_names},
        "tasks": by_task(records, arm_names, unit),
        "paired": {other: paired(records, arm_names[0], other, unit) for other in arm_names[1:]},
        "failures": [
            {"run_id": r["run_id"], "end_reason": r.get("end_reason"), "reason": (r.get("check") or {}).get("reason")}
            for r in records
            if not r.get("passed")
        ],
        "flags": flags(records, arm_names, unit),
    }


def _num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _diff(name: str, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.0%}" if name in ("pass", "model_share") else f"{value:+,.1f}"


def _arm_table(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Arm | Runs | Pass | Steps, mean (median) | Wall s, mean (median) | Model share | Thinking per step | "
        "Longest call s | Refused calls | Tool errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, s in report["arms"].items():
        lines.append(
            f"| {arm} | {s['runs']} | {s['passed']}/{s['runs']} ({_pct(s['pass_rate'])}) | "
            f"{_num(s['steps_mean'])} ({_num(s['steps_median'])}) | "
            f"{_num(s['wall_s_mean'])} ({_num(s['wall_s_median'])}) | {_pct(s['model_share_mean'])} | "
            f"{_num(s['thinking_per_step_mean'], 0)} | {_num(s['longest_call_s'])} | {s['refused_calls']} | "
            f"{s['tool_errors']} |"
        )
    return lines


def _end_table(report: Mapping[str, Any]) -> list[str]:
    reasons = sorted({reason for s in report["arms"].values() for reason in s["end_reasons"]})
    lines = ["| Arm | " + " | ".join(reasons) + " |", "|---|" + "---|" * len(reasons)]
    for arm, s in report["arms"].items():
        lines.append(f"| {arm} | " + " | ".join(str(s["end_reasons"].get(reason, 0)) for reason in reasons) + " |")
    return lines


def _task_table(report: Mapping[str, Any]) -> list[str]:
    arms = list(report["arms"])
    head = " | ".join(f"{arm}: pass, steps, wall s" for arm in arms)
    lines = [f"| Task | {head} |", "|---|" + "---|" * len(arms)]
    for task, row in report["tasks"].items():
        cells = [
            f"{row[arm]['passed']}/{row[arm]['runs']}, {_num(row[arm]['steps_mean'])}, {_num(row[arm]['wall_s_mean'])}"
            for arm in arms
        ]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return lines


def _paired_table(table: Mapping[str, Any]) -> list[str]:
    tasks = list(dict.fromkeys(task for row in table.values() for task in row["by_task"]))
    lines = [
        "| Metric | Pairs | Mean by task | 95% interval | " + " | ".join(tasks) + " |",
        "|---|---|---|---|" + "---|" * len(tasks),
    ]
    for name, row in table.items():
        interval = row["interval"]
        shown = "-" if interval is None else f"{_diff(name, interval[0])} to {_diff(name, interval[1])}"
        cells = [_diff(name, row["by_task"].get(task)) for task in tasks]
        lines.append(
            f"| {name} | {row['pairs']} | {_diff(name, row['mean_difference'])} | {shown} | " + " | ".join(cells) + " |"
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    config = report["config"]
    environment = report["environment"]
    lines = [
        "# Forms study",
        "",
        f"{report['n_runs']} runs{'' if report['complete'] else ' so far (the study is not complete)'}. "
        f"Model {config.get('model')}, localmelo {str(config.get('localmelo_sha'))[:8]}, "
        f"wall cap {config.get('wall_cap_s')} s, {config.get('max_calls')} model calls.",
        f"{' '.join((environment.get('sw_vers') or '').split())}; "
        f"{' '.join((environment.get('xcodebuild_version') or '').split())}; "
        f"Chrome {environment.get('chrome_version')}.",
        "",
        f"Thinking per step is in {UNITS[report['thinking_unit']]}.",
        "",
        "## Per arm",
        "",
        *_arm_table(report),
        "",
        "## How the runs ended",
        "",
        *_end_table(report),
        "",
        "## Per task",
        "",
        *_task_table(report),
    ]
    arms = list(report["arms"])
    for other, table in report["paired"].items():
        lines += ["", f"## {other} minus {arms[0]}, paired by task and repeat", "", *_paired_table(table)]
    if report["failures"]:
        lines += ["", "## Runs that did not pass", ""]
        lines += [f"- {f['run_id']}: {f['end_reason']}; {f['reason']}" for f in report["failures"]]
    if report["flags"]:
        lines += ["", "## Flags", ""]
        lines += [f"- {flag}" for flag in report["flags"]]
    return "\n".join(lines) + "\n"


def write(out: Path) -> dict[str, Any]:
    report = build_report(out)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "report.md").write_text(render_markdown(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    write(args.run_dir.resolve())
    print((args.run_dir / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
