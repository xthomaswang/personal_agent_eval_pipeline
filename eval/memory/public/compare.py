"""Arms of different LongMemEval-S runs side by side: accuracy, paired differences, why answers were wrong, retrieval, control.

    cd eval
    uv run python -m memory.public.compare <run-dir>:<arm> <run-dir>:<arm> [...] [--note TEXT] [--out compare.md]

Each `<run-dir>:<arm>` is one column, named `<run name>/<arm>`. The runs must ask the same questions of the same data with
the same chat, embedding and judge models; anything else (a localmelo build, an arm's settings) is what is compared, and the
heading lists each column's build and settings. The paired difference of two columns is taken over the questions both
were asked, each question's share of right repeats against the other's, with the same bootstrap as the report
(`memory.public.report`); `--note` adds a line, such as the other changes between the builds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import report

SAME = ("questions", "data", "model", "embedding_model", "judge_model")


def load_columns(specs: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """The columns (name, run, arm, config) for ``<run-dir>:<arm>`` specs, and what makes them not comparable."""
    columns, problems = [], []
    for spec in specs:
        path, _, arm = spec.rpartition(":")
        run_dir = Path(path)
        run = report.load_run(run_dir)
        if arm not in (run["config"].get("arms") or {}):
            problems.append(f"{spec}: the run has no arm {arm!r}")
            continue
        columns.append({"name": f"{run_dir.name}/{arm}", "run": run, "arm": arm, "config": run["config"]})
    for col in columns[1:]:
        for key in SAME:
            if col["config"].get(key) != columns[0]["config"].get(key):
                problems.append(f"{col['name']}: {key} differs from {columns[0]['name']}")
    return columns, problems


def merged(columns: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    """The rows (``key`` "rows") or controls of every column, each renamed to its column."""
    return [{**r, "arm": col["name"]} for col in columns for r in col["run"][key] if r["arm"] == col["arm"]]


def compare(columns: Sequence[Mapping[str, Any]], resamples: int = report.BOOTSTRAP, seed: int = report.SEED) -> dict[str, Any]:
    names = [c["name"] for c in columns]
    rows, controls = merged(columns, "rows"), merged(columns, "controls")
    expected = {c["name"]: len(c["config"].get("questions") or []) * int(c["config"].get("repeats") or 1) for c in columns}
    return {
        "columns": [{"name": c["name"], "localmelo_sha": c["config"].get("localmelo_sha"), "arm": (c["config"].get("arms") or {})[c["arm"]],
                     "rows": sum(1 for r in rows if r["arm"] == c["name"]), "expected": expected[c["name"]]} for c in columns],
        "accuracy": report.accuracy_table(rows, names, resamples, seed),
        "paired": report.paired_table(rows, names, resamples, seed),
        "misses": report.miss_table(rows, names),
        "retrieval": report.retrieval_table(rows, names),
        "control": report.control_table(controls, rows, names),
        "resamples": resamples,
    }


def render(cmp: Mapping[str, Any], note: str = "") -> str:
    names = [c["name"] for c in cmp["columns"]]
    out = ["# LongMemEval-S: arms of different runs", ""]
    for c in cmp["columns"]:
        arm = c["arm"]
        settings = ", ".join([*arm.get("x", []), *(f"{k}={v}" for k, v in (arm.get("config") or {}).items())])
        out.append(f"- `{c['name']}`: localmelo `{str(c['localmelo_sha'])[:10]}`, {settings or 'no settings'}; "
                   f"{c['rows']} of {c['expected']} question cells")
    out += ([f"- {note}"] if note else []) + [""]
    out += ["## Accuracy", "", f"Per question type, as in the report ({cmp['resamples']} resamples).", ""]
    out += report.table(["group", *names], [[g, *(report.acc(cmp["accuracy"][g][n]) for n in names)] for g in report.ROWS])
    out += ["## Paired differences", "", "First column minus second, over the questions both were asked (percentage points).", ""]
    body = []
    for p in cmp["paired"]:
        ci = p["ci"]
        diff = "-" if p["diff"] is None else f"{100 * p['diff']:+.1f}" + (f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else "")
        body.append([f"{p['a']} - {p['b']}", p["group"], str(p["questions"]), diff, f"+{p['a_better']} / -{p['b_better']}"])
    out += report.table(["columns", "group", "questions", "difference (pp)", "better"], body)
    out += ["## Why answers were wrong", ""]
    steps = [*report.MISS_STEPS, *(s for s in report.ITEM_STEPS if any(cmp["misses"][n][s] for n in names))]
    out += report.table(["", *names], [[s, *(str(cmp["misses"][n][s]) for n in names)] for s in steps])
    rt = cmp["retrieval"][report.ALL]
    out += ["## Retrieval (all questions with an answer session)", ""]
    out += report.table(["", *names], [
        ["auto hit", *(report.hit(rt[n]["auto_hit"]) for n in names)], ["turns that searched", *(report.hit(rt[n]["searched"]) for n in names)],
        ["search hit (of those)", *(report.hit(rt[n]["search_hit"]) for n in names)], ["either", *(report.hit(rt[n]["either"]) for n in names)],
        ["turns that opened a listed turn", *(report.hit(rt[n]["opened"]) for n in names)]])
    ct = cmp["control"]
    out += ["## Control and cost", ""]
    out += report.table(["", *names], [
        ["false recall", *(report.hit(ct[n]["false_recall"]) for n in names)],
        ["control prompts that opened a listed turn", *(report.hit(ct[n]["opened"]) for n in names)],
        ["question prompt tokens", *(report.med(ct[n]["question_prompt_tokens"]) for n in names)],
        ["question first token ms", *(report.med(ct[n]["question_ttft_ms"]) for n in names)]])
    return "\n".join(out).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("columns", nargs="+", help="<run-dir>:<arm>")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int, default=report.BOOTSTRAP)
    args = parser.parse_args(argv)
    columns, problems = load_columns(args.columns)
    if problems or len(columns) < 2:
        print("\n".join(problems or ["give at least two columns"]), file=sys.stderr)
        return 2
    cmp = compare(columns, args.resamples)
    text = render(cmp, args.note)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(cmp, indent=1, default=str), encoding="utf-8")
    print(text if not args.out else f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
