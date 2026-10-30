"""The report of a web study: accuracy and cost per harness and arm, and D against S.

    cd eval
    uv run python agent_ability/web/report.py <dir>   # writes <dir>/report.md and report.json

Reads <dir>/web.json (the records run.py keeps). Nothing here needs a model, a server or Chrome, and a partial study
reports what it has. For each harness and arm:

- verdicts by tier, with costs: runs, the count of each verdict, accuracy (pass over counted runs) and the means
  per counted run of model calls, tool calls, output tokens, uncached prompt tokens, images and wall time. Runs the
  CAPTCHA blocked (``env_blocked``) and runs that were ``interrupted`` are not counted; a blocked run's retry is;
- zero-navigation answers (D, S and S-cua): an answer (pass or fail) given with no page visit, and how often it was
  right;
- D and S accuracy (and S-cua's, when the study has it), split by whether C of the same harness got the question right
  in any repeat;
- D against S, paired by question: a question counts for the arm with the higher pass rate over its counted repeats,
  ties are dropped, and the exact two-sided sign test says whether the split is unlikely under chance. The minimum
  detectable difference is the smallest net margin (questions won minus questions lost) that the test calls
  significant, for the number of questions that differ and for all the paired questions. The test and the margin are
  the backend report's (backend/src/modeleval/stats.py), written out again here to keep this file standalone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ALPHA = 0.05
EXCLUDED = ("env_blocked", "interrupted")
COUNTED = ("pass", "fail", "abstain", "no_answer", "timeout", "harness_error")
COSTS = ("n_llm", "n_tool", "output_tokens", "uncached_tokens", "n_images", "wall_s")
TIERS = ("easy", "medium")
DONE = ("finished", "interrupted")
DOUBLE_ARMS = ("D", "S")  # the pair that is compared
BROWSING_ARMS = ("D", "S", "S-cua")  # the arms with a browser: zero-navigation answers and the closed-book split


# --- the sign test (backend/src/modeleval/stats.py) ---


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test, ties dropped: P(|X - n/2| >= |w - n/2|) under Binomial(n, 1/2)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**n)


def min_margin(discordant: int, alpha: float = ALPHA) -> int | None:
    """The smallest net margin |wins - losses| that the sign test calls significant when ``discordant`` questions
    differ; None when even a clean sweep is not significant."""
    for w in range((discordant + 1) // 2, discordant + 1):  # p falls as the split gets more lopsided
        if sign_test(w, discordant - w) < alpha:
            return w - (discordant - w)
    return None


def min_discordant(alpha: float = ALPHA) -> int:
    """The fewest differing questions for which a clean sweep is significant (6 at alpha 0.05)."""
    d = 1
    while sign_test(d, 0) >= alpha:
        d += 1
    return d


def mdd(n_questions: int, discordant: int, alpha: float = ALPHA) -> dict[str, Any]:
    return {
        "n_questions": n_questions,
        "discordant": discordant,
        "min_discordant": min_discordant(alpha),
        "margin_at_discordant": min_margin(discordant, alpha),
        "margin_all_questions": min_margin(n_questions, alpha),
    }


# --- records ---


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def load_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every finished or interrupted record, in schedule order."""
    runs = [r for r in state["runs"].values() if r.get("state") in DONE]
    return sorted(runs, key=lambda r: (r["index"], r.get("attempt", 1)))


def counted(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("verdict") not in EXCLUDED]


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verdict counts, accuracy and mean costs of some counted records."""
    verdicts = Counter(r["verdict"] for r in records)
    n_pass = verdicts["pass"]
    return {
        "runs": len(records),
        "verdicts": {v: verdicts[v] for v in COUNTED},
        "n_pass": n_pass,
        "accuracy": _rate(n_pass, len(records)),
        "costs": {c: _mean(r.get(c) for r in records) for c in COSTS},
    }


def summarize_cell(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One harness and arm: all counted runs and each tier, plus what was left out."""
    kept = counted(records)
    return {
        "all": summarize(kept),
        "tiers": {t: summarize([r for r in kept if r["tier"] == t]) for t in TIERS},
        "excluded": dict(Counter(r["verdict"] for r in records if r["verdict"] in EXCLUDED)),
    }


def zero_navigation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Answers (pass or fail) of D or S that came with no page visit, and their accuracy against the others'."""
    answered = [r for r in counted(records) if r["verdict"] in ("pass", "fail") and r.get("n_nav") is not None]
    zero = [r for r in answered if r.get("zero_nav")]
    rest = [r for r in answered if not r.get("zero_nav")]
    return {
        "answered": len(answered),
        "zero_nav": len(zero),
        "share": _rate(len(zero), len(answered)),
        "zero_nav_pass": sum(r["verdict"] == "pass" for r in zero),
        "zero_nav_accuracy": _rate(sum(r["verdict"] == "pass" for r in zero), len(zero)),
        "other_pass": sum(r["verdict"] == "pass" for r in rest),
        "other_answered": len(rest),
        "other_accuracy": _rate(sum(r["verdict"] == "pass" for r in rest), len(rest)),
    }


def c_solved(records: Sequence[dict[str, Any]], harness: str) -> set[int] | None:
    """The questions that C of a harness got right in any repeat; None when the study has no C runs for it."""
    c_runs = counted(r for r in records if r["harness"] == harness and r["arm"] == "C")
    if not c_runs:
        return None
    return {r["question"] for r in c_runs if r["verdict"] == "pass"}


def split_by_closed_book(records: Sequence[dict[str, Any]], solved: set[int] | None) -> dict[str, Any] | None:
    """D or S runs on the questions that C answered right (``c_right``) and those it did not (``c_wrong``)."""
    if solved is None:
        return None
    kept = counted(records)
    out: dict[str, Any] = {}
    for name, rows in (
        ("c_right", [r for r in kept if r["question"] in solved]),
        ("c_wrong", [r for r in kept if r["question"] not in solved]),
    ):
        out[name] = {
            "questions": len({r["question"] for r in rows}),
            "runs": len(rows),
            "n_pass": sum(r["verdict"] == "pass" for r in rows),
            "accuracy": _rate(sum(r["verdict"] == "pass" for r in rows), len(rows)),
        }
    return out


def paired_d_vs_s(records: Sequence[dict[str, Any]], harness: str) -> dict[str, Any] | None:
    """D against S on the questions both have counted runs for. None when a side has no counted run at all."""
    rates: dict[str, dict[int, tuple[int, int]]] = {}
    tiers: dict[int, str] = {}
    for arm in DOUBLE_ARMS:
        by_question: dict[int, list[dict[str, Any]]] = {}
        for r in counted(x for x in records if x["harness"] == harness and x["arm"] == arm):
            by_question.setdefault(r["question"], []).append(r)
            tiers[r["question"]] = r["tier"]
        rates[arm] = {q: (sum(r["verdict"] == "pass" for r in rs), len(rs)) for q, rs in by_question.items()}
    if not rates["D"] or not rates["S"]:
        return None
    pairs = []
    for question in sorted(set(rates["D"]) & set(rates["S"])):
        (dp, dn), (sp, sn) = rates["D"][question], rates["S"][question]
        d_rate, s_rate = dp / dn, sp / sn
        result = "D" if d_rate > s_rate else "S" if s_rate > d_rate else "tie"
        pairs.append(
            {"question": question, "tier": tiers[question], "d": f"{dp}/{dn}", "s": f"{sp}/{sn}", "result": result}
        )
    wins = sum(p["result"] == "D" for p in pairs)
    losses = sum(p["result"] == "S" for p in pairs)
    p_value = sign_test(wins, losses)
    return {
        "questions": len(pairs),
        "d_better": wins,
        "s_better": losses,
        "ties": len(pairs) - wins - losses,
        "net": wins - losses,
        "p_value": p_value,
        "significant": p_value < ALPHA,
        "alpha": ALPHA,
        "mdd": mdd(len(pairs), wins + losses),
        "pairs": pairs,
    }


def question_marks(records: Sequence[dict[str, Any]], harness: str, arm: str, question: int, repeats: int) -> str:
    """One mark per repeat: P pass, - counted but not a pass, x left out (blocked or interrupted), . not run."""
    marks = []
    for repeat in range(1, repeats + 1):
        runs = [
            r for r in records
            if (r["harness"], r["arm"], r["question"], r["repeat"]) == (harness, arm, question, repeat)
        ]  # fmt: skip
        if not runs:
            marks.append(".")
            continue
        last = max(runs, key=lambda r: r.get("attempt", 1))  # a retry stands in for the blocked first attempt
        marks.append("x" if last["verdict"] in EXCLUDED else "P" if last["verdict"] == "pass" else "-")
    return " ".join(marks)


def flags(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "captcha": [
            r["run_id"] for r in records if r.get("verdict") == "env_blocked" and r.get("blocked_by") != "space"
        ],
        "space_lost": [r["run_id"] for r in records if r.get("blocked_by") == "space"],
        "answer_key_suspect": [r["run_id"] for r in records if r.get("answer_key_suspect")],
        "search_tool_leak": [r["run_id"] for r in records if r.get("search_tool_leak")],
        "foreign_tools": sorted({t for r in records for t in r.get("foreign_tools") or []}),
        "tap_child_failed": [r["run_id"] for r in records if (r.get("tap") or {}).get("child_failed")],
    }


def build_report(out: Path) -> dict[str, Any]:
    state = json.loads((out / "web.json").read_text())
    config = state["config"]
    records = load_records(state)
    harnesses, arms_, repeats = config["harnesses"], config["arms"], config["repeats"]
    cells: dict[str, dict[str, Any]] = {}
    zero: dict[str, dict[str, Any]] = {}
    split: dict[str, dict[str, Any]] = {}
    paired: dict[str, Any] = {}
    for harness in harnesses:
        solved = c_solved(records, harness)
        cells[harness], zero[harness], split[harness] = {}, {}, {}
        for arm in arms_:
            rows = [r for r in records if r["harness"] == harness and r["arm"] == arm]
            cells[harness][arm] = summarize_cell(rows)
            if arm in BROWSING_ARMS:
                zero[harness][arm] = zero_navigation(rows)
                split[harness][arm] = split_by_closed_book(rows, solved)
        if all(a in arms_ for a in DOUBLE_ARMS):
            paired[harness] = paired_d_vs_s(records, harness)
    planned = len(config["questions"]) * repeats * len(harnesses) * len(arms_)
    return {
        "schema_version": 1,
        "protocol": config.get("protocol"),
        "config": config,
        "complete": state.get("complete", False),
        "chrome_version": state.get("chrome_version"),
        "runs_recorded": len(counted(records)),
        "runs_planned": planned,
        "excluded": dict(Counter(r["verdict"] for r in records if r["verdict"] in EXCLUDED)),
        "cells": cells,
        "zero_nav": zero,
        "closed_book_split": split,
        "paired": paired,
        "marks": {
            q: {h: {a: question_marks(records, h, a, q, repeats) for a in arms_} for h in harnesses}
            for q in config["questions"]
        },
        "tiers": {r["question"]: r["tier"] for r in records},
        "flags": flags(records),
    }


# --- Markdown ---


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _tier_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Verdicts by tier",
        "",
        (
            "Accuracy is pass over counted runs. Costs are means per counted run; for C, model calls include its "
            "tool-less requests, and it has no tool calls to speak of."
        ),
        "",
        (
            "| Harness | Arm | Tier | Runs | Pass | Fail | Abstain | No answer | Timeout | Error | Accuracy "
            "| Model calls | Tool calls | Output tokens | Uncached tokens | Images | Wall s |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for harness, arms_ in report["cells"].items():
        for arm, cell in arms_.items():
            for name, s in (*cell["tiers"].items(), ("all", cell["all"])):
                v, c = s["verdicts"], s["costs"]
                lines.append(
                    f"| {harness} | {arm} | {name} | {s['runs']} | {v['pass']} | {v['fail']} | {v['abstain']} | "
                    f"{v['no_answer']} | {v['timeout']} | {v['harness_error']} | {_pct(s['accuracy'])} | "
                    f"{_num(c['n_llm'], 1)} | {_num(c['n_tool'], 1)} | {_num(c['output_tokens'])} | "
                    f"{_num(c['uncached_tokens'])} | {_num(c['n_images'], 1)} | {_num(c['wall_s'])} |"
                )
    return [*lines, ""]


def _zero_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Zero-navigation answers",
        "",
        (
            "An answer (pass or fail) given with no page visit since the start of the run: from the model's own "
            "memory or from the search page. C has no browser and is not listed."
        ),
        "",
        "| Harness | Arm | Answered | Zero-navigation | Share | Zero-navigation accuracy | Accuracy of the others |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for harness, arms_ in report["zero_nav"].items():
        for arm, z in arms_.items():
            lines.append(
                f"| {harness} | {arm} | {z['answered']} | {z['zero_nav']} | {_pct(z['share'])} | "
                f"{z['zero_nav_pass']}/{z['zero_nav']} ({_pct(z['zero_nav_accuracy'])}) | "
                f"{z['other_pass']}/{z['other_answered']} ({_pct(z['other_accuracy'])}) |"
            )
    return [*lines, ""]


def _split_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "## D and S by closed-book knowledge",
        "",
        (
            "Runs split by whether C of the same harness got the question right in any repeat. A browser that adds "
            "little over memory scores alike on both rows."
        ),
        "",
        "| Harness | Arm | C ever right | Questions | Runs | Pass | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for harness, arms_ in report["closed_book_split"].items():
        for arm, split in arms_.items():
            if split is None:
                lines.append(f"| {harness} | {arm} | no C runs | | | | |")
                continue
            for key, label in (("c_right", "yes"), ("c_wrong", "no")):
                s = split[key]
                lines.append(
                    f"| {harness} | {arm} | {label} | {s['questions']} | {s['runs']} | {s['n_pass']} | {_pct(s['accuracy'])} |"
                )
    return [*lines, ""]


def _paired_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## D against S, paired by question",
        "",
        (
            "A question goes to the arm with the higher pass rate over its counted repeats; ties are dropped. The "
            "p-value is the exact two-sided sign test. The minimum detectable difference is the smallest net margin "
            f"(questions won minus questions lost) that the test calls significant at {ALPHA}."
        ),
        "",
        "| Harness | Paired questions | D better | S better | Tie | Net (D - S) | p | Minimum detectable difference |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    details: list[str] = []
    for harness, p in report["paired"].items():
        if p is None:
            lines.append(f"| {harness} | 0 | | | | | | needs counted D and S runs |")
            continue
        m = p["mdd"]
        at = m["margin_at_discordant"]
        text = (
            f"net {at} of {m['discordant']} differing questions" if at is not None
            else f"none: {m['discordant']} differing questions are fewer than {m['min_discordant']}"
        )  # fmt: skip
        full = m["margin_all_questions"]
        text += f"; if all {m['n_questions']} differed, net {full}" if full is not None else ""
        lines.append(
            f"| {harness} | {p['questions']} | {p['d_better']} | {p['s_better']} | {p['ties']} | {p['net']:+d} | "
            f"{p['p_value']:.3f}{' *' if p['significant'] else ''} | {text} |"
        )
        differing = [x for x in p["pairs"] if x["result"] != "tie"]
        if differing:
            details.append(
                f"- {harness}: " + ", ".join(f"{x['question']} ({x['tier']}) D {x['d']} S {x['s']}" for x in differing)
            )
    lines += ["", f"A star marks p < {ALPHA}. A clean sweep needs at least {min_discordant()} differing questions.", ""]
    if details:
        lines += ["Questions that differ (pass counts of D and S):", "", *details, ""]
    return lines


def _marks_table(report: dict[str, Any]) -> list[str]:
    columns = [(h, a) for h, arms_ in report["cells"].items() for a in arms_]
    lines = [
        "## Passes per question (one mark per repeat)",
        "",
        "P pass, - counted but not a pass, x left out (blocked or interrupted), . not run.",
        "",
        "| Question | Tier | " + " | ".join(f"{h} {a}" for h, a in columns) + " |",
        "|---|---|" + "---|" * len(columns),
    ]
    for question, by_harness in report["marks"].items():
        tier = report["tiers"].get(int(question), report["tiers"].get(question, ""))
        lines.append(f"| {question} | {tier} | " + " | ".join(by_harness[h][a] for h, a in columns) + " |")
    return [*lines, ""]


def _flag_lines(report: dict[str, Any]) -> list[str]:
    f = report["flags"]

    def ids(values: Sequence[str]) -> str:
        return ", ".join(f"`{v}`" for v in values) if values else "none"

    return [
        "## Flags",
        "",
        f"- Blocked by a CAPTCHA (left out of the counts): {ids(f['captcha'])}",
        f"- S-cua window off the Space on screen (left out of the counts): {ids(f['space_lost'])}",
        f"- Answer key suspected (a URL or title naming SimpleQA): {ids(f['answer_key_suspect'])}",
        f"- A search tool outside the web server was offered to the model: {ids(f['search_tool_leak'])}",
        f"- Tools offered besides the web server's: {', '.join(f['foreign_tools']) or 'none'}",
        f"- The tap's browser server failed to start: {ids(f['tap_child_failed'])}",
        "",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    excluded = ", ".join(f"{n} {v}" for v, n in sorted(report["excluded"].items())) or "none"
    lines = [
        f"# Web search suite: {', '.join(cfg['harnesses'])} on arms {', '.join(cfg['arms'])}",
        "",
        (
            f"- Model `{cfg.get('model')}`; {len(cfg['questions'])} questions x {cfg['repeats']} repeats x "
            f"{len(cfg['harnesses'])} harnesses x {len(cfg['arms'])} arms; dataset revision "
            f"`{(cfg.get('dataset_revision') or 'unknown')[:12]}`"
        ),
        (
            f"- Chrome {report.get('chrome_version') or 'n/a'}; "
            f"chrome-devtools-mcp {cfg.get('chrome_devtools_mcp') or 'n/a'}; "
            f"localmelo `{(cfg.get('localmelo_sha') or 'n/a')[:7]}`"
        ),
        f"- Counted runs: {report['runs_recorded']} of {report['runs_planned']} planned"
        + ("" if report["complete"] else " (the study is not complete)"),
        f"- Left out of the counts: {excluded}",
        "",
        *_tier_table(report),
        *_zero_table(report),
        *_split_table(report),
        *(_paired_section(report) if report["paired"] else []),
        *_marks_table(report),
        *_flag_lines(report),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="a web study's output directory (it has web.json)")
    args = ap.parse_args(argv)
    out = args.run_dir.resolve()
    if not (out / "web.json").exists():
        print(f"{out} has no web.json", file=sys.stderr)
        return 2
    report = build_report(out)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "report.md").write_text(render_markdown(report))
    print(f"{out / 'report.md'}: {report['runs_recorded']}/{report['runs_planned']} runs counted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
