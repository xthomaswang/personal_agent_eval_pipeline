"""Report of a LongMemEval-S study: accuracy per question type, retrieval, ingest, judge and the false-recall control.

    cd eval
    uv run python -m memory.public.report <run-dir>      # writes <run-dir>/report.md and report.json

Everything is recomputed from the run's files (results.jsonl, controls.jsonl, questions/*/ingest.json, public.json,
environment.json); the dataset itself is not read. Sections:

- accuracy per question type, abstention and all, per arm: the mean over questions of each question's share of right
  repeats, a 95% interval by bootstrap over questions (10000 resamples, seed 0) and the repeat spread (the lowest and highest
  accuracy of a single repeat). A turn that did not finish counts as wrong; a reply the judge never answered is left out
  and counted;
- paired arm differences: for every two arms, per group, the mean of the per-question difference with a bootstrap interval
  over the questions both arms were asked;
- retrieval, the auto hit (what recall put in front of the model) and the search hit (what the `search_memory` tool
  returned) apart, with Wilson intervals: the rate of questions whose evidence session was reached, the rate with every
  evidence session reached, the share of turns that searched, and how a search hit was read (the memory event's `searched`
  items, the items the tool event names, or matched by words);
- ingest: how many turns the index cut (a turn longer than the index keeps), per type with single-session-assistant on its own
  (its answers are in the assistant's long replies), the time and notes per ingest, and any embedding failure;
- replies asked again: per arm, the turns in which localmelo asked the model again after a reply that was no answer (empty,
  only thinking, or a tool call written as text; the counts per notice kind), the turns that ended on a tool call written as
  text, and the model requests retried after an error; rows from before these fields were recorded are counted apart;
- the judge: replies that needed a second ask (`judge_retry`), replies that stayed neither yes nor no (`judge_unparsed`),
  unreachable judges and turns that were not judged;
- the control: the share of ordinary prompts for which recall used an item (false recall), and the prompt tokens and time to
  first token of those prompts against the questions';
- the sampling each arm ran with (`memory.across.report.sampling_line`).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memory.across.report import ms, num, sampling_line, stats
from memory.across.report_v2 import rate

from . import data

BOOTSTRAP = 10000
SEED = 0
ALL = "all"
ROWS = (*data.TYPES, data.ABSTENTION, ALL)  # the accuracy table's rows


# --- statistics ---


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_ci(values: Sequence[float], resamples: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float] | None:
    """The 95% percentile interval of the mean of ``values``, resampling them (here the questions) with replacement.

    A fresh generator per call, so an interval does not depend on which intervals were computed before it."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]


# --- loading ---


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_run(run_dir: Path) -> dict[str, Any]:
    """The run's files. A row that appears twice (a cell run again) counts once, the last."""
    config = json.loads((run_dir / "public.json").read_text(encoding="utf-8")).get("config", {})
    results = {(r["question_id"], r["arm"], r["repeat"]): r for r in read_jsonl(run_dir / "results.jsonl")}
    controls = {(r["question_id"], r["arm"]): r for r in read_jsonl(run_dir / "controls.jsonl")}
    ingests = {}
    for path in sorted((run_dir / "questions").glob("*/ingest.json")) if (run_dir / "questions").is_dir() else []:
        ingests[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    env_path = run_dir / "environment.json"
    questions_path = run_dir / "questions.json"
    return {
        "config": config, "rows": list(results.values()), "controls": list(controls.values()), "ingests": ingests,
        "environment": json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {},
        "questions": json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.exists() else [],
    }


def group_of(row: Mapping[str, Any]) -> str:
    return data.ABSTENTION if row.get("abstention") else str(row.get("question_type"))


def in_group(row: Mapping[str, Any], name: str) -> bool:
    return name == ALL or group_of(row) == name


# --- accuracy ---


def per_question(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """arm -> question id -> the 0/1 verdicts of its repeats (a reply the judge never answered is not in it)."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r.get("correct") is not None:
            out.setdefault(r["arm"], {}).setdefault(r["question_id"], []).append(1.0 if r["correct"] else 0.0)
    return out


def accuracy_cell(rows: Sequence[Mapping[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    """One arm and group: questions, mean accuracy, its bootstrap interval, right/asked cells, and the repeat spread."""
    scores: dict[str, list[float]] = {}
    by_repeat: dict[int, list[float]] = {}
    for r in rows:
        if r.get("correct") is None:
            continue
        scores.setdefault(r["question_id"], []).append(1.0 if r["correct"] else 0.0)
        by_repeat.setdefault(int(r["repeat"]), []).append(1.0 if r["correct"] else 0.0)
    per_q = [sum(v) / len(v) for v in scores.values()]
    ci = bootstrap_ci(per_q, resamples, seed)
    repeats = {k: mean(v) for k, v in sorted(by_repeat.items())}
    return {
        "questions": len(per_q), "cells": sum(len(v) for v in scores.values()), "right": int(sum(sum(v) for v in scores.values())),
        "accuracy": mean(per_q), "ci": list(ci) if ci else None,
        "repeats": repeats, "spread": [min(repeats.values()), max(repeats.values())] if len(repeats) > 1 else None,
    }


def accuracy_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str], resamples: int, seed: int) -> dict[str, dict[str, Any]]:
    return {
        g: {a: accuracy_cell([r for r in rows if r["arm"] == a and in_group(r, g)], resamples, seed) for a in arms}
        for g in ROWS
    }


def paired_table(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str], resamples: int, seed: int,
) -> list[dict[str, Any]]:
    """For every two arms (in the order given) and group: the mean of (a - b) over the questions both were asked, the
    bootstrap interval of that mean, and how many questions went each way."""
    scores = per_question(rows)
    group_of_q = {r["question_id"]: group_of(r) for r in rows}
    out = []
    for a, b in itertools.combinations(arms, 2):
        for g in ROWS:
            shared = sorted(q for q in set(scores.get(a, {})) & set(scores.get(b, {})) if g == ALL or group_of_q[q] == g)
            diffs = [sum(scores[a][q]) / len(scores[a][q]) - sum(scores[b][q]) / len(scores[b][q]) for q in shared]
            ci = bootstrap_ci(diffs, resamples, seed)
            out.append({
                "a": a, "b": b, "group": g, "questions": len(diffs), "diff": mean(diffs), "ci": list(ci) if ci else None,
                "a_better": sum(1 for d in diffs if d > 0), "b_better": sum(1 for d in diffs if d < 0),
            })
    return out


# --- retrieval ---


def retrieval_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Hit rates of one arm and group over the turns of questions that have an answer session."""
    mine = [r for r in rows if (r.get("retrieval") or {}).get("answer_sessions")]
    ret = [r["retrieval"] for r in mine]
    searched = [x for x in ret if x.get("searches")]
    return {
        "turns": len(mine),
        "auto_hit": rate(sum(1 for x in ret if x.get("auto_hit")), len(ret)),
        "auto_all": rate(sum(1 for x in ret if x.get("auto_cover") == 1.0), len(ret)),
        "searched": rate(len(searched), len(ret)),
        "search_hit": rate(sum(1 for x in searched if x.get("search_hit")), len(searched)),
        "either": rate(sum(1 for x in ret if x.get("auto_hit") or x.get("search_hit")), len(ret)),
        "via_memory": sum(1 for x in searched if x.get("search_via") == "memory"),
        "via_items": sum(1 for x in searched if x.get("search_via") == "items"),
        "via_text": sum(1 for x in searched if x.get("search_via") == "text"),
        "hints": sum(1 for x in ret if x.get("search_hint") not in (None, "")),
    }


def retrieval_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {g: {a: retrieval_cell([r for r in rows if r["arm"] == a and in_group(r, g)]) for a in arms} for g in ROWS}


# --- ingest ---


def ingest_table(ingests: Mapping[str, Mapping[str, Any]], questions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per question type (and for all, and for all but single-session-assistant): the turns the index cut, the time, the notes."""
    type_of = {q["question_id"]: q["question_type"] for q in questions}
    keep = ("single-session-assistant",)

    def summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        turns, cut = sum(int(r.get("turns") or 0) for r in reports), sum(int(r.get("cut") or 0) for r in reports)
        return {
            "ingests": len(reports), "turns": turns, "cut": cut, "cut_rate": rate(cut, turns),
            "skipped": sum(int(r.get("skipped") or 0) for r in reports), "trimmed": sum(int(r.get("trimmed") or 0) for r in reports),
            "seconds": stats([num(r.get("seconds")) for r in reports]), "notes": stats([num(r.get("notes")) for r in reports]),
            "embedded_chars": stats([num(r.get("embedded_chars")) for r in reports]),
            "errors": sum(1 for r in reports if r.get("error")),
        }

    out = {t: summary([r for q, r in ingests.items() if type_of.get(q) == t]) for t in data.TYPES}
    out[ALL] = summary(list(ingests.values()))
    out["all but single-session-assistant"] = summary([r for q, r in ingests.items() if type_of.get(q) not in keep])
    return out


# --- judge ---


def judge_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, int]]:
    out = {}
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        judged = [r["judge"] for r in mine if isinstance(r.get("judge"), Mapping)]
        out[arm] = {
            "rows": len(mine), "judged": len(judged), "not_judged": len(mine) - len(judged),
            "judge_retry": sum(int(r.get("judge_retry") or 0) for r in mine),
            "judge_unparsed": sum(int(r.get("judge_unparsed") or 0) for r in mine),
            "judge_error": sum(1 for j in judged if j.get("error")),
        }
    return out


REPLY_KINDS = ("empty_reply", "unfinished_answer", "tool_call_as_text")


def reply_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, int]]:
    """Per arm, the no-answer replies localmelo asked the model again for (from each row's ``notices``)."""
    out = {}
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        seen = [r for r in mine if isinstance(r.get("notices"), Mapping)]
        cell = {
            "rows": len(mine), "not_recorded": len(mine) - len(seen),
            "turns_asked_again": sum(1 for r in seen if int(r.get("reply_retries") or 0)),
            "asked_again": sum(int(r.get("reply_retries") or 0) for r in seen),
            "final_text_tool_call": sum(1 for r in seen if r.get("final_text_tool_call")),
            "request_retry": sum(int(r["notices"].get("retry") or 0) for r in seen),
        }
        for kind in REPLY_KINDS:
            cell[kind] = sum(int(r["notices"].get(kind) or 0) for r in seen)
        out[arm] = cell
    return out


# --- why a question was missed ---

MISS_STEPS = (
    "turn failed", "abstention: answered anyway", "2 evidence cut at ingest, not retrieved", "3 not retrieved",
    "4 retrieved, evidence cut at ingest", "5a session retrieved, evidence turn not shown",
    "5b evidence turn shown, not used by the reader", "5 retrieved, evidence turn not checked")
EVIDENCE_SEEN = 0.6  # the share of an evidence turn's words the model must have been shown for it to count as shown


def miss_step(row: Mapping[str, Any]) -> str | None:
    """The step a wrong answer lost the evidence at (the spec's 1-6 attribution), from the row alone; None for a right one.

    Ingest (2): an evidence turn (``has_answer``) is longer than the index keeps (``evidence.best`` < 1). Retrieval (3):
    neither the recall block nor a search reached an answer session. Then (5) one did and the answer is still wrong: 5a
    the model was shown other turns of the session but not the evidence turn (``retrieval.evidence_seen`` below
    EVIDENCE_SEEN; retrieval, not the reader), 5b it was shown the evidence turn (the reader); rows without
    ``evidence_seen`` are not split."""
    if row.get("correct"):
        return None
    if row.get("state") != "finished":
        return "turn failed"
    if row.get("abstention"):
        return "abstention: answered anyway"
    ret = row.get("retrieval") or {}
    best = (row.get("evidence") or {}).get("best")
    cut = best is not None and best < 1.0
    if not (ret.get("auto_hit") or ret.get("search_hit")):
        return "2 evidence cut at ingest, not retrieved" if cut else "3 not retrieved"
    if cut:
        return "4 retrieved, evidence cut at ingest"
    seen = ret.get("evidence_seen")
    if seen is None:
        return "5 retrieved, evidence turn not checked"
    return "5a session retrieved, evidence turn not shown" if seen < EVIDENCE_SEEN else "5b evidence turn shown, not used by the reader"


def miss_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, int]]:
    """Per arm: the wrong answers by miss_step."""
    out = {}
    for arm in arms:
        steps = [miss_step(r) for r in rows if r["arm"] == arm]
        out[arm] = {step: sum(1 for s in steps if s == step) for step in MISS_STEPS}
    return out


# --- control ---


def control_table(
    controls: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], arms: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Per arm: the control cells, the false recalls among them, and prompt tokens / time to first token for the control
    prompts and, for comparison, for the questions."""
    out = {}
    for arm in arms:
        mine = [c for c in controls if c["arm"] == arm]
        asked = [r for r in rows if r["arm"] == arm and r.get("state") == "finished"]
        finished = [c for c in mine if c.get("state") == "finished"]
        out[arm] = {
            "cells": len(mine), "failed": len(mine) - len(finished),
            "false_recall": rate(sum(1 for c in finished if c.get("false_recall")), len(finished)),
            "items_used": stats([float(len(c.get("used") or [])) for c in finished]),
            "prompt_tokens": stats([num(c.get("prompt_tokens")) for c in finished]),
            "ttft_ms": stats([num(c.get("ttft_ms")) for c in finished]),
            "question_prompt_tokens": stats([num(r.get("prompt_tokens")) for r in asked]),
            "question_ttft_ms": stats([num(r.get("ttft_ms")) for r in asked]),
        }
    return out


# --- the report ---


def analyse(run_dir: Path, resamples: int = BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    run = load_run(run_dir)
    config = run["config"]
    rows = run["rows"]
    arms = [a for a in (config.get("arms") or {}) if any(r["arm"] == a for r in rows)] or sorted({r["arm"] for r in rows})
    expected = len(config.get("questions") or []) * len(config.get("arms") or {}) * int(config.get("repeats") or 1)
    notes = []
    if rows and len(rows) < expected:
        notes.append(f"{len(rows)} of {expected} question cells have a row; the rest were not run (yet).")
    pending = sum(1 for r in rows if isinstance(r.get("judge"), Mapping) and r["judge"].get("error"))
    if pending:
        notes.append(f"{pending} replies were not judged (the judge could not be reached); run the driver again to judge them.")
    if any(i.get("error") for i in run["ingests"].values()):
        notes.append("The embedding model failed during some ingests; the part that failed was indexed by words only.")
    return {
        "run_dir": run_dir.name, "config": config, "arms": arms, "notes": notes, "environment": run["environment"],
        "resamples": resamples, "seed": seed, "questions": len(config.get("questions") or run["questions"]),
        "accuracy": accuracy_table(rows, arms, resamples, seed), "paired": paired_table(rows, arms, resamples, seed),
        "retrieval": retrieval_table(rows, arms), "ingest": ingest_table(run["ingests"], run["questions"]),
        "judge": judge_table(rows, arms), "replies": reply_table(rows, arms), "control": control_table(run["controls"], rows, arms),
        "misses": miss_table(rows, arms),
        "rows": [
            {k: r.get(k) for k in ("question_id", "arm", "repeat", "state", "correct")}
            | {"auto_hit": (r.get("retrieval") or {}).get("auto_hit"), "search_hit": (r.get("retrieval") or {}).get("search_hit"),
               "miss_step": miss_step(r)}
            for r in rows
        ],
    }


def table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    rule = "|" + "|".join("---" for _ in head) + "|"
    return ["| " + " | ".join(head) + " |", rule, *("| " + " | ".join(r) + " |" for r in body), ""]


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}"


def acc(cell: Mapping[str, Any]) -> str:
    """"71.2 [62.0, 79.5]" (percent) and, when there are repeats, "r 70.0-72.5"."""
    if not cell["questions"]:
        return "-"
    ci = cell["ci"]
    text = f"{pct(cell['accuracy'])} [{pct(ci[0])}, {pct(ci[1])}]" if ci else pct(cell["accuracy"])
    spread = f" · r {pct(cell['spread'][0])}-{pct(cell['spread'][1])}" if cell["spread"] else ""
    return f"{text}{spread} (n={cell['questions']})"


def hit(cell: Mapping[str, Any]) -> str:
    """"57.1% [34.0, 78.2] (8/14)" for a rate dict made by report_v2.rate; "-" without trials."""
    if not cell["n"]:
        return "-"
    return f"{pct(cell['p'])} [{pct(cell['lo'])}, {pct(cell['hi'])}] ({cell['k']}/{cell['n']})"


def med(s: Mapping[str, Any]) -> str:
    return f"{ms(s['median'])} / {ms(s['p90'])} (n={s['n']})"


def render(rep: Mapping[str, Any]) -> str:
    arms: list[str] = rep["arms"]
    cfg = rep["config"]
    if not arms:
        return f"# LongMemEval-S: {rep['run_dir']}\n\nNo cell has run yet.\n"
    start = (rep["environment"] or {}).get("start") or {}
    stage = cfg.get("stage") or {}
    data_ = cfg.get("data") or {}

    def one_line(text: Any, default: str) -> str:
        return " ".join(str(text or "").split()) or default

    out = [f"# LongMemEval-S: {rep['run_dir']}", ""]
    out += [
        f"localmelo `{(cfg.get('localmelo_sha') or '')[:10]}`, benchmark driver `{(start.get('bench_sha') or '')[:10] or 'unknown'}`, "
        f"chat model `{cfg.get('model')}`, embedding model `{cfg.get('embedding_model')}`, judge `{cfg.get('judge_model')}`.",
        f"Data: `{data_.get('name')}` (sha256 `{(data_.get('sha256') or '')[:12]}`); {rep['questions']} questions (stage set: "
        f"{stage.get('per_type')} per type + {stage.get('abstention')} abstention, seed {stage.get('seed')}), {cfg.get('repeats')} repeats.",
        f"Machine: {one_line(start.get('sw_vers'), 'unknown')} | {one_line(start.get('xcodebuild_version'), 'no Xcode')}.",
        "",
    ]
    out += [f"> {n}" for n in rep["notes"]] + ([""] if rep["notes"] else [])
    arm_cfg = cfg.get("arms") or {}
    out += ["## Arms", ""]
    out += table(["arm", "melo serve -X", "config.toml", "sampling"], [
        [a, " ".join(f"`{x}`" for x in (arm_cfg.get(a) or {}).get("x") or []) or "-",
         ", ".join(f"`{k}={v}`" for k, v in ((arm_cfg.get(a) or {}).get("config") or {}).items()) or "-",
         sampling_line((arm_cfg.get(a) or {}).get("extra_body"))] for a in arms])

    out += ["## Accuracy", "",
            "Percent right, with the 95% bootstrap interval over questions in brackets "
            f"({rep['resamples']} resamples, seed {rep['seed']}); `r a-b` is the lowest and highest accuracy of a single repeat. "
            "A question's score is its share of right repeats. Abstention groups the `_abs` questions of every type.", ""]
    out += table(["group", *arms], [[g, *(acc(rep["accuracy"][g][a]) for a in arms)] for g in ROWS])

    if len(arms) > 1:
        out += ["## Paired arm differences", "",
                "Per question, the first arm's score minus the second's, over the questions both were asked; the mean difference "
                "in percentage points with the 95% bootstrap interval over those questions. `+n / -n`: questions the first / "
                "second arm did better on.", ""]
        body = []
        for p in rep["paired"]:
            ci = p["ci"]
            diff = "-" if p["diff"] is None else f"{100 * p['diff']:+.1f}" + (f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else "")
            body.append([f"{p['a']} - {p['b']}", p["group"], str(p["questions"]), diff, f"+{p['a_better']} / -{p['b_better']}"])
        out += table(["arms", "group", "questions", "difference (pp)", "better"], body)

    out += ["## Retrieval", "",
            "Over the turns of questions that have an answer session. Auto hit: an item of `turn.memory.used` (recall's own pick) "
            "comes from an answer session; all: from every answer session. Search: the turn called `search_memory`; search hit: a "
            "result is from an answer session (of the turns that searched). Either: the auto or the search hit. Rates with 95% "
            "Wilson intervals.", ""]
    for g in ROWS:
        cells = rep["retrieval"][g]
        if not any(cells[a]["turns"] for a in arms):
            continue
        out += [f"**{g}**", ""]
        out += table(["", *arms], [
            ["auto hit", *(hit(cells[a]["auto_hit"]) for a in arms)],
            ["auto hit, all answer sessions", *(hit(cells[a]["auto_all"]) for a in arms)],
            ["turns that searched", *(hit(cells[a]["searched"]) for a in arms)],
            ["search hit (of those)", *(hit(cells[a]["search_hit"]) for a in arms)],
            ["either", *(hit(cells[a]["either"]) for a in arms)],
            ["search hit read from memory event / tool items / words",
             *(f"{cells[a]['via_memory']} / {cells[a]['via_items']} / {cells[a]['via_text']}" for a in arms)],
            ["turns with a `search_hint`", *(str(cells[a]["hints"]) for a in arms)]])

    ing = rep["ingest"]
    out += ["## Ingest", "",
            "`melo memory ingest` indexes a turn up to the length the index keeps; a longer one is cut and its tail cannot be "
            "recalled. single-session-assistant questions ask about what the assistant said, and its replies are the long ones, "
            "so it is shown on its own.", ""]
    shown = [(t, ing[t]) for t in (*data.TYPES, ALL, "all but single-session-assistant") if ing[t]["ingests"]]
    out += table(
        ["type", "ingests", "turns", "cut", "cut rate", "seconds (median / p90)", "notes (median / p90)", "skipped", "trimmed",
         "embedding errors"],
        [[("**" + t + "**") if t == "single-session-assistant" else t, str(s["ingests"]), str(s["turns"]), str(s["cut"]),
          hit(s["cut_rate"]), med(s["seconds"]), med(s["notes"]), str(s["skipped"]), str(s["trimmed"]), str(s["errors"])]
         for t, s in shown])

    ms = rep["misses"]
    out += ["## Why answers were wrong", "",
            "Each wrong answer at the step it lost the evidence: cut at ingest (an evidence turn is longer than the index keeps), "
            "not retrieved (neither the recall block nor a search reached an answer session), the session retrieved but its "
            f"evidence turn not shown (under {EVIDENCE_SEEN:.0%} of its words in what recall and search showed the model), "
            "or shown but not used. "
            "From the row alone; read the answers before trusting a step.", ""]
    out += table(["", *arms], [[step, *(str(ms[a][step]) for a in arms)] for step in MISS_STEPS])

    rp = rep["replies"]
    out += ["## Replies asked again", "",
            "A reply that was no answer (empty, only thinking, or a tool call written as text) makes localmelo ask the model "
            "again; the notice counts include the reply that ended a turn. Builds before 8bb1b123 send no `tool_call_as_text` "
            "notice, so their text tool calls are not counted here. Rows not recorded: written before the driver kept notices.", ""]
    out += table(["", *arms], [
        ["turns asked again", *(f"{rp[a]['turns_asked_again']} / {rp[a]['rows'] - rp[a]['not_recorded']}" for a in arms)],
        ["times asked again", *(str(rp[a]["asked_again"]) for a in arms)],
        *([kind, *(str(rp[a][kind]) for a in arms)] for kind in REPLY_KINDS),
        ["turn ended on a text tool call", *(str(rp[a]["final_text_tool_call"]) for a in arms)],
        ["model requests retried", *(str(rp[a]["request_retry"]) for a in arms)],
        ["rows not recorded", *(str(rp[a]["not_recorded"]) for a in arms)]])

    out += ["## Judge", "",
            "`judge_retry`: the first reply was neither yes nor no, so it was asked again without sampling fields. `judge_unparsed`: "
            "the second was no clearer (graded wrong). Not judged: the turn did not finish (graded wrong) or the judge was "
            "unreachable (left out).", ""]
    jd = rep["judge"]
    out += table(["", *arms], [
        ["rows", *(str(jd[a]["rows"]) for a in arms)], ["judged", *(str(jd[a]["judged"]) for a in arms)],
        ["judge_retry", *(str(jd[a]["judge_retry"]) for a in arms)],
        ["judge_unparsed", *(str(jd[a]["judge_unparsed"]) for a in arms)],
        ["judge unreachable", *(str(jd[a]["judge_error"]) for a in arms)],
        ["not judged (turn failed)", *(str(jd[a]["not_judged"]) for a in arms)]])

    ct = rep["control"]
    if any(ct[a]["cells"] for a in arms):
        out += ["## Control: ordinary prompts", "",
                "One prompt that needs no memory, asked on an ingested HOME. False recall: recall used at least one item for it. "
                "Prompt tokens and chat time to first token (the timing proxy: the first byte of the agent's request, a token for a "
                "request that streams) are medians / p90; the question rows are the benchmark's own turns for comparison.", ""]
        out += table(["", *arms], [
            ["control cells (failed)", *(f"{ct[a]['cells']} ({ct[a]['failed']})" for a in arms)],
            ["false recall", *(hit(ct[a]["false_recall"]) for a in arms)],
            ["items used per control prompt", *(med(ct[a]["items_used"]) for a in arms)],
            ["control prompt tokens", *(med(ct[a]["prompt_tokens"]) for a in arms)],
            ["control first token ms", *(med(ct[a]["ttft_ms"]) for a in arms)],
            ["question prompt tokens", *(med(ct[a]["question_prompt_tokens"]) for a in arms)],
            ["question first token ms", *(med(ct[a]["question_ttft_ms"]) for a in arms)]])
    return "\n".join(out).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP, help="bootstrap resamples (default %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED, help="bootstrap seed (default %(default)s)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rep = analyse(args.run_dir, args.resamples, args.seed)
    (args.run_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "report.md").write_text(render(rep), encoding="utf-8")
    print(f"wrote {args.run_dir / 'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
