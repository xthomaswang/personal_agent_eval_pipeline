"""The E3 report: cache efficiency, accuracy and reflection health per arm, and the decision rule.

    cd eval
    uv run python agent_ability/e3/report.py <dir> [--localmelo <clone>]   # writes <dir>/report.md and report.json

Reads <dir>/e3.json and each run's directory (calls.jsonl, bodies/, summary.json). Nothing here
needs a model or a server, and a partial study reports what it has.

Definitions:
- cache efficiency: sum(cached_tokens) / sum(prompt_tokens) over a run's model calls except its first,
  from usage_raw of the proxy's calls.jsonl; per arm both pooled over its runs and the median of the
  per-run values;
- a reflection request is one whose last user message starts with localmelo's REFLECTION_PROMPT, or a
  tool-less one that carries that text in a system message (the default mode's, and the fixed-prefix
  mode's fallback and retry). Tool-less requests without the text are counted apart (toolless_other);
- a reflection is the first request of a chain: a reflection request that follows an unreadable
  reflection request at once is its retry, not a new reflection. It is unreadable when its reply has no
  JSON object with a recommended_action key;
- loss: S0 passed a task in every repeat and the arm did not; discordant: the arm's pass count of a task
  differs from S0's (such pairs are listed for a rerun).
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

AGENT_ABILITY = Path(__file__).resolve().parent.parent
EVAL_ROOT = AGENT_ABILITY.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from bench.proxy.steps import is_generation, is_ok  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "vendor" / "localmelo"
PROMPT_SOURCE = Path("melo") / "agent" / "_chat.py"
# A tie in cache efficiency goes to the simpler arm
SIMPLICITY = ("S0", "C1", "S2", "B1", "B2")
UNREADABLE_MARGIN = 0.02


def reflection_prompt(localmelo: Path) -> str:
    """REFLECTION_PROMPT, read from the clone's source without importing it."""
    tree = ast.parse((localmelo / PROMPT_SOURCE).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "REFLECTION_PROMPT" for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ValueError(f"REFLECTION_PROMPT not found in {localmelo / PROMPT_SOURCE}")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def is_reflection_request(body: dict[str, Any], prompt: str) -> bool | None:
    """True for a reflection request, None for a tool-less request that is not one, else False."""
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is not None and _text(last_user.get("content")).startswith(prompt):
        return True
    if not body.get("tools"):
        if any(prompt in _text(m.get("content")) for m in messages if m.get("role") == "system"):
            return True
        return None
    return False


def reply_text(raw: bytes, name: str) -> str:
    """The assistant's content of a stored response body (chat completions, JSON or SSE)."""
    text = raw.decode("utf-8", errors="replace")
    if name.endswith(".sse"):
        parts = []
        for line in text.splitlines():
            if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
                continue
            try:
                event = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            for choice in (event.get("choices") or []) if isinstance(event, dict) else []:
                delta = choice.get("delta") or choice.get("message") or {}
                parts.append(_text(delta.get("content")))
        return "".join(parts)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ""
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if choices and isinstance(choices[0], dict):
        return _text((choices[0].get("message") or {}).get("content"))
    return ""


def reply_excerpt(raw: bytes, name: str, limit: int = 200) -> str:
    """Why a reply could not be read: its content, or, when it has none, the tool call(s) it made instead."""
    text = reply_text(raw, name)
    if text.strip():
        return text[:limit]
    calls = []
    try:
        obj = json.loads(raw.decode("utf-8", errors="replace")) if not name.endswith(".sse") else None
    except json.JSONDecodeError:
        obj = None
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if choices and isinstance(choices[0], dict):
        for call in (choices[0].get("message") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            calls.append(f"{fn.get('name')}({str(fn.get('arguments') or '')[:80]})")
        finish = choices[0].get("finish_reason")
        if calls:
            return f"<tool call instead of text, finish {finish}: {'; '.join(calls)}>"[:limit]
        return f"<empty reply, finish {finish}>"
    return "<empty reply>"


def has_decision(text: str) -> bool:
    """Whether text holds a JSON object with a recommended_action key (markdown fences and prose around it are fine)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict) and "recommended_action" in obj:
            return True
        start = text.find("{", start + 1)
    return False


def _calls(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "calls.jsonl"
    rows = []
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and is_generation(row) and is_ok(row):
            rows.append(row)
    return sorted(rows, key=lambda r: (r.get("mono_start") or 0, r.get("call_index") or 0))


def _usage(call: dict[str, Any]) -> tuple[int, int] | None:
    """(prompt_tokens, cached_tokens) of a call, or None without usage."""
    usage = call.get("usage_raw")
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int):
        return None
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return usage["prompt_tokens"], cached if isinstance(cached, int) else 0


def _body(run_dir: Path, call: dict[str, Any], kind: str) -> tuple[bytes, str] | None:
    rel = (call.get("bodies") or {}).get(kind)
    if not rel or not (run_dir / rel).is_file():
        return None
    return (run_dir / rel).read_bytes(), rel


def run_metrics(run_dir: Path, prompt: str) -> dict[str, Any]:
    """Tokens, cache and reflection counts of one run directory."""
    calls = _calls(run_dir)
    used = [(c, u) for c in calls if (u := _usage(c)) is not None]
    after_first = used[1:]
    prompt_tokens = sum(u[0] for _, u in used)
    cached_all = sum(u[1] for _, u in used)
    later_prompt = sum(u[0] for _, u in after_first)
    later_cached = sum(u[1] for _, u in after_first)
    reflection_requests = reflections = unreadable = other = 0
    unreadable_replies: list[str] = []   # the first 200 characters of each unreadable reflection's reply
    previous_unreadable = False
    for call in calls:
        request = _body(run_dir, call, "request")
        try:
            body = json.loads(request[0]) if request else None
        except json.JSONDecodeError:
            body = None
        kind = is_reflection_request(body, prompt) if isinstance(body, dict) else False
        if kind is None:
            other += 1
        if kind is not True:
            previous_unreadable = False
            continue
        reflection_requests += 1
        response = _body(run_dir, call, "response")
        readable = response is not None and has_decision(reply_text(*response))
        if not previous_unreadable:
            reflections += 1
            unreadable += 0 if readable else 1
            if not readable:
                unreadable_replies.append(reply_excerpt(*response) if response is not None else "<no response body>")
        previous_unreadable = not readable
    return {
        "model_calls": len(used),
        "prompt_tokens": prompt_tokens,
        "uncached_tokens": prompt_tokens - cached_all,
        "later_prompt_tokens": later_prompt,
        "later_cached_tokens": later_cached,
        "cache_efficiency": later_cached / later_prompt if later_prompt else None,
        "reflection_requests": reflection_requests,
        "reflections": reflections,
        "unreadable_reflections": unreadable,
        "unreadable_replies": unreadable_replies,
        "toolless_other": other,
    }


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


def _median(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.median(kept) if kept else None


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _result_line(out: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = out / record["run_dir"] / "summary.json" if record.get("run_dir") else None
    if path is None or not path.exists():
        return {}
    return ((json.loads(path.read_text()).get("harness_report") or {}).get("run_record")) or {}


def load_runs(out: Path, state: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    runs = []
    for record in state["runs"].values():
        if record.get("state") not in ("finished", "interrupted"):
            continue
        line = _result_line(out, record)
        metrics = run_metrics(out / record["run_dir"], prompt) if record.get("run_dir") else {}
        runs.append(
            {
                "run_id": record["run_id"],
                "index": record["index"],
                "arm": record["arm"],
                "exercise": record["exercise"],
                "repeat": record["repeat"],
                "state": record["state"],
                "verdict": record.get("verdict"),
                "passed": bool(record.get("passed")) and record["state"] == "finished",
                "wall_s": record.get("wall_s"),
                "attempts": record.get("attempts") if record.get("attempts") is not None else line.get("attempts"),
                "n_aux": record.get("n_aux") if record.get("n_aux") is not None else line.get("n_aux"),
                **metrics,
            }
        )
    return sorted(runs, key=lambda r: r["index"])


def summarize_arm(rows: Sequence[dict[str, Any]], exercises: Sequence[str], repeats: int) -> dict[str, Any]:
    finished = [r for r in rows if r["state"] == "finished"]
    pass_by = {(r["exercise"], r["repeat"]): r["passed"] for r in rows}
    later_prompt = sum(r.get("later_prompt_tokens") or 0 for r in rows)
    later_cached = sum(r.get("later_cached_tokens") or 0 for r in rows)
    reflections = sum(r.get("reflections") or 0 for r in rows)
    unreadable = sum(r.get("unreadable_reflections") or 0 for r in rows)
    per_task: dict[str, Any] = {}
    for exercise in exercises:
        task_rows = [r for r in finished if r["exercise"] == exercise]
        per_task[exercise] = {
            "passes": [bool(pass_by.get((exercise, n), False)) for n in range(1, repeats + 1)],
            "n_pass": sum(1 for n in range(1, repeats + 1) if pass_by.get((exercise, n))),
            "runs": len(task_rows),
            "prompt_tokens": _mean(r.get("prompt_tokens") for r in task_rows),
            "uncached_tokens": _mean(r.get("uncached_tokens") for r in task_rows),
            "steps": _mean(r.get("model_calls") for r in task_rows),
            "wall_s": _mean(r.get("wall_s") for r in task_rows),
        }
    return {
        "runs": len(rows),
        "interrupted": len(rows) - len(finished),
        "n_pass": sum(1 for r in rows if r["passed"]),
        "cache_efficiency": _rate(later_cached, later_prompt),
        "cache_efficiency_median": _median(r.get("cache_efficiency") for r in rows),
        "prompt_tokens_per_run": _mean(r.get("prompt_tokens") for r in finished),
        "uncached_tokens_per_run": _mean(r.get("uncached_tokens") for r in finished),
        "steps_per_run": _mean(r.get("model_calls") for r in finished),
        "wall_s_per_run": _mean(r.get("wall_s") for r in finished),
        "reflection_requests": sum(r.get("reflection_requests") or 0 for r in rows),
        "reflections": reflections,
        "unreadable_reflections": unreadable,
        "unreadable_rate": _rate(unreadable, reflections),
        "toolless_other": sum(r.get("toolless_other") or 0 for r in rows),
        "attempts": sum(r.get("attempts") or 0 for r in rows),
        "attempts_per_run": _mean(r.get("attempts") for r in finished),
        "n_aux": sum(r.get("n_aux") or 0 for r in rows),
        "n_aux_per_run": _mean(r.get("n_aux") for r in finished),
        "per_task": per_task,
    }


def decide(arms: dict[str, dict[str, Any]], exercises: Sequence[str], repeats: int) -> dict[str, Any]:
    """The plan's rule: highest cache efficiency among the arms that lose no task S0 passes and whose
    unreadable-reflection rate is at most 2 points above S0's; a tie goes to the simpler arm."""
    base = arms.get("S0")
    if base is None:
        return {"winner": None, "reason": "no S0 arm in this study", "arms": {}, "discordant": []}
    base_rate = base["unreadable_rate"] or 0.0
    verdicts: dict[str, Any] = {}
    discordant = []
    for name, arm in arms.items():
        losses = [
            e
            for e in exercises
            if base["per_task"][e]["n_pass"] == repeats and arm["per_task"][e]["n_pass"] < repeats
        ]
        rate = arm["unreadable_rate"] or 0.0
        rate_ok = rate <= base_rate + UNREADABLE_MARGIN + 1e-12
        verdicts[name] = {
            "cache_efficiency": arm["cache_efficiency"],
            "losses": losses,
            "unreadable_rate": arm["unreadable_rate"],
            "unreadable_rate_ok": rate_ok,
            "eligible": not losses and rate_ok and arm["cache_efficiency"] is not None,
        }
        if name == "S0":
            continue
        for e in exercises:
            if arm["per_task"][e]["n_pass"] != base["per_task"][e]["n_pass"]:
                discordant.append(
                    {
                        "task": e,
                        "arm": name,
                        "s0_passes": base["per_task"][e]["n_pass"],
                        "arm_passes": arm["per_task"][e]["n_pass"],
                        "loss": e in losses,
                    }
                )
    eligible = [n for n, v in verdicts.items() if v["eligible"]]
    winner = None
    if eligible:
        best = max(round(verdicts[n]["cache_efficiency"], 4) for n in eligible)
        tied = [n for n in eligible if round(verdicts[n]["cache_efficiency"], 4) == best]
        winner = min(tied, key=lambda n: SIMPLICITY.index(n) if n in SIMPLICITY else len(SIMPLICITY))
    return {
        "winner": winner,
        "reason": "no arm is eligible" if winner is None else f"{winner} has the highest cache efficiency among the eligible arms",
        "arms": verdicts,
        "discordant": discordant,
    }


def build_report(out: Path, localmelo: Path = VENDOR) -> dict[str, Any]:
    state = json.loads((out / "e3.json").read_text())
    config = state["config"]
    exercises, repeats = config["exercises"], config["repeats"]
    runs = load_runs(out, state, reflection_prompt(localmelo))
    arms = {a: summarize_arm([r for r in runs if r["arm"] == a], exercises, repeats) for a in config["arms"]}
    return {
        "schema_version": 1,
        "protocol": config.get("protocol"),
        "config": config,
        "complete": state.get("complete", False),
        "runs_recorded": len(runs),
        "runs_planned": len(exercises) * repeats * len(config["arms"]),
        "arms": arms,
        "decision": decide(arms, exercises, repeats),
        "runs": runs,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _unreadable_section(runs: Sequence[dict[str, Any]]) -> list[str]:
    """Each unreadable reflection's reply, first 200 characters, one line per reflection."""
    lines = [
        f"- `{r['run_id']}`: {' '.join(text.split()).replace('`', chr(39))}"
        for r in runs
        for text in r.get("unreadable_replies") or []
    ]
    return ["### Unreadable reflection replies (first 200 characters)", "", *lines, ""] if lines else []


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    arms = report["arms"]
    exercises = cfg["exercises"]
    lines = [
        "# E3: prompt-cache A/B of five localmelo flag sets",
        "",
        f"- localmelo `{cfg.get('localmelo_sha', 'unknown')[:7]}`; model `{cfg.get('model')}`; "
        f"{len(exercises)} tasks x {cfg['repeats']} repeats x {len(arms)} arms",
        f"- Runs recorded: {report['runs_recorded']} of {report['runs_planned']}"
        + ("" if report["complete"] else " (the study is not complete)"),
        "",
        "## Arms",
        "",
        "| Arm | Flags |",
        "|---|---|",
        *[f"| {a} | `{' '.join(f for f in flags if f not in ('--output-format', 'json')) or 'none'}` |" for a, flags in cfg["arms"].items()],
        "",
        "## Cache efficiency and cost",
        "",
        "Efficiency is cached / prompt tokens over each run's model calls except its first. Costs are means per run.",
        "",
        "| Arm | Pass | Efficiency (pooled) | Efficiency (median run) | Prompt tokens | Uncached tokens | Steps | Wall s | Attempts | n_aux |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        lines.append(
            f"| {name} | {arm['n_pass']}/{arm['runs']} | {_pct(arm['cache_efficiency'])} | "
            f"{_pct(arm['cache_efficiency_median'])} | {_num(arm['prompt_tokens_per_run'])} | "
            f"{_num(arm['uncached_tokens_per_run'])} | {_num(arm['steps_per_run'], 1)} | "
            f"{_num(arm['wall_s_per_run'])} | {_num(arm['attempts_per_run'], 2)} | {_num(arm['n_aux_per_run'], 1)} |"
        )
    lines += [
        "",
        "## Reflection health",
        "",
        "| Arm | Reflections | Unreadable | Unreadable rate | Reflection requests | Tool-less, not a reflection |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {n} | {a['reflections']} | {a['unreadable_reflections']} | {_pct(a['unreadable_rate'])} | "
            f"{a['reflection_requests']} | {a['toolless_other']} |"
            for n, a in arms.items()
        ],
        "",
        *_unreadable_section(report["runs"]),
        "## Passes per task (one mark per repeat)",
        "",
        "| Task | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for e in exercises:
        cells = [" ".join("P" if p else "-" for p in arm["per_task"][e]["passes"]) for arm in arms.values()]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per task (mean over finished repeats): prompt tokens / uncached tokens / steps / wall s",
        "",
        "| Task | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for e in exercises:
        cells = []
        for arm in arms.values():
            t = arm["per_task"][e]
            cells.append(
                f"{_num(t['prompt_tokens'])} / {_num(t['uncached_tokens'])} / {_num(t['steps'], 1)} / {_num(t['wall_s'])}"
            )
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    decision = report["decision"]
    lines += [
        "",
        "## Decision rule",
        "",
        "The arm with the highest cache efficiency that loses no task S0 passed in every repeat, and whose "
        "unreadable-reflection rate is at most 2 points above S0's. A tie goes to the simpler arm "
        f"({', '.join(SIMPLICITY)}).",
        "",
        "| Arm | Efficiency | Losses | Unreadable rate | Rate within +2 points | Eligible |",
        "|---|---:|---|---:|---|---|",
        *[
            f"| {n} | {_pct(v['cache_efficiency'])} | {', '.join(v['losses']) or 'none'} | "
            f"{_pct(v['unreadable_rate'])} | {'yes' if v['unreadable_rate_ok'] else 'no'} | "
            f"{'yes' if v['eligible'] else 'no'} |"
            for n, v in decision["arms"].items()
        ],
        "",
        f"**Result: {decision['winner'] or 'none'}** ({decision['reason']}).",
        "",
        "## Discordant (task, arm) pairs for a rerun",
        "",
    ]
    if decision["discordant"]:
        lines += [
            "| Task | Arm | S0 passes | Arm passes | Loss |",
            "|---|---|---:|---:|---|",
            *[
                f"| {d['task']} | {d['arm']} | {d['s0_passes']} | {d['arm_passes']} | {'yes' if d['loss'] else 'no'} |"
                for d in decision["discordant"]
            ],
        ]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="an E3 output directory (it has e3.json)")
    ap.add_argument("--localmelo", type=Path, default=VENDOR, help="the clone REFLECTION_PROMPT is read from")
    args = ap.parse_args(argv)
    out = args.run_dir.resolve()
    if not (out / "e3.json").exists():
        print(f"{out} has no e3.json", file=sys.stderr)
        return 2
    report = build_report(out, args.localmelo.resolve())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "report.md").write_text(render_markdown(report))
    print(f"{out / 'report.md'}: {report['runs_recorded']}/{report['runs_planned']} runs; result {report['decision']['winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
