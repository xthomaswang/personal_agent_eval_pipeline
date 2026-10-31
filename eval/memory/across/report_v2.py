"""Report of a memory-bench-v2 study (tasks_v2): the in-chat kinds, per kind and arm.

    cd eval
    uv run python -m memory.across.report <run-dir> --tasks memory/across/data/tasks_v2.yaml   # report.md, report.json

``report.main`` sends a run of the v2 protocol here. Everything is recomputed from the journals (cells/*/turns.jsonl),
the proxy logs and the task file, and the SQLite files of each finished cell's HOME, which are opened read-only
(stores.py). Sections:

- items that are not valid for an arm, with the reason, and left out of everything below. A long-range item is valid
  for an arm when, at the question, the fact turn is out of the working window (N0, M2) or still in it (FC), as the
  stored window and the message arithmetic around it say (stores.judge); a full-context chat must also keep every
  request under the prompt cap. Any item whose fact or teaching turn did not finish is invalid too;
- accuracy per kind and arm with Wilson 95% intervals: long range also split by `note_expected`, follow-up as Q1,
  Q2 and Q2 given Q1 right (a failed question turn counts as wrong);
- miss classes of every wrong answer (score.MISS_CLASSES) and, for the right long-range answers, where they came from;
- notes made in the long chats against what the note rules were expected to save;
- the paired comparison of the arms of each kind (exact McNemar, as the v1 report);
- turn wall time, recall time, time to the first event, the size of the agent's request and the recall block, with
  its characters read from `turn.memory.recall.block_chars` when the build reports it and estimated from the used
  items otherwise (the table says which); the `turn.memory.recall.context` each question had (off, used, none);
- with ``--compare <other run dir> [arm]``, the same question of the same arm in two runs of one task file (two
  localmelo builds, say) paired: exact McNemar per kind and the questions whose verdict differs. Without an arm, every
  arm both runs have.

An arm may be a variant of another (``base`` in the arms file, e.g. M2 with one more setting): it follows its base's
row in each kind's arm table, and the tables pair it with every other arm of the kind.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import data_v2, report, run, score, stores

FC_REQUEST_LIMIT = 100_000  # localmelo's prompt cap in characters; a request's bytes are at least its characters
NON_READ_TOOLS = ("file_write", "file_edit", "python_exec", "shell_exec")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval of k successes in n trials (proportions); None for no trials."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else None, "lo": ci[0] if ci else None, "hi": ci[1] if ci else None}


def ask_label(item: data_v2.Item, n: int) -> str:
    """The name of ask ``n`` (from 0) of an item in the tables: the kind, or kind.q<n> when it has several asks."""
    return item.kind if len(item.asks) == 1 else f"{item.kind}.q{n + 1}"


# --- one cell ---


@dataclass
class Ctx:
    """A cell as the row builders see it."""

    cell: Mapping[str, Any]
    arm: str
    base: str  # the arm whose row in each kind's arm table this arm follows (a variant of M2 follows M2)
    repeat: int
    off: bool  # the arm switches memory off (N0, FC)
    step_memory: bool  # the arm has an embedding model, so localmelo's legacy step memory runs
    finished: bool
    turns: Mapping[tuple[str, str], Mapping[str, Any]]
    req: Mapping[tuple[str, int, str, str], Mapping[str, Any]]
    notes: Mapping[str, list[dict[str, Any]]] | None  # saved notes by chat; None: memory off or not read
    problems: list[str] = field(default_factory=list)

    @property
    def home(self) -> Path:
        return Path(self.cell["dir"]) / "home"

    def chat_requests(self, phase_labels: Sequence[tuple[str, str]]) -> float | None:
        """The biggest agent request (bytes) over the given turns, from the proxy log; None without a log."""
        sizes = [v for key in phase_labels if (r := self.req.get((self.arm, self.repeat, *key))) and (v := r.get("req_bytes_max")) is not None]
        return max(sizes) if sizes else None


def saved_gold(ctx: Ctx, session: str | None, targets: Sequence[str]) -> tuple[bool | None, bool | None]:
    """(a note from chat ``session`` holds a gold string, the chat made any note); None, None when not known."""
    if ctx.off or ctx.notes is None or not session:
        return None, None
    made = ctx.notes.get(session, [])
    return any(score.contains(n["text"], t) for n in made for t in targets), bool(made)


def ask_row(ctx: Ctx, item: data_v2.Item, n: int, chat: Mapping[str, Any], gold_session: str | None, in_window: bool | None) -> dict[str, Any] | None:
    ask = item.asks[n]
    t = ctx.turns.get(("ask", ask.id))
    if t is None:
        return None
    grade = score.grade(ask, t.get("answer"))
    memory = t.get("memory")
    rec = t.get("recall") if isinstance(t.get("recall"), Mapping) else None
    recall = score.recall_score(memory, [gold_session] if gold_session else [])
    saved, _ = saved_gold(ctx, gold_session, ask.targets)
    failed = t.get("state") != "finished"
    chars, estimated = score.block_chars(memory, rec)
    req = ctx.req.get((ctx.arm, ctx.repeat, "ask", ask.id)) or {}
    valid = bool(chat["valid"])
    row = {
        "arm": ctx.arm, "repeat": ctx.repeat, "item": item.id, "qid": ask.id, "label": ask_label(item, n), "kind": item.kind, "lang": item.lang,
        "note_expected": item.note_expected, "valid": valid, "reason": chat["reason"], "state": t.get("state"),
        "outcome": grade.outcome, "correct": grade.correct, "stale": bool(grade.rejected), "abstained": grade.abstained, "answer": t.get("answer", ""),
        "gold_hit": bool(recall.get("hit_any")), "gold_rank": recall.get("gold_rank"), "n_used": recall["n_used"], "saved": saved,
        "remembered": [str(r.get("text") or "") for r in (memory or {}).get("remembered", []) if isinstance(r, Mapping)],
        "recall_ms": report.recall_ms(rec), "wall_ms": report.num(t.get("wall_ms")), "ttfe_ms": report.num(t.get("ttfe_ms")),
        "req_bytes": report.num(req.get("req_bytes")), "block_chars": chars, "block_estimated": estimated,
        "timed_out": bool(t.get("timed_out")), "denied": int(t.get("approvals_denied") or 0),
        "context": (rec or {}).get("context"),  # off | used | none, on builds that report it
        "miss": "", "source": "",
    }
    if valid and not grade.correct:
        row["miss"] = score.miss_class(
            memory_on=not ctx.off, failed=failed, saved=saved, recalled=row["gold_hit"], in_window=in_window)
    if valid and grade.correct and item.kind == "longrange":
        row["source"] = score.answer_source(memory, ask.targets, in_window=in_window, step_memory=ctx.step_memory)
    return row


def longrange_rows(ctx: Ctx, item: data_v2.Item) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ask = item.asks[0]
    rule = data_v2.KINDS[item.kind].arms[ctx.base]
    fact = ctx.turns.get(("long", f"{item.id}.t{item.fact_turn:02d}"))
    q = ctx.turns.get(("ask", ask.id))
    mine = [(phase, label) for (phase, label) in ctx.turns if label.startswith(f"{item.id}.") and phase in ("long", "ask")]
    session = (q or {}).get("session_id") or (fact or {}).get("session_id")
    window: stores.WindowFacts | None = None
    reason = ""
    if fact is None or fact.get("state") != "finished":
        reason = "the fact turn did not finish"
    elif q is None or not session:
        reason = "the chat did not reach the question"
    elif rule and not ctx.finished:
        reason = "the cell has not finished, so its stores are not read"
    elif rule:
        try:
            window = stores.window_facts(ctx.home, session, item.fact)
            valid, why = stores.judge(window, rule)
            reason = "" if valid else why
        except stores.StoreError as exc:
            reason = f"the gateway store cannot be read: {exc}"
    biggest = ctx.chat_requests(mine)
    if not reason and rule == "in" and biggest is not None and biggest >= FC_REQUEST_LIMIT:
        reason = f"a request of {biggest:,.0f} bytes, over the prompt cap ({FC_REQUEST_LIMIT:,} characters)"
    saved, made = saved_gold(ctx, session, ask.targets)
    tools = Counter(name for (phase, label), t in ctx.turns.items() if label.startswith(f"{item.id}.") and phase == "long" for name in t.get("tools") or [])
    chat = {
        "arm": ctx.arm, "repeat": ctx.repeat, "item": item.id, "kind": item.kind, "lang": item.lang, "note_expected": item.note_expected,
        "rule": rule, "valid": not reason, "reason": reason, "session_id": session, "window": asdict(window) if window else None,
        "fact_made_note": None if ctx.off or not fact or not fact.get("memory") else bool(fact["memory"].get("remembered")),
        "fact_note_saved": saved, "chat_made_a_note": made, "req_bytes_max": biggest, "tools": dict(tools),
        "denied": sum(int(t.get("approvals_denied") or 0) for key, t in ctx.turns.items() if key in mine),
    }
    row = ask_row(ctx, item, 0, chat, session, window.fact_in_window if window else None)
    return chat, [row] if row else []


def followup_rows(ctx: Ctx, item: data_v2.Item) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    teach = [ctx.turns.get(("teach", f"{item.id}.f{n}")) for n in (1, 2)]
    sessions = [(t or {}).get("session_id") for t in teach]
    reason = next((f"teaching turn f{n} did not finish" for n, t in enumerate(teach, 1) if t is None or t.get("state") != "finished"), "")
    chat = {
        "arm": ctx.arm, "repeat": ctx.repeat, "item": item.id, "kind": item.kind, "lang": item.lang, "note_expected": None,
        "rule": "", "valid": not reason, "reason": reason, "session_id": None, "window": None, "teach_sessions": sessions,
        "tools": {}, "denied": sum(int(t.get("approvals_denied") or 0) for t in teach if t),
    }
    rows = [ask_row(ctx, item, n, chat, sessions[n], None) for n in range(len(item.asks))]
    return chat, [r for r in rows if r]


def recall_rows(ctx: Ctx, item: data_v2.Item) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A recall question: nothing was taught, so it is always valid, and a miss is classed by whether an item that holds
    the gold string was in ``turn.memory.used``."""
    chat = {
        "arm": ctx.arm, "repeat": ctx.repeat, "item": item.id, "kind": item.kind, "lang": item.lang, "note_expected": None,
        "rule": "", "valid": True, "reason": "", "session_id": None, "window": None, "teach_sessions": [], "tools": {}, "denied": 0,
    }
    row = ask_row(ctx, item, 0, chat, None, None)
    if row is None:
        return chat, []
    t = ctx.turns.get(("ask", item.asks[0].id)) or {}
    used = score.used_items(t.get("memory"))
    holds = [u for u in used if any(x and score.contains(str(u.get("text") or ""), x) for x in item.asks[0].targets)]
    row.update(category=item.category, used_sources=[str(u.get("source") or "")[:80] for u in used])
    if not row["correct"]:
        row["miss"] = "failed" if t.get("state") != "finished" else ("recalled but not used" if holds else "not recalled")
    return chat, [row]


def sleep_rows(ctx: Ctx, item: data_v2.Item) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A sleep item: valid when every teaching turn finished; its question is graded and classed as a recall one."""
    teach = [ctx.turns.get(("teach", f"{item.id}.f{n}")) for n in range(1, len(item.facts) + 1)]
    chat, rows = recall_rows(ctx, item)
    reason = next((f"teaching turn f{n} did not finish" for n, t in enumerate(teach, 1) if t is None or t.get("state") != "finished"), "")
    chat.update(valid=not reason, reason=reason, teach_sessions=[(t or {}).get("session_id") for t in teach])
    for r in rows:
        r["valid"] = not reason
        if reason:
            r["miss"] = ""
    return chat, rows


ROW_BUILDERS = {"longrange": longrange_rows, "followup": followup_rows, "recall": recall_rows, "sleep": sleep_rows}


def cell_rows(
    cell: Mapping[str, Any], tasks: data_v2.TaskSet, req: Mapping[Any, Any], bases: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    info = cell["info"]
    off = run.memory_is_off(info.get("x") or [])
    finished = info.get("state") == "finished"
    ctx = Ctx(
        cell, info["arm"], bases.get(info["arm"], info["arm"]), info["cell"]["repeat"], off, bool(info.get("embedding")) and not off, finished,
        {(t["phase"], t["item"]): t for t in cell["turns"]}, req, None)
    if finished and not off:
        try:
            ctx.notes = stores.notes_by_chat(ctx.home)
        except stores.StoreError as exc:
            ctx.problems.append(f"{cell['id']}: the notes cannot be read ({exc}); misses are not classed as saved or not")
    ran = {t["item"].split(".")[0] for t in cell["turns"]}
    chats: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    for item in (i for i in tasks.items if i.id in ran and ctx.base in data_v2.KINDS[i.kind].arms):
        chat, rows = ROW_BUILDERS[item.kind](ctx, item)
        chats.append(chat)
        asks += rows
    return chats, asks, ctx.problems


# --- tables ---


def counts(rows: Sequence[Mapping[str, Any]], key: str, classes: Sequence[str]) -> dict[str, int]:
    found = Counter(r[key] for r in rows)
    return {c: found[c] for c in classes if found[c]} | {c: n for c, n in found.items() if c not in classes and c}


def accuracy(rows: Sequence[Mapping[str, Any]], arms: Sequence[str], tasks: data_v2.TaskSet) -> dict[str, dict[str, dict[str, Any]]]:
    """kind -> arm -> row name -> rate, over the valid items."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in data_v2.KINDS:
        out[kind] = {}
        for arm in arms:
            mine = [r for r in rows if r["valid"] and r["kind"] == kind and r["arm"] == arm]
            if not mine:
                continue
            by_label = {label: [r for r in mine if r["label"] == label] for label in dict.fromkeys(r["label"] for r in mine)}
            table: dict[str, dict[str, Any]] = {}
            if kind in data_v2.CATEGORIES:
                table["all"] = rate(sum(r["correct"] for r in mine), len(mine))
                for category in data_v2.CATEGORIES[kind]:
                    sub = [r for r in mine if r.get("category") == category]
                    if sub:
                        table[category] = rate(sum(r["correct"] for r in sub), len(sub))
            elif kind == "longrange":
                table["all"] = rate(sum(r["correct"] for r in mine), len(mine))
                for flag, name in ((False, "note not expected"), (True, "note expected")):
                    sub = [r for r in mine if r["note_expected"] is flag]
                    table[name] = rate(sum(r["correct"] for r in sub), len(sub))
            else:
                for label, sub in by_label.items():
                    table[label.removeprefix(kind + ".").upper()] = rate(sum(r["correct"] for r in sub), len(sub))
                first = {(r["repeat"], r["item"]): r["correct"] for r in by_label.get(f"{kind}.q1", [])}
                follow = [r for r in by_label.get(f"{kind}.q2", []) if first.get((r["repeat"], r["item"]))]
                table["Q2 given Q1 right"] = rate(sum(r["correct"] for r in follow), len(follow))
            out[kind][arm] = table
    return out


def latency(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in data_v2.KINDS:
        out[kind] = {}
        for arm in arms:
            mine = [r for r in rows if r["valid"] and r["kind"] == kind and r["arm"] == arm and r["state"] == "finished"]
            if not mine:
                continue
            out[kind][arm] = {
                "wall_ms": report.stats([r["wall_ms"] for r in mine]), "recall_ms": report.stats([r["recall_ms"] for r in mine]),
                "ttfe_ms": report.stats([r["ttfe_ms"] for r in mine]), "req_bytes": report.stats([r["req_bytes"] for r in mine]),
                "block_real": report.stats([r["block_chars"] for r in mine if r["block_chars"] is not None and not r["block_estimated"]]),
                "block_estimated": report.stats([r["block_chars"] for r in mine if r["block_chars"] is not None and r["block_estimated"]]),
            }
    return out


def kind_questions(tasks: data_v2.TaskSet, kind: str) -> SimpleNamespace:
    """The asks of the items of a kind, as ``report.paired`` reads questions: ``id``, ``kind`` (the ask's label), ``text``."""
    return SimpleNamespace(questions=[
        SimpleNamespace(id=a.id, kind=ask_label(i, n), text=a.text) for i in tasks.items if i.kind == kind for n, a in enumerate(i.asks)])


def paired_rows(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    """The valid rows of a kind in the fields ``report.paired`` reads."""
    return [{"arm": r["arm"], "repeat": r["repeat"], "qid": r["qid"], "correct": r["correct"], "wall_ms": r["wall_ms"], "recall_ms": r["recall_ms"]}
            for r in rows if r["valid"] and r["kind"] == kind]


def pairs(rows: Sequence[Mapping[str, Any]], arms: Sequence[str], tasks: data_v2.TaskSet, bases: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """kind -> "A-B" -> the v1 paired comparison over the valid items of both arms (labels are the kind's asks). The arms
    of a kind are those of the run that its arm table lists, variants of a listed arm included."""
    out: dict[str, dict[str, Any]] = {}
    for kind, spec in data_v2.KINDS.items():
        names = [a for a in arms if bases.get(a, a) in spec.arms]
        adapter = kind_questions(tasks, kind)
        kinds = list(dict.fromkeys(q.kind for q in adapter.questions))
        out[kind] = {}
        for x, a in enumerate(names):
            for b in names[x + 1 :]:
                result = report.paired(paired_rows(rows, kind), a, b, adapter, kinds)
                if result:
                    out[kind][f"{a}-{b}"] = {k: result[k] for k in ("a", "b", "overall", "by_kind", "ci_diff_pp")}
    return out


def contexts(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, int]]:
    """arm -> how many asked questions had each `turn.memory.recall.context` (off | used | none); "not reported" for a build
    or an arm that sends no recall."""
    out: dict[str, dict[str, int]] = {}
    for arm in arms:
        found = Counter(str(r["context"]) if r["context"] is not None else "not reported" for r in rows if r["arm"] == arm and r["valid"])
        out[arm] = dict(found)
    return out


# --- two runs ---


def compare(this: Mapping[str, Any], other: Mapping[str, Any], tasks: data_v2.TaskSet, arm: str | None = None) -> dict[str, Any]:
    """The arms of two analysed runs of one task file, paired question by question (the same repeat of the same ask):
    exact McNemar per kind over the items valid in both, and the asks whose verdict differs. ``arm``: just that arm
    (it must be in both), else every arm both runs have."""
    names = [arm] if arm else [a for a in this["arms"] if a in other["arms"]]
    missing = [a for a in names if a not in this["arms"] or a not in other["arms"]]
    if missing:
        raise ValueError(f"arm {', '.join(missing)} is not in both runs (this run: {', '.join(this['arms'])}; the other: {', '.join(other['arms'])})")
    notes = []
    if this["config"].get("tasks_sha256") != other["config"].get("tasks_sha256"):
        notes.append("The two runs used different task files; asks are paired by id, so only ones with the same id are compared.")
    out: dict[str, Any] = {"notes": notes, "other": other["run_dir"], "localmelo": [this["config"].get("localmelo_sha"), other["config"].get("localmelo_sha")], "arms": {}}
    for name in names:
        mine = [{**r, "arm": "this"} for r in this["rows"] if r["arm"] == name]
        theirs = [{**r, "arm": "other"} for r in other["rows"] if r["arm"] == name]
        by = {(r["arm"], r["repeat"], r["qid"]): r for r in [*mine, *theirs] if r["valid"]}
        kinds: dict[str, Any] = {}
        for kind in data_v2.KINDS:
            adapter = kind_questions(tasks, kind)
            result = report.paired(paired_rows([*mine, *theirs], kind), "this", "other", adapter, list(dict.fromkeys(q.kind for q in adapter.questions)))
            if result:
                kinds[kind] = {k: result[k] for k in ("overall", "by_kind", "ci_diff_pp")}
        keys = sorted({(rep, q) for (a, rep, q) in by if a == "this"} & {(rep, q) for (a, rep, q) in by if a == "other"})
        differences = [
            {"qid": q, "repeat": rep, "label": by[("this", rep, q)]["label"], "this": by[("this", rep, q)]["correct"], "other": by[("other", rep, q)]["correct"],
             "this_answer": by[("this", rep, q)]["answer"], "other_answer": by[("other", rep, q)]["answer"],
             "this_miss": by[("this", rep, q)]["miss"], "other_miss": by[("other", rep, q)]["miss"]}
            for rep, q in keys if by[("this", rep, q)]["correct"] != by[("other", rep, q)]["correct"]]
        both = set(keys)
        out["arms"][name] = {
            "paired": len(keys), "kinds": kinds, "differences": differences,
            "only_in_this": sum(1 for (a, rep, q) in by if a == "this" and (rep, q) not in both),
            "only_in_other": sum(1 for (a, rep, q) in by if a == "other" and (rep, q) not in both)}
    return out


def render_compare(cmp: Mapping[str, Any]) -> list[str]:
    out = [f"## Compared with {cmp['other']}", "",
           f"The same ask of the same arm in this run (localmelo `{(cmp['localmelo'][0] or '')[:10]}`) and in `{cmp['other']}` "
           f"(localmelo `{(cmp['localmelo'][1] or '')[:10]}`), over the items valid in both. McNemar: exact two-sided p on the discordant pairs; "
           "this - other is in percentage points.", ""]
    out += [f"> {n}" for n in cmp["notes"]] + ([""] if cmp["notes"] else [])
    for name, result in cmp["arms"].items():
        out += [f"### {name}: {result['paired']} asks paired", ""]
        head = ["ask", "pairs", "both right", "this only", "other only", "both wrong", "this - other (pp)", "McNemar p"]
        body = [[label, str(s["n"]), str(s["both"]), str(s["only_a"]), str(s["only_b"]), str(s["neither"]), f"{s['diff_pp']:+.1f}", f"{s['p']:.3f}"]
                for k in result["kinds"].values() for label, s in k["by_kind"].items()]
        out += ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|", *("| " + " | ".join(r) + " |" for r in body), ""]
        if result["only_in_this"] or result["only_in_other"]:
            out += [f"Valid in only one run: {result['only_in_this']} in this run, {result['only_in_other']} in the other (left out).", ""]
        for d in result["differences"]:
            winner = "this run" if d["this"] else "the other run"
            out.append(f"- {d['qid']} ({d['label']}): right in {winner} only. this: {d['this_answer'][:80]!r} ({d['this_miss'] or 'right'}); "
                       f"other: {d['other_answer'][:80]!r} ({d['other_miss'] or 'right'})")
        out += [""] if result["differences"] else ["No ask differs.", ""]
    return out


# --- the report ---


def analyse(run_dir: Path, tasks: data_v2.TaskSet) -> dict[str, Any]:
    loaded = report.load_run(run_dir)
    config = loaded["state"]["config"]
    arms = [a for a in config["arms"] if any(c["info"]["arm"] == a for c in loaded["cells"])]
    proxy_rows, proxy_counts = report.proxy_turns(loaded)
    req = {(r["arm"], r["repeat"], r["phase"], r["item"]): r for r in proxy_rows}
    chats: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    problems: list[str] = []
    bases = {a: (spec or {}).get("base") or a for a, spec in config["arms"].items()}
    for cell in loaded["cells"]:
        c, a, p = cell_rows(cell, tasks, req, bases)
        chats, asks, problems = chats + c, asks + a, problems + p
    notes = list(problems)
    if config.get("tasks_sha256") != tasks.sha256():
        notes.append("The task file differs from the one the study ran with; replies are graded against the current file.")
    unfinished = [c["id"] for c in loaded["cells"] if c["info"].get("state") != "finished"]
    if unfinished:
        notes.append(f"Not finished: {', '.join(unfinished)}. Their long-range items are invalid (their stores are not read); other finished turns are included.")
    if not proxy_counts:
        notes.append("No cell has a proxy log, so there are no request sizes and the full-context arm is not checked against the prompt cap.")
    if any(r["block_estimated"] for r in asks if r["block_chars"] is not None):
        notes.append("This localmelo build does not report `turn.memory.recall.block_chars`: the recall-block characters below are an estimate (the used items' text plus their source lines).")
    invalid = {a: [{"item": c["item"], "kind": c["kind"], "repeat": c["repeat"], "reason": c["reason"]} for c in chats if c["arm"] == a and not c["valid"]] for a in arms}
    valid_asks = [r for r in asks if r["valid"]]
    sources = {a: counts([r for r in valid_asks if r["arm"] == a and r["source"]], "source", score.SOURCES) for a in arms}
    misses = {
        kind: {a: counts([r for r in valid_asks if r["arm"] == a and r["kind"] == kind and r["miss"]], "miss", score.MISS_CLASSES) for a in arms}
        for kind in data_v2.KINDS
    }
    return {
        "run_dir": str(run_dir.name), "config": config, "arms": arms, "notes": notes,
        "environment": {k: (loaded["environment"].get(k) or {}) for k in ("start", "end")},
        "kinds": {k: dict(spec.arms) for k, spec in data_v2.KINDS.items() if any(i.kind == k for i in tasks.items)},
        "bases": {a: b for a, b in bases.items() if a in arms and b != a},
        "imports": {c["id"]: c["info"]["import"] for c in loaded["cells"] if c["info"].get("import")},
        "sleeps": {c["id"]: sleep_detail(c, tasks) for c in loaded["cells"] if c["info"].get("sleep")},
        "ran": {a: {"items": sum(1 for c in chats if c["arm"] == a), "valid": sum(1 for c in chats if c["arm"] == a and c["valid"])} for a in arms},
        "invalid": invalid, "accuracy": accuracy(asks, arms, tasks), "misses": misses, "sources": sources,
        "latency": latency(asks, arms), "paired": pairs(asks, arms, tasks, bases), "contexts": contexts(asks, arms), "chats": chats, "rows": asks,
    }


def sleep_detail(cell: Mapping[str, Any], tasks: data_v2.TaskSet) -> dict[str, Any]:
    """A cell's sleep: the summary, each change with the texts of the notes it touched (from the copy of the notes the
    night left), and per sleep item the active notes that hold its gold string and those that hold a reject string."""
    cdir = Path(cell["dir"])
    doc = json.loads((cdir / "sleep.json").read_text()) if (cdir / "sleep.json").exists() else {}
    notes: dict[int, tuple[str, str]] = {}
    db = cdir / "after_sleep_records.db"
    if db.exists():
        with closing(stores.open_ro(db)) as con:
            notes = {i: (t, st) for i, t, st in con.execute("SELECT id, text, status FROM long_term")}
    changes = [{
        "op": c.get("op"), "after": notes.get(c.get("after_id"), ("", ""))[0] or c.get("text", ""),
        "before": [notes.get(b, (f"#{b}", ""))[0] for b in c.get("before_ids") or []], "source_turn": c.get("source_turn"),
    } for c in (doc.get("report") or {}).get("changes") or []]
    active = [t for t, st in notes.values() if st == "active"]
    items = {}
    for item in (i for i in tasks.items if i.kind == "sleep" and i.category != "abstain"):
        ask = item.asks[0]
        items[item.id] = {
            "category": item.category,
            "gold_active": sum(1 for t in active if any(score.contains(t, x) for x in ask.targets)),
            "reject_active": sum(1 for t in active if any(score.contains(t, r) for r in ask.reject)),
        }
    return {"summary": cell["info"]["sleep"], "changes": changes, "active_notes": len(active), "items": items}


def pct(r: Mapping[str, Any]) -> str:
    if not r["n"]:
        return "-"
    return f"{100 * r['p']:.0f}% ({r['k']}/{r['n']}) [{100 * r['lo']:.0f}-{100 * r['hi']:.0f}]"


def render(rep: Mapping[str, Any]) -> str:
    arms: list[str] = rep["arms"]
    cfg = rep["config"]
    if not arms:
        return f"# Memory bench v2: {rep['run_dir']}\n\nNo cell has run yet.\n"
    start = rep["environment"].get("start") or {}
    out = [f"# Memory bench v2: {rep['run_dir']}", ""]
    out += [
        f"localmelo `{(cfg.get('localmelo_sha') or '')[:10]}`, chat model `{cfg.get('model')}`, embedding model `{cfg.get('embedding_model')}`, "
        f"{cfg.get('repeats')} repeat(s), {len(cfg.get('questions') or [])} items, long-chat turns stop after {cfg.get('long_timeout_s')} s.",
        f"Machine: {' '.join((start.get('sw_vers') or '').split()) or 'unknown'} | {' '.join((start.get('xcodebuild_version') or '').split()) or 'no Xcode'}.",
        "",
    ]
    out += [f"> {n}" for n in rep["notes"]] + ([""] if rep["notes"] else [])

    def table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
        return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|", *("| " + " | ".join(r) + " |" for r in body), ""]

    out += ["## Which arm runs what", ""]
    out += table(["kind", "arm: window rule at the question"], [
        [k, ", ".join(f"{a}: {r or 'none'}" for a, r in arms_.items())] for k, arms_ in rep["kinds"].items()])
    if rep.get("bases"):
        out += ["Variants follow their base arm's row: " + ", ".join(f"{a} follows {b}" for a, b in rep["bases"].items()) + ".", ""]
    specs = cfg.get("arms") or {}
    out += ["Arm settings the driver passed:", ""]
    out += table(["arm", "melo serve -X", "config.toml", "embedding model"], [
        [a, " ".join((specs.get(a) or {}).get("x") or []) or "-", ", ".join(f"{k}={v}" for k, v in ((specs.get(a) or {}).get("config") or {}).items()) or "-",
         "yes" if (specs.get(a) or {}).get("embedding") else "no"] for a in arms])
    out += ["Window rule `out`: the fact turn has left the working window; `in`: it is still in it. An item that fails its arm's rule, "
            "or whose fact or teaching turn did not finish, is invalid for that arm: it is listed below and left out of everything after it.", ""]
    if rep.get("imports"):
        out += ["## Memory import", "", "What localmelo's memory import found in the cell's seeded HOME and brought in before the first question "
                "(the whole scan is in the cell's `import.json`).", ""]
        out += table(["cell", "sources", "lines kept", "notes written", "skipped", "scan s", "import s", "partial scan", "error"], [
            [cid, str(i.get("sources", 0)), str(i.get("kept", 0)), str(i.get("written", 0)), str(i.get("skipped", 0)),
             str(i.get("scan_s", "-")), str(i.get("apply_s", "-")), "yes" if i.get("partial") else "no", i.get("error") or "-"]
            for cid, i in rep["imports"].items()])
    if rep.get("sleeps"):
        out += ["## Memory sleep", "", "The night of each cell: what the sleep changed (O1 duplicates merged, O2 a newer value written over "
                "older ones, O3 a lost fact written), with the texts of the notes it touched, for judging each change against the teaching "
                "turns. After it: per sleep item, the active notes that hold the gold string and those that hold a reject (old) string.", ""]
        out += table(["cell", "variant", "changes", "by op", "model calls", "refused", "refusals", "ms", "capped", "active notes after", "error"], [
            [cid, d["summary"].get("variant", "") + "".join(f" {k}={v}" for k, v in (d["summary"].get("options") or {}).items()),
             str(d["summary"].get("changes", 0)), ", ".join(f"{k} {v}" for k, v in d["summary"].get("by_op", {}).items()) or "-",
             str(d["summary"].get("model_calls")), str(d["summary"].get("refused", "-")), str(d["summary"].get("refusals", "-")),
             str(d["summary"].get("ms", "-")), str(d["summary"].get("capped")), str(d["active_notes"]), d["summary"].get("error") or "-"]
            for cid, d in rep["sleeps"].items()])
        for cid, d in rep["sleeps"].items():
            if d["changes"]:
                out += [f"**{cid}** changes", ""]
                out += table(["op", "before", "after", "source turn"], [
                    [c["op"] or "", " / ".join(c["before"]) or "-", c["after"], str(c["source_turn"])] for c in d["changes"]])
            if d["items"]:
                out += [f"**{cid}** after the night", ""]
                out += table(["item", "category", "active notes with the gold", "active notes with an old value"], [
                    [iid, i["category"], str(i["gold_active"]), str(i["reject_active"])] for iid, i in d["items"].items()])
    out += ["## Items run and invalid items", ""]
    out += table(["", *arms], [
        ["items run", *(str(rep["ran"][a]["items"]) for a in arms)], ["valid", *(str(rep["ran"][a]["valid"]) for a in arms)],
        ["invalid", *(str(rep["ran"][a]["items"] - rep["ran"][a]["valid"]) for a in arms)]])
    for a in arms:
        for bad in rep["invalid"][a]:
            out.append(f"- {a} {bad['item']} (repeat {bad['repeat']}): {bad['reason']}")
    out += [""]

    out += ["## Accuracy", "",
            "Correct / valid items with the 95% Wilson interval in brackets (percent). A reply is correct when it holds the gold answer or an alias and none of "
            "the reject strings (a stale or look-alike value).", ""]
    for kind in data_v2.KINDS:
        acc = rep["accuracy"][kind]
        names = list(dict.fromkeys(n for a in arms for n in acc.get(a, {})))
        if names:
            out += [f"**{kind}**", ""]
            out += table(["", *arms], [[n, *(pct(acc[a][n]) if a in acc and n in acc[a] else "-" for a in arms)] for n in names])

    out += ["## Misses", "", "Every wrong answer of a valid item, by class. `not saved`: no note holding the gold string from the chat that said the fact; "
            "`not recalled`: saved, but that chat's item is not in `turn.memory.used`; `recalled but not used`: it is, and the answer is wrong. "
            "Memory-off arms: `out of window` or `in window but not used`.", ""]
    for kind in data_v2.KINDS:
        classes = list(dict.fromkeys(c for a in arms for c in rep["misses"][kind][a]))
        if classes:
            out += [f"**{kind}**", ""]
            out += table(["", *arms], [[c, *(str(rep["misses"][kind][a].get(c, 0)) for a in arms)] for c in classes])
    srcs = list(dict.fromkeys(c for a in arms for c in rep["sources"][a]))
    if srcs:
        out += ["Where the right long-range answers came from, as far as the events say. A `note` or a `chat turn` is a used item that holds the gold; "
                "`own-chat turn` is the fact turn still in the window. localmelo's legacy step memory (M2 only) sends no event, so an unexplained right answer "
                "there may be one of its step summaries.", ""]
        out += table(["source", *arms], [[c, *(str(rep["sources"][a].get(c, 0)) for a in arms)] for c in srcs])

    long = [c for c in rep["chats"] if c["kind"] == "longrange" and c["valid"] and c.get("fact_note_saved") is not None]
    if long:
        out += ["## Notes made in the long chats", "",
                "`note_expected` says whether the note rules were expected to save the fact. Fact turn made a note: `turn.memory.remembered` of the fact turn. "
                "A note holds the gold: a `records.db` row from the chat with the gold string.", ""]
        body = []
        for flag, name in ((False, "note not expected"), (True, "note expected")):
            body.append([f"{name}: fact turn made a note", *(frac([c for c in long if c["arm"] == a and c["note_expected"] is flag], "fact_made_note") for a in arms)])
            body.append([f"{name}: a note holds the gold", *(frac([c for c in long if c["arm"] == a and c["note_expected"] is flag], "fact_note_saved") for a in arms)])
        out += table(["", *arms], body)

    tools: dict[str, Counter[str]] = {a: Counter() for a in arms}
    for c in rep["chats"]:
        tools[c["arm"]].update(c.get("tools") or {})
    if any(tools.values()):
        names = sorted({n for t in tools.values() for n in t})
        out += ["## Tools the long chats used", "", "Tool calls in the opener, bridge, fact and filler turns. The filler is meant to use "
                f"{', '.join(data_v2.READ_TOOLS)} only; anything in {', '.join(NON_READ_TOOLS)} needs approval, which the driver denies.", ""]
        out += table(["tool", *arms], [[n, *(str(tools[a].get(n, 0)) for a in arms)] for n in names]
                     + [["approval requests denied (long chats)", *(str(sum(c.get("denied", 0) for c in rep["chats"] if c["arm"] == a and c["kind"] == "longrange")) for a in arms)]])

    shown = list(dict.fromkeys(c for a in arms for c in rep["contexts"][a]))
    if any(c != "not reported" for c in shown):
        out += ["## Recall context", "", "Valid questions by `turn.memory.recall.context` (off: the recall query was the message alone; used: the earlier turns of the chat "
                "were part of it; none: there was nothing to add).", ""]
        out += table(["context", *arms], [[c, *(str(rep["contexts"][a].get(c, 0)) for a in arms)] for c in shown])

    out += ["## Paired comparison of the arms", "",
            "The same item (and ask) valid for both arms. McNemar: exact two-sided p on the discordant pairs. A - B is in percentage points.", ""]
    for kind in data_v2.KINDS:
        body = []
        for name, pr in rep["paired"][kind].items():
            for label, s in pr["by_kind"].items():
                body.append([name, label, str(s["n"]), str(s["both"]), str(s["only_a"]), str(s["only_b"]), str(s["neither"]), f"{s['diff_pp']:+.1f}", f"{s['p']:.3f}"])
        if body:
            out += [f"**{kind}**", ""]
            out += table(["arms (A-B)", "ask", "pairs", "both right", "A only", "B only", "both wrong", "A - B (pp)", "McNemar p"], body)

    out += ["## Time, request size and recall block", "",
            "Median / p90 over the valid questions whose turn finished. Request bytes: the agent's first chat request of the question's turn, from the "
            "timing proxy. Recall block characters are the reported `block_chars` (real) or the estimate from the used items.", ""]
    for kind in data_v2.KINDS:
        lt = rep["latency"][kind]
        if not lt:
            continue

        def lrow(name: str, key: str, kind_: str = kind) -> list[str]:
            return [name, *(f"{report.ms(rep['latency'][kind_][a][key]['median'])} / {report.ms(rep['latency'][kind_][a][key]['p90'])} (n={rep['latency'][kind_][a][key]['n']})"
                            if a in rep["latency"][kind_] else "-" for a in arms)]

        out += [f"**{kind}**", ""]
        out += table(["", *arms], [
            lrow("turn wall (ms)", "wall_ms"), lrow("recall (ms)", "recall_ms"), lrow("first event (ms)", "ttfe_ms"), lrow("request (bytes)", "req_bytes"),
            lrow("recall block (chars, real)", "block_real"), lrow("recall block (chars, ESTIMATE)", "block_estimated")])
    if rep.get("compare"):
        out += render_compare(rep["compare"])
    return "\n".join(out).rstrip() + "\n"


def frac(chats: Sequence[Mapping[str, Any]], key: str) -> str:
    known = [c for c in chats if c.get(key) is not None]
    return report.frac(sum(1 for c in known if c[key]), len(known))


def main(run_dir: Path, tasks_path: Path, compare_with: Sequence[str] | None = None) -> int:
    """Write report.md and report.json of a v2 run. ``compare_with``: the other run's directory, and optionally one arm."""
    tasks = data_v2.load(tasks_path)
    rep = analyse(run_dir, tasks)
    if compare_with:
        if len(compare_with) > 2:
            raise SystemExit("--compare takes a run directory and optionally an arm")
        try:
            rep["compare"] = compare(rep, analyse(Path(compare_with[0]), tasks), tasks, compare_with[1] if len(compare_with) > 1 else None)
        except (ValueError, OSError) as exc:
            raise SystemExit(f"--compare: {exc}") from None
    (run_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render(rep), encoding="utf-8")
    print(f"wrote {run_dir / 'report.md'} and report.json")
    return 0
