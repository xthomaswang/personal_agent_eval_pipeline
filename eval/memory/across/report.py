"""Report of a memory-across-chats study: accuracy, recall, latency and the cost of the second model.

    cd eval
    uv run python -m memory.across.report <run-dir>      # writes <run-dir>/report.md and report.json

Everything is recomputed from the journals (cells/*/turns.jsonl) and the task file, so a change to the scorer
or the data needs no new run. Sections:

- accuracy by question kind and arm (a reply is correct per memory.across.score);
- recall: hit@k from `turn.memory.used` against the chat that taught the gold fact, wrong recall (a stale or
  look-alike fact used), and false recall on questions with nothing to recall;
- latency: median and p90 of the recall time (embed + search + select, from `turn.memory.recall`, which a
  build with the retriever switch reports; "-" otherwise) and of the turn (wall clock from the POST to the
  end of the turn, and the gateway's own pre-run time), and the tokens the chat model reads to select;
- chat time to first byte per arm from the timing proxy's log (`cells/*/proxy.jsonl`), and for arms with an
  embedding model how it relates to the embeddings requests in the same turn (overlap, per-turn embed ms);
- the second model: oMLX rss and loaded models before the cell, after the warm-up, after teaching and after the
  cell (was the embedding model loaded, and what the cell added), and the one-token warm-up times;
- teaching, and what the driver did to keep turns moving: notes made, approval requests denied, turns stopped
  by the timeout;
- a paired comparison of two arms (M2 against L1 by default) by question: wins and losses, an exact McNemar
  test, a bootstrap interval of the accuracy difference, and the paired latency.

A run on a v2 task file (the in-chat kinds) is reported by report_v2.py instead.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import data, proxy, score

PAIR = ("M2", "L1")
BOOTSTRAP = 2000


# --- statistics ---


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0..100); None for no values."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: Sequence[float], n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile interval of the mean of ``diffs``, resampling the items."""
    if not diffs:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]


def num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# --- loading ---


def load_run(run_dir: Path) -> dict[str, Any]:
    state = json.loads((run_dir / "across.json").read_text(encoding="utf-8"))
    cells = []
    for cell_dir in sorted((run_dir / "cells").glob("*")):
        info_path = cell_dir / "cell.json"
        if ".aborted-" in cell_dir.name or not info_path.exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        rows = []
        for line in (cell_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines() if (cell_dir / "turns.jsonl").exists() else []:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        cells.append({"id": cell_dir.name, "dir": cell_dir, "info": info, "turns": [r for r in rows if r.get("event") == "turn"]})
    env_path = run_dir / "environment.json"
    return {"state": state, "cells": cells, "environment": json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}}


def recall_ms(recall: Mapping[str, Any] | None) -> float | None:
    """embed + search + select milliseconds of one recall; None when the build did not report it."""
    if not isinstance(recall, Mapping):
        return None
    parts = [num(recall.get(k)) for k in ("embed_ms", "search_ms", "select_ms")]
    return sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None


def select_parts(recall: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    """(prompt, completion) tokens of the chat model's selection; a bare number counts as prompt tokens, and None
    stands for a recall that made no selection."""
    value = (recall or {}).get("select_tokens")
    prompt, completion = (num(value.get("prompt")), num(value.get("completion"))) if isinstance(value, Mapping) else (num(value), None)
    if not prompt and not completion:
        return None, None  # no selection was made (not the llm retriever, or no candidates)
    return prompt, completion


def has_fallback(recall: Mapping[str, Any] | None) -> bool | None:
    """Whether a recall fell back to words only (`fallback` is "none" when it did not); None without a recall."""
    if not isinstance(recall, Mapping):
        return None
    return recall.get("fallback") not in (None, "", "none", False)


def ask_rows(run: Mapping[str, Any], tasks: data.TaskSet) -> list[dict[str, Any]]:
    """One graded row per asked question of every cell."""
    questions = {q.id: q for q in tasks.questions}
    rows: list[dict[str, Any]] = []
    for cell in run["cells"]:
        info = cell["info"]
        sessions = {t["item"]: t.get("session_id") for t in cell["turns"] if t["phase"] == "teach" and t.get("session_id")}
        taught = [s for s in sessions.values() if s]
        for t in cell["turns"]:
            q = questions.get(t["item"]) if t["phase"] == "ask" else None
            if q is None:
                continue
            grade = score.grade(q, t.get("answer"))
            gold = [sessions.get(f) for f in q.facts]
            wrong = [sessions.get(f) for f in q.wrong_facts]
            recall = score.recall_score(
                t.get("memory"), [g for g in gold if g], [w for w in wrong if w], taught, nothing=q.kind == "nothing")
            if q.kind != "nothing" and not all(gold):  # the gold fact never reached a chat: it cannot be recalled
                recall.update(hit_at_1=False, hit_at_3=False, hit_at_5=False, hit_any=False)
            server_ms = t.get("server_ms") or {}
            rec = t.get("recall") if isinstance(t.get("recall"), Mapping) else None
            prompt_tokens, completion_tokens = select_parts(rec)
            rows.append({
                "arm": info["arm"], "repeat": info["cell"]["repeat"], "qid": q.id, "kind": q.kind, "lang": q.lang,
                "state": t.get("state"), "outcome": grade.outcome, "correct": grade.correct, "abstained": grade.abstained,
                "recall": recall, "recall_ms": recall_ms(rec), "select_prompt": prompt_tokens, "select_completion": completion_tokens,
                "fallback": has_fallback(rec), "retriever": (rec or {}).get("retriever"),
                "denied": int(t.get("approvals_denied") or 0), "timed_out": bool(t.get("timed_out")),
                "wall_ms": num(t.get("wall_ms")), "ttfe_ms": num(t.get("ttfe_ms")),
                "pre_run_ms": num(server_ms.get("run_started")), "answer": t.get("answer", ""),
            })
    return rows


# --- tables ---


def frac(k: int, n: int) -> str:
    return f"{100.0 * k / n:.0f}% ({k}/{n})" if n else "-"


def ms(value: float | None) -> str:
    return "-" if value is None else (f"{value:.0f}" if value >= 100 else f"{value:.1f}")


def stats(values: Sequence[float | None]) -> dict[str, Any]:
    xs = [v for v in values if v is not None]
    return {"n": len(xs), "median": percentile(xs, 50), "p90": percentile(xs, 90), "mean": sum(xs) / len(xs) if xs else None}


def accuracy(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, tuple[int, int]]]:
    """kind (and "all") -> arm -> (correct, asked)."""
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for kind in (*data.KINDS, "all"):
        table[kind] = {}
        for arm in arms:
            chosen = [r for r in rows if r["arm"] == arm and (kind == "all" or r["kind"] == kind)]
            table[kind][arm] = (sum(1 for r in chosen if r["correct"]), len(chosen))
    return table


def recall_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        answerable = [r for r in mine if r["kind"] != "nothing"]
        nothing = [r for r in mine if r["kind"] == "nothing"]
        confusable = [r for r in mine if r["kind"] in ("superseded", "lookalike")]
        out[arm] = {
            **{f"hit_at_{k}": (sum(1 for r in answerable if r["recall"].get(f"hit_at_{k}")), len(answerable)) for k in score.HIT_KS},
            "hit_any": (sum(1 for r in answerable if r["recall"].get("hit_any")), len(answerable)),
            "wrong_recall": (sum(1 for r in confusable if r["recall"].get("wrong_recall")), len(confusable)),
            "false_recall": (sum(1 for r in nothing if r["recall"].get("false_recall")), len(nothing)),
            "false_recall_taught": (sum(1 for r in nothing if r["recall"].get("false_recall_taught")), len(nothing)),
            "hit_at_3_by_kind": {
                k: (sum(1 for r in answerable if r["kind"] == k and r["recall"].get("hit_at_3")), sum(1 for r in answerable if r["kind"] == k))
                for k in data.KINDS if k != "nothing"
            },
        }
    return out


def latency_table(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Latency over the turns that finished; a turn the timeout stopped says nothing about speed."""
    out = {}
    for arm in arms:
        every = [r for r in rows if r["arm"] == arm]
        mine = [r for r in every if r["state"] == "finished"]
        prompt = [r["select_prompt"] for r in mine if r["select_prompt"] is not None]
        completion = [r["select_completion"] for r in mine if r["select_completion"] is not None]
        out[arm] = {
            "recall_ms": stats([r["recall_ms"] for r in mine]),
            "wall_ms": stats([r["wall_ms"] for r in mine]),
            "ttfe_ms": stats([r["ttfe_ms"] for r in mine]),
            "pre_run_ms": stats([r["pre_run_ms"] for r in mine]),
            "select_prompt": {**stats(prompt), "total": sum(prompt)},
            "select_completion": {**stats(completion), "total": sum(completion)},
            "fallbacks": (sum(1 for r in mine if r["fallback"]), sum(1 for r in mine if r["fallback"] is not None)),
            "failed": len(every) - len(mine),
        }
    return out


def rss_mb(infos: Sequence[Mapping[str, Any]], stage: str) -> list[float]:
    """The oMLX rss in MB recorded at ``stage`` of each cell."""
    return [v / 1024.0 for i in infos if (v := num(((i.get("rss") or {}).get(stage) or {}).get("rss_kb"))) is not None]


STAGES = ("before", "after_warmup", "after_teach", "after")


def loaded_at(infos: Sequence[Mapping[str, Any]], stage: str, model: str) -> tuple[int, int]:
    """(cells whose model server had ``model`` loaded at ``stage``, cells where that is known)."""
    states = [((i.get("rss") or {}).get(stage) or {}).get("loaded") for i in infos]
    known = [st for st in states if isinstance(st, list)]
    return sum(1 for st in known if model in st), len(known)


def deltas(infos: Sequence[Mapping[str, Any]], first: str, last: str, field: str) -> list[float]:
    """``last`` minus ``first`` of a recorded number (rss in kB, pool bytes) for each cell that has both."""
    out = []
    for i in infos:
        rss = i.get("rss") or {}
        a, b = num(_dig(rss.get(first), field)), num(_dig(rss.get(last), field))
        if a is not None and b is not None:
            out.append(b - a)
    return out


def _dig(row: Any, field: str) -> Any:
    for part in field.split("."):
        row = row.get(part) if isinstance(row, Mapping) else None
    return row


def started_cold(info: Mapping[str, Any], embedding_model: str) -> bool:
    """Whether the cell began with the embedding model unloaded: the driver confirmed it, and the server's own list
    taken just after agrees."""
    cold = info.get("cold_start") or {}
    before = ((info.get("rss") or {}).get("before") or {}).get("loaded")
    return bool(cold) and not cold.get("skipped") and cold.get("loaded_after") is False and not (isinstance(before, list) and embedding_model in before)


def cold_start_summary(run: Mapping[str, Any]) -> dict[str, int]:
    model = str(run["state"]["config"].get("embedding_model"))
    infos = [c["info"] for c in run["cells"]]
    return {
        "cells": len(infos), "cold": sum(1 for i in infos if started_cold(i, model)),
        "unloaded_by_driver": sum(1 for i in infos if (i.get("cold_start") or {}).get("was_loaded")),
        "skipped": sum(1 for i in infos if (i.get("cold_start") or {}).get("skipped")),
    }


def cost_table(run: Mapping[str, Any], arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    config = run["state"]["config"]
    out = {}
    for arm in arms:
        infos = [c["info"] for c in run["cells"] if c["info"]["arm"] == arm]
        warm = [i.get("warmup") or {} for i in infos]
        out[arm] = {
            "cells": len(infos),
            "cold": sum(1 for i in infos if started_cold(i, str(config.get("embedding_model")))),
            "unload_wait_ms": percentile([v for i in infos if (v := num((i.get("cold_start") or {}).get("wait_ms"))) is not None and (i.get("cold_start") or {}).get("was_loaded")], 50),
            "rss_mb": {s: percentile(rss_mb(infos, s), 50) for s in STAGES},
            "rss_delta_mb": {
                "warmup": percentile([d / 1024.0 for d in deltas(infos, "before", "after_warmup", "rss_kb")], 50),
                "cell": percentile([d / 1024.0 for d in deltas(infos, "before", "after", "rss_kb")], 50),
            },
            "pool_delta_mb": percentile([d / 1048576.0 for d in deltas(infos, "before", "after", "status.model_memory_used")], 50),
            "embedding_loaded": {s: loaded_at(infos, s, str(config.get("embedding_model"))) for s in STAGES},
            "chat_loaded": {s: loaded_at(infos, s, str(config.get("model"))) for s in STAGES},
            "chat_warmup_ms": percentile([v for w in warm if (v := num(w.get("chat_ms"))) is not None], 50),
            "embed_warmup_ms": percentile([v for w in warm if (v := num(w.get("embed_ms"))) is not None], 50),
            "cell_wall_s": percentile([v for i in infos if (v := num(i.get("wall_s"))) is not None], 50),
        }
    return out


def interventions(run: Mapping[str, Any], arms: Sequence[str]) -> dict[str, dict[str, dict[str, int]]]:
    """Per arm and phase: turns, notes made, approval requests denied, turns stopped by the timeout, failures."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for arm in arms:
        out[arm] = {}
        for phase in ("teach", "ask"):
            turns = [t for c in run["cells"] if c["info"]["arm"] == arm for t in c["turns"] if t["phase"] == phase]
            out[arm][phase] = {
                "turns": len(turns),
                "failed": sum(1 for t in turns if t.get("state") != "finished"),
                "made_a_note": sum(1 for t in turns if (t.get("memory") or {}).get("remembered")),
                "refused": sum(1 for t in turns if (t.get("memory") or {}).get("refused")),
                "approvals_denied": sum(int(t.get("approvals_denied") or 0) for t in turns),
                "turns_with_denied": sum(1 for t in turns if t.get("approvals_denied")),
                "timed_out": sum(1 for t in turns if t.get("timed_out")),
            }
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman's rank correlation (ties share their mean rank); None for fewer than 3 pairs or no spread."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda k: values[k])
        out = [0.0] * len(values)
        k = 0
        while k < len(order):
            m = k
            while m + 1 < len(order) and values[order[m + 1]] == values[order[k]]:
                m += 1
            for idx in order[k : m + 1]:
                out[idx] = (k + m) / 2.0 + 1.0
            k = m + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / math.sqrt(sxx * syy)


def proxy_turns(run: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """One row per turn with the proxy's view of it, and per arm the request counts.

    The turn's chat call is the first request in its window (POST to turn end) that offers tools, which is the
    agent's call; a recall selection offers none. Its time to first byte is t_first_byte - t_start. An embeddings
    request overlaps it when it was in flight at any point between the chat request's start and first byte.
    """
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for cell in run["cells"]:
        log = proxy.read_log(cell["dir"] / "proxy.jsonl")
        if not log:
            continue
        arm = cell["info"]["arm"]
        total = counts.setdefault(arm, {"chat": 0, "embeddings": 0, "embeddings_in_turns": 0, "chat_streamed": 0})
        chats = sorted((r for r in log if str(r.get("path", "")).endswith("/chat/completions") and r.get("t_start")), key=lambda r: r["t_start"])
        embeds = [r for r in log if str(r.get("path", "")).endswith("/embeddings") and r.get("t_start")]
        total["chat"] += len(chats)
        total["embeddings"] += len(embeds)
        total["chat_streamed"] += sum(1 for r in chats if r.get("stream"))
        for t in cell["turns"]:
            a, b = num(t.get("started_epoch")), num(t.get("ended_epoch"))
            if not a or not b:
                continue
            inside = [r for r in chats if a <= r["t_start"] <= b]
            agent = [r for r in inside if r.get("tools")]
            main = agent[0] if agent else (inside[-1] if inside and all(r.get("tools") is None for r in inside) else None)
            mine = [e for e in embeds if a <= e["t_start"] <= b]
            total["embeddings_in_turns"] += len(mine)
            first = num(main.get("t_first_byte")) if main else None
            rec = t.get("recall") if isinstance(t.get("recall"), Mapping) else None
            rows.append({
                "arm": arm, "repeat": cell["info"]["cell"]["repeat"], "phase": t["phase"], "item": t["item"], "state": t.get("state"),
                # the size of the agent's chat request (the first one with tools), and of the biggest one in the turn
                "req_bytes": num(main.get("req_bytes")) if main else None,
                "req_bytes_max": max((num(r.get("req_bytes")) or 0 for r in agent), default=None),
                "ttft_ms": round((first - main["t_start"]) * 1000.0, 1) if main and first else None,
                "stream": bool(main.get("stream")) if main else None, "n_embed": len(mine), "embed_ms": num((rec or {}).get("embed_ms")),
                "overlap": any(e["t_start"] < first and (num(e.get("t_end")) or e["t_start"]) > main["t_start"] for e in embeds) if main and first else None,
            })
    return rows, counts


def ttft_table(turns: Sequence[Mapping[str, Any]], counts: Mapping[str, Mapping[str, int]], arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        mine = [r for r in turns if r["arm"] == arm and r["state"] == "finished" and r["ttft_ms"] is not None]
        pairs = [(r["embed_ms"], r["ttft_ms"]) for r in mine if r["embed_ms"] is not None]
        total = counts.get(arm, {"chat": 0, "embeddings": 0, "embeddings_in_turns": 0, "chat_streamed": 0})
        out[arm] = {
            "ask": stats([r["ttft_ms"] for r in mine if r["phase"] == "ask"]),
            "teach": stats([r["ttft_ms"] for r in mine if r["phase"] == "teach"]),
            "streamed": (sum(1 for r in mine if r["stream"]), len(mine)),
            "with_embedding": stats([r["ttft_ms"] for r in mine if r["n_embed"] > 0]),
            "without_embedding": stats([r["ttft_ms"] for r in mine if r["n_embed"] == 0]),
            "overlapped": stats([r["ttft_ms"] for r in mine if r["overlap"]]),
            "clear": stats([r["ttft_ms"] for r in mine if r["overlap"] is False]),
            "spearman_embed_ms": spearman([x for x, _ in pairs], [y for _, y in pairs]),
            "pairs": len(pairs),
            "chat_requests": total["chat"], "embeddings": total["embeddings"], "embeddings_in_turns": total["embeddings_in_turns"],
        }
    return out


def paired(
    rows: Sequence[Mapping[str, Any]], a: str, b: str, tasks: data.TaskSet, kinds: Sequence[str] = data.KINDS,
) -> dict[str, Any] | None:
    """Arm ``a`` against arm ``b`` question by question (the same repeat of the same question). ``tasks`` is anything
    with ``questions`` (``id``, ``kind``, ``text``); ``kinds`` are the kinds to break the pairs down by."""
    by = {(r["arm"], r["repeat"], r["qid"]): r for r in rows}
    keys = sorted({(rep, q) for (arm, rep, q) in by if arm == a} & {(rep, q) for (arm, rep, q) in by if arm == b})
    if not keys:
        return None
    kind_of = {q.id: q.kind for q in tasks.questions}
    text = {q.id: q.text for q in tasks.questions}

    def summary(selected: Sequence[tuple[int, str]]) -> dict[str, Any]:
        pairs = [(by[(a, rep, q)]["correct"], by[(b, rep, q)]["correct"]) for rep, q in selected]
        only_a = sum(1 for x, y in pairs if x and not y)
        only_b = sum(1 for x, y in pairs if y and not x)
        return {
            "n": len(pairs), "both": sum(1 for x, y in pairs if x and y), "only_a": only_a, "only_b": only_b,
            "neither": sum(1 for x, y in pairs if not x and not y), "p": mcnemar_p(only_a, only_b),
            "diff_pp": 100.0 * (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / len(pairs),
        }

    by_kind = {k: summary([key for key in keys if kind_of[key[1]] == k]) for k in kinds if any(kind_of[q] == k for _, q in keys)}
    per_question: dict[str, list[float]] = defaultdict(list)
    wall: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rec: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rep, q in keys:
        ra, rb = by[(a, rep, q)], by[(b, rep, q)]
        per_question[q].append(float(ra["correct"]) - float(rb["correct"]))
        if ra["wall_ms"] is not None and rb["wall_ms"] is not None:
            wall[q].append((ra["wall_ms"], rb["wall_ms"]))
        if ra["recall_ms"] is not None and rb["recall_ms"] is not None:
            rec[q].append((ra["recall_ms"], rb["recall_ms"]))
    mean_diffs = [sum(v) / len(v) for v in per_question.values()]

    def lat(table: Mapping[str, Sequence[tuple[float, float]]]) -> dict[str, Any]:
        mean = {q: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v)) for q, v in table.items()}
        return {
            "n": len(mean), "median_a": percentile([x for x, _ in mean.values()], 50), "median_b": percentile([y for _, y in mean.values()], 50),
            "median_diff": percentile([x - y for x, y in mean.values()], 50),
        }

    consistent_a = sorted(q for q, v in per_question.items() if len(v) > 0 and all(d > 0 for d in v))
    consistent_b = sorted(q for q, v in per_question.items() if len(v) > 0 and all(d < 0 for d in v))
    return {
        "a": a, "b": b, "overall": summary(keys), "by_kind": by_kind, "ci_diff_pp": [100 * x for x in (bootstrap_ci(mean_diffs) or ())],
        "wall_ms": lat(wall), "recall_ms": lat(rec),
        "only_a_always": [(q, text[q]) for q in consistent_a], "only_b_always": [(q, text[q]) for q in consistent_b],
    }


# --- the report ---


def analyse(run_dir: Path, tasks: data.TaskSet | None = None, pair: Sequence[str] = PAIR) -> dict[str, Any]:
    run = load_run(run_dir)
    tasks = tasks or data.load()
    config = run["state"]["config"]
    arms = [a for a in config["arms"] if any(c["info"]["arm"] == a for c in run["cells"])]
    rows = ask_rows(run, tasks)
    turns_proxy, proxy_counts = proxy_turns(run)
    notes = []
    if config.get("tasks_sha256") != tasks.sha256():
        notes.append("The task file differs from the one the study ran with; replies are graded against the current file.")
    unfinished = [c["id"] for c in run["cells"] if c["info"].get("state") != "finished"]
    missing = [cid for cid in (c["cell_id"] for c in json.loads((run_dir / "schedule.json").read_text())) if cid not in {c["id"] for c in run["cells"]}]
    if unfinished or missing:
        notes.append(f"Not finished: {', '.join(unfinished + missing)}. Their finished turns are included.")
    cold = cold_start_summary(run)
    if cold["cold"] != cold["cells"]:
        notes.append(f"Only {cold['cold']} of {cold['cells']} cells started with the embedding model unloaded; load times and the embedding model's cost are not comparable across arms.")
    if not proxy_counts:
        notes.append("No cell has a proxy log, so there is no chat time to first byte (the run used --no-proxy).")
    if any(r["recall"].get("n_used") for r in rows) and not any(r["recall_ms"] is not None for r in rows):
        notes.append("No turn reported recall timings (`turn.memory.recall`): this localmelo build predates them, so the recall columns are empty.")
    return {
        "run_dir": str(run_dir.name), "config": config, "arms": arms, "notes": notes,
        "environment": {k: (run["environment"].get(k) or {}) for k in ("start", "end")},
        "accuracy": accuracy(rows, arms), "recall": recall_table(rows, arms), "latency": latency_table(rows, arms),
        "ttft": ttft_table(turns_proxy, proxy_counts, arms), "cost": cost_table(run, arms), "cold_start": cold_start_summary(run), "interventions": interventions(run, arms),
        "paired": paired(rows, pair[0], pair[1], tasks) if pair[0] in arms and pair[1] in arms else None,
        "rows": rows,
    }


def render(rep: Mapping[str, Any]) -> str:
    arms: list[str] = rep["arms"]
    cfg = rep["config"]
    if not arms:
        return f"# Memory across chats: {rep['run_dir']}\n\nNo cell has run yet.\n"
    start = rep["environment"].get("start") or {}
    out = [f"# Memory across chats: {rep['run_dir']}", ""]
    out += [
        f"localmelo `{(cfg.get('localmelo_sha') or '')[:10]}`, chat model `{cfg.get('model')}`, embedding model `{cfg.get('embedding_model')}`, "
        f"{cfg.get('repeats')} repeats, {len(cfg.get('questions') or [])} questions per cell, "
        f"ask chats {'kept' if cfg.get('keep_ask_chats') else 'deleted after each answer'}.",
        f"Machine: {' '.join((start.get('sw_vers') or '').split()) or 'unknown'} | {' '.join((start.get('xcodebuild_version') or '').split()) or 'no Xcode'}.",
        "",
    ]
    out += [f"> {n}" for n in rep["notes"]] + ([""] if rep["notes"] else [])

    def table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
        return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|", *("| " + " | ".join(r) + " |" for r in body), ""]

    iv = rep["interventions"]
    out += ["## Teaching turns that asked for tool approval", "",
            "When the chat model also tries to save a stated fact to a file, the tool call asks for approval and the turn would wait; the driver denies "
            f"every request and stops a teaching turn after {cfg.get('teach_timeout_s') or 60:g} s. This build tells the model not to save such facts, so "
            "a nonzero count here is the check that it still does.", ""]
    out += table(["", *arms], [
        ["teaching turns that asked for approval", *(frac(iv[a]["teach"]["turns_with_denied"], iv[a]["teach"]["turns"]) for a in arms)],
        ["approval requests denied (teaching)", *(str(iv[a]["teach"]["approvals_denied"]) for a in arms)],
        ["teaching turns stopped by the timeout", *(frac(iv[a]["teach"]["timed_out"], iv[a]["teach"]["turns"]) for a in arms)],
        ["question turns that asked for approval", *(frac(iv[a]["ask"]["turns_with_denied"], iv[a]["ask"]["turns"]) for a in arms)],
        ["question turns stopped by the timeout", *(frac(iv[a]["ask"]["timed_out"], iv[a]["ask"]["turns"]) for a in arms)],
    ])

    out += ["## Accuracy", "", "A reply is correct when it holds the gold answer (or an alias) and none of the question's reject strings; for nothing to recall, when it says it does not know.", ""]
    out += table(["kind", *arms], [[k, *(frac(*rep["accuracy"][k][a]) for a in arms)] for k in (*data.KINDS, "all")])

    out += ["## Recall", "", "hit@k: the chat that taught the gold fact is among the first k items of `turn.memory.used`. Wrong recall: the stale or look-alike fact's chat was used (superseded and lookalike questions). False recall: any item used on a question with nothing to recall (taught: an item from a teaching chat).", ""]
    rc = rep["recall"]
    out += table(["measure", *arms], [
        ["hit@1", *(frac(*rc[a]["hit_at_1"]) for a in arms)], ["hit@3", *(frac(*rc[a]["hit_at_3"]) for a in arms)],
        ["hit@5", *(frac(*rc[a]["hit_at_5"]) for a in arms)], ["wrong recall", *(frac(*rc[a]["wrong_recall"]) for a in arms)],
        ["false recall", *(frac(*rc[a]["false_recall"]) for a in arms)], ["false recall (taught)", *(frac(*rc[a]["false_recall_taught"]) for a in arms)],
    ])
    out += ["hit@3 by kind", ""]
    out += table(["kind", *arms], [[k, *(frac(*rc[a]["hit_at_3_by_kind"][k]) for a in arms)] for k in rc[arms[0]]["hit_at_3_by_kind"]] if arms else [])

    out += ["## Latency", "", "Milliseconds over every asked turn of the arm that finished. Recall = embed + search + select from `turn.memory.recall`; turn = the POST to the end of the turn; pre-run = the gateway's clock from turn.started to run.started (recall and agent start); first event = the POST to the first event frame. The reply is one message, so the first reply message arrives with the end of the turn.", ""]
    lt = rep["latency"]
    def lrow(name: str, key: str) -> list[str]:
        return [name, *(f"{ms(lt[a][key]['median'])} / {ms(lt[a][key]['p90'])} (n={lt[a][key]['n']})" for a in arms)]
    out += table(["median / p90 (ms)", *arms], [lrow("recall", "recall_ms"), lrow("turn wall", "wall_ms"), lrow("pre-run", "pre_run_ms"), lrow("first event", "ttfe_ms")])
    out += table(["select tokens (chat model's choice)", *arms], [
        ["prompt median / p90", *(f"{ms(lt[a]['select_prompt']['median'])} / {ms(lt[a]['select_prompt']['p90'])}" for a in arms)],
        ["completion median / p90", *(f"{ms(lt[a]['select_completion']['median'])} / {ms(lt[a]['select_completion']['p90'])}" for a in arms)],
        ["total prompt + completion", *(str(int(lt[a]["select_prompt"]["total"] + lt[a]["select_completion"]["total"])) for a in arms)],
        ["fell back to words", *(frac(*lt[a]["fallbacks"]) for a in arms)], ["failed turns (left out above)", *(str(lt[a]["failed"]) for a in arms)],
    ])

    tf = rep["ttft"]
    out += ["## Chat time to first byte", "",
            "From the timing proxy: the agent's chat request (the first one with tools in the turn) from its start to the first byte of the response, in ms. "
            "Only a request that streams has a first token; the gateway sends the others whole, and for those the first byte is about the end of the reply. "
            "The row \"streamed\" says which this is. An embeddings request overlaps when it was in flight between the chat request's start and its first byte; "
            "the query embedding comes before the chat request, so an overlap is a neighbour, usually the previous turn being indexed.", ""]
    if not any(tf[a]["chat_requests"] for a in arms):
        out += ["No proxy log.", ""]
    else:
        def trow(name: str, key: str) -> list[str]:
            return [name, *(f"{ms(tf[a][key]['median'])} / {ms(tf[a][key]['p90'])} (n={tf[a][key]['n']})" for a in arms)]
        out += table(["median / p90 (ms)", *arms], [
            trow("question turns", "ask"), trow("teaching turns", "teach"), trow("turns with an embeddings request in them", "with_embedding"),
            trow("turns without one", "without_embedding"), trow("overlapped by an embeddings request", "overlapped"), trow("not overlapped", "clear"),
            ["streamed", *(frac(*tf[a]["streamed"]) for a in arms)],
            ["chat requests / embeddings requests (in turns)", *(f"{tf[a]['chat_requests']} / {tf[a]['embeddings']} ({tf[a]['embeddings_in_turns']})" for a in arms)],
            ["Spearman: turn embed ms and chat first byte", *("-" if tf[a]["spearman_embed_ms"] is None else f"{tf[a]['spearman_embed_ms']:+.2f} (n={tf[a]['pairs']})" for a in arms)],
        ])

    cold = rep["cold_start"]
    out += ["## The second model", "",
            (f"Every cell started with the embedding model unloaded ({cold['cold']} of {cold['cells']} cells; the driver unloaded it in {cold['unloaded_by_driver']}, the chat model never)."
             if cold["cells"] and cold["cold"] == cold["cells"] else
             f"**Not every cell started with the embedding model unloaded**: {cold['cold']} of {cold['cells']} did."),
            "", "oMLX process rss (median over cells, MB) and the models it had loaded, before a cell, after the one-token warm-up requests (the chat model always, the embedding model only in M2), after teaching and after the cell. Loaded = cells in which the model was in memory / cells where the server said.", ""]
    cs = rep["cost"]
    def mbs(a: str, s: str) -> str:
        v = cs[a]["rss_mb"][s]
        return "-" if v is None else f"{v:,.0f}"
    def signed(v: float | None) -> str:
        return "-" if v is None else f"{v:+,.0f}"
    def seen(a: str, kind: str, s: str) -> str:
        k, n = cs[a][kind][s]
        return f"{k}/{n}" if n else "-"
    out += table(["", *arms], [
        *([name, *(mbs(a, s) for a in arms)] for name, s in (("rss before", "before"), ("rss after warm-up", "after_warmup"), ("rss after teaching", "after_teach"), ("rss after cell", "after"))),
        ["rss added by the warm-up", *(signed(cs[a]["rss_delta_mb"]["warmup"]) for a in arms)], ["rss added by the cell", *(signed(cs[a]["rss_delta_mb"]["cell"]) for a in arms)],
        ["pool memory added by the cell (MB)", *(signed(cs[a]["pool_delta_mb"]) for a in arms)],
        *([f"embedding model loaded {name}", *(seen(a, "embedding_loaded", s) for a in arms)] for name, s in (("before the cell", "before"), ("after warm-up", "after_warmup"), ("after teaching", "after_teach"), ("after the cell", "after"))),
        ["chat model loaded after the cell", *(seen(a, "chat_loaded", "after") for a in arms)],
        ["unload wait ms (when it had to unload)", *(ms(cs[a]["unload_wait_ms"]) for a in arms)],
        ["chat warm-up ms", *(ms(cs[a]["chat_warmup_ms"]) for a in arms)], ["embedding warm-up ms (M2: the load)", *(ms(cs[a]["embed_warmup_ms"]) for a in arms)],
        ["cell wall s", *(ms(cs[a]["cell_wall_s"]) for a in arms)],
    ])

    out += ["## Teaching, and what the driver did", "",
            "Teaching messages that made a note (`turn.memory.remembered`). The driver denies every approval request (the chat model trying to save a fact to a file leaves the turn waiting) and stops a turn that takes too long; both are counted here, per phase, as turns / approval requests denied. Questions can also be answered from the chat that taught them.", ""]
    def cnt(a: str, phase: str, key: str) -> str:
        return str(iv[a][phase][key])
    out += table(["", *arms], [
        ["teaching turns", *(cnt(a, "teach", "turns") for a in arms)], ["made a note", *(cnt(a, "teach", "made_a_note") for a in arms)],
        ["refused", *(cnt(a, "teach", "refused") for a in arms)], ["teaching turns failed", *(cnt(a, "teach", "failed") for a in arms)],
        ["teaching: turns with a denied approval / requests denied", *(f"{cnt(a, 'teach', 'turns_with_denied')} / {cnt(a, 'teach', 'approvals_denied')}" for a in arms)],
        ["teaching: turns stopped by the timeout", *(cnt(a, "teach", "timed_out") for a in arms)],
        ["questions: turns with a denied approval / requests denied", *(f"{cnt(a, 'ask', 'turns_with_denied')} / {cnt(a, 'ask', 'approvals_denied')}" for a in arms)],
        ["questions: turns stopped by the timeout", *(cnt(a, "ask", "timed_out") for a in arms)],
    ])

    pr = rep["paired"]
    out += [f"## {pr['a']} against {pr['b']}, by question" if pr else "## Paired comparison", ""]
    if pr is None:
        out += ["Both arms are needed.", ""]
    else:
        a, b = pr["a"], pr["b"]
        out += [f"Pairs are the same question in the same repeat. McNemar: exact two-sided p on the discordant pairs. {a} - {b} is in percentage points.", ""]
        def prow(name: str, s: Mapping[str, Any]) -> list[str]:
            return [name, str(s["n"]), str(s["both"]), str(s["only_a"]), str(s["only_b"]), str(s["neither"]), f"{s['diff_pp']:+.1f}", f"{s['p']:.3f}"]
        out += table(["kind", "pairs", "both right", f"{a} only", f"{b} only", "both wrong", f"{a} - {b} (pp)", "McNemar p"], [*(prow(k, s) for k, s in pr["by_kind"].items()), prow("all", pr["overall"])])
        ci = pr["ci_diff_pp"]
        if ci:
            out += [f"Accuracy difference {a} - {b}: {pr['overall']['diff_pp']:+.1f} pp, 95% bootstrap interval over questions [{ci[0]:+.1f}, {ci[1]:+.1f}].", ""]
        for name, key in (("recall ms", "recall_ms"), ("turn wall ms", "wall_ms")):
            s = pr[key]
            out += [f"{name}: median per question {a} {ms(s['median_a'])}, {b} {ms(s['median_b'])}, median paired difference {ms(s['median_diff'])} (n={s['n']} questions)."]
        out += [""]
        for label, key in ((f"Right with {a}, wrong with {b}, in every repeat", "only_a_always"), (f"Right with {b}, wrong with {a}, in every repeat", "only_b_always")):
            out += [f"{label}: " + ("; ".join(f"{q} ({t})" for q, t in pr[key]) if pr[key] else "none"), ""]
    return "\n".join(out).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tasks", type=Path, default=None, help="the task file the run used (default: tasks_v1.yaml, or tasks_v2.yaml for a v2 run)")
    ap.add_argument("--pair", default=",".join(PAIR), help="v1: the two arms to compare question by question (default %(default)s)")
    ap.add_argument(
        "--compare", nargs="+", metavar=("OTHER_RUN_DIR", "ARM"),
        help="v2: pair this run with another run of the same task file, ask by ask: the arm in both (default: every arm both have)")
    return ap.parse_args(argv)


def v2_task_file(sha256: str | None) -> Path:
    """The shipped v2 task file (``data/*.yaml`` with ``version: 2``) whose fingerprint is ``sha256``, so a run on
    tasks_sleep.yaml is read with that file; tasks_v2.yaml when none matches (the report then says the file differs)."""
    from . import data_v2

    for path in sorted(data_v2.DATA_FILE.parent.glob("*.yaml")):
        try:
            if sha256 and data_v2.load(path).sha256() == sha256:
                return path
        except (data.DataError, ValueError, OSError):
            continue
    return data_v2.DATA_FILE


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads((args.run_dir / "across.json").read_text(encoding="utf-8")).get("config", {})
    if config.get("protocol") == "across-run-v3":  # a run on a v2 task file (run.PROTOCOL_V2): report_v2 reads it
        from . import data_v2, report_v2

        return report_v2.main(args.run_dir, args.tasks or v2_task_file(config.get("tasks_sha256")), args.compare)
    if args.compare:
        raise SystemExit("--compare reads runs of the v2 task file")
    pair = tuple(a.strip() for a in args.pair.split(","))
    if len(pair) != 2:
        raise SystemExit("--pair takes two arms")
    rep = analyse(args.run_dir, data.load(args.tasks or data.DATA_FILE), pair)
    (args.run_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "report.md").write_text(render(rep), encoding="utf-8")
    print(f"wrote {args.run_dir / 'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
