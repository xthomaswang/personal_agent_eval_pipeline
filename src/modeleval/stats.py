"""Capability statistics (DESIGN §6.6): per-task scores, exact sign tests, the minimum detectable
difference and the decision rules fixed in advance.

The unit of analysis is the task, not the run: a task's score is the number of harnesses that solved
it (0-3). The three harnesses run the same task with the same model, so their runs are correlated;
treating 66 runs as 66 independent trials (McNemar, Wilson over runs) would overstate the power.
Wilson intervals are therefore only printed next to per-harness counts, never used for a decision.

A comparison scores only the (harness, task) pairs that have a valid final record on both sides. A
pair excluded on one side (invalid twice, or missing from a run that finished) is never scored as an
unsolved run: that would cap the task at 0-2 against the other model's 0-3 and count a loss the
model never had. Excluded pairs are listed, and a decision that rests on fewer pairs is provisional.

Capability records (`capability/results.json`, written by the capability runner) are read through
small accessors, so the field names live in one place:

    harness, task, set ("gate" | "ranking"), verdict, solved (bool, else check.passed),
    solved_within_1200 (bool, else solved and wall_s <= 1200), wall_s, model_s, steps, anomalies
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

ALPHA = 0.05
GATE_MARGIN = 5  # gate set: "worse" at a net loss of >= 5 tasks ...
GATE_HARNESSES = 2  # ... with the sign consistent in at least two harnesses
FAST_WALL_S = 1200  # the reference-equivalent time budget of the "solved within 1200 s" column
EXCLUDED_VERDICTS = frozenset({"invalid", "server_lost"})  # pipeline mismatches; re-run, never scored


# ------------------------------------------------------------------ record accessors
def rec_set(r: dict[str, Any]) -> str | None:
    return r.get("set") or r.get("task_set")


def rec_task(r: dict[str, Any]) -> str | None:
    return r.get("task") or r.get("task_id")


def rec_harness(r: dict[str, Any]) -> str | None:
    return r.get("harness")


def rec_valid(r: dict[str, Any]) -> bool:
    """Only pipeline mismatches are excluded (DESIGN §6.4); a 5xx or a timeout is the model's result."""
    if r.get("valid") is False:
        return False
    return str(r.get("verdict") or "").lower() not in EXCLUDED_VERDICTS


def rec_solved(r: dict[str, Any]) -> bool:
    """Solved = the checker passed within the step budget, also after a wall timeout (DESIGN §6.4).
    A smoke verdict of `timeout` with a passing check is therefore solved."""
    if isinstance(r.get("solved"), bool):
        return r["solved"]
    check = r.get("check")
    if isinstance(check, dict) and isinstance(check.get("passed"), bool):
        return check["passed"]
    for k in ("check_passed", "passed"):
        if isinstance(r.get(k), bool):
            return r[k]
    return str(r.get("verdict") or "").lower() == "pass"


def rec_solved_fast(r: dict[str, Any]) -> bool:
    for k in ("solved_within_1200", "solved_1200s", "solved_within_budget"):
        if isinstance(r.get(k), bool):
            return r[k]
    wall = rec_float(r, "wall_s")
    return rec_solved(r) and wall is not None and wall <= FAST_WALL_S


def rec_float(r: dict[str, Any], key: str) -> float | None:
    v = r.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def rec_steps(r: dict[str, Any]) -> float | None:
    for k in ("steps", "n_llm"):
        v = rec_float(r, k)
        if v is not None:
            return v
    return None


def rec_anomalies(r: dict[str, Any]) -> list[str]:
    a = r.get("anomalies")
    if isinstance(a, dict):
        return sorted(k for k, v in a.items() if v)
    if isinstance(a, (list, tuple)):
        return [str(x) for x in a if x]
    return []


def final_records(results: Iterable[dict[str, Any]], set_name: str | None = None) -> list[dict[str, Any]]:
    """The valid record per (harness, task), the last one when a pair was re-run."""
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        if set_name is not None and rec_set(r) != set_name:
            continue
        if not rec_valid(r) or rec_task(r) is None or rec_harness(r) is None:
            continue
        last[(str(rec_harness(r)), str(rec_task(r)))] = r
    return list(last.values())


# ------------------------------------------------------------------ scores
def per_task_scores(results: list[dict], set_name: str) -> dict[str, int]:
    """task -> number of harnesses that solved it (0-3)."""
    scores: dict[str, int] = {}
    for r in final_records(results, set_name):
        t = str(rec_task(r))
        scores[t] = scores.get(t, 0) + (1 if rec_solved(r) else 0)
    return dict(sorted(scores.items()))


def per_harness(results: list[dict], set_name: str) -> dict[str, dict[str, bool]]:
    """harness -> {task: solved}, for the harness-consistency part of the gate rule."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in final_records(results, set_name):
        out[str(rec_harness(r))][str(rec_task(r))] = rec_solved(r)
    return {h: dict(sorted(v.items())) for h, v in sorted(out.items())}


# ------------------------------------------------------------------ tests and intervals
def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped): P(|X - n/2| >= |w - n/2|) under Binomial(n, 1/2)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def min_margin(discordant: int, alpha: float = ALPHA) -> int | None:
    """Smallest net margin |wins - losses| that the sign test calls significant when `discordant`
    tasks differ; None when even a clean sweep is not significant."""
    for w in range((discordant + 1) // 2, discordant + 1):  # p falls as the split gets more lopsided
        if sign_test(w, discordant - w) < alpha:
            return w - (discordant - w)
    return None


def min_discordant(alpha: float = ALPHA) -> int:
    """Fewest differing tasks for which a clean sweep is significant (6 at alpha 0.05)."""
    d = 1
    while sign_test(d, 0) >= alpha:
        d += 1
    return d


def mdd(n_tasks: int, discordant: int, set_name: str | None = None, alpha: float = ALPHA) -> dict[str, Any]:
    """Minimum detectable difference at the current sizes, in tasks and as a share of the set."""
    out: dict[str, Any] = {"n_tasks": n_tasks, "discordant": discordant, "min_discordant": min_discordant(alpha),
                           "margin_at_discordant": min_margin(discordant, alpha),
                           "margin_all_tasks": min_margin(n_tasks, alpha)}
    if set_name == "gate":
        out["gate_margin"] = GATE_MARGIN
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """Wilson score interval for k of n (display only)."""
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------ comparison and decision
def _harness_nets(new_h: dict[str, dict[str, bool]], ref_h: dict[str, dict[str, bool]]) -> dict[str, dict[str, int]]:
    out = {}
    for h in sorted(set(new_h) & set(ref_h)):
        common = set(new_h[h]) & set(ref_h[h])
        won = sum(1 for t in common if new_h[h][t] and not ref_h[h][t])
        lost = sum(1 for t in common if ref_h[h][t] and not new_h[h][t])
        out[h] = {"wins": won, "losses": lost, "net": won - lost, "tasks": len(common)}
    return out


def decide(set_name: str | None, wins: int, losses: int, p: float,
           harness_nets: dict[str, dict[str, int]] | None) -> tuple[str, str]:
    """The decision rule of DESIGN §6.6, fixed in advance. Returns (decision, rule text)."""
    if set_name == "gate":
        rule = (f"not worse unless the new model loses on >= {GATE_MARGIN} tasks more than it wins, "
                f"with the sign consistent in at least {GATE_HARNESSES} harnesses")
        behind = sum(1 for v in (harness_nets or {}).values() if v["net"] < 0)
        if losses - wins >= GATE_MARGIN and behind >= GATE_HARNESSES:
            return "worse", rule
        return "not worse", rule
    rule = f"better / worse only at sign-test p < {ALPHA}; otherwise no detectable difference"
    if p < ALPHA and wins != losses:
        return ("better" if wins > losses else "worse"), rule
    return "no detectable difference", rule


def _all_pairs(results: Iterable[dict[str, Any]], set_name: str | None) -> set[tuple[str, str]]:
    """Every (harness, task) a run has any record of, valid or not."""
    return {(str(rec_harness(r)), str(rec_task(r))) for r in results
            if (set_name is None or rec_set(r) == set_name) and rec_task(r) is not None and rec_harness(r) is not None}


def compare(new_results: list[dict], ref_results: list[dict], *, set_name: str | None = None) -> dict:
    """The two models' records of one set, compared per task over the (harness, task) pairs that
    have a valid final record on both sides: a task's difference is the sum over those pairs of
    solved_new - solved_ref. Then the sign test across tasks, the task counts where each model is
    ahead, the per-harness nets of the gate rule, the MDD and the decision. `excluded` lists the
    pairs left out ("harness/task: why"); with any, `provisional` is true."""
    new_h, ref_h = per_harness(new_results, set_name), per_harness(ref_results, set_name)
    valid_new = {(h, t) for h, ts in new_h.items() for t in ts}
    valid_ref = {(h, t) for h, ts in ref_h.items() for t in ts}
    common = valid_new & valid_ref
    excluded = []
    for h, t in sorted((_all_pairs(new_results, set_name) | _all_pairs(ref_results, set_name)) - common):
        side = [name for name, valid in (("new", valid_new), ("reference", valid_ref)) if (h, t) not in valid]
        excluded.append(f"{h}/{t}: no valid record for {' and '.join(side)}")
    diffs: dict[str, int] = {}
    score_new = score_ref = 0
    for h, t in sorted(common):
        diffs[t] = diffs.get(t, 0) + int(new_h[h][t]) - int(ref_h[h][t])
        score_new, score_ref = score_new + int(new_h[h][t]), score_ref + int(ref_h[h][t])
    diffs = dict(sorted(diffs.items()))
    wins = sum(1 for d in diffs.values() if d > 0)
    losses = sum(1 for d in diffs.values() if d < 0)
    p = sign_test(wins, losses)
    nets = _harness_nets(new_h, ref_h)
    decision, rule = decide(set_name, wins, losses, p, nets)
    tasks_new, tasks_ref = {t for _, t in valid_new}, {t for _, t in valid_ref}
    return {"set": set_name, "tasks": len(diffs), "pairs": len(common), "diffs": diffs, "wins": wins,
            "losses": losses, "ties": len(diffs) - wins - losses, "net": wins - losses, "p": p,
            "score_new": score_new, "score_ref": score_ref,
            "only_new": sorted(tasks_new - tasks_ref), "only_ref": sorted(tasks_ref - tasks_new),
            "excluded": excluded, "provisional": bool(excluded),
            "per_harness": nets, "mdd": mdd(len(diffs), wins + losses, set_name),
            "decision": decision, "rule": rule}


# ------------------------------------------------------------------ time to solution
def time_to_solution(new_results: list[dict], ref_results: list[dict], set_name: str) -> dict[str, Any]:
    """Paired over the (harness, task) pairs both models solved (DESIGN §6.6): sums and ratios of wall
    and model time, and steps per pair. Unpaired medians would favour the model that solves only
    the quick tasks."""
    new = {(rec_harness(r), rec_task(r)): r for r in final_records(new_results, set_name) if rec_solved(r)}
    ref = {(rec_harness(r), rec_task(r)): r for r in final_records(ref_results, set_name) if rec_solved(r)}
    pairs = sorted(set(new) & set(ref))

    def tot(side: dict, key: str) -> tuple[float | None, int]:
        vals = [rec_float(side[k], key) for k in pairs]
        have = [v for v in vals if v is not None]
        return (sum(have) if have else None), len(have)

    def ratio(a: float | None, b: float | None) -> float | None:
        return a / b if a is not None and b else None

    wall_n, wn = tot(new, "wall_s")
    wall_r, wr = tot(ref, "wall_s")
    model_n, mn = tot(new, "model_s")
    model_r, mr = tot(ref, "model_s")
    steps_n = [rec_steps(new[k]) for k in pairs]
    steps_r = [rec_steps(ref[k]) for k in pairs]
    sn = [x for x in steps_n if x is not None]
    sr = [x for x in steps_r if x is not None]
    complete_model = mn == mr == len(pairs)
    return {"pairs": len(pairs), "wall_new": wall_n, "wall_ref": wall_r,
            "wall_ratio": ratio(wall_n, wall_r) if wn == wr == len(pairs) else None,
            "model_new": model_n if complete_model else None, "model_ref": model_r if complete_model else None,
            "model_ratio": ratio(model_n, model_r) if complete_model else None,
            "steps_new": sum(sn) / len(sn) if sn else None, "steps_ref": sum(sr) / len(sr) if sr else None}


def harness_counts(results: list[dict], set_name: str) -> dict[str, dict[str, Any]]:
    """Per harness: solved, solved within 1200 s, scored runs, anomalies, Wilson interval (display)."""
    out: dict[str, dict[str, Any]] = {}
    for r in final_records(results, set_name):
        h = str(rec_harness(r))
        d = out.setdefault(h, {"solved": 0, "fast": 0, "runs": 0, "anomalies": 0})
        d["runs"] += 1
        d["solved"] += rec_solved(r)
        d["fast"] += rec_solved_fast(r)
        d["anomalies"] += bool(rec_anomalies(r))
    for d in out.values():
        d["wilson"] = wilson(d["solved"], d["runs"])
    return dict(sorted(out.items()))
