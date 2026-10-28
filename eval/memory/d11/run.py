"""Run D11's cross-task protocol (aider12-v0 x 5 passes x arms off/on) and build the D2 report.

    BENCH_LOCALMELO_REPO=memory/vendor/localmelo uv run python -m memory.d11.run --run-dir <trial>

See memory/native_t4_and_d11.md, section 2. Each arm has one shared memory directory
(<trial>/mem-off, <trial>/mem-on), runs with --admit-on-outcome <arm> and --reasoning compress,
and gets a fresh copy of the exercise on every run so only memory carries over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "memory" / "vendor" / "localmelo"
AGENT_ABILITY = ROOT / "agent_ability"
if str(AGENT_ABILITY) not in sys.path:
    sys.path.insert(0, str(AGENT_ABILITY))

from smoke import adapters, checks, paths, runner, tasks  # noqa: E402

ARMS: tuple[str, str] = ("off", "on")
EXERCISES: tuple[str, ...] = (*tasks.AIDER_CORE, *tasks.AIDER_EXTENDED)
DEFAULT_PASSES = 5
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_STEPS = 30
DEFAULT_THINKING = "off"
TASK_LOCAL_KINDS = frozenset({"note", "draft", "thought", "digest", "reflection"})


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def log_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def arms_for_pass(pass_num: int) -> tuple[str, str]:
    """Alternate which arm goes first by pass number so drift falls on both."""
    return ("off", "on") if pass_num % 2 == 1 else ("on", "off")


def pass_permutation(pass_num: int, exercises: Sequence[str] = EXERCISES) -> list[str]:
    """Per-pass exercise order seeded by the pass number, identical for both arms."""
    order = list(exercises)
    random.Random(pass_num).shuffle(order)
    return order


def schedule(passes: int = DEFAULT_PASSES, exercises: Sequence[str] = EXERCISES) -> list[dict[str, Any]]:
    """The deterministic D11 schedule (120 runs for 5 passes x 12 exercises x 2 arms)."""
    if passes < 1:
        raise ValueError(f"passes must be at least 1, got {passes}")
    if not exercises:
        raise ValueError("exercises must not be empty")
    planned: list[dict[str, Any]] = []
    for pass_num in range(1, passes + 1):
        order = pass_permutation(pass_num, exercises)
        arms = arms_for_pass(pass_num)
        for pos, exercise in enumerate(order, 1):
            for arm in arms:
                run_id = f"p{pass_num:02d}__{exercise}__{arm}"
                planned.append(
                    {
                        "index": len(planned) + 1,
                        "pass": pass_num,
                        "order_in_pass": pos,
                        "exercise": exercise,
                        "task": f"aider/{exercise}",
                        "arm": arm,
                        "first_arm": arms[0],
                        "run_id": run_id,
                    }
                )
    return planned


def identity(
    localmelo: Path,
    *,
    passes: int = DEFAULT_PASSES,
    exercises: Sequence[str] = EXERCISES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_steps: int = DEFAULT_MAX_STEPS,
    thinking: str = DEFAULT_THINKING,
) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "-C", str(localmelo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "protocol": "d11-v1",
        "suite": "aider12-v0",
        "reasoning": "compress",
        "arms": list(ARMS),
        "passes": passes,
        "exercises": list(exercises),
        "budget": {"max_steps": max_steps, "timeout_s": timeout_s, "thinking": thinking},
        "localmelo_sha": sha,
        "localmelo_repo": str(localmelo),
        "polyglot_sha": paths.POLYGLOT_SHA,
        "model": os.environ.get("BENCH_MODEL", "Qwen3.8-27B-oQ5-mtp"),
    }


def _history_task_ids(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall()
    except sqlite3.Error:
        return []
    return [str(r[0]) for r in rows if r and r[0]]


def _history_task_info(db_path: Path, task_id: str | None) -> dict[str, Any]:
    if not task_id or not db_path.exists():
        return {"steps": None, "llm_calls": None, "task_status": None, "recalls": []}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            steps_row = conn.execute("SELECT COUNT(*) FROM steps WHERE task_id = ?", (task_id,)).fetchone()
            calls_row = conn.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE task_id = ? AND kind IN ('plan', 'reflect')",
                (task_id,),
            ).fetchone()
            status_row = conn.execute("SELECT status FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            recall_rows = conn.execute(
                "SELECT recall_json FROM llm_calls WHERE task_id = ? AND recall_json IS NOT NULL ORDER BY seq",
                (task_id,),
            ).fetchall()
    except sqlite3.Error:
        return {"steps": None, "llm_calls": None, "task_status": None, "recalls": []}
    recalls: list[dict[str, Any]] = []
    for (raw,) in recall_rows:
        recalls.extend(parse_recall_records(raw))
    return {
        "steps": int(steps_row[0]) if steps_row and steps_row[0] is not None else None,
        "llm_calls": int(calls_row[0]) if calls_row and calls_row[0] is not None else None,
        "task_status": str(status_row[0]) if status_row and status_row[0] is not None else None,
        "recalls": recalls,
    }


def parse_recall_records(raw: Any) -> list[dict[str, Any]]:
    """Normalize a recall_json cell (JSON string, list of records, or single record dict)."""
    if not raw:
        return []
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _invoke_harness(
    prompt: str,
    workspace: Path,
    run_dir: Path,
    mem_dir: Path,
    home_dir: Path,
    env_file: Path,
    localmelo: Path,
    *,
    arm: str,
    timeout_s: float,
    max_steps: int,
    thinking: str,
) -> dict[str, Any]:
    from bench import harness, manifest

    manifest.LOCALMELO_REPO = localmelo.resolve()
    return harness.run(
        "localmelo",
        prompt,
        workspace,
        run_dir,
        timeout_s=timeout_s,
        max_steps=max_steps,
        thinking=thinking,
        harness_args=["--admit-on-outcome", arm, "--reasoning", "compress", "--output-format", "json"],
        extra_env={
            **adapters.agent_env(env_file),
            "LOCALMELO_PERSIST_MEMORY": "1",
            "LOCALMELO_MEMORY_DIR": str(mem_dir),
            "HOME": str(home_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def _run_item(
    item: dict[str, Any],
    trial: Path,
    localmelo: Path,
    env_file: Path,
    *,
    timeout_s: float,
    max_steps: int,
    thinking: str,
) -> dict[str, Any]:
    from bench.proxy.steps import step_counts

    task = tasks.aider_task(item["exercise"])
    arm = item["arm"]
    mem_dir = trial / f"mem-{arm}"
    home_dir = trial / "home"
    mem_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    ws = trial / "ws" / task.slug
    tasks.build_workspace(task, ws)
    prompt_text = tasks.prompt(task, ws)

    run_dir = trial / "runs" / item["run_id"]
    if run_dir.exists():
        shutil.rmtree(run_dir)

    before_tasks = set(item.get("before_tasks") or _history_task_ids(mem_dir / "history.db"))
    summary = _invoke_harness(
        prompt_text,
        ws,
        run_dir,
        mem_dir,
        home_dir,
        env_file,
        localmelo,
        arm=arm,
        timeout_s=timeout_s,
        max_steps=max_steps,
        thinking=thinking,
    )

    check_result = checks.check(task, ws, run_dir)
    save(run_dir / "check.json", check_result)
    if ws.exists():
        shutil.move(ws, run_dir / "workspace")

    stdout_records = log_records(run_dir / "stdout.log")
    run_record = next((r for r in reversed(stdout_records) if "task_id" in r and "status" in r), {})
    task_id = run_record.get("task_id")
    if not task_id:
        after_tasks = _history_task_ids(mem_dir / "history.db")
        new_tasks = [t for t in after_tasks if t not in before_tasks]
        task_id = new_tasks[-1] if new_tasks else (after_tasks[-1] if after_tasks else None)

    hist = _history_task_info(mem_dir / "history.db", task_id)
    calls = log_records(run_dir / "calls.jsonl")
    steps = (summary or {}).get("steps") or (step_counts(calls) if calls else {})
    n_llm = steps.get("n_llm")
    if n_llm is None:
        n_llm = hist["llm_calls"]
    if summary is not None and n_llm is None:
        n_llm = 0

    clean_item = {k: v for k, v in item.items() if k != "before_tasks"}
    verdict = runner.verdict(summary, check_result, None)
    record = {
        **clean_item,
        "state": "finished",
        "task_id": task_id,
        "verdict": verdict,
        "passed": verdict == "pass",
        "wall_s": (summary or {}).get("wall_s"),
        "exit_code": (summary or {}).get("exit_code"),
        "timed_out": bool(summary and summary.get("timed_out")),
        "budget_exhausted": bool(summary and summary.get("timed_out")) or any(c.get("budget_exhausted") for c in calls),
        "n_llm": n_llm,
        "n_tool": steps.get("n_tool"),
        "n_llm_aux": steps.get("n_llm_aux"),
        "n_llm_err": steps.get("n_llm_err"),
        "history_steps": hist["steps"],
        "history_llm_calls": hist["llm_calls"],
        "task_status": run_record.get("status") or hist["task_status"],
        "check": check_result,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "run_dir": str(run_dir.relative_to(trial)),
        "mem_dir": str(mem_dir.relative_to(trial)),
        "error": None,
    }
    save(run_dir / "record.json", record)
    return record


def _salvage_interrupted(item: dict[str, Any], trial: Path) -> dict[str, Any]:
    """Record a 'started' item with no record.json as 'interrupted', salvaging summary.json and history.db."""
    from bench.proxy.steps import step_counts

    run_id = item["run_id"]
    arm = item["arm"]
    run_dir = trial / "runs" / run_id
    mem_dir = trial / f"mem-{arm}"
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    check_path = run_dir / "check.json"
    check_result = json.loads(check_path.read_text()) if check_path.exists() else None

    stdout_records = log_records(run_dir / "stdout.log")
    run_record = next((r for r in reversed(stdout_records) if "task_id" in r and "status" in r), {})
    task_id = run_record.get("task_id")
    if not task_id:
        before_tasks = set(item.get("before_tasks") or ())
        after_tasks = _history_task_ids(mem_dir / "history.db")
        new_tasks = [t for t in after_tasks if t not in before_tasks]
        if new_tasks:
            task_id = new_tasks[-1]
        elif "before_tasks" not in item and after_tasks:
            task_id = after_tasks[-1]

    hist = _history_task_info(mem_dir / "history.db", task_id)
    calls = log_records(run_dir / "calls.jsonl")
    steps = (summary.get("steps") if summary else None) or (step_counts(calls) if calls else {})
    n_llm = steps.get("n_llm")
    if n_llm is None:
        n_llm = hist["llm_calls"]
    timed_out = bool(summary and summary.get("timed_out"))
    budget = timed_out or any(c.get("budget_exhausted") for c in calls)
    clean_item = {k: v for k, v in item.items() if k != "before_tasks"}
    return {
        **clean_item,
        "state": "interrupted",
        "task_id": task_id,
        "verdict": "interrupted",
        "passed": False,
        "wall_s": summary.get("wall_s") if summary else None,
        "exit_code": summary.get("exit_code") if summary else None,
        "timed_out": timed_out,
        "budget_exhausted": budget,
        "n_llm": n_llm,
        "n_tool": steps.get("n_tool"),
        "n_llm_aux": steps.get("n_llm_aux"),
        "n_llm_err": steps.get("n_llm_err"),
        "history_steps": hist["steps"],
        "history_llm_calls": hist["llm_calls"],
        "task_status": run_record.get("status") or hist["task_status"] or "interrupted",
        "check": check_result,
        "run_dir": str(run_dir.relative_to(trial)),
        "mem_dir": str(mem_dir.relative_to(trial)),
        "error": "interrupted",
    }


def _export(localmelo: Path, mem_dir: Path, home_dir: Path, flag: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    missing = not (mem_dir / "long_term.db").exists()
    command = [str(localmelo / ".venv" / "bin" / "melo-retrieve"), flag, str(mem_dir)]
    errors = target.with_suffix(".err")
    with tmp.open("w") as out, errors.open("w") as err:
        result = subprocess.run(command, stdout=out, stderr=err, env={**os.environ, "HOME": str(home_dir)})
    save(
        target.with_suffix(".diagnostic.json"),
        {"exit_code": result.returncode, "stderr": errors.read_text(), "missing_store": missing},
    )
    if result.returncode and not missing:
        raise subprocess.CalledProcessError(result.returncode, command)
    if missing:
        tmp.write_text("")
    tmp.replace(target)


def export_arms(trial: Path, localmelo: Path) -> dict[str, dict[str, str]]:
    """Export active store entries and admission decisions for both arms."""
    home_dir = trial / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        mem_dir = trial / f"mem-{arm}"
        mem_dir.mkdir(parents=True, exist_ok=True)
        store_path = mem_dir / "export.jsonl"
        decisions_path = mem_dir / "decisions.jsonl"
        _export(localmelo, mem_dir, home_dir, "--export-store", store_path)
        _export(localmelo, mem_dir, home_dir, "--export-decisions", decisions_path)
        exports[arm] = {
            "mem_dir": str(mem_dir.relative_to(trial)),
            "store": str(store_path.relative_to(trial)),
            "decisions": str(decisions_path.relative_to(trial)),
        }
    return exports


def decision_written(row: dict[str, Any]) -> bool:
    """Whether a decision row wrote an entry into long_term.db."""
    return row.get("entry_id") is not None and str(row.get("action") or "").upper() != "IGNORE"


def classify_recall_hits(
    recalls: Sequence[dict[str, Any]],
    *,
    current_exercise: str,
    current_task_id: str | None,
    task_to_exercise: dict[str, str],
) -> dict[str, int]:
    """Split recalled long-memory hits into same_exercise vs cross_exercise by origin_task_ids."""
    same = cross = task_local = unmapped = 0
    for record in recalls:
        for hit in record.get("hits") or ():
            if not isinstance(hit, dict):
                continue
            kind = hit.get("kind")
            if kind in TASK_LOCAL_KINDS:
                task_local += 1
                continue
            raw_origins = hit.get("origin_task_ids") or ()
            if isinstance(raw_origins, str):
                raw_origins = [raw_origins]
            origins = [tid for tid in raw_origins if isinstance(tid, str) and tid]
            if current_task_id and any(tid != current_task_id for tid in origins):
                origins = [tid for tid in origins if tid != current_task_id]
            elif current_task_id and origins == [current_task_id] and kind is not None and kind != "long":
                task_local += 1
                continue
            exercises = {task_to_exercise[tid] for tid in origins if tid in task_to_exercise}
            if current_exercise in exercises:
                same += 1
            elif exercises:
                cross += 1
            else:
                unmapped += 1
    return {
        "same_exercise": same,
        "cross_exercise": cross,
        "total": same + cross,
        "task_local": task_local,
        "unmapped": unmapped,
    }


def mcnemar_test(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Paired McNemar test over (off_passed, on_passed) pairs with exact binomial two-sided p-value."""
    both_pass = sum(1 for off_ok, on_ok in pairs if off_ok and on_ok)
    on_only = sum(1 for off_ok, on_ok in pairs if not off_ok and on_ok)
    off_only = sum(1 for off_ok, on_ok in pairs if off_ok and not on_ok)
    both_fail = sum(1 for off_ok, on_ok in pairs if not off_ok and not on_ok)
    discordant = on_only + off_only
    if discordant == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = (on_only - off_only) ** 2 / discordant
        k = min(on_only, off_only)
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant))
    return {
        "n_pairs": len(pairs),
        "both_pass": both_pass,
        "on_only": on_only,
        "off_only": off_only,
        "both_fail": both_fail,
        "n_discordant": discordant,
        "statistic": round(stat, 6),
        "p_value": round(p_value, 6),
    }


def bootstrap_ci(
    diffs: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI for paired differences (on - off), reporting both mean and median."""
    values = [float(x) for x in diffs]
    if not values:
        return {
            "n_pairs": 0,
            "mean_diff": None,
            "median_diff": None,
            "ci95_mean": [None, None],
            "ci95_median": [None, None],
            "n_boot": n_boot,
        }
    n = len(values)
    rng = random.Random(seed)
    means: list[float] = []
    medians: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
        medians.append(float(statistics.median(sample)))
    means.sort()
    medians.sort()
    lo_idx = max(0, min(n_boot - 1, int(math.floor((alpha / 2.0) * n_boot))))
    hi_idx = max(0, min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1))
    return {
        "n_pairs": n,
        "mean_diff": round(sum(values) / n, 6),
        "median_diff": round(float(statistics.median(values)), 6),
        "ci95_mean": [round(means[lo_idx], 6), round(means[hi_idx], 6)],
        "ci95_median": [round(medians[lo_idx], 6), round(medians[hi_idx], 6)],
        "n_boot": n_boot,
    }


def _is_interrupted(row: dict[str, Any]) -> bool:
    return row.get("state") == "interrupted" or row.get("verdict") == "interrupted"


def _summarize_slice(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    finished = [r for r in rows if not _is_interrupted(r)]
    n = len(finished)
    n_interrupted = len(rows) - n
    n_pass = sum(1 for r in finished if r.get("verdict") == "pass" or r.get("passed") is True)
    llm_vals = [float(r["n_llm"]) for r in finished if r.get("n_llm") is not None]
    wall_vals = [float(r["wall_s"]) for r in finished if r.get("wall_s") is not None]
    written_vals = [int(r.get("entries_written", 0)) for r in finished]
    signals: Counter[str] = Counter()
    for r in finished:
        for sig, cnt in (r.get("admitted_by") or {}).items():
            signals[sig] += int(cnt)
    same_hits = sum(int((r.get("recall_hits") or {}).get("same_exercise", 0)) for r in finished)
    cross_hits = sum(int((r.get("recall_hits") or {}).get("cross_exercise", 0)) for r in finished)
    task_local_hits = sum(int((r.get("recall_hits") or {}).get("task_local", 0)) for r in finished)
    return {
        "n_runs": n,
        "n_interrupted": n_interrupted,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 6) if n else None,
        "n_llm": {
            "sum": int(sum(llm_vals)) if llm_vals else 0,
            "mean": round(sum(llm_vals) / len(llm_vals), 4) if llm_vals else None,
            "median": round(float(statistics.median(llm_vals)), 4) if llm_vals else None,
        },
        "wall_s": {
            "sum": round(sum(wall_vals), 3) if wall_vals else 0.0,
            "mean": round(sum(wall_vals) / len(wall_vals), 3) if wall_vals else None,
            "median": round(float(statistics.median(wall_vals)), 3) if wall_vals else None,
        },
        "admitted_by": dict(signals),
        "entries_written": {
            "total": sum(written_vals),
            "mean_per_run": round(sum(written_vals) / n, 4) if n else 0.0,
            "max_per_run": max(written_vals) if written_vals else 0,
        },
        "recall_hits": {
            "same_exercise": same_hits,
            "cross_exercise": cross_hits,
            "total": same_hits + cross_hits,
            "task_local": task_local_hits,
        },
    }


def evaluate_d2(enriched_runs: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Evaluate the D2 decision rule and paired tests on passes 2-5, leaving interrupted runs out of pairs."""
    by_pair: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    interrupted_ids: list[str] = []
    for r in enriched_runs:
        if _is_interrupted(r):
            interrupted_ids.append(str(r.get("run_id") or f"p{int(r['pass']):02d}__{r['exercise']}__{r['arm']}"))
            continue
        if int(r["pass"]) >= 2:
            by_pair.setdefault((int(r["pass"]), str(r["exercise"])), {})[str(r["arm"])] = r

    paired_keys = sorted(k for k, arms in by_pair.items() if "off" in arms and "on" in arms)
    off_rows = [by_pair[k]["off"] for k in paired_keys]
    on_rows = [by_pair[k]["on"] for k in paired_keys]
    n_pairs = len(paired_keys)

    pass_pairs = [
        (
            bool(off.get("verdict") == "pass" or off.get("passed") is True),
            bool(on.get("verdict") == "pass" or on.get("passed") is True),
        )
        for off, on in zip(off_rows, on_rows, strict=True)
    ]
    llm_pairs = [
        (float(off["n_llm"]), float(on["n_llm"]))
        for off, on in zip(off_rows, on_rows, strict=True)
        if off.get("n_llm") is not None and on.get("n_llm") is not None
    ]
    wall_pairs = [
        (float(off["wall_s"]), float(on["wall_s"]))
        for off, on in zip(off_rows, on_rows, strict=True)
        if off.get("wall_s") is not None and on.get("wall_s") is not None
    ]

    pass_rate_off = sum(1 for off_ok, _ in pass_pairs if off_ok) / n_pairs if n_pairs else None
    pass_rate_on = sum(1 for _, on_ok in pass_pairs if on_ok) / n_pairs if n_pairs else None
    pass_rate_diff = (pass_rate_on - pass_rate_off) if (pass_rate_on is not None and pass_rate_off is not None) else None

    med_llm_off = float(statistics.median(off for off, _ in llm_pairs)) if llm_pairs else None
    med_llm_on = float(statistics.median(on for _, on in llm_pairs)) if llm_pairs else None
    med_wall_off = float(statistics.median(off for off, _ in wall_pairs)) if wall_pairs else None
    med_wall_on = float(statistics.median(on for _, on in wall_pairs)) if wall_pairs else None

    on_written_p25 = [int(r.get("entries_written", 0)) for r in on_rows]
    mean_written_on = sum(on_written_p25) / len(on_written_p25) if on_written_p25 else 0.0
    max_written_on = max(on_written_p25) if on_written_p25 else 0

    all_on_written = [int(r.get("entries_written", 0)) for r in enriched_runs if r.get("arm") == "on" and not _is_interrupted(r)]
    all_mean_written_on = sum(all_on_written) / len(all_on_written) if all_on_written else 0.0
    all_max_written_on = max(all_on_written) if all_on_written else 0

    non_inferior = pass_rate_diff is not None and pass_rate_diff >= -0.10 - 1e-9
    fewer_calls = med_llm_on is not None and med_llm_off is not None and med_llm_on < med_llm_off
    less_wall = med_wall_on is not None and med_wall_off is not None and med_wall_on < med_wall_off
    efficiency_gain = bool(fewer_calls or less_wall)
    bounded_growth = bool(on_rows) and (mean_written_on <= 3.0 + 1e-9) and (max_written_on <= 8)
    decision_pass = bool(n_pairs > 0 and non_inferior and efficiency_gain and bounded_growth)

    return {
        "passes": [2, 3, 4, 5],
        "n_pairs": n_pairs,
        "interrupted_runs": interrupted_ids,
        "pass_rate_off": round(pass_rate_off, 6) if pass_rate_off is not None else None,
        "pass_rate_on": round(pass_rate_on, 6) if pass_rate_on is not None else None,
        "pass_rate_diff": round(pass_rate_diff, 6) if pass_rate_diff is not None else None,
        "pass_rate_diff_pp": round(pass_rate_diff * 100.0, 4) if pass_rate_diff is not None else None,
        "median_n_llm_off": round(med_llm_off, 4) if med_llm_off is not None else None,
        "median_n_llm_on": round(med_llm_on, 4) if med_llm_on is not None else None,
        "median_wall_s_off": round(med_wall_off, 3) if med_wall_off is not None else None,
        "median_wall_s_on": round(med_wall_on, 3) if med_wall_on is not None else None,
        "entries_written_on": {
            "mean_per_run": round(mean_written_on, 4),
            "max_per_run": max_written_on,
            "all_passes_mean_per_run": round(all_mean_written_on, 4),
            "all_passes_max_per_run": all_max_written_on,
        },
        "criteria": {
            "non_inferior_pass_rate": non_inferior,
            "fewer_calls": fewer_calls,
            "less_wall_time": less_wall,
            "efficiency_gain": efficiency_gain,
            "bounded_growth": bounded_growth,
        },
        "admit_on_outcome_default_on": decision_pass,
        "mcnemar": mcnemar_test(pass_pairs),
        "bootstrap_n_llm": bootstrap_ci([on - off for off, on in llm_pairs], n_boot=n_boot, seed=seed),
        "bootstrap_wall_s": bootstrap_ci([on - off for off, on in wall_pairs], n_boot=n_boot, seed=seed),
    }


def build_report(
    runs: Sequence[dict[str, Any]],
    *,
    decisions_by_arm: dict[str, Sequence[dict[str, Any]]] | None = None,
    recalls_by_task: dict[str, Sequence[dict[str, Any]]] | None = None,
    entries_by_arm: dict[str, Sequence[dict[str, Any]]] | None = None,
    config: dict[str, Any] | None = None,
    exports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-arm/pass measures, recall hit splits, and the D2 decision rule."""
    decisions_by_arm = decisions_by_arm or {}
    recalls_by_task = recalls_by_task or {}
    entries_by_arm = entries_by_arm or {}

    task_to_exercise: dict[str, str] = {
        str(r["task_id"]): str(r["exercise"])
        for r in runs
        if r.get("task_id") and r.get("exercise")
    }

    enriched: list[dict[str, Any]] = []
    for r in runs:
        arm = str(r["arm"])
        task_id = r.get("task_id")
        exercise = str(r["exercise"])
        arm_decisions = [d for d in decisions_by_arm.get(arm, ()) if task_id and d.get("task_id") == task_id]
        if arm_decisions:
            admitted_by = dict(
                Counter(
                    str(d.get("admitted_by") or "unspecified")
                    for d in arm_decisions
                    if decision_written(d)
                )
            )
            entries_written = sum(1 for d in arm_decisions if decision_written(d))
        else:
            admitted_by = dict(r.get("admitted_by") or {})
            entries_written = int(r["entries_written"]) if "entries_written" in r else sum(admitted_by.values())

        raw_recalls = recalls_by_task.get(str(task_id)) if task_id and str(task_id) in recalls_by_task else r.get("recalls")
        if raw_recalls is not None:
            recall_hits = classify_recall_hits(
                raw_recalls,
                current_exercise=exercise,
                current_task_id=str(task_id) if task_id else None,
                task_to_exercise=task_to_exercise,
            )
        elif "recall_hits" in r:
            rh = r["recall_hits"]
            same = int(rh.get("same_exercise", 0))
            cross = int(rh.get("cross_exercise", 0))
            recall_hits = {
                "same_exercise": same,
                "cross_exercise": cross,
                "total": same + cross,
                "task_local": int(rh.get("task_local", 0)),
                "unmapped": int(rh.get("unmapped", 0)),
            }
        else:
            recall_hits = {"same_exercise": 0, "cross_exercise": 0, "total": 0, "task_local": 0, "unmapped": 0}

        enriched.append(
            {
                **r,
                "passed": False if _is_interrupted(r) else bool(r.get("verdict") == "pass" or r.get("passed") is True),
                "admitted_by": admitted_by,
                "entries_written": entries_written,
                "recall_hits": recall_hits,
            }
        )

    passes = sorted({int(r["pass"]) for r in enriched})
    by_arm_and_pass: dict[str, dict[str, Any]] = {}
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for r in enriched if r["arm"] == arm]
        by_arm[arm] = {
            **_summarize_slice(arm_rows),
            "active_store_entries": len(entries_by_arm.get(arm, ())),
            "total_decisions": len(decisions_by_arm.get(arm, ())),
        }
        by_arm_and_pass[arm] = {
            str(p): _summarize_slice([r for r in arm_rows if int(r["pass"]) == p])
            for p in passes
        }

    interrupted = [
        {
            "run_id": r.get("run_id") or f"p{int(r['pass']):02d}__{r['exercise']}__{r['arm']}",
            "pass": r["pass"],
            "exercise": r["exercise"],
            "arm": r["arm"],
            "task_id": r.get("task_id"),
            "n_llm": r.get("n_llm"),
            "wall_s": r.get("wall_s"),
        }
        for r in enriched
        if _is_interrupted(r)
    ]
    d2 = evaluate_d2(enriched)
    return {
        "schema_version": 1,
        "protocol": "d11-v1",
        "config": config or {},
        "exports": exports or {},
        "interrupted": interrupted,
        "by_arm": by_arm,
        "by_arm_and_pass": by_arm_and_pass,
        "d2": d2,
        "runs": enriched,
    }


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report.get("config") or {}
    d2 = report["d2"]
    interrupted = report.get("interrupted") or []
    lines_out = [
        "# D11 cross-task memory report (--admit-on-outcome on vs off)",
        "",
        f"- localmelo SHA: `{cfg.get('localmelo_sha', 'unknown')}` (`--reasoning {cfg.get('reasoning', 'compress')}`)",
        f"- Model: `{cfg.get('model', 'unknown')}`; passes: {cfg.get('passes')}; "
        f"exercises: {len(cfg.get('exercises') or ())}",
        f"- Interrupted runs ({len(interrupted)}): "
        + (", ".join(f"`{r['run_id']}`" for r in interrupted) if interrupted else "none"),
        "",
        "## Per arm and pass",
        "",
        "| Arm | Pass | Runs | Pass | Pass rate | Median n_llm | Median wall_s | Written (mean / max) | Admitted by signal | Recall hits (same / cross) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for arm in ARMS:
        for pass_key, row in report["by_arm_and_pass"].get(arm, {}).items():
            rate = f"{row['pass_rate'] * 100:.1f}%" if row["pass_rate"] is not None else "n/a"
            med_llm = row["n_llm"]["median"] if row["n_llm"]["median"] is not None else "n/a"
            med_wall = row["wall_s"]["median"] if row["wall_s"]["median"] is not None else "n/a"
            ew = row["entries_written"]
            signals = ", ".join(f"{k}:{v}" for k, v in sorted(row["admitted_by"].items())) or "none"
            rh = row["recall_hits"]
            lines_out.append(
                f"| {arm} | {pass_key} | {row['n_runs']} | {row['n_pass']} | {rate} | {med_llm} | "
                f"{med_wall} | {ew['mean_per_run']:.2f} / {ew['max_per_run']} (tot {ew['total']}) | "
                f"{signals} | {rh['same_exercise']} / {rh['cross_exercise']} |"
            )

    c = d2["criteria"]
    mc = d2["mcnemar"]
    b_llm = d2["bootstrap_n_llm"]
    b_wall = d2["bootstrap_wall_s"]
    lines_out += [
        "",
        "## D2 decision rule (passes 2-5)",
        "",
        f"- Paired runs (passes 2-5): {d2['n_pairs']}",
        f"- Pass rate: off={d2['pass_rate_off']}, on={d2['pass_rate_on']}, diff={d2['pass_rate_diff_pp']} pp "
        f"(non-inferior >= -10 pp: `{c['non_inferior_pass_rate']}`)",
        f"- Efficiency (medians): n_llm off={d2['median_n_llm_off']} vs on={d2['median_n_llm_on']}; "
        f"wall_s off={d2['median_wall_s_off']} vs on={d2['median_wall_s_on']} "
        f"(efficiency gain: `{c['efficiency_gain']}`)",
        f"- Bounded growth (on): mean={d2['entries_written_on']['mean_per_run']} (<= 3), "
        f"max={d2['entries_written_on']['max_per_run']} (<= 8) (bounded: `{c['bounded_growth']}`)",
        f"- **D2 decision (`--admit-on-outcome on` by default)**: `{d2['admit_on_outcome_default_on']}`",
        "",
        "## Paired statistical tests (passes 2-5)",
        "",
        f"- McNemar (pass/fail): both_pass={mc['both_pass']}, on_only={mc['on_only']}, "
        f"off_only={mc['off_only']}, both_fail={mc['both_fail']}, statistic={mc['statistic']}, p={mc['p_value']}",
        f"- Bootstrap 95% CI (on - off n_llm): mean_diff={b_llm['mean_diff']} {b_llm['ci95_mean']}, "
        f"median_diff={b_llm['median_diff']} {b_llm['ci95_median']}",
        f"- Bootstrap 95% CI (on - off wall_s): mean_diff={b_wall['mean_diff']} {b_wall['ci95_mean']}, "
        f"median_diff={b_wall['median_diff']} {b_wall['ci95_median']}",
    ]
    return "\n".join(lines_out) + "\n"


def load_trial_report(trial: Path) -> dict[str, Any]:
    state = json.loads((trial / "d11.json").read_text())
    planned = state["schedule"]
    runs = [
        state["runs"][item["run_id"]]
        for item in planned
        if item["run_id"] in state["runs"]
        and state["runs"][item["run_id"]].get("state") in ("finished", "interrupted")
    ]
    decisions_by_arm: dict[str, list[dict[str, Any]]] = {}
    entries_by_arm: dict[str, list[dict[str, Any]]] = {}
    recalls_by_task: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        mem_dir = trial / f"mem-{arm}"
        decisions_by_arm[arm] = lines(mem_dir / "decisions.jsonl")
        entries_by_arm[arm] = lines(mem_dir / "export.jsonl")
        for r in runs:
            if r["arm"] == arm and r.get("task_id"):
                tid = str(r["task_id"])
                info = _history_task_info(mem_dir / "history.db", tid)
                recalls_by_task[tid] = info["recalls"]
    return build_report(
        runs,
        decisions_by_arm=decisions_by_arm,
        recalls_by_task=recalls_by_task,
        entries_by_arm=entries_by_arm,
        config=state.get("config"),
        exports=state.get("exports"),
    )


def run_d11(
    trial: Path,
    *,
    passes: int = DEFAULT_PASSES,
    exercises: Sequence[str] = EXERCISES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_steps: int = DEFAULT_MAX_STEPS,
    thinking: str = DEFAULT_THINKING,
) -> dict[str, Any]:
    trial = trial.resolve()
    localmelo = Path(os.environ.get("BENCH_LOCALMELO_REPO", VENDOR)).expanduser().resolve()
    os.environ["BENCH_LOCALMELO_REPO"] = str(localmelo)
    config = identity(
        localmelo,
        passes=passes,
        exercises=exercises,
        timeout_s=timeout_s,
        max_steps=max_steps,
        thinking=thinking,
    )
    os.environ.setdefault("BENCH_MODEL", config["model"])
    planned = schedule(passes=passes, exercises=exercises)

    checkpoint = trial / "d11.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config:
            raise ValueError("D11 config, localmelo SHA, exercises, passes or model changed; use a fresh directory")
    else:
        if trial.exists() and any(trial.iterdir()):
            raise ValueError("Non-empty trial directory has no d11.json checkpoint; use a fresh directory")
        state = {"config": config, "schedule": planned, "runs": {}, "exports": {}, "complete": False}
        save(checkpoint, state)

    for name in ("mem-off", "mem-on", "home", "runs", "ws"):
        (trial / name).mkdir(parents=True, exist_ok=True)
    env_file = adapters.write_env_file(trial / "agent-env.sh")

    for item in planned:
        run_id = item["run_id"]
        existing = state["runs"].get(run_id)
        if existing and existing.get("state") in ("finished", "interrupted"):
            continue
        record_file = trial / "runs" / run_id / "record.json"
        if existing and existing.get("state") == "started":
            if record_file.exists():
                state["runs"][run_id] = json.loads(record_file.read_text())
            else:
                state["runs"][run_id] = _salvage_interrupted(existing, trial)
            save(checkpoint, state)
            continue
        mem_dir = trial / f"mem-{item['arm']}"
        before_tasks = _history_task_ids(mem_dir / "history.db")
        state["runs"][run_id] = {**item, "state": "started", "before_tasks": before_tasks}
        save(checkpoint, state)
        record = _run_item(
            state["runs"][run_id],
            trial,
            localmelo,
            env_file,
            timeout_s=timeout_s,
            max_steps=max_steps,
            thinking=thinking,
        )
        state["runs"][run_id] = record
        save(checkpoint, state)

    if (trial / "ws").exists() and not any((trial / "ws").iterdir()):
        (trial / "ws").rmdir()

    state["exports"] = export_arms(trial, localmelo)
    state["complete"] = True
    save(checkpoint, state)

    report = load_trial_report(trial)
    save(trial / "report.json", report)
    save(trial / "results.json", report)
    save_text(trial / "report.md", render_markdown(report))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="trial directory for runs, shared memory and reports")
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="number of passes over the exercises (default 5)")
    ap.add_argument(
        "--exercises",
        default=",".join(EXERCISES),
        help="comma-separated Aider exercises (default: aider12-v0)",
    )
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per run (default 600)")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="proxy step budget per run (default 30)")
    ap.add_argument("--thinking", default=DEFAULT_THINKING, help="proxy thinking mode (default off)")
    ap.add_argument("--print-schedule", action="store_true", help="print the planned schedule as JSON and exit")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exercises = tuple(e.strip() for e in args.exercises.split(",") if e.strip())
    if args.print_schedule:
        print(json.dumps(schedule(passes=args.passes, exercises=exercises), indent=2))
        return 0
    report = run_d11(
        args.run_dir,
        passes=args.passes,
        exercises=exercises,
        timeout_s=args.timeout,
        max_steps=args.max_steps,
        thinking=args.thinking,
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "runs": len(report["runs"]),
                "d2_default_on": report["d2"]["admit_on_outcome_default_on"],
                "exports": report["exports"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
