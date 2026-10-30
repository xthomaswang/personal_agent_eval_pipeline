"""E3: an A/B of five localmelo flag sets (arms) for prompt-cache reuse, on twelve Aider exercises.

    cd eval
    uv run python agent_ability/e3/run.py --run-dir <dir> --dry-run   # schedule and checks, no model
    uv run python agent_ability/e3/run.py --run-dir <dir>             # 120 runs; run it again to resume
    uv run python agent_ability/e3/report.py <dir>                    # report.md and report.json

The order is task-major: for each (task, repeat) all five arms run back to back, and the arm order
is rotated by (task index + repeat), so no arm is always first. Every run goes through the smoke
runner's run_one (so bench.harness, the recorder proxy and the Aider check) with the capability
leg's budget. The operator holds the GPU lock (holder: agent-benchmark); the driver never takes or
releases it. It stops cleanly before the next run when <dir>/STOP exists, when the lock is gone, or
on battery below 30%.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

AGENT_ABILITY = Path(__file__).resolve().parent.parent
EVAL_ROOT = AGENT_ABILITY.parent
for _root in (EVAL_ROOT, AGENT_ABILITY):  # bench (for bench.harness) and smoke
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from smoke import adapters, data, paths, runner, tasks  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "vendor" / "localmelo"
LOCALMELO_SHA = "ce4a3a8"
GPU_LOCK = Path.home() / "Developer" / "mlsys" / "experiments" / ".gpu-lock"
LOW_BATTERY_PCT = 30

# The extra localmelo flags of each arm; everything else is the bench default. --output-format json makes
# localmelo print its RunResult line, which has attempts and n_aux (it changes nothing else).
COMMON_ARGS = ("--output-format", "json")
ARMS: dict[str, tuple[str, ...]] = {
    "S0": (),
    "C1": ("--window-eviction", "chunk"),
    "B1": ("--window-eviction", "boundary"),
    "B2": ("--window-eviction", "boundary", "--fixed-prefix", "on"),
    "S2": ("--fixed-prefix", "on"),
    # E3b, added after E3: chunked eviction with the v2 fixed prefix
    "C2": ("--window-eviction", "chunk", "--fixed-prefix", "on"),
    # C2 + the reflect request keeps the plan's tools and asks for JSON only (needs a build that has the flag)
    "C3": ("--window-eviction", "chunk", "--fixed-prefix", "on", "--reflect-tool-guard", "on"),
    # E4, on a build whose defaults are C3 (S0 then means "the defaults"): the v3 cache layout, alone and with a
    # 64-message window
    "V3": ("--cache-layout", "v3"),
    "V64": ("--cache-layout", "v3", "--window-messages", "64"),
}
DEFAULT_ARMS = ("S0", "C1", "B1", "B2", "S2")   # E3's five
EXERCISES = (
    "sgf-parsing",
    "book-store",
    "zebra-puzzle",
    "pov",
    "scale-generator",
    "connect",
    "transpose",
    "bowling",
    "paasio",
    "wordy",
    "zipper",
    "react",
)
REPEATS = 2
# The capability leg's budget per run
BUDGET = {"max_steps": 30, "max_turns": 40, "timeout_s": 3600.0, "thinking": "off", "log_bodies": True}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def rotated(arms: Sequence[str], offset: int) -> list[str]:
    k = offset % len(arms)
    return [*arms[k:], *arms[:k]]


def schedule(
    exercises: Sequence[str] = EXERCISES, repeats: int = REPEATS, arms: Sequence[str] = DEFAULT_ARMS
) -> list[dict[str, Any]]:
    """Task-major runs. The arms of (task index i, repeat r) start at arm number (i + r) mod len(arms),
    with i counted from 0 and r from 1."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not exercises or not arms:
        raise ValueError("exercises and arms must not be empty")
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm {', '.join(unknown)}; choose from {', '.join(ARMS)}")
    planned: list[dict[str, Any]] = []
    for task_index, exercise in enumerate(exercises):
        for repeat in range(1, repeats + 1):
            for position, arm in enumerate(rotated(list(arms), task_index + repeat), 1):
                index = len(planned) + 1
                planned.append(
                    {
                        "index": index,
                        "exercise": exercise,
                        "task": f"aider/{exercise}",
                        "task_index": task_index,
                        "repeat": repeat,
                        "arm": arm,
                        "position": position,
                        "run_id": f"{index:03d}__{exercise}__r{repeat}__{arm}",
                    }
                )
    return planned


def arm_args(arm: str) -> list[str]:
    return [*COMMON_ARGS, *ARMS[arm]]


def _out(cmd: Sequence[str], timeout: float = 30.0) -> str:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def localmelo_sha(localmelo: Path) -> str:
    return _out(["git", "-C", str(localmelo), "rev-parse", "HEAD"])


def system_snapshot(localmelo: Path) -> dict[str, Any]:
    """The macOS and Xcode builds, the localmelo build and the model server (it records the oMLX commit)."""
    from bench import manifest

    return {
        "timestamp": time.time(),
        "sw_vers": _out(["sw_vers"]),
        "xcodebuild_version": _out(["xcodebuild", "-version"]),
        "localmelo_sha": localmelo_sha(localmelo),
        "model_server": manifest._model_server(),
        "pmset_batt": _out(["pmset", "-g", "batt"]),
    }


def identity(localmelo: Path, exercises: Sequence[str], repeats: int, arms: Sequence[str]) -> dict[str, Any]:
    from bench import manifest

    return {
        "protocol": "e3-v1",
        "arms": {arm: arm_args(arm) for arm in arms},
        "exercises": list(exercises),
        "repeats": repeats,
        "budget": BUDGET,
        "localmelo_sha": localmelo_sha(localmelo),
        "polyglot_sha": paths.POLYGLOT_SHA,
        "model": manifest.BENCH_MODEL,
    }


# --- guards ---


def lock_problem() -> str | None:
    """The operator's GPU lock must exist with a holder line `holder: agent-benchmark`."""
    try:
        lines = GPU_LOCK.read_text().splitlines()
    except FileNotFoundError:
        return f"{GPU_LOCK} is absent; the operator takes it first (personal-agent-bench/scripts/gpu-lock.sh acquire)"
    if not any(line.startswith(paths.GPU_LOCK_HOLDER) for line in lines):
        holder = next((line for line in lines if line.startswith("holder:")), "no holder line")
        return f"{GPU_LOCK} is held by another session ({holder})"
    return None


def battery_problem(low_pct: int = LOW_BATTERY_PCT) -> str | None:
    text = _out(["pmset", "-g", "batt"])
    found = re.search(r"(\d+)%", text)
    if "'Battery Power'" in text and found and int(found.group(1)) < low_pct:
        return f"on battery at {found.group(1)}% (below {low_pct}%)"
    return None


def stop_reason(out: Path) -> str | None:
    """Why the driver must not start the next run, or None."""
    if (out / "STOP").exists():
        return "STOP file found"
    return lock_problem() or battery_problem()


def preflight(
    settings: runner.Settings, harness: ModuleType, localmelo: Path, expect_sha: str = LOCALMELO_SHA
) -> list[str]:
    """Everything that would make the study invalid, as messages."""
    problems: list[str] = []
    sha = localmelo_sha(localmelo)
    if not sha.startswith(expect_sha):
        problems.append(f"localmelo at {localmelo} is {sha or 'not a git checkout'}, not {expect_sha}")
    elif _out(["git", "-C", str(localmelo), "status", "--porcelain"]):
        problems.append(f"localmelo at {localmelo} has uncommitted changes")
    known = set(data.exercise_names())
    problems += [f"exercise {t.name} is not in the pinned Aider data" for t in settings.tasks if t.name not in known]
    saved = paths.GPU_LOCK
    paths.GPU_LOCK = GPU_LOCK  # runner.lock_problem reads this
    try:
        problems += runner.preflight(settings, harness)
    finally:
        paths.GPU_LOCK = saved
    return problems


def _settings(exercises: Sequence[str], arm: str | None = None) -> runner.Settings:
    return runner.Settings(
        harnesses=["localmelo"],
        tasks=[tasks.aider_task(e) for e in exercises],
        warmup=False,
        timeout_s=BUDGET["timeout_s"],
        max_steps=BUDGET["max_steps"],
        max_turns=BUDGET["max_turns"],
        thinking=BUDGET["thinking"],
        log_bodies=BUDGET["log_bodies"],
        localmelo_args=arm_args(arm) if arm else [],
    )


# --- one run ---


def _result_line(summary: dict[str, Any] | None) -> dict[str, Any]:
    return ((summary or {}).get("harness_report") or {}).get("run_record") or {}


def execute(item: dict[str, Any], out: Path, harness: ModuleType, env_file: Path) -> dict[str, Any]:
    """One run through the smoke runner's run_one; its record, plus what the report needs."""
    task = tasks.aider_task(item["exercise"])
    settings = _settings([item["exercise"]], item["arm"])
    run_root = out / "runs" / item["run_id"]
    record = runner.run_one(harness, "localmelo", task, settings, run_root, out / "ws", env_file)
    line = record["harness_report"].get("run_record") or {}
    done = {
        **item,
        "state": "finished",
        "verdict": record["verdict"],
        "passed": record["verdict"] == "pass",
        "wall_s": record["wall_s"],
        "exit_code": record["exit_code"],
        "timed_out": record["timed_out"],
        "n_llm": record["n_llm"],
        "n_llm_aux": record["n_llm_aux"],
        "n_llm_err": record["n_llm_err"],
        "attempts": line.get("attempts"),
        "n_aux": line.get("n_aux"),
        "task_status": record["harness_report"].get("task_status") or line.get("status"),
        "check": record["check"],
        "prompt_sha256": record["prompt_sha256"],
        "run_dir": str((run_root / record["run_dir"]).relative_to(out)),
        "error": record["error"],
    }
    save(run_root / "record.json", done)
    return done


def salvage_interrupted(item: dict[str, Any], out: Path) -> dict[str, Any]:
    """A run that was started and left no record.json: recorded as interrupted, never replayed."""
    run_dir = out / "runs" / item["run_id"] / "localmelo" / tasks.aider_task(item["exercise"]).slug
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    line = _result_line(summary)
    return {
        **{k: v for k, v in item.items() if k != "state"},
        "state": "interrupted",
        "verdict": "interrupted",
        "passed": False,
        "wall_s": (summary or {}).get("wall_s"),
        "n_llm": ((summary or {}).get("steps") or {}).get("n_llm"),
        "attempts": line.get("attempts"),
        "n_aux": line.get("n_aux"),
        "run_dir": str(run_dir.relative_to(out)),
        "error": "interrupted",
    }


# --- the study ---


def _log(path: Path, text: str) -> None:
    with path.open("a") as log:
        log.write(f"{dt.datetime.now().astimezone().replace(microsecond=0).isoformat()} {text}\n")


def load_state(out: Path, config: dict[str, Any], planned: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = out / "e3.json"
    digest = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config or state.get("schedule_sha256") != digest:
            raise ValueError("E3 config, schedule or localmelo SHA changed; use a fresh directory")
        return state
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no e3.json checkpoint; use a fresh directory")
    state = {"config": config, "schedule_sha256": digest, "runs": {}, "complete": False}
    save(out / "schedule.json", planned)
    save(checkpoint, state)
    return state


def run_e3(
    out: Path,
    *,
    exercises: Sequence[str] = EXERCISES,
    repeats: int = REPEATS,
    arms: Sequence[str] = DEFAULT_ARMS,
    localmelo: Path = VENDOR,
    harness: ModuleType | None = None,
    expect_sha: str = LOCALMELO_SHA,
) -> int:
    """0 when every run is finished or interrupted, 2 when refused to start, 3 when stopped early."""
    out = out.resolve()
    localmelo = localmelo.resolve()
    if harness is None:
        from bench import harness as harness_module

        harness = harness_module
    harness.manifest.LOCALMELO_REPO = localmelo
    planned = schedule(exercises, repeats, arms)
    problems = preflight(_settings(exercises), harness, localmelo, expect_sha)
    if problems:
        print("Preflight failed:", *problems, sep="\n  - ", file=sys.stderr)
        return 2
    state = load_state(out, identity(localmelo, exercises, repeats, arms), planned)
    log_path = out / "e3.log"
    env_path = out / "environment.json"
    env_doc = json.loads(env_path.read_text()) if env_path.exists() else {}
    env_doc.setdefault("start", system_snapshot(localmelo))
    save(env_path, env_doc)
    env_file = adapters.write_env_file(out / "agent-env.sh")
    checkpoint = out / "e3.json"
    stopped = False

    with runner.workspace_lock(out / "ws"):
        for item in planned:
            run_id = item["run_id"]
            existing = state["runs"].get(run_id)
            if existing and existing.get("state") in ("finished", "interrupted"):
                continue
            if existing and existing.get("state") == "started":
                record_file = out / "runs" / run_id / "record.json"
                state["runs"][run_id] = (
                    json.loads(record_file.read_text()) if record_file.exists() else salvage_interrupted(existing, out)
                )
                save(checkpoint, state)
                _log(log_path, f"{run_id}: found started, recorded as {state['runs'][run_id]['state']}")
                continue
            reason = stop_reason(out)
            if reason:
                _log(log_path, f"== {reason} before {run_id}; stopping")
                stopped = True
                break
            state["runs"][run_id] = {**item, "state": "started"}
            save(checkpoint, state)
            record = execute(item, out, harness, env_file)
            state["runs"][run_id] = record
            save(checkpoint, state)
            _log(
                log_path,
                f"[{item['index']}/{len(planned)}] {run_id}: {record['verdict']} "
                f"n_llm={record['n_llm']} wall_s={record['wall_s']}",
            )

    ws = out / "ws"
    if ws.exists() and not any(ws.iterdir()):
        ws.rmdir()
    env_doc["end"] = system_snapshot(localmelo)
    save(env_path, env_doc)
    state["complete"] = all(
        state["runs"].get(i["run_id"], {}).get("state") in ("finished", "interrupted") for i in planned
    )
    save(checkpoint, state)
    done = sum(1 for r in state["runs"].values() if r.get("state") in ("finished", "interrupted"))
    print(f"{done}/{len(planned)} runs recorded in {out}" + (" (stopped early; see e3.log)" if stopped else ""))
    return 3 if stopped else 0


def print_plan(
    out: Path,
    planned: list[dict[str, Any]],
    exercises: Sequence[str],
    repeats: int,
    arms: Sequence[str],
    localmelo: Path,
    problems: list[str],
    environment: dict[str, Any],
) -> None:
    budget = BUDGET
    print(
        f"E3 plan: {len(planned)} runs = {len(exercises)} tasks x {repeats} repeats x {len(arms)} arms; "
        f"{budget['max_steps']} proxy steps, {budget['max_turns']} turns, {budget['timeout_s']:g} s per run, "
        f"thinking {budget['thinking']}, bodies logged"
    )
    print(f"Output: {out} (e3.json, schedule.json, environment.json, runs/)")
    for arm in arms:
        print(f"Arm {arm}: localmelo {' '.join(arm_args(arm))}")
    for item in planned:
        print(f"{item['index']:3d}. {item['exercise']:16s} r{item['repeat']} {item['arm']}")
    state_path = out / "e3.json"
    if state_path.exists():
        runs = json.loads(state_path.read_text()).get("runs", {})
        finished = sum(1 for r in runs.values() if r.get("state") in ("finished", "interrupted"))
        print(f"Checkpoint: {finished}/{len(planned)} runs already recorded; a resume skips them")
    print(
        f"Environment: {(environment.get('sw_vers') or '').replace(chr(10), ' ')} | "
        f"{(environment.get('xcodebuild_version') or '').replace(chr(10), ' ')} | "
        f"localmelo {environment.get('localmelo_sha', '')[:7]} at {localmelo} | "
        f"oMLX {((environment.get('model_server') or {}).get('repo') or {}).get('commit')}"
    )
    print("Preflight: " + ("ok" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"  - {problem}")
    battery = battery_problem()
    print(f"Battery: {battery or 'ok'}; STOP file: {'present' if (out / 'STOP').exists() else 'absent'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="output directory; a run there resumes from e3.json")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and the checks; run no model")
    ap.add_argument("--repeats", type=int, default=REPEATS, help=f"repeats per task (default {REPEATS})")
    ap.add_argument("--tasks", default=",".join(EXERCISES), help="comma-separated Aider exercises (default: the 12)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated arms (default: E3's five)")
    ap.add_argument("--localmelo", type=Path, default=VENDOR, help="the pinned localmelo clone")
    ap.add_argument("--localmelo-sha", default=LOCALMELO_SHA, help=f"the commit it must be at (default {LOCALMELO_SHA})")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, harness: ModuleType | None = None) -> int:
    args = parse_args(argv)
    exercises = tuple(e.strip() for e in args.tasks.split(",") if e.strip())
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    try:
        planned = schedule(exercises, args.repeats, arms)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        if harness is None:
            from bench import harness as harness_module

            harness = harness_module
        localmelo = args.localmelo.resolve()
        harness.manifest.LOCALMELO_REPO = localmelo
        problems = preflight(_settings(exercises), harness, localmelo, args.localmelo_sha)
        print_plan(
            args.run_dir.resolve(), planned, exercises, args.repeats, arms, localmelo, problems,
            system_snapshot(localmelo),
        )
        return 2 if problems else 0
    try:
        return run_e3(
            args.run_dir, exercises=exercises, repeats=args.repeats, arms=arms, localmelo=args.localmelo,
            harness=harness, expect_sha=args.localmelo_sha,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
