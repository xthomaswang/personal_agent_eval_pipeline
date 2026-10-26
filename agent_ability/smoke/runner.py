"""Run the smoke suite through personal-agent-bench's bench.harness.

Order is harness-major: for each harness one unscored warm-up, then every task, in --repeat passes
over the task list (pass 1 over all tasks, then pass 2, ...). Each run gets a fresh workspace at the
task's fixed path under ws/, so the prompt (which names that path) is the same bytes for every
harness and pass; after the check the workspace moves into the run directory.

Verdicts:
- timeout: bench.harness killed the harness at the wall-clock budget (even if the check passes);
- harness_error: the run raised, or no model call succeeded and the check failed;
- pass / fail: the check's result otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from smoke import adapters, checks, data, paths, report, tasks, watch
from smoke.tasks import Task

HARNESSES = ("claude_code", "hermes", "localmelo")
ALIASES = {"claude-code": "claude_code", "cc": "claude_code", "melo": "localmelo"}
WATCH_NOTE = "--watch-solution watches Aider tasks only; the MCPMark tasks run unwatched"


@dataclass
class Settings:
    harnesses: list[str]
    tasks: list[Task]
    mcp_arm: bool = False
    warmup: bool = True
    timeout_s: float = 600
    max_steps: int = 30
    max_turns: int = 40
    thinking: str = "off"
    random_seed: bool = False
    one_reasoning_field: bool = False
    log_bodies: bool = False
    localmelo_args: list[str] = field(default_factory=list)
    repeat: int = 1
    watch_solution: bool = False
    not_runnable: list[str] = field(default_factory=list)

    def runs(self) -> list[tuple[str, Task, int]]:
        """(harness, task, pass number) in run order; the warm-up runs once, as pass 1."""
        planned = []
        for harness in self.harnesses:
            if self.warmup:
                planned.append((harness, tasks.WARMUP, 1))
            planned += [(harness, task, n) for n in range(1, self.repeat + 1) for task in self.tasks]
        return planned

    def run_name(self, task: Task, n: int) -> str:
        """The run directory's name: the task slug, plus __r<n> for scored tasks when there are passes."""
        return f"{task.slug}__r{n}" if self.repeat > 1 and task.scored else task.slug


def _now() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verdict(summary: dict[str, Any] | None, check: dict[str, Any], error: str | None) -> str:
    if error is not None or summary is None:
        return "harness_error"
    if summary.get("timed_out"):
        return "timeout"
    if check.get("passed"):
        return "pass"
    if not (summary.get("steps") or {}).get("n_llm"):
        return "harness_error"
    return "fail"


def _binaries(harness: ModuleType) -> dict[str, Path]:
    return {
        "claude_code": Path(harness.manifest.CLAUDE_BIN),
        "hermes": Path(harness.HERMES_BIN),
        "localmelo": Path(harness.manifest.LOCALMELO_REPO) / ".venv" / "bin" / "melo",
    }


def lock_problem() -> str | None:
    """The GPU lock must be held by this session (personal-agent-bench/scripts/gpu-lock.sh acquire)."""
    try:
        lines = paths.GPU_LOCK.read_text().splitlines()
    except FileNotFoundError:
        return f"{paths.GPU_LOCK} is absent; take it first with personal-agent-bench/scripts/gpu-lock.sh acquire"
    if not any(line.startswith(paths.GPU_LOCK_HOLDER) for line in lines):
        holder = next((line for line in lines if line.startswith("holder:")), "no holder line")
        return f"{paths.GPU_LOCK} is held by another session ({holder})"
    return None


def preflight(settings: Settings, harness: ModuleType) -> list[str]:
    """Everything that would make the run invalid, as messages."""
    categories = sorted({t.category for t in settings.tasks if t.kind == "mcpmark"})
    problems = data.integrity_problems(categories)
    pytest_ok = paths.SMOKE_PYTHON.exists() and (
        subprocess.run([str(paths.SMOKE_PYTHON), "-I", "-c", "import pytest"], capture_output=True).returncode == 0
    )
    if not pytest_ok:
        problems.append(f"{paths.SMOKE_PYTHON} cannot import pytest; run `uv sync` in {paths.SMOKE_ROOT}")
    binaries = _binaries(harness)
    for name in settings.harnesses:
        if name in binaries and not binaries[name].exists():
            problems.append(f"{name}: {binaries[name]} is missing")
    lock = lock_problem()
    if lock:
        problems.append(lock)
    url = f"{harness.UPSTREAM}/v1/models"
    request = urllib.request.Request(url)
    token_cmd = os.environ.get(harness.UPSTREAM_TOKEN_CMD_ENV)
    if token_cmd:  # a server that needs a bearer token, which the proxy fetches the same way
        from bench.proxy.config import upstream_token

        try:
            request.add_header("Authorization", f"Bearer {upstream_token(token_cmd)}")
        except ValueError as exc:
            problems.append(str(exc))
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status != 200:
                problems.append(f"the model server answered {response.status} at {url}")
    except OSError as exc:
        problems.append(f"the model server is not reachable at {url} ({exc})")
    if settings.mcp_arm:
        problems += adapters.mcp_problems()
    return problems


@contextlib.contextmanager
def workspace_lock(ws_root: Path) -> Iterator[None]:
    ws_root.mkdir(parents=True, exist_ok=True)
    lock = ws_root / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another run is using the workspaces (delete it if it is stale)") from None
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_one(
    harness: ModuleType,
    name: str,
    task: Task,
    settings: Settings,
    run_root: Path,
    ws_root: Path,
    env_file: Path,
    n: int = 1,
) -> dict[str, Any]:
    ws = ws_root / task.slug
    tasks.build_workspace(task, ws)
    text = tasks.prompt(task, ws, mcp_arm=settings.mcp_arm)
    run_dir = run_root / name / settings.run_name(task, n)
    builders = adapters.overrides(harness, ws, mcp=settings.mcp_arm, env_file=env_file)
    watcher = watch.SolutionWatcher(task, ws) if settings.watch_solution and task.kind == "aider" else None
    summary: dict[str, Any] | None = None
    error: str | None = None
    try:
        with builders, watcher or contextlib.nullcontext():
            summary = harness.run(
                name,
                text,
                ws,
                run_dir,
                timeout_s=settings.timeout_s,
                max_steps=settings.max_steps,
                max_turns=settings.max_turns,
                thinking=settings.thinking,
                random_seed=settings.random_seed,
                one_reasoning_field=settings.one_reasoning_field,
                log_bodies=settings.log_bodies,
                harness_args=settings.localmelo_args if name == "localmelo" else (),
                extra_env=adapters.agent_env(env_file),
            )
    except Exception as exc:  # the run did not complete; record it and go on with the next one
        error = f"{type(exc).__name__}: {exc}"
        run_dir.mkdir(parents=True, exist_ok=True)
    result = checks.check(task, ws, run_dir)
    shutil.move(ws, run_dir / "workspace")
    watched = watcher.finish(run_dir) if watcher else None
    steps = (summary or {}).get("steps") or {}
    if summary is not None and not steps:  # the proxy recorded no call at all
        steps = {"n_llm": 0, "n_tool": 0, "n_llm_aux": 0, "n_llm_err": 0}
    harness_report = (summary or {}).get("harness_report") or {}
    return {
        "harness": name,
        "task": task.id,
        "repeat": n,
        "kind": task.kind,
        "scored": task.scored,
        "verdict": verdict(summary, result, error),
        "wall_s": (summary or {}).get("wall_s"),
        "exit_code": (summary or {}).get("exit_code"),
        "timed_out": (summary or {}).get("timed_out"),
        "n_llm": steps.get("n_llm"),
        "n_tool": steps.get("n_tool"),
        "n_llm_aux": steps.get("n_llm_aux"),
        "n_llm_err": steps.get("n_llm_err"),
        "native_steps": report.native_steps(name, harness_report),
        "harness_report": harness_report,
        "check": result,
        "prompt_sha256": _sha256(text),
        "run_dir": str(run_dir.relative_to(run_root)),
        "error": error,
        "watch": watched,
    }


def _smoke_commit() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        proc = subprocess.run(["git", "-C", str(paths.SMOKE_ROOT), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


@functools.cache
def _system() -> dict[str, str | None]:
    """The macOS and Xcode builds, recorded with every result: this machine runs macOS betas."""

    def out(*cmd: str) -> str | None:
        try:
            proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    return {
        "macos": out("sw_vers", "-productVersion"),
        "macos_build": out("sw_vers", "-buildVersion"),
        "xcodebuild_version": out("xcodebuild", "-version"),
    }


def header(settings: Settings, run_root: Path, started_at: str, finished: bool) -> dict[str, Any]:
    manifest_path = paths.MCPMARK_ENVS / "manifest.json"
    return {
        "schema_version": 1,
        "suite": "harness-smoke",
        "arm": "mcp" if settings.mcp_arm else "builtin",
        "run_root": str(run_root),
        "started_at": started_at,
        "updated_at": _now(),
        "finished": finished,
        "settings": {
            **{k: v for k, v in asdict(settings).items() if k != "tasks"},
            "tasks": [t.id for t in settings.tasks],
        },
        "not_runnable": settings.not_runnable,
        "data": {
            "polyglot_sha": paths.POLYGLOT_SHA,
            "mcpmark_sha": paths.MCPMARK_SHA,
            "zip_manifest_sha256": data.sha256_file(manifest_path) if manifest_path.exists() else "",
        },
        "smoke": _smoke_commit(),
        "system": _system(),
    }


def run_suite(harness: ModuleType, settings: Settings, run_root: Path, ws_root: Path) -> list[dict[str, Any]]:
    run_root.mkdir(parents=True)
    env_file = adapters.write_env_file(run_root / "agent-env.sh")
    started_at = _now()
    records: list[dict[str, Any]] = []
    report.write(run_root, header(settings, run_root, started_at, finished=False), records)
    if settings.watch_solution and any(t.kind != "aider" for t in settings.tasks):
        print(WATCH_NOTE, flush=True)
    with workspace_lock(ws_root):
        planned = settings.runs()
        for index, (name, task, n) in enumerate(planned, 1):
            label = f"{task.id} #{n}" if settings.repeat > 1 and task.scored else task.id
            print(f"[{index}/{len(planned)}] {name} {label} ...", flush=True)
            record = run_one(harness, name, task, settings, run_root, ws_root, env_file, n)
            records.append(record)
            print(f"    {record['verdict']} wall_s={record['wall_s']} n_llm={record['n_llm']}", flush=True)
            report.write(run_root, header(settings, run_root, started_at, finished=False), records)
    report.write(run_root, header(settings, run_root, started_at, finished=True), records)
    return records


def _harness_list(values: list[str] | None, mcp: bool) -> tuple[list[str], list[str]]:
    default = list(adapters.MCP_HARNESSES if mcp else HARNESSES)
    if not values:
        return default, (["localmelo (no MCP client)"] if mcp else [])
    names = [ALIASES.get(n.strip(), n.strip()) for v in values for n in v.split(",") if n.strip()]
    unknown = [n for n in names if n not in HARNESSES]
    if unknown:
        raise SystemExit(f"unknown harness {', '.join(unknown)}; choose from {', '.join(HARNESSES)}")
    if mcp and "localmelo" in names:
        raise SystemExit("localmelo cannot run the MCP arm: it has no MCP client yet")
    return list(dict.fromkeys(names)), []


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the harness smoke suite (10 MCPMark filesystem-easy tasks + 5 Aider Python exercises).",
    )
    parser.add_argument(
        "--harness",
        action="append",
        help="claude_code, hermes or localmelo; repeat or comma-separate (default: all three; MCP arm: claude_code,hermes)",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="fnmatch pattern on task ids, e.g. 'aider/*' or 'mcpmark/papers/*'; repeatable (default: the 15-task suite)",
    )
    parser.add_argument("--extended", action="store_true", help="add the 7 extended Aider exercises to the default suite")
    parser.add_argument(
        "--mcp-arm",
        action="store_true",
        help="MCP-only arm: Claude Code and Hermes on the MCPMark tasks, with only the filesystem MCP server as tools",
    )
    parser.add_argument("--no-warmup", action="store_true", help="skip the unscored warm-up task")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the preflight result; run nothing")
    parser.add_argument("--show-prompts", action="store_true", help="with --dry-run, also print every prompt")
    parser.add_argument("--list-tasks", action="store_true", help="list every selectable task id and exit")
    parser.add_argument("--timeout", type=float, default=600, help="wall-clock seconds per task (default 600)")
    parser.add_argument("--max-steps", type=int, default=30, help="proxy step budget per task (default 30)")
    parser.add_argument("--max-turns", type=int, default=40, help="the harness's own turn limit (default 40)")
    parser.add_argument("--thinking", default="off", help="proxy thinking mode (default off)")
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="the proxy sets a fresh random seed on every request (for servers whose default seed is fixed, e.g. rec)",
    )
    parser.add_argument(
        "--one-reasoning-field",
        action="store_true",
        help="the proxy keeps one of reasoning_content and reasoning on replayed assistant messages (rec rejects both)",
    )
    parser.add_argument(
        "--log-bodies",
        action="store_true",
        help="the proxy also stores full request and response bodies in each run's bodies/ (synthetic tasks only)",
    )
    parser.add_argument(
        "--localmelo-args",
        default="",
        metavar="FLAGS",
        help='extra localmelo flags, shell-quoted, before the prompt: --localmelo-args="--tool-exposure all"',
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="passes over the selected tasks, one after another (default 1)"
    )
    parser.add_argument(
        "--watch-solution",
        action="store_true",
        help="Aider tasks: keep every version of the solution while the harness runs, check each afterwards, "
        "and count the model requests after the first passing one (summary.json 'watch')",
    )
    parser.add_argument("--runs-dir", type=Path, default=paths.RUNS_ROOT, help="where runs/<timestamp> goes")
    parser.add_argument("--skip-preflight", action="store_true", help="debugging only: run even if preflight fails")
    return parser.parse_args(argv)


def settings_from(args: argparse.Namespace) -> Settings:
    harnesses, not_runnable = _harness_list(args.harness, args.mcp_arm)
    if args.task:
        selected = tasks.select(args.task)
        if args.extended:
            selected += [t for t in map(tasks.aider_task, tasks.AIDER_EXTENDED) if t not in selected]
    else:
        selected = tasks.default_suite(extended=args.extended, mcp_arm=args.mcp_arm)
    if args.mcp_arm and any(t.kind != "mcpmark" for t in selected):
        raise SystemExit("the MCP arm runs MCPMark tasks only: without a shell there is no way to run the unit tests")
    localmelo_args = shlex.split(args.localmelo_args)
    if localmelo_args and "localmelo" not in harnesses:
        raise SystemExit("--localmelo-args needs localmelo among the harnesses")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    return Settings(
        harnesses=harnesses,
        tasks=selected,
        mcp_arm=args.mcp_arm,
        warmup=not args.no_warmup,
        timeout_s=args.timeout,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        thinking=args.thinking,
        random_seed=args.random_seed,
        one_reasoning_field=args.one_reasoning_field,
        log_bodies=args.log_bodies,
        localmelo_args=localmelo_args,
        repeat=args.repeat,
        watch_solution=args.watch_solution,
        not_runnable=not_runnable,
    )


def print_plan(settings: Settings, run_root: Path, ws_root: Path, problems: list[str], show_prompts: bool) -> None:
    arm = "MCP-only (filesystem MCP server)" if settings.mcp_arm else "built-in tools"
    seed = "; a random seed per request" if settings.random_seed else ""
    seed += "; one reasoning field per replayed message" if settings.one_reasoning_field else ""
    seed += "; request and response bodies logged" if settings.log_bodies else ""
    seed += f"; localmelo flags {shlex.join(settings.localmelo_args)}" if settings.localmelo_args else ""
    seed += f"; {settings.repeat} passes over the tasks" if settings.repeat > 1 else ""
    seed += "; Aider solutions watched" if settings.watch_solution else ""
    print(f"Plan: {arm}; thinking {settings.thinking}{seed}; {settings.max_steps} proxy steps, "
          f"{settings.max_turns} harness turns, {settings.timeout_s:g} s per task")
    print(f"Results: {run_root}/results.json and results.md")
    if settings.not_runnable:
        print(f"Not runnable in this arm: {', '.join(settings.not_runnable)}")
    if settings.watch_solution and any(t.kind != "aider" for t in settings.tasks):
        print(WATCH_NOTE)
    planned = settings.runs()
    texts: dict[str, str] = {}
    for index, (name, task, n) in enumerate(planned, 1):
        ws = ws_root / task.slug
        text = texts.setdefault(task.id, tasks.prompt(task, ws, mcp_arm=settings.mcp_arm))
        flag = "" if task.scored else "  (warm-up, unscored)"
        label = f"{task.id} #{n}" if settings.repeat > 1 and task.scored else task.id
        print(f"{index:3d}. {name:12s} {label:46s} prompt {_sha256(text)[:12]} ({len(text.encode())} B){flag}")
    scored = sum(1 for _, t, _ in planned if t.scored)
    print(f"{len(planned)} runs ({scored} scored) in {ws_root}")
    print("Preflight: " + ("ok" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"  - {problem}")
    if show_prompts:
        for task_id, text in texts.items():
            print(f"\n----- prompt for {task_id} -----\n{text}", end="")


def main(argv: list[str] | None = None, harness: ModuleType | None = None) -> int:
    args = parse_args(argv)
    if args.list_tasks:
        for task in tasks.known_tasks():
            suite = " (default suite)" if task in tasks.default_suite() else ""
            print(f"{task.id}{suite}")
        return 0
    settings = settings_from(args)
    if harness is None:
        from bench import harness as harness_module

        harness = harness_module
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-mcp" if settings.mcp_arm else "")
    run_root = args.runs_dir.resolve() / stamp
    ws_root = paths.WS_ROOT
    problems = [] if args.skip_preflight and not args.dry_run else preflight(settings, harness)
    if args.dry_run:
        print_plan(settings, run_root, ws_root, problems, args.show_prompts)
        return 0
    if problems:
        print("Preflight failed:", *problems, sep="\n  - ", file=sys.stderr)
        return 2
    records = run_suite(harness, settings, run_root, ws_root)
    print(f"\nResults: {run_root / 'results.md'}")
    print(json.dumps(report.totals(records), indent=2))
    return 0
