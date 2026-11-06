"""The forms suite's driver: localmelo fills in web forms on a local fixture site, as an A/B of its thinking.

    cd eval
    uv run python agent_ability/forms/run.py --run-dir <dir> --dry-run   # the plan and the checks; no model, no run
    uv run python agent_ability/forms/run.py --run-dir <dir>             # the study; run it again to resume
    uv run python agent_ability/forms/report.py <dir>                    # report.md and report.json again

Eight tasks (tasks.py) on a fixture site (fixture.py), times the arms (arms.py), times ``--repeats``. The order is
repeat-major: for each repeat and task the arms run back to back, rotated by (task index + repeat) as in web/run.py, so
no arm is always first. Runs are serial: wall time and the model's share of it are what is measured.

One run: a fresh site on a free port of 127.0.0.1; a run folder with the task's files; a headless Chrome on an empty
profile, open on the task's page; then bench.harness.run (the recorder proxy in between, ``--thinking passthrough``)
with the arm's config. The site's records decide the verdict (``tasks.check``). Chrome's command line is read back from
``ps`` and the study stops if it is not headless; it is kept in each record.

A run ends on the first of: the model's last reply; melo's own budget (``--max-calls``, one call under the proxy's) or
its ``-t``; or the wall cap (``--wall-cap-s``), when bench.harness kills it. ``end_reason`` says which, with melo's
status, or ``asked_person`` for a reply that is a question. Nothing ever waits for an answer (arms.py).

A run in which no model call succeeded says nothing about melo: it is dropped, run once more after a pause, and a second
one in a row stops the study.

For the night runner (a KIND=cmd job, in eval/, with NIGHT_URL, NIGHT_DEADLINE and NIGHT_DIR set): ``--base-url`` and
``--deadline`` default to the first two, and the report is copied to NIGHT_DIR. The study stops between runs on
``<dir>/STOP``, or when a run could pass the deadline, given its wall cap, and exits 3. It exits 0 when every run is
recorded and 2 when it refuses to start. A rerun resumes; a run that was cut off is dropped and run again, its folder
kept as ``<run_id>.dropped-<n>``. SIGTERM ends the run under way the same way, and exits 3.

Files in ``<dir>``: forms.json (the checkpoint), schedule.json, environment.json, forms.log, and per run
``runs/<run_id>/`` with record.json, tap.jsonl, mcp.json, site/ (submissions.jsonl, requests.jsonl), workspace/, chrome/
and localmelo/ (the bench run directory: calls.jsonl, stdout.log, mem/history.db and the run's HOME with config.toml).
The operator holds the GPU lock; the driver only reads it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

AGENT_ABILITY = Path(__file__).resolve().parent.parent
EVAL_ROOT = AGENT_ABILITY.parent
for _root in (EVAL_ROOT, AGENT_ABILITY):  # bench, memory; smoke, web, forms
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from bench.proxy.steps import is_generation, is_ok  # noqa: E402
from forms import arms, fixture, report, tasks  # noqa: E402
from smoke import runner  # noqa: E402
from web import arms as web_arms  # noqa: E402
from web import chrome  # noqa: E402
from web import run as web_run  # noqa: E402

PROTOCOL = "forms-v1"
REPEATS = 3
WALL_CAP_S = 900.0
MAX_CALLS = 50  # the proxy's budget of main model calls; melo stops one short of it
MAX_TURNS = 60  # bench.harness passes it on; localmelo's builder does not use it
THINKING = "passthrough"  # the proxy must leave the arms' thinking alone
STOPPED = 3
REFUSED = 2
RUNS_ROOT = AGENT_ABILITY / "runs" / "forms"
NO_MODEL_PAUSE_S = 30.0
VERDICTS = ("pass", "fail", "harness_error")
STATE_FILE = "forms.json"
HEADLESS_FLAG = "--headless=new"


class Terminated(Exception):
    """SIGTERM: the night is over. The run under way is dropped."""


@dataclass
class Context:
    out: Path
    harness: ModuleType
    tasks: Mapping[str, tasks.Task]
    arms: Mapping[str, arms.Arm]
    chrome_version: str | None
    model: str = arms.DEFAULT_MODEL
    embedding_model: str | None = arms.EMBEDDING_MODEL
    wall_cap_s: float = WALL_CAP_S
    max_calls: int = MAX_CALLS
    extra: tuple[str, ...] = ()
    log_bodies: bool = False


# --- the schedule ---


def schedule(
    chosen: Sequence[tasks.Task], repeats: int = REPEATS, arm_names: Sequence[str] = arms.DEFAULT_ARMS
) -> list[dict[str, Any]]:
    """Repeat-major runs. The arms of (task index i, repeat r) start at arm number (i + r) mod len(arms)."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not chosen or not arm_names:
        raise ValueError("tasks and arms must not be empty")
    planned: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for task_index, task in enumerate(chosen):
            for position, arm in enumerate(web_run.rotated(list(arm_names), task_index + repeat), 1):
                index = len(planned) + 1
                planned.append(
                    {
                        "index": index,
                        "task": task.id,
                        "repeat": repeat,
                        "harness": arms.HARNESS,
                        "arm": arm,
                        "position": position,
                        "run_id": f"{index:03d}__{task.id}__r{repeat}__{arm}",
                    }
                )
    return planned


def select_tasks(spec: str | None) -> list[tasks.Task]:
    """Task ids, comma-separated, or ``all``."""
    if not spec or spec.strip() == "all":
        return list(tasks.TASKS)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in tasks.BY_ID]
    if unknown:
        raise ValueError(f"unknown task {', '.join(unknown)}; choose from {', '.join(tasks.BY_ID)}")
    return [tasks.BY_ID[n] for n in dict.fromkeys(names)]


def select_arms(spec: str | None, known: Mapping[str, arms.Arm]) -> list[str]:
    if not spec:
        return list(arms.DEFAULT_ARMS)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown arm {', '.join(unknown)}; choose from {', '.join(known)}")
    return list(dict.fromkeys(names))


# --- guards ---


def stop_reason(out: Path, *, deadline: float | None, wall_cap_s: float, gpu_lock: bool = True) -> str | None:
    """Why the driver must not start the next run, or None. A run may last its wall cap, so one that could pass the
    deadline is not started."""
    if (out / "STOP").exists():
        return "STOP file found"
    if deadline is not None and time.time() + wall_cap_s > deadline:
        return f"the deadline {deadline:.0f} is less than a wall cap away"
    return (runner.lock_problem() if gpu_lock else None) or web_run.battery_problem()


def fixture_problems() -> list[str]:
    """Every task's start page must answer, and every picture must be there."""
    found = []
    site = fixture.Site()
    try:
        for task in tasks.TASKS:
            try:
                chrome.http_get(site.url(task.start))
            except OSError as exc:
                found.append(f"the fixture site does not serve {task.start} ({exc})")
    finally:
        site.stop()
    found += [
        f"tile {n} is missing"
        for n in range(1, fixture.TILE_COUNT + 1)
        if not (fixture.PAGES / "tiles" / f"tile-{n:02d}.svg").exists()
    ]
    return found


def preflight(
    harness: ModuleType,
    localmelo: Path,
    out: Path,
    *,
    expect_sha: str = arms.LOCALMELO_SHA,
    expected_chrome_version: str | None = None,
    gpu_lock: bool = True,
    check_model: bool = True,
) -> list[str]:
    """Everything that would make the study invalid, as messages. Run it inside ``bench_settings``. ``check_model``
    False (a dry run) leaves the model server alone."""
    problems = fixture_problems()
    melo = runner._binaries(harness)["localmelo"]
    if not melo.exists():
        problems.append(f"localmelo: {melo} is missing")
    problems += web_arms.localmelo_problems(localmelo, expect_sha)
    problems += web_arms.mcp_problems(("D",))
    if gpu_lock:
        problems += [p for p in (runner.lock_problem(),) if p]
    if check_model:
        problems += [p for p in (web_run.model_problem(harness),) if p]
    problems += chrome.preflight((chrome.BENCH_ROOT, RUNS_ROOT, out))
    try:
        chrome.check_version(chrome.chrome_version(), expected_chrome_version)
    except chrome.ChromeError as exc:
        problems.append(str(exc))
    return problems


# --- Chrome ---


@contextlib.contextmanager
def forms_chrome(chrome_dir: Path, start_url: str, *, expected_version: str | None = None) -> Iterator[chrome.Run]:
    """A headless Chrome on an empty profile of its own, open on ``start_url``. On exit its tabs and History are kept
    and the profile deleted (chrome.py's ``Run``). The command line is read back from the process: a Chrome that is not
    headless is stopped at once, and that stops the study."""
    version = chrome.chrome_version()
    chrome.check_version(version, expected_version)
    problems = chrome.preflight((chrome.BENCH_ROOT, RUNS_ROOT, chrome_dir))
    if problems:
        raise chrome.ChromeError("; ".join(problems))
    profile = chrome_dir / "profile"
    profile.mkdir(parents=True)
    chrome._write_marker(profile, role="forms", chrome_version=version)
    browser = chrome.launch(profile, version=version, headless=True, log_path=chrome_dir / "chrome.log")
    command = (chrome.command_of(browser.proc.pid) or "").replace(str(browser.profile), "<profile>")
    meta: dict[str, Any] = {
        "chrome_version": version,
        "headless": HEADLESS_FLAG in command.split(),
        "command": command,
        "port": browser.port,
        "pid": browser.proc.pid,
    }
    run = chrome.Run(chrome_dir, browser, version, {}, "empty", meta)
    try:
        if not meta["headless"]:
            raise chrome.ChromeError(f"the launched Chrome is not headless: {command}")
        run.start_tab = meta["start_tab"] = chrome.leave_start_tab(browser, start_url, settle_s=0.2)
        yield run
    finally:
        run.finish()


# --- what a run leaves behind ---


def reasoning_tokens(record: Mapping[str, Any]) -> int | None:
    """What the server said a reply spent on reasoning (``completion_tokens_details``), when it says."""
    details = (record.get("usage_raw") or {}).get("completion_tokens_details")
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) else None


def template_kwargs(record: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = (record.get("thinking_forwarded") or {}).get("chat_template_kwargs")
    return kwargs if isinstance(kwargs, dict) else {}


def model_metrics(calls: Path, arm: arms.Arm) -> dict[str, Any]:
    """Counts, time and thinking of the run's model calls, from the proxy's calls.jsonl.

    ``model_s`` sums the duration of every generation request, failed ones too; melo's embedding requests are apart.
    ``reasoning_tokens`` is None unless every successful call reported them; ``reasoning_chars`` is the length of the
    reasoning text in the responses. ``arm_applied`` is False when a request left the proxy without the arm's
    ``chat_template_kwargs``.
    """
    every = web_run._json_lines(calls)
    records = [r for r in every if is_generation(r)]
    embeddings = [r for r in every if str(r.get("path")).endswith("/embeddings")]
    ok = [r for r in records if is_ok(r)]
    durations = [float(r.get("duration_s") or 0.0) for r in records]
    tokens = [reasoning_tokens(r) for r in ok]
    wanted = arm.extra_body.get("chat_template_kwargs") or {}
    offered = {name for r in ok for name in r.get("request_tool_names") or []}
    return {
        "n_calls": len(ok),
        "n_main": sum(1 for r in ok if r.get("role") == "main"),
        "n_aux": sum(1 for r in ok if r.get("role") != "main"),
        "n_failed_calls": len(records) - len(ok),
        "model_s": round(sum(durations), 3),
        "longest_call_s": max(durations, default=None),
        "n_embedding_calls": len(embeddings),
        "embedding_s": round(sum(float(r.get("duration_s") or 0.0) for r in embeddings), 3),
        "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok),
        "output_tokens": sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in ok),
        "reasoning_tokens": sum(t or 0 for t in tokens) if ok and all(t is not None for t in tokens) else None,
        "reasoning_chars": sum(r.get("output_reasoning_chars") or 0 for r in ok),
        "longest_reasoning_chars": max((r.get("output_reasoning_chars") or 0 for r in ok), default=0),
        "thinking_off_calls": sum(1 for r in ok if template_kwargs(r).get("enable_thinking") is False),
        "arm_applied": all(template_kwargs(r).get(k) == v for r in records for k, v in wanted.items()),
        "foreign_tools": sorted(n for n in offered if not n.startswith(arms.TOOL_PREFIX)),
    }


def history_thinking_chars(db: Path) -> int | None:
    """The thinking melo kept (history.db), for a server that returns it inside the reply's text."""
    from bench import harness as bench_harness

    value = bench_harness._sqlite_value(db, "SELECT SUM(LENGTH(thinking)) FROM llm_calls")
    return int(value) if value is not None else None


def end_reason(
    summary: Mapping[str, Any] | None, error: str | None, status: str | None, reply: str, passed: bool
) -> str:
    """Why the run ended: ``harness_error``, ``wall_cap``, ``no_record`` (melo left no result), ``asked_person`` (it
    stopped on a question), or melo's own status."""
    if error is not None or summary is None:
        return "harness_error"
    if summary.get("timed_out"):
        return "wall_cap"
    if status is None:
        return "no_record"
    if status == "completed" and not passed and reply.rstrip().endswith("?"):
        return "asked_person"
    return status


def kill_leftovers(run_root: Path) -> list[int]:
    """End the melo of a cut-off run and what it started: bench.harness leaves it in a session of its own. Only a
    process with this run's mcp.json on its command line is touched."""
    marker = str(run_root / "mcp.json")
    killed = []
    for proc in chrome.process_table():
        if marker in proc.command and proc.pid != os.getpid():
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGTERM)
                killed.append(proc.pid)
    return killed


# --- one run ---


def execute(item: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """One run. Writes and returns its record."""
    task, arm = ctx.tasks[item["task"]], ctx.arms[item["arm"]]
    run_root = ctx.out / "runs" / item["run_id"]
    workspace = run_root / "workspace"
    agent_dir = run_root / arms.HARNESS
    chrome_dir = run_root / "chrome"
    tasks.prepare_workspace(task, workspace)
    text = task.text(workspace)
    summary: dict[str, Any] | None = None
    error: str | None = None
    meta: dict[str, Any] = {}
    site = fixture.Site(run_root / "site")
    try:
        with forms_chrome(chrome_dir, site.url(task.start), expected_version=ctx.chrome_version) as browser:
            meta = browser.meta
            try:
                with arms.overrides(
                    ctx.harness, arm, run_dir=run_root, browser_url=browser.browser_url, workspace=workspace,
                    site_url=site.base_url, password=task.password, model=ctx.model, embedding_model=ctx.embedding_model,
                    max_calls=ctx.max_calls, wall_cap_s=ctx.wall_cap_s, extra=ctx.extra,
                ) as plan:  # fmt: skip
                    summary = ctx.harness.run(
                        arms.HARNESS, text, workspace, agent_dir, timeout_s=ctx.wall_cap_s, max_steps=ctx.max_calls,
                        max_turns=MAX_TURNS, thinking=THINKING, log_bodies=ctx.log_bodies,
                        harness_args=plan.harness_args,
                    )  # fmt: skip
            except Terminated:
                kill_leftovers(run_root)
                raise
            except Exception as exc:  # noqa: BLE001 - the run did not complete; record it and go on with the next one
                error = f"{type(exc).__name__}: {exc}"
                agent_dir.mkdir(parents=True, exist_ok=True)
    except chrome.ChromeError as exc:
        raise web_run.StopStudy(f"Chrome: {exc}") from exc
    finally:
        site.stop()

    stdout = agent_dir / "stdout.log"
    reply = web_run.final_text(arms.HARNESS, stdout) if stdout.exists() else ""
    outcome = tasks.check(task, site.submissions, reply)
    model = model_metrics(agent_dir / "calls.jsonl", arm)
    tap = web_run.tap_metrics(run_root / "tap.jsonl")
    melo = ((summary or {}).get("harness_report") or {}).get("run_record") or {}
    status = melo.get("status")
    wall_s = (summary or {}).get("wall_s")
    refused_by_tap = tap["by_status"].get("refused", 0)
    refused_by_melo = max(
        (melo.get("n_tool") or 0) - tap["n_calls"], 0
    )  # handed to its executor, never reached the tap
    verdict = "harness_error" if error else "pass" if outcome["passed"] else "fail"
    record: dict[str, Any] = {
        **item,
        "state": "finished",
        "title": task.title,
        "verdict": verdict,
        "passed": verdict == "pass",
        "check": outcome,
        "end_reason": end_reason(summary, error, status, reply, outcome["passed"]),
        "melo_status": status,
        "melo_error": melo.get("error") or None,
        "melo_empty_reply": melo.get("empty_reply"),  # its fallback may retry a turn with thinking off, in any arm
        "exit_code": (summary or {}).get("exit_code"),
        "timed_out": (summary or {}).get("timed_out"),
        "error": error,
        "reply": reply[-2000:],
        # cost
        "wall_s": wall_s,
        "wall_cap_s": ctx.wall_cap_s,
        **model,
        "model_share": round(model["model_s"] / wall_s, 4) if wall_s else None,
        "melo_thinking_chars": history_thinking_chars(agent_dir / "mem" / "history.db"),
        "n_tool_melo": melo.get("n_tool"),
        "n_tool_tap": tap["n_calls"],
        "tools": tap["by_tool"],
        "n_images": tap["n_images"],
        "refused_calls": refused_by_tap + refused_by_melo,
        "refused_by_tap": refused_by_tap,
        "refused_by_melo": refused_by_melo,
        "tool_errors": sum(tap["by_status"].get(s, 0) for s in ("tool_error", "rpc_error", "internal_error")),
        # the site
        "submissions": dict(Counter(f"{s['kind']}:{s['status']}" for s in site.submissions)),
        "n_requests": len(site.requests),
        # Chrome: the run's own, and whether melo started another
        "headless": meta.get("headless"),
        "chrome_command": meta.get("command"),
        "melo_chrome_profile": (agent_dir / "localmelo-home" / ".cache" / "localmelo" / "browser" / "chrome").exists(),
        # provenance
        "t0": (summary or {}).get("started_at"),
        "finished_at": time.time(),
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "run_dir": str(run_root.relative_to(ctx.out)),
    }
    web_run.save(run_root / "record.json", record)
    return record


def drop_run(out: Path, run_id: str) -> None:
    """Keep what a cut-off run left, out of the way of its second try."""
    folder = out / "runs" / run_id
    if not folder.exists():
        return
    n = 1
    while (target := folder.with_name(f"{run_id}.dropped-{n}")).exists():
        n += 1
    folder.rename(target)


# --- the study ---


def task_digest(chosen: Sequence[tasks.Task]) -> str:
    """Changes when a task's text or its answer changes, so a resume cannot mix two sets."""
    text = json.dumps(
        [[t.id, t.start, t.prompt, t.kind, t.pick, dict(t.expected), t.answer] for t in chosen], sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()


def identity(
    chosen: Sequence[tasks.Task],
    repeats: int,
    arm_names: Sequence[str],
    known: Mapping[str, arms.Arm],
    *,
    model: str,
    embedding_model: str | None,
    wall_cap_s: float,
    max_calls: int,
    extra: Sequence[str],
    localmelo_sha: str,
) -> dict[str, Any]:
    """What a resume must match."""
    return {
        "protocol": PROTOCOL,
        "tasks": [t.id for t in chosen],
        "tasks_sha256": task_digest(chosen),
        "repeats": repeats,
        "arms": {name: known[name].identity() for name in arm_names},
        "wall_cap_s": wall_cap_s,
        "max_calls": max_calls,
        "thinking": THINKING,
        "extra": list(extra),
        "localmelo_sha": localmelo_sha,
        "chrome_devtools_mcp": web_arms.mcp_versions()["pinned"],
        "model": model,
        "embedding_model": embedding_model,
    }


def load_state(out: Path, config: dict[str, Any], planned: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = out / STATE_FILE
    digest = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config or state.get("schedule_sha256") != digest:
            raise ValueError("the study's config or schedule changed; use a fresh directory")
        return state
    if out.exists() and any(p.name != "STOP" for p in out.iterdir()):
        raise ValueError(f"{out} is not empty and has no {STATE_FILE} checkpoint; use a fresh directory")
    state = {"config": config, "schedule_sha256": digest, "chrome_version": None, "runs": {}, "complete": False}
    web_run.save(out / "schedule.json", planned)
    web_run.save(checkpoint, state)
    return state


def clear_stale_lock(ws: Path) -> None:
    """A lock left by a run that was killed outright would refuse every rerun."""
    lock = ws / ".lock"
    with contextlib.suppress(OSError, ValueError):
        os.kill(int(lock.read_text().strip()), 0)
        return
    lock.unlink(missing_ok=True)


@contextlib.contextmanager
def terminate_as_exception() -> Iterator[None]:
    def raise_it(*_: Any) -> None:
        raise Terminated("SIGTERM")

    saved = signal.signal(signal.SIGTERM, raise_it)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, saved)


def write_report(out: Path, night_dir: str | None) -> None:
    """The report so far; it must never change how the study ends."""
    try:
        report.write(out)
        if night_dir:
            shutil.copyfile(out / "report.md", Path(night_dir) / f"forms-{out.name}.md")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        web_run._log(out / "forms.log", f"the report failed: {type(exc).__name__}: {exc}")


def run_study(
    out: Path,
    *,
    chosen: Sequence[tasks.Task] | None = None,
    repeats: int = REPEATS,
    arm_names: Sequence[str] = arms.DEFAULT_ARMS,
    known_arms: Mapping[str, arms.Arm] = arms.ARMS,
    localmelo: Path = arms.LOCALMELO,
    expect_sha: str = arms.LOCALMELO_SHA,
    extra: Sequence[str] = (),
    base_url: str | None = None,
    model: str = arms.DEFAULT_MODEL,
    embedding_model: str | None = arms.EMBEDDING_MODEL,
    wall_cap_s: float = WALL_CAP_S,
    max_calls: int = MAX_CALLS,
    deadline: float | None = None,
    expected_chrome_version: str | None = None,
    gpu_lock: bool = True,
    log_bodies: bool = False,
    night_dir: str | None = None,
    harness: ModuleType | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """0 when every run is recorded, 2 when refused to start, 3 when stopped early."""
    out = out.resolve()
    localmelo = localmelo.resolve()
    extra = arms.check_extra(extra)
    chosen = list(chosen) if chosen is not None else list(tasks.TASKS)
    if harness is None:
        from bench import harness as harness_module

        harness = harness_module
    planned = schedule(chosen, repeats, arm_names)
    with web_run.bench_settings(harness, localmelo, base_url, model):
        problems = preflight(
            harness, localmelo, out, expect_sha=expect_sha, expected_chrome_version=expected_chrome_version,
            gpu_lock=gpu_lock,
        )  # fmt: skip
        if problems:
            print("Preflight failed:", *problems, sep="\n  - ", file=sys.stderr)
            return REFUSED
        config = identity(
            chosen, repeats, arm_names, known_arms, model=model, embedding_model=embedding_model,
            wall_cap_s=wall_cap_s, max_calls=max_calls, extra=extra, localmelo_sha=expect_sha,
        )  # fmt: skip
        state = load_state(out, config, planned)
        installed = chrome.chrome_version()
        try:
            chrome.check_version(installed, state.get("chrome_version"))
        except chrome.ChromeVersionChanged as exc:
            print(exc, file=sys.stderr)
            return REFUSED
        state["chrome_version"] = installed
        checkpoint = out / STATE_FILE
        log_path = out / "forms.log"
        env_path = out / "environment.json"
        env_doc = json.loads(env_path.read_text()) if env_path.exists() else {}
        env_doc.setdefault("base_url", base_url or harness.manifest.OMLX_BASE_URL)
        env_doc.setdefault("start", web_run.system_snapshot(localmelo, installed, model=model))
        web_run.save(env_path, env_doc)
        web_run.save(checkpoint, state)
        ctx = Context(
            out, harness, {t.id: t for t in chosen}, known_arms, installed, model, embedding_model, wall_cap_s,
            max_calls, tuple(extra), log_bodies,
        )  # fmt: skip

        def settle(item: dict[str, Any]) -> dict[str, Any] | None:
            """The item's record, running it if it needs running; None when the study must stop."""
            run_id = item["run_id"]
            existing = state["runs"].get(run_id)
            if existing and existing.get("state") == "finished":
                return existing
            if existing:  # started and cut off: its record, if it got that far, else a second try
                record_file = out / "runs" / run_id / "record.json"
                if record_file.exists():
                    state["runs"][run_id] = json.loads(record_file.read_text())
                    web_run.save(checkpoint, state)
                    return state["runs"][run_id]
                drop_run(out, run_id)
                del state["runs"][run_id]
                web_run.save(checkpoint, state)
                web_run._log(log_path, f"{run_id}: found cut off, running it again")
            reason = stop_reason(out, deadline=deadline, wall_cap_s=wall_cap_s, gpu_lock=gpu_lock)
            if reason:
                web_run._log(log_path, f"== {reason} before {run_id}; stopping")
                return None
            state["runs"][run_id] = {**item, "state": "started"}
            web_run.save(checkpoint, state)
            try:
                record = execute(item, ctx)
            except web_run.StopStudy as exc:
                state["runs"].pop(run_id, None)
                web_run.save(checkpoint, state)
                drop_run(out, run_id)
                web_run._log(log_path, f"== {exc} at {run_id}; stopping")
                return None
            state["runs"][run_id] = record
            web_run.save(checkpoint, state)
            web_run._log(
                log_path,
                f"[{item['index']}/{len(planned)}] {run_id}: {record['verdict']} ({record['end_reason']}) "
                f"calls={record['n_calls']} wall_s={record['wall_s']}",
            )
            return record

        stopped = False
        no_model_streak = 0
        clear_stale_lock(out / "ws")
        with runner.workspace_lock(out / "ws"), terminate_as_exception():
            try:
                for item in planned:
                    while True:
                        record = settle(item)
                        if record is None:
                            stopped = True
                            break
                        if record.get("n_calls"):
                            no_model_streak = 0
                            break
                        # No model call succeeded: the run says nothing about melo.
                        no_model_streak += 1
                        state["runs"].pop(item["run_id"], None)
                        web_run.save(checkpoint, state)
                        drop_run(out, item["run_id"])
                        web_run._log(
                            log_path, f"{item['run_id']}: no model call succeeded ({no_model_streak} in a row)"
                        )
                        if no_model_streak >= 2 or web_run._pause(out, NO_MODEL_PAUSE_S, sleep):
                            stopped = True
                            break
                    if stopped:
                        break
            except Terminated:
                web_run._log(log_path, "== SIGTERM; stopping")
                current = next((r for r, v in state["runs"].items() if v.get("state") == "started"), None)
                if current:
                    state["runs"].pop(current, None)
                    drop_run(out, current)
                stopped = True
        with contextlib.suppress(OSError):
            (out / "ws").rmdir()
        env_doc["end"] = web_run.system_snapshot(localmelo, installed, model=model)
        web_run.save(env_path, env_doc)
        state["complete"] = all(state["runs"].get(i["run_id"], {}).get("state") == "finished" for i in planned)
        web_run.save(checkpoint, state)
        write_report(out, night_dir)
        done = sum(1 for r in state["runs"].values() if r.get("state") == "finished")
        print(f"{done}/{len(planned)} runs recorded in {out}" + (" (stopped early; see forms.log)" if stopped else ""))
        return STOPPED if stopped else 0


# --- the plan, and the command line ---


def print_plan(
    out: Path,
    planned: Sequence[dict[str, Any]],
    chosen: Sequence[tasks.Task],
    repeats: int,
    arm_names: Sequence[str],
    known: Mapping[str, arms.Arm],
    *,
    localmelo: Path,
    base_url: str,
    model: str,
    embedding_model: str | None,
    wall_cap_s: float,
    max_calls: int,
    deadline: float | None,
    extra: Sequence[str],
    problems: Sequence[str],
    environment: Mapping[str, Any],
) -> None:
    print(
        f"Forms plan: {len(planned)} runs = {len(chosen)} tasks x {repeats} repeats x {len(arm_names)} arms; "
        f"{max_calls} model calls, wall cap {wall_cap_s:g} s "
        f"(melo's -t {wall_cap_s - arms.MELO_TIMEOUT_MARGIN_S:g} s), so at most {len(planned) * wall_cap_s / 60:.0f} "
        "GPU minutes"
    )
    print(f"Output: {out} ({STATE_FILE}, schedule.json, environment.json, runs/)")
    print(
        f"Model endpoint: {base_url} (not contacted in a dry run); model {model}; embeddings {embedding_model}; "
        f"proxy --thinking {THINKING}"
    )
    print("Browser: chrome-devtools-mcp through forms/tap.py, on a headless Chrome on an empty profile per run")
    for name in arm_names:
        arm = known[name]
        print(f"Arm {name}: {arm.description}; extra_body {json.dumps(dict(arm.extra_body))}")
        print(f"  melo flags: {' '.join(arms.flag_template(arm, max_calls, wall_cap_s, extra))}")
    for task in chosen:
        print(f"Task {task.id}: {task.title}, from {task.start}")
    for item in planned:
        print(f"{item['index']:3d}. {item['task']:13s} r{item['repeat']} {item['arm']}")
    state_path = out / STATE_FILE
    if state_path.exists():
        runs = json.loads(state_path.read_text()).get("runs", {})
        finished = sum(1 for r in runs.values() if r.get("state") == "finished")
        print(f"Checkpoint: {finished}/{len(planned)} runs already recorded; a resume skips them")
    print(
        f"Environment: {' '.join((environment.get('sw_vers') or '').split())} | "
        f"{' '.join((environment.get('xcodebuild_version') or '').split())} | "
        f"Chrome {environment.get('chrome_version') or 'n/a'} | "
        f"chrome-devtools-mcp {environment['chrome_devtools_mcp']['pinned']} | "
        f"localmelo {(environment.get('localmelo_sha') or 'n/a')[:8]} at {localmelo}"
    )
    print("Preflight: " + ("ok" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"  - {problem}")
    print(
        f"Battery: {web_run.battery_problem() or 'ok'}; "
        f"STOP file: {'present' if (out / 'STOP').exists() else 'absent'}; "
        f"deadline: {f'{deadline:.0f}' if deadline else 'none'}"
    )


def env_number(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run-dir", type=Path, required=True, help=f"output directory; a run there resumes from {STATE_FILE}"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the checks; start nothing, ask no model")
    ap.add_argument("--tasks", default="all", help=f"comma-separated, from {', '.join(tasks.BY_ID)} (default: all)")
    ap.add_argument("--arms", default=None, help=f"comma-separated (default: {','.join(arms.DEFAULT_ARMS)})")
    ap.add_argument("--arms-file", type=Path, default=None, help="a JSON file of more arms (forms/arms.py)")
    ap.add_argument("--repeats", type=int, default=REPEATS, help=f"repeats per task and arm (default {REPEATS})")
    ap.add_argument("--localmelo", type=Path, default=arms.LOCALMELO, help="the localmelo clone (default: the pinned)")
    ap.add_argument(
        "--localmelo-sha", default=arms.LOCALMELO_SHA, help="the commit the clone must be at, 7 digits or more"
    )
    ap.add_argument("--localmelo-args", default="", help="more flags for every run, shell-quoted, after the arm's own")
    ap.add_argument(
        "--base-url",
        default=os.environ.get("NIGHT_URL"),
        help="the model endpoint, http://host:port/v1 (default: $NIGHT_URL, else the bench manifest's)",
    )
    ap.add_argument("--model", default=arms.DEFAULT_MODEL, help=f"the chat model (default {arms.DEFAULT_MODEL})")
    ap.add_argument(
        "--embedding-model",
        default=arms.EMBEDDING_MODEL,
        help=f"the embedding model, or none (default {arms.EMBEDDING_MODEL})",
    )
    ap.add_argument("--wall-cap-s", type=float, default=WALL_CAP_S, help=f"seconds per run (default {WALL_CAP_S:g})")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS, help=f"main model calls per run (default {MAX_CALLS})")
    ap.add_argument(
        "--deadline", type=float, default=env_number("NIGHT_DEADLINE"), help="epoch seconds (default: $NIGHT_DEADLINE)"
    )
    ap.add_argument("--expected-chrome-version", default=None, help="refuse any other installed Chrome")
    ap.add_argument("--no-gpu-lock", action="store_true", help="do not require the GPU lock (a stub model)")
    ap.add_argument("--log-bodies", action="store_true", help="the proxy keeps every request and response body")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, harness: ModuleType | None = None) -> int:
    args = parse_args(argv)
    embedding_model = None if args.embedding_model.lower() == "none" else args.embedding_model
    try:
        known = arms.load_arms(args.arms_file)
        chosen = select_tasks(args.tasks)
        arm_names = select_arms(args.arms, known)
        planned = schedule(chosen, args.repeats, arm_names)
        extra = arms.check_extra(shlex.split(args.localmelo_args))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
    if harness is None:
        from bench import harness as harness_module

        harness = harness_module
    localmelo = args.localmelo.resolve()
    out = args.run_dir.resolve()
    if args.dry_run:
        with web_run.bench_settings(harness, localmelo, args.base_url, args.model):
            problems = preflight(
                harness, localmelo, out, expect_sha=args.localmelo_sha,
                expected_chrome_version=args.expected_chrome_version, gpu_lock=not args.no_gpu_lock, check_model=False,
            )  # fmt: skip
            version = None
            with contextlib.suppress(chrome.ChromeError):
                version = chrome.chrome_version()
            print_plan(
                out, planned, chosen, args.repeats, arm_names, known, localmelo=localmelo,
                base_url=args.base_url or harness.manifest.OMLX_BASE_URL, model=args.model,
                embedding_model=embedding_model, wall_cap_s=args.wall_cap_s, max_calls=args.max_calls,
                deadline=args.deadline, extra=extra, problems=problems,
                environment=web_run.system_snapshot(localmelo, version, model=args.model),
            )  # fmt: skip
        return REFUSED if problems else 0
    try:
        return run_study(
            out, chosen=chosen, repeats=args.repeats, arm_names=arm_names, known_arms=known, localmelo=localmelo,
            expect_sha=args.localmelo_sha, extra=extra, base_url=args.base_url, model=args.model,
            embedding_model=embedding_model, wall_cap_s=args.wall_cap_s, max_calls=args.max_calls,
            deadline=args.deadline, expected_chrome_version=args.expected_chrome_version,
            gpu_lock=not args.no_gpu_lock, log_bodies=args.log_bodies, night_dir=os.environ.get("NIGHT_DIR"),
            harness=harness,
        )  # fmt: skip
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    sys.exit(main())
