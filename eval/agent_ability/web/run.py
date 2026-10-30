"""The web search suite's study runner: agents answer SimpleQA Verified questions by operating Chrome (DESIGN.md).

    cd eval
    uv run python agent_ability/web/run.py --run-dir <dir> --dry-run   # schedule and checks, nothing started
    uv run python agent_ability/web/run.py --run-dir <dir>             # the study; run it again to resume
    uv run python agent_ability/web/report.py <dir>                    # report.md and report.json

The cells are harness x arm (claude_code, hermes, localmelo x D, S, C; S-cua only when it is named in ``--arms``). The
order is repeat-major: for each repeat, for each question, all cells run back to back, rotated by (question index +
repeat), so no cell is always first. Every run goes through bench.harness.run (the recorder proxy in between) with the
arm's MCP config from arms.py. D and S launch a headless Chrome on a clone of the profile template (chrome.py) first,
S-cua a headed one; C launches none. t0 is the start of the harness. Afterwards the run's History gives its navigation
counts (history.py), and the reply's ``FINAL ANSWER:`` line is graded against the question (answer.py, score.py).

``--model`` reaches all three harnesses (bench.harness.MODEL for Hermes and localmelo, a per-run env and settings copy
for Claude Code), and ``--claude-bin`` is the Claude Code binary every run uses.

A run that reaches Google's ``/sorry/`` CAPTCHA page (in its History or its tabs) is ``env_blocked``: the study pauses
(30 minutes by default) and runs it once more, in a run directory of its own. So is an S-cua run whose Chrome window
left the Space on screen (the tap refused its calls), which is run again without a pause. Two blocked runs in a row stop
the study. Before each S-cua run the Space on screen must be ``--cua-space``; the study waits for it up to 30 minutes,
then stops and records nothing. It also stops before the next run when ``<dir>/STOP`` exists, when the lock is gone
(unless ``--no-gpu-lock``), on battery below 30%, or when Chrome's version is no longer the one the study started with.

Files in ``<dir>``: web.json (the checkpoint), schedule.json, environment.json, web.log, and per run
``runs/<run_id>/`` with record.json, tap.jsonl, chrome/ (History, tabs.json, chrome.json) and ``<harness>/`` (the bench
run directory). The operator holds the GPU lock (holder: agent-benchmark); the driver never takes or releases it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

AGENT_ABILITY = Path(__file__).resolve().parent.parent
EVAL_ROOT = AGENT_ABILITY.parent
for _root in (EVAL_ROOT, AGENT_ABILITY):  # bench (for bench.harness) and smoke, web
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from bench.proxy.steps import is_generation, is_ok  # noqa: E402
from smoke import runner  # noqa: E402
from web import answer, arms, chrome, data, history, score  # noqa: E402
from web.chrome import bench_chrome  # noqa: E402

PROTOCOL = "web-v1"
HARNESS_ALIASES = {"claude-code": "claude_code", "cc": "claude_code", "melo": "localmelo"}
REPEATS = 1
# The wall cap per arm; D and S browse, C answers from memory. Steps and turns are the capability leg's.
WALL_CAP_S = {"D": 1800.0, "S": 1800.0, "C": 600.0, "S-cua": 1800.0}
BUDGET = {"max_steps": 30, "max_turns": 40, "thinking": "off"}
LOW_BATTERY_PCT = 30
CAPTCHA_PAUSE_S = 1800.0
SPACE_WAIT_S = 1800.0  # how long S-cua waits for its desktop Space to be the one on screen
SPACE_INTERVAL_S = 60.0
PAUSE_CHUNK_S = 30.0  # how often a pause looks for the STOP file
DONE = ("finished", "interrupted")
EXCLUDED = ("env_blocked", "interrupted")  # verdicts the report leaves out of its counts
VERDICTS = ("pass", "fail", "abstain", "no_answer", "timeout", "harness_error", "env_blocked", "interrupted")

_CAPTCHA = re.compile(r"\bgoogle\.[a-z]{2,3}(?:\.[a-z]{2})?/sorry\b", re.IGNORECASE)
_ANSWER_KEY = re.compile(r"simple[ _-]?qa", re.IGNORECASE)
# The tap's refusals when the browser's Space is not the one on screen (tap.py: _cua_window and _cua_target)
_SPACE_LOST = re.compile(r"not on screen|on the desktop on screen", re.IGNORECASE)
_MCP_TOOL_PREFIX = f"mcp__{arms.SERVER}__"
_SEARCH_TOOLS = ("web_search", "web_extract")  # a harness's own search tools, which no arm offers


class StopStudy(Exception):
    """Something outside the agents that makes the next run meaningless (Chrome cannot be run, or changed)."""


@dataclass
class Context:
    out: Path
    harness: ModuleType
    rows: dict[int, data.Row]
    cwd: Path
    chrome_version: str | None  # the version the study started with; bench_chrome refuses any other
    model: str = arms.DEFAULT_MODEL
    claude_bin: Path = arms.CLAUDE_BIN
    cua_bin: str | None = None  # S-cua: the cua-driver binary, and the Space its runs need on screen
    cua_space: int | None = None
    sleep: Callable[[float], None] = time.sleep
    localmelo_extra: tuple[str, ...] = ()  # appended to the flags of every localmelo run


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _log(path: Path, text: str) -> None:
    with path.open("a") as log:
        log.write(f"{dt.datetime.now().astimezone().replace(microsecond=0).isoformat()} {text}\n")


def _out(cmd: Sequence[str], timeout: float = 30.0) -> str:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


# --- the schedule ---


def rotated(cells: Sequence[Any], offset: int) -> list[Any]:
    k = offset % len(cells)
    return [*cells[k:], *cells[:k]]


def schedule(
    rows: Sequence[data.Row],
    repeats: int = REPEATS,
    harnesses: Sequence[str] = arms.HARNESSES,
    arm_names: Sequence[str] = arms.DEFAULT_ARMS,
) -> list[dict[str, Any]]:
    """Repeat-major runs. The cells (harness x arm) of (question index i, repeat r) start at cell number (i + r) mod
    len(cells), with i counted from 0 and r from 1, so each cell takes every position over the questions."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not rows or not harnesses or not arm_names:
        raise ValueError("questions, harnesses and arms must not be empty")
    unknown = [h for h in harnesses if h not in arms.HARNESSES]
    if unknown:
        raise ValueError(f"unknown harness {', '.join(unknown)}; choose from {', '.join(arms.HARNESSES)}")
    unknown = [a for a in arm_names if a not in arms.ARMS]
    if unknown:
        raise ValueError(f"unknown arm {', '.join(unknown)}; choose from {', '.join(arms.ARMS)}")
    cells = [(h, a) for h in harnesses for a in arm_names]
    planned: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for question_index, row in enumerate(rows):
            for position, (harness, arm) in enumerate(rotated(cells, question_index + repeat), 1):
                index = len(planned) + 1
                planned.append(
                    {
                        "index": index,
                        "question": row.id,
                        "tier": row.tier,
                        "repeat": repeat,
                        "harness": harness,
                        "arm": arm,
                        "position": position,
                        "attempt": 1,
                        "run_id": f"{index:03d}__q{row.id}__r{repeat}__{harness}__{arm}",
                    }
                )
    return planned


def retry_of(item: dict[str, Any]) -> dict[str, Any]:
    """The second attempt of a run that a CAPTCHA blocked: the same cell, in a run directory of its own."""
    return {**item, "attempt": 2, "retry_of": item["run_id"], "run_id": f"{item['run_id']}__retry"}


def select_questions(spec: str | None, pilot: data.Pilot) -> list[data.Row]:
    """Question ids, tiers (easy, medium) or ``all``, comma-separated, in the order given and without repeats."""
    if not spec or spec.strip() == "all":
        return list(pilot.rows)
    chosen: list[data.Row] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token in data.TIERS:
            found = list(pilot.tier(token))
        elif token == "all":
            found = list(pilot.rows)
        else:
            try:
                found = [pilot.by_id(int(token))]
            except (ValueError, KeyError):
                raise ValueError(f"unknown question {token!r}; use ids of the pilot set, easy, medium or all") from None
        chosen += [r for r in found if r not in chosen]
    return chosen


def parse_names(spec: str | None, choices: Sequence[str], what: str, default: Sequence[str] | None = None) -> list[str]:
    if not spec:
        return list(default or choices)
    names = [HARNESS_ALIASES.get(n.strip(), n.strip()) for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in choices]
    if unknown:
        raise ValueError(f"unknown {what} {', '.join(unknown)}; choose from {', '.join(choices)}")
    return list(dict.fromkeys(names))


# --- guards ---


def battery_problem(low_pct: int = LOW_BATTERY_PCT) -> str | None:
    text = _out(["pmset", "-g", "batt"])
    found = re.search(r"(\d+)%", text)
    if "'Battery Power'" in text and found and int(found.group(1)) < low_pct:
        return f"on battery at {found.group(1)}% (below {low_pct}%)"
    return None


def stop_reason(out: Path, *, gpu_lock: bool = True) -> str | None:
    """Why the driver must not start the next run, or None."""
    if (out / "STOP").exists():
        return "STOP file found"
    return (runner.lock_problem() if gpu_lock else None) or battery_problem()


def model_problem(harness: ModuleType) -> str | None:
    """The model endpoint (harness.UPSTREAM) must answer /v1/models."""
    url = f"{harness.UPSTREAM}/v1/models"
    request = urllib.request.Request(url)
    token_cmd = os.environ.get(harness.UPSTREAM_TOKEN_CMD_ENV)
    if token_cmd:  # a server that needs a bearer token, which the proxy fetches the same way
        from bench.proxy.config import upstream_token

        try:
            request.add_header("Authorization", f"Bearer {upstream_token(token_cmd)}")
        except ValueError as exc:
            return str(exc)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status != 200:
                return f"the model server answered {response.status} at {url}"
    except OSError as exc:
        return f"the model server is not reachable at {url} ({exc})"
    return None


def installed_chrome_version() -> str:
    return chrome.chrome_version()


def cua_problems(cua_space: int | None) -> list[str]:
    """S-cua needs its Space number and a running cua-driver daemon."""
    found = []
    if cua_space is None:
        found.append("S-cua needs --cua-space, the desktop Space its Chrome window will be on")
    try:
        problem = arms.cua_status_problem(arms.resolve_cua_bin())
    except ValueError as exc:
        problem = str(exc)
    return [*found, *([problem] if problem else [])]


def wait_for_space(ctx: Context) -> str | None:
    """Before an S-cua run: None once the Space on screen is the study's, waiting up to SPACE_WAIT_S for it; else the
    reason to stop. The tap refuses to act while the browser's Space is off screen, so a run begun on the wrong
    one would be wasted."""
    assert ctx.cua_bin is not None and ctx.cua_space is not None
    waited = 0.0
    while True:
        try:
            current: int | None = arms.cua_current_space(ctx.cua_bin)
        except arms.CuaError as exc:
            current, note = None, f" ({exc})"
        else:
            note = ""
            if current == ctx.cua_space:
                return None
        if (ctx.out / "STOP").exists():
            return "STOP file found"
        if waited >= SPACE_WAIT_S:
            return f"the Space on screen is {current}, not {ctx.cua_space}, after {waited:g} s{note}"
        step = min(SPACE_INTERVAL_S, SPACE_WAIT_S - waited)
        ctx.sleep(step)
        waited += step


def preflight(
    harness: ModuleType,
    harnesses: Sequence[str],
    arm_names: Sequence[str],
    localmelo: Path,
    *,
    expected_chrome_version: str | None = None,
    expect_sha: str = arms.LOCALMELO_SHA,
    gpu_lock: bool = True,
    model: str = arms.DEFAULT_MODEL,
    claude_bin: Path = arms.CLAUDE_BIN,
    cua_space: int | None = None,
) -> list[str]:
    """Everything that would make the study invalid, as messages. Run it inside ``bench_settings``."""
    problems: list[str] = []
    try:
        data.load()
    except (data.DataError, OSError, ValueError) as exc:
        problems.append(f"the question file is invalid: {exc}")
    binaries = runner._binaries(harness)
    problems += [
        f"{h}: {binaries[h]} is missing"
        for h in harnesses
        if h in binaries and h != "claude_code" and not binaries[h].exists()
    ]
    if "claude_code" in harnesses:  # the binary, the per-run env and settings copies, and the safety variables
        problems += arms.claude_problems(Path(harness.CLAUDE_ENV), Path(harness.CLAUDE_HOME), model, claude_bin)
    if "localmelo" in harnesses:
        problems += arms.localmelo_problems(localmelo, expect_sha)
    if "S-cua" in arm_names:
        problems += cua_problems(cua_space)
    if gpu_lock:
        problems += [p for p in (runner.lock_problem(),) if p]
    problems += [p for p in (model_problem(harness),) if p]
    problems += arms.mcp_problems(arm_names)
    if any(arms.NEEDS_CHROME[a] for a in arm_names):
        problems += chrome.preflight()
        if not chrome.is_bench_profile(chrome.TEMPLATE_DIR):
            problems.append(f"{chrome.TEMPLATE_DIR} is not a profile template; run web/chrome.py init")
        else:
            try:
                chrome.check_version(
                    installed_chrome_version(),
                    chrome.read_marker(chrome.TEMPLATE_DIR).get("chrome_version"),
                    expected_chrome_version,
                )
            except chrome.ChromeError as exc:
                problems.append(str(exc))
    return problems


@contextlib.contextmanager
def bench_settings(
    harness: ModuleType,
    localmelo: Path,
    base_url: str | None,
    model: str = arms.DEFAULT_MODEL,
    claude_bin: Path = arms.CLAUDE_BIN,
) -> Iterator[None]:
    """Point bench.harness at the pinned localmelo checkout, the model endpoint, the model and the Claude Code binary,
    for the length of the study.

    ``base_url`` is the model server (``http://host:port/v1``); the recorder proxy forwards to it, and the run
    manifests record it. None keeps the bench manifest's default. ``model`` is what Hermes's config and localmelo's
    ``--chat-model`` read (``harness.MODEL``) and what the manifests record; Claude Code gets it through arms.py.
    """
    manifest = harness.manifest
    saved = (
        manifest.LOCALMELO_REPO,
        manifest.OMLX_BASE_URL,
        harness.UPSTREAM,
        manifest.BENCH_MODEL,
        harness.MODEL,
        manifest.CLAUDE_BIN,
    )
    try:
        manifest.LOCALMELO_REPO = localmelo
        manifest.BENCH_MODEL = harness.MODEL = model
        manifest.CLAUDE_BIN = claude_bin
        if base_url:
            manifest.OMLX_BASE_URL = base_url.rstrip("/")
            harness.UPSTREAM = manifest.OMLX_BASE_URL.removesuffix("/v1")
        yield
    finally:
        (
            manifest.LOCALMELO_REPO,
            manifest.OMLX_BASE_URL,
            harness.UPSTREAM,
            manifest.BENCH_MODEL,
            harness.MODEL,
            manifest.CLAUDE_BIN,
        ) = saved


# --- what a run leaves behind ---


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def final_text(harness: str, stdout: Path) -> str:
    """The agent's final reply, from the harness's stdout.log.

    - claude_code (stream-json): the ``result`` event's ``result``; else the last assistant message's text.
    - hermes (stream-json): the ``result`` event's ``text``; else the text deltas after the last tool event.
    - localmelo (``-o json``): the ``result`` of the run's one JSON line.
    """
    events = _json_lines(stdout)
    if harness == "localmelo":
        line = next((e for e in reversed(events) if "task_id" in e and "status" in e), None)
        return str(line.get("result") or "") if line else ""
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if harness == "claude_code":
        if result and isinstance(result.get("result"), str) and result["result"].strip():
            return result["result"]
        for event in reversed(events):
            if event.get("type") == "assistant":
                text = _blocks_text((event.get("message") or {}).get("content"))
                if text.strip():
                    return text
        return ""
    if harness == "hermes":
        if result and isinstance(result.get("text"), str) and result["text"].strip():
            return result["text"]
        tail: list[str] = []
        for event in events:
            if event.get("type") in ("tool_use", "tool_result"):
                tail = []
            elif event.get("type") == "text":
                tail.append(str(event.get("text") or ""))
        return "".join(tail)
    raise ValueError(f"unknown harness {harness!r}")


def call_metrics(calls: Path) -> dict[str, Any]:
    """Tokens and the tool names a harness offered, from the proxy's calls.jsonl (ok generation requests)."""
    prompt = cached = output = 0
    offered: set[str] = set()
    for record in _json_lines(calls):
        if not (is_generation(record) and is_ok(record)):
            continue
        usage = record.get("usage") or {}
        prompt += usage.get("prompt_tokens") or 0
        cached += usage.get("cached_tokens") or 0
        output += usage.get("completion_tokens") or 0
        offered.update(record.get("request_tool_names") or [])
    foreign = sorted(n for n in offered if not n.startswith(_MCP_TOOL_PREFIX))
    return {
        "prompt_tokens": prompt,
        "uncached_tokens": prompt - cached,
        "output_tokens": output,
        "foreign_tools": foreign,
        "search_tool_leak": any(n in _SEARCH_TOOLS for n in foreign),
    }


def tap_metrics(log: Path) -> dict[str, Any]:
    """What the agent did through the tap: calls by status and tool, pictures, the longest text result."""
    events = _json_lines(log)
    calls = [e for e in events if e.get("event") == "call"]
    return {
        "n_calls": len(calls),
        "by_status": dict(Counter(str(c.get("status")) for c in calls)),
        "by_tool": dict(Counter(str(c.get("tool")) for c in calls)),
        "n_images": sum(c.get("n_images") or 0 for c in calls),
        "max_chars_out": max((c.get("chars_out") or 0 for c in calls), default=0),
        # S-cua: calls the tap refused because the browser's Space was not on screen
        "space_lost": sum(
            1 for c in calls if c.get("status") == "refused" and _SPACE_LOST.search(str(c.get("head") or ""))
        ),
        "child_failed": any(e.get("event") == "child_start_failed" for e in events),
    }


def read_tabs(path: Path) -> list[dict[str, Any]]:
    try:
        tabs = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [t for t in tabs if isinstance(t, dict)] if isinstance(tabs, list) else []


def captcha_seen(urls: Sequence[str], tabs: Sequence[dict[str, Any]]) -> bool:
    """Google's ``/sorry/`` page: in a top-level visit of the run, or in a tab that was open at the end."""
    return any(_CAPTCHA.search(u or "") for u in [*urls, *(t.get("url") or "" for t in tabs)])


def answer_key_suspect(urls: Sequence[str], tabs: Sequence[dict[str, Any]]) -> bool:
    """A visited URL, or a tab's URL or title, that names SimpleQA."""
    seen = [*urls, *(str(t.get(k) or "") for t in tabs for k in ("url", "title"))]
    return any(_ANSWER_KEY.search(s) for s in seen)


def gold_url_visited(urls: Sequence[str], row: data.Row) -> bool:
    hosts = row.gold_hosts()
    return any(data.url_host(u) in hosts for u in urls)


def decide(*, blocked: bool, error: str | None, summary: dict[str, Any] | None, n_llm_total: int, graded: str) -> str:
    """env_blocked, harness_error, timeout, else the answer's grade (pass, fail, abstain or no_answer)."""
    if blocked:
        return "env_blocked"
    if error is not None or summary is None:
        return "harness_error"
    if summary.get("timed_out"):
        return "timeout"
    if n_llm_total == 0:  # no model call succeeded
        return "harness_error"
    return graded


# --- one run ---


def execute(item: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """One run: Chrome (D, S and S-cua), the harness, then History, the answer and its grade. Writes and returns its
    record."""
    row = ctx.rows[item["question"]]
    name, arm = item["harness"], item["arm"]
    run_root = ctx.out / "runs" / item["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    agent_dir = run_root / name
    chrome_dir = run_root / "chrome"
    text = arms.prompt(arm, row.problem)
    summary: dict[str, Any] | None = None
    error: str | None = None
    fallback_t0 = time.time()
    try:
        with contextlib.ExitStack() as stack:
            browser_url = chrome_pid = None
            if arms.NEEDS_CHROME[arm]:
                browser = stack.enter_context(
                    bench_chrome(chrome_dir, expected_version=ctx.chrome_version, headless=arm not in arms.HEADED)
                )
                browser_url = browser.browser_url
                chrome_pid = browser.chrome.proc.pid if arm in arms.HEADED else None
            fallback_t0 = time.time()
            try:
                with arms.overrides(
                    ctx.harness,
                    name,
                    arm,
                    run_dir=run_root,
                    browser_url=browser_url,
                    chrome_pid=chrome_pid,
                    cua_bin=ctx.cua_bin,
                    model=ctx.model,
                    claude_bin=ctx.claude_bin,
                    localmelo_extra=ctx.localmelo_extra,
                ) as plan:
                    summary = ctx.harness.run(
                        name,
                        text,
                        ctx.cwd,
                        agent_dir,
                        timeout_s=WALL_CAP_S[arm],
                        max_steps=BUDGET["max_steps"],
                        max_turns=BUDGET["max_turns"],
                        thinking=BUDGET["thinking"],
                        harness_args=plan.harness_args,
                    )
            except Exception as exc:  # noqa: BLE001 - the run did not complete; record it and go on with the next one
                error = f"{type(exc).__name__}: {exc}"
                agent_dir.mkdir(parents=True, exist_ok=True)
            # leaving the stack closes Chrome, which commits History
    except chrome.ChromeError as exc:
        raise StopStudy(f"Chrome: {exc}") from exc
    t0 = float((summary or {}).get("started_at") or fallback_t0)
    t_end = t0 + float((summary or {}).get("wall_s") or max(time.time() - t0, 0.0))

    nav: history.HistorySummary | None = None
    nav_error: str | None = None
    tabs: list[dict[str, Any]] = []
    if arms.NEEDS_CHROME[arm]:
        tabs = read_tabs(chrome_dir / "tabs.json")
        if (chrome_dir / "History").exists():
            try:
                nav = history.summarize(chrome_dir / "History", t0, t_end)
            except (sqlite3.Error, OSError) as exc:
                nav_error = f"{type(exc).__name__}: {exc}"
        else:
            nav_error = "no History file"
    urls = nav.urls if nav else []
    tap = tap_metrics(run_root / "tap.jsonl")
    captcha = arms.NEEDS_CHROME[arm] and captcha_seen(urls, tabs)
    space_lost = arm == "S-cua" and tap["space_lost"] > 0  # its browser's Space left the screen: the run saw nothing
    blocked_by = "captcha" if captcha else "space" if space_lost else None

    stdout = agent_dir / "stdout.log"
    reply = final_text(name, stdout) if stdout.exists() else ""
    extraction = answer.extract(reply)
    grade = score.grade(reply, **row.scoring())
    steps = (summary or {}).get("steps") or {}
    n_main, n_aux = steps.get("n_llm") or 0, steps.get("n_llm_aux") or 0
    verdict = decide(
        blocked=blocked_by is not None, error=error, summary=summary, n_llm_total=n_main + n_aux, graded=grade.verdict
    )
    calls = call_metrics(agent_dir / "calls.jsonl")
    report = ((summary or {}).get("harness_report")) or {}

    n_nav = nav.n_nav if nav else None
    record: dict[str, Any] = {
        **item,
        "state": "finished",
        "kind": row.kind,
        # verdict
        "verdict": verdict,
        "passed": verdict == "pass",
        "answer": extraction.answer,
        "answer_status": extraction.status,
        "matched": grade.matched,
        "timed_out": (summary or {}).get("timed_out"),
        "exit_code": (summary or {}).get("exit_code"),
        "error": error,
        # cost; C counts its tool-less requests, which the proxy calls aux
        "wall_s": (summary or {}).get("wall_s"),
        "wall_cap_s": WALL_CAP_S[arm],
        "n_llm": n_main + n_aux if arm == "C" else n_main,
        "n_llm_aux": n_aux,
        "n_llm_err": steps.get("n_llm_err"),
        "n_tool": steps.get("n_tool"),
        "output_tokens": calls["output_tokens"],
        "uncached_tokens": calls["uncached_tokens"],
        "prompt_tokens": calls["prompt_tokens"],
        "n_images": steps.get("images_new_sum"),
        # navigation; null without Chrome (C) or without a readable History
        "n_nav": n_nav,
        "n_search": nav.n_search if nav else None,
        "n_pages": nav.n_pages if nav else None,
        "gold_url_visited": gold_url_visited(urls, row) if nav else None,
        "zero_nav": (n_nav == 0 and verdict in ("pass", "fail")) if nav else None,
        "serp_only": (nav.n_search > 0 and nav.n_pages == 0) if nav else None,
        "urls": urls,
        "nav_error": nav_error,
        # flags
        "captcha": captcha,
        "space_lost": tap["space_lost"] if arm == "S-cua" else None,  # S-cua calls refused: Space off screen
        "blocked_by": blocked_by,  # captcha or space when the verdict is env_blocked
        "cua_space": ctx.cua_space if arm == "S-cua" else None,
        "answer_key_suspect": answer_key_suspect(urls, tabs) if arms.NEEDS_CHROME[arm] else None,
        "mcp_output_truncated": None,  # not derivable from the logs yet
        "foreign_tools": calls["foreign_tools"],
        "search_tool_leak": calls["search_tool_leak"],
        "tap": tap if (run_root / "tap.jsonl").exists() else None,
        "harness_status": report.get("subtype") or report.get("task_status") or report.get("exit_code"),
        # provenance
        "t0": t0,
        "t_end": t_end,
        "finished_at": time.time(),
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "run_dir": str(run_root.relative_to(ctx.out)),
    }
    save(run_root / "record.json", record)
    return record


def salvage_interrupted(item: dict[str, Any], out: Path) -> dict[str, Any]:
    """A run that was started and left no record.json: recorded as interrupted, never replayed."""
    run_root = out / "runs" / item["run_id"]
    summary_path = run_root / item["harness"] / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return {
        **{k: v for k, v in item.items() if k != "state"},
        "state": "interrupted",
        "verdict": "interrupted",
        "passed": False,
        "wall_s": (summary or {}).get("wall_s"),
        "n_llm": ((summary or {}).get("steps") or {}).get("n_llm"),
        "run_dir": str(run_root.relative_to(out)),
        "error": "interrupted",
    }


# --- the study ---


def system_snapshot(
    localmelo: Path | None,
    chrome_version: str | None,
    *,
    model: str | None = None,
    claude_bin: Path | None = None,
    cua_bin: str | None = None,
) -> dict[str, Any]:
    """The macOS and Xcode builds, the versions of the pieces the study runs, and the model server."""
    from bench import manifest

    return {
        "timestamp": time.time(),
        "sw_vers": _out(["sw_vers"]),
        "xcodebuild_version": _out(["xcodebuild", "-version"]),
        "model": model,
        "chrome_version": chrome_version,
        "chrome_devtools_mcp": arms.mcp_versions(),
        "node": arms.node_version(),
        "claude_code_version": _out([str(claude_bin), "--version"]) if claude_bin else None,
        "cua_driver_version": _out([cua_bin, "--version"]) if cua_bin else None,
        "localmelo_sha": arms.localmelo_sha(localmelo) if localmelo else None,
        "model_server": manifest._model_server(),
        "pmset_batt": _out(["pmset", "-g", "batt"]),
    }


def identity(
    pilot: data.Pilot,
    rows: Sequence[data.Row],
    repeats: int,
    harnesses: Sequence[str],
    arm_names: Sequence[str],
    model: str = arms.DEFAULT_MODEL,
    localmelo_extra: Sequence[str] = (),
    localmelo_sha: str = arms.LOCALMELO_SHA,
) -> dict[str, Any]:
    """What a resume must match. localmelo's flags include ``localmelo_extra``; ``localmelo_sha`` is the commit (or the
    prefix) the study expects of its checkout, not the checkout's own hash (that is in environment.json)."""
    return {
        "protocol": PROTOCOL,
        "harnesses": list(harnesses),
        "arms": list(arm_names),
        "questions": [r.id for r in rows],
        "repeats": repeats,
        "budget": BUDGET,
        "wall_cap_s": {a: WALL_CAP_S[a] for a in arm_names},
        "localmelo_flags": arms.harness_arg_templates(arm_names, localmelo_extra) if "localmelo" in harnesses else None,
        "localmelo_sha": localmelo_sha if "localmelo" in harnesses else None,
        "dataset_revision": pilot.source["revision"],
        "chrome_devtools_mcp": arms.mcp_versions()["pinned"],
        "model": model,
    }


def load_state(out: Path, config: dict[str, Any], planned: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = out / "web.json"
    digest = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config or state.get("schedule_sha256") != digest:
            raise ValueError("the web study's config or schedule changed; use a fresh directory")
        return state
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no web.json checkpoint; use a fresh directory")
    state = {"config": config, "schedule_sha256": digest, "chrome_version": None, "runs": {}, "complete": False}
    save(out / "schedule.json", planned)
    save(checkpoint, state)
    return state


def _pause(out: Path, seconds: float, sleep: Callable[[float], None]) -> str | None:
    """Wait ``seconds`` in chunks, looking for the STOP file between them. Returns why it ended early, or None."""
    remaining = seconds
    while remaining > 0:
        if (out / "STOP").exists():
            return "STOP file found"
        chunk = min(remaining, PAUSE_CHUNK_S)
        sleep(chunk)
        remaining -= chunk
    return None


def run_study(
    out: Path,
    *,
    rows: Sequence[data.Row] | None = None,
    repeats: int = REPEATS,
    harnesses: Sequence[str] = arms.HARNESSES,
    arm_names: Sequence[str] = arms.DEFAULT_ARMS,
    localmelo: Path = arms.LOCALMELO,
    base_url: str | None = None,
    expected_chrome_version: str | None = None,
    expect_sha: str = arms.LOCALMELO_SHA,
    captcha_pause_s: float = CAPTCHA_PAUSE_S,
    gpu_lock: bool = True,
    model: str = arms.DEFAULT_MODEL,
    claude_bin: Path = arms.CLAUDE_BIN,
    cua_space: int | None = None,
    harness: ModuleType | None = None,
    sleep: Callable[[float], None] = time.sleep,
    localmelo_extra: Sequence[str] = (),
) -> int:
    """0 when every run is recorded, 2 when refused to start, 3 when stopped early.

    ``expect_sha`` is the commit (or a prefix of at least 7 digits) the localmelo checkout must be at, and
    ``localmelo_extra`` is appended to every localmelo run's flags; both are part of the study's identity."""
    out = out.resolve()
    localmelo_extra = arms.check_localmelo_extra(localmelo_extra)
    localmelo = localmelo.resolve()
    claude_bin = Path(claude_bin)
    if "S-cua" in arm_names and cua_space is None:
        raise ValueError("S-cua needs cua_space (--cua-space): the desktop Space its Chrome window will be on")
    if harness is None:
        from bench import harness as harness_module

        harness = harness_module
    pilot = data.load()
    rows = list(rows) if rows is not None else list(pilot.rows)
    planned = schedule(rows, repeats, harnesses, arm_names)
    with bench_settings(harness, localmelo, base_url, model, claude_bin):
        problems = preflight(
            harness, harnesses, arm_names, localmelo,
            expected_chrome_version=expected_chrome_version, expect_sha=expect_sha, gpu_lock=gpu_lock,
            model=model, claude_bin=claude_bin, cua_space=cua_space,
        )  # fmt: skip
        if problems:
            print("Preflight failed:", *problems, sep="\n  - ", file=sys.stderr)
            return 2
        config = identity(pilot, rows, repeats, harnesses, arm_names, model, localmelo_extra, expect_sha)
        state = load_state(out, config, planned)
        uses_chrome = any(arms.NEEDS_CHROME[a] for a in arm_names)
        if uses_chrome:
            installed = installed_chrome_version()
            recorded = state.get("chrome_version")
            try:
                chrome.check_version(installed, recorded, expected_chrome_version)
            except chrome.ChromeVersionChanged as exc:
                print(exc, file=sys.stderr)
                return 2
            state["chrome_version"] = installed
        else:
            installed = None
        log_path = out / "web.log"
        env_path = out / "environment.json"
        env_doc = json.loads(env_path.read_text()) if env_path.exists() else {}
        env_doc.setdefault("base_url", base_url or harness.manifest.OMLX_BASE_URL)
        cua_bin = arms.resolve_cua_bin() if "S-cua" in arm_names else None
        snapshot_args: dict[str, Any] = {
            "model": model,
            "claude_bin": claude_bin if "claude_code" in harnesses else None,
            "cua_bin": cua_bin,
        }
        melo = localmelo if "localmelo" in harnesses else None
        env_doc.setdefault("start", system_snapshot(melo, installed, **snapshot_args))
        save(env_path, env_doc)
        save(out / "web.json", state)
        checkpoint = out / "web.json"
        ctx = Context(
            out,
            harness,
            {r.id: r for r in rows},
            out / "ws" / "cwd",
            installed,
            model,
            claude_bin,
            cua_bin,
            cua_space,
            sleep,
            tuple(localmelo_extra),
        )
        stopped = False

        def settle(item: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
            """The item's final record and whether this call ran it; (None, False) when the study must stop."""
            run_id = item["run_id"]
            existing = state["runs"].get(run_id)
            if existing and existing.get("state") in DONE:
                return existing, False
            if existing and existing.get("state") == "started":
                record_file = out / "runs" / run_id / "record.json"
                state["runs"][run_id] = (
                    json.loads(record_file.read_text()) if record_file.exists() else salvage_interrupted(existing, out)
                )
                save(checkpoint, state)
                _log(log_path, f"{run_id}: found started, recorded as {state['runs'][run_id]['state']}")
                return state["runs"][run_id], False
            reason = stop_reason(out, gpu_lock=gpu_lock)
            if not reason and item["arm"] == "S-cua":
                reason = wait_for_space(ctx)
            if reason:
                _log(log_path, f"== {reason} before {run_id}; stopping")
                return None, False
            state["runs"][run_id] = {**item, "state": "started"}
            save(checkpoint, state)
            try:
                record = execute(item, ctx)
            except StopStudy as exc:
                state["runs"].pop(run_id, None)  # nothing was recorded; a resume starts it again
                save(checkpoint, state)
                _log(log_path, f"== {exc} at {run_id}; stopping")
                return None, False
            state["runs"][run_id] = record
            save(checkpoint, state)
            _log(
                log_path,
                f"[{item['index']}/{len(planned)}] {run_id}: {record['verdict']} n_llm={record['n_llm']} "
                f"n_nav={record['n_nav']} wall_s={record['wall_s']}",
            )
            return record, True

        ctx.cwd.mkdir(parents=True, exist_ok=True)
        streak = 0  # blocked runs in a row, in this invocation
        with runner.workspace_lock(out / "ws"):
            for item in planned:
                current = item
                while not stopped:
                    record, ran = settle(current)
                    if record is None:
                        stopped = True
                        break
                    blocked = record["verdict"] == "env_blocked"
                    if ran:
                        streak = streak + 1 if blocked else 0
                        if streak >= 2:
                            _log(
                                log_path,
                                f"== two runs in a row were blocked at {current['run_id']}; stopping",
                            )
                            stopped = True
                            break
                    again = retry_of(current)
                    if (
                        not blocked
                        or current["attempt"] > 1
                        or state["runs"].get(again["run_id"], {}).get("state") in DONE
                    ):
                        break
                    # A CAPTCHA wants a long pause. A lost Space does not: the next S-cua run waits for the Space itself.
                    pause_s = captcha_pause_s if record.get("blocked_by", "captcha") == "captcha" else 0.0
                    waited = time.time() - float(record.get("finished_at") or 0)
                    _log(
                        log_path,
                        f"{current['run_id']}: blocked ({record.get('blocked_by', 'captcha')}); running it again",
                    )
                    reason = _pause(out, max(pause_s - waited, 0.0), sleep)
                    if reason:
                        _log(log_path, f"== {reason} during the pause; stopping")
                        stopped = True
                        break
                    current = again
                if stopped:
                    break
        ws = out / "ws"
        with contextlib.suppress(OSError):
            ctx.cwd.rmdir()
            ws.rmdir()
        env_doc["end"] = system_snapshot(melo, installed, **snapshot_args)
        save(env_path, env_doc)
        state["complete"] = all(state["runs"].get(i["run_id"], {}).get("state") in DONE for i in planned)
        save(checkpoint, state)
        done = sum(1 for r in state["runs"].values() if r.get("state") in DONE)
        print(f"{done}/{len(planned)} runs recorded in {out}" + (" (stopped early; see web.log)" if stopped else ""))
        return 3 if stopped else 0


# --- the plan, and the command line ---


def print_plan(
    out: Path,
    planned: list[dict[str, Any]],
    rows: Sequence[data.Row],
    repeats: int,
    harnesses: Sequence[str],
    arm_names: Sequence[str],
    localmelo: Path,
    base_url: str,
    problems: list[str],
    environment: dict[str, Any],
    captcha_pause_s: float,
    *,
    model: str = arms.DEFAULT_MODEL,
    claude_bin: Path = arms.CLAUDE_BIN,
    cua_note: str | None = None,
    localmelo_extra: Sequence[str] = (),
) -> None:
    caps = ", ".join(f"{a} {WALL_CAP_S[a]:g}" for a in arm_names)
    print(
        f"Web plan: {len(planned)} runs = {len(rows)} questions x {repeats} repeats x {len(harnesses)} harnesses x "
        f"{len(arm_names)} arms; {BUDGET['max_steps']} proxy steps, {BUDGET['max_turns']} turns, "
        f"thinking {BUDGET['thinking']}; wall caps (s): {caps}"
    )
    print(f"Output: {out} (web.json, schedule.json, environment.json, runs/)")
    print(
        f"Model endpoint: {base_url}; model {model}; CAPTCHA pause {captcha_pause_s:g} s, one retry, two blocked in a row stop"
    )
    if "claude_code" in harnesses:
        print(f"Claude Code: {claude_bin}")
    for arm in arm_names:
        if not arms.NEEDS_CHROME[arm]:
            chrome_note = "no Chrome"
        elif arm in arms.HEADED:
            chrome_note = "headed Chrome on a profile clone, on the Space on screen"
        else:
            chrome_note = "headless Chrome on a profile clone"
        print(f"Arm {arm}: tap --mode {arms.TAP_MODES[arm]}; {chrome_note}")
    if cua_note:
        print(cua_note)
    if "localmelo" in harnesses:
        for arm, flags in arms.harness_arg_templates(arm_names, localmelo_extra).items():
            print(f"localmelo {arm}: {' '.join(flags)}")
    for item in planned:
        retry = "" if item["attempt"] == 1 else " (retry)"
        print(
            f"{item['index']:3d}. q{item['question']:<5d} {item['tier']:6s} r{item['repeat']} "
            f"{item['harness']:11s} {item['arm']}{retry}"
        )
    state_path = out / "web.json"
    if state_path.exists():
        runs = json.loads(state_path.read_text()).get("runs", {})
        finished = sum(1 for r in runs.values() if r.get("state") in DONE)
        print(f"Checkpoint: {finished}/{len(planned)} runs already recorded; a resume skips them")
    print(
        f"Environment: {' '.join((environment.get('sw_vers') or '').split())} | "
        f"{' '.join((environment.get('xcodebuild_version') or '').split())} | "
        f"Claude Code {environment.get('claude_code_version') or 'n/a'} | "
        f"Chrome {environment.get('chrome_version') or 'n/a'} | "
        f"chrome-devtools-mcp {environment['chrome_devtools_mcp']['pinned']} | "
        f"localmelo {(environment.get('localmelo_sha') or 'n/a')[:7]} at {localmelo}"
    )
    print("Preflight: " + ("ok" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"  - {problem}")
    battery = battery_problem()
    print(f"Battery: {battery or 'ok'}; STOP file: {'present' if (out / 'STOP').exists() else 'absent'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="output directory; a run there resumes from web.json")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and the checks; start nothing")
    ap.add_argument("--harnesses", default=",".join(arms.HARNESSES), help="comma-separated (default: all three)")
    ap.add_argument(
        "--arms",
        default=",".join(arms.DEFAULT_ARMS),
        help=f"comma-separated arms, from {', '.join(arms.ARMS)} (default: {','.join(arms.DEFAULT_ARMS)}; S-cua only when named)",
    )
    ap.add_argument(
        "--questions", default="all", help="comma-separated pilot ids, easy, medium or all (default: all 24)"
    )
    ap.add_argument("--repeats", type=int, default=REPEATS, help=f"repeats per question (default {REPEATS})")
    ap.add_argument("--localmelo", type=Path, default=arms.LOCALMELO, help="the localmelo clone (default: the pinned)")
    ap.add_argument(
        "--localmelo-sha",
        default=arms.LOCALMELO_SHA,
        help="the commit, in full or a prefix of at least 7 digits, that the --localmelo clone must be at "
        f"(default {arms.LOCALMELO_SHA[:7]})",
    )
    ap.add_argument(
        "--localmelo-args",
        default="",
        help="extra flags for every localmelo run, shell-quoted, appended after the arm's own "
        '(e.g. "-X agent.reflection_gate=patient"; one flag without a space needs the = form, --localmelo-args=-v); '
        "not --mcp, --mcp-only, -o, -m, -u or --",
    )
    ap.add_argument(
        "--base-url", default=None, help="the model endpoint, http://host:port/v1 (default: the bench manifest's)"
    )
    ap.add_argument(
        "--expected-chrome-version", default=None, help="refuse any other installed Chrome (default: record it)"
    )
    ap.add_argument(
        "--captcha-pause-s",
        type=float,
        default=CAPTCHA_PAUSE_S,
        help=f"pause before a retry (default {CAPTCHA_PAUSE_S:g})",
    )
    ap.add_argument(
        "--no-gpu-lock", action="store_true", help="do not require the GPU lock (a stub model, not the shared GPU)"
    )
    ap.add_argument(
        "--model",
        default=arms.DEFAULT_MODEL,
        help=f"the model id every harness is told to use (default {arms.DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--claude-bin",
        type=Path,
        default=arms.CLAUDE_BIN,
        help="the Claude Code binary (default: the capability reference run's, backend/vendor/claude-code/2.1.312)",
    )
    ap.add_argument(
        "--cua-space",
        type=int,
        default=None,
        help="S-cua: the desktop Space whose Chrome window the runs drive; required when S-cua is among the arms",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, harness: ModuleType | None = None) -> int:
    args = parse_args(argv)
    try:
        pilot = data.load()
        rows = select_questions(args.questions, pilot)
        harnesses = parse_names(args.harnesses, arms.HARNESSES, "harness")
        arm_names = parse_names(args.arms, arms.ARMS, "arm", arms.DEFAULT_ARMS)
        planned = schedule(rows, args.repeats, harnesses, arm_names)
        localmelo_extra = arms.parse_localmelo_extra(args.localmelo_args)
        if "localmelo" not in harnesses and (localmelo_extra or args.localmelo_sha != arms.LOCALMELO_SHA):
            raise ValueError("--localmelo-args and --localmelo-sha need localmelo among --harnesses")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if harness is None:
        from bench import harness as harness_module

        harness = harness_module
    localmelo = args.localmelo.resolve()
    if args.dry_run:
        with bench_settings(harness, localmelo, args.base_url, args.model, args.claude_bin):
            problems = preflight(
                harness, harnesses, arm_names, localmelo,
                expected_chrome_version=args.expected_chrome_version, expect_sha=args.localmelo_sha,
                gpu_lock=not args.no_gpu_lock, model=args.model, claude_bin=args.claude_bin, cua_space=args.cua_space,
            )  # fmt: skip
            chrome_version = None
            if any(arms.NEEDS_CHROME[a] for a in arm_names):
                with contextlib.suppress(chrome.ChromeError):
                    chrome_version = installed_chrome_version()
            cua_bin, cua_note = None, None
            if "S-cua" in arm_names:  # read-only: which Space is on screen
                try:
                    cua_bin = arms.resolve_cua_bin()
                    space = arms.cua_current_space(cua_bin)
                    wait = "" if space == args.cua_space else f"; a run waits up to {SPACE_WAIT_S:g} s for it"
                    cua_note = f"S-cua: Space on screen {space}, wanted {args.cua_space}{wait}"
                except (ValueError, arms.CuaError) as exc:
                    cua_note = f"S-cua: cua-driver cannot be asked ({exc})"
            snapshot = system_snapshot(
                localmelo if "localmelo" in harnesses else None,
                chrome_version,
                model=args.model,
                claude_bin=args.claude_bin if "claude_code" in harnesses else None,
                cua_bin=cua_bin,
            )
            print_plan(
                args.run_dir.resolve(), planned, rows, args.repeats, harnesses, arm_names, localmelo,
                args.base_url or harness.manifest.OMLX_BASE_URL, problems, snapshot, args.captcha_pause_s,
                model=args.model, claude_bin=args.claude_bin, cua_note=cua_note, localmelo_extra=localmelo_extra,
            )  # fmt: skip
        return 2 if problems else 0
    try:
        return run_study(
            args.run_dir, rows=rows, repeats=args.repeats, harnesses=harnesses, arm_names=arm_names,
            localmelo=localmelo, base_url=args.base_url, expected_chrome_version=args.expected_chrome_version,
            captcha_pause_s=args.captcha_pause_s, gpu_lock=not args.no_gpu_lock, model=args.model,
            claude_bin=args.claude_bin, cua_space=args.cua_space, harness=harness,
            expect_sha=args.localmelo_sha, localmelo_extra=localmelo_extra,
        )  # fmt: skip
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
