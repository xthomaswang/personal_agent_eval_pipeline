"""Memory across chats: one model that does both jobs, or a chat model plus an embedding model?

    cd eval
    uv run python -m memory.across.run --run-dir <dir> --localmelo <checkout> --base-url http://127.0.0.1:8500/v1 --dry-run
    uv run python -m memory.across.run ... --arms L1 --repeats 1     # one arm, once
    uv run python -m memory.across.run ...                           # 4 arms x 3 repeats; run again to resume
    uv run python -m memory.across.run ... --tasks memory/across/data/tasks_v2.yaml --arms M2,B0,N0,FC   # the in-chat kinds
    uv run python -m memory.across.report <dir>                      # report.md and report.json

A cell is one arm in one repeat. Each cell gets a fresh HOME (so fresh memory files), starts its own
`melo serve` on a free port, teaches the facts (each in a new chat), asks the questions (each in a new
chat, deleted afterwards so a question never feeds the next), and stops the gateway. The arms are rotated
across repeats. The arms (ARMS below) are data: the `-X` settings each one passes to `melo serve`.

The checkout must be at LOCALMELO_SHA (or --localmelo-sha) with a clean tree. Both models are served from one
endpoint (--base-url) for all arms, but the gateway is pointed at a timing proxy (proxy.py) that forwards to it
and logs every request. Before each cell the embedding model is unloaded (the chat model never is), and the warm-up
loads only the models the arm uses, so M2's warm-up times the embedding model's load. The driver denies every
tool approval a turn asks for (the chat model trying to save a fact to a file leaves the turn waiting) and stops
a teaching turn after 60 s; both are counted per turn.

A task file of version 2 (``--tasks data/tasks_v2.yaml``, see data_v2.py) selects the other protocol: items that need a
chat of their own. Each kind lists the arms that run it (the KINDS table in data_v2.py), so a cell runs only the items whose
kind lists its arm: a long chat (opener, fact, about 30 filler turns that read files, the question) or teaching chats
and a two-question chat. The long chats are kept (the report reads their window and notes from the cell's SQLite files
after the cell). Turns of a long chat stop after --long-timeout (120 s). An arm of an --arms-file may be a variant of
another (``base``): it runs the kinds, and meets the window rule, that its base's row in the table says.

The operator holds the GPU lock (holder: agent-benchmark); the driver never takes or releases it. It stops
cleanly before the next turn when <dir>/STOP exists, when the lock is gone, or when the battery is low;
running the same command again resumes: finished cells are skipped, a stopped cell continues in its own
HOME, and a cell that died in the middle of a turn is set aside and run again from scratch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import client, data, data_v2, proxy, server

# <repo>/eval/memory/across/run.py: four levels up is the directory that holds the shared lock
GPU_LOCK = Path(__file__).resolve().parents[4] / ".gpu-lock"
LOCK_HOLDER = "agent-benchmark"
BATTERY_START_PCT = 40  # do not start a cell on battery below this
BATTERY_STOP_PCT = 20  # stop before the next turn on battery below this
REPEATS = 3
TEACH_TIMEOUT_S = 60.0  # a teaching turn that takes longer is stopped and counted
LONG_TIMEOUT_S = 120.0  # so is a turn of a long chat (v2)
LOCALMELO_SHA = "5f1498f4"  # the retriever switch, the recall timings, and a chat told not to save stated facts in files
CHAT_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"
PROTOCOL = "across-run-v2"
PROTOCOL_V2 = "across-run-v3"  # a study on a v2 task file; v1 studies keep PROTOCOL and their identity
DEFAULT_EXTRA_BODY = {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    x: tuple[str, ...]  # `-X KEY=VALUE` settings for `melo serve`
    embedding: bool  # whether the gateway is given the embedding model
    v1: bool = True  # whether the arm runs the v1 questions (an in-chat-only arm does not)
    base: str = ""  # tasks_v2: the arm whose row in each kind's arm table this one follows (a variant of M2 says "M2")
    config: tuple[tuple[str, Any], ...] = ()  # dotted config.toml entries the driver writes for the cell, e.g. agent.window_messages

    def settings(self) -> dict[str, Any]:
        return dict(self.config)


# What "memory off" sets. N0 and FC both use it, so a change to what they switch off is made here, once. With
# `memory.across_chats=false` there is no recall (same-chat notes included), no pinned notes and no remember or
# search_memory tool; Arm.embedding=False writes embedding_backend="none", which also switches off the agent loop's
# legacy step memory.
MEMORY_OFF_X = ("memory.across_chats=false",)
# localmelo's WINDOW_MESSAGES_MAX: a window big enough that a long chat evicts nothing. `melo serve -X
# agent.window_messages=N` exits ("only for direct mode"); a gateway session reads its cap from `[agent] window_messages`
# of config.toml, so FC carries it as a config entry, which the driver writes into the cell's config.toml.
WINDOW_MESSAGES_FULL = 256

ARMS: dict[str, Arm] = {
    "M2": Arm("M2", "mixed: chat model + embedding model", ("memory.retriever=embedding",), True),
    "L1": Arm("L1", "one model as two: the chat model picks from BM25 candidates", ("memory.retriever=llm",), False),
    "B0": Arm("B0", "words only: BM25", ("memory.retriever=bm25",), False),
    "N0": Arm("N0", "memory off", MEMORY_OFF_X, False),
    "FC": Arm(
        "FC", "full context: memory off, the whole chat in a 256-message window", MEMORY_OFF_X, False, v1=False,
        config=(("agent.window_messages", WINDOW_MESSAGES_FULL),)),
}
DEFAULT_ARMS = ("M2", "L1", "B0", "N0")
DEFAULT_ARMS_V2 = ("M2", "B0", "N0", "FC")
REPEATS_V2 = 1  # repeats are correlated at temperature 0: v2 spends its budget on items


def memory_is_off(x: Sequence[str]) -> bool:
    """Whether an arm's ``-X`` settings switch memory off (N0, FC)."""
    return set(MEMORY_OFF_X) <= set(x)


def load_arms(path: Path) -> dict[str, Arm]:
    """Arms from a JSON file ``{name: {"label", "x": [...], "embedding": bool, "v1": bool, "base": str, "config": {...}}}``,
    in place of ARMS. ``config`` maps dotted config.toml keys to values, e.g. ``{"agent.window_messages": 256}``.

    An arm that is a variant of another, such as M2 with one more `-X` setting, gives that arm as ``base``: on a v2 task
    file it then runs the kinds, and meets the window rule, that the base's row in the kind's arm table says."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: Arm(
            name, str(row.get("label", name)), tuple(row["x"]), bool(row.get("embedding", False)), bool(row.get("v1", True)),
            str(row.get("base", "")), tuple(sorted(dict(row.get("config") or {}).items())))
        for name, row in raw.items()
    }


# --- files ---


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


class Journal:
    """The turns of one cell, one JSON line each: a ``start`` before a message is sent, a ``turn`` after it ends."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a line cut short by a crash
        return rows

    def append(self, row: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def journal_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a journal says: finished turns by (phase, item), deleted sessions, and starts with no result."""
    turns = {(r["phase"], r["item"]): r for r in rows if r.get("event") == "turn"}
    started = {(r["phase"], r["item"]) for r in rows if r.get("event") == "start"}
    return {
        "turns": turns,
        "dangling": sorted(started - set(turns)),
        "deleted": {r["session_id"] for r in rows if r.get("event") == "deleted" and r.get("ok")},
    }


# --- schedule ---


def rotated(items: Sequence[str], k: int) -> list[str]:
    k %= len(items)
    return [*items[k:], *items[:k]]


def schedule(arms: Sequence[str], repeats: int) -> list[dict[str, Any]]:
    """Repeat-major cells; the arms of repeat r start at arm number (r - 1) mod len(arms)."""
    if repeats < 1 or not arms:
        raise ValueError("repeats must be at least 1 and arms must not be empty")
    if len(set(arms)) != len(arms):
        raise ValueError(f"repeated arm in {list(arms)}")
    cells: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for position, arm in enumerate(rotated(list(arms), repeat - 1), 1):
            cells.append({
                "index": len(cells) + 1, "repeat": repeat, "arm": arm, "position": position,
                "cell_id": f"r{repeat}__{arm}",
            })
    return cells


# --- guards ---


def lock_problem(lock: Path, holder: str = LOCK_HOLDER) -> str | None:
    """The operator's GPU lock must exist with a `holder: <holder>` line."""
    try:
        lines = lock.read_text().splitlines()
    except FileNotFoundError:
        return f"the GPU lock is absent; the operator takes it first ({lock.name})"
    if not any(line.startswith(f"holder: {holder}") for line in lines):
        other = next((line for line in lines if line.startswith("holder:")), "no holder line")
        return f"the GPU lock is held by another session ({other})"
    return None


def battery_state() -> tuple[bool, int | None]:
    """(on battery power, charge in percent) from `pmset -g batt`; (False, None) when unknown."""
    text = server.out(["pmset", "-g", "batt"])
    found = re.search(r"(\d+)%", text)
    return "'Battery Power'" in text, int(found.group(1)) if found else None


def battery_problem(low_pct: int, state: Callable[[], tuple[bool, int | None]] = battery_state) -> str | None:
    on_battery, pct = state()
    if on_battery and pct is not None and pct < low_pct:
        return f"on battery at {pct}% (below {low_pct}%)"
    return None


def make_guards(
    run_dir: Path, lock: Path = GPU_LOCK, holder: str = LOCK_HOLDER,
    start_pct: int = BATTERY_START_PCT, stop_pct: int = BATTERY_STOP_PCT,
) -> tuple[Callable[[], str | None], Callable[[], str | None]]:
    """(turn guard, cell guard): why the driver must not send the next turn, or not start the next cell."""
    def turn_guard() -> str | None:
        if (run_dir / "STOP").exists():
            return "STOP file found"
        return lock_problem(lock, holder) or battery_problem(stop_pct)

    def cell_guard() -> str | None:
        return turn_guard() or battery_problem(start_pct)

    return turn_guard, cell_guard


# --- the study ---


@dataclass
class Study:
    run_dir: Path
    tasks: data.TaskSet | data_v2.TaskSet
    arms: dict[str, Arm]
    arm_names: tuple[str, ...]
    repeats: int
    localmelo: Path
    expect_sha: str
    base_url: str
    model: str
    embedding_model: str = EMBEDDING_MODEL
    extra_body: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_EXTRA_BODY))
    question_ids: tuple[str, ...] = ()  # empty: all
    keep_ask_chats: bool = False
    turn_timeout_s: float = 900.0
    teach_timeout_s: float = TEACH_TIMEOUT_S
    long_timeout_s: float = LONG_TIMEOUT_S  # v2: every turn of a long chat
    cold_start: bool = True  # unload the embedding model before every cell, so M2's warm-up times its load
    unload_timeout_s: float = 60.0
    use_proxy: bool = True  # route the gateway's model calls through the timing proxy
    poll_s: float = 0.25  # how often to look at a turn when there is no event stream
    use_stream: bool = True
    do_warmup: bool = True
    scratch: Path | None = None  # where the preflight makes its throwaway HOME (default: the system temp dir)
    # seams, replaced in tests
    launcher: Callable[[server.LaunchSpec], Any] = server.launch
    turn_guard: Callable[[], str | None] = lambda: None
    cell_guard: Callable[[], str | None] = lambda: None
    probe_rss: Callable[[], dict[str, Any]] | None = None
    warm: Callable[[Study, Arm], dict[str, Any]] | None = None
    snapshot: Callable[[Study], dict[str, Any]] | None = None
    sleep: Callable[[float], None] = time.sleep

    @property
    def melo(self) -> Path:
        return self.localmelo / ".venv" / "bin" / "melo"

    @property
    def v2(self) -> bool:
        """Whether the task file is a v2 one (items, not facts and questions)."""
        return isinstance(self.tasks, data_v2.TaskSet)

    def asked(self) -> tuple[data.Question, ...]:
        order = self.tasks.ask_order()
        return tuple(q for q in order if not self.question_ids or q.id in self.question_ids)

    def taught(self) -> tuple[data.Fact, ...]:
        """The facts to teach, in teaching order: those the selected questions need, and every filler."""
        if not self.question_ids:
            return self.tasks.facts
        need = {f for q in self.asked() for f in (*q.facts, *q.wrong_facts)}
        return tuple(f for f in self.tasks.facts if f.id in need or f.role == "filler")

    def items(self, arm: str | None = None) -> tuple[data_v2.Item, ...]:
        """v2: the selected items in the order a cell runs them (kinds in the KINDS table's order, items in the task
        file's fixed shuffle). With ``arm``, only the items whose kind lists that arm."""
        order = self.tasks.order()
        base = (self.arms[arm].base or arm) if arm in self.arms else arm
        return tuple(
            i for kind, spec in data_v2.KINDS.items() for i in order
            if i.kind == kind and (not self.question_ids or i.id in self.question_ids) and (arm is None or base in spec.arms))

    def rss(self) -> dict[str, Any]:
        """The model server process's rss and loaded models now."""
        return self.probe_rss() if self.probe_rss else server.model_server_state(self.base_url)

    def identity(self) -> dict[str, Any]:
        doc = {
            "protocol": PROTOCOL,
            "tasks_sha256": self.tasks.sha256(),
            "arms": {a: {"x": list(self.arms[a].x), "embedding": self.arms[a].embedding, **({"base": self.arms[a].base} if self.arms[a].base else {}),
                    **({"config": self.arms[a].settings()} if self.arms[a].config else {})} for a in self.arm_names},
            "repeats": self.repeats,
            "questions": [q.id for q in self.asked()] if not self.v2 else [i.id for i in self.items()],
            "localmelo_sha": server.git_sha(self.localmelo),
            "base_url": self.base_url,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "extra_body": self.extra_body,
            "keep_ask_chats": self.keep_ask_chats,
            "teach_timeout_s": self.teach_timeout_s,
            "cold_start": self.cold_start,
            "use_proxy": self.use_proxy,
        }
        if self.v2:  # a v1 study's identity stays what it was
            doc.update(protocol=PROTOCOL_V2, long_timeout_s=self.long_timeout_s)
        return doc


def preflight(study: Study) -> list[str]:
    """Everything that would make the study invalid, as messages."""
    problems: list[str] = []
    sha = server.git_sha(study.localmelo)
    if not study.expect_sha or not sha.startswith(study.expect_sha):
        problems.append(f"localmelo at {study.localmelo.name} is {sha[:10] or 'not a git checkout'}, not {study.expect_sha or '(no sha given)'}")
    elif server.git_dirty(study.localmelo):
        problems.append(f"localmelo at {study.localmelo.name} has uncommitted changes")
    if not study.melo.exists():
        problems.append(f"{study.melo} does not exist (run uv sync in the checkout)")
    else:
        with tempfile.TemporaryDirectory(dir=study.scratch) as probe_home:
            for arm in (study.arms[a] for a in study.arm_names):
                for key in [*(pair.partition("=")[0] for pair in arm.x), *(k for k, _ in arm.config)]:
                    if not _config_get_ok(study, Path(probe_home), key):
                        problems.append(f"arm {arm.name}: this localmelo build does not know the setting {key}")
    try:
        ids, _ = server.model_ids(study.base_url)
    except (OSError, ValueError) as exc:
        ids = []
        problems.append(f"{study.base_url}/models is not readable: {exc}")
    else:
        if study.model not in ids:
            problems.append(f"chat model {study.model!r} is not listed by {study.base_url}/models")
        if any(study.arms[a].embedding for a in study.arm_names) and study.embedding_model not in ids:
            problems.append(f"embedding model {study.embedding_model!r} is not listed by {study.base_url}/models (arm M2 needs it)")
    known = {i.id for i in study.tasks.items} if study.v2 else {x.id for x in study.tasks.questions}
    problems += [f"selected question {q} does not exist" for q in study.question_ids if q not in known]
    for name in study.arm_names:
        if study.v2 and not study.items(name):
            kinds = sorted(k for k, spec in data_v2.KINDS.items() if (study.arms[name].base or name) in spec.arms)
            problems.append(f"arm {name} has nothing to run here: the selected items are of kinds that do not list it (it runs {kinds or 'no kind'})")
        elif not study.v2 and not study.arms[name].v1:
            problems.append(f"arm {name} runs only the in-chat kinds of a v2 task file, not the v1 questions")
    return problems


def _config_get_ok(study: Study, probe_home: Path, key: str) -> bool:
    """`melo config get KEY` on a throwaway HOME: whether the build knows the setting."""
    try:
        done = subprocess.run(
            [str(study.melo), "config", "get", key], capture_output=True, text=True, timeout=60, check=False,
            stdin=subprocess.DEVNULL, env=server.serve_env(probe_home),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def system_snapshot(study: Study) -> dict[str, Any]:
    """The macOS and Xcode builds, the localmelo build and the model server (models listed, process, rss)."""
    snap: dict[str, Any] = {
        "timestamp": time.time(),
        "sw_vers": server.out(["sw_vers"]),
        "xcodebuild_version": server.out(["xcodebuild", "-version"]),
        "localmelo_sha": server.git_sha(study.localmelo),
        "pmset_batt": server.out(["pmset", "-g", "batt"]),
        "python": sys.version.split()[0],
    }
    try:
        ids, listing = server.model_ids(study.base_url)
        snap["model_server"] = {"base_url": study.base_url, "models": ids, "models_response": listing, **server.model_server_state(study.base_url)}
    except (OSError, ValueError) as exc:
        snap["model_server"] = {"base_url": study.base_url, "error": str(exc)}
    return snap


# --- one cell ---


def prepare_cell_dir(cell_dir: Path) -> bool:
    """True to continue the cell in its HOME. A cell that died mid-turn is set aside; a new one starts empty."""
    info_path = cell_dir / "cell.json"
    if cell_dir.exists():
        state = journal_state(Journal(cell_dir / "turns.jsonl").read())
        if info_path.exists() and (cell_dir / "home").is_dir() and not state["dangling"]:
            return True
        n = 1
        while (aside := cell_dir.with_name(f"{cell_dir.name}.aborted-{n}")).exists():
            n += 1
        cell_dir.rename(aside)
    cell_dir.mkdir(parents=True)
    return False


def turn_key(cell: Mapping[str, Any], item: str) -> str:
    return f"acm-r{cell['repeat']}-{re.sub('[^A-Za-z0-9]', '', cell['arm'])}-{item}"


def turn_record(cell: Mapping[str, Any], phase: str, item: str, key: str, res: client.TurnResult | None, error: str = "") -> dict[str, Any]:
    base = {"event": "turn", "cell": cell["cell_id"], "phase": phase, "item": item, "key": key, "at": now_iso()}
    if res is None:
        return {**base, "state": "failed", "error": error, "timed_out": False, "approvals_denied": 0, "denied_tools": []}
    memory = res.memory()
    ok = res.status == "completed"
    return {
        **base,
        "state": "finished" if ok else "failed",
        "status": res.status, "cause": res.cause, "error": res.error or error,
        "session_id": res.session_id, "turn_id": res.turn_id, "answer": res.answer,
        "wall_ms": res.wall_ms, "ttfe_ms": res.ttfe_ms, "wall_source": res.wall_source,
        "started_epoch": res.started_epoch, "ended_epoch": res.ended_epoch,
        "server_ms": res.server_ms(), "usage": res.usage,
        "timed_out": res.status == "timeout", "cancelled": res.cancelled,
        "approvals_denied": len(res.denied), "denied_tools": [d["tool"] for d in res.denied],
        "memory": memory, "recall": (memory or {}).get("recall") or res.memory_recall,
        "event_types": [e.get("type") for e in res.events],
    }


def run_memory_stats(res: client.TurnResult) -> dict[str, float]:
    """The numeric counters of ``run.finished``'s ``result.memory_stats``, which localmelo sends in its explicit
    reasoning modes only; {} when it is absent (the default legacy mode). The recall log and other lists are left out."""
    event = res.event("run.finished")
    stats = (((event or {}).get("data") or {}).get("result") or {}).get("memory_stats")
    if not isinstance(stats, Mapping):
        return {}
    return {k: v for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def cell_counts(rows: Sequence[Mapping[str, Any]], phases: Sequence[str] = ("teach", "ask")) -> dict[str, dict[str, int]]:
    """Per phase: turns, failed turns, turns stopped by the timeout, approvals denied, and turns that had one."""
    out = {}
    for phase in phases:
        turns = [r for r in rows if r.get("event") == "turn" and r.get("phase") == phase]
        out[phase] = {
            "turns": len(turns), "failed": sum(1 for r in turns if r.get("state") != "finished"),
            "timed_out": sum(1 for r in turns if r.get("timed_out")),
            "approvals_denied": sum(int(r.get("approvals_denied") or 0) for r in turns),
            "turns_with_denied": sum(1 for r in turns if r.get("approvals_denied")),
        }
    return out


def delete_chat(gw: client.Gateway, session_id: str, sleep: Callable[[float], None], attempts: int = 5) -> bool:
    for attempt in range(attempts):
        try:
            gw.delete_session(session_id)
            return True
        except client.GatewayError as exc:
            if exc.status == 404:
                return True
            sleep(0.5 * (attempt + 1))  # a delete right after a turn can meet 409 session_busy
        except (OSError, http.client.HTTPException):
            sleep(0.5)
    return False


# --- v2: the kinds that need a chat of their own ---


@dataclass
class Live:
    """A cell that is running: what the runners of the v2 kinds use."""

    study: Study
    arm: Arm
    cell: Mapping[str, Any]
    gw: client.Gateway
    journal: Journal
    work: Path
    done: dict[str, Any]  # journal_state at the start of this run of the cell
    send: Callable[..., dict[str, Any]]


def prepare_workspace(tasks: data_v2.TaskSet, path: Path) -> None:
    """The fixture files of the task file in ``path``, so file_read and list_files have something to read.

    A file that is there already is left as it is (a long chat that a stop interrupted continues in its workspace)."""
    path.mkdir(parents=True, exist_ok=True)
    for fixture in tasks.fixtures:
        target = path / fixture.path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fixture.text, encoding="utf-8")


def settle(gw: client.Gateway, session_id: str, sleep: Callable[[float], None], attempts: int = 120) -> None:
    """Wait until the chat's newest turn is no longer running: a turn that was cancelled needs a moment to stop,
    and the next message of the chat is refused until it has."""
    for _ in range(attempts):
        try:
            row = gw.last_turn(session_id)
        except (client.GatewayError, OSError, http.client.HTTPException):
            row = None
        if row is None or row.get("status") != "running":
            return
        sleep(0.5)


def done_session(done: Mapping[str, Any], steps: Sequence[data_v2.Step]) -> str | None:
    """The chat the journal already holds for these steps (the newest finished turn's), or None."""
    sessions = [r.get("session_id") for step in steps if (r := done["turns"].get((step.phase, step.label)))]
    return next((s for s in reversed(sessions) if s), None)


def run_longrange(live: Live, items: Sequence[data_v2.Item]) -> bool:
    """One chat per item: opener, fact, filler, the question. True when a guard stopped the cell.

    A chat that a stop interrupted continues in its own session and workspace. A chat whose first message failed to
    start one is left as it is (the question is then never asked). The chats are not deleted: the report reads each
    chat's window and notes from the cell's SQLite files."""
    study = live.study
    for item in items:
        steps = study.tasks.long_steps(item)
        workspace = live.work / item.id
        session = done_session(live.done, steps)
        for number, step in enumerate(steps):
            if (step.phase, step.label) in live.done["turns"]:
                continue
            if session is None and number:
                break  # the chat never started
            if study.turn_guard():
                return True
            if number == 0:
                prepare_workspace(study.tasks, workspace)
            row = live.send(
                step.phase, step.label, step.text, session_id=session, workspace=workspace, timeout_s=study.long_timeout_s)
            session = row.get("session_id") or session
            if row.get("timed_out") and session:
                settle(live.gw, session, study.sleep)
    return False


def run_followup(live: Live, items: Sequence[data_v2.Item]) -> bool:
    """Each item's two facts, each in its own new chat; then one new chat per item with the two questions, deleted
    afterwards (unless --keep-ask-chats). True when a guard stopped the cell."""
    study = live.study
    tasks = study.tasks
    if not study.keep_ask_chats:  # a crash between the last question of a chat and its delete
        for item in items:
            asked = [live.done["turns"].get(("ask", a.id)) for a in item.asks]
            session = next((r.get("session_id") for r in asked if r and r.get("session_id")), None)
            if all(asked) and session and session not in live.done["deleted"]:
                ok = delete_chat(live.gw, session, study.sleep)
                live.journal.append({"event": "deleted", "item": item.id, "session_id": session, "ok": ok})
    for item in items:
        for step in tasks.teach_steps(item):
            if (step.phase, step.label) in live.done["turns"]:
                continue
            if study.turn_guard():
                return True
            live.send(step.phase, step.label, step.text)
    for item in items:
        steps = tasks.ask_steps(item)
        session = done_session(live.done, steps)
        if all((step.phase, step.label) in live.done["turns"] for step in steps):
            continue
        for number, step in enumerate(steps):
            if (step.phase, step.label) in live.done["turns"]:
                continue
            if study.turn_guard():
                return True
            if session is None and number:  # the first question started no chat: the follow-up has nothing to follow
                live.journal.append(turn_record(
                    live.cell, step.phase, step.label, turn_key(live.cell, step.label), None, "the first question started no chat"))
                continue
            row = live.send(step.phase, step.label, step.text, session_id=session)
            session = row.get("session_id") or session
        if session and not study.keep_ask_chats:
            ok = delete_chat(live.gw, session, study.sleep)
            live.journal.append({"event": "deleted", "item": item.id, "session_id": session, "ok": ok})
    return False


# one runner per kind of data_v2.KINDS: (live cell, the items of that kind this arm runs) -> stopped by a guard
KIND_RUNNERS: dict[str, Callable[[Live, Sequence[data_v2.Item]], bool]] = {"longrange": run_longrange, "followup": run_followup}


def run_items(live: Live) -> bool:
    """Run the kinds one after the other, each on the items of its kind that list this cell's arm."""
    for kind in data_v2.KINDS:
        items = [i for i in live.study.items(live.arm.name) if i.kind == kind]
        if items and KIND_RUNNERS[kind](live, items):
            return True
    return False


def run_cell(study: Study, cell: Mapping[str, Any]) -> str:
    """Run one cell to the end ("finished") or to a guard ("stopped"). Raises server.LaunchError on a dead gateway."""
    arm = study.arms[cell["arm"]]
    cell_dir = study.run_dir / "cells" / cell["cell_id"]
    resume = prepare_cell_dir(cell_dir)
    journal = Journal(cell_dir / "turns.jsonl")
    home, work = cell_dir / "home", cell_dir / "work"
    embedding_model = study.embedding_model if arm.embedding else None  # only M2 has (and loads) the second model
    if resume:
        info = json.loads((cell_dir / "cell.json").read_text())
        info["resumed"] = info.get("resumed", 0) + 1
    else:
        home.mkdir()
        work.mkdir()
        info = {
            "cell": dict(cell), "arm": arm.name, "x": list(arm.x), "embedding": arm.embedding,
            "home": "home", "started_at": now_iso(), "state": "started", "rss": {},
            **({"config": arm.settings()} if arm.config else {}),
        }
    info["state"] = "started"
    save(cell_dir / "cell.json", info)
    # Each step runs once per cell, in this order, and is kept in cell.json: a cell that stopped before one of them
    # does it when it continues. The embedding model is unloaded first (never the chat model) so every cell starts
    # without it; only M2 loads it again, and its warm-up then times the load.
    if "cold_start" not in info:
        info["cold_start"] = (
            server.ensure_unloaded(study.base_url, study.embedding_model, timeout=study.unload_timeout_s, sleep=study.sleep)
            if study.cold_start else {"model": study.embedding_model, "skipped": True})
        save(cell_dir / "cell.json", info)
    if "before" not in info["rss"]:
        info["rss"]["before"] = study.rss()
    if "warmup" not in info:
        info["warmup"] = (study.warm(study, arm) if study.warm else server.warmup(
            study.base_url, study.model, embedding_model)) if study.do_warmup else None
        if info["warmup"] is not None:
            info["rss"]["after_warmup"] = study.rss()
        save(cell_dir / "cell.json", info)
    started = time.monotonic()

    timing = proxy.TimingProxy(study.base_url, cell_dir / "proxy.jsonl") if study.use_proxy else None
    spec = server.LaunchSpec(study.melo, home, server.free_port(), arm.x, cell_dir / "gateway.log")
    running = None
    stopped = False
    try:
        # the gateway talks to the proxy when there is one; the port is new at every start, so the config is too
        server.write_config(home, server.config_document(
            timing.url if timing else study.base_url, study.model, embedding_model, study.extra_body, arm.settings()))
        running = study.launcher(spec)
        gw = server.wait_ready(running)
        info["port"] = spec.port
        if timing:
            info["proxy"] = {"log": "proxy.jsonl", "upstream": study.base_url}
        done = journal_state(journal.read())

        seen: dict[str, int] = {}  # the newest event id read per chat: the next turn's stream starts after it

        def send(
            phase: str, item: str, text: str, *, session_id: str | None = None, workspace: Path | None = None,
            timeout_s: float | None = None,
        ) -> dict[str, Any]:
            """One message: a new chat (in the shared workspace, or ``workspace``), or ``session_id``'s next turn."""
            key = turn_key(cell, item)
            journal.append({"event": "start", "phase": phase, "item": item, "key": key, "at": now_iso()})
            try:
                res = client.run_turn(
                    gw, text, key, session_id=session_id, workspace=str((workspace or work).resolve()), poll_s=study.poll_s,
                    use_stream=study.use_stream, after=seen.get(session_id or "", 0),
                    timeout_s=timeout_s or (study.teach_timeout_s if phase == "teach" else study.turn_timeout_s))
            except (client.GatewayError, OSError, http.client.HTTPException, ValueError) as exc:
                if not running.alive():
                    raise server.LaunchError(f"the gateway died during {item}: {exc}") from exc
                row = turn_record(cell, phase, item, key, None, f"{type(exc).__name__}: {exc}")
            else:
                seen[res.session_id] = max(seen.get(res.session_id, 0), res.last_event_id)
                row = turn_record(cell, phase, item, key, res)
                if study.v2:  # the tools a turn used, to check that a long chat stayed with the read-tier ones
                    row["tools"] = [(e.get("data") or {}).get("name") for e in res.events if e.get("type") == "tool.started"]
                    if stats := run_memory_stats(res):
                        row["memory_stats"] = stats
            journal.append(row)
            return row

        def teach() -> bool:
            """Send the teaching messages not sent yet; True when a guard stopped it."""
            for fact in study.taught():
                if ("teach", fact.id) in done["turns"]:
                    continue
                if study.turn_guard():
                    return True
                send("teach", fact.id, fact.text)
            return False

        def ask() -> bool:
            """Ask the questions not asked yet, deleting each chat; True when a guard stopped it."""
            if not study.keep_ask_chats:  # a crash between a question and its delete
                for row in done["turns"].values():
                    if row["phase"] == "ask" and row.get("session_id") and row["session_id"] not in done["deleted"]:
                        ok = delete_chat(gw, row["session_id"], study.sleep)
                        journal.append({"event": "deleted", "item": row["item"], "session_id": row["session_id"], "ok": ok})
            for q in study.asked():
                if ("ask", q.id) in done["turns"]:
                    continue
                if study.turn_guard():
                    return True
                row = send("ask", q.id, q.text)
                if row.get("session_id") and not study.keep_ask_chats:
                    ok = delete_chat(gw, row["session_id"], study.sleep)
                    journal.append({"event": "deleted", "item": q.id, "session_id": row["session_id"], "ok": ok})
            return False

        if study.v2:
            stopped = run_items(Live(study, arm, cell, gw, journal, work, done, send))
        else:
            stopped = teach()
            if not stopped:
                info["rss"].setdefault("after_teach", study.rss())
                stopped = ask()
    finally:
        if running is not None:
            info["gateway_exit"] = running.stop()
        if timing:
            timing.stop()
        info["wall_s"] = round(info.get("wall_s", 0.0) + time.monotonic() - started, 1)
        if not stopped:
            info["rss"]["after"] = study.rss()
        info["counts"] = cell_counts(journal.read(), ("teach", "long", "ask") if study.v2 else ("teach", "ask"))
        save(cell_dir / "cell.json", info)
    if stopped:
        info["state"] = "paused"
        save(cell_dir / "cell.json", info)
        return "stopped"
    shutil.rmtree(home / ".cache" / "localmelo" / "gateway" / "auth", ignore_errors=True)  # the token stays out of run records
    info["state"] = "finished"
    info["finished_at"] = now_iso()
    save(cell_dir / "cell.json", info)
    return "finished"


# --- running it ---


def log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(f"{now_iso()} {text}\n")


def load_state(study: Study, planned: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = study.run_dir / "across.json"
    config = study.identity()
    digest = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config or state.get("schedule_sha256") != digest:
            raise ValueError("the study's config, schedule, task set or localmelo build changed; use a fresh directory")
        return state
    if study.run_dir.exists() and any(study.run_dir.iterdir()):
        raise ValueError(f"{study.run_dir} is not empty and has no across.json checkpoint; use a fresh directory")
    state = {"config": config, "schedule_sha256": digest, "cells": {}, "complete": False}
    save(study.run_dir / "schedule.json", planned)
    if study.v2:  # the items in the order a cell runs them, and the arms each kind lists
        save(study.run_dir / "tasks.json", {
            "protocol": study.tasks.protocol, "tasks_sha256": study.tasks.sha256(), "ask_seed": study.tasks.ask_seed,
            "kinds": {k: dict(spec.arms) for k, spec in data_v2.KINDS.items()},
            "items": [{"id": i.id, "kind": i.kind, "lang": i.lang, "note_expected": i.note_expected} for i in study.items()],
        })
    else:
        save(study.run_dir / "tasks.json", {  # the teaching order (an old value before its replacement) and the ask order
            "protocol": study.tasks.protocol, "tasks_sha256": study.tasks.sha256(), "ask_seed": study.tasks.ask_seed,
            "teach_order": [{"id": f.id, "order": f.order, "block": f.block, "role": f.role, "supersedes": f.supersedes} for f in study.taught()],
            "ask_order": [q.id for q in study.asked()],
        })
    save(checkpoint, state)
    return state


def run_study(study: Study) -> int:
    """0: every cell finished. 2: refused to start. 3: stopped early (run again to resume). 1: a gateway failed."""
    study.run_dir = study.run_dir.resolve()
    study.localmelo = study.localmelo.resolve()
    study.run_dir.mkdir(parents=True, exist_ok=True)
    planned = schedule(study.arm_names, study.repeats)
    problems = preflight(study)
    if problems:
        print("Preflight failed:", *problems, sep="\n  - ", file=sys.stderr)
        return 2
    try:
        state = load_state(study, planned)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    log_path = study.run_dir / "across.log"
    env_path = study.run_dir / "environment.json"
    env_doc = json.loads(env_path.read_text()) if env_path.exists() else {}
    env_doc.setdefault("start", (study.snapshot or system_snapshot)(study))
    save(env_path, env_doc)
    checkpoint = study.run_dir / "across.json"
    code = 0
    for cell in planned:
        cid = cell["cell_id"]
        if state["cells"].get(cid, {}).get("state") == "finished":
            continue
        reason = study.cell_guard()
        if reason:
            log(log_path, f"== {reason} before {cid}; stopping")
            code = 3
            break
        log(log_path, f"[{cell['index']}/{len(planned)}] {cid}: start")
        try:
            outcome = run_cell(study, cell)
        except server.ColdStartError as exc:
            log(log_path, f"{cid}: {exc}")
            print(f"Preflight failed: {cid} cannot start cold for the embedding model:\n  - {exc}", file=sys.stderr)
            state["cells"][cid] = {"state": "error", "arm": cell["arm"], "repeat": cell["repeat"], "error": str(exc)}
            save(checkpoint, state)
            code = 2
            break
        except server.LaunchError as exc:
            log(log_path, f"{cid}: {exc}")
            print(f"{cid}: {exc}", file=sys.stderr)
            state["cells"][cid] = {"state": "error", "arm": cell["arm"], "repeat": cell["repeat"], "error": str(exc)}
            save(checkpoint, state)
            code = 1
            break
        state["cells"][cid] = {"state": "finished" if outcome == "finished" else "paused", "arm": cell["arm"], "repeat": cell["repeat"]}
        save(checkpoint, state)
        log(log_path, f"{cid}: {outcome}")
        if outcome == "stopped":
            log(log_path, f"== {study.turn_guard() or 'stopped'} during {cid}; stopping")
            code = 3
            break
    env_doc["end"] = (study.snapshot or system_snapshot)(study)
    save(env_path, env_doc)
    state["complete"] = all(state["cells"].get(c["cell_id"], {}).get("state") == "finished" for c in planned)
    save(checkpoint, state)
    done = sum(1 for c in state["cells"].values() if c.get("state") == "finished")
    print(f"{done}/{len(planned)} cells finished in {study.run_dir}" + (" (stopped early; see across.log)" if code == 3 else ""))
    return code


# --- command line ---


def v2_turns(study: Study, arm: str) -> dict[str, int]:
    """The turns one cell of ``arm`` sends on a v2 task file, by phase."""
    tasks, turns = study.tasks, {"long": 0, "teach": 0, "ask": 0}
    for item in study.items(arm):
        if item.kind == "longrange":
            steps = tasks.long_steps(item)
            turns["long"] += len(steps) - len(item.asks)
            turns["ask"] += len(item.asks)
        else:
            turns["teach"] += len(tasks.teach_steps(item))
            turns["ask"] += len(tasks.ask_steps(item))
    return turns


def print_plan(study: Study, planned: list[dict[str, Any]], problems: list[str]) -> None:
    if study.v2:
        per_arm = {a: v2_turns(study, a) for a in study.arm_names}
        total = sum(sum(per_arm[c["arm"]].values()) for c in planned)
        print(
            f"Memory-bench-v2 plan: {len(planned)} cells = {len(study.arm_names)} arms x {study.repeats} repeats, "
            f"{len(study.items())} items ({', '.join(f'{k} {sum(1 for i in study.items() if i.kind == k)}' for k in data_v2.KINDS)}), {total} turns in all."
        )
        for name in study.arm_names:
            t = per_arm[name]
            kinds = sorted({i.kind for i in study.items(name)})
            print(f"  {name}: {sum(t.values())} turns ({t['long']} in long chats, {t['teach']} teaching, {t['ask']} questions); kinds {', '.join(kinds) or 'none'}")
        print(f"Ask chats of followup items are {'kept' if study.keep_ask_chats else 'deleted after the item'}; long chats are kept.")
    else:
        asked, taught = study.asked(), study.taught()
        turns = len(taught) + len(asked)
        print(
            f"Memory-across-chats plan: {len(planned)} cells = {len(study.arm_names)} arms x {study.repeats} repeats; "
            f"each teaches {len(taught)} facts and asks {len(asked)} questions ({turns} turns), "
            f"{len(planned) * turns} turns in all. Ask chats are {'kept' if study.keep_ask_chats else 'deleted after each answer'}."
        )
    print(f"Output: {study.run_dir} (across.json, schedule.json, tasks.json, environment.json, cells/)")
    print(f"Every cell starts with {study.embedding_model} unloaded (the chat model is never unloaded): "
          + (f"asked to unload through the server, {study.unload_timeout_s:g} s to finish." if study.cold_start else "NOT done (--no-cold-start)."))
    print(f"Teaching turns stop after {study.teach_timeout_s:g} s, questions after {study.turn_timeout_s:g} s"
          + (f", turns of a long chat after {study.long_timeout_s:g} s" if study.v2 else "") + "; every approval request is denied and counted. "
          f"Model calls {'go through the timing proxy (cells/<cell>/proxy.jsonl)' if study.use_proxy else 'go straight to the model server'}.")
    for name in study.arm_names:
        arm = study.arms[name]
        embedding = f" (+ {study.embedding_model})" if arm.embedding else ""
        config = f" [config.toml: {', '.join(f'{k}={v}' for k, v in arm.config)}]" if arm.config else ""
        print(f"Arm {name}: melo serve {' '.join(f'-X {x}' for x in arm.x)}{config}{embedding}  # {arm.label}")
    for cell in planned:
        print(f"{cell['index']:3d}. {cell['cell_id']}")
    state_path = study.run_dir / "across.json"
    if state_path.exists():
        cells = json.loads(state_path.read_text()).get("cells", {})
        print(f"Checkpoint: {sum(1 for c in cells.values() if c.get('state') == 'finished')}/{len(planned)} cells finished; a run resumes there")
    print(f"Model server {study.base_url}: chat {study.model}" + (f", embedding {study.embedding_model}" if any(study.arms[a].embedding for a in study.arm_names) else ""))
    print("Preflight: " + ("ok" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"  - {problem}")
    print(f"Guards: {study.cell_guard() or 'ok'}; STOP file: {'present' if (study.run_dir / 'STOP').exists() else 'absent'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="output directory; running again there resumes from across.json")
    ap.add_argument("--localmelo", type=Path, required=True, help="the localmelo checkout (its .venv/bin/melo is used; it is never changed)")
    ap.add_argument("--localmelo-sha", default=LOCALMELO_SHA, help="the commit prefix the checkout must be at, with a clean tree (default %(default)s)")
    ap.add_argument("--base-url", required=True, help="the model server's URL up to /v1, e.g. http://127.0.0.1:8500/v1")
    ap.add_argument("--model", default=CHAT_MODEL, help="the chat model id as the server lists it (default %(default)s)")
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="the embedding model id, listed by the same server, for arms with embedding (default %(default)s)")
    ap.add_argument("--arms", default=None, help=f"comma-separated arms (default {','.join(DEFAULT_ARMS)} on tasks_v1, {','.join(DEFAULT_ARMS_V2)} on tasks_v2)")
    ap.add_argument("--arms-file", type=Path, help="JSON file that replaces the built-in arms: {name: {label, x: [KEY=VALUE...], embedding}}")
    ap.add_argument("--repeats", type=int, default=None, help=f"repeats per arm (default {REPEATS} on tasks_v1, {REPEATS_V2} on tasks_v2)")
    ap.add_argument("--tasks", type=Path, default=data.DATA_FILE, help="the task file: tasks_v1.yaml (default) or tasks_v2.yaml, which selects the v2 protocol")
    ap.add_argument(
        "--questions", default="",
        help="comma-separated question ids to ask (default all); teaches only the facts they need, plus the fillers. "
        "On tasks_v2: item ids (L01,P01...), e.g. two per kind for a smoke run")
    ap.add_argument("--extra-body", default=json.dumps(DEFAULT_EXTRA_BODY), help="JSON for [vllm.extra_body] of the gateway config (default %(default)s)")
    ap.add_argument("--keep-ask-chats", action="store_true", help="keep each question's chat (default: delete it after reading the answer, so questions do not feed each other)")
    ap.add_argument("--turn-timeout", type=float, default=900.0, help="seconds a question's turn may take (default %(default)s)")
    ap.add_argument("--teach-timeout", type=float, default=TEACH_TIMEOUT_S, help="seconds a teaching turn may take before it is stopped and counted (default %(default)s)")
    ap.add_argument("--long-timeout", type=float, default=LONG_TIMEOUT_S, help="tasks_v2: seconds a turn of a long chat may take, the question included, before it is stopped and counted (default %(default)s)")
    ap.add_argument(
        "--admin-token-cmd",
        default=os.environ.get("BENCH_UPSTREAM_TOKEN_CMD"),
        help="a command that prints the model server's API key, for its admin routes (status, unload); run once, "
        "never recorded (default: $BENCH_UPSTREAM_TOKEN_CMD)",
    )
    ap.add_argument("--no-cold-start", action="store_true", help="do not unload the embedding model before a cell (the report then says cells did not start cold)")
    ap.add_argument("--unload-timeout", type=float, default=60.0, help="seconds to wait for the embedding model to unload before a cell (default %(default)s)")
    ap.add_argument("--no-proxy", action="store_true", help="point the gateway at the model server directly (no per-request timing log, no chat TTFT)")
    ap.add_argument("--no-stream", action="store_true", help="poll for the end of a turn instead of following the event stream (no time to first event)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the one-token requests that load the models before a cell")
    ap.add_argument("--gpu-lock", type=Path, default=GPU_LOCK, help="the shared GPU lock file (default %(default)s)")
    ap.add_argument("--lock-holder", default=LOCK_HOLDER, help="the holder the lock must name (default %(default)s)")
    ap.add_argument("--battery-start", type=int, default=BATTERY_START_PCT, help="do not start a cell on battery below this percent (default %(default)s)")
    ap.add_argument("--battery-stop", type=int, default=BATTERY_STOP_PCT, help="stop before the next turn on battery below this percent (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and the checks; start no gateway and send no request to a model")
    return ap.parse_args(argv)


def build_study(args: argparse.Namespace) -> Study:
    arms = load_arms(args.arms_file) if args.arms_file else dict(ARMS)
    try:
        tasks = data_v2.load_any(args.tasks)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
    v2 = isinstance(tasks, data_v2.TaskSet)
    names = tuple(a.strip() for a in (args.arms or ",".join(DEFAULT_ARMS_V2 if v2 else DEFAULT_ARMS)).split(",") if a.strip())
    unknown = [a for a in names if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arm {', '.join(unknown)}; choose from {', '.join(arms)}")
    repeats = args.repeats if args.repeats is not None else (REPEATS_V2 if v2 else REPEATS)
    try:
        schedule(names, repeats)
        extra = json.loads(args.extra_body)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
    run_dir = args.run_dir.resolve()
    turn_guard, cell_guard = make_guards(run_dir, args.gpu_lock, args.lock_holder, args.battery_start, args.battery_stop)
    return Study(
        run_dir=run_dir, tasks=tasks, arms=arms, arm_names=names, repeats=repeats, localmelo=args.localmelo,
        expect_sha=args.localmelo_sha, base_url=args.base_url, model=args.model, embedding_model=args.embedding_model,
        extra_body=extra, question_ids=tuple(q.strip() for q in args.questions.split(",") if q.strip()),
        keep_ask_chats=args.keep_ask_chats, turn_timeout_s=args.turn_timeout, teach_timeout_s=args.teach_timeout, long_timeout_s=args.long_timeout,
        cold_start=not args.no_cold_start, unload_timeout_s=args.unload_timeout,
        use_proxy=not args.no_proxy,
        use_stream=not args.no_stream,
        do_warmup=not args.no_warmup, turn_guard=turn_guard, cell_guard=cell_guard,
    )


def admin_token(cmd: str | None) -> str | None:
    """The model server's API key, printed by ``cmd`` (no shell); None without a command."""
    if not cmd:
        return None
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SystemExit(f"--admin-token-cmd failed (exit {proc.returncode}); its output is not shown")
    return proc.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    server.set_admin_token(admin_token(args.admin_token_cmd))
    study = build_study(args)
    if args.dry_run:
        study.localmelo = study.localmelo.resolve()
        problems = preflight(study)
        print_plan(study, schedule(study.arm_names, study.repeats), problems)
        return 2 if problems else 0
    return run_study(study)


if __name__ == "__main__":
    sys.exit(main())
