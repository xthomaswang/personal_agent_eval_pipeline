"""Memory across chats: one model that does both jobs, or a chat model plus an embedding model?

    cd eval
    uv run python -m memory.across.run --run-dir <dir> --localmelo <checkout> --base-url http://127.0.0.1:8500/v1 --dry-run
    uv run python -m memory.across.run ... --arms L1 --repeats 1     # one arm, once
    uv run python -m memory.across.run ...                           # 4 arms x 3 repeats; run again to resume
    uv run python -m memory.across.report <dir>                      # report.md and report.json

A cell is one arm in one repeat. Each cell gets a fresh HOME (so fresh memory files), starts its own
`melo serve` on a free port, teaches the facts (each in a new chat), asks the questions (each in a new
chat, deleted afterwards so a question never feeds the next), and stops the gateway. The arms are rotated
across repeats. The four arms (ARMS below) are data: the `-X` settings each one passes to `melo serve`.

The checkout must be at LOCALMELO_SHA (or --localmelo-sha) with a clean tree. Both models are served from one
endpoint (--base-url) for all arms, but the gateway is pointed at a timing proxy (proxy.py) that forwards to it
and logs every request. Before each cell the embedding model is unloaded (the chat model never is), and the warm-up
loads only the models the arm uses, so M2's warm-up times the embedding model's load. The driver denies every
tool approval a turn asks for (the chat model trying to save a fact to a file leaves the turn waiting) and stops
a teaching turn after 60 s; both are counted per turn.

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

from . import client, data, proxy, server

# <repo>/eval/memory/across/run.py: four levels up is the directory that holds the shared lock
GPU_LOCK = Path(__file__).resolve().parents[4] / ".gpu-lock"
LOCK_HOLDER = "agent-benchmark"
BATTERY_START_PCT = 40  # do not start a cell on battery below this
BATTERY_STOP_PCT = 20  # stop before the next turn on battery below this
REPEATS = 3
TEACH_TIMEOUT_S = 60.0  # a teaching turn that takes longer is stopped and counted
LOCALMELO_SHA = "5f1498f4"  # the retriever switch, the recall timings, and a chat told not to save stated facts in files
CHAT_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"
PROTOCOL = "across-run-v2"
DEFAULT_EXTRA_BODY = {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    x: tuple[str, ...]  # `-X KEY=VALUE` settings for `melo serve`
    embedding: bool  # whether the gateway is given the embedding model


ARMS: dict[str, Arm] = {
    "M2": Arm("M2", "mixed: chat model + embedding model", ("memory.retriever=embedding",), True),
    "L1": Arm("L1", "one model as two: the chat model picks from BM25 candidates", ("memory.retriever=llm",), False),
    "B0": Arm("B0", "words only: BM25", ("memory.retriever=bm25",), False),
    "N0": Arm("N0", "memory off", ("memory.across_chats=false",), False),
}
DEFAULT_ARMS = ("M2", "L1", "B0", "N0")


def load_arms(path: Path) -> dict[str, Arm]:
    """Arms from a JSON file ``{name: {"label", "x": [...], "embedding": bool}}``, in place of ARMS."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: Arm(name, str(row.get("label", name)), tuple(row["x"]), bool(row.get("embedding", False)))
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
    tasks: data.TaskSet
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

    def asked(self) -> tuple[data.Question, ...]:
        order = self.tasks.ask_order()
        return tuple(q for q in order if not self.question_ids or q.id in self.question_ids)

    def taught(self) -> tuple[data.Fact, ...]:
        """The facts to teach, in teaching order: those the selected questions need, and every filler."""
        if not self.question_ids:
            return self.tasks.facts
        need = {f for q in self.asked() for f in (*q.facts, *q.wrong_facts)}
        return tuple(f for f in self.tasks.facts if f.id in need or f.role == "filler")

    def rss(self) -> dict[str, Any]:
        """The model server process's rss and loaded models now."""
        return self.probe_rss() if self.probe_rss else server.model_server_state(self.base_url)

    def identity(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "tasks_sha256": self.tasks.sha256(),
            "arms": {a: {"x": list(self.arms[a].x), "embedding": self.arms[a].embedding} for a in self.arm_names},
            "repeats": self.repeats,
            "questions": [q.id for q in self.asked()],
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
                for pair in arm.x:
                    key = pair.partition("=")[0]
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
    problems += [f"selected question {q} does not exist" for q in study.question_ids if q not in {x.id for x in study.tasks.questions}]
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


def cell_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per phase: turns, failed turns, turns stopped by the timeout, approvals denied, and turns that had one."""
    out = {}
    for phase in ("teach", "ask"):
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
        server.write_config(home, server.config_document(timing.url if timing else study.base_url, study.model, embedding_model, study.extra_body))
        running = study.launcher(spec)
        gw = server.wait_ready(running)
        info["port"] = spec.port
        if timing:
            info["proxy"] = {"log": "proxy.jsonl", "upstream": study.base_url}
        done = journal_state(journal.read())

        def send(phase: str, item: str, text: str) -> dict[str, Any]:
            key = turn_key(cell, item)
            journal.append({"event": "start", "phase": phase, "item": item, "key": key, "at": now_iso()})
            try:
                res = client.run_turn(
                    gw, text, key, workspace=str(work.resolve()), poll_s=study.poll_s, use_stream=study.use_stream,
                    timeout_s=study.teach_timeout_s if phase == "teach" else study.turn_timeout_s)
            except (client.GatewayError, OSError, http.client.HTTPException, ValueError) as exc:
                if not running.alive():
                    raise server.LaunchError(f"the gateway died during {item}: {exc}") from exc
                row = turn_record(cell, phase, item, key, None, f"{type(exc).__name__}: {exc}")
            else:
                row = turn_record(cell, phase, item, key, res)
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
        info["counts"] = cell_counts(journal.read())
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


def print_plan(study: Study, planned: list[dict[str, Any]], problems: list[str]) -> None:
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
    print(f"Teaching turns stop after {study.teach_timeout_s:g} s, questions after {study.turn_timeout_s:g} s; every approval request is denied and counted. "
          f"Model calls {'go through the timing proxy (cells/<cell>/proxy.jsonl)' if study.use_proxy else 'go straight to the model server'}.")
    for name in study.arm_names:
        arm = study.arms[name]
        embedding = f" (+ {study.embedding_model})" if arm.embedding else ""
        print(f"Arm {name}: melo serve {' '.join(f'-X {x}' for x in arm.x)}{embedding}  # {arm.label}")
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
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated arms (default %(default)s)")
    ap.add_argument("--arms-file", type=Path, help="JSON file that replaces the built-in arms: {name: {label, x: [KEY=VALUE...], embedding}}")
    ap.add_argument("--repeats", type=int, default=REPEATS, help="repeats per arm (default %(default)s)")
    ap.add_argument("--tasks", type=Path, default=data.DATA_FILE, help="the task file (default tasks_v1.yaml)")
    ap.add_argument("--questions", default="", help="comma-separated question ids to ask (default all); teaches only the facts they need, plus the fillers")
    ap.add_argument("--extra-body", default=json.dumps(DEFAULT_EXTRA_BODY), help="JSON for [vllm.extra_body] of the gateway config (default %(default)s)")
    ap.add_argument("--keep-ask-chats", action="store_true", help="keep each question's chat (default: delete it after reading the answer, so questions do not feed each other)")
    ap.add_argument("--turn-timeout", type=float, default=900.0, help="seconds a question's turn may take (default %(default)s)")
    ap.add_argument("--teach-timeout", type=float, default=TEACH_TIMEOUT_S, help="seconds a teaching turn may take before it is stopped and counted (default %(default)s)")
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
    names = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in names if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arm {', '.join(unknown)}; choose from {', '.join(arms)}")
    try:
        tasks = data.load(args.tasks)
        schedule(names, args.repeats)
        extra = json.loads(args.extra_body)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
    run_dir = args.run_dir.resolve()
    turn_guard, cell_guard = make_guards(run_dir, args.gpu_lock, args.lock_holder, args.battery_start, args.battery_stop)
    return Study(
        run_dir=run_dir, tasks=tasks, arms=arms, arm_names=names, repeats=args.repeats, localmelo=args.localmelo,
        expect_sha=args.localmelo_sha, base_url=args.base_url, model=args.model, embedding_model=args.embedding_model,
        extra_body=extra, question_ids=tuple(q.strip() for q in args.questions.split(",") if q.strip()),
        keep_ask_chats=args.keep_ask_chats, turn_timeout_s=args.turn_timeout, teach_timeout_s=args.teach_timeout, cold_start=not args.no_cold_start, unload_timeout_s=args.unload_timeout,
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
