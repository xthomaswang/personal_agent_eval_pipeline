"""Grade every stored answer of the public cell again with one fixed judge model, and compare it with the old verdicts.

Every run under ``memory/runs`` was graded by a judge that was the same model as the chat model under test. This tool sends the
stored ``answer`` of each finished row to one other judge model and sets the two verdicts side by side. It asks no chat again.

    cd eval
    uv run python -m memory.public.rejudge --runs-root memory/runs --base-url <url> --judge-model <id> --out <dir> \\
        [--data <longmemeval_s_cleaned.json>] [--runs NAME ...] [--workers 4] [--limit N] [--dry-run] [--report-only]

How it works:

- A run is a directory with a ``results.jsonl`` (``<root>/<run>/`` or one level deeper, ``<root>/<run>/<part>/``). The rows only
  keep the answer, so the question text and the gold answer come from the LongMemEval file (``--data``, default
  ``$LONGMEMEVAL_DATA``). A run is used only when its ``public.json`` records the same hash as that file, its ``questions.json``
  is there, and every row names a question the file knows with the type the row records. Any other run is named with the
  reason and left out; nothing is guessed.
- Every row that is finished and has an answer is one cell. A cell's judgment is keyed by (question id, SHA-256 of the answer
  text, abstention flag), so identical answers to one question are judged once and the verdict is shared.
- Each finished judge call is appended to ``<out>/verdicts.jsonl`` at once. On the next start, keys with a verdict are skipped;
  a verdict with an error (the judge could not be reached) is asked again. The pass ends cleanly, on SIGTERM or SIGINT, at the
  time in the environment variable NIGHT_DEADLINE (seconds since the epoch), or when the file named by ``--stop-file`` exists: no
  new call is started, the calls in flight finish and are written, and the tool says how many answers are still open. A second
  signal aborts at once.
- When every key has a verdict (or with ``--report-only``) ``<out>/rejudge.json`` and ``<out>/rejudge.md`` are written: per run and
  arm the correct counts under the old and the new judge and the flips; the agreement overall and the flips by question type;
  per run the difference between every two arms under both judges, with the pairs whose sign changed first; and how often the
  new judge needed a second ask or stayed unclear.

Exit codes: 0 every key has a verdict and the report is written, or a ``--dry-run``, ``--report-only`` or finished ``--limit`` pilot;
75 (EX_TEMPFAIL) stopped on request with answers still open, so a later start resumes; 1 some answers got no reply from the judge;
2 refused to start (bad arguments, no usable run, verdicts of another judge model in ``--out``).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self, TextIO

from memory.across import server

from . import data, judge

Key = tuple[str, str, bool]  # (question id, SHA-256 of the answer text, abstention flag)
Post = Callable[..., tuple[float, Any]]

VERDICTS_FILE = "verdicts.jsonl"
REPORT_JSON = "rejudge.json"
REPORT_MD = "rejudge.md"
EXIT_STOPPED = 75  # EX_TEMPFAIL: stopped on request with keys left; the next start resumes
PROGRESS_EVERY = 200  # a progress line after this many judge calls
SHIFT_FLAG = 3  # a pair of arms whose difference moved by this many correct answers or more is listed first
MAX_DEAD_IN_A_ROW = 20  # this many calls in a row with no reply from the judge end the pass
REQUESTED = ("signal", "deadline", "stop file")  # the reasons a pass is stopped on request (the exit code is then EXIT_STOPPED)
JUDGE_DOWN = "judge down"
RUN_INTERNALS = frozenset({"work", "cells", "questions"})  # directories inside a run, never runs of their own


def _say(text: str) -> None:
    print(text, flush=True)


class SkipRun(Exception):
    """A run cannot be matched to the question file; the message says why (no local path in it)."""


# --- the question file ---


@dataclass(frozen=True)
class Gold:
    """What the judge needs from a question of the file."""

    question_type: str
    question: str
    answer: Any


@dataclass(frozen=True)
class Dataset:
    """The question file reduced to the fields the judge reads, with its name and SHA-256."""

    name: str
    sha256: str
    gold: Mapping[str, Gold]


def load_dataset(path: Path) -> Dataset:
    """Read and check the LongMemEval file (``data.DataError`` when it is unusable); the haystacks are dropped at once."""
    questions = data.load(path)
    gold = {str(q["question_id"]): Gold(str(q["question_type"]), str(q["question"]), q["answer"]) for q in questions}
    del questions
    return Dataset(Path(path).name, data.file_sha256(path), gold)


# --- the stored rows ---


@dataclass(frozen=True)
class Cell:
    """One finished row of a run: where it came from and what the old judge said."""

    run: str
    arm: str
    repeat: int
    question_id: str
    question_type: str
    abstention: bool
    key: Key
    old: bool | None  # the old judge's verdict (the row's ``correct``); None when the row has none
    old_model: str


@dataclass(frozen=True)
class Task:
    """One judge call: the question with its gold answer and the stored answer to grade."""

    key: Key
    question_type: str
    question: str
    gold: Any
    abstention: bool
    answer: str


@dataclass
class RunLoad:
    """The cells of one usable run and what was left out of it."""

    name: str
    cells: list[Cell] = field(default_factory=list)
    rows: int = 0
    unfinished: int = 0  # rows whose turn did not finish
    no_answer: int = 0  # rows with an empty answer
    duplicates: int = 0  # rows dropped because a later row has the same question, arm and repeat
    unreadable: int = 0  # lines of results.jsonl that are not a JSON object


@dataclass
class Collection:
    """Every usable run, the runs left out with the reason, and the distinct judge calls (in the order they first appear)."""

    found: int
    loaded: list[RunLoad]
    skipped: list[tuple[str, str]]
    tasks: dict[Key, Task]

    @property
    def cells(self) -> int:
        return sum(len(load.cells) for load in self.loaded)

    @property
    def saved(self) -> int:
        """Judge calls saved by sharing one verdict between identical (question, answer) keys."""
        return self.cells - len(self.tasks)


def answer_key(question_id: str, answer: str, abstention: bool) -> Key:
    """The key a judgment is shared under."""
    return (question_id, hashlib.sha256(answer.encode("utf-8")).hexdigest(), abstention)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(the JSON objects of a file, the number of non-empty lines that are not one); a line cut short by a crash is skipped."""
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SkipRun(f"{what} cannot be read") from None


def find_runs(root: Path) -> dict[str, Path]:
    """Every directory with a results.jsonl at ``<root>/<run>`` or ``<root>/<run>/<part>``, by its name relative to the root."""
    found: dict[str, Path] = {}
    for pattern in ("*/results.jsonl", "*/*/results.jsonl"):
        for path in root.glob(pattern):
            run_dir = path.parent
            if run_dir.parent != root and run_dir.name in RUN_INTERNALS:
                continue
            found[run_dir.relative_to(root).as_posix()] = run_dir
    return dict(sorted(found.items()))


def select_runs(found: Mapping[str, Path], names: Sequence[str]) -> tuple[dict[str, Path], list[str]]:
    """(the runs the names pick, the names that pick none); a name picks the run of that name and the runs inside a directory of it."""
    if not names:
        return dict(found), []

    def picks(name: str, run: str) -> bool:
        return run == name.rstrip("/") or run.startswith(name.rstrip("/") + "/")

    chosen = {run: path for run, path in found.items() if any(picks(n, run) for n in names)}
    return chosen, [n for n in names if not any(picks(n, run) for run in found)]


def is_finished(row: Mapping[str, Any]) -> bool:
    """A row whose turn finished: ``state`` is "finished" (a row with no ``state`` is read by ``status`` "completed")."""
    state = row.get("state")
    return row.get("status") == "completed" if state is None else state == "finished"


def load_run(name: str, run_dir: Path, dataset: Dataset) -> tuple[RunLoad, dict[Key, Task]]:
    """The finished rows with an answer of one run and their judge calls; ``SkipRun`` (with the reason) when the run cannot be matched."""
    if not (run_dir / "public.json").exists():
        raise SkipRun("no public.json, so it is not a public-cell run")
    public = _read_json(run_dir / "public.json", "public.json")
    try:
        recorded = str(public["config"]["data"]["sha256"])
    except (KeyError, TypeError):
        raise SkipRun("public.json does not record the hash of the question file") from None
    if recorded != dataset.sha256:
        raise SkipRun(f"the run used another question file (hash {recorded[:12]}, loaded {dataset.sha256[:12]})")
    if not (run_dir / "questions.json").exists():
        raise SkipRun("no questions.json")
    listing = _read_json(run_dir / "questions.json", "questions.json")
    if not isinstance(listing, list) or not all(isinstance(q, dict) and "question_id" in q for q in listing):
        raise SkipRun("questions.json is not a list of questions")
    listed = {str(q["question_id"]) for q in listing}
    unknown = sorted(listed - set(dataset.gold))
    if unknown:
        raise SkipRun(f"questions.json names {len(unknown)} questions that the question file does not have (first: {unknown[0]})")
    try:
        rows, unreadable = read_jsonl(run_dir / "results.jsonl")
    except (OSError, ValueError):
        raise SkipRun("results.jsonl cannot be read") from None
    latest: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("question_id")), str(row.get("arm")), row.get("repeat"))] = row
    load = RunLoad(name, rows=len(latest), duplicates=len(rows) - len(latest), unreadable=unreadable)
    tasks: dict[Key, Task] = {}
    outside: list[str] = []
    differing: list[str] = []
    for row in latest.values():
        qid = str(row.get("question_id"))
        if qid not in listed:
            outside.append(qid)
            continue
        gold = dataset.gold[qid]
        abstention = data.is_abstention({"question_id": qid})
        if row.get("question_type", gold.question_type) != gold.question_type or bool(row.get("abstention", abstention)) != abstention:
            differing.append(qid)
            continue
        answer = row.get("answer")
        text = answer if isinstance(answer, str) and answer.strip() else ""
        if not text:
            load.no_answer += 1
        elif not is_finished(row):
            load.unfinished += 1
        else:
            key = answer_key(qid, text, abstention)
            judged = row.get("judge")
            old = row.get("correct")
            repeat = row.get("repeat")
            load.cells.append(Cell(
                name, str(row.get("arm")), repeat if isinstance(repeat, int) else 0, qid, gold.question_type, abstention, key,
                old if isinstance(old, bool) else None, str(judged.get("model") or "") if isinstance(judged, dict) else ""))
            tasks.setdefault(key, Task(key, gold.question_type, gold.question, gold.answer, abstention, text))
    if outside:
        raise SkipRun(f"{len(outside)} rows ask about questions that questions.json does not list (first: {outside[0]})")
    if differing:
        raise SkipRun(f"{len(differing)} rows record another question type than the question file (first: {differing[0]})")
    return load, tasks


def collect(runs: Mapping[str, Path], dataset: Dataset) -> Collection:
    """Load every run; the ones that cannot be matched are listed with the reason and contribute nothing."""
    loaded: list[RunLoad] = []
    skipped: list[tuple[str, str]] = []
    tasks: dict[Key, Task] = {}
    for name, run_dir in runs.items():
        try:
            load, found = load_run(name, run_dir, dataset)
        except SkipRun as exc:
            skipped.append((name, str(exc)))
            continue
        loaded.append(load)
        for key, task in found.items():
            tasks.setdefault(key, task)
    return Collection(len(runs), loaded, skipped, tasks)


# --- the verdict file ---


def is_settled(verdict: Mapping[str, Any] | None) -> bool:
    """A verdict the judge gave; one with an error (the judge could not be reached) is asked again."""
    return isinstance(verdict, Mapping) and not verdict.get("error")


def read_verdicts(path: Path) -> dict[Key, dict[str, Any]]:
    """The judge verdict of every key in a verdicts file; a settled verdict is never replaced by an unsettled one."""
    verdicts: dict[Key, dict[str, Any]] = {}
    if not path.exists():
        return verdicts
    rows, _ = read_jsonl(path)
    for row in rows:
        try:
            key: Key = (str(row["question_id"]), str(row["answer_sha256"]), bool(row["abstention"]))
        except KeyError:
            continue
        verdict = row.get("judge")
        if isinstance(verdict, dict) and not (is_settled(verdicts.get(key)) and not is_settled(verdict)):
            verdicts[key] = verdict
    return verdicts


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as src:
        src.seek(-1, os.SEEK_END)
        return src.read(1) == b"\n"


class VerdictLog:
    """The append-only verdicts file: one JSON line per finished judge call, flushed at once; a lock keeps each line whole."""

    def __init__(self, path: Path, verdicts: dict[Key, dict[str, Any]]) -> None:
        self.path = path
        self.verdicts = verdicts  # the latest verdict of every key, kept in step with the file
        self._lock = threading.Lock()
        self._file: TextIO | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torn = self.path.exists() and self.path.stat().st_size > 0 and not _ends_with_newline(self.path)
        self._file = self.path.open("a", encoding="utf-8")
        if torn:  # a crash cut the last line short; the next line must not be glued to it
            self._file.write("\n")
            self._file.flush()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None:
            self._file.close()

    def append(self, task: Task, verdict: judge.Verdict) -> None:
        """Write the verdict of one call; only keys without a settled verdict are asked, so none is replaced."""
        doc = verdict.to_json()
        line = json.dumps({
            "question_id": task.key[0], "answer_sha256": task.key[1], "abstention": task.abstention,
            "question_type": task.question_type, "at": datetime.now().astimezone().isoformat(timespec="seconds"), "judge": doc,
        }, ensure_ascii=False) + "\n"
        with self._lock:
            assert self._file is not None
            self._file.write(line)
            self._file.flush()
            self.verdicts[task.key] = doc


# --- asking the judge ---


class Stop:
    """Ends the pass. ``reason`` says what came first: a signal, the deadline, the stop file or a judge that looks dead.

    Workers ask ``is_set`` before each call, so the deadline and the stop file are looked at there; no call is started afterwards."""

    def __init__(self, *, deadline: float | None = None, stop_file: Path | None = None, now: Callable[[], float] = time.time) -> None:
        self._event = threading.Event()
        self.deadline, self.stop_file, self.now = deadline, stop_file, now
        self.reason = ""
        self.signalled = False  # a signal arrived (whatever came first)

    def set(self, reason: str) -> None:
        self.signalled = self.signalled or reason == "signal"
        self.reason = self.reason or reason
        self._event.set()

    def is_set(self) -> bool:
        if not self._event.is_set():
            if self.deadline is not None and self.now() >= self.deadline:
                self.set("deadline")
            elif self.stop_file is not None and self.stop_file.exists():
                self.set("stop file")
        return self._event.is_set()


@contextmanager
def stop_on_signals(stop: Stop) -> Iterator[None]:
    """SIGTERM and SIGINT set ``stop`` (a second one raises KeyboardInterrupt); the old handlers come back afterwards."""

    def handler(signum: int, frame: Any) -> None:
        if stop.signalled:
            raise KeyboardInterrupt
        stop.set("signal")

    old: dict[int, Any] = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, handler)
    except ValueError:  # not the main thread: the signals keep their default
        pass
    try:
        yield
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


@dataclass
class Outcome:
    """What one pass did."""

    calls: int = 0
    settled: int = 0
    errors: int = 0
    dead: bool = False  # MAX_DEAD_IN_A_ROW calls in a row got no reply
    seconds: float = 0.0
    judge_ms: float = 0.0  # the sum of the judge calls' own times


class _Count:
    """The counters of a pass, behind a lock, and the progress line."""

    def __init__(self, total: int, done: int, clock: Callable[[], float], say: Callable[[str], None]) -> None:
        self.total, self.done0, self.clock, self.say = total, done, clock, say
        self.started = clock()
        self.out = Outcome()
        self.in_a_row = 0
        self.lock = threading.Lock()

    def note(self, verdict: judge.Verdict) -> bool:
        """Count one finished call, say where the pass is every PROGRESS_EVERY calls; True when the judge looks dead."""
        with self.lock:
            out = self.out
            out.calls += 1
            out.judge_ms += verdict.ms
            if verdict.error:
                out.errors += 1
                self.in_a_row += 1
                out.dead = out.dead or self.in_a_row >= MAX_DEAD_IN_A_ROW
            else:
                out.settled += 1
                self.in_a_row = 0
            if out.calls % PROGRESS_EVERY == 0:
                self.say(self.line())
            return out.dead

    def line(self) -> str:
        elapsed = max(self.clock() - self.started, 1e-9)
        rate = self.out.calls / elapsed * 60.0
        done = self.done0 + self.out.settled
        return f"progress: {done}/{self.total} done, {rate:.1f} per minute, about {(self.total - done) / rate:.0f} minutes left"


def judge_pending(
    tasks: Sequence[Task], log: VerdictLog, *, total: int, done: int, base_url: str, model: str, workers: int, post: Post,
    timeout: float, stop: Stop, clock: Callable[[], float] = time.monotonic, say: Callable[[str], None] = _say,
) -> Outcome:
    """Ask the judge about ``tasks`` with ``workers`` threads and write each verdict as it arrives.

    After ``stop`` is set the calls in flight finish and are written; the ones not started are left. ``total`` and ``done`` (the keys
    of the whole job and the ones that already have a verdict) are for the progress line."""
    count = _Count(total, done, clock, say)
    waiting = len(tasks)
    waiting_lock = threading.Lock()
    all_done = threading.Event()

    def one(task: Task) -> None:
        nonlocal waiting
        try:
            if stop.is_set():
                return
            verdict = judge.judge(
                base_url, model, task.question_type, task.question, task.gold, task.answer, abstention=task.abstention,
                post=post, timeout=timeout)
            log.append(task, verdict)
            if count.note(verdict):
                stop.set(JUDGE_DOWN)
        except BaseException:  # a bug or an interrupt: let the other workers stop, then raise
            stop.set("error")
            raise
        finally:
            with waiting_lock:
                waiting -= 1
                if waiting == 0:
                    all_done.set()

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(one, task) for task in tasks]
    try:
        while tasks and not all_done.wait(0.2):  # a timeout: a signal that reached another thread still gets its handler run
            pass
    except KeyboardInterrupt:  # the second signal: do not wait for the calls in flight
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    for future in futures:
        future.result()  # raises a worker's unexpected error
    count.out.seconds = clock() - count.started
    return count.out


# --- the comparison ---


def tally(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """Counts of (old verdict, new verdict) pairs: n, correct by each judge, agreement and the two kinds of flip."""
    n = old_correct = new_correct = yes_to_no = no_to_yes = 0
    for old, new in pairs:
        n += 1
        old_correct += old
        new_correct += new
        yes_to_no += old and not new
        no_to_yes += new and not old
    agree = n - yes_to_no - no_to_yes
    return {
        "n": n, "old_correct": old_correct, "new_correct": new_correct, "yes_to_no": yes_to_no, "no_to_yes": no_to_yes,
        "agree": agree, "agreement": agree / n if n else None,
    }


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def arm_pairs(arms: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every pair of arms of a run with the difference of their correct counts (first minus second) under each judge.

    A pair is flagged when the sign of the difference changed (a tie counts as its own sign) or the difference moved by SHIFT_FLAG
    correct answers or more; flagged pairs come first, the largest move first."""
    pairs = []
    for a, b in itertools.combinations(sorted(arms), 2):
        old = arms[a]["old_correct"] - arms[b]["old_correct"]
        new = arms[a]["new_correct"] - arms[b]["new_correct"]
        changed = _sign(old) != _sign(new)
        pairs.append({
            "a": a, "b": b, "n_a": arms[a]["n"], "n_b": arms[b]["n"], "old_difference": old, "new_difference": new,
            "move": new - old, "sign_changed": changed, "flagged": changed or abs(new - old) >= SHIFT_FLAG,
        })
    pairs.sort(key=lambda p: (not p["flagged"], -abs(p["move"]), p["a"], p["b"]))
    return pairs


def build_report(collection: Collection, verdicts: Mapping[Key, Mapping[str, Any]], dataset: Dataset, judge_model: str) -> dict[str, Any]:
    """Everything the report says, as plain data. Cells with no verdict yet (or no old verdict) are counted apart, not compared."""
    compared: dict[str, list[tuple[Cell, bool, bool]]] = defaultdict(list)
    pending = old_missing = 0
    for load in collection.loaded:
        for cell in load.cells:
            verdict = verdicts.get(cell.key)
            if not is_settled(verdict):
                pending += 1
            elif cell.old is None:
                old_missing += 1
            else:
                assert verdict is not None
                compared[load.name].append((cell, cell.old, verdict.get("label") is True))
    everything = [item for items in compared.values() for item in items]
    by_type = {}
    for question_type in data.TYPES:
        items = [(o, n) for c, o, n in everything if c.question_type == question_type]
        if items:
            by_type[question_type] = tally(items)
    runs: dict[str, Any] = {}
    for load in collection.loaded:
        arms: dict[str, dict[str, Any]] = {}
        for arm in sorted({c.arm for c, _, _ in compared[load.name]}):
            items = [(c, o, n) for c, o, n in compared[load.name] if c.arm == arm]
            models = sorted({c.old_model for c, _, _ in items if c.old_model})
            arms[arm] = {**tally((o, n) for _, o, n in items), "old_judge_models": models}
        runs[load.name] = {
            "rows": load.rows, "unfinished": load.unfinished, "no_answer": load.no_answer, "duplicates": load.duplicates,
            "unreadable_lines": load.unreadable, "arms": arms, "pairs": arm_pairs(arms),
        }
    given = [verdicts[k] for k in collection.tasks if is_settled(verdicts.get(k))]
    return {
        "judge_model": judge_model,
        "question_file": {"name": dataset.name, "sha256": dataset.sha256},
        "runs_found": collection.found,
        "runs_used": len(collection.loaded),
        "runs_skipped": [{"run": name, "reason": reason} for name, reason in collection.skipped],
        "cells": collection.cells,
        "cells_compared": len(everything),
        "cells_without_new_verdict": pending,
        "cells_without_old_verdict": old_missing,
        "rows_unfinished": sum(load.unfinished for load in collection.loaded),
        "rows_no_answer": sum(load.no_answer for load in collection.loaded),
        "distinct_keys": len(collection.tasks),
        "calls_saved": collection.saved,
        "overall": tally((o, n) for _, o, n in everything),
        "by_question_type": by_type,
        "abstention_questions": tally((o, n) for c, o, n in everything if c.abstention),
        "new_judge": {
            "keys_judged": len(given),
            "retry": sum(bool(v.get("retry")) for v in given),
            "unparsed": sum(bool(v.get("unparsed")) for v in given),
            "keys_with_error": sum(1 for k in collection.tasks if k in verdicts and not is_settled(verdicts[k])),
            "keys_without_verdict": sum(1 for k in collection.tasks if k not in verdicts),
            "mean_ms_per_call": round(sum(float(v.get("ms") or 0.0) for v in given) / len(given), 1) if given else None,
        },
        "runs": runs,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines + [""]


def _signed(x: int) -> str:
    return f"{x:+d}" if x else "0"


def _tally_row(label: str, t: Mapping[str, Any]) -> list[Any]:
    return [label, t["n"], t["old_correct"], t["new_correct"], _pct(t["agreement"]), t["yes_to_no"], t["no_to_yes"]]


TALLY_HEADERS = ("", "Answers", "Old judge: correct", "New judge: correct", "Agreement", "Yes to no", "No to yes")


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as short plain English with a table per section; it holds no local path."""
    new = report["new_judge"]
    out = [
        "# Stored answers graded again by a fixed judge", "",
        f"The new judge is `{report['judge_model']}`. It was sent only the stored answers. No chat was run again.",
        "The old judge is the model that also wrote the answer. Its verdict is the one stored in each run.",
        "A yes means the judge called the answer correct. A flip \"yes to no\" is an old yes that the new judge turned into a no.", "",
        "## What was compared", "",
    ]
    out += _table(("Item", "Count"), [
        ("Runs found", report["runs_found"]), ("Runs used", report["runs_used"]), ("Runs left out", len(report["runs_skipped"])),
        ("Answers compared", report["cells_compared"]),
        ("Rows left out: turn not finished", report["rows_unfinished"]), ("Rows left out: no answer", report["rows_no_answer"]),
        ("Answers still without a new verdict", report["cells_without_new_verdict"]),
        ("Answers without an old verdict", report["cells_without_old_verdict"]),
        ("Distinct (question, answer) pairs judged", report["distinct_keys"]),
        ("Judge calls saved by sharing a verdict", report["calls_saved"]),
    ])
    out += ["## Overall", ""] + _table(TALLY_HEADERS, [_tally_row("All answers", report["overall"])])
    out += ["## Flips by question type", ""]
    out += _table(TALLY_HEADERS, [_tally_row(t, v) for t, v in report["by_question_type"].items()]
                  + [_tally_row("Abstention questions (also counted in their type)", report["abstention_questions"])])
    mean = "n/a" if new["mean_ms_per_call"] is None else f"{new['mean_ms_per_call']:.0f} ms"
    out += ["## The new judge's own replies", ""] + _table(("Item", "Count"), [
        ("Judge calls with a reply", new["keys_judged"]), ("Needed a second ask (first reply was not yes or no)", new["retry"]),
        ("Still not yes or no after the second ask (graded wrong)", new["unparsed"]),
        ("Calls with no reply from the judge", new["keys_with_error"]), ("Calls not made yet", new["keys_without_verdict"]),
        ("Mean time per call", mean),
    ])
    out += ["## Runs", "",
            "Each pair table gives the difference of the correct counts, the first arm minus the second, under each judge.",
            f"A pair is listed when the sign of the difference changed or the difference moved by {SHIFT_FLAG} or more.", ""]
    for name, run in report["runs"].items():
        if not run["arms"]:
            continue
        models = sorted({m for arm in run["arms"].values() for m in arm["old_judge_models"]})
        out += [f"### {name}", "", f"Old judge: {', '.join(models) if models else 'not recorded'}.", ""]
        out += _table(("Arm", "Answers", "Old: correct", "New: correct", "Yes to no", "No to yes"), [
            (arm, t["n"], t["old_correct"], t["new_correct"], t["yes_to_no"], t["no_to_yes"]) for arm, t in run["arms"].items()])
        flagged = [p for p in run["pairs"] if p["flagged"]]
        if flagged:
            out += _table(("Pair", "Old difference", "New difference", "Move", "Sign changed"), [
                (f"{p['a']} vs {p['b']}" + (f" (answers {p['n_a']} and {p['n_b']})" if p["n_a"] != p["n_b"] else ""),
                 _signed(p["old_difference"]), _signed(p["new_difference"]), _signed(p["move"]), "yes" if p["sign_changed"] else "no")
                for p in flagged])
        if len(run["pairs"]) > len(flagged):
            out += [f"{len(run['pairs']) - len(flagged)} other pairs kept their sign and moved by less than {SHIFT_FLAG}.", ""]
    if report["runs_skipped"]:
        out += ["## Runs left out", ""] + _table(("Run", "Reason"), [(s["run"], s["reason"]) for s in report["runs_skipped"]])
    return "\n".join(out).rstrip("\n") + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_reports(out_dir: Path, report: Mapping[str, Any]) -> None:
    """Write rejudge.json and rejudge.md into ``out_dir``."""
    _write_atomic(out_dir / REPORT_JSON, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(out_dir / REPORT_MD, render_markdown(report))


# --- the command line ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m memory.public.rejudge",
        description="Grade every stored answer of the public cell again with one fixed judge model.")
    parser.add_argument(
        "--runs-root", type=Path, default=Path(__file__).resolve().parents[1] / "runs", help="directory of the runs (default: memory/runs)")
    parser.add_argument(
        "--runs", nargs="+", default=[], metavar="NAME", help="only these runs (names relative to the root; a directory picks its runs)")
    parser.add_argument(
        "--data", type=Path, default=os.environ.get("LONGMEMEVAL_DATA"), help="longmemeval_s_cleaned.json (default: $LONGMEMEVAL_DATA)")
    parser.add_argument("--base-url", help="the judge server's OpenAI-compatible base URL, ending in /v1")
    parser.add_argument("--judge-model", help="the id of the judge model")
    parser.add_argument("--out", type=Path, help="directory for verdicts.jsonl, rejudge.json and rejudge.md")
    parser.add_argument("--workers", type=int, default=4, help="judge calls in parallel (default 4)")
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N distinct answers (a timing pilot; no report)")
    parser.add_argument("--stop-file", type=Path, default=None, help="stop (exit 75) as soon as this file exists; it is never removed here")
    parser.add_argument("--timeout", type=float, default=judge.TIMEOUT_S, help="seconds to wait for one reply (default %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="print the counts and exit; ask no judge")
    parser.add_argument("--report-only", action="store_true", help="write the report from the verdicts already in --out; ask no judge")
    return parser


def main(
    argv: Sequence[str] | None = None, *, post: Post = server.http_json, say: Callable[[str], None] = _say,
    now: Callable[[], float] = time.time,
) -> int:
    """Run the tool; ``post`` is the HTTP call of ``judge.judge`` and ``now`` the clock NIGHT_DEADLINE is read against (tests pass fakes).

    Returns the exit code (see the module doc)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.out is None:
        parser.error("--out is required (except with --dry-run)")
    if not (args.dry_run or args.report_only) and not (args.base_url and args.judge_model):
        parser.error("--base-url and --judge-model are required (except with --dry-run or --report-only)")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be at least 1")
    if args.data is None:
        parser.error("--data (or $LONGMEMEVAL_DATA) names the LongMemEval-S file")
    try:
        deadline = float(os.environ["NIGHT_DEADLINE"]) if os.environ.get("NIGHT_DEADLINE", "").strip() else None
    except ValueError:
        say("NIGHT_DEADLINE must be a number of seconds since the epoch")
        return 2
    if not args.runs_root.is_dir():
        say(f"--runs-root is not a directory: {args.runs_root}")
        return 2
    runs, unknown = select_runs(find_runs(args.runs_root), args.runs)
    if unknown:
        say(f"no run named {', '.join(unknown)} under the runs root")
        return 2
    try:
        dataset = load_dataset(args.data)
    except data.DataError as exc:
        say(f"the question file cannot be used: {exc}")
        return 2
    collection = collect(runs, dataset)
    keys = list(collection.tasks)
    work = keys[: args.limit]
    verdicts = read_verdicts(args.out / VERDICTS_FILE) if args.out else {}
    settled = {k for k in keys if is_settled(verdicts.get(k))}
    todo = [collection.tasks[k] for k in work if k not in settled]
    say(f"runs: {collection.found} found, {len(collection.loaded)} usable, {len(collection.skipped)} left out")
    for name, reason in collection.skipped:
        say(f"  left out {name}: {reason}")
    say(f"rows in the usable runs: {sum(load.rows for load in collection.loaded)}; "
        f"left out: {sum(load.unfinished for load in collection.loaded)} not finished, "
        f"{sum(load.no_answer for load in collection.loaded)} without an answer")
    say(f"answers to grade: {collection.cells}; distinct (question, answer) keys: {len(keys)}; "
        f"judge calls saved by sharing: {collection.saved}")
    say(f"keys with a verdict already: {len(settled)}; to judge now: {len(todo)}" + (f" (first {args.limit} keys)" if args.limit else ""))
    if args.dry_run:
        return 0
    models = {str(verdicts[k].get("model") or "") for k in settled}
    if args.report_only:
        if len(models) > 1:
            say(f"the verdicts come from more than one judge model: {', '.join(sorted(models))}")
            return 2
        return _finish(args.out, collection, verdicts, dataset, next(iter(models), args.judge_model or ""), say)
    if models - {args.judge_model}:
        say(f"{VERDICTS_FILE} already holds verdicts of {', '.join(sorted(models))}; use a new --out for {args.judge_model}")
        return 2
    if not keys:
        say("no answers to grade")
        return 2
    if todo:
        stop = Stop(deadline=deadline, stop_file=args.stop_file, now=now)
        with stop_on_signals(stop), VerdictLog(args.out / VERDICTS_FILE, verdicts) as log:
            outcome = judge_pending(
                todo, log, total=len(keys), done=len(settled), base_url=args.base_url, model=args.judge_model, workers=args.workers,
                post=post, timeout=args.timeout, stop=stop, say=say)
        rate = outcome.calls / outcome.seconds * 60.0 if outcome.seconds > 0 else 0.0
        say(f"{outcome.calls} judge calls in {outcome.seconds:.1f} s ({rate:.1f} per minute with {args.workers} workers); "
            f"{outcome.errors} got no reply; mean judge time {outcome.judge_ms / max(outcome.calls, 1):.0f} ms")
        if stop.reason == JUDGE_DOWN:
            say(f"stopped: the judge gave no reply {MAX_DEAD_IN_A_ROW} times in a row; check the server and run the same command again")
            return 1
        open_work = sum(1 for k in work if not is_settled(verdicts.get(k)))
        if stop.reason in REQUESTED and open_work:
            say(f"stopped on request ({stop.reason}): {open_work} of {len(work)} answers still wait for a verdict; "
                "run the same command again to continue")
            return EXIT_STOPPED
    failed = sum(1 for k in work if k in verdicts and not is_settled(verdicts[k]))
    if failed:
        say(f"{failed} answers got no reply from the judge; run the same command again to ask them once more")
        return 1
    if len(work) < len(keys):
        say(f"pilot done: {len(work)} of {len(keys)} keys have a verdict; the report is written when all of them do")
        return 0
    return _finish(args.out, collection, verdicts, dataset, args.judge_model, say)


def provenance() -> dict[str, Any]:
    """What wrote the report: this repo's commit, the machine's build numbers and the time."""
    bench = Path(__file__).resolve().parents[3]  # <repo>/eval/memory/public/rejudge.py
    return {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bench_sha": server.git_sha(bench), "bench_dirty": server.git_dirty(bench),
        "sw_vers_build": server.out(["sw_vers", "-buildVersion"]), "xcodebuild_version": server.out(["xcodebuild", "-version"]),
    }


def _finish(
    out_dir: Path, collection: Collection, verdicts: Mapping[Key, Mapping[str, Any]], dataset: Dataset, judge_model: str,
    say: Callable[[str], None],
) -> int:
    report = build_report(collection, verdicts, dataset, judge_model)
    report["provenance"] = provenance()
    write_reports(out_dir, report)
    overall = report["overall"]
    say(f"wrote {REPORT_JSON} and {REPORT_MD} in {out_dir}: {overall['n']} answers compared, agreement {_pct(overall['agreement'])}, "
        f"{overall['yes_to_no']} yes to no, {overall['no_to_yes']} no to yes; {report['cells_without_new_verdict']} without a new verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
