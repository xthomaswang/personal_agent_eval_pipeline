"""LongMemEval-S against localmelo on this machine: ingest a question's history once, ask it in a fresh gateway per arm.

    cd eval
    uv run python -m memory.public.run --run-dir <dir> --localmelo <checkout> --localmelo-sha <sha> \\
        --base-url http://127.0.0.1:8500/v1 --data <longmemeval_s_cleaned.json> --arms-file <arms.json> --dry-run
    uv run python -m memory.public.run ... --questions 6          # a smoke run: 6 of the stage set, spread over the types
    uv run python -m memory.public.run ... --questions id1,id2    # or these questions
    uv run python -m memory.public.run ...                        # the stage set (120 questions); run again to resume
    uv run python -m memory.public.report <dir>                   # report.md and report.json

The benchmark (data.py) asks one question about a history of about 50 chats. Per question this driver

1. builds a template HOME (config.toml with `[ui] mode = "developer"`, the embedding retriever and the model URLs, straight
   to the model server) and runs `melo memory ingest FILE -j -X notes=rules` on it, with no gateway running. localmelo writes
   the sessions through its chat store with their dates, embeds every turn and keeps the lasting facts as notes; no chat
   model is called. The report it prints goes to `questions/<id>/ingest.json` (its `ids` map a file session id to the id
   the store gave it);
2. for every arm and repeat copies the template to a cell HOME, rewrites config.toml with the arm's settings (the gateway's
   model calls go through a timing proxy), starts `melo serve` with `LOCALMELO_NOW` set to the day of the question, asks the
   question in a new chat, stops the gateway, and has a judge model grade the reply (judge.py; --judge-url and
   --judge-model name a fixed judge on its own server, so that the judge is not the model under test). One row per
   (question, arm, repeat) goes to `results.jsonl`; running again skips the rows that are there (a turn that failed or timed
   out has its row too: delete the row to ask it again; a gateway that dies stops the run without one).

Arms are data (`--arms-file`): `{name: {"label", "x": ["memory.retriever=bm25"], "config": {"memory.recall_records": 8},
"extra_body": {...}, "embedding": true}}`. `x` are `melo serve -X` pairs, `config` dotted config.toml entries, `extra_body` the
gateway's `[vllm.extra_body]` (none: production decoding, no sampling field sent) and `embedding` whether the gateway has
the embedding model (default true).

A hit is retrieval, not the answer. The auto hit: an item of the turn's `turn.memory.used` (what recall put in front of
the model) comes from an answer session. The search hit: the same for what the `search_memory` tool returned in the turn;
a build that sends the memory event's `searched` items (its events reference documents them) is read from those alone, a
search that found nothing being a miss; an older build whose tool result names no session is matched by the words of the
answer sessions' user turns instead, and the row says which (`search_via`). Both are recorded apart, with the
`search_hint` of the memory event when the build sends one. A hit is per session; the evidence turns themselves are
checked by their words (four letters or more) found in the recalled items and the search results, as shown (a short
head of each turn): `evidence_seen` is the largest share of one evidence turn, `evidence_all` the smallest over all of
them (None when the question marks no evidence turn).

`--control-prompts FILE` (`default` for the built-in list) adds a control: on the first `--control-homes` ingested HOMEs, per
arm, one ordinary prompt that needs no memory (a unit conversion, a short poem, a regex) is asked instead of the question. Any
item recall used is a false recall; the prompt tokens and the chat time to first token come with it (`controls.jsonl`).

The operator holds the GPU lock (holder: agent-benchmark); the driver never takes or releases it. It stops cleanly before the
next cell when <dir>/STOP exists, the lock is gone or the battery is low. The checkout must be at --localmelo-sha with a
clean tree. The history of a question is the dataset: ingest files and HOMEs are removed after each question (unless
`--keep-homes`) and nothing of the dataset's text is written outside the run directory, which stays local.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.across import client, proxy, server
from memory.across import run as across

from . import data, judge

PROTOCOL = "public-lme-v1"
REPEATS = 1
BASE_SETTINGS: dict[str, Any] = {"ui.mode": "developer", "memory.retriever": "embedding"}  # every config.toml; an arm may override
INGEST_NOTES = "rules"
INGEST_TIMEOUT_S = 3600.0
TURN_TIMEOUT_S = 900.0
ARM_KEYS = {"label", "x", "config", "extra_body", "embedding"}
SEARCH_TOOL = "search_memory"
OPEN_TOOL = "open_memory"  # builds with [memory] recall_form = "list": the model opens listed turns whole
# Notices of a reply that was no answer; localmelo asks the model again after each, except for the reply that ends the turn.
NO_ANSWER_NOTICES = ("empty_reply", "unfinished_answer", "tool_call_as_text")

# Ordinary prompts that need nothing from memory, for the control (`--control-prompts default`).
CONTROL_PROMPTS = (
    "Convert 72 degrees Fahrenheit to Celsius and show the arithmetic.",
    "How many kilometres is 26.2 miles? Round to one decimal place.",
    "Write a three-line poem about a quiet train station at dawn.",
    "Write a regular expression that matches a US ZIP code with an optional four-digit extension, and explain it in one sentence.",
    "Summarize this sentence in five words or fewer: The committee postponed the vote until next quarter because two members "
    "were travelling and the budget figures were not final.",
    "What is the capital of Australia?",
    "Write a Python function that returns the n-th Fibonacci number without recursion.",
    "Explain the difference between a mutex and a semaphore in two sentences.",
    "Convert 3.5 hours to minutes.",
    "Give me three synonyms for the word quick.",
    "What is 17 multiplied by 23?",
    "Rewrite this sentence in the passive voice: The chef prepared a delicious meal.",
    "Write a SQL query that counts the rows of a table named orders for each value of its status column.",
    "Translate good morning into French, Spanish and German.",
    "Explain what a prime number is to a ten-year-old.",
    "List the planets of the solar system in order from the sun.",
    "Write a limerick about a cat who learns to code.",
    "What does the shell command ls -la do?",
    "How many seconds are in a day?",
    "Give a one-sentence definition of photosynthesis.",
)


class IngestError(RuntimeError):
    """`melo memory ingest` failed (exit code, or a report that is not JSON)."""


# --- arms ---


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    x: tuple[str, ...] = ()  # `-X KEY=VALUE` settings for `melo serve`
    config: Mapping[str, Any] = field(default_factory=dict, hash=False)  # dotted config.toml entries, e.g. memory.recall_records
    extra_body: Mapping[str, Any] = field(default_factory=dict, hash=False)  # [vllm.extra_body]; empty: no sampling field is sent
    embedding: bool = True  # whether the gateway is given the embedding model

    def identity(self) -> dict[str, Any]:
        return {"x": list(self.x), "config": dict(self.config), "extra_body": dict(self.extra_body), "embedding": self.embedding}


def load_arms(path: Path) -> dict[str, Arm]:
    """Arms from a JSON file ``{name: {"label", "x": [...], "config": {...}, "extra_body": {...}, "embedding": bool}}``.
    A key that is not one of these is an error (a misspelt setting would otherwise run as the default)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the arms file {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected an object with one entry per arm")
    arms: dict[str, Arm] = {}
    for name, row in raw.items():
        if not isinstance(row, dict) or set(row) - ARM_KEYS:
            got = sorted(row) if isinstance(row, dict) else repr(row)
            raise ValueError(f"arm {name}: expected an object with keys from {sorted(ARM_KEYS)}, got {got}")
        x = tuple(str(p) for p in row.get("x", ())) if isinstance(row.get("x", ()), list) else ()
        if any("=" not in p for p in x):
            raise ValueError(f"arm {name}: every x entry is KEY=VALUE, got {list(x)}")
        config, extra_body = row.get("config", {}), row.get("extra_body", {})
        if not isinstance(config, dict) or not isinstance(extra_body, dict) or not isinstance(row.get("x", []), list):
            raise ValueError(f"arm {name}: x is a list, config and extra_body are objects")
        arms[name] = Arm(name, str(row.get("label", name)), x, dict(config), dict(extra_body), bool(row.get("embedding", True)))
    return arms


def load_control_prompts(spec: str) -> tuple[str, ...]:
    """The control prompts: ``default`` for the built-in list, else a JSON list of strings or a text file with one prompt per line."""
    if spec == "default":
        return CONTROL_PROMPTS
    text = Path(spec).read_text(encoding="utf-8")
    if Path(spec).suffix == ".json":
        prompts = json.loads(text)
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError(f"{spec}: expected a JSON list of strings")
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not prompts:
        raise ValueError(f"{spec}: no prompts")
    return tuple(p.strip() for p in prompts if p.strip())


# --- the study ---


def gateway_problem(melo: Path, home: Path) -> str | None:
    """Why the checkout of ``melo`` cannot start `melo serve` (its gateway's packages are not installed), or None."""
    python = melo.parent / "python"
    try:
        done = subprocess.run(
            [str(python), "-c", "import localmelo.support.gateway"], capture_output=True, text=True, timeout=60, check=False,
            stdin=subprocess.DEVNULL, env=server.serve_env(home))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{python} cannot be run: {exc}"
    if done.returncode == 0:
        return None
    last = (done.stderr.strip().splitlines() or ["no output"])[-1]
    return f"this localmelo checkout cannot start melo serve ({last}); set it up with uv sync --frozen --all-extras"


@dataclass
class Study:
    run_dir: Path
    questions: list[data.Question]  # the whole file
    ids: tuple[str, ...]  # the questions this run asks, sorted
    arms: dict[str, Arm]
    arm_names: tuple[str, ...]
    repeats: int
    localmelo: Path
    expect_sha: str
    base_url: str
    model: str
    embedding_model: str
    judge_model: str
    data_path: Path = Path("longmemeval_s_cleaned.json")
    judge_base_url: str = ""  # the judge's server when it is not the chat model's (empty: base_url)
    data_sha256: str = ""
    stage: tuple[int, int, int] = (data.PER_TYPE, data.ABSTENTION_COUNT, data.SEED)  # per type, abstention, seed
    work_dir: Path | None = None  # where ingest files and HOMEs live (default <run-dir>/work)
    keep_homes: bool = False
    ingest_x: tuple[str, ...] = ()  # more `melo memory ingest -X` pairs, e.g. ingest=compress (the same for every arm)
    ingest_settings: dict[str, Any] = field(default_factory=dict)  # dotted config.toml entries of the ingest HOME only
    ingest_extra_body: dict[str, Any] = field(default_factory=dict)  # the ingest HOME's [vllm.extra_body] (e.g. thinking off)
    ingest_fail_fast: bool = False  # stop the run after its first ingest when the encoder failed (ingest_problem)
    encode_share: float | None = None  # ingest_x for the oldest ceil(share x n) sessions only, the rest plain (split_sessions)
    encode_turn_share: float | None = None  # oldest sessions until this share of the ingest file's turns is covered
    keep_answer_sessions_plain: bool = False  # turn-share cut only: never encode a session named by the answer metadata
    control_prompts: tuple[str, ...] = ()
    control_homes: int = 20
    cold_start: bool = True  # unload the embedding model before each cell, so every cell starts without it
    unload_timeout_s: float = 60.0
    turn_timeout_s: float = TURN_TIMEOUT_S
    ingest_timeout_s: float = INGEST_TIMEOUT_S
    poll_s: float = 0.25
    use_stream: bool = True
    do_warmup: bool = True
    scratch: Path | None = None  # where the preflight makes its throwaway HOME (default: the system temp dir)
    # seams, replaced in tests
    launcher: Callable[[server.LaunchSpec], Any] = server.launch
    turn_guard: Callable[[], str | None] = lambda: None
    cell_guard: Callable[[], str | None] = lambda: None
    snapshot: Callable[[Study], dict[str, Any]] | None = None
    serve_check: Callable[[Path, Path], str | None] | None = None  # None: gateway_problem
    verdict: Callable[[data.Question, str], judge.Verdict] | None = None
    sleep: Callable[[float], None] = time.sleep

    @property
    def melo(self) -> Path:
        return self.localmelo / ".venv" / "bin" / "melo"

    @property
    def work(self) -> Path:
        return self.work_dir or self.run_dir / "work"

    def by_id(self) -> dict[str, data.Question]:
        return {str(q["question_id"]): q for q in self.questions}

    def judge_answer(self, question: data.Question, answer: str) -> judge.Verdict:
        if self.verdict:
            return self.verdict(question, answer)
        return judge.judge(
            self.judge_base_url or self.base_url, self.judge_model, str(question["question_type"]), str(question["question"]), question["answer"], answer,
            abstention=data.is_abstention(question))

    def identity(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "data": {"name": self.data_path.name, "sha256": self.data_sha256},
            "stage": {"per_type": self.stage[0], "abstention": self.stage[1], "seed": self.stage[2]},
            "questions": list(self.ids),
            "arms": {a: self.arms[a].identity() for a in self.arm_names},
            "repeats": self.repeats,
            "localmelo_sha": server.git_sha(self.localmelo),
            "base_url": self.base_url,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "judge_model": self.judge_model,
            **({"judge_base_url": self.judge_base_url} if self.judge_base_url and self.judge_base_url != self.base_url else {}),
            "base_settings": BASE_SETTINGS,
            "ingest": {"notes": INGEST_NOTES, **({"x": list(self.ingest_x)} if self.ingest_x else {}),
                       **({"settings": dict(self.ingest_settings)} if self.ingest_settings else {}),
                       **({"extra_body": dict(self.ingest_extra_body)} if self.ingest_extra_body else {}),
                       **({"encode_share": self.encode_share} if self.encode_share is not None else {}),
                       **({"encode_turn_share": self.encode_turn_share} if self.encode_turn_share is not None else {}),
                       **({"keep_answer_sessions_plain": True} if self.keep_answer_sessions_plain else {})},
            "control": {"prompts": list(self.control_prompts), "homes": self.control_homes} if self.control_prompts else None,
            "cold_start": self.cold_start,
            "turn_timeout_s": self.turn_timeout_s,
        }


def settings_of(arm: Arm) -> dict[str, Any]:
    """The dotted config.toml entries of an arm's cell: the shared ones, then the arm's own over them."""
    return {**BASE_SETTINGS, **arm.config}


# --- preflight and environment ---


def knows_setting(melo: Path, home: Path, key: str) -> bool:
    """`melo config get KEY` on a throwaway HOME: whether this build knows the setting."""
    try:
        done = subprocess.run(
            [str(melo), "config", "get", key], capture_output=True, text=True, timeout=60, check=False,
            stdin=subprocess.DEVNULL, env=server.serve_env(home))
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def preflight(study: Study) -> list[str]:
    """Everything that would make the study invalid, as messages."""
    problems: list[str] = []
    sha = server.git_sha(study.localmelo)
    if not study.expect_sha or not sha.startswith(study.expect_sha):
        problems.append(
            f"localmelo at {study.localmelo.name} is {sha[:10] or 'not a git checkout'}, not {study.expect_sha or '(no sha given)'}")
    elif server.git_dirty(study.localmelo):
        problems.append(f"localmelo at {study.localmelo.name} has uncommitted changes")
    if not study.melo.exists():
        problems.append(f"{study.melo} does not exist (run uv sync in the checkout)")
    else:
        with tempfile.TemporaryDirectory(dir=study.scratch) as probe:
            keys = {k for a in study.arm_names for k in (*(p.partition("=")[0] for p in study.arms[a].x), *settings_of(study.arms[a]))}
            problems += [
                f"this localmelo build does not know the setting {k}" for k in sorted(keys) if not knows_setting(study.melo, Path(probe), k)]
            problems += [p for p in [(study.serve_check or gateway_problem)(study.melo, Path(probe))] if p]
    try:
        listed, _ = server.model_ids(study.base_url)
    except (OSError, ValueError) as exc:
        problems.append(f"{study.base_url}/models is not readable: {exc}")
    else:
        elsewhere = bool(study.judge_base_url) and study.judge_base_url != study.base_url
        for role, model in (("chat", study.model), ("embedding", study.embedding_model), *(() if elsewhere else (("judge", study.judge_model),))):
            if model not in listed:
                problems.append(f"{role} model {model!r} is not listed by {study.base_url}/models")
        if elsewhere:  # a fixed judge on its own server: it is never the model under test
            try:
                judges, _ = server.model_ids(study.judge_base_url)
            except (OSError, ValueError) as exc:
                problems.append(f"{study.judge_base_url}/models is not readable: {exc}")
            else:
                if study.judge_model not in judges:
                    problems.append(f"judge model {study.judge_model!r} is not listed by {study.judge_base_url}/models")
    if study.cold_start and server.loaded_models(study.base_url).get("loaded") is None:
        problems.append(f"{study.base_url} does not say which models are loaded (its admin routes need --admin-token-cmd); "
                        "pass it, or --no-cold-start")
    known = {str(q["question_id"]) for q in study.questions}
    problems += [f"selected question {q} is not in the file" for q in study.ids if q not in known]
    if not study.ids:
        problems.append("no question selected")
    if study.control_prompts and study.control_homes < 1:
        problems.append("--control-homes must be at least 1")
    return problems


def system_snapshot(study: Study) -> dict[str, Any]:
    """The machine, the builds and the data a run was made on: macOS and Xcode builds, the localmelo and benchmark commits,
    the dataset's fingerprint, the model server (models listed, process, rss)."""
    bench = Path(__file__).resolve().parents[3]  # <repo>/eval/memory/public/run.py
    snap: dict[str, Any] = {
        "timestamp": time.time(),
        "sw_vers": server.out(["sw_vers"]),
        "xcodebuild_version": server.out(["xcodebuild", "-version"]),
        "localmelo_sha": server.git_sha(study.localmelo),
        "localmelo_sends_searched": sends_searched(study.localmelo),
        "bench_sha": server.git_sha(bench),
        "bench_dirty": server.git_dirty(bench),
        "pmset_batt": server.out(["pmset", "-g", "batt"]),
        "python": sys.version.split()[0],
        "data": {"name": study.data_path.name, "sha256": study.data_sha256, "questions": len(study.questions)},
    }
    try:
        listed, listing = server.model_ids(study.base_url)
        snap["model_server"] = {
            "base_url": study.base_url, "listed": listed, "models_response": listing, **server.model_server_state(study.base_url)}
    except (OSError, ValueError) as exc:
        snap["model_server"] = {"base_url": study.base_url, "error": str(exc)}
    return snap


# --- ingest ---


def template_home(study: Study, qdir: Path, url: str | None = None) -> Path:
    """A new HOME for ``qdir``'s question with the config.toml ingest needs: developer mode, the model URLs (``url``, else
    the model server), the embedding model, and the ingest's own extra body and settings."""
    home = qdir / "template"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    write_ingest_config(study, home, url or study.base_url)
    return home


def write_ingest_config(study: Study, home: Path, url: str) -> None:
    settings = {**BASE_SETTINGS, **study.ingest_settings}  # e.g. memory.compress_cache, which `melo memory ingest -X` refuses
    server.write_config(home, server.config_document(
        url, study.model, study.embedding_model, study.ingest_extra_body or None, settings))


def ingest_requests(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The chat requests an ingest made (the encoder's, with ingest=compress), from its proxy log: how many, how many stopped
    at the token cap (``finish_reason`` length) or did not say, and the largest reply."""
    chat = [r for r in rows if str(r.get("path", "")).endswith("/chat/completions")]
    tokens = [int(r["completion_tokens"]) for r in chat if isinstance(r.get("completion_tokens"), int)]
    return {"chat": len(chat), "length": sum(1 for r in chat if r.get("finish_reason") == "length"),
            "finish_unknown": sum(1 for r in chat if not r.get("finish_reason")), "completion_tokens_max": max(tokens, default=None)}


def compress_summary(compress: Any) -> dict[str, int] | None:
    """A compress ingest's windows (failed, from the cache, cut), the items it wrote and the items the rules forced."""
    if not isinstance(compress, Mapping):
        return None
    sessions = [s for s in compress.values() if isinstance(s, Mapping)]
    windows = [w for s in sessions for w in (s.get("windows") or []) if isinstance(w, Mapping)]
    return {"windows": len(windows), "failed": sum(1 for w in windows if w.get("failed")),
            "cached": sum(1 for w in windows if w.get("cached")), "cut": sum(1 for w in windows if w.get("cut")),
            "items": sum(len(s.get("items") or []) for s in sessions),
            "forced": sum(len((s.get("gate") or {}).get("forced") or []) for s in sessions),
            **({"split": sum(int(w.get("split") or 0) for w in windows),
                "failed_parts": sum(int(w.get("failed_parts") or 0) for w in windows)}
               if any("split" in w for w in windows) else {})}


INGEST_FAILED_MAX = 0.10  # --ingest-fail-fast: the most windows the first ingest may fail


def ingest_problem(report: Mapping[str, Any]) -> str:
    """Why an ingest's encoder cannot be trusted ("" when it can): more than INGEST_FAILED_MAX of the windows failed, or,
    on a build whose encoder does not split a window (no ``split`` in the summary), a reply that stopped at the token cap.
    A build that splits answers a capped reply with a split, so only its failed windows count."""
    summary = report.get("compress_summary") or {}
    length = (report.get("requests") or {}).get("length") or 0
    if length and "split" not in summary:
        return f"{length} encoder repl{'y' if length == 1 else 'ies'} stopped at the token cap (finish_reason length)"
    if summary.get("windows") and summary["failed"] / summary["windows"] > INGEST_FAILED_MAX:
        return f"{summary['failed']} of {summary['windows']} encoder windows failed (more than {INGEST_FAILED_MAX:.0%})"
    return ""


def parse_report(stdout: str) -> dict[str, Any]:
    """The JSON object `melo memory ingest -j` printed: the whole output, else its last line that is one."""
    text = stdout.strip()
    candidates = [text, *reversed(text.splitlines())]
    for candidate in candidates:
        try:
            doc = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(doc, dict) and "sessions" in doc:
            return doc
    raise IngestError(f"the ingest report is not JSON: {text[:200]!r}")


def split_sessions(document: Mapping[str, Any], share: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ingest file's oldest ceil(share x n) sessions by date (the file's order breaks a tie, as ingest's does) and the
    rest, as two ingest files."""
    ordered = sorted(document["sessions"], key=lambda s: s["date"])
    cut = math.ceil(share * len(ordered))
    return {"sessions": ordered[:cut]}, {"sessions": ordered[cut:]}


def split_sessions_by_turns(
    document: Mapping[str, Any], share: float, plain_ids: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The smallest oldest eligible prefix covering ``share`` of all file turns.

    Stable date order breaks ties by file order, just as ``split_sessions`` does.
    Protected sessions always stay plain, but their turns remain in the denominator.
    If they prevent reaching the target, encode every eligible session; never move
    protected evidence into the encoded part or discard the question.
    """
    if not 0 < share < 1:
        raise ValueError("the turn share must be between 0 and 1")
    ordered = sorted(document["sessions"], key=lambda s: s["date"])
    total = sum(len(s["turns"]) for s in ordered)
    protected = set(plain_ids)
    encoded, plain = [], []
    covered = 0
    for session in ordered:
        if session["id"] not in protected and total and covered / total < share:
            encoded.append(session)
            covered += len(session["turns"])
        else:
            plain.append(session)
    return {"sessions": encoded}, {"sessions": plain}


def answer_file_ids(question: data.Question, document: Mapping[str, Any]) -> list[str]:
    """Map answer-session metadata to every retained ingest copy, without text matching."""
    if "answer_session_ids" not in question:
        raise IngestError("the answer-preserving cut needs answer_session_ids metadata")
    answers = {str(sid) for sid in question["answer_session_ids"]}
    file_to_original = data.haystack_ids(question)
    if answers - set(file_to_original.values()):
        raise IngestError("an answer session is absent from the question's haystack")
    protected = [file for file, original in file_to_original.items() if original in answers]
    if set(protected) - {s["id"] for s in document["sessions"]}:
        raise IngestError("the ingest document omits an answer session needed by the cut")
    return protected


SUMMED = ("sessions", "turns", "docs", "notes", "embedded_chars", "cut", "skipped", "trimmed", "seconds", "model_calls",
          "cache_hits", "split", "failed_parts", "refused", "input_tokens", "output_tokens")


def merge_reports(encoded: Mapping[str, Any], plain: Mapping[str, Any]) -> dict[str, Any]:
    """One report for an ingest done in two parts: counts summed, ``ids`` joined, the encoder's own fields from the first."""
    report = {**plain, **encoded}
    for key in SUMMED:
        if key in encoded or key in plain:
            report[key] = encoded.get(key, 0) + plain.get(key, 0)
    report["ids"] = {**(plain.get("ids") or {}), **(encoded.get("ids") or {})}
    report["error"] = "; ".join(e for e in (encoded.get("error"), plain.get("error")) if e)
    report["encoded_sessions"], report["plain_sessions"] = encoded.get("sessions", 0), plain.get("sessions", 0)
    return report


def run_ingest(study: Study, question: data.Question, home: Path, source: Path, document: Mapping[str, Any],
               xs: Sequence[str]) -> dict[str, Any]:
    """One `melo memory ingest` of ``document`` (written to ``source``, removed afterwards unless homes are kept)."""
    source.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    cmd = [str(study.melo), "memory", "ingest", str(source), "-j", "-X", f"notes={INGEST_NOTES}"]
    cmd += [part for pair in xs for part in ("-X", pair)]
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=study.ingest_timeout_s, check=False, stdin=subprocess.DEVNULL,
            cwd=home, env=server.serve_env(home))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IngestError(f"{question['question_id']}: melo memory ingest did not run to the end: {exc}") from exc
    finally:
        if not study.keep_homes:
            source.unlink(missing_ok=True)
    if done.returncode != 0:
        raise IngestError(f"{question['question_id']}: melo memory ingest exited {done.returncode}: {done.stderr.strip()[-300:]}")
    return parse_report(done.stdout)


def ingest(study: Study, question: data.Question, qdir: Path, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Ingest a question's history into a new template HOME, with no gateway on it. Returns the `-j` report (`ids` included).

    The ingest file is written under ``qdir`` and removed afterwards (it is the dataset's text). IngestError for an exit
    code other than 0 (2: not in developer mode or a bad file, 1: a gateway owns the HOME) or an output that is not JSON.
    With ``encode_share`` the oldest share of the sessions is ingested first with the run's -X pairs and the rest after it
    without its ``ingest=`` pair (plain), into the same HOME. ``encode_turn_share`` instead cuts by file turns, optionally
    protecting every answer session. ``document`` is an ingest file to load instead of the question's haystack
    (another bench's chats, as across/item_poison.py loads them)."""
    if study.encode_share is not None and study.encode_turn_share is not None:
        raise IngestError("session-share and turn-share cuts are mutually exclusive")
    if study.keep_answer_sessions_plain and study.encode_turn_share is None:
        raise IngestError("the answer-preserving cut needs encode_turn_share")
    # the ingest talks to the model server through a proxy, whose log says how each encoder reply ended
    timing = proxy.TimingProxy(study.base_url, qdir / "ingest_proxy.jsonl") if study.base_url.startswith("http") else None
    home = template_home(study, qdir, timing.url if timing else None)
    document = data.to_ingest(question) if document is None else document
    try:
        if study.encode_share is None and study.encode_turn_share is None:
            report = run_ingest(study, question, home, qdir / "ingest_input.json", document, study.ingest_x)
        else:
            if study.encode_turn_share is not None:
                protected = answer_file_ids(question, document) if study.keep_answer_sessions_plain else []
                old, new = split_sessions_by_turns(document, study.encode_turn_share, protected)
            else:
                assert study.encode_share is not None
                old, new = split_sessions(document, study.encode_share)
            plain_x = [x for x in study.ingest_x if not x.startswith("ingest=")]
            encoded = run_ingest(study, question, home, qdir / "ingest_input.json", old, study.ingest_x) if old["sessions"] else {}
            plain = run_ingest(study, question, home, qdir / "ingest_plain.json", new, plain_x) if new["sessions"] else {}
            report = merge_reports(encoded, plain)
    finally:
        if timing:
            timing.stop()
    shutil.rmtree(home / ".cache" / "localmelo" / "gateway" / "auth", ignore_errors=True)  # no stale token for a cell to read
    write_ingest_config(study, home, study.base_url)  # the proxy is gone (a cell writes its own config.toml anyway)
    if timing:
        report["requests"] = ingest_requests(proxy.read_log(qdir / "ingest_proxy.jsonl"))
    summary = compress_summary(report.get("compress"))
    if summary is not None:
        report["compress_summary"] = summary
    return report


# --- reading a turn ---


def used_items(memory: Mapping[str, Any] | None, store_to_file: Mapping[str, str]) -> list[dict[str, Any]]:
    """The items of ``turn.memory.used`` without their text: kind, id, the chat they came from (store id) and that chat's
    id in the ingest file (``file_id``, None for a chat the file does not hold, such as the question's own)."""
    out = []
    for item in (memory or {}).get("used") or []:
        if isinstance(item, Mapping):
            sid = str(item.get("session_id") or "")
            entry = {"kind": item.get("kind"), "id": item.get("id"), "turn": item.get("turn"), "session_id": sid,
                     "file_id": store_to_file.get(sid)}
            if item.get("kind") == "item":  # a memory item (compress ingest): its chats are its sources'
                entry["item"] = item.get("item")
                entry["file_ids"] = sorted({store_to_file[str(src[0])] for src in source_pairs(item) if str(src[0]) in store_to_file})
            out.append(entry)
    return out


def source_pairs(entry: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The (session, turn or exchange) pairs of an item's ``sources``; malformed ones are left out."""
    out = []
    for src in entry.get("sources") or []:
        if isinstance(src, (list, tuple)) and len(src) == 2 and isinstance(src[1], int):
            out.append((str(src[0]), src[1]))
    return out


def compress_evidence(question: data.Question, compress: Any) -> dict[str, Any] | None:
    """What a compress ingest did with the question's evidence turns, from the `-j` report's ``compress`` (per ingest
    file session id: the gate's exchange lists and the items with their sources): how many evidence exchanges and how
    many the gate kept, ``gated_out`` (none kept), the items that hold one (``items``), ``answer_in_item`` (the words of a
    short gold answer are all in one of them; None for a long answer or no item) and the route actions of those items.
    None without a ``compress`` report."""
    if not isinstance(compress, Mapping):
        return None
    chats = {str(fid): c for fid, c in compress.items() if isinstance(c, Mapping)}
    evidence = set(data.evidence_exchanges(question))
    kept = {(fid, i) for fid, c in chats.items() for i in ((c.get("gate") or {}).get("kept") or []) if isinstance(i, int)}
    holding = [it for c in chats.values() for it in (c.get("items") or [])
               if isinstance(it, Mapping) and evidence & set(source_pairs(it))]
    ids = {it.get("id") for it in holding}
    terms = data.answer_terms(question)
    words = [set(re.findall(r"[a-z0-9]+", f"{it.get('title') or ''} {it.get('text') or ''}".lower())) for it in holding]
    routes = [str(r.get("action")) for c in chats.values() for r in (c.get("routes") or [])
              if isinstance(r, Mapping) and (r.get("item") in ids or r.get("target") in ids)]
    return {
        "evidence": len(evidence), "kept": len(evidence & kept), "gated_out": bool(evidence) and not evidence & kept,
        "items": sorted(i for i in ids if i is not None),
        "answer_in_item": None if terms is None or not holding else any(terms <= w for w in words), "routes": routes,
    }


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def structured_sessions(finished: Mapping[str, Any]) -> list[str] | None:
    """The store session ids a ``tool.finished`` event names when it carries its results as data (``items``, ``results`` or
    a JSON ``output``); None when it carries text only, which is what a build without them sends."""
    for key in ("items", "results", "output"):
        value = finished.get(key)
        if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, Mapping):
            value = value.get("items") or value.get("results")
        if isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
            return [str(v["session_id"]) for v in value if v.get("session_id")]
    return None


def text_sessions(output: str, answer_texts: Mapping[str, Sequence[str]]) -> set[str]:
    """The answer sessions (file ids) a search result's text shows, by words, for a build that names no session.

    A result line reads `N. User: <what the user typed> melo: ... (chat, day)` for a chat turn and `N. the user said
    “<their words>” (chat, day)` for a note. A turn is shown when its first 48 characters start a line's text; a note when
    its quoted words are part of one of the session's user turns. Turns under 12 characters are too short to tell apart."""
    lines = [_flat(line) for line in output.splitlines() if line.strip()]
    found: set[str] = set()
    for sid, texts in answer_texts.items():
        for text in texts:
            flat = _flat(text)
            if len(flat) < 12:
                continue
            head = "User: " + flat[:48]
            for line in lines:
                quoted = re.search(r"“(.+)”", line)
                if head in line or (quoted and len(quoted.group(1)) >= 12 and quoted.group(1) in flat):
                    found.add(sid)
    return found


def searches(events: Sequence[Mapping[str, Any]], tool: str = SEARCH_TOOL) -> list[dict[str, Any]]:
    """The calls of ``tool`` (``search_memory`` unless named) in a turn, in order, each with its ``finished`` event's data."""
    started = [e for e in events if e.get("type") == "tool.started" and (e.get("data") or {}).get("name") == tool]
    finished = [e.get("data") or {} for e in events if e.get("type") == "tool.finished" and (e.get("data") or {}).get("name") == tool]
    calls = []
    for n in range(max(len(started), len(finished))):
        begun = (started[n].get("data") or {}) if n < len(started) else {}
        calls.append({"arguments": begun.get("arguments"), "finished": finished[n] if n < len(finished) else None})
    return calls


def sends_searched(localmelo: Path) -> bool:
    """Whether the build lists what ``search_memory`` found in the memory event's ``searched`` (its events reference documents
    the key; b9c02def and later). Such a build leaves the key out when the turn's searches found nothing."""
    doc = localmelo / "docs" / "reference" / "events.md"
    return doc.is_file() and "`searched` (`[" in doc.read_text(encoding="utf-8")


def opens(memory: Mapping[str, Any] | None, store_to_file: Mapping[str, str], wanted: set[str]) -> dict[str, Any]:
    """The turns the model opened with ``open_memory`` (the memory event's ``opened``, list-form builds): how many, and
    ``open_hit``, whether one is from an answer session (None when nothing was opened or the question has no answer
    session)."""
    opened = [o for o in ((memory or {}).get("opened") or []) if isinstance(o, Mapping)]
    sessions = {str(o.get("session_id") or "") for o in opened} | {sid for o in opened for sid, _ in source_pairs(o)}
    files = {store_to_file.get(sid) for sid in sessions} - {None}
    return {"opens": len(opened), "open_hit": bool(files & wanted) if opened and wanted else None,
            "opened_items": sorted({o["item"] for o in opened if o.get("item") is not None})}


def digest(memory: Mapping[str, Any] | None, store_to_file: Mapping[str, str], wanted: set[str]) -> dict[str, Any]:
    """The other chats' turns a new chat's first turn got as a digest (the memory event's ``digest``, newer builds): how
    many, and ``digest_hit``, whether one is from an answer session (None when there is no digest or no answer session)."""
    items = [i for i in (((memory or {}).get("digest") or {}).get("items") or []) if isinstance(i, Mapping)]
    files = {store_to_file.get(str(i.get("session_id") or "")) for i in items} - {None}
    return {"digest_items": len(items), "digest_hit": bool(files & wanted) if items and wanted else None}


def search_hint(memory: Mapping[str, Any] | None) -> Any:
    """The ``search_hint`` a build puts in the memory event (or in its ``recall``), as it sent it; None when it sends none."""
    found = (memory or {}).get("search_hint")
    return found if found is not None else ((memory or {}).get("recall") or {}).get("search_hint")


def retrieval(
    question: data.Question, memory: Mapping[str, Any] | None, events: Sequence[Mapping[str, Any]], ids: Mapping[str, str],
    exact: bool = False,
) -> dict[str, Any]:
    """Whether the memory reached the evidence: the auto hit and the search hit of a turn, apart (see the module docstring).

    ``ids`` is the ingest report's file id -> store id. ``auto_hit``: an item of ``turn.memory.used`` is from an answer
    session; ``auto_cover``: the share of the answer sessions that some item is from. ``search_hit`` is None when the
    turn made no search call, else whether a result is from an answer session; ``search_via`` is "memory" (the memory
    event's ``searched`` items, newer builds), "items" (the tool event names the sessions) or "text" (matched by words).
    ``exact``: the build sends ``searched`` (``sends_searched``), so a turn that searched without it found nothing and is
    never matched by words. Everything is None when the question has no answer session."""
    wanted = set(data.answer_sessions(question))
    origin = data.haystack_ids(question)  # a repeated session's copies are both the haystack session
    store_to_file = {store: origin.get(file, file) for file, store in ids.items()}
    used = used_items(memory, store_to_file)
    row: dict[str, Any] = {
        "answer_sessions": sorted(wanted), "used": used,
        "search_hint": search_hint(memory), **evidence_seen(question, memory, events),
        "auto_hit": None, "auto_cover": None, "searches": 0, "search_hit": None, "search_via": None,
        **opens(memory, store_to_file, wanted), **digest(memory, store_to_file, wanted),
    }
    calls = searches(events)
    row["searches"] = len(calls)
    reached = {f for u in used for f in (u.get("file_ids") or [u["file_id"]]) if f in wanted}
    if wanted:
        row["auto_hit"] = bool(reached)
        row["auto_cover"] = round(len(reached) / len(wanted), 4)
    searched = (memory or {}).get("searched")
    if isinstance(searched, list) and wanted:  # newer builds list every item the search tool returned in the memory event
        named = {str(i["session_id"]) for i in searched if isinstance(i, Mapping) and i.get("session_id")}
        named |= {sid for i in searched if isinstance(i, Mapping) for sid, _ in source_pairs(i)}
        row["search_hit"] = bool({store_to_file[s] for s in named if s in store_to_file} & wanted)
        row["search_via"] = "memory"
    elif exact and calls and wanted:  # the build leaves searched out when its searches found nothing
        row["search_hit"], row["search_via"] = False, "memory"
    elif calls and wanted:
        found: set[str] = set()
        via = None
        texts = data.answer_user_texts(question)
        for call in calls:
            done = call["finished"] or {}
            named = structured_sessions(done)
            if named is not None:
                found |= {store_to_file[s] for s in named if s in store_to_file} & wanted
                via = via or "items"
            else:
                found |= text_sessions(str(done.get("output") or ""), texts)
                via = via or "text"
        row["search_hit"] = bool(found)
        row["search_via"] = via
    return row


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def evidence_seen(question: data.Question, memory: Mapping[str, Any] | None, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How much of the evidence turns the model was shown, in the text of the recalled items and the ``search_memory`` and
    ``open_memory`` results: ``evidence_seen`` the largest share of one evidence turn's words, ``evidence_all`` the smallest over all of
    them (a question can need every one). Both None when the question marks no evidence turn with words."""
    # an entry's ``shown`` (newer builds) is the line the model read; ``text`` is the stored head, which may differ
    shown = [str(u.get("shown") or u.get("text") or "") for u in ((memory or {}).get("used") or []) if isinstance(u, Mapping)]
    shown += [str(c["finished"].get("output") or "") for tool in (SEARCH_TOOL, OPEN_TOOL) for c in searches(events, tool) if c["finished"]]
    seen = _words(" ".join(shown))
    shares = [len(w & seen) / len(w) for w in map(_words, data.evidence_texts(question)) if w]
    return {"evidence_seen": round(max(shares), 3) if shares else None, "evidence_all": round(min(shares), 3) if shares else None}


def first_prompt_tokens(res: client.TurnResult) -> int | None:
    """Prompt tokens of the turn's first model call (the prompt with the recalled memory in it), from the usage of its first
    ``assistant.message``; else the turn's total prompt tokens, else None."""
    for event in res.events:
        usage = (event.get("data") or {}).get("usage") if event.get("type") == "assistant.message" else None
        if isinstance(usage, Mapping) and isinstance(usage.get("prompt_tokens"), (int, float)):
            return int(usage["prompt_tokens"])
    value = res.usage.get("prompt_tokens")
    return int(value) if value is not None else None


def proxy_view(log: Sequence[Mapping[str, Any]], started: float, ended: float) -> dict[str, Any]:
    """The timing proxy's view of a turn: its agent request (the first chat completion in the turn that offers tools) and the
    time from its start to its first byte; for a request that streams, that is the first token."""
    chats = sorted((r for r in log if str(r.get("path", "")).endswith("/chat/completions") and r.get("t_start")), key=lambda r: r["t_start"])
    inside = [r for r in chats if not (started and ended) or started <= r["t_start"] <= ended]
    agent = next((r for r in inside if r.get("tools")), None)
    first = agent.get("t_first_byte") if agent else None
    embeds = [r for r in log if str(r.get("path", "")).endswith("/embeddings")]
    return {
        "ttft_ms": round((first - agent["t_start"]) * 1000.0, 1) if agent and first else None,
        "req_bytes": agent.get("req_bytes") if agent else None, "streamed": bool(agent.get("stream")) if agent else None,
        "chat_requests": len(inside), "embed_requests": len(embeds),
    }


def reply_notices(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The turn's ``notice`` events: a count per kind, the no-answer replies localmelo asked the model again for (a no-answer
    notice with a model reply after it) and whether the turn ended on a tool call written as text (that notice with no reply
    after it). Builds before 8bb1b123 send no ``tool_call_as_text`` notice."""
    kinds: dict[str, int] = {}
    retries, final_text_call = 0, False
    for i, event in enumerate(events):
        if event.get("type") != "notice":
            continue
        kind = str((event.get("data") or {}).get("kind") or "")
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind in NO_ANSWER_NOTICES:
            if any(e.get("type") == "assistant.message" for e in events[i + 1:]):
                retries += 1
            elif kind == "tool_call_as_text":
                final_text_call = True
    return {"notices": kinds, "reply_retries": retries, "final_text_tool_call": final_text_call}


def turn_fields(res: client.TurnResult | None, error: str) -> dict[str, Any]:
    """What a row keeps of a turn, whatever its kind: the outcome, the timings and the usage, the tools it called."""
    if res is None:
        return {"state": "failed", "status": "", "cause": "", "error": error, "answer": "", "timed_out": False, "denied": 0,
                **reply_notices([])}
    return {
        "state": "finished" if res.status == "completed" else "failed", "status": res.status, "cause": res.cause,
        "error": res.error or error, "answer": res.answer, "session_id": res.session_id, "turn_id": res.turn_id,
        "wall_ms": res.wall_ms, "ttfe_ms": res.ttfe_ms, "server_ms": res.server_ms(), "usage": res.usage,
        "timed_out": res.status == "timeout", "denied": len(res.denied),
        "tools": [(e.get("data") or {}).get("name") for e in res.events if e.get("type") == "tool.started"],
        "prompt_tokens": first_prompt_tokens(res), **reply_notices(res.events),
    }


# --- one cell ---


@dataclass
class Asked:
    """What came back from a cell's gateway, read after it was stopped."""

    res: client.TurnResult | None
    error: str
    cold: dict[str, Any]
    log: list[dict[str, Any]]
    seconds: float


def ask(study: Study, arm: Arm, question: data.Question, text: str, tag: str, template: Path) -> Asked:
    """One question in a new chat of a fresh gateway on a copy of ``template``.

    The cell's HOME is a copy of the template (the ingested memory), its config.toml is rewritten for the arm with the gateway's
    model URLs pointing at a timing proxy, and ``melo serve`` starts with ``LOCALMELO_NOW`` at the question's day. The gateway
    is stopped before this returns, and the HOME removed unless ``keep_homes``. server.LaunchError for a gateway that dies."""
    qid = str(question["question_id"])
    cell_dir = study.run_dir / "cells" / f"{qid}__{tag}"
    shutil.rmtree(cell_dir, ignore_errors=True)  # a cell that died halfway is started again
    cell_dir.mkdir(parents=True)
    root = study.work / qid / tag
    shutil.rmtree(root, ignore_errors=True)
    home, work = root / "home", root / "work"
    shutil.copytree(template, home, symlinks=True)
    work.mkdir(parents=True)
    timing = proxy.TimingProxy(study.base_url, cell_dir / "proxy.jsonl")
    spec = server.LaunchSpec(
        study.melo, home, server.free_port(), arm.x, cell_dir / "gateway.log", env={"LOCALMELO_NOW": data.question_now(question)})
    running = None
    res: client.TurnResult | None = None
    error = ""
    cold: dict[str, Any] = {"skipped": True}
    started = time.monotonic()
    try:
        server.write_config(home, server.config_document(
            timing.url, study.model, study.embedding_model if arm.embedding else None, arm.extra_body or None,
            {**dict(server.optional_settings(study.melo)), **settings_of(arm)}))
        if study.cold_start and arm.embedding:
            cold = server.ensure_unloaded(study.base_url, study.embedding_model, timeout=study.unload_timeout_s, sleep=study.sleep)
        running = study.launcher(spec)
        gw = server.wait_ready(running)
        key = f"plm-{re.sub('[^A-Za-z0-9]', '', qid)}-{re.sub('[^A-Za-z0-9]', '', tag)}"
        try:
            res = client.run_turn(
                gw, text, key, workspace=str(work.resolve()), poll_s=study.poll_s, use_stream=study.use_stream,
                timeout_s=study.turn_timeout_s)
        except (client.GatewayError, OSError, http.client.HTTPException, ValueError) as exc:
            if not running.alive():
                raise server.LaunchError(f"the gateway died during {qid} {tag}: {exc}") from exc
            error = f"{type(exc).__name__}: {exc}"
    finally:
        if running is not None:
            running.stop()
        timing.stop()
        if not study.keep_homes:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(home / ".cache" / "localmelo" / "gateway" / "auth", ignore_errors=True)  # the token stays out of run records
    if res is not None:
        across.save(cell_dir / "turn.json", {"events": res.events, "memory_recall": res.memory_recall})
    return Asked(res, error, cold, proxy.read_log(cell_dir / "proxy.jsonl"), round(time.monotonic() - started, 1))


def question_row(
    study: Study, arm: Arm, repeat: int, question: data.Question, ids: Mapping[str, str], asked: Asked, compress: Any = None,
) -> dict[str, Any]:
    """The results.jsonl row of one question cell: the turn, what memory did, the retrieval hits and the judge's verdict.

    For an arm that recalls memory items (``memory.recall_source = "items"``), ``compress`` (the ingest report's) adds
    ``compress`` (compress_evidence) and the item hits: ``item_hit`` (recall used or listed an item that holds the
    evidence), ``item_search_hit`` (a search returned one), ``item_opened`` (the model opened one)."""
    res = asked.res
    memory = res.memory() if res else None
    row: dict[str, Any] = {
        "protocol": PROTOCOL, "at": across.now_iso(), "question_id": question["question_id"],
        "question_type": question["question_type"], "abstention": data.is_abstention(question), "arm": arm.name, "repeat": repeat,
        **turn_fields(res, asked.error),
        "recall": ((memory or {}).get("recall") or (res.memory_recall if res else None)),
        "retrieval": retrieval(question, memory, res.events if res else [], ids, exact=sends_searched(study.localmelo)),
        "evidence": data.evidence_kept(question),
        "remembered": len((memory or {}).get("remembered") or []),
        "cold_start": asked.cold, "cell_s": asked.seconds,
    }
    settings = settings_of(arm)
    row["list_form"] = settings.get("memory.recall_form") == "list"
    if settings.get("memory.recall_source") == "items":
        row["compress"] = found = compress_evidence(question, compress)
        if found is not None:
            held, ret = set(found["items"]), row["retrieval"]
            listed = {u.get("item") for u in ret["used"] if u.get("kind") == "item"}
            searched = {i.get("item") for i in ((memory or {}).get("searched") or []) if isinstance(i, Mapping) and i.get("kind") == "item"}
            ret["item_hit"] = bool(listed & held) if held else None
            ret["item_search_hit"] = bool(searched & held) if held else None
            ret["item_opened"] = bool(set(ret.get("opened_items") or []) & held) if held else None
    if res is not None:
        row.update(proxy_view(asked.log, res.started_epoch, res.ended_epoch))
    if row["state"] == "finished":
        verdict = study.judge_answer(question, row["answer"])
        row["judge"] = verdict.to_json()
        row["correct"] = verdict.correct
        row["judge_retry"], row["judge_unparsed"] = int(verdict.retry), int(verdict.unparsed)
    else:
        row["judge"] = None
        row["correct"] = False  # a turn that did not finish is a wrong answer, not a missing one
        row["judge_retry"] = row["judge_unparsed"] = 0
    return row


def control_row(
    arm: Arm, question: data.Question, index: int, prompt: str, ids: Mapping[str, str], asked: Asked,
) -> dict[str, Any]:
    """The controls.jsonl row of one control cell: any item recall used is a false recall."""
    res = asked.res
    memory = res.memory() if res else None
    used = used_items(memory, {s: f for f, s in ids.items()})
    row: dict[str, Any] = {
        "protocol": PROTOCOL, "at": across.now_iso(), "question_id": question["question_id"], "arm": arm.name, "index": index,
        "prompt": prompt, **turn_fields(res, asked.error), "used": used, "false_recall": bool(used),
        "opens": len((memory or {}).get("opened") or []),
        "recall": ((memory or {}).get("recall") or (res.memory_recall if res else None)), "cold_start": asked.cold,
    }
    row["answer"] = row["answer"][:300]
    if res is not None:
        row.update(proxy_view(asked.log, res.started_epoch, res.ended_epoch))
    return row


# --- running the study ---


def read_rows(path: Path) -> list[dict[str, Any]]:
    return across.Journal(path).read()


def load_state(study: Study) -> dict[str, Any]:
    checkpoint = study.run_dir / "public.json"
    config = study.identity()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config:
            raise ValueError("the study's config, question set, arms or localmelo build changed; use a fresh directory")
        return state
    if study.run_dir.exists() and any(study.run_dir.iterdir()):
        raise ValueError(f"{study.run_dir} is not empty and has no public.json checkpoint; use a fresh directory")
    state = {"config": config, "complete": False}
    across.save(study.run_dir / "questions.json", [data.summary(study.by_id()[q]) for q in study.ids])
    across.save(checkpoint, state)
    return state


def cell_plan(study: Study, index: int) -> list[tuple[str, int]]:
    """(arm, repeat) of one question's cells in the order they run: repeat by repeat, the arms rotated by the question's
    number and the repeat, so no arm always goes first."""
    plan = []
    for repeat in range(1, study.repeats + 1):
        for arm in across.rotated(list(study.arm_names), index + repeat - 1):
            plan.append((arm, repeat))
    return plan


def finished_cells(study: Study) -> set[tuple[str, str, int]]:
    """(question, arm, repeat) of every row of results.jsonl that belongs to this study."""
    wanted = set(study.ids)
    return {(r["question_id"], r["arm"], r["repeat"]) for r in read_rows(study.run_dir / "results.jsonl") if r.get("question_id") in wanted}


def control_prompt(study: Study, index: int) -> str | None:
    """The control prompt of the ``index``-th question's HOME, or None when that HOME has no control cells."""
    if not study.control_prompts or index >= study.control_homes:
        return None
    return study.control_prompts[index % len(study.control_prompts)]


def missing_controls(study: Study) -> list[tuple[str, str]]:
    """(question, arm) of the control cells that have no row yet."""
    have = {(r["question_id"], r["arm"]) for r in read_rows(study.run_dir / "controls.jsonl")}
    return [
        (qid, a) for i, qid in enumerate(study.ids) if control_prompt(study, i) is not None
        for a in study.arm_names if (qid, a) not in have]


def rejudge_pending(study: Study, path: Path) -> int:
    """Ask the judge again for every row whose judge could not be reached; rewrite results.jsonl. Returns how many are settled."""
    rows = read_rows(path)
    todo = [r for r in rows if isinstance(r.get("judge"), dict) and r["judge"].get("error")]
    if not todo:
        return 0
    questions = study.by_id()
    settled = 0
    for row in todo:
        question = questions.get(str(row["question_id"]))
        if question is None:
            continue
        verdict = study.judge_answer(question, str(row.get("answer") or ""))
        row["judge"], row["correct"] = verdict.to_json(), verdict.correct
        row["judge_retry"], row["judge_unparsed"] = int(verdict.retry), int(verdict.unparsed)
        settled += int(not verdict.error)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return settled


def run_study(study: Study) -> int:
    """0: every cell has a row. 2: refused to start. 3: stopped early (run again to resume). 1: a gateway or an ingest failed."""
    study.run_dir = study.run_dir.resolve()
    study.localmelo = study.localmelo.resolve()
    study.run_dir.mkdir(parents=True, exist_ok=True)
    problems = preflight(study)
    if problems:
        print("Preflight failed:", *problems, sep="\n  - ", file=sys.stderr)
        return 2
    try:
        state = load_state(study)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    log_path = study.run_dir / "public.log"
    results, controls = study.run_dir / "results.jsonl", study.run_dir / "controls.jsonl"
    env_path = study.run_dir / "environment.json"
    env_doc = json.loads(env_path.read_text()) if env_path.exists() else {}
    env_doc.setdefault("start", (study.snapshot or system_snapshot)(study))
    bench = Path(__file__).resolve().parents[3]
    env_doc.setdefault("resumes", []).append(  # the driver may change between windows of one run
        {"at": across.now_iso(), "bench_sha": server.git_sha(bench), "bench_dirty": server.git_dirty(bench)})
    if study.do_warmup and "warmup" not in env_doc:  # the chat model is loaded once, before the first cell is timed
        env_doc["warmup"] = server.warmup(study.base_url, study.model, None)
    across.save(env_path, env_doc)
    if (fixed := rejudge_pending(study, results)):
        across.log(log_path, f"judged again: {fixed} rows")
    questions = study.by_id()
    code = 0
    first_ingest = True  # --ingest-fail-fast checks the first ingest of this run
    for index, qid in enumerate(study.ids):
        question = questions[qid]
        done = {(r["arm"], r["repeat"]) for r in read_rows(results) if r.get("question_id") == qid}
        ctl_done = {r["arm"] for r in read_rows(controls) if r.get("question_id") == qid}
        todo = [(a, r) for a, r in cell_plan(study, index) if (a, r) not in done]
        prompt = control_prompt(study, index)
        todo_ctl = [a for a in study.arm_names if prompt is not None and a not in ctl_done]
        if not todo and not todo_ctl:
            continue
        reason = study.cell_guard()
        if reason:
            across.log(log_path, f"== {reason} before {qid}; stopping")
            code = 3
            break
        qdir = study.work / qid
        try:
            across.log(log_path, f"[{index + 1}/{len(study.ids)}] {qid}: ingest, {len(todo)} cells, {len(todo_ctl)} controls")
            ids_path = study.run_dir / "questions" / qid / "ingest.json"
            report = ingest(study, question, qdir)
            across.save(ids_path, report)
            if report.get("error"):
                across.log(log_path, f"{qid}: the embedding model failed during ingest: {report['error']}")
            problem = ingest_problem(report) if study.ingest_fail_fast and first_ingest else ""
            first_ingest = False
            if problem:
                across.log(log_path, f"== {qid}: {problem}; stopping the run (--ingest-fail-fast)")
                print(f"{qid}: {problem}; stopping the run (--ingest-fail-fast)", file=sys.stderr)
                code = 4
                break
            encoder = {**report["compress_summary"], "length": (report.get("requests") or {}).get("length")} \
                if report.get("compress_summary") else None
            template = qdir / "template"
            ids = report.get("ids") or {}
            stopped = False
            for arm_name, repeat in todo:
                reason = study.turn_guard()
                if reason:
                    across.log(log_path, f"== {reason} before {qid} {arm_name} r{repeat}; stopping")
                    stopped = True
                    break
                arm = study.arms[arm_name]
                asked = ask(study, arm, question, str(question["question"]), f"{arm_name}__r{repeat}", template)
                row = question_row(study, arm, repeat, question, ids, asked, report.get("compress"))
                if encoder:
                    row["ingest_encoder"] = encoder  # the question's compress ingest: windows, failed, forced, replies at the cap
                across.Journal(results).append(row)
                across.log(log_path, f"{qid} {arm_name} r{repeat}: {row['state']}, correct={row['correct']}")
            for arm_name in [] if stopped else todo_ctl:  # the controls come after the question's own cells
                reason = study.turn_guard()
                if reason:
                    across.log(log_path, f"== {reason} before {qid} {arm_name} control; stopping")
                    stopped = True
                    break
                arm = study.arms[arm_name]
                asked = ask(study, arm, question, str(prompt), f"{arm_name}__ctl", template)
                row = control_row(arm, question, index, str(prompt), ids, asked)
                across.Journal(controls).append(row)
                across.log(log_path, f"{qid} {arm_name} control: {row['state']}, false_recall={row['false_recall']}")
            if stopped:
                code = 3
                break
        except (IngestError, server.LaunchError, server.ColdStartError) as exc:
            across.log(log_path, f"{qid}: {exc}")
            print(f"{qid}: {exc}", file=sys.stderr)
            code = 2 if isinstance(exc, server.ColdStartError) else 1
            break
        finally:
            if not study.keep_homes:
                shutil.rmtree(qdir, ignore_errors=True)
    have, total = len(finished_cells(study)), len(study.ids) * len(study.arm_names) * study.repeats
    state["complete"] = have == total and not missing_controls(study)
    across.save(study.run_dir / "public.json", state)
    print(f"{have}/{total} rows in {results}" + (" (stopped early; see public.log)" if code == 3 else ""))
    return code


# --- command line ---


def print_plan(study: Study, problems: list[str]) -> None:
    by_id = study.by_id()
    groups: dict[str, int] = {}
    for qid in study.ids:
        name = data.group(by_id[qid]) if qid in by_id else "unknown"
        groups[name] = groups.get(name, 0) + 1
    cells = len(study.ids) * len(study.arm_names) * study.repeats
    controls = len(study.arm_names) * min(study.control_homes, len(study.ids)) if study.control_prompts else 0
    print(f"LongMemEval-S plan: {len(study.ids)} questions ({', '.join(f'{g} {n}' for g, n in sorted(groups.items()))}) x "
          f"{len(study.arm_names)} arms x {study.repeats} repeats = {cells} question cells, {controls} control cells, "
          f"{len(study.ids)} ingests.")
    print(f"Output: {study.run_dir} (public.json, questions.json, environment.json, results.jsonl, controls.jsonl, cells/, questions/)")
    if study.ingest_x:
        print(f"Ingest: melo memory ingest -X notes={INGEST_NOTES} {' '.join(f'-X {x}' for x in study.ingest_x)}")
    if study.encode_share is not None:
        print(f"Encode share: the oldest {study.encode_share:g} of each haystack's sessions with those pairs, the rest plain "
              f"(no ingest= pair) in a second ingest on the same HOME")
    if study.encode_turn_share is not None:
        protected = "; answer sessions stay plain even if the target cannot be reached" if study.keep_answer_sessions_plain else ""
        print(f"Encode turn share: the oldest eligible prefix covering at least {study.encode_turn_share:g} of each file's turns"
              f"{protected}; the rest plain")
    if study.ingest_settings:
        print(f"Ingest HOME config.toml: {', '.join(f'{k}={v!r}' for k, v in study.ingest_settings.items())}")
    if study.ingest_extra_body:
        print(f"Ingest HOME [vllm.extra_body]: {json.dumps(study.ingest_extra_body)}")
    if study.ingest_fail_fast:
        print(f"Ingest fail-fast: the run stops after its first ingest if an encoder reply hit the token cap or over "
              f"{INGEST_FAILED_MAX:.0%} of its windows failed")
    for name in study.arm_names:
        arm = study.arms[name]
        config = f" [config.toml: {', '.join(f'{k}={v}' for k, v in arm.config.items())}]" if arm.config else ""
        embedding = "" if arm.embedding else " [no embedding model]"
        print(f"Arm {name}: melo serve {' '.join(f'-X {x}' for x in arm.x)}{config}{embedding}  # {arm.label}")
    print(f"Model server {study.base_url}: chat {study.model}, embedding {study.embedding_model}, judge {study.judge_model}; "
          f"turns stop after {study.turn_timeout_s:g} s; the embedding model is "
          + ("unloaded before every cell." if study.cold_start else "left as it is (--no-cold-start)."))
    print("Preflight: " + ("ok" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"  - {problem}")
    print(f"Guards: {study.cell_guard() or 'ok'}; STOP file: {'present' if (study.run_dir / 'STOP').exists() else 'absent'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="output directory; running again there resumes from results.jsonl")
    ap.add_argument("--localmelo", type=Path, required=True, help="the localmelo checkout (its .venv/bin/melo is used; it is never changed)")
    ap.add_argument("--localmelo-sha", required=True, help="the commit prefix the checkout must be at, with a clean tree")
    ap.add_argument("--base-url", required=True, help="the model server's URL up to /v1, e.g. http://127.0.0.1:8500/v1")
    ap.add_argument("--model", default=across.CHAT_MODEL, help="the chat model id as the server lists it (default %(default)s)")
    ap.add_argument(
        "--embedding-model", default=across.EMBEDDING_MODEL, help="the embedding model id of the same server (default %(default)s)")
    ap.add_argument("--judge-model", default=None, help="the model id that grades the replies (default: --model)")
    ap.add_argument("--judge-url", default=None, help="the judge model's server up to /v1, when it is not --base-url (a fixed judge that is not the model under test)")
    ap.add_argument(
        "--data", type=Path, default=os.environ.get("LONGMEMEVAL_DATA"), help="longmemeval_s_cleaned.json (default: $LONGMEMEVAL_DATA)")
    ap.add_argument(
        "--arms-file", type=Path, required=True,
        help='JSON file: {name: {label, x: ["KEY=VALUE"], config: {}, extra_body: {}, embedding: true}}')
    ap.add_argument("--arms", default=None, help="comma-separated arms of the file (default: all of them)")
    ap.add_argument("--repeats", type=int, default=REPEATS, help="repeats per arm and question (default %(default)s)")
    ap.add_argument(
        "--questions", default="",
        help="a count (that many of the stage set, spread over the types) or comma-separated ids (default: the stage set)")
    ap.add_argument("--per-type", type=int, default=data.PER_TYPE, help="stage set: non-abstention questions per type (default %(default)s)")
    ap.add_argument("--abstention", type=int, default=data.ABSTENTION_COUNT, help="stage set: abstention questions (default %(default)s)")
    ap.add_argument("--seed", type=int, default=data.SEED, help="stage set: the seed (default %(default)s)")
    ap.add_argument(
        "--control-prompts", default=None, metavar="FILE",
        help="run the false-recall control with the prompts of this JSON list or text file (`default`: the built-in 20)")
    ap.add_argument(
        "--control-homes", type=int, default=20, help="the control runs on the first N ingested HOMEs, once per arm (default %(default)s)")
    ap.add_argument(
        "--work-dir", type=Path, default=None, help="where ingest files and HOMEs live while a question runs (default <run-dir>/work)")
    ap.add_argument("--keep-homes", action="store_true", help="keep the ingest files and HOMEs (they hold the dataset's text)")
    ap.add_argument(
        "--ingest-x", action="append", default=[], metavar="KEY=VALUE",
        help="one more `melo memory ingest -X` pair for every question, e.g. ingest=compress (repeat for more)")
    ap.add_argument(
        "--ingest-extra-body", type=json.loads, default={}, metavar="JSON",
        help='the ingest HOME\'s [vllm.extra_body], e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\' (cells keep their arm\'s)')
    ap.add_argument(
        "--ingest-fail-fast", action="store_true",
        help=f"stop the run (exit 4) when its first ingest had an encoder reply at the token cap or more than "
        f"{INGEST_FAILED_MAX:.0%} failed windows")
    cuts = ap.add_mutually_exclusive_group()
    cuts.add_argument(
        "--encode-share", type=float, default=None, metavar="F",
        help="give the --ingest-x pairs to the oldest ceil(F x n) sessions of each haystack only; the rest is ingested plain")
    cuts.add_argument(
        "--encode-turn-share", type=float, default=None, metavar="F",
        help="encode the smallest oldest prefix covering at least F of all ingest-file turns; the rest is ingested plain")
    ap.add_argument(
        "--keep-answer-sessions-plain", action="store_true",
        help="with --encode-turn-share, exclude every answer session from encoding, even if the target becomes unreachable")
    ap.add_argument(
        "--ingest-setting", action="append", default=[], metavar="KEY=VALUE",
        help="a dotted config.toml entry of the ingest HOME only, e.g. memory.compress_cache=/path (a JSON value, else a string; "
        "repeat for more)")
    ap.add_argument("--turn-timeout", type=float, default=TURN_TIMEOUT_S, help="seconds a question's turn may take (default %(default)s)")
    ap.add_argument("--ingest-timeout", type=float, default=INGEST_TIMEOUT_S, help="seconds one ingest may take (default %(default)s)")
    ap.add_argument(
        "--admin-token-cmd", default=os.environ.get("BENCH_UPSTREAM_TOKEN_CMD"),
        help="a command that prints the model server's API key, for its admin routes (status, unload); run once, never recorded "
        "(default: $BENCH_UPSTREAM_TOKEN_CMD)")
    ap.add_argument("--no-cold-start", action="store_true", help="do not unload the embedding model before a cell")
    ap.add_argument(
        "--unload-timeout", type=float, default=60.0, help="seconds to wait for the embedding model to unload (default %(default)s)")
    ap.add_argument("--no-stream", action="store_true", help="poll for the end of a turn instead of following the event stream")
    ap.add_argument("--no-warmup", action="store_true", help="skip the one-token request that loads the chat model before the first cell")
    ap.add_argument("--gpu-lock", type=Path, default=across.GPU_LOCK, help="the shared GPU lock file (default %(default)s)")
    ap.add_argument("--lock-holder", default=across.LOCK_HOLDER, help="the holder the lock must name (default %(default)s)")
    ap.add_argument(
        "--battery-start", type=int, default=across.BATTERY_START_PCT,
        help="do not start a cell on battery below this percent (default %(default)s)")
    ap.add_argument(
        "--battery-stop", type=int, default=across.BATTERY_STOP_PCT,
        help="stop before the next cell on battery below this percent (default %(default)s)")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and the checks; ingest nothing, start no gateway, send no request to a model")
    return ap.parse_args(argv)


def encode_share(share: float | None, ingest_x: Sequence[str]) -> float | None:
    if share is None:
        return None
    if not 0 < share < 1:
        raise SystemExit("--encode-share takes a share between 0 and 1 (without it every session gets the --ingest-x pairs)")
    if not any(x.startswith("ingest=") for x in ingest_x):
        raise SystemExit("--encode-share needs an --ingest-x ingest=... pair to give to the oldest sessions")
    return share


def encode_turn_share(share: float | None, ingest_x: Sequence[str], keep_answers_plain: bool = False) -> float | None:
    if share is None:
        if keep_answers_plain:
            raise SystemExit("--keep-answer-sessions-plain needs --encode-turn-share")
        return None
    if not 0 < share < 1:
        raise SystemExit("--encode-turn-share takes a share between 0 and 1")
    if "ingest=compress" not in ingest_x:
        raise SystemExit("--encode-turn-share needs --ingest-x ingest=compress")
    return share


def ingest_body(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit("--ingest-extra-body takes a JSON object")
    return value


def parse_settings(pairs: Sequence[str]) -> dict[str, Any]:
    """``KEY=VALUE`` pairs as dotted config entries: a value that reads as JSON (true, 4000) is that, else the string."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, equals, value = pair.partition("=")
        if not equals or not key.strip():
            raise SystemExit(f"--ingest-setting takes KEY=VALUE, got {pair!r}")
        try:
            out[key.strip()] = json.loads(value)
        except ValueError:
            out[key.strip()] = value
    return out


def build_study(args: argparse.Namespace) -> Study:
    if not args.data:
        raise SystemExit("--data (or $LONGMEMEVAL_DATA) names the LongMemEval-S file")
    try:
        arms = load_arms(args.arms_file)
        questions = data.load(args.data)
        ids = tuple(data.select(questions, args.questions, args.per_type, args.abstention, args.seed))
        prompts = load_control_prompts(args.control_prompts) if args.control_prompts else ()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
    names = tuple(a.strip() for a in (args.arms or ",".join(arms)).split(",") if a.strip())
    unknown = [a for a in names if a not in arms]
    if unknown or len(set(names)) != len(names) or not names:
        raise SystemExit(f"unknown or repeated arm {', '.join(unknown) or ', '.join(names)}; choose from {', '.join(arms)}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    run_dir = args.run_dir.resolve()
    turn_guard, cell_guard = across.make_guards(run_dir, args.gpu_lock, args.lock_holder, args.battery_start, args.battery_stop)
    return Study(
        run_dir=run_dir, questions=questions, ids=ids, arms=arms, arm_names=names, repeats=args.repeats, localmelo=args.localmelo,
        expect_sha=args.localmelo_sha, base_url=args.base_url, model=args.model, embedding_model=args.embedding_model,
        judge_model=args.judge_model or args.model, judge_base_url=args.judge_url or "", data_path=args.data, data_sha256=data.file_sha256(args.data),
        stage=(args.per_type, args.abstention, args.seed), work_dir=args.work_dir, keep_homes=args.keep_homes, ingest_x=tuple(args.ingest_x),
        ingest_settings=parse_settings(args.ingest_setting), ingest_extra_body=ingest_body(args.ingest_extra_body),
        ingest_fail_fast=args.ingest_fail_fast, encode_share=encode_share(args.encode_share, args.ingest_x),
        encode_turn_share=encode_turn_share(args.encode_turn_share, args.ingest_x, args.keep_answer_sessions_plain),
        keep_answer_sessions_plain=args.keep_answer_sessions_plain,
        control_prompts=prompts, control_homes=args.control_homes, cold_start=not args.no_cold_start,
        unload_timeout_s=args.unload_timeout, turn_timeout_s=args.turn_timeout, ingest_timeout_s=args.ingest_timeout,
        use_stream=not args.no_stream, do_warmup=not args.no_warmup, turn_guard=turn_guard, cell_guard=cell_guard)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    server.set_admin_token(across.admin_token(args.admin_token_cmd))
    study = build_study(args)
    if args.dry_run:
        study.localmelo = study.localmelo.resolve()
        problems = preflight(study)
        print_plan(study, problems)
        return 2 if problems else 0
    return run_study(study)


if __name__ == "__main__":
    sys.exit(main())
