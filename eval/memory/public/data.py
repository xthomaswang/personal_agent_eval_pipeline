"""LongMemEval-S as the driver reads it: the file, a seeded stage set, and the two conversions localmelo needs.

LongMemEval (Wu et al., 2024; MIT, https://github.com/xiaowu0162/LongMemEval) asks one question about a long history of
chats: about 50 sessions per question, 115k tokens, the `_S` ("small") setting. The file is one JSON list of 500 questions.
It is large (277 MB), lives outside this repository and is never committed; nothing here copies a record of it.

A question holds:

    question_id            ends with `_abs` for an abstention question (the history never says what is asked)
    question_type          single-session-user | single-session-assistant | single-session-preference | multi-session |
                           temporal-reasoning | knowledge-update
    question, answer       the question text; the gold answer (a string, or an integer for some counting questions) or,
                           for single-session-preference, the rubric of the desired reply
    question_date          when the question is asked, "2023/05/30 (Tue) 23:40"
    haystack_session_ids   one id per session; haystack_dates the same with the date of each; haystack_sessions the
                           sessions, lists of {role, content[, has_answer]} turns
    answer_session_ids     the sessions that hold the evidence (abstention questions have some too)

Two conversions follow. `to_ingest` writes the haystack as the file `melo memory ingest` loads (one session per haystack
session, its id the haystack id with a repeat mark on a repeated one, dates as ISO local time), and `question_now` writes the day the question is asked as the
`LOCALMELO_NOW` value that pins localmelo's "today". The stage set is a seeded sample of the 500, stratified by question
type, so a run costs a quarter of the full set and every run of it asks the same questions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Question = dict[str, Any]

TYPES = (
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "temporal-reasoning", "knowledge-update",
)
ABSTENTION = "abstention"  # the report's name for the `_abs` questions, which it groups apart from their type
PER_TYPE = 17  # non-abstention questions of each type in the stage set (6 x 17 = 102)
ABSTENTION_COUNT = 18  # `_abs` questions, spread over the types that have some (18 + 102 = 120)
SEED = 0
# The stage set's ids as first drawn (2026-10-31, per_type 17, abstention 18, seed 0), frozen so every stage asks the same
# questions even if the sampling code or the data file changes. Ids only: the questions stay in the data file.
STAGE_SET_FILE = Path(__file__).with_name("stage_set_v1.txt")
REQUIRED = (
    "question_id", "question_type", "question", "answer", "question_date",
    "haystack_session_ids", "haystack_dates", "haystack_sessions", "answer_session_ids",
)
_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?: \([A-Za-z]{3}\))? (\d{2}):(\d{2})(?::(\d{2}))?")


class DataError(ValueError):
    pass


# --- the file ---


def load(path: Path) -> list[Question]:
    """The questions of a LongMemEval file, each checked for its keys and for equal-length haystack lists."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(q, dict) for q in raw):
        raise DataError(f"{path}: expected a JSON list of questions")
    seen: set[str] = set()
    for number, q in enumerate(raw, 1):
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            raise DataError(f"question {number}: missing {', '.join(missing)}")
        if q["question_id"] in seen:
            raise DataError(f"question id {q['question_id']} is used twice")
        seen.add(q["question_id"])
        if q["question_type"] not in TYPES:
            raise DataError(f"{q['question_id']}: unknown question type {q['question_type']!r}")
        if not len(q["haystack_session_ids"]) == len(q["haystack_dates"]) == len(q["haystack_sessions"]):
            raise DataError(f"{q['question_id']}: haystack ids, dates and sessions differ in length")
    return raw


def file_sha256(path: Path) -> str:
    """The SHA-256 of the file's bytes, which a run records so a later report knows which file it was."""
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_abstention(question: Mapping[str, Any]) -> bool:
    return str(question["question_id"]).endswith("_abs")


def group(question: Mapping[str, Any]) -> str:
    """The row of the report's accuracy table a question counts in: its type, or "abstention"."""
    return ABSTENTION if is_abstention(question) else str(question["question_type"])


# --- the stage set ---


def stage_set(
    questions: Sequence[Mapping[str, Any]], per_type: int = PER_TYPE, abstention: int = ABSTENTION_COUNT, seed: int = SEED,
) -> list[str]:
    """The ids of the stage set, sorted: ``per_type`` non-abstention questions of every type plus ``abstention`` `_abs`
    questions. The same file, counts and seed always give the same ids (the sample of a type is drawn from its ids sorted
    first, by a generator seeded with the seed and the type). The abstention questions go round the types that have some,
    in alphabetical order, one at a time, so no type is left out until the count runs out; a type with fewer questions
    than asked contributes all of them."""
    plain: dict[str, list[str]] = {}
    abstaining: dict[str, list[str]] = {}
    for q in questions:
        (abstaining if is_abstention(q) else plain).setdefault(str(q["question_type"]), []).append(str(q["question_id"]))
    chosen: list[str] = []
    for type_ in TYPES:
        ids = sorted(plain.get(type_, []))
        chosen += random.Random(f"{seed}:{type_}").sample(ids, min(per_type, len(ids)))
    queues = {t: random.Random(f"{seed}:abs:{t}").sample(sorted(ids), len(ids)) for t, ids in abstaining.items()}
    taken = 0
    while taken < abstention and any(queues.values()):
        for type_ in sorted(queues):
            if queues[type_] and taken < abstention:
                chosen.append(queues[type_].pop())
                taken += 1
    return sorted(chosen)


def subset(questions: Sequence[Mapping[str, Any]], ids: Sequence[str], count: int, seed: int = SEED) -> list[str]:
    """``count`` of ``ids`` (sorted), spread over the report's groups: every group in turn, drawn in a seeded order, so a
    small count (a smoke run) still meets several types and an abstention question."""
    by_id = {str(q["question_id"]): q for q in questions}
    groups: dict[str, list[str]] = {}
    for qid in sorted(ids):
        groups.setdefault(group(by_id[qid]), []).append(qid)
    queues = {g: random.Random(f"{seed}:subset:{g}").sample(v, len(v)) for g, v in groups.items()}
    picked: list[str] = []
    while len(picked) < min(count, len(ids)):
        for name in sorted(queues):
            if queues[name] and len(picked) < count:
                picked.append(queues[name].pop())
    return sorted(picked)


def frozen_stage_set(questions: Sequence[Mapping[str, Any]], path: Path | None = None) -> list[str] | None:
    """The frozen stage set (``STAGE_SET_FILE``, one id per line), or None when there is no such file or none of its ids is
    in ``questions`` (another data file, such as a test's). DataError when only some of them are."""
    path = path or STAGE_SET_FILE
    if not path.is_file():
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    known = {str(q["question_id"]) for q in questions}
    unknown = [i for i in ids if i not in known]
    if len(unknown) == len(ids):
        return None
    if unknown:
        raise DataError(f"{path.name} names questions the data file does not have: {', '.join(unknown[:5])}")
    return sorted(ids)


def select(
    questions: Sequence[Mapping[str, Any]], spec: str = "", per_type: int = PER_TYPE, abstention: int = ABSTENTION_COUNT,
    seed: int = SEED,
) -> list[str]:
    """The ids a run asks. ``spec`` empty: the whole stage set. A number N: N of the stage set (see `subset`). Otherwise
    comma-separated question ids, which may be any question of the file."""
    stage = frozen_stage_set(questions) if (per_type, abstention, seed) == (PER_TYPE, ABSTENTION_COUNT, SEED) else None
    stage = stage or stage_set(questions, per_type, abstention, seed)
    spec = spec.strip()
    if not spec:
        return stage
    if spec.isdigit():
        return subset(questions, stage, int(spec), seed)
    known = {str(q["question_id"]) for q in questions}
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise DataError(f"unknown question id {', '.join(unknown)}")
    return sorted(dict.fromkeys(ids))


# --- the two conversions ---


def iso_local(text: str) -> str:
    """"2023/05/30 (Tue) 23:40" as "2023-05-30T23:40:00", which localmelo reads as local time (no zone)."""
    found = _DATE.fullmatch(text.strip())
    if not found:
        raise DataError(f"not a LongMemEval date: {text!r}")
    year, month, day, hour, minute, second = found.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second or '00'}"


REPEAT_MARK = "#"  # no haystack id of the file holds it


def file_session_ids(question: Mapping[str, Any]) -> list[str | None]:
    """The id of each haystack session in the ingest file, in order: the haystack id, ``<id>#<n>`` for its n-th use when the
    haystack repeats it with other turns or another date, and None for a word-for-word repeat (same id, date and turns),
    which is written once, as localmelo reads such a repeat once from TM21 on. 13 of the 500 questions hold one session
    twice, the same turns at two dates, and ingest refuses an id used twice; both copies are kept, as the benchmark gives
    them."""
    uses: dict[str, int] = {}
    seen: list[tuple[str, Any, Any]] = []
    out: list[str | None] = []
    for sid, date, turns in zip(question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"], strict=True):
        sid = str(sid)
        if (sid, date, turns) in seen:
            out.append(None)
            continue
        seen.append((sid, date, turns))
        uses[sid] = uses.get(sid, 0) + 1
        out.append(sid if uses[sid] == 1 else f"{sid}{REPEAT_MARK}{uses[sid]}")
    return out


def haystack_ids(question: Mapping[str, Any]) -> dict[str, str]:
    """Ingest file session id -> haystack session id (the same but for a repeat)."""
    pairs = zip(file_session_ids(question), (str(s) for s in question["haystack_session_ids"]), strict=True)
    return {file: sid for file, sid in pairs if file is not None}


def to_ingest(question: Mapping[str, Any]) -> dict[str, Any]:
    """The haystack as the file ``melo memory ingest`` loads: ``{"sessions": [{"id", "date", "turns": [{"role", "text"}]}]}``.

    One session per haystack session, its ``id`` the haystack session id, marked when repeated (``file_session_ids``; the
    store gives its own ids; ``ids`` of the ingest report maps one to the other) and its ``date`` the haystack date as ISO local time. A turn keeps its role and its
    text; LongMemEval's `has_answer` marks are dropped, so the file tells localmelo nothing about the evidence. No title
    is set: a session id such as `answer_...` would show up in the chat label the model reads. Sessions stay in the
    file's order; ingest sorts them by date itself."""
    sessions = []
    for sid, date, turns in zip(file_session_ids(question), question["haystack_dates"], question["haystack_sessions"], strict=True):
        if sid is None:  # a word-for-word repeat
            continue
        sessions.append({
            "id": str(sid), "date": iso_local(date),
            "turns": [{"role": str(t["role"]), "text": str(t["content"])} for t in turns],
        })
    return {"sessions": sessions}


def question_now(question: Mapping[str, Any]) -> str:
    """The day the question is asked, as ISO local time, for ``LOCALMELO_NOW``."""
    return iso_local(str(question["question_date"]))


def answer_sessions(question: Mapping[str, Any]) -> list[str]:
    return [str(s) for s in question["answer_session_ids"]]


def answer_user_texts(question: Mapping[str, Any]) -> dict[str, list[str]]:
    """The user turns of each answer session (file session id -> texts), for matching a search result by its words when
    the gateway reports no session id. Held in memory only; the run never writes them."""
    wanted = set(answer_sessions(question))
    out: dict[str, list[str]] = {}
    for sid, turns in zip(question["haystack_session_ids"], question["haystack_sessions"], strict=True):
        if str(sid) in wanted:
            out[str(sid)] = [str(t["content"]) for t in turns if t.get("role") == "user" and str(t.get("content", "")).strip()]
    return out


def evidence_texts(question: Mapping[str, Any]) -> list[str]:
    """The evidence turns (``has_answer``) of the answer sessions, for checking whether the model was shown one. Held in
    memory only; the run never writes them."""
    wanted = set(answer_sessions(question))
    return [str(t.get("content") or "") for sid, turns in zip(question["haystack_session_ids"], question["haystack_sessions"], strict=True)
            if str(sid) in wanted for t in turns if t.get("has_answer")]


DOC_CHARS = 2000  # localmelo's index keeps the first 2000 characters of "User: <typed>\nmelo: <answer>" (support/memory/index.py)


def evidence_kept(question: Mapping[str, Any], doc_chars: int = DOC_CHARS) -> dict[str, Any]:
    """How much of each evidence turn (``has_answer``) the index keeps: a user turn and the assistant turn after it make one
    document, cut at ``doc_chars``. ``turns``: evidence turns; ``full``: those kept whole; ``best``: the largest kept share
    of one (1.0 when one is whole, None without evidence turns). Lengths only; no text is recorded."""
    shares: list[float] = []
    for session in question["haystack_sessions"]:
        for at, turn in enumerate(session):
            if not turn.get("has_answer"):
                continue
            text = str(turn.get("content") or "")
            if turn.get("role") == "user":
                offset = len("User: ")
            else:
                typed = next((str(t.get("content") or "") for t in reversed(session[:at]) if t.get("role") == "user"), "")
                offset = len("User: ") + len(typed.strip()) + len("\nmelo: ")
            shares.append(1.0 if not text else max(0.0, min(1.0, (doc_chars - offset) / len(text))))
    return {"turns": len(shares), "full": sum(s >= 1.0 for s in shares), "best": round(max(shares), 3) if shares else None}


def summary(question: Mapping[str, Any]) -> dict[str, Any]:
    """What a run records about a question, with none of its text: id, type, abstention flag, answer sessions, sizes."""
    return {
        "question_id": question["question_id"], "question_type": question["question_type"], "abstention": is_abstention(question),
        "answer_sessions": answer_sessions(question), "sessions": len(question["haystack_sessions"]),
        "turns": sum(len(s) for s in question["haystack_sessions"]),
    }
