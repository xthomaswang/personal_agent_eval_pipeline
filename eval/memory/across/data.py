"""Load and validate the task set of the memory-across-chats study (data/tasks_v1.yaml).

``load()`` returns a ``TaskSet`` or raises ``DataError`` with every problem it found, so a typo fails here
and not in the middle of a study. The checks:

- the schema: top-level keys, and for every fact and question its fields, types and known enum values;
- the shape of v1: 43 facts (20 plain, 5 old + 5 new, 5 target + 5 look-alike, 3 filler) in 10 blocks, and
  60 questions (20 exact, 15 paraphrase, 10 crosslang, 5 superseded, 5 lookalike, 5 nothing);
- ids: unique, every gold and wrong fact exists, the teaching order is 1..N without gaps and blocks never
  go back;
- the facts: a superseded value is taught before its replacement, twins point at each other, the fillers
  are last and long enough to fill the pinned-quote window, every value occurs in its fact's text;
- the questions: the language fits the kind (cross-language asks in the other language than it was taught
  in), the gold answer occurs in the gold fact's text, a reject string occurs in the wrong fact and none
  in the gold one, and every plain fact is asked at least once.

The ask order is fixed by the file: questions sorted by ``sha256("<ask_seed>:<id>")``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import score

DATA_FILE = Path(__file__).resolve().parent / "data" / "tasks_v1.yaml"
KINDS = ("exact", "paraphrase", "crosslang", "superseded", "lookalike", "nothing")
ROLES = ("plain", "old", "new", "target", "lookalike", "filler")
LANGS = ("en", "zh")
# v1 shape: questions per kind, facts per role, teaching blocks
V1_QUESTIONS = {"exact": 20, "paraphrase": 15, "crosslang": 10, "superseded": 5, "lookalike": 5, "nothing": 5}
V1_ROLES = {"plain": 20, "old": 5, "new": 5, "target": 5, "lookalike": 5, "filler": 3}
V1_BLOCKS = 10
# localmelo pins the newest "about"/"preferences" quotes into every chat's system prompt, 600 characters
# in all (each quote counts its length + 5). The last fillers must exceed that on their own, so no asked
# fact is pinned. A quote is one sentence of at most 300 characters.
PIN_WINDOW = 600
FILLER_MIN_CHARS = 200
FILLER_MAX_CHARS = 300

_FACT_KEYS = {"id", "order", "block", "lang", "role", "text", "value", "twin", "supersedes"}
_FACT_REQUIRED = {"id", "order", "block", "lang", "role", "text"}
_QUESTION_KEYS = {"id", "kind", "lang", "text", "facts", "answer", "aliases", "reject", "wrong_facts"}
_QUESTION_REQUIRED = {"id", "kind", "lang", "text", "facts"}


class DataError(ValueError):
    """The task file is malformed; the message lists every problem."""


@dataclass(frozen=True)
class Fact:
    id: str
    order: int
    block: str
    lang: str
    role: str
    text: str  # the teaching message, sent as the first turn of a new chat
    value: str = ""
    twin: str = ""
    supersedes: str = ""


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    lang: str
    text: str
    facts: tuple[str, ...]  # the gold facts
    answer: str = ""
    aliases: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()
    wrong_facts: tuple[str, ...] = ()

    @property
    def targets(self) -> tuple[str, ...]:
        """The gold answer, then the aliases; empty for a question with nothing to recall."""
        return (self.answer, *self.aliases) if self.answer else ()


@dataclass(frozen=True)
class TaskSet:
    version: int
    protocol: str
    ask_seed: int
    facts: tuple[Fact, ...]  # in teaching order
    questions: tuple[Question, ...]  # in file order
    by_fact: Mapping[str, Fact] = field(default_factory=dict, repr=False)

    def fact(self, fact_id: str) -> Fact:
        return self.by_fact[fact_id]

    def ask_order(self) -> tuple[Question, ...]:
        """The questions in the order they are asked: a fixed shuffle by ``ask_seed``."""
        def key(q: Question) -> str:
            return hashlib.sha256(f"{self.ask_seed}:{q.id}".encode()).hexdigest()

        return tuple(sorted(self.questions, key=key))

    def sha256(self) -> str:
        """A fingerprint of everything a study depends on, for checkpoints."""
        parts = [self.protocol, str(self.ask_seed)]
        parts += [f"{f.id}|{f.order}|{f.text}" for f in self.facts]
        parts += [f"{q.id}|{q.kind}|{q.text}|{q.answer}|{','.join(q.aliases)}|{','.join(q.reject)}" for q in self.questions]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _str_list(row: Mapping[str, Any], key: str, where: str, problems: list[str]) -> tuple[str, ...]:
    value = row.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        problems.append(f"{where}: {key} must be a list of non-empty strings")
        return ()
    return tuple(value)


def _parse_fact(row: Any, index: int, problems: list[str]) -> Fact | None:
    where = f"facts[{index}]"
    if not isinstance(row, Mapping):
        problems.append(f"{where}: not a mapping")
        return None
    where = f"fact {row.get('id', index)}"
    extra, missing = set(row) - _FACT_KEYS, _FACT_REQUIRED - set(row)
    if extra or missing:
        problems.append(f"{where}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
        return None
    for key in ("id", "block", "lang", "role", "text", "value", "twin", "supersedes"):
        if key in row and not isinstance(row[key], str):
            problems.append(f"{where}: {key} must be a string")
            return None
    if isinstance(row["order"], bool) or not isinstance(row["order"], int):
        problems.append(f"{where}: order must be an integer")
        return None
    if row["lang"] not in LANGS:
        problems.append(f"{where}: lang {row['lang']!r} is not one of {LANGS}")
    if row["role"] not in ROLES:
        problems.append(f"{where}: role {row['role']!r} is not one of {ROLES}")
    if not row["text"].strip():
        problems.append(f"{where}: empty text")
    return Fact(**{k: row[k] for k in row})


def _parse_question(row: Any, index: int, problems: list[str]) -> Question | None:
    where = f"questions[{index}]"
    if not isinstance(row, Mapping):
        problems.append(f"{where}: not a mapping")
        return None
    where = f"question {row.get('id', index)}"
    extra, missing = set(row) - _QUESTION_KEYS, _QUESTION_REQUIRED - set(row)
    if extra or missing:
        problems.append(f"{where}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
        return None
    for key in ("id", "kind", "lang", "text", "answer"):
        if key in row and not isinstance(row[key], str):
            problems.append(f"{where}: {key} must be a string")
            return None
    if row["kind"] not in KINDS:
        problems.append(f"{where}: kind {row['kind']!r} is not one of {KINDS}")
    if row["lang"] not in LANGS:
        problems.append(f"{where}: lang {row['lang']!r} is not one of {LANGS}")
    return Question(
        id=row["id"], kind=row["kind"], lang=row["lang"], text=row["text"],
        facts=_str_list(row, "facts", where, problems), answer=row.get("answer", ""),
        aliases=_str_list(row, "aliases", where, problems), reject=_str_list(row, "reject", where, problems),
        wrong_facts=_str_list(row, "wrong_facts", where, problems),
    )


def _check_facts(facts: Sequence[Fact], by_id: Mapping[str, Fact], problems: list[str]) -> None:
    if [f.order for f in facts] != list(range(1, len(facts) + 1)):
        problems.append("facts: order must be 1..N in file order, without gaps or repeats")
    blocks: list[str] = []
    for f in facts:
        if not blocks or blocks[-1] != f.block:
            if f.block in blocks:
                problems.append(f"fact {f.id}: block {f.block} is not contiguous")
            blocks.append(f.block)
    texts = [f.text for f in facts]
    if len(set(texts)) != len(texts):
        problems.append("facts: two facts have the same teaching text")
    for f in facts:
        if f.role != "filler":
            if not f.value:
                problems.append(f"fact {f.id}: a {f.role} fact needs a value")
            elif not score.contains(f.text, f.value):
                problems.append(f"fact {f.id}: value {f.value!r} is not in its text")
        if f.role == "new":
            old = by_id.get(f.supersedes)
            if old is None or old.role != "old":
                problems.append(f"fact {f.id}: supersedes {f.supersedes!r}, which is not an old fact")
            elif old.order >= f.order:
                problems.append(f"fact {f.id}: taught before the fact it supersedes ({old.id})")
        elif f.supersedes:
            problems.append(f"fact {f.id}: only a new fact supersedes another")
        if f.role in ("target", "lookalike"):
            twin = by_id.get(f.twin)
            if twin is None or twin.twin != f.id or {f.role, twin.role} != {"target", "lookalike"}:
                problems.append(f"fact {f.id}: twin {f.twin!r} does not point back as the other of a target/lookalike pair")
        elif f.twin:
            problems.append(f"fact {f.id}: only a target or lookalike fact has a twin")
    olds = [f.id for f in facts if f.role == "old"]
    replaced = [f.supersedes for f in facts if f.role == "new"]
    if sorted(olds) != sorted(replaced):
        problems.append("facts: every old fact must be superseded by exactly one new fact")
    fillers = [f for f in facts if f.role == "filler"]
    if fillers:
        if [f.id for f in facts[-len(fillers):]] != [f.id for f in fillers]:
            problems.append("facts: the fillers must be the last facts taught")
        for f in fillers:
            if not FILLER_MIN_CHARS <= len(f.text) <= FILLER_MAX_CHARS:
                problems.append(f"fact {f.id}: a filler needs {FILLER_MIN_CHARS}-{FILLER_MAX_CHARS} characters, has {len(f.text)}")
        if len(fillers) < 3 or sum(len(f.text) + 5 for f in fillers[-3:]) <= PIN_WINDOW:
            problems.append(f"facts: the last three fillers must exceed the {PIN_WINDOW}-character pinned-quote window")


def _check_questions(ts: Sequence[Question], by_id: Mapping[str, Fact], facts: Sequence[Fact], problems: list[str]) -> None:
    ids = [q.id for q in ts]
    if len(set(ids)) != len(ids):
        problems.append("questions: duplicate ids")
    if len({q.text for q in ts}) != len(ts):
        problems.append("questions: two questions have the same text")
    asked: set[str] = set()
    for q in ts:
        where = f"question {q.id}"
        unknown = [f for f in (*q.facts, *q.wrong_facts) if f not in by_id]
        if unknown:
            problems.append(f"{where}: unknown fact ids {unknown}")
            continue
        asked.update(q.facts)
        gold = [by_id[f] for f in q.facts]
        wrong = [by_id[f] for f in q.wrong_facts]
        if q.kind == "nothing":
            if q.facts or q.wrong_facts or q.answer or q.aliases:
                problems.append(f"{where}: a nothing question has no facts, answer or aliases")
        else:
            if len(q.facts) != 1 or not q.answer:
                problems.append(f"{where}: needs exactly one gold fact and an answer")
                continue
            if not score.contains(gold[0].text, q.answer):
                problems.append(f"{where}: answer {q.answer!r} is not in the text of {gold[0].id}")
        if q.kind in ("exact", "paraphrase"):
            if gold and (gold[0].role != "plain" or gold[0].lang != q.lang):
                problems.append(f"{where}: a {q.kind} question asks a plain fact in the language it was taught in")
        elif q.kind == "crosslang":
            if gold and (gold[0].role != "plain" or gold[0].lang == q.lang):
                problems.append(f"{where}: a crosslang question asks a plain fact in the other language")
        elif q.kind == "superseded":
            if gold and (gold[0].role != "new" or [w.id for w in wrong] != [gold[0].supersedes] or gold[0].lang != q.lang):
                problems.append(f"{where}: gold must be a new fact, wrong_facts its old fact, in one language")
            if not q.reject:
                problems.append(f"{where}: a superseded question needs reject strings")
        elif q.kind == "lookalike":
            if gold and (gold[0].role != "target" or [w.id for w in wrong] != [gold[0].twin] or gold[0].lang != q.lang):
                problems.append(f"{where}: gold must be a target fact, wrong_facts its look-alike, in one language")
            if not q.reject:
                problems.append(f"{where}: a lookalike question needs reject strings")
        if q.kind not in ("superseded", "lookalike") and q.wrong_facts:
            problems.append(f"{where}: only superseded and lookalike questions have wrong_facts")
        for r in q.reject:
            if gold and score.contains(gold[0].text, r):
                problems.append(f"{where}: reject {r!r} is in the gold fact's text")
        if q.kind in ("superseded", "lookalike") and wrong and not any(score.contains(wrong[0].text, r) for r in q.reject):
            problems.append(f"{where}: no reject string is in the wrong fact's text")
        for t in q.targets:
            if any(score.contains(t, r) or score.contains(r, t) for r in q.reject):
                problems.append(f"{where}: answer {t!r} and a reject string contain one another")
    for f in facts:
        if f.role == "plain" and f.id not in asked:
            problems.append(f"fact {f.id}: a plain fact no question asks")


def _check_shape(facts: Sequence[Fact], qs: Sequence[Question], expect: Mapping[str, Any] | None, problems: list[str]) -> None:
    if expect is None:
        return
    kinds = {k: sum(1 for q in qs if q.kind == k) for k in KINDS}
    roles = {r: sum(1 for f in facts if f.role == r) for r in ROLES}
    blocks = len({f.block for f in facts})
    if kinds != expect["questions"]:
        problems.append(f"questions per kind are {kinds}, expected {expect['questions']}")
    if roles != expect["roles"]:
        problems.append(f"facts per role are {roles}, expected {expect['roles']}")
    if blocks != expect["blocks"]:
        problems.append(f"{blocks} teaching blocks, expected {expect['blocks']}")


V1 = {"questions": V1_QUESTIONS, "roles": V1_ROLES, "blocks": V1_BLOCKS}


def parse(raw: Any, expect: Mapping[str, Any] | None = V1) -> TaskSet:
    """Validate a parsed YAML document; ``expect=None`` skips the v1 counts."""
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        raise DataError("the task file is not a mapping")
    unknown = set(raw) - {"version", "protocol", "ask_seed", "facts", "questions"}
    if unknown or not {"version", "protocol", "ask_seed", "facts", "questions"} <= set(raw):
        raise DataError(f"top-level keys must be version, protocol, ask_seed, facts, questions; got {sorted(raw)}")
    if raw["version"] != 1 or not isinstance(raw["protocol"], str) or not isinstance(raw["ask_seed"], int):
        problems.append("version must be 1, protocol a string, ask_seed an integer")
    if not isinstance(raw["facts"], list) or not isinstance(raw["questions"], list):
        raise DataError("facts and questions must be lists")
    facts = [f for i, row in enumerate(raw["facts"]) if (f := _parse_fact(row, i, problems))]
    questions = [q for i, row in enumerate(raw["questions"]) if (q := _parse_question(row, i, problems))]
    by_id = {f.id: f for f in facts}
    if len(by_id) != len(facts):
        problems.append("facts: duplicate ids")
    if len(facts) != len(raw["facts"]) or len(questions) != len(raw["questions"]):
        raise DataError("\n".join(problems))
    _check_facts(facts, by_id, problems)
    _check_questions(questions, by_id, facts, problems)
    _check_shape(facts, questions, expect, problems)
    if problems:
        raise DataError("\n".join(problems))
    return TaskSet(raw["version"], raw["protocol"], raw["ask_seed"], tuple(facts), tuple(questions), by_id)


def load(path: Path = DATA_FILE, expect: Mapping[str, Any] | None | str = "auto") -> TaskSet:
    """The task file at ``path``. ``expect="auto"`` holds the shipped file to the v1 counts and any other file to
    the schema only; pass ``V1`` or ``None`` to choose."""
    if expect == "auto":
        expect = V1 if Path(path).resolve() == DATA_FILE else None
    return parse(yaml.safe_load(Path(path).read_text(encoding="utf-8")), expect)  # type: ignore[arg-type]
