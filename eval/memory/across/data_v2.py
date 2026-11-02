"""Load and validate the task set of the memory bench v2 (data/tasks_v2.yaml), kind by kind.

v1 (``data.py``) asks facts taught in other chats. v2 adds kinds that need a chat of their own; the file holds the
kinds built so far (``KINDS``) and later steps add more. A kind says how many items the shipped file has, which arms
run it, and for each arm what the working window must look like at the question (``"out"``: the fact has left it,
``"in"``: it is still in it, ``""``: no condition). The arms table is data, so a run asks an arm only for the kinds
that list it.

- ``longrange``: one chat. Turn 1 is an opener, the fact is said at turn 2 or 3, about 40 filler turns follow (at
  least half use only the read-tier tools ``file_read`` and ``list_files``, each asking for a one
  sentence answer), then the question comes in the same chat. ``note_expected`` says whether localmelo's note rules
  are expected to save the fact (at least 70% of the items are details of the conversation it should not save).
- ``followup``: two facts, each taught in its own chat, then one chat with two questions: Q1 recalls the first
  fact, Q2 is a pronoun follow-up whose answer is the second fact.
- ``recall``: one question in a new chat, nothing taught: what memory holds before the cell starts (an arm's seeded
  HOME and memory import) is all there is to recall. ``category`` says what the question tests (``RECALL_CATEGORIES``);
  an ``abstain`` question has no gold (the reply must say it does not know) and a ``harm`` one asks something unrelated
  whose reply must not drag in remembered trivia (its reject strings). ``workspace`` puts the chat in that folder
  under the cell's HOME. The shipped file has no recall items: private task files hold them.
- ``sleep``: a messy-memory night. Day 1 teaches every item's facts, each in a new chat, round by round (the first
  fact of every item, then the second ...); the arm's sleep variant then runs once; day 2 asks each item's question
  in a new chat. ``category`` is the defect the facts make (``SLEEP_CATEGORIES``): the same fact twice in other
  words, a change said in passing, a fact in a busy turn, a turn likely to draw "noted" with no save, a control that
  must not change (a control with an empty answer, such as a fact only asked about or taken back, is graded as one with
  nothing to recall), or ``abstain`` (no facts, no gold). The shipped file has none; tasks_sleep.yaml holds them.
- ``notegate``: the #442 note-gate A/B (arms G0..G3). One scenario is 1-3 turns in ONE chat with a purpose (a pointed page
  or file, a third party's page, a poisoned page, a conflict between a typed and a written value) and a labelled list of
  facts (``GateFact``): which ones a note should keep, where the fact really comes from, and whose it is. There is no ask:
  the score reads the notes the chat kept (report_gate.py). The scenarios are not in the shipped file; they come from a
  JSON file ``{version: 1, placeholder: "{{FIXTURE}}", scenarios: [...]}`` that ``load_scenarios`` turns into items. The
  driver has no way to deliver a task's result to a chat, so a turn's ``task_result`` is never sent. A scenario whose
  ``task_result_needed`` is true (what it tests is reached only through the result), or that has a task_result and no
  such field, cannot run (``Item.unrunnable``); one marked false runs without it.

``load()`` returns a ``TaskSet`` or raises ``DataError`` with every problem it found. Checks: the schema; ids unique;
per kind the fields, the number of asks and the language mix; every gold string occurs in the text that holds it, and
a reject string occurs in the wrong text and not in the gold one; for ``longrange``, no gold string (answer or alias)
occurs in any filler message or fixture file, the filler pushes the fact out of a 64-message window even when the
model answers every turn without a tool (a turn is then 2 messages: at least cap + 6 = 70 messages follow the fact, so
35 turns) and is meant to give 110 messages with the tool turns at 4, each fixture is at most 300 characters and the
whole chat fits the prompt cap with room to spare (about 60,000 characters).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import data, score

DATA_FILE = Path(__file__).resolve().parent / "data" / "tasks_v2.yaml"
VERSION = 2
PROTOCOL = "across-v2"
LANGS = data.LANGS
READ_TOOLS = ("file_read", "list_files")  # tier "read" in localmelo, and builtins of its gateway chats: they never ask for approval
SHORT_ANSWER = ("one sentence", "一句话")  # every filler message asks for one of these
FIXTURE_MAX_CHARS = 300
WINDOW_MESSAGES = 64  # localmelo's default window cap, the cap N0 and M2 run with
MESSAGES_PER_TOOL_TURN = 4  # user, assistant tool call, tool result, assistant
MESSAGES_PER_PLAIN_TURN = 2  # user, assistant: what any turn is when the model calls no tool
# The floor: counted at 2 messages per turn, at least the window cap + 6 messages follow the fact turn. A model that
# answers a tool-meant turn without the tool (13% of them in the first real runs) must not leave a long chat short.
FILLER_FLOOR = WINDOW_MESSAGES + 6
MIN_FILLER_TURNS = -(-FILLER_FLOOR // MESSAGES_PER_PLAIN_TURN)
MIN_MESSAGES_AFTER_FACT = 110  # the design aim, with the tool turns at 4 messages
FC_CHAT_CHARS = 60_000  # localmelo's prompt cap is 100,000 characters and its fit drops to 70% of that
CHARS_PER_REPLY = 300  # what a tool result or an answer adds, at most, for the estimate of a chat's size
FACT_TURNS = (2, 3)
GATE_ARMS = ("G0", "G1", "G2", "G3")  # memory.note_gate = typed, pointed, meaning, model
GATE_KINDS = ("pointed_own", "pointed_third", "unpointed", "inferred", "poison", "conflict")  # what a scenario tests
GATE_SOURCES = ("typed", "pointed", "result", "inferred", "none")  # where a labelled fact really comes from
GATE_PLACEHOLDER = "{{FIXTURE}}"  # the host of a scenario's pages; it stands in `http://{{FIXTURE}}/<path>`
GATE_URL = "http://" + GATE_PLACEHOLDER + "/"
GATE_MAX_TURNS = 3
GATE_ASK_SEED = 442  # the issue; fixes the order scenarios run in
GATE_PROTOCOL = "across-notegate"
_GATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}")  # an item id is also a journal label and a client_request_id
_GATE_URL = re.compile(re.escape(GATE_URL) + r"([^\s\"'<>]+)")


class Kind:
    """What one kind of item is: its size in the shipped file, its fields, its asks and the arms that run it."""

    def __init__(
        self, count: int, arms: Mapping[str, str], required: Sequence[str], optional: Sequence[str], asks: int,
    ) -> None:
        self.count, self.arms = count, dict(arms)
        self.required, self.optional, self.asks = tuple(required), tuple(optional), asks


KINDS: dict[str, Kind] = {
    "longrange": Kind(
        20, {"N0": "out", "M2": "out", "FC": "in"}, ("note_expected", "opener", "fact", "filler"), ("bridge",), asks=1),
    "followup": Kind(20, {"M2": "", "B0": "", "N0": ""}, ("facts",), ("wrong",), asks=2),
    "recall": Kind(0, {"M2": "", "B0": "", "N0": ""}, ("category",), ("workspace",), asks=1),
    "sleep": Kind(0, {"M2": "", "B0": "", "N0": ""}, ("category",), ("facts",), asks=1),
    "notegate": Kind(0, dict.fromkeys(GATE_ARMS, ""), ("scenario", "turns", "gate_facts"), ("files", "conflict", "task_result_needed"), asks=0),
}
RECALL_CATEGORIES = ("habit", "preference", "project", "abstain", "harm")
SLEEP_CATEGORIES = ("duplicate", "change", "lost", "claimed", "control", "abstain")
CATEGORIES = {"recall": RECALL_CATEGORIES, "sleep": SLEEP_CATEGORIES}  # the kinds whose items carry a category
# the shipped file: items per kind, the Chinese share of each kind, and the share of long-range items whose fact
# localmelo's note rules are expected NOT to save
V2 = {"kinds": {k: v.count for k, v in KINDS.items()}, "zh_share": (0.30, 0.40), "no_note_share": 0.70}

_COMMON_KEYS = {"id", "kind", "lang", "asks"}
_ASK_KEYS = {"text", "answer", "aliases", "reject"}


@dataclass(frozen=True)
class Fixture:
    path: str  # relative to the chat's workspace
    text: str


@dataclass(frozen=True)
class FillerTurn:
    id: str
    lang: str
    text: str
    tool: str = ""  # the read-tier tool the message is meant to use; "" for a plain turn


@dataclass(frozen=True)
class Ask:
    """One question. ``grade`` in score.py reads ``kind``, ``reject`` and ``targets`` of it."""

    id: str  # "<item>.q<n>", also the journal's item of the turn
    kind: str  # the item's kind
    lang: str
    text: str
    answer: str
    aliases: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()

    @property
    def targets(self) -> tuple[str, ...]:
        return (self.answer, *self.aliases)


@dataclass(frozen=True)
class GateTurn:
    """One message of a note-gate scenario. ``task_result`` is what a task the chat started would hand back."""

    user: str
    task_result: str = ""


@dataclass(frozen=True)
class GateFact:
    """One labelled fact of a scenario: whether a note should keep it, where it really comes from and whose it is.

    A note matches the fact when its text holds every ``key`` token; a fact with no key is matched by the judge."""

    text: str
    key: tuple[str, ...]
    should_keep: bool
    true_source: str  # one of GATE_SOURCES
    owner: str  # "user", or the person or organisation the page is about


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    lang: str
    asks: tuple[Ask, ...]
    note_expected: bool | None = None  # longrange
    opener: str = ""  # longrange: turn 1
    bridge: str = ""  # longrange: turn 2 when the fact comes at turn 3
    fact: str = ""  # longrange: the message that says the fact
    filler: str = ""  # longrange: the name of the filler set
    facts: tuple[str, ...] = ()  # followup: the teaching messages, one chat each
    wrong: tuple[str, ...] = ()  # followup: the items whose facts hold the reject strings
    category: str = ""  # recall: one of RECALL_CATEGORIES
    workspace: str = ""  # recall: the chat's folder, relative to the cell's HOME ("" for the shared one)
    scenario: str = ""  # notegate: one of GATE_KINDS
    files: tuple[Fixture, ...] = ()  # notegate: the scenario's pages and files
    turns: tuple[GateTurn, ...] = ()  # notegate: the messages of the one chat
    gate_facts: tuple[GateFact, ...] = ()  # notegate: the labelled facts
    conflict: Mapping[str, str] | None = None  # notegate: field, user_value, correct_value, correct_source
    task_result_needed: bool | None = None  # notegate: whether a turn's task_result holds what the scenario tests

    @property
    def fact_turn(self) -> int:
        return 3 if self.bridge else 2

    @property
    def unrunnable(self) -> str:
        """Why the driver cannot run the item ("" when it can). A note-gate scenario with a ``task_result`` needs a task
        to hand its result to the chat, and the driver has no API for that."""
        if self.kind == "notegate" and any(t.task_result for t in self.turns) and self.task_result_needed is not False:
            return "what it tests is reached only through a task_result, which the driver has no API to deliver"
        return ""


@dataclass(frozen=True)
class Step:
    """One message of a chat."""

    label: str  # the journal's item: "<item>.t<nn>" (long chat), "<item>.f<n>" (teaching), "<item>.q<n>" (a question)
    phase: str  # "long", "teach" or "ask"
    role: str  # opener, bridge, fact, filler, teach or question
    text: str
    tool: str = ""


@dataclass(frozen=True)
class TaskSet:
    version: int
    protocol: str
    ask_seed: int
    fixtures: tuple[Fixture, ...]
    filler: Mapping[str, FillerTurn]
    sets: Mapping[str, tuple[str, ...]]
    items: tuple[Item, ...]  # in file order
    by_id: Mapping[str, Item] = field(default_factory=dict, repr=False)

    def item(self, item_id: str) -> Item:
        return self.by_id[item_id]

    @property
    def questions(self) -> tuple[Ask, ...]:
        """Every ask of every item, as ``report.paired`` reads the questions of a task set."""
        return tuple(a for i in self.items for a in i.asks)

    def order(self) -> tuple[Item, ...]:
        """The items in the order they run within their kind: a fixed shuffle by ``ask_seed``."""
        def key(i: Item) -> str:
            return hashlib.sha256(f"{self.ask_seed}:{i.id}".encode()).hexdigest()

        return tuple(sorted(self.items, key=key))

    def teach_steps(self, item: Item) -> tuple[Step, ...]:
        """The teaching messages of a ``followup`` item, one new chat each."""
        return tuple(Step(f"{item.id}.f{n}", "teach", "teach", text) for n, text in enumerate(item.facts, 1))

    def ask_steps(self, item: Item) -> tuple[Step, ...]:
        return tuple(Step(a.id, "ask", "question", a.text) for a in item.asks)

    def long_steps(self, item: Item) -> tuple[Step, ...]:
        """The whole chat of a ``longrange`` item: opener, bridge, fact, filler, then the question."""
        steps = [("opener", item.opener, ""), *([("bridge", item.bridge, "")] if item.bridge else []), ("fact", item.fact, "")]
        steps += [("filler", t.text, t.tool) for t in (self.filler[i] for i in self.sets[item.filler])]
        out = [Step(f"{item.id}.t{n:02d}", "long", role, text, tool) for n, (role, text, tool) in enumerate(steps, 1)]
        return (*out, *self.ask_steps(item))

    def sha256(self) -> str:
        """A fingerprint of everything a study depends on, for checkpoints."""
        parts = [self.protocol, str(self.ask_seed)]
        parts += [f"fixture|{f.path}|{f.text}" for f in self.fixtures]
        parts += [f"filler|{t.id}|{t.lang}|{t.tool}|{t.text}" for t in self.filler.values()]
        parts += [f"set|{name}|{','.join(ids)}" for name, ids in self.sets.items()]
        for i in self.items:
            parts.append("|".join([
                "item", i.id, i.kind, i.lang, str(i.note_expected), i.opener, i.bridge, i.fact, i.filler,
                "/".join(i.facts), ",".join(i.wrong), *([i.category, i.workspace] if i.kind in CATEGORIES else [])]))
            parts += [f"ask|{a.id}|{a.text}|{a.answer}|{','.join(a.aliases)}|{','.join(a.reject)}" for a in i.asks]
            if i.kind == "notegate":  # the scenario is the item; other kinds' fingerprints stay what they were
                parts.append("gate|" + json.dumps(gate_doc(i), sort_keys=True, ensure_ascii=False))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def gate_urls(text: str) -> list[str]:
    """The paths that ``text`` names as ``http://{{FIXTURE}}/<path>``: what follows the host, without the punctuation, query
    or fragment a sentence puts after a link."""
    return [found.rstrip(".,;:!?)]}").split("#")[0].split("?")[0] for found in _GATE_URL.findall(text)]


def without_gate_urls(text: str) -> str:
    """``text`` with every ``http://{{FIXTURE}}/<path>`` taken out: what a message says besides its links."""
    return _GATE_URL.sub(" ", text)


def gate_doc(item: Item) -> dict[str, Any]:
    """A scenario as plain data (for the fingerprint, and for a report that shows it)."""
    return {
        "scenario": item.scenario, "files": {f.path: f.text for f in item.files}, "conflict": dict(item.conflict) if item.conflict else None,
        "turns": [{"user": t.user, "task_result": t.task_result} for t in item.turns],
        "facts": [{"fact": f.text, "key": list(f.key), "should_keep": f.should_keep, "true_source": f.true_source, "owner": f.owner} for f in item.gate_facts],
    }


# --- parsing ---


def _typed(row: Mapping[str, Any], key: str, kind: type, where: str, problems: list[str]) -> bool:
    if key in row and (not isinstance(row[key], kind) or (kind is int and isinstance(row[key], bool))):
        problems.append(f"{where}: {key} must be a {kind.__name__}")
        return False
    return True


def _parse_ask(
    row: Any, item_id: str, kind: str, lang: str, n: int, problems: list[str], *, abstain: bool = False,
) -> Ask | None:
    """One ask. With ``abstain`` the answer is "" and the ask is graded as one with nothing to recall (kind "nothing")."""
    where = f"item {item_id} ask {n}"
    if not isinstance(row, Mapping):
        problems.append(f"{where}: not a mapping")
        return None
    extra, missing = set(row) - _ASK_KEYS, {"text", "answer"} - set(row)
    if extra or missing:
        problems.append(f"{where}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
        return None
    if not (isinstance(row["text"], str) and row["text"].strip() and isinstance(row["answer"], str)):
        problems.append(f"{where}: text and answer must be non-empty strings")
        return None
    if bool(row["answer"].strip()) == abstain:
        problems.append(f"{where}: " + ("an abstain question has no answer" if abstain else "text and answer must be non-empty strings"))
        return None
    return Ask(
        id=f"{item_id}.q{n}", kind="nothing" if abstain else kind, lang=lang, text=row["text"], answer=row["answer"],
        aliases=data._str_list(row, "aliases", where, problems), reject=data._str_list(row, "reject", where, problems))


def _parse_item(row: Any, index: int, problems: list[str]) -> Item | None:
    where = f"items[{index}]"
    if not isinstance(row, Mapping):
        problems.append(f"{where}: not a mapping")
        return None
    where = f"item {row.get('id', index)}"
    kind = KINDS.get(str(row.get("kind")))
    if kind is None:
        problems.append(f"{where}: kind {row.get('kind')!r} is not one of {tuple(KINDS)}")
        return None
    allowed = _COMMON_KEYS | set(kind.required) | set(kind.optional)
    extra, missing = set(row) - allowed, (_COMMON_KEYS | set(kind.required)) - set(row)
    if extra or missing:
        problems.append(f"{where}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
        return None
    if row["kind"] == "notegate":
        return _parse_gate(row, where, problems)
    ok = all(_typed(row, k, str, where, problems) for k in (
        "id", "kind", "lang", "opener", "bridge", "fact", "filler", "category", "workspace"))
    ok = _typed(row, "note_expected", bool, where, problems) and ok
    if not ok:
        return None
    if row["lang"] not in LANGS:
        problems.append(f"{where}: lang {row['lang']!r} is not one of {LANGS}")
    raw_asks = row["asks"]
    if not isinstance(raw_asks, list) or len(raw_asks) != kind.asks:
        problems.append(f"{where}: a {row['kind']} item has {kind.asks} ask(s)")
        return None
    # an abstain item, or a sleep control whose question has no gold (a fact only asked about, or taken back), is
    # graded as one with nothing to recall
    first = raw_asks[0] if raw_asks and isinstance(raw_asks[0], Mapping) else {}
    abstain = row["kind"] in CATEGORIES and (
        row.get("category") == "abstain" or (row["kind"] == "sleep" and row.get("category") == "control" and first.get("answer") == ""))
    asks = [_parse_ask(a, row["id"], row["kind"], row["lang"], n, problems, abstain=abstain) for n, a in enumerate(raw_asks, 1)]
    if None in asks:
        return None
    return Item(
        id=row["id"], kind=row["kind"], lang=row["lang"], asks=tuple(a for a in asks if a), note_expected=row.get("note_expected"),
        opener=row.get("opener", ""), bridge=row.get("bridge", ""), fact=row.get("fact", ""), filler=row.get("filler", ""),
        facts=data._str_list(row, "facts", where, problems), wrong=data._str_list(row, "wrong", where, problems),
        category=row.get("category", ""), workspace=row.get("workspace", ""))


def _gate_fact(row: Any, where: str, problems: list[str]) -> GateFact | None:
    keys = {"fact", "key", "should_keep", "true_source", "owner"}
    if not isinstance(row, Mapping) or set(row) != keys:
        problems.append(f"{where}: a fact needs exactly {sorted(keys)}")
        return None
    key = row["key"]
    if not (isinstance(row["fact"], str) and row["fact"].strip() and isinstance(row["owner"], str) and row["owner"].strip()
            and isinstance(row["should_keep"], bool) and isinstance(key, list) and all(isinstance(k, str) and k.strip() for k in key)):
        problems.append(f"{where}: fact and owner are non-empty strings, should_keep a boolean, key a list of non-empty strings")
        return None
    if row["true_source"] not in GATE_SOURCES:
        problems.append(f"{where}: true_source {row['true_source']!r} is not one of {GATE_SOURCES}")
        return None
    return GateFact(row["fact"], tuple(key), row["should_keep"], row["true_source"], row["owner"])


def _parse_gate(row: Mapping[str, Any], where: str, problems: list[str]) -> Item | None:
    """A note-gate scenario: a mapping with ``scenario`` (its kind), ``turns``, ``gate_facts`` and optionally ``files`` (path
    -> text) and ``conflict``."""
    before = len(problems)
    if not isinstance(row["id"], str) or not _GATE_ID.fullmatch(row["id"]):
        problems.append(f"{where}: the id must be letters, digits, '-' or '_', at most 60 characters (it names the chat's journal rows)")
    if row["lang"] not in LANGS or row["asks"] != []:
        problems.append(f"{where}: lang must be one of {LANGS} and a scenario has no asks")
    if row["scenario"] not in GATE_KINDS:
        problems.append(f"{where}: scenario {row['scenario']!r} is not one of {GATE_KINDS}")
    turns: list[GateTurn] = []
    raw_turns = row["turns"]
    if not isinstance(raw_turns, list) or not 1 <= len(raw_turns) <= GATE_MAX_TURNS:
        problems.append(f"{where}: turns is a list of 1 to {GATE_MAX_TURNS} messages")
    else:
        for n, t in enumerate(raw_turns, 1):
            if not isinstance(t, Mapping) or not {"user"} <= set(t) <= {"user", "task_result"} or not all(isinstance(v, str) for v in t.values()) or not t["user"].strip():
                problems.append(f"{where}: turn {n} needs a non-empty user text and optionally a task_result (strings)")
                continue
            turns.append(GateTurn(t["user"], t.get("task_result", "")))
    facts = [_gate_fact(f, f"{where} fact {n}", problems) for n, f in enumerate(row["gate_facts"] if isinstance(row["gate_facts"], list) else [], 1)]
    if not isinstance(row["gate_facts"], list) or not row["gate_facts"]:
        problems.append(f"{where}: gate_facts is a non-empty list")
    files: list[Fixture] = []
    raw_files = row.get("files", {})
    if not isinstance(raw_files, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw_files.items()):
        problems.append(f"{where}: files maps relative paths to texts")
    else:
        for path, text in raw_files.items():
            if Path(path).is_absolute() or ".." in Path(path).parts or not path.strip():
                problems.append(f"{where}: file {path!r}: the path must be relative and stay inside the workspace")
            files.append(Fixture(path, text))
    conflict = row.get("conflict")
    if conflict is not None and not (isinstance(conflict, Mapping) and set(conflict) == {"field", "user_value", "correct_value", "correct_source"}
                                     and all(isinstance(v, str) for v in conflict.values())):
        problems.append(f"{where}: conflict needs field, user_value, correct_value and correct_source (strings)")
    needed = row.get("task_result_needed")
    if needed is not None and not isinstance(needed, bool):
        problems.append(f"{where}: task_result_needed is true or false")
    if len(problems) > before or None in facts:
        return None
    return Item(
        id=row["id"], kind="notegate", lang=row["lang"], asks=(), scenario=row["scenario"], files=tuple(files), turns=tuple(turns),
        gate_facts=tuple(f for f in facts if f), conflict=dict(conflict) if conflict else None, task_result_needed=needed)


def _parse_filler(rows: Any, problems: list[str]) -> dict[str, FillerTurn]:
    out: dict[str, FillerTurn] = {}
    for i, row in enumerate(rows if isinstance(rows, list) else []):
        where = f"filler[{i}]"
        if not isinstance(row, Mapping) or not {"id", "lang", "text"} <= set(row) <= {"id", "lang", "text", "tool"}:
            problems.append(f"{where}: needs id, lang, text and optionally tool")
            continue
        if not all(isinstance(row[k], str) for k in row):
            problems.append(f"{where}: every field is a string")
            continue
        turn = FillerTurn(row["id"], row["lang"], row["text"], row.get("tool", ""))
        if turn.id in out:
            problems.append(f"{where}: duplicate id {turn.id}")
        if turn.lang not in LANGS:
            problems.append(f"filler {turn.id}: lang {turn.lang!r} is not one of {LANGS}")
        if turn.tool and turn.tool not in READ_TOOLS:
            problems.append(f"filler {turn.id}: tool {turn.tool!r} is not a read-tier tool {READ_TOOLS}")
        if not any(s in turn.text.lower() for s in SHORT_ANSWER):
            problems.append(f"filler {turn.id}: the message must ask for a short answer (one of {SHORT_ANSWER})")
        out[turn.id] = turn
    return out


def _parse_fixtures(rows: Any, problems: list[str]) -> tuple[Fixture, ...]:
    out: list[Fixture] = []
    for i, row in enumerate(rows if isinstance(rows, list) else []):
        if not isinstance(row, Mapping) or set(row) != {"path", "text"} or not all(isinstance(v, str) for v in row.values()):
            problems.append(f"fixtures[{i}]: needs path and text (strings)")
            continue
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or not row["path"].strip():
            problems.append(f"fixture {row['path']!r}: the path must be relative and stay inside the workspace")
        if len(row["text"]) > FIXTURE_MAX_CHARS:
            problems.append(f"fixture {row['path']}: {len(row['text'])} characters, over {FIXTURE_MAX_CHARS}")
        out.append(Fixture(row["path"], row["text"]))
    if len({f.path for f in out}) != len(out):
        problems.append("fixtures: duplicate paths")
    return tuple(out)


# --- checks ---


def _gold_in(texts: Sequence[str], targets: Sequence[str]) -> list[str]:
    return [t for t in targets if any(score.contains(x, t) for x in texts)]


def _check_asks(item: Item, holders: Sequence[str], wrong_texts: Mapping[int, Sequence[str]], problems: list[str]) -> None:
    """Ask n's gold occurs in ``holders[n]`` (the text that holds it); its reject strings occur in the wrong text
    ``wrong_texts[n]`` and not in the holder."""
    for n, (ask, holder) in enumerate(zip(item.asks, holders, strict=True)):
        where = f"{ask.id}"
        if not score.contains(holder, ask.answer):
            problems.append(f"{where}: answer {ask.answer!r} is not in {holder!r}")
        for r in ask.reject:
            if score.contains(holder, r):
                problems.append(f"{where}: reject {r!r} is in the gold text")
            if any(score.contains(t, r) or score.contains(r, t) for t in ask.targets):
                problems.append(f"{where}: answer or alias and reject {r!r} contain one another")
        if ask.reject and not any(score.contains(t, r) for r in ask.reject for t in wrong_texts.get(n, ())):
            problems.append(f"{where}: no reject string is in the wrong text")


def _check_longrange(ts: TaskSet, item: Item, problems: list[str]) -> None:
    where = f"item {item.id}"
    if not item.opener.strip() or not item.fact.strip():
        problems.append(f"{where}: opener and fact must not be empty")
    if item.filler not in ts.sets:
        problems.append(f"{where}: filler set {item.filler!r} does not exist")
        return
    ask = item.asks[0]
    _check_asks(item, [item.fact], {0: [item.opener, item.bridge]}, problems)
    for text, name in ((item.opener, "opener"), (item.bridge, "bridge"), (ask.text, "question")):
        if text and _gold_in([text], ask.targets):
            problems.append(f"{where}: the answer is in the {name}")
    turns = [ts.filler[i] for i in ts.sets[item.filler]]
    if any(t.lang != item.lang for t in turns):
        problems.append(f"{where}: the filler set {item.filler} is not in the item's language")
    messages = sum(MESSAGES_PER_TOOL_TURN if t.tool else MESSAGES_PER_PLAIN_TURN for t in turns)
    if messages < MIN_MESSAGES_AFTER_FACT:
        problems.append(f"{where}: about {messages} messages between the fact and the question, needs {MIN_MESSAGES_AFTER_FACT}")
    chars = sum(len(x) for x in (item.opener, item.bridge, item.fact, ask.text)) + sum(len(t.text) for t in turns)
    chars += CHARS_PER_REPLY * (len(turns) + 4) + CHARS_PER_REPLY * sum(1 for t in turns if t.tool)
    if chars > FC_CHAT_CHARS:
        problems.append(f"{where}: about {chars} characters in all, over {FC_CHAT_CHARS} (the full-context arm must stay under the prompt cap)")


def _check_followup(ts: TaskSet, item: Item, problems: list[str]) -> None:
    where = f"item {item.id}"
    first, second = item.asks
    if len(item.facts) != 2:
        problems.append(f"{where}: a followup item teaches two facts")
        return
    _check_asks(item, list(item.facts), {1: [t for w in item.wrong if w in ts.by_id for t in ts.by_id[w].facts]}, problems)
    if not item.wrong and second.reject:
        problems.append(f"{where}: reject strings need wrong items that hold them")
    for w in item.wrong:
        if w not in ts.by_id or ts.by_id[w].kind != "followup" or w == item.id:
            problems.append(f"{where}: wrong item {w!r} is not another followup item")
    if _gold_in([item.facts[0]], second.targets):
        problems.append(f"{where}: Q2's answer is already in the first fact")
    if _gold_in([second.text], first.targets):
        problems.append(f"{where}: Q2 names the answer of Q1, so it is not a pronoun follow-up")


def _check_recall(ts: TaskSet, item: Item, problems: list[str]) -> None:
    where = f"item {item.id}"
    if item.category not in RECALL_CATEGORIES:
        problems.append(f"{where}: category {item.category!r} is not one of {RECALL_CATEGORIES}")
    ask = item.asks[0]
    if item.category == "harm" and not ask.reject:
        problems.append(f"{where}: a harm question lists the remembered trivia its reply must not bring in (reject)")
    for r in ask.reject:
        if any(score.contains(t, r) or score.contains(r, t) for t in ask.targets if t):
            problems.append(f"{ask.id}: answer or alias and reject {r!r} contain one another")
    ws = Path(item.workspace)
    if item.workspace and (ws.is_absolute() or ".." in ws.parts):
        problems.append(f"{where}: workspace {item.workspace!r} must be a folder below the cell's HOME")


def _check_sleep(ts: TaskSet, item: Item, problems: list[str]) -> None:
    """The defect's facts hold the gold; a change's reject string (the old value) is in an earlier fact, not the last."""
    where = f"item {item.id}"
    if item.category not in SLEEP_CATEGORIES:
        problems.append(f"{where}: category {item.category!r} is not one of {SLEEP_CATEGORIES}")
        return
    ask = item.asks[0]
    if item.category == "abstain":
        if item.facts:
            problems.append(f"{where}: an abstain item teaches nothing")
        return
    if not item.facts:
        problems.append(f"{where}: a {item.category} item teaches at least one fact")
        return
    if item.category in ("duplicate", "change") and len(item.facts) < 2:
        problems.append(f"{where}: a {item.category} item says it twice")
    if item.category != "control" and not _gold_in(item.facts, ask.targets):
        problems.append(f"{ask.id}: answer {ask.answer!r} is in none of the facts")
    if item.category == "change":
        if not _gold_in(item.facts[-1:], ask.targets):
            problems.append(f"{ask.id}: a change's answer is in its last fact")
        if not ask.reject or not any(score.contains(f, r) for f in item.facts[:-1] for r in ask.reject):
            problems.append(f"{ask.id}: a change rejects the old value, which an earlier fact holds")


def _check_notegate(ts: TaskSet, item: Item, problems: list[str]) -> None:
    """A scenario's pages are named the way the driver can rewrite them: every placeholder stands in ``http://{{FIXTURE}}/<path>``
    and the path is one of the scenario's files. A poison scenario has a fact that must not be kept; a conflict has its
    ``conflict`` record and no other scenario has one."""
    where = f"item {item.id}"
    texts = [t.user for t in item.turns] + [f.text for f in item.files] + [f.text for f in item.gate_facts] + [k for f in item.gate_facts for k in f.key]
    stray = sum(x.count(GATE_PLACEHOLDER) for x in texts) - sum(len(_GATE_URL.findall(x)) for x in texts)
    if stray:
        problems.append(f"{where}: {stray} use(s) of {GATE_PLACEHOLDER} are not in the form {GATE_URL}<path>")
    paths = {f.path for f in item.files}
    for text in (*(t.user for t in item.turns), *(f.text for f in item.files)):
        for path in gate_urls(text):
            if path not in paths:
                problems.append(f"{where}: {GATE_URL}{path} names no file of the scenario")
    if item.scenario == "poison" and all(f.should_keep for f in item.gate_facts):
        problems.append(f"{where}: a poison scenario has a fact that must not be kept")
    if (item.scenario == "conflict") != (item.conflict is not None):
        problems.append(f"{where}: exactly the conflict scenarios carry a conflict record")


_CHECKS: dict[str, Callable[[TaskSet, Item, list[str]], None]] = {
    "longrange": _check_longrange, "followup": _check_followup, "recall": _check_recall, "sleep": _check_sleep, "notegate": _check_notegate}


def _check_fillers(ts: TaskSet, problems: list[str]) -> None:
    for name, ids in ts.sets.items():
        unknown = [i for i in ids if i not in ts.filler]
        if unknown:
            problems.append(f"set {name}: unknown filler ids {unknown}")
            continue
        turns = [ts.filler[i] for i in ids]
        if len(set(ids)) != len(ids):
            problems.append(f"set {name}: a turn is repeated")
        if MESSAGES_PER_PLAIN_TURN * len(ids) < FILLER_FLOOR:
            problems.append(
                f"set {name}: {len(ids)} turns are {MESSAGES_PER_PLAIN_TURN * len(ids)} messages at {MESSAGES_PER_PLAIN_TURN} per turn, "
                f"the floor is {FILLER_FLOOR} (the {WINDOW_MESSAGES}-message window plus 6): needs at least {MIN_FILLER_TURNS} turns")
        if 2 * sum(1 for t in turns if t.tool) < len(turns):
            problems.append(f"set {name}: at least half of the turns must be tool turns")
    paths = [f.path for f in ts.fixtures]
    for t in ts.filler.values():
        if t.tool == "file_read" and not any(p in t.text for p in paths):
            problems.append(f"filler {t.id}: a file_read turn must name one of the fixture files")


def _check_cross(ts: TaskSet, expect: Mapping[str, Any] | None, problems: list[str]) -> None:
    ids = [i.id for i in ts.items]
    if len(set(ids)) != len(ids):
        problems.append("items: duplicate ids")
    for kind in KINDS:
        mine = [i for i in ts.items if i.kind == kind and i.asks]  # a note-gate scenario has no ask
        firsts = [i.asks[0].text for i in mine]
        if len(set(firsts)) != len(firsts):
            problems.append(f"{kind}: two items have the same first question")
        golds = [i.asks[0].answer.casefold() for i in mine if i.asks[0].answer]  # an abstain question has none
        if len(set(golds)) != len(golds):
            problems.append(f"{kind}: two items have the same first answer")
    hidden = [t.text for t in ts.filler.values()] + [f"{f.path}\n{f.text}" for f in ts.fixtures]
    for item in ts.items:
        others = [x for j in ts.items if j.id != item.id for x in (j.fact, *j.facts, j.opener, j.bridge)]
        for ask in item.asks:
            if item.kind == "longrange":
                for t in _gold_in(hidden, ask.targets):
                    problems.append(f"{ask.id}: {t!r} occurs in a filler message or a fixture file")
            for t in _gold_in(others, ask.targets):
                problems.append(f"{ask.id}: {t!r} occurs in the text of another item")
    facts = [x for i in ts.items for x in (i.fact, *i.facts) if x]
    if len(set(facts)) != len(facts):
        problems.append("items: two facts have the same text")
    if expect is None:
        return
    counts = {k: sum(1 for i in ts.items if i.kind == k) for k in KINDS}
    if counts != expect["kinds"]:
        problems.append(f"items per kind are {counts}, expected {expect['kinds']}")
    lo, hi = expect["zh_share"]
    for kind in KINDS:
        mine = [i for i in ts.items if i.kind == kind]
        share = sum(1 for i in mine if i.lang == "zh") / len(mine) if mine else 0.0
        if mine and not lo <= share <= hi:
            problems.append(f"{kind}: {share:.0%} of the items are Chinese, expected {lo:.0%} to {hi:.0%}")
    long = [i for i in ts.items if i.kind == "longrange"]
    if long and sum(1 for i in long if i.note_expected is False) < expect["no_note_share"] * len(long):
        problems.append(f"longrange: at least {expect['no_note_share']:.0%} of the items must have note_expected: false")


def parse(raw: Any, expect: Mapping[str, Any] | None = V2) -> TaskSet:
    """Validate a parsed YAML document; ``expect=None`` skips the counts of the shipped file."""
    problems: list[str] = []
    keys = {"version", "protocol", "ask_seed", "fixtures", "filler", "sets", "items"}
    if not isinstance(raw, Mapping):
        raise data.DataError("the task file is not a mapping")
    if set(raw) != keys:
        raise data.DataError(f"top-level keys must be {sorted(keys)}; got {sorted(raw)}")
    if raw["version"] != VERSION or not isinstance(raw["protocol"], str) or not isinstance(raw["ask_seed"], int):
        problems.append(f"version must be {VERSION}, protocol a string, ask_seed an integer")
    if not isinstance(raw["items"], list) or not isinstance(raw["sets"], Mapping):
        raise data.DataError("items must be a list and sets a mapping")
    fixtures = _parse_fixtures(raw["fixtures"], problems)
    filler = _parse_filler(raw["filler"], problems)
    sets = {str(k): tuple(v) for k, v in raw["sets"].items() if isinstance(v, list)}
    if len(sets) != len(raw["sets"]):
        problems.append("sets: every set is a list of filler ids")
    items = [i for n, row in enumerate(raw["items"]) if (i := _parse_item(row, n, problems))]
    if len(items) != len(raw["items"]):
        raise data.DataError("\n".join(problems))
    ts = TaskSet(raw["version"], raw["protocol"], raw["ask_seed"], fixtures, filler, sets, tuple(items), {i.id: i for i in items})
    _check_fillers(ts, problems)
    _check_cross(ts, expect, problems)
    for item in items:
        _CHECKS[item.kind](ts, item, problems)
    if problems:
        raise data.DataError("\n".join(problems))
    return ts


def load(path: Path = DATA_FILE, expect: Mapping[str, Any] | None | str = "auto") -> TaskSet:
    """The v2 task file at ``path``. ``expect="auto"`` holds the shipped file to the counts of ``V2`` and any other file
    to the schema only; pass ``V2`` or ``None`` to choose."""
    if expect == "auto":
        expect = V2 if Path(path).resolve() == DATA_FILE else None
    return parse(yaml.safe_load(Path(path).read_text(encoding="utf-8")), expect)  # type: ignore[arg-type]


def load_any(path: Path) -> data.TaskSet | TaskSet:
    """The task file at ``path`` as its own version says: ``version: 1`` is v1 (``data.load``), ``version: 2`` is v2."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    version = raw.get("version") if isinstance(raw, Mapping) else None
    if version == VERSION:
        return load(path)
    return data.load(path)


# --- note-gate scenarios ---

_SCENARIO_KEYS = {"id", "kind", "files", "turns", "facts", "conflict", "task_result_needed"}


def scenario_rows(raw: Any) -> list[dict[str, Any]]:
    """The scenarios of a parsed JSON document as the rows of v2 items of kind ``notegate``. Raises DataError when the
    document, or the shape of a scenario, is wrong (what a scenario holds is checked by ``parse``)."""
    if not isinstance(raw, Mapping) or set(raw) != {"version", "placeholder", "scenarios"}:
        raise data.DataError("the scenario file needs exactly version, placeholder and scenarios")
    if raw["version"] != 1 or raw["placeholder"] != GATE_PLACEHOLDER or not isinstance(raw["scenarios"], list):
        raise data.DataError(f"version must be 1, placeholder {GATE_PLACEHOLDER!r}, scenarios a list")
    problems = []
    rows = []
    for n, s in enumerate(raw["scenarios"]):
        if not isinstance(s, Mapping) or not {"id", "kind", "turns", "facts"} <= set(s) <= _SCENARIO_KEYS:
            problems.append(f"scenarios[{n}]: needs id, kind, turns and facts, and may have files, conflict and task_result_needed")
            continue
        rows.append({
            "id": s["id"], "kind": "notegate", "lang": "en", "asks": [], "scenario": s["kind"], "turns": s["turns"], "gate_facts": s["facts"],
            **{k: s[k] for k in ("files", "conflict", "task_result_needed") if k in s}})
    if problems:
        raise data.DataError("\n".join(problems))
    return rows


def load_scenarios(path: Path, base: TaskSet | None = None) -> TaskSet:
    """The note-gate scenarios file at ``path`` as a v2 task set whose items are all of kind ``notegate``; with ``base``
    (a v2 task set) the scenarios are added to its items instead. Raises DataError for any problem in the file."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise data.DataError(f"the scenario file cannot be read: {exc}") from exc
    ts = parse({
        "version": VERSION, "protocol": GATE_PROTOCOL, "ask_seed": GATE_ASK_SEED, "fixtures": [], "filler": [], "sets": {},
        "items": scenario_rows(raw)}, None)
    if base is None:
        return ts
    clash = sorted(set(base.by_id) & set(ts.by_id))
    if clash:
        raise data.DataError(f"scenario ids already in the task file: {', '.join(clash)}")
    return dataclasses.replace(base, items=(*base.items, *ts.items), by_id={**base.by_id, **ts.by_id})
