"""Deterministic scoring for the memory-across-chats study. No LLM judge.

Answer correctness. A reply is correct when it contains the gold answer or one of its aliases and none of
the question's ``reject`` strings (the stale or look-alike values). Matching is on normalized text, in the
spirit of ``agent_ability/web/score.py`` (NFKD fold, lower case, punctuation dropped): an ASCII target must
match whole words, so "14" is not found in "214"; a target with CJK characters is a substring match with
the spaces removed. A reply that contains both the gold and a reject string is ``mixed`` and counts as
wrong, because the newer value must win and a look-alike must not be used.

A question with nothing to recall is answered correctly when the reply says it does not know (a phrase from
``ABSTAIN_EN`` or ``ABSTAIN_ZH``) and gives none of its ``reject`` strings. Any other reply invented something.
A hedge followed by a guess still counts as not knowing unless the guess is one of the reject strings (the
values taught for nearby facts), so the figure is a lenient bound on inventing.

Recall. ``used`` is the ``turn.memory`` list of what the gateway put into the message, in rank order, each
item with the ``session_id`` of the chat it came from. A fact counts as recalled at k when an item among the
first k comes from the chat that taught it. Items from other teaching chats are wrong recall; on a question
with nothing to recall any item at all is false recall.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Outcomes of one reply
CORRECT = "correct"
WRONG = "wrong"
ABSTAIN = "abstain"  # an answerable question answered with "I don't know"
MIXED = "mixed"  # the gold and a reject string together
EMPTY = "empty"  # no reply text (the turn failed or the reply was only thinking)

# Phrases that say "I do not know". Matched on normalized text, so apostrophes and case do not matter.
ABSTAIN_EN = (
    "don t know", "do not know", "don t have", "do not have", "haven t told", "have not told",
    "haven t mentioned", "have not mentioned", "haven t shared", "have not shared", "hasn t been mentioned",
    "didn t mention", "did not mention", "no record", "no information", "no details", "not aware",
    "not sure", "can t find", "cannot find", "couldn t find", "could not find", "unable to find",
    "don t recall", "do not recall", "don t see", "do not see", "no memory", "nothing in my memory",
    "not in my memory", "no way of knowing", "no way to know", "can t tell", "cannot tell",
    "i m not certain", "i do not see", "never told me", "not told me", "has not come up", "hasn t come up",
)
ABSTAIN_ZH = (
    "不知道", "不清楚", "不确定", "无法确定", "没有告诉", "没告诉", "没有提过", "没提过", "没有提到", "没提到",
    "未提及", "没有记录", "没有相关", "没有关于", "没有找到", "找不到", "记不得", "不记得", "尚未告诉",
    "还没告诉", "还没有告诉", "还没提", "没有这方面", "无从得知",
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PUNCT_OR_SYMBOL = ("P", "S")


def strip_thinking(text: str | None) -> str:
    """The reply without a reasoning block: closed ``<think>`` blocks, or everything up to a lone ``</think>``."""
    text = text or ""
    text = _THINK.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    if "<think>" in text.lower():  # an unfinished block: there is no answer after it
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


def normalize(text: str) -> str:
    """NFKD fold (accents dropped), lower case, punctuation and symbols as spaces, whitespace collapsed."""
    folded = unicodedata.normalize("NFKD", text).replace("’", "'")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = unicodedata.normalize("NFKC", folded).casefold()
    spaced = "".join(" " if unicodedata.category(c)[0] in _PUNCT_OR_SYMBOL else c for c in folded)
    return " ".join(spaced.split())


def contains(text: str, target: str) -> bool:
    """Whether the normalized ``text`` holds the normalized ``target``: whole words for ASCII, else a substring."""
    want = normalize(target)
    have = normalize(text)
    if not want or not have:
        return False
    if want.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(want)}(?![a-z0-9])", have) is not None
    return want.replace(" ", "") in have.replace(" ", "")


def abstained(text: str) -> bool:
    """Whether the reply says it does not know."""
    folded = normalize(text)
    squashed = folded.replace(" ", "")
    return any(p in folded for p in ABSTAIN_EN) or any(p in squashed for p in ABSTAIN_ZH)


@dataclass(frozen=True)
class Grade:
    outcome: str
    correct: bool
    matched: str | None = None  # the gold or alias found
    rejected: tuple[str, ...] = ()  # the reject strings found
    abstained: bool = False


def grade(question: Any, reply: str | None) -> Grade:
    """Grade one reply against a question (``data.Question`` or any object with the same fields)."""
    text = strip_thinking(reply)
    if not text:
        return Grade(EMPTY, False)
    said_unknown = abstained(text)
    rejected = tuple(r for r in question.reject if contains(text, r))
    if question.kind == "nothing":
        ok = said_unknown and not rejected
        return Grade(CORRECT if ok else WRONG, ok, None, rejected, said_unknown)
    matched = next((t for t in question.targets if contains(text, t)), None)
    if matched is not None and rejected:
        return Grade(MIXED, False, matched, rejected, said_unknown)
    if matched is not None:
        return Grade(CORRECT, True, matched, (), said_unknown)
    if said_unknown:
        return Grade(ABSTAIN, False, None, rejected, True)
    return Grade(WRONG, False, None, rejected, False)


# --- recall ---

HIT_KS = (1, 3, 5)


def used_items(memory: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The ``used`` list of a ``turn.memory`` payload; empty when the event was absent."""
    items = (memory or {}).get("used")
    return [i for i in items if isinstance(i, Mapping)] if isinstance(items, list) else []


def recall_score(
    memory: Mapping[str, Any] | None,
    gold_sessions: Sequence[str],
    wrong_sessions: Sequence[str] = (),
    taught_sessions: Sequence[str] = (),
    *,
    nothing: bool = False,
) -> dict[str, Any]:
    """Recall facts about one asked question.

    ``gold_sessions`` are the chats that taught the gold facts, ``wrong_sessions`` those of the stale or
    look-alike facts, ``taught_sessions`` every teaching chat. ``hit_at_k`` is True when every gold chat is
    among the sources of the first k items; it is None for a question with nothing to recall.
    """
    items = used_items(memory)
    sources = [str(i.get("session_id") or "") for i in items]
    taught = set(taught_sessions)
    gold = set(gold_sessions)
    wrong = set(wrong_sessions)
    out: dict[str, Any] = {
        "n_used": len(items),
        "n_taught": sum(1 for s in sources if s in taught),
        "n_other": sum(1 for s in sources if s not in taught),
        "wrong_recall": any(s in wrong for s in sources),
        "other_taught": sum(1 for s in sources if s in taught and s not in gold),
    }
    if nothing:
        out["false_recall"] = bool(items)
        out["false_recall_taught"] = out["n_taught"] > 0
        return out
    for k in HIT_KS:
        out[f"hit_at_{k}"] = bool(gold) and gold <= set(sources[:k])
    out["hit_any"] = bool(gold) and gold <= set(sources)
    ranks = [sources.index(g) + 1 for g in gold if g in sources]
    out["gold_rank"] = max(ranks) if gold and len(ranks) == len(gold) else None
    return out


# --- in-chat kinds (tasks_v2): where a right answer came from, and why a wrong one was wrong ---

# Why a wrong answer was wrong. With memory off (N0, FC) the working window decides: the fact had left it, or it was
# still in it and not used. With memory on: if the chat's item is in ``turn.memory.used`` the answer is wrong
# although the fact was recalled, whether as a note or, on builds that recall a chat's own evicted turns, as the
# turn itself (so no note is needed); if it is not, the note was never made (no record from the chat that said the
# fact holds the gold string) or it was made and the recall did not bring it.
MISS_NOT_SAVED = "not saved"
MISS_NOT_RECALLED = "not recalled"
MISS_NOT_USED = "recalled but not used"
MISS_OUT_OF_WINDOW = "out of window"
MISS_IN_WINDOW = "in window but not used"
MISS_MEMORY_OFF = "memory off"  # memory off and no window to look at (a follow-up item)
MISS_FAILED = "turn failed"
MISS_UNCHECKED = "unchecked"  # the cell's stores were not read
MISS_CLASSES = (
    MISS_NOT_SAVED, MISS_NOT_RECALLED, MISS_NOT_USED, MISS_OUT_OF_WINDOW, MISS_IN_WINDOW, MISS_MEMORY_OFF, MISS_FAILED,
    MISS_UNCHECKED)

# Where a right answer came from, as far as the events say: a recalled note (a ``record`` item of ``turn.memory.used``
# that holds the gold; its source reads "you said it ..."), the chat's own turn (a ``chat`` item whose source reads
# "this chat, <day>", which newer builds recall after the turn has left the window; or the turn still in the window),
# a recalled turn of another chat, or unknown. localmelo's legacy step memory (a summary per tool step, recalled by an
# embedding model; M2 only) sends no event, so for an arm with an embedding model "unknown" can be that.
THIS_CHAT = "this chat"  # the start of the source line of a used item that is a turn of the question's own chat
SRC_NOTE = "note"
SRC_CHAT = "chat turn"
SRC_WINDOW = "own-chat turn"
SRC_UNKNOWN = "unknown"
SRC_STEP = "unknown (step summary possible)"
SOURCES = (SRC_NOTE, SRC_CHAT, SRC_WINDOW, SRC_UNKNOWN, SRC_STEP)


def miss_class(*, memory_on: bool, failed: bool, saved: bool | None, recalled: bool, in_window: bool | None) -> str:
    """The class of one wrong answer: ``saved`` (a note with the gold exists; None: stores not read), ``recalled`` (the
    gold chat's item is in ``used``), ``in_window`` (the fact turn is in the working window; None: not known).

    Order: a failed turn; memory off (the window classes); recalled (not used, whatever notes exist); then not recalled,
    which is "not saved" without a note and "not recalled" with one."""
    if failed:
        return MISS_FAILED
    if not memory_on:
        return MISS_MEMORY_OFF if in_window is None else MISS_IN_WINDOW if in_window else MISS_OUT_OF_WINDOW
    if recalled:
        return MISS_NOT_USED
    if saved is None:
        return MISS_UNCHECKED
    return MISS_NOT_RECALLED if saved else MISS_NOT_SAVED


def answer_source(memory: Mapping[str, Any] | None, targets: Sequence[str], *, in_window: bool | None, step_memory: bool) -> str:
    """Where a right answer came from; ``step_memory``: the arm has an embedding model, so legacy step memory runs."""
    held = [i for i in used_items(memory) if any(contains(str(i.get("text") or ""), t) for t in targets)]
    if any(i.get("kind") == "record" for i in held):
        return SRC_NOTE
    if any(str(i.get("source") or "").startswith(THIS_CHAT) for i in held):
        return SRC_WINDOW
    if any(i.get("kind") == "chat" for i in held):
        return SRC_CHAT
    if in_window:
        return SRC_WINDOW
    return SRC_STEP if step_memory else SRC_UNKNOWN


def block_chars(memory: Mapping[str, Any] | None, recall: Mapping[str, Any] | None) -> tuple[int | None, bool]:
    """(characters of the recall block, whether that is an estimate). localmelo reports ``recall.block_chars``; a build
    that does not gets an estimate, the used items' text plus their source lines; None when there was no recall."""
    reported = (recall or {}).get("block_chars")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        return int(reported), False
    if memory is None:
        return None, False
    return sum(len(str(i.get("text") or "")) + len(str(i.get("source") or "")) for i in used_items(memory)), True

