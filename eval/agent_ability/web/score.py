"""Deterministic scoring of one answer against a gold answer and its aliases (DESIGN.md, section 7). No LLM judge.

Both sides are pre-normalized, then compared by the row's ``kind``:

- ``number``: ``$``, ``%`` and ``,`` are dropped; what is left must be one number, plus an optional unit of one
  to three plain words. The tolerance depends on the gold: exact up to 50, within 1% up to 10,000, within 5%
  above. A hedge such as ``12 or 13`` is not one number and fails.
- ``year``: exact. A tolerance would accept 1968 for 1987.
- ``date``: day, month name and year in any order, or ISO (``1962-03-03``). Each component the gold has must match;
  extra components in the answer are ignored.
- ``string``: GAIA's ``normalize_str`` on both sides, then equality.
- ``list``: GAIA's rule. Split on ``,`` and ``;``, require equal length, then compare element by element, in order,
  or as multisets when ``unordered``.

An answer is correct when it matches the gold or any alias. Aliases are hand-written per row (data/).

Sources of the rules:

- ``normalize_str`` and the list comparison are GAIA's, from ``scorer.py`` of the GAIA leaderboard space
  (https://huggingface.co/spaces/gaia-benchmark/leaderboard): whitespace and punctuation are removed and the text
  is lower-cased; a list is split on ``,`` and ``;``, the lengths must be equal, and each pair of elements is
  compared as a number when the gold element is one, else as a string with punctuation kept.
- The three tolerance tiers follow SimpleQA Verified (https://arxiv.org/abs/2509.07968, Table 6): exact for small
  counts, about 1% for medium quantities and about 5% for very large ones. The paper sets a range per question for
  an LLM judge; here one rule on the gold's size stands in for it, with the cut-offs 50 and 10,000 chosen for this
  suite. Unlike that judge, a reply that hedges between candidates does not pass.

Differences from GAIA, all small: pre-normalization (below); ``normalize_str`` also drops non-ASCII punctuation
such as curly quotes; and in a list, a leading "and " on an element is dropped, so "a, b, and c" equals "a, b, c".
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from . import answer as answer_mod

KINDS = ("number", "year", "date", "string", "list")
PASS = "pass"
FAIL = "fail"

_NUMBER = re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*(.*))?", re.DOTALL)
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")
_YEAR = re.compile(r"\d{1,4}")
_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?")
_DAY = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?")
_UNIT_WORDS_MAX = 3
# Words that turn "number plus unit" into a hedge or a range ("12 or thirteen", "12 to 13").
_HEDGE_WORDS = frozenset(
    ("or", "nor", "and", "to", "either", "between", "maybe", "perhaps", "possibly", "probably")
    + ("about", "around", "approximately", "roughly", "versus", "vs")
)
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_ASCII_PUNCTUATION = frozenset(string.punctuation)


def prenormalize(text: str) -> str:
    """NFKD fold (accents dropped), trim, drop one trailing ``.`` and a leading ``the ``."""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c)).strip()
    if folded.endswith("."):
        folded = folded[:-1].rstrip()
    return re.sub(r"^the\s+", "", folded, flags=re.IGNORECASE).strip()


def normalize_str(text: str, remove_punct: bool = True) -> str:
    """GAIA's ``normalize_str``: no whitespace, lower case, and (by default) no punctuation."""
    lowered = re.sub(r"\s", "", text).lower()
    if not remove_punct:
        return lowered
    return "".join(
        c for c in lowered if c not in _ASCII_PUNCTUATION and not unicodedata.category(c).startswith("P")
    )


# -- number ---------------------------------------------------------------------------------------------------


def parse_number(text: str) -> Fraction | None:
    """The value of "one number plus an optional unit" after ``$``, ``%`` and ``,`` are dropped, else None."""
    cleaned = text.replace("−", "-").translate({ord(c): None for c in "$%,"}).strip()
    found = _NUMBER.fullmatch(cleaned)
    if not found:
        return None
    unit = (found.group(2) or "").split()
    if len(unit) > _UNIT_WORDS_MAX:
        return None
    for word in unit:
        if not word.replace(".", "").isalpha() or word.lower().strip(".") in _HEDGE_WORDS:
            return None
    return Fraction(found.group(1))


def tolerance(gold: Fraction) -> Fraction:
    """The allowed absolute difference: 0 up to 50, 1% of the gold up to 10,000, 5% above."""
    size = abs(gold)
    if size <= 50:
        return Fraction(0)
    if size <= 10_000:
        return size / 100
    return size / 20


def _match_number(answer: str, gold: str, unordered: bool) -> bool:
    got, want = parse_number(answer), parse_number(gold)
    return got is not None and want is not None and abs(got - want) <= tolerance(want)


# -- year -----------------------------------------------------------------------------------------------------


def _match_year(answer: str, gold: str, unordered: bool) -> bool:
    return bool(_YEAR.fullmatch(answer) and _YEAR.fullmatch(gold)) and int(answer) == int(gold)


# -- date -----------------------------------------------------------------------------------------------------


def parse_date(text: str) -> dict[str, int] | None:
    """The ``year``, ``month`` and ``day`` found in ``text``, or None when it is not just a date.

    Accepted: ISO (``1962-03-03``, ``1962-03``), or month names, 4-digit years and 1-2 digit days (with an ordinal
    suffix) in any order, with commas and the words "the" and "of" ignored. Anything else, or a repeated
    component, makes it None.
    """
    lowered = text.strip().lower()
    iso = _ISO_DATE.fullmatch(lowered)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), iso.group(3)
        if not 1 <= month <= 12 or (day is not None and not 1 <= int(day) <= 31):
            return None
        parts = {"year": year, "month": month}
        if day is not None:
            parts["day"] = int(day)
        return parts
    parts = {}
    for token in lowered.replace(",", " ").split():
        token = token.strip(".")
        if token in ("", "the", "of"):
            continue
        day = _DAY.fullmatch(token)
        if token in _MONTHS:
            key, value = "month", _MONTHS[token]
        elif re.fullmatch(r"\d{4}", token):
            key, value = "year", int(token)
        elif day and 1 <= int(day.group(1)) <= 31:
            key, value = "day", int(day.group(1))
        else:
            return None
        if key in parts:
            return None
        parts[key] = value
    return parts or None


def _match_date(answer: str, gold: str, unordered: bool) -> bool:
    got, want = parse_date(answer), parse_date(gold)
    return got is not None and want is not None and all(got.get(k) == v for k, v in want.items())


# -- string ---------------------------------------------------------------------------------------------------


def _match_string(answer: str, gold: str, unordered: bool) -> bool:
    got = normalize_str(answer)
    return bool(got) and got == normalize_str(gold)


# -- list -----------------------------------------------------------------------------------------------------


def _list_key(element: str) -> tuple[str, Fraction | str]:
    """A number when the element is one ($, % and , dropped), else GAIA's string form with punctuation kept."""
    element = re.sub(r"^and\s+", "", element.strip(), flags=re.IGNORECASE)
    plain = element.translate({ord(c): None for c in "$%,"}).strip()
    if _PLAIN_NUMBER.fullmatch(plain):
        return ("number", Fraction(plain))
    return ("string", normalize_str(element, remove_punct=False))


def _match_list(answer: str, gold: str, unordered: bool) -> bool:
    got = [_list_key(e) for e in re.split(r"[,;]", answer)]
    want = [_list_key(e) for e in re.split(r"[,;]", gold)]
    if len(got) != len(want):
        return False
    return Counter(got) == Counter(want) if unordered else got == want


_COMPARE = {
    "number": _match_number,
    "year": _match_year,
    "date": _match_date,
    "string": _match_string,
    "list": _match_list,
}


# -- public ---------------------------------------------------------------------------------------------------


def matches(answer: str, target: str, kind: str, *, unordered: bool = False) -> bool:
    """Whether ``answer`` equals the single ``target`` (the gold or one alias) under the rules of ``kind``."""
    if kind not in _COMPARE:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    got = prenormalize(answer)
    return bool(got) and _COMPARE[kind](got, prenormalize(target), unordered)


def score(
    answer: str, *, gold: str, kind: str, aliases: Sequence[str] = (), unordered: bool = False
) -> str | None:
    """The gold or alias that ``answer`` matches, or None."""
    for target in (gold, *aliases):
        if matches(answer, target, kind, unordered=unordered):
            return target
    return None


@dataclass(frozen=True)
class Grade:
    """``verdict`` is pass, fail, abstain or no_answer; ``matched`` is the gold or alias that passed."""

    verdict: str
    answer: str
    matched: str | None = None


def grade(
    reply: str | None, *, gold: str, kind: str, aliases: Sequence[str] = (), unordered: bool = False
) -> Grade:
    """Grade an agent's whole reply: extract its ``FINAL ANSWER:`` line, then score it."""
    extracted = answer_mod.extract(reply)
    if extracted.status != answer_mod.ANSWER:
        return Grade(extracted.status, extracted.answer)
    matched = score(extracted.answer, gold=gold, kind=kind, aliases=aliases, unordered=unordered)
    return Grade(PASS if matched is not None else FAIL, extracted.answer, matched)
