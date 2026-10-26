"""Typed values for T4: normalizing the reader's answers, and extracting key values from free text.

Key values are what `source_precision` and `duplicate_ratio` compare: dates (ISO), clock times (HH:MM,
24 h), money (integer cents) and entities (ids from facts.yaml, through their aliases). The owner's own
aliases are never a key value.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from typing import Any

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
MONTHS |= {m[:3]: i for m, i in list(MONTHS.items())} | {"sept": 9}
NULLS = {"", "not_found", "not found", "null", "none"}

_ISO_DT = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?")
_ISO_D = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_MONTH_D = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
_D_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\b\.?(?:,?\s+(\d{{4}}))?", re.I)
_NUM_D = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{4}))?(?![\d/])")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?\b)?", re.I)
_TIME_H = re.compile(r"(?<![\d:.$])\b(\d{1,2})\s*([ap])\.?m\.?\b", re.I)
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")


class Entities:
    """Alias lookup over facts.yaml's entities."""

    def __init__(self, entities: dict[str, dict[str, Any]], owner: str) -> None:
        self.owner = owner
        self.types = {eid: e["type"] for eid, e in entities.items()}
        pairs = []
        for eid, e in entities.items():
            for alias in [e["name"], *e.get("aliases", [])]:
                pairs.append((alias, eid))
        self.exact = {_fold(a): eid for a, eid in pairs}
        # Longest alias first; short all-caps aliases (DEN, BOS) match case-sensitively
        self.patterns = []
        for alias, eid in sorted(pairs, key=lambda p: -len(p[0])):
            flags = 0 if (len(alias) <= 3 and alias.isupper()) else re.I
            self.patterns.append((re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags), eid))

    def find(self, text: str) -> list[str]:
        """Entity ids mentioned in text, in order of first mention, each once; overlapping mentions count once."""
        taken: list[tuple[int, int]] = []
        found: list[tuple[int, str]] = []
        for pattern, eid in self.patterns:
            for m in pattern.finditer(text):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((m.start(), eid))
        out: list[str] = []
        for _, eid in sorted(found):
            if eid not in out:
                out.append(eid)
        return out

    def resolve(self, answer: str, etype: str | None = None) -> str | None:
        """One entity id for a short answer, or None when it names none or several (the owner aside).
        With etype, only entities of that type count, so "Ken Ito (Riverside Dental)" is a person."""
        folded = _fold(re.sub(r"^\s*dr\.?\s+", "", answer, flags=re.I))
        if folded in self.exact:
            return self.exact[folded]
        if _fold(answer) in self.exact:
            return self.exact[_fold(answer)]
        hits = [e for e in self.find(answer) if e != self.owner and (etype is None or self.types[e] == etype)]
        return hits[0] if len(hits) == 1 else None


def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w&#]+", " ", text.casefold()).split())


def _date(y: int, m: int, d: int) -> str | None:
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def _clock(h: int, m: int, ampm: str | None) -> str | None:
    if ampm:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ampm.lower() == "p" else 0)
    return f"{h:02d}:{m:02d}" if 0 <= h <= 23 and 0 <= m <= 59 else None


def dates(text: str, year: int) -> list[str]:
    out = [f"{y}-{m}-{d}" for y, m, d in _ISO_D.findall(text) if _date(int(y), int(m), int(d))]
    for mon, d, y in _MONTH_D.findall(text):
        out.append(_date(int(y or year), MONTHS[mon.lower()], int(d)))
    for d, mon, y in _D_MONTH.findall(text):
        out.append(_date(int(y or year), MONTHS[mon.lower()], int(d)))
    for m, d, y in _NUM_D.findall(text):
        out.append(_date(int(y or year), int(m), int(d)))
    return [v for v in out if v]


def times(text: str) -> list[str]:
    text = _ISO_DT.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)
    out = [_clock(int(h), int(m), ap or None) for h, m, ap in _TIME.findall(text)]
    out += [_clock(int(h), 0, ap) for h, ap in _TIME_H.findall(text)]
    return [v for v in out if v]


def money(text: str) -> list[int]:
    return [int(whole.replace(",", "")) * 100 + int((cents or "0").ljust(2, "0")) for whole, cents in _MONEY.findall(text)]


def key_values(text: str, ents: Entities, year: int) -> set[str]:
    """Every key value in text, as typed strings: date:…, time:…, money:…, entity:…"""
    vals = {f"date:{d}" for d in dates(text, year)}
    vals |= {f"time:{t}" for t in times(text)}
    vals |= {f"money:{c}" for c in money(text)}
    vals |= {f"entity:{e}" for e in ents.find(text) if e != ents.owner}
    return vals


def canonical_text(text: str, year: int) -> str:
    """Text with dates, times and amounts in one form, lower case, punctuation removed: equal for duplicates."""
    text = _ISO_DT.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)
    for pattern, conv in ((_MONTH_D, lambda m: _date(int(m.group(3) or year), MONTHS[m.group(1).lower()], int(m.group(2)))),
                          (_D_MONTH, lambda m: _date(int(m.group(3) or year), MONTHS[m.group(2).lower()], int(m.group(1)))),
                          (_TIME, lambda m: _clock(int(m.group(1)), int(m.group(2)), m.group(3))),
                          (_TIME_H, lambda m: _clock(int(m.group(1)), 0, m.group(2))),
                          (_MONEY, lambda m: str(money(m.group(0))[0]))):
        text = pattern.sub(lambda m, c=conv: c(m) or m.group(0), text)
    return _fold(text)


# ---- the reader's answers ------------------------------------------------------------------------------

def is_null(answer: Any) -> bool:
    return answer is None or (isinstance(answer, str) and answer.strip().casefold() in NULLS)


def _one(values: list[str]) -> str | None:
    distinct = list(dict.fromkeys(values))
    return distinct[0] if len(distinct) == 1 else None


def normalize(answer: Any, kind: str, ents: Entities, year: int, tz: str, etype: str | None = None) -> Any:
    """The answer in its type's canonical form, or None when it is null or cannot be read as that type.
    etype narrows entity and list answers to one entity type (the gold answer's)."""
    if is_null(answer):
        return None
    if kind == "money":
        if isinstance(answer, (int, float)) and not isinstance(answer, bool):
            return round(answer * 100)
        text = str(answer)
        found = money(text) or [round(float(x) * 100) for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?", text.replace(",", ""))]
        return found[0] if len(set(found)) == 1 else None
    text = str(answer)
    if kind == "date":
        return _one(dates(text, year))
    if kind == "time":
        return _one(times(text))
    if kind == "datetime":
        m = _ISO_DT.search(text)
        day, clock = (m.group(1), m.group(2)) if m else (_one(dates(text, year)), _one(times(text)))
        if not day or not clock or len(set(dates(text, year))) > 1:
            return None
        offset = re.search(r"([+-]\d{2}):?(\d{2})$|Z$", text.strip())
        local = dt.datetime.fromisoformat(f"{day}T{clock}")
        zone = tz if not offset else ("+00:00" if offset.group(0) == "Z" else f"{offset.group(1)}:{offset.group(2)}")
        aware = local.replace(tzinfo=dt.datetime.fromisoformat(f"2000-01-01T00:00{zone}").tzinfo)
        return aware.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%MZ")
    if kind == "entity":
        return ents.resolve(text, etype)
    if kind == "list":
        items: Iterable[Any] = answer if isinstance(answer, list) else re.split(r",|;|\band\b|&", text)
        resolved = [ents.resolve(str(i), etype) for i in items if str(i).strip()]
        return frozenset(resolved) if resolved and all(resolved) else None
    raise ValueError(f"unknown answer type {kind}")


def expected(answer: Any, kind: str, tz: str) -> Any:
    """A probe's gold answer in the same canonical form as normalize()."""
    if answer is None:
        return None
    if kind == "datetime":
        local = dt.datetime.fromisoformat(f"{answer}:00{tz}")
        return local.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%MZ")
    if kind == "list":
        return frozenset(answer)
    return answer
