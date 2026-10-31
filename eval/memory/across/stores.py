"""Read the gateway's and the memory's SQLite files of a finished cell, read-only (stdlib only).

A cell's HOME holds the files the in-chat checks need (paths below are under ``<home>/.cache/localmelo``):

- ``gateway/sessions.db``: ``windows`` has ONE row per chat (``INSERT OR REPLACE`` on the chat's id), the working
  window as it stood when the chat's newest finished turn ended: ``turn`` is that turn, ``data`` is JSON
  ``{"v", "messages": [{"role", "content", ...}]}``. There is no row for "the turn before the question"; after
  the question has run, the row is the window the question's turn ended with. ``events`` holds every turn's events,
  ``turn.started`` (with ``user_text``), ``assistant.message``, ``tool.finished`` and ``run.finished`` (whose
  ``result.config.window_messages`` is the cap the chat ran with);
- ``memory/records.db``: ``long_term`` has one row per saved note, with the ``session_id`` of the chat it was
  made in, its ``text`` and its ``status`` (``active`` or ``superseded``).

The files are opened with ``file:<path>?immutable=1``: SQLite then ignores locks and the write-ahead log, which is
only right for a gateway that has stopped, so call these after the cell ends. A file with an unmerged ``-wal`` is
refused (it would be read stale).

Why the window of the question's turn tells where the fact was at the question. The window drops its oldest messages
when it grows past its cap, and only ever from the front. A message that is in the window the turn ended with was in
it for every model call of that turn. A message that is absent from it was out of the window when the question
arrived if at least ``cap`` messages came after the fact turn and before the question turn: the window held at most
``cap`` messages then, the newest ones, and the fact is older than all of those. ``judge`` applies both; the counts come
from the stored events (they can fall short of the real ones after an empty-reply nudge, never exceed them).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SESSIONS_DB = Path(".cache") / "localmelo" / "gateway" / "sessions.db"
RECORDS_DB = Path(".cache") / "localmelo" / "memory" / "records.db"


class StoreError(RuntimeError):
    """A store cannot be read for the check (missing, or not settled on disk)."""


def open_ro(path: Path) -> sqlite3.Connection:
    """The file at ``path`` opened read-only and immutable; StoreError when it is missing or has an unmerged WAL."""
    path = Path(path).resolve()
    if not path.is_file():
        raise StoreError(f"{path.name} does not exist")
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise StoreError(f"{path.name} has an unmerged write-ahead log (the gateway did not close it)")
    try:
        return sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    except sqlite3.Error as exc:
        raise StoreError(f"{path.name} cannot be opened: {exc}") from exc


def _query(db: Path, sql: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    conn = open_ro(db)
    try:
        return conn.execute(sql, args).fetchall()
    except sqlite3.Error as exc:
        raise StoreError(f"{Path(db).name}: {exc}") from exc
    finally:
        conn.close()


# --- the working window ---


@dataclass(frozen=True)
class WindowFacts:
    """What the store says about one chat's window at its newest turn (the question's)."""

    session_id: str
    last_turn: int  # the chat's newest turn
    window_turn: int | None  # the turn the stored window belongs to; None: no row
    window_len: int  # messages in the stored window
    fact_in_window: bool  # a user message with the fact's text is in the stored window
    fact_turn: int | None  # the turn that said the fact (from turn.started); None: not found
    fact_index: int | None  # messages in the chat before the fact turn
    messages_before: int  # messages added by the turns before the last one
    after_fact: int | None  # messages added by the turns between the fact turn and the last one; None: no fact turn
    cap: int | None  # window_messages the last turn ran with (run.finished); None: not reported


def turn_message_counts(events: list[tuple[int, str]]) -> dict[int, int]:
    """Messages each turn added to the window: its user message, and one per ``assistant.message`` and per
    ``tool.finished``. This matches the stored window for all but the turns that hit an empty-reply nudge (the
    label and the nudge are two more messages with no event), so it can undercount and never overcount."""
    counts: dict[int, int] = {}
    for turn, kind in events:
        counts.setdefault(turn, 1)
        if kind in ("assistant.message", "tool.finished"):
            counts[turn] += 1
    return counts


def window_facts(home: Path, session_id: str, fact_text: str) -> WindowFacts:
    """The window row of chat ``session_id`` and the message arithmetic around it, for a fact said as ``fact_text``."""
    db = Path(home) / SESSIONS_DB
    rows = _query(db, "SELECT turn, data FROM windows WHERE session_id = ?", (session_id,))
    window_turn, messages = None, []
    if rows:
        window_turn = int(rows[0][0])
        doc = json.loads(rows[0][1])
        messages = [m for m in doc.get("messages", []) if isinstance(m, Mapping)]
    wanted = fact_text.strip()
    in_window = any(m.get("role") == "user" and str(m.get("content") or "").strip() == wanted for m in messages)
    events = _query(
        db, "SELECT turn, type FROM events WHERE session_id = ? AND turn IS NOT NULL AND type IN "
        "('turn.started', 'assistant.message', 'tool.finished') ORDER BY id", (session_id,))
    counts = turn_message_counts([(int(t), str(k)) for t, k in events])
    last = max(counts, default=0)
    fact_turn = None
    for turn, text in _query(db, "SELECT turn, event FROM events WHERE session_id = ? AND type = 'turn.started' ORDER BY id", (session_id,)):
        started = json.loads(text)
        if str(((started.get("data") or {}).get("user_text")) or "").strip() == wanted:
            fact_turn = int(turn)
            break
    cap = None
    for (text,) in _query(db, "SELECT event FROM events WHERE session_id = ? AND turn = ? AND type = 'run.finished'", (session_id, last)):
        value = ((((json.loads(text).get("data") or {}).get("result") or {}).get("config") or {}).get("window_messages"))
        if isinstance(value, int) and not isinstance(value, bool):
            cap = value
    return WindowFacts(
        session_id=session_id, last_turn=last, window_turn=window_turn, window_len=len(messages), fact_in_window=in_window,
        fact_turn=fact_turn, fact_index=None if fact_turn is None else sum(n for t, n in counts.items() if t < fact_turn),
        messages_before=sum(n for t, n in counts.items() if t < last),
        after_fact=None if fact_turn is None else sum(n for t, n in counts.items() if fact_turn < t < last), cap=cap,
    )


def judge(facts: WindowFacts, rule: str) -> tuple[bool, str]:
    """(valid, reason) of one chat for an arm whose rule is ``"out"`` (the fact must have left the window by the
    question), ``"in"`` (it must still be in it), or ``""`` (no condition)."""
    if not rule:
        return True, ""
    if facts.window_turn is None:
        return False, "no window row for the chat"
    if facts.window_turn != facts.last_turn:
        return False, f"the stored window is from turn {facts.window_turn}, not the question's turn {facts.last_turn}"
    if rule == "in":
        return (True, "") if facts.fact_in_window else (False, "the fact turn left the window (it should still be in it)")
    if facts.fact_in_window:
        return False, "the fact turn is still in the window"
    if facts.cap is None or facts.after_fact is None:
        return False, "the events do not give the window cap or the fact's place"
    if facts.after_fact < facts.cap:
        return False, (f"only {facts.after_fact} messages between the end of the fact turn and the question (cap {facts.cap}): "
                       "the fact may still have been in the window")
    return True, ""


# --- saved notes ---


def notes_by_chat(home: Path) -> dict[str, list[dict[str, Any]]]:
    """Every saved note (``long_term`` row) by the id of the chat that made it."""
    rows = _query(Path(home) / RECORDS_DB, "SELECT session_id, text, status, kind FROM long_term ORDER BY id")
    out: dict[str, list[dict[str, Any]]] = {}
    for session_id, text, status, kind in rows:
        out.setdefault(str(session_id), []).append({"text": str(text), "status": str(status), "kind": str(kind)})
    return out
