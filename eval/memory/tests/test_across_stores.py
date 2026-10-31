"""The in-chat checks on a cell's SQLite files (stores.py) and the scoring of in-chat answers (score.py): eviction
validity from a windows table, saved notes, miss classes, answer sources and the recall block."""
import json
import sqlite3
from pathlib import Path

import pytest
from memory.across import score, stores

FACT = "Today's call is in room 4B."


def make_sessions_db(home: Path, chats: dict[str, dict]) -> None:
    """A gateway store as the real one lays it out for the checks: one ``windows`` row per chat and every turn's events.

    ``chats[sid]``: ``counts`` (messages each turn added), ``fact_turn``, ``window`` (the stored messages), ``window_turn``
    (default: the last turn; None: no row) and ``cap`` (what run.finished reports; None: nothing)."""
    path = home / stores.SESSIONS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE windows (session_id TEXT PRIMARY KEY, turn INTEGER NOT NULL, version INTEGER NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL, thinking_since REAL)")
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, turn INTEGER, type TEXT NOT NULL, event TEXT NOT NULL)")

    def event(sid, turn, kind, data):
        conn.execute("INSERT INTO events (session_id, turn, type, event) VALUES (?, ?, ?, ?)", (sid, turn, kind, json.dumps({"type": kind, "data": data})))

    for sid, chat in chats.items():
        counts = chat["counts"]
        for turn, n in enumerate(counts, 1):
            event(sid, turn, "turn.started", {"turn": turn, "user_text": FACT if turn == chat["fact_turn"] else f"message {turn}"})
            for _ in range(n - 1):  # the user message is the first; the rest are assistant messages and tool results
                event(sid, turn, "assistant.message", {"step": 1})
            if chat.get("cap") is not None and turn == len(counts):
                event(sid, turn, "run.finished", {"result": {"config": {"window_messages": chat["cap"]}}})
            event(sid, turn, "turn.finished", {"turn": turn})
        window_turn = chat.get("window_turn", len(counts))
        if window_turn is not None:
            conn.execute("INSERT INTO windows VALUES (?, ?, 1, ?, 0, NULL)", (sid, window_turn, json.dumps({"v": 1, "messages": chat["window"]})))
    conn.commit()
    conn.close()


def messages(*texts):
    return [{"role": "user" if n % 2 == 0 else "assistant", "content": t} for n, t in enumerate(texts)]


def long_chat(**kw):
    """30 filler turns of 4 messages after an opener and a fact turn of 2, then the question turn of 2: 116 before it."""
    chat = {"counts": [2, 2, *[4] * 28, 2], "fact_turn": 2, "cap": 64, "window": messages("message 30", "ok", "message 31", "ok")}
    chat.update(kw)
    return chat


def judged(tmp_path, rule, **kw):
    make_sessions_db(tmp_path, {"s1": long_chat(**kw)})
    facts = stores.window_facts(tmp_path, "s1", FACT)
    return facts, stores.judge(facts, rule)


def test_the_facts_come_from_the_window_row_and_the_events(tmp_path):
    facts, verdict = judged(tmp_path, "out")
    assert (facts.last_turn, facts.window_turn, facts.window_len, facts.fact_in_window) == (31, 31, 4, False)
    assert (facts.fact_turn, facts.fact_index, facts.messages_before, facts.after_fact, facts.cap) == (2, 2, 116, 112, 64)
    assert verdict == (True, "")


def test_out_arms_need_the_fact_out_of_the_window_after_a_long_enough_chat(tmp_path):
    kept = messages("message 1", FACT, "ok")  # the fact is still in the stored window
    kept[1]["role"] = "user"
    assert judged(tmp_path / "a", "out", window=kept)[1] == (False, "the fact turn is still in the window")
    short = long_chat(counts=[2, 2, *[4] * 8, 2])  # 32 messages between the end of the fact turn and the question
    ok, why = judged(tmp_path / "b", "out", **short)[1]
    assert not ok and "only 32 messages between the end of the fact turn and the question (cap 64)" in why
    assert judged(tmp_path / "d", "out", cap=None)[1] == (False, "the events do not give the window cap or the fact's place")


def test_it_does_not_matter_where_in_the_chat_the_fact_turn_starts(tmp_path):
    """A model that calls a tool in the opener pushes the fact turn past message 9; 112 messages followed it all the same."""
    late = long_chat(counts=[12, 2, *[4] * 28, 2])
    facts, verdict = judged(tmp_path, "out", **late)
    assert facts.fact_index == 12 and facts.after_fact == 112 and verdict == (True, "")


def test_the_fact_is_out_when_the_messages_after_it_reach_the_cap_and_not_before(tmp_path):
    exactly = long_chat(counts=[2, 2, *[4] * 16, 2])  # 64 messages after the fact turn: the window held the newest 64 or fewer
    facts, verdict = judged(tmp_path / "a", "out", **exactly)
    assert facts.after_fact == 64 and verdict == (True, "")
    one_less = long_chat(counts=[2, 2, *[4] * 15, 3, 2])
    facts, (ok, why) = judged(tmp_path / "b", "out", **one_less)
    assert facts.after_fact == 63 and not ok and "only 63 messages between the end of the fact turn and the question (cap 64)" in why
    assert judged(tmp_path / "c", "out", **long_chat(counts=[2, 2, *[4] * 31, 2], cap=128))[1][0] is False  # 124 after the fact, cap 128


def test_the_in_arm_needs_the_fact_in_the_stored_window(tmp_path):
    kept = [{"role": "user", "content": FACT}, {"role": "assistant", "content": "ok"}]
    assert judged(tmp_path / "a", "in", window=kept, cap=256)[1] == (True, "")
    ok, why = judged(tmp_path / "b", "in", cap=256)[1]
    assert not ok and "left the window" in why


def test_a_window_row_from_another_turn_or_no_row_is_not_the_questions_window(tmp_path):
    ok, why = judged(tmp_path / "a", "out", window_turn=30)[1]
    assert not ok and "from turn 30, not the question's turn 31" in why
    ok, why = judged(tmp_path / "b", "in", window_turn=None)[1]
    assert not ok and why == "no window row for the chat"
    assert stores.judge(stores.window_facts(tmp_path / "a", "s1", FACT), "") == (True, "")  # no rule: always valid


def test_message_counts_per_turn_follow_the_events():
    events = [(1, "turn.started"), (1, "assistant.message"), (2, "turn.started"), (2, "assistant.message"), (2, "tool.finished"), (2, "assistant.message")]
    assert stores.turn_message_counts(events) == {1: 2, 2: 4}


def test_stores_are_opened_read_only_and_immutable(tmp_path):
    make_sessions_db(tmp_path, {"s1": long_chat()})
    conn = stores.open_ro(tmp_path / stores.SESSIONS_DB)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM windows")
    conn.close()
    with pytest.raises(stores.StoreError, match="does not exist"):
        stores.open_ro(tmp_path / "nowhere.db")
    path = tmp_path / stores.SESSIONS_DB
    path.with_name(path.name + "-wal").write_bytes(b"unmerged")
    with pytest.raises(stores.StoreError, match="unmerged write-ahead log"):
        stores.window_facts(tmp_path, "s1", FACT)


def test_saved_notes_by_the_chat_that_made_them(tmp_path):
    path = tmp_path / stores.RECORDS_DB
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE long_term (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'profile', status TEXT NOT NULL DEFAULT 'active', session_id TEXT NOT NULL DEFAULT '')")
    conn.executemany("INSERT INTO long_term (text, status, session_id) VALUES (?, ?, ?)", [("Your sister is Priya.", "active", "a"), ("You live in Lisbon.", "superseded", "a"), ("Room 4B.", "active", "b")])
    conn.commit()
    conn.close()
    notes = stores.notes_by_chat(tmp_path)
    assert [n["text"] for n in notes["a"]] == ["Your sister is Priya.", "You live in Lisbon."] and notes["a"][1]["status"] == "superseded" and set(notes) == {"a", "b"}
    with pytest.raises(stores.StoreError):
        stores.notes_by_chat(tmp_path / "elsewhere")


def test_miss_classes():
    kw = {"failed": False, "in_window": None}
    assert score.miss_class(memory_on=True, saved=False, recalled=False, **kw) == score.MISS_NOT_SAVED
    assert score.miss_class(memory_on=True, saved=True, recalled=False, **kw) == score.MISS_NOT_RECALLED
    assert score.miss_class(memory_on=True, saved=True, recalled=True, **kw) == score.MISS_NOT_USED
    assert score.miss_class(memory_on=True, saved=None, recalled=False, **kw) == score.MISS_UNCHECKED
    assert score.miss_class(memory_on=False, saved=None, recalled=False, failed=False, in_window=False) == score.MISS_OUT_OF_WINDOW
    assert score.miss_class(memory_on=False, saved=None, recalled=False, failed=False, in_window=True) == score.MISS_IN_WINDOW
    assert score.miss_class(memory_on=False, saved=None, recalled=False, failed=False, in_window=None) == score.MISS_MEMORY_OFF
    assert score.miss_class(memory_on=True, saved=False, recalled=False, failed=True, in_window=None) == score.MISS_FAILED
    assert set(score.MISS_CLASSES) >= {score.MISS_NOT_SAVED, score.MISS_NOT_RECALLED, score.MISS_NOT_USED, score.MISS_OUT_OF_WINDOW, score.MISS_IN_WINDOW}


def test_a_recalled_fact_is_recalled_but_not_used_even_without_a_note():
    """A build that recalls a chat's own evicted turns: the fact is in `used`, no note was made, the answer is wrong."""
    kw = {"memory_on": True, "failed": False, "in_window": None}
    assert score.miss_class(saved=False, recalled=True, **kw) == score.MISS_NOT_USED
    assert score.miss_class(saved=None, recalled=True, **kw) == score.MISS_NOT_USED  # the stores need not be read to know it
    assert score.miss_class(saved=False, recalled=False, **kw) == score.MISS_NOT_SAVED and score.miss_class(saved=True, recalled=False, **kw) == score.MISS_NOT_RECALLED
    assert score.miss_class(saved=False, recalled=True, memory_on=True, failed=True, in_window=None) == score.MISS_FAILED  # a failed turn first
    assert score.miss_class(saved=False, recalled=True, memory_on=False, failed=False, in_window=False) == score.MISS_OUT_OF_WINDOW  # memory off: the window classes


def used(kind, text, session="s1"):
    return {"used": [{"kind": kind, "text": text, "source": "you said it", "session_id": session}]}


def test_where_a_right_answer_came_from():
    gold = ["4B"]
    assert score.answer_source(used("record", FACT), gold, in_window=False, step_memory=True) == score.SRC_NOTE
    assert score.answer_source(used("chat", FACT), gold, in_window=False, step_memory=False) == score.SRC_CHAT
    assert score.answer_source(None, gold, in_window=True, step_memory=False) == score.SRC_WINDOW
    assert score.answer_source(used("record", "Something else."), gold, in_window=False, step_memory=False) == score.SRC_UNKNOWN
    assert score.answer_source(used("record", "Something else."), gold, in_window=False, step_memory=True) == score.SRC_STEP  # M2: step summaries send no event
    assert score.answer_source(used("record", FACT), gold, in_window=True, step_memory=True) == score.SRC_NOTE  # a used note that holds the gold wins
    own = {"used": [{"kind": "chat", "text": f"User: {FACT}\nmelo: Noted.", "source": "this chat, Oct 31", "session_id": "s1"}]}
    assert score.answer_source(own, gold, in_window=False, step_memory=True) == score.SRC_WINDOW  # a turn of the chat itself that left the window
    other = {"used": [{**own["used"][0], "source": "an earlier chat, Oct 30"}]}
    assert score.answer_source(other, gold, in_window=False, step_memory=True) == score.SRC_CHAT
    both = {"used": [*own["used"], *used("record", FACT)["used"]]}
    assert score.answer_source(both, gold, in_window=False, step_memory=False) == score.SRC_NOTE


def test_the_recall_block_is_real_when_reported_and_an_estimate_otherwise():
    memory = {"used": [{"text": "abcd", "source": "xy"}, {"text": "ef", "source": "g"}]}
    assert score.block_chars(memory, {"block_chars": 321}) == (321, False)
    assert score.block_chars(memory, {"embed_ms": 3}) == (9, True) and score.block_chars(memory, None) == (9, True)
    assert score.block_chars({"used": []}, None) == (0, True)
    assert score.block_chars(None, None) == (None, False)  # memory off: no recall, no block
    assert score.block_chars(memory, {"block_chars": True}) == (9, True)  # a bool is not a count
