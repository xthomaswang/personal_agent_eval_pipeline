"""T4 offline self-tests: fixtures, the pab server, the oracle's checks and the reader's parsing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from memory.pab.mcp import Tool, ToolError, handle
from memory.t4 import gen, reader
from memory.t4 import store as st
from memory.t4.oracle import ANSWERABLE, Fixtures, fabricated, score, store_checks

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fx() -> Fixtures:
    return Fixtures()


def test_committed_fixtures_are_what_seed_1_generates() -> None:
    assert subprocess.run([sys.executable, "-m", "memory.t4.gen", "--seed", "1", "--check"], cwd=ROOT).returncode == 0


def test_other_seeds_move_distractors_not_the_story(fx: Fixtures) -> None:
    a, b = gen.build(1), gen.build(2)
    assert a["probes"]["probes"][0]["question"] == b["probes"]["probes"][0]["question"]
    assert a["days"][1] != b["days"][1]


def test_probes(fx: Fixtures) -> None:
    cats = [p["category"] for p in fx.probes]
    assert {c: cats.count(c) for c in set(cats)} == {c: 12 for c in (*ANSWERABLE, "abstention")}
    for p in fx.probes:
        assert all(rid in fx.records for rid in p["support"])
        assert (p["answer"] is None) == (p["category"] == "abstention") == (not p["support"])


def test_record_ids_are_not_uuid_shaped(fx: Fixtures) -> None:
    uuid = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    text = "".join(p.read_text() for p in (fx.path / "days").glob("*.jsonl"))
    assert not __import__("re").search(uuid, text, __import__("re").I)


@pytest.mark.parametrize("day", range(1, 11))
def test_gold_store_is_precise_and_free_of_duplicates(fx: Fixtures, day: int) -> None:
    checks = store_checks(fx, st.gold(fx.facts, day))
    assert checks["source_precision"] == 1.0, checks["imprecise"]
    assert checks["n_duplicates"] == 0
    assert fabricated(fx, st.gold(fx.facts, day), None) == []


def test_gold_answers_pass_their_own_normalization(fx: Fixtures) -> None:
    names = {k: v["name"] for k, v in fx.facts["entities"].items()}
    for p in fx.probes:
        a: Any = p["answer"]
        if p["type"] == "entity" and a:
            a = names[a]
        elif p["type"] == "list":
            a = [names[x] for x in a]
        elif p["type"] == "money" and a is not None:
            a = a / 100
        elif p["type"] == "datetime" and a:
            a = a.replace("T", " ")
        assert fx.correct(p, a), p["id"]


@pytest.mark.parametrize(("probe_id", "answer", "ok"), [
    ("cs01", "2026-04-17 20:55", True), ("cs01", "2026-04-17 18:40", False), ("cs01", "April 17, 2026 8:55 PM", True),
    ("ku04", "Dr. Ken Ito", True), ("ku04", "Ken Ito (Riverside Dental)", True), ("ku04", "Ana Sousa", False),
    ("ku12", 2095, True), ("ku12", "$2,095.00", True), ("ku12", 2150, False), ("ku08", "9:00 PM", True),
    ("cs08", ["Dev Kapoor", "Priya"], True), ("cs08", ["Priya Raman"], False), ("ku02", "Harbor Room", False),
    ("ab03", None, True), ("ab03", "NOT_FOUND", True), ("ab03", "Imani Cole", False), ("ex01", None, False),
])
def test_answers(fx: Fixtures, probe_id: str, answer: Any, ok: bool) -> None:
    probe = next(p for p in fx.probes if p["id"] == probe_id)
    assert fx.correct(probe, answer) is ok


def test_abstention_probes_are_never_dropped(fx: Fixtures) -> None:
    fx.dropped = {"ab01", "ex01"}
    try:
        answers = [{"probe_id": p["id"], "rep": 0, "answer": None, "error": None} for p in fx.probes]
        s = score(fx, answers)
    finally:
        fx.dropped = set()
    assert s["per_category"]["abstention"] == {"correct": 12, "n": 12}
    assert s["per_category"]["extraction"]["n"] == 11
    assert s["answerable"] == {"correct": 0, "n": 47}


# ---- pab over stdio: the scripted agent and a fabricating one ----

class Client:
    def __init__(self, run_dir: Path, session: int) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", "memory.pab", "--run-dir", str(run_dir), "--session",
                                      str(session)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")

    def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        assert reply["id"] == self.n
        return reply

    def call(self, name: str, **args: Any) -> tuple[Any, bool]:
        res = self.rpc("tools/call", {"name": name, "arguments": args})["result"]
        text = res["content"][0]["text"]
        return (text if res["isError"] else json.loads(text)), res["isError"]

    def close(self) -> None:
        self.proc.stdin.close()
        assert self.proc.wait(timeout=10) == 0


def test_scripted_agent_builds_the_gold_store(fx: Fixtures, tmp_path: Path) -> None:
    ids: dict[str, str] = {}
    for day in range(1, 11):
        c = Client(tmp_path, day)
        fed, err = c.call("feed_day")
        assert not err and fed["date"] == f"2026-04-{5 + day:02d}"
        for item in fx.facts["items"]:
            for v in item["versions"]:
                if v["day"] != day:
                    continue
                key, value = item["key"], v["text"]
                if key in ids:
                    _, err = c.call("memory_update", id=ids[key], value=value, source_ids=v["source_ids"])
                else:
                    got, err = c.call("memory_write", key=key, value=value, source_ids=v["source_ids"])
                    ids[key] = got["id"]
                assert not err
        assert c.call("submit", status="done") == ({"ok": True}, False)
        c.close()
    entries = st.load(tmp_path / "t4" / "store.json")
    assert sorted(e["text"] for e in entries) == sorted(f"{g['id']}: {g['text']}" for g in st.gold(fx.facts))
    assert fabricated(fx, entries, tmp_path / "events" / "pab.jsonl") == []
    assert store_checks(fx, entries)["source_precision"] == 1.0


def test_citing_a_record_before_it_is_fed_is_a_violation(fx: Fixtures, tmp_path: Path) -> None:
    later = next(rid for rid, r in fx.facts["records"].items() if r["day"] == 2)
    c = Client(tmp_path, 1)
    c.call("feed_day")
    assert c.call("memory_write", key="k", value="v", source_ids=[later])[1] is False  # accepted, then flagged
    assert c.call("memory_write", key="k", value="v", source_ids=[])[1] is True  # refused
    assert c.call("memory_update", id="m999", value="v")[1] is True
    c.close()
    bad = fabricated(fx, st.load(tmp_path / "t4" / "store.json"), tmp_path / "events" / "pab.jsonl")
    assert bad == [{"session": 1, "tool": "memory_write", "id": "m001", "ids": [later]}]


def test_mcp_protocol_edges() -> None:
    def boom(args: dict[str, Any]) -> Any:
        raise ToolError("nope")

    tools = {"t": Tool("t", "d", {"type": "object"}, boom)}
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}},
                  tools, "pab", "1")
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, tools, "pab", "1") is None
    assert handle({"jsonrpc": "2.0", "id": 2, "method": "x"}, tools, "pab", "1")["error"]["code"] == -32601
    assert handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "u"}}, tools, "pab",
                  "1")["error"]["code"] == -32602
    res = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "t"}}, tools, "pab", "1")
    assert res["result"]["isError"] is True and res["result"]["content"][0]["text"] == "nope"


# ---- reader ----

@pytest.mark.parametrize(("text", "answer", "error"), [
    ('{"answer": "2026-04-17"}', "2026-04-17", None), ('Sure.\n{"answer": null}', None, None),
    ('```json\n{"answer": ["Priya", "Dev"]}\n```', ["Priya", "Dev"], None), ("I don't know", None, "no JSON object with an answer"),
])
def test_reader_parse(text: str, answer: Any, error: str | None) -> None:
    assert reader.parse(text) == (answer, error)


def test_reader_request(fx: Fixtures) -> None:
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "Harbor Hall"}'}}],
                                         "usage": {"prompt_tokens": 10}})

    probe = next(p for p in fx.probes if p["id"] == "ku02")
    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        row = reader.ask(client, "http://proxy", "m", "SYSTEM", probe)
    assert row["answer"] == "Harbor Hall" and row["error"] is None and fx.correct(probe, row["answer"])
    body = seen["body"]
    assert body["temperature"] == 0 and body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert "top_p" not in body and probe["question"] in body["messages"][1]["content"]
