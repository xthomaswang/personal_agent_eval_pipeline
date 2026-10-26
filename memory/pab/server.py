"""pab: the bench's own MCP server. For T4 it feeds one day of the timeline and holds the memory store.

    python -m memory.pab --run-dir <run> --session <1-10> [--fixtures memory/t4/fixtures/v1]
    python -m memory.pab --run-dir <run> --task submit          # only the submit tool

One process serves one ingestion session (one harness process, one day). The memory store persists
across sessions in <run>/t4/store.json. Every call appends one line to <run>/events/pab.jsonl with the
session and both clocks; the oracle reads the feed_day lines to check that every cited source_id had
been fed by the time it was cited. The timeline is synthetic, so the arguments are logged in full.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from memory.pab.mcp import Tool, ToolError, serve

FIXTURES = Path(__file__).resolve().parents[1] / "t4" / "fixtures" / "v1"
MAX_KEY, MAX_VALUE = 120, 2000
STATUSES = ("done", "failed", "abstain")


class Pab:
    def __init__(self, run_dir: Path, session: int, fixtures: Path) -> None:
        self.run_dir, self.session, self.fixtures = run_dir, session, fixtures
        self.store_path = run_dir / "t4" / "store.json"
        self.events_path = run_dir / "events" / "pab.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- records and store ----

    def day(self) -> tuple[str, list[dict[str, Any]]]:
        timeline = yaml.safe_load((self.fixtures / "facts.yaml").read_text())["timeline"]
        date = dt.date.fromisoformat(timeline["start"]) + dt.timedelta(days=self.session - 1)
        path = self.fixtures / "days" / f"day{self.session:02d}.jsonl"
        return date.isoformat(), [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def load(self) -> dict[str, Any]:
        if self.store_path.exists():
            return json.loads(self.store_path.read_text())
        return {"next_id": 1, "entries": []}

    def save(self, store: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1))
        os.replace(tmp, self.store_path)

    def log(self, tool: str, t0: float, w0: float, args: dict[str, Any], **extra: Any) -> None:
        line = {"session": self.session, "tool": tool, "t_start": t0, "t_end": time.monotonic(),
                "wall_start": w0, "args": args, **extra}
        with self.events_path.open("a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    @staticmethod
    def public(entry: dict[str, Any]) -> dict[str, Any]:
        return {k: entry[k] for k in ("id", "key", "value", "source_ids")}

    # ---- tools ----

    def wrap(self, name: str, fn):  # noqa: ANN001, ANN205 - a thin logging wrapper
        def call(args: dict[str, Any]) -> Any:
            t0, w0 = time.monotonic(), time.time()
            try:
                result, extra = fn(args)
            except ToolError as e:
                self.log(name, t0, w0, args, ok=False, error=str(e))
                raise
            self.log(name, t0, w0, args, ok=True, **extra)
            return result
        return call

    def feed_day(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        date, records = self.day()
        return {"date": date, "results": records}, {"fed_ids": [r["record_id"] for r in records]}

    def _fields(self, args: dict[str, Any], partial: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, limit in (("key", MAX_KEY), ("value", MAX_VALUE)):
            if name in args:
                v = args[name]
                if not isinstance(v, str) or not v.strip() or len(v) > limit:
                    raise ToolError(f"{name} must be a non-empty string of at most {limit} characters")
                out[name] = v.strip()
            elif not partial:
                raise ToolError(f"{name} is required")
        if "source_ids" in args:
            ids = args["source_ids"]
            if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i.strip() for i in ids):
                raise ToolError("source_ids must be a non-empty list of record_id strings")
            out["source_ids"] = list(dict.fromkeys(i.strip() for i in ids))
        elif not partial:
            raise ToolError("source_ids is required: list the record_ids this entry comes from")
        return out

    def _find(self, store: dict[str, Any], eid: Any) -> dict[str, Any]:
        for e in store["entries"]:
            if e["id"] == eid:
                return e
        raise ToolError(f"no entry with id {eid!r}")

    def memory_write(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        fields = self._fields(args, partial=False)
        store = self.load()
        eid = f"m{store['next_id']:03d}"
        store["next_id"] += 1
        store["entries"].append({"id": eid, **fields, "created_session": self.session, "updated_session": self.session})
        self.save(store)
        return {"id": eid}, {"id": eid}

    def memory_update(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        fields = self._fields(args, partial=True)
        if not fields:
            raise ToolError("give at least one of key, value, source_ids")
        store = self.load()
        entry = self._find(store, args.get("id"))
        entry.update(fields, updated_session=self.session)
        self.save(store)
        return {"id": entry["id"], "updated": True}, {"id": entry["id"]}

    def memory_delete(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        store = self.load()
        entry = self._find(store, args.get("id"))
        store["entries"].remove(entry)
        self.save(store)
        return {"deleted": entry["id"]}, {"id": entry["id"]}

    def memory_search(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        limit = args.get("limit", 10)
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ToolError("limit must be an integer from 1 to 50")
        words = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for e in self.load()["entries"]:
            hits = len(words & set(re.findall(r"\w+", f"{e['key']} {e['value']}".casefold())))
            if hits:
                scored.append((-hits, e["id"], e))
        results = [self.public(e) for _, _, e in sorted(scored)[:limit]]
        return {"results": results}, {"n": len(results)}

    def memory_list(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        results = [self.public(e) for e in self.load()["entries"]]
        return {"results": results}, {"n": len(results)}

    def submit(self, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        status = args.get("status")
        if status not in STATUSES:
            raise ToolError(f"status must be one of {', '.join(STATUSES)}")
        return {"ok": True}, {"status": status}

    def tools(self, task: str) -> list[Tool]:
        string = {"type": "string"}
        ids = {"type": "array", "items": string, "minItems": 1,
               "description": "the record_ids of the records this entry comes from"}
        submit = Tool("submit", "Finish this session. status: done, failed or abstain.",
                      {"type": "object", "properties": {"status": {"type": "string", "enum": list(STATUSES)},
                                                        "note": string}, "required": ["status"]},
                      self.wrap("submit", self.submit))
        if task == "submit":
            return [submit]
        return [
            Tool("feed_day", "Return today's records (messages, emails and calendar entries) as JSON. Every "
                 "record has a record_id. Only today's records are available.",
                 {"type": "object", "properties": {}}, self.wrap("feed_day", self.feed_day)),
            Tool("memory_write", "Add one entry to the user's memory store and return its id. key: a short topic "
                 "label; value: the facts, with absolute dates; source_ids: the record_ids it comes from.",
                 {"type": "object", "properties": {"key": string, "value": string, "source_ids": ids},
                  "required": ["key", "value", "source_ids"]}, self.wrap("memory_write", self.memory_write)),
            Tool("memory_update", "Change an existing entry. Fields you leave out stay as they are; source_ids "
                 "replaces the entry's list.",
                 {"type": "object", "properties": {"id": string, "key": string, "value": string, "source_ids": ids},
                  "required": ["id"]}, self.wrap("memory_update", self.memory_update)),
            Tool("memory_delete", "Delete an entry by id.",
                 {"type": "object", "properties": {"id": string}, "required": ["id"]},
                 self.wrap("memory_delete", self.memory_delete)),
            Tool("memory_search", "Return the entries whose key or value shares words with the query, best first.",
                 {"type": "object", "properties": {"query": string, "limit": {"type": "integer", "minimum": 1,
                                                                             "maximum": 50}},
                  "required": ["query"]}, self.wrap("memory_search", self.memory_search)),
            Tool("memory_list", "Return every entry in the memory store.", {"type": "object", "properties": {}},
                 self.wrap("memory_list", self.memory_list)),
            submit,
        ]


def main() -> None:
    ap = argparse.ArgumentParser(description="pab MCP server (stdio)")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--session", type=int, default=1, help="the ingestion session, which is also the day (1-10)")
    ap.add_argument("--task", choices=("t4", "submit"), default="t4")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = ap.parse_args()
    pab = Pab(args.run_dir, args.session, args.fixtures)
    serve(pab.tools(args.task), "pab", "1")
