"""The memory store as the reader sees it, whichever agent wrote it.

One entry shape for every source: {"id", "text", "source_ids"}.
- pab's store (<run>/t4/store.json): text is "key: value".
- An exported store (JSON list or JSONL, e.g. localmelo's `melo-retrieve --export-store`): entries with
  "text", or with "key" and "value"; source_ids from "source_ids" or "provenance.source_ids".
- The gold store: every item's last version up to a day (facts.yaml).

render() is the exact text the reader gets and memory_tokens counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entry(raw: dict[str, Any], n: int) -> dict[str, Any]:
    if "text" in raw:
        text = str(raw["text"])
    elif "key" in raw and "value" in raw:
        text = f"{raw['key']}: {raw['value']}"
    else:
        raise ValueError(f"entry {n} has neither text nor key and value")
    ids = raw.get("source_ids") or (raw.get("provenance") or {}).get("source_ids") or []
    return {"id": str(raw.get("id", n + 1)), "text": text, "source_ids": [str(i) for i in ids]}


def load(path: Path) -> list[dict[str, Any]]:
    """Entries from pab's store.json or an exported store; a missing pab store is an empty store."""
    if not path.exists() and path.name == "store.json":
        return []
    text = path.read_text()
    if path.suffix == ".jsonl":
        raws = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        raws = data["entries"] if isinstance(data, dict) else data
    return [_entry(r, n) for n, r in enumerate(raws)]


def gold(facts: dict[str, Any], upto_day: int | None = None) -> list[dict[str, Any]]:
    """The ideal store after upto_day (default: the whole timeline)."""
    out = []
    for item in facts["items"]:
        versions = [v for v in item["versions"] if upto_day is None or v["day"] <= upto_day]
        if versions:
            v = versions[-1]
            out.append({"id": item["key"], "text": v["text"], "source_ids": v["source_ids"]})
    return out


def render(entries: list[dict[str, Any]]) -> str:
    """The store as the reader gets it: a JSON array of {id, text}, one entry per line."""
    if not entries:
        return "[]"
    return "[\n" + ",\n".join(json.dumps({"id": e["id"], "text": e["text"]}, ensure_ascii=False) for e in entries) + "\n]"
