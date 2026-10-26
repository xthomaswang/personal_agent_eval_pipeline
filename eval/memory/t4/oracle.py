"""T4's oracle: probe accuracy from the reader's answers, and the store checks.

    uv run python -m memory.t4.oracle --run-dir <run> [--store <export>] [--tokenizer <tokenizer.json>]

Reads <run>/t4/store.json (or --store), <run>/events/pab.jsonl when it exists, and
<run>/reader/answers.jsonl; writes <run>/t4/oracle.json.

- Probe accuracy: the reader's answer is normalized by type and compared exactly. Probes on the frozen
  drop list (drop_list.yaml next to the fixtures) are left out. Abstention probes are never on it: the
  empty store answers them right by construction, so they are scored apart, as the share of probes the
  reader still answers null from the full store. The headline is the accuracy over the four answerable
  categories.
- fabricated_source (hard violation): a memory_write or memory_update citing a record_id that feed_day
  had not returned yet. Without pab events (an exported store), an id that is not a record of the
  timeline.
- source_precision: among entries with at least one key value (date, time, amount, entity), the share
  whose key values all appear in the records they cite (resolved relative dates included).
- duplicate_ratio: the share of entries whose text equals an earlier entry's after normalization.
- memory_tokens: tokens of the store as the reader sees it, with the model's tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from memory.t4 import store as st
from memory.t4.values import Entities, canonical_text, expected, key_values, normalize

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "v1"
ANSWERABLE = ("extraction", "cross_session", "temporal", "knowledge_update")
TEXT_FIELDS = ("sender", "to", "subject", "body", "chat", "sender", "text", "title", "location", "notes", "start", "end", "date")


class Fixtures:
    def __init__(self, path: Path = FIXTURES) -> None:
        self.path = path
        self.facts = yaml.safe_load((path / "facts.yaml").read_text())
        self.probes = yaml.safe_load((path / "probes.yaml").read_text())["probes"]
        tl = self.facts["timeline"]
        self.tz, self.year, self.owner = tl["tz"], int(tl["start"][:4]), tl["owner"]
        self.ents = Entities(self.facts["entities"], self.owner)
        self.records: dict[str, dict[str, Any]] = {}
        for day in range(1, tl["days"] + 1):
            for line in (path / "days" / f"day{day:02d}.jsonl").read_text().splitlines():
                rec = json.loads(line)
                self.records[rec["record_id"]] = rec
        drop = path / "drop_list.yaml"
        self.drop_info = yaml.safe_load(drop.read_text()) if drop.exists() else None
        self.dropped = set(self.drop_info["dropped"]) if self.drop_info else set()
        self._values: dict[str, set[str]] = {}

    def record_values(self, rid: str) -> set[str]:
        if rid not in self._values:
            rec = self.records[rid]
            text = "\n".join(str(rec[f]) for f in TEXT_FIELDS if rec.get(f))
            vals = key_values(text, self.ents, self.year) | set(self.facts["records"][rid]["resolved"])
            self._values[rid] = vals
        return self._values[rid]

    def entity_type(self, probe: dict[str, Any]) -> str | None:
        gold = probe["answer"]
        if probe["type"] == "entity" and gold:
            return self.facts["entities"][gold]["type"]
        if probe["type"] == "list" and gold:
            return self.facts["entities"][gold[0]]["type"]
        return None

    def correct(self, probe: dict[str, Any], answer: Any) -> bool:
        got = normalize(answer, probe["type"], self.ents, self.year, self.tz, self.entity_type(probe))
        if probe["answer"] is None:
            return got is None and (answer is None or isinstance(answer, str))
        return got is not None and got == expected(probe["answer"], probe["type"], self.tz)


def fabricated(fx: Fixtures, entries: list[dict[str, Any]], events: Path | None, exported: bool = False) -> list[dict[str, Any]]:
    """Writes through pab are checked call by call against what feed_day had returned by then. An exported
    store (no memory calls to check) is checked against every id fed in the run, or, without pab's
    events, against the timeline's records."""
    out = []
    fed: set[str] = set()
    lines = events.read_text().splitlines() if events is not None and events.exists() else []
    for line in lines:
        ev = json.loads(line)
        if ev["tool"] == "feed_day" and ev.get("ok"):
            fed |= set(ev["fed_ids"])
        if not exported and ev["tool"] in ("memory_write", "memory_update") and ev.get("ok"):
            bad = [i for i in ev["args"].get("source_ids", []) if i not in fed]
            if bad:
                out.append({"session": ev["session"], "tool": ev["tool"], "id": ev.get("id"), "ids": bad})
    if lines and not exported:
        return out
    known = fed if lines else set(fx.records)
    for e in entries:
        bad = [i for i in e["source_ids"] if i not in known]
        if bad:
            out.append({"entry": e["id"], "ids": bad})
    return out


def store_checks(fx: Fixtures, entries: list[dict[str, Any]]) -> dict[str, Any]:
    checked = precise = dups = 0
    seen: set[str] = set()
    imprecise = []
    for e in entries:
        vals = key_values(e["text"], fx.ents, fx.year)
        if vals:
            checked += 1
            cited: set[str] = set()
            for rid in e["source_ids"]:
                if rid in fx.records:
                    cited |= fx.record_values(rid)
            if vals <= cited:
                precise += 1
            else:
                imprecise.append({"entry": e["id"], "not_in_sources": sorted(vals - cited)})
        canon = canonical_text(e["text"], fx.year)
        dups += canon in seen
        seen.add(canon)
    n = len(entries)
    return {"n_entries": n, "source_precision": precise / checked if checked else None, "n_checked": checked,
            "imprecise": imprecise, "duplicate_ratio": dups / n if n else 0.0, "n_duplicates": dups}


def memory_tokens(entries: list[dict[str, Any]], tokenizer: Path | None) -> int | None:
    if tokenizer is None or not tokenizer.exists():
        return None
    from tokenizers import Tokenizer  # the model's tokenizer.json; imported only when counting

    return len(Tokenizer.from_file(str(tokenizer)).encode(st.render(entries)).ids)


def score(fx: Fixtures, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy per category over the retained probes (rep 0 of each probe)."""
    by_probe = {a["probe_id"]: a for a in answers if a.get("rep", 0) == 0}
    per: dict[str, list[bool]] = {}
    rows = []
    for p in fx.probes:
        if p["id"] in fx.dropped and p["category"] != "abstention":
            continue
        a = by_probe.get(p["id"])
        ok = a is not None and a.get("error") is None and fx.correct(p, a.get("answer"))
        per.setdefault(p["category"], []).append(ok)
        rows.append({"probe_id": p["id"], "category": p["category"], "correct": ok})
    acc = {c: {"correct": sum(v), "n": len(v)} for c, v in per.items()}
    ans = [ok for c in ANSWERABLE for ok in per.get(c, [])]
    return {"answerable": {"correct": sum(ans), "n": len(ans)}, "per_category": acc, "probes": rows,
            "missing": sorted(p["id"] for p in fx.probes if p["id"] not in by_probe)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--store", type=Path, help="an exported store instead of <run>/t4/store.json")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--tokenizer", type=Path, default=Path(os.environ["BENCH_MODEL_PATH"]) / "tokenizer.json"
                    if os.environ.get("BENCH_MODEL_PATH") else None)
    args = ap.parse_args()
    fx = Fixtures(args.fixtures)
    entries = st.load(args.store or args.run_dir / "t4" / "store.json")
    answers_path = args.run_dir / "reader" / "answers.jsonl"
    answers = [json.loads(x) for x in answers_path.read_text().splitlines()] if answers_path.exists() else []
    result = {
        "fixtures": "v1",
        "drop_list": bool(fx.drop_info),
        "probes": score(fx, answers) if answers else None,
        "fabricated_source": fabricated(fx, entries, args.run_dir / "events" / "pab.jsonl", exported=bool(args.store)),
        **store_checks(fx, entries),
        "memory_tokens": memory_tokens(entries, args.tokenizer),
    }
    out = args.run_dir / "t4" / "oracle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    pr = result["probes"]
    if pr:
        cats = " ".join(f"{c}={v['correct']}/{v['n']}" for c, v in pr["per_category"].items())
        print(f"answerable {pr['answerable']['correct']}/{pr['answerable']['n']}  {cats}")
    print(f"entries {result['n_entries']}  fabricated {len(result['fabricated_source'])}  "
          f"precision {result['source_precision']}  duplicates {result['duplicate_ratio']:.3f}  "
          f"tokens {result['memory_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
