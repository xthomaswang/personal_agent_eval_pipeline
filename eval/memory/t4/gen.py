"""Generate T4's fixtures from scenario.py.

    uv run python -m memory.t4.gen --seed 1            # writes memory/t4/fixtures/v1/
    uv run python -m memory.t4.gen --seed 1 --check    # exit 1 if the committed fixtures differ

The seed places the distractors (which day, what time) and draws every id; the scenario's own records,
items and probes are fixed. Output:

    days/day01.jsonl ... day10.jsonl   the records feed_day returns, one JSON object per line, by time
    facts.yaml                         timeline, entities, every record's resolved values, the gold items
    probes.yaml                        the 60 probes with their typed answers and supporting record ids
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from memory.t4 import scenario as sc

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "v1"
CATEGORIES = ("extraction", "cross_session", "temporal", "knowledge_update", "abstention")
PROBES_PER_CATEGORY = 12
INTERNAL = ("key", "day", "time", "resolved", "window", "at", "who")


def day_date(day: int) -> dt.date:
    return dt.date.fromisoformat(sc.START) + dt.timedelta(days=day - 1)


def stamp(day: int, hhmm: str) -> str:
    return f"{day_date(day).isoformat()}T{hhmm}:00{sc.TZ}"


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class Ids:
    """Opaque ids drawn from the seed; never UUID-shaped, so they cannot be mistaken for machine ids."""

    def __init__(self, rng: random.Random) -> None:
        self.rng, self.seen = rng, set()

    def new(self, prefix: str, nbytes: int = 5) -> str:
        while True:
            value = f"{prefix}{self.rng.getrandbits(8 * nbytes):0{2 * nbytes}x}"
            if value not in self.seen:
                self.seen.add(value)
                return value


def _distractor(spec: dict[str, Any], day: int, rng: random.Random) -> dict[str, Any]:
    lo, hi = (_minutes(t) for t in spec["window"])
    time = _hhmm(rng.randint(lo, hi))
    if spec["kind"] == "mail":
        return sc.mail("", day, time, spec["sender"], spec["subject"], spec["body"], to=spec.get("to", sc.OWNER_MAIL))
    if spec["kind"] == "message":
        return sc.msg("", day, time, spec["who"], spec["text"])
    start, end = spec["at"]
    return sc.event("", day, time, spec["title"], stamp(day, start), stamp(day, end), calendar=spec["calendar"])


def _wire(rec: dict[str, Any], record_id: str, ids: Ids) -> dict[str, Any]:
    """The record as feed_day returns it: the macos-apps-mcp read shape plus record_id and kind."""
    out: dict[str, Any] = {"record_id": record_id, "kind": rec["kind"]}
    if rec["kind"] == "mail":
        domain = rec["sender"].rsplit("@", 1)[-1].rstrip(">")
        out |= {"id": f"<{ids.new('', 6)}@{domain}>", "date": stamp(rec["day"], rec["time"]), "mailbox": "INBOX"}
    elif rec["kind"] == "message":
        out |= {"id": ids.new("msg-", 6), "date": stamp(rec["day"], rec["time"])}
    else:
        out |= {"id": ids.new("evt-", 6)}
    out |= {k: v for k, v in rec.items() if k not in INTERNAL and k != "kind"}
    return out


def build(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ids = Ids(rng)
    pool = list(sc.DISTRACTORS)
    rng.shuffle(pool)
    per_day = sc.DISTRACTORS_PER_DAY
    if len(pool) < per_day * sc.DAYS:
        raise SystemExit(f"need {per_day * sc.DAYS} distractors, have {len(pool)}")

    by_key: dict[str, str] = {}
    days: dict[int, list[dict[str, Any]]] = {}
    index: dict[str, dict[str, Any]] = {}
    for day in range(1, sc.DAYS + 1):
        recs = [r for r in sc.RECORDS if r["day"] == day]
        recs += [_distractor(spec, day, rng) for spec in pool[(day - 1) * per_day : day * per_day]]
        wired = []
        for rec in recs:
            rid = ids.new("r")
            if rec["key"]:
                by_key[rec["key"]] = rid
            index[rid] = {"day": day, "kind": rec["kind"], "key": rec["key"] or None, "resolved": rec["resolved"]}
            wired.append((rec["time"], rid, _wire(rec, rid, ids)))
        days[day] = [w for _, _, w in sorted(wired, key=lambda x: (x[0], x[1]))]

    def rids(keys: list[str], where: str) -> list[str]:
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(f"{where}: unknown record keys {missing}")
        return [by_key[k] for k in keys]

    items = []
    for item in sc.ITEMS:
        versions = []
        for v in item["versions"]:
            late = [k for k in v["sources"] if next(r["day"] for r in sc.RECORDS if r["key"] == k) > v["day"]]
            if late:
                raise SystemExit(f"item {item['key']} day {v['day']}: cites records of later days {late}")
            version = {"day": v["day"], "text": v["text"], "source_ids": rids(v["sources"], item["key"])}
            if "change" in v:
                version["change"] = v["change"]
            versions.append(version)
        items.append({"key": item["key"], "versions": versions})

    probes = []
    for p in sc.PROBES:
        if p["category"] not in CATEGORIES:
            raise SystemExit(f"{p['id']}: unknown category {p['category']}")
        probes.append({**{k: p[k] for k in ("id", "category", "type", "question", "answer")},
                       "support": rids(p["support"], p["id"])})
    counts = {c: sum(p["category"] == c for p in probes) for c in CATEGORIES}
    if set(counts.values()) != {PROBES_PER_CATEGORY}:
        raise SystemExit(f"need {PROBES_PER_CATEGORY} probes per category, have {counts}")

    facts = {
        "version": 1,
        "seed": seed,
        "timeline": {"start": sc.START, "days": sc.DAYS, "tz": sc.TZ, "owner": sc.OWNER},
        "entities": {k: {"type": t, "name": n, "aliases": list(a)} for k, (t, n, a) in sc.ENTITIES.items()},
        "records": index,
        "items": items,
    }
    return {"days": days, "facts": facts, "probes": {"version": 1, "probes": probes}}


def write(out: Path, built: dict[str, Any]) -> None:
    (out / "days").mkdir(parents=True, exist_ok=True)
    for day, recs in built["days"].items():
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        (out / "days" / f"day{day:02d}.jsonl").write_text(lines)
    for name in ("facts", "probes"):
        text = yaml.safe_dump(built[name], sort_keys=False, allow_unicode=True, width=120)
        (out / f"{name}.yaml").write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=FIXTURES)
    ap.add_argument("--check", action="store_true", help="compare with the fixtures in --out instead of writing")
    args = ap.parse_args()
    built = build(args.seed)
    if not args.check:
        write(args.out, built)
        n = sum(len(r) for r in built["days"].values())
        print(f"{n} records over {len(built['days'])} days, {len(built['probes']['probes'])} probes -> {args.out}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        write(Path(tmp), built)
        differ = [p.relative_to(tmp) for p in sorted(Path(tmp).rglob("*")) if p.is_file()
                  and (not (args.out / p.relative_to(tmp)).exists()
                       or (args.out / p.relative_to(tmp)).read_bytes() != p.read_bytes())]
    for p in differ:
        print(f"differs: {p}", file=sys.stderr)
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
