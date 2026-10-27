"""Check a page's items against its rendered elements (README.md, "Items").

    python3 validate.py pages/<page>.tasks.json <out>.elements.json

Exit 1 with one line per problem: unknown ids, ids not visible on the rendered page, malformed items,
duplicate item ids. Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TYPES = {"click": {"instruction", "target"}, "fill": {"instruction", "profile", "expect"},
         "read": {"question", "answer", "match"}}
OUTCOMES = {"value", "checked", "option"}


def problems(tasks: dict, elements: dict) -> list[str]:
    out = []
    qa = elements.get("qa", {})
    if not isinstance(tasks.get("page"), str) or tasks.get("language") not in ("en", "zh"):
        out.append("top level: needs page (string) and language (en or zh)")
    items = tasks.get("items")
    if not isinstance(items, list) or not items:
        return out + ["top level: items must be a non-empty list"]
    seen = set()
    for it in items:
        iid = it.get("id", "?")
        if iid in seen:
            out.append(f"{iid}: duplicate id")
        seen.add(iid)
        kind = it.get("type")
        if kind not in TYPES:
            out.append(f"{iid}: type must be one of {', '.join(TYPES)}")
            continue
        missing = TYPES[kind] - it.keys()
        if missing:
            out.append(f"{iid}: missing {', '.join(sorted(missing))}")
            continue
        if kind == "click":
            if not isinstance(it["target"], list) or not it["target"]:
                out.append(f"{iid}: target must be a non-empty list of data-qa ids")
            out += [f"{iid}: target {t!r} is not a visible data-qa element" for t in it["target"] if t not in qa]
        elif kind == "fill":
            if not isinstance(it["expect"], dict) or not it["expect"]:
                out.append(f"{iid}: expect must be a non-empty object")
                continue
            for k, v in it["expect"].items():
                if k not in qa:
                    out.append(f"{iid}: expect {k!r} is not a visible data-qa element")
                if not isinstance(v, dict) or len(v) != 1 or next(iter(v)) not in OUTCOMES:
                    out.append(f"{iid}: expect {k!r} must be one of {{value|checked|option: ...}}")
            out += [f"{iid}: forbid {f!r} is not a visible data-qa element" for f in it.get("forbid", []) if f not in qa]
            out += [f"{iid}: {f!r} is both expected and forbidden" for f in it.get("forbid", []) if f in it["expect"]]
        elif kind == "read":
            if it["match"] not in ("text", "number", "set"):
                out.append(f"{iid}: match must be text, number or set")
            if not str(it["answer"]).strip():
                out.append(f"{iid}: empty answer")
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    elements = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    found = problems(tasks, elements)
    for p in found:
        print(p)
    counts = {k: sum(i.get("type") == k for i in tasks.get("items", [])) for k in TYPES}
    print(f"{tasks.get('page')}: {counts}; {'OK' if not found else f'{len(found)} problem(s)'}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
