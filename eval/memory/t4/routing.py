"""Check Hippo decisions against T4's introduced versions and admitted-store state.

No inference from candidate text: source_ids stay literal. Unknown ids and story records that neither
introduce an item nor are marked distractors are reported as unscorable, not silently credited.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def admitted(row: dict[str, Any]) -> bool:
    return row.get("entry_id") is not None and row.get("action", "").upper() in {"NEW", "UPDATE", "RELATED", "MERGE"}


def check(facts: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    records = facts["records"]
    items = {item["key"]: item["versions"] for item in facts["items"]}
    introductions: dict[str, list[tuple[str, int]]] = {}
    all_sources: set[str] = set()
    for key, versions in items.items():
        for i, version in enumerate(versions):
            if i and version.get("change") not in {"replaces", "adds"}:
                raise ValueError(f"{key} day {version['day']} needs a replaces/adds label")
            all_sources.update(version["source_ids"])
            for rid in version["source_ids"]:
                if records[rid]["day"] == version["day"]:
                    introductions.setdefault(rid, []).append((key, i))
    state: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    pairs, rows = [], []
    actions: Counter[str] = Counter()
    for n, decision in enumerate(decisions, 1):
        ids = set(decision.get("source_ids") or [])
        action = str(decision.get("action") or "").upper()
        actions[action or "NOT_ADMITTED"] += 1
        carried: dict[str, int] = {}
        for rid in ids:
            for key, i in introductions.get(rid, []):
                carried[key] = max(i, carried.get(key, -1))
        # Newer versions own their supporting records. {C1,C4} carries family_dinner only;
        # C1 is that version's support, not a second verdict for sister. C1 alone carries sister.
        support: set[str] = set()
        selected = []
        for key, i in sorted(carried.items(), key=lambda p: (-items[p[0]][p[1]]["day"], p[0])):
            version = items[key][i]
            triggers = {rid for rid in ids if (key, i) in introductions.get(rid, [])}
            if triggers <= support:
                continue
            selected.append((key, i))
            support.update(version["source_ids"])
        unknown = sorted(ids - records.keys())
        unmapped = sorted(rid for rid in ids & records.keys()
                          if rid not in all_sources and records[rid].get("key") is not None)
        missing = not ids
        verdicts = []
        for key, i in selected:
            current = state.get(key, -1)
            version = items[key][i]
            conflict = False
            if current < 0:
                expected, accepted = ["NEW"], {"NEW", "RELATED"}
            elif current >= i:
                expected, accepted = ["IGNORE"], {"IGNORE"}
            elif version["change"] == "replaces":
                expected, accepted = ["UPDATE"], {"UPDATE", "CONFLICT"}
                conflict = action == "CONFLICT"
            else:
                expected, accepted = ["UPDATE", "RELATED"], {"UPDATE", "RELATED", "NEW"}
            verdict = {"row": n, "seq": decision.get("seq", n), "item": key, "version": i + 1,
                       "day": version["day"], "stored_version": current + 1, "action": action,
                       "expected": expected, "accepted": action in accepted, "conflict_left_open": conflict}
            verdicts.append(verdict)
            pairs.append(verdict)
            if admitted(decision):
                state[key] = max(current, i)
                seen.add((key, i))
        distractor = bool(ids) and not selected and not unknown and not unmapped and all(
            records[rid].get("key") is None for rid in ids)
        distractor_admitted = distractor and action in {"NEW", "RELATED"} and decision.get("entry_id") is not None
        if distractor:
            ok = action == "IGNORE" or decision.get("entry_id") is None or distractor_admitted
        else:
            ok = all(v["accepted"] for v in verdicts) if verdicts else None
        unscorable = missing or bool(unknown or unmapped) or (not verdicts and not distractor)
        rows.append({"row": n, "seq": decision.get("seq", n), "candidate_id": decision.get("candidate_id"),
                     "action": action, "n_pairs": len(verdicts), "accepted": ok,
                     "unscorable": unscorable, "missing_source_ids": missing,
                     "unknown_source_ids": unknown, "unmapped_source_ids": unmapped,
                     "distractor_only": distractor, "distractor_admitted": distractor_admitted})
    return {
        "per_pair": {"n": len(pairs), "accepted": sum(p["accepted"] for p in pairs),
                     "errors": sum(not p["accepted"] for p in pairs),
                     "conflict_left_open": sum(p["conflict_left_open"] for p in pairs)},
        "per_row": {"n": len(rows), "scorable": sum(not r["unscorable"] for r in rows),
                    "accepted": sum(r["accepted"] is True and not r["unscorable"] for r in rows),
                    "errors": sum(r["accepted"] is False for r in rows),
                    "unscorable": sum(r["unscorable"] for r in rows),
                    "distractor_admitted": sum(r["distractor_admitted"] for r in rows)},
        "actions": dict(actions), "pairs": pairs, "rows": rows,
        "missed_versions": [{"item": key, "version": i + 1, "day": version["day"]}
                            for key, versions in items.items() for i, version in enumerate(versions)
                            if (key, i) not in seen],
    }
