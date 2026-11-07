"""Freeze the TM57 ID-only list without importing evaluation/model code.

For each of the four sampled types, use a separate
random.Random(429).sample(sorted(eligible_ids), 25), then sort the selection.
Keep every remaining assistant ID. Reuse exactly the heldout428 preference
and abstention IDs as 25 NOT FRESH rows, outside the fresh/primary IDs.
Only question_id and question_type are inspected from dataset records.

Run with EVAL/.venv/bin/python -I -B. Default mode creates fresh_429.json
exclusively; --check regenerates and compares exact bytes without writing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


PUBLIC = Path(__file__).resolve().parent
EVAL = PUBLIC.parent.parent
OUTPUT = PUBLIC / "fresh_429.json"
HELDOUT = PUBLIC / "heldout_428.json"
OLD_CONFIG = EVAL / "memory/runs/lme-pkg-9003264/public.json"
OLD_QUESTIONS = EVAL / "memory/runs/lme-pkg-9003264/questions.json"
DATASET = (
    Path.home()
    / "Developer/ml_source/datasets/longmemeval/longmemeval_s_cleaned.json"
)
SAMPLED_TYPES = (
    "single-session-user",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)
FRESH_COUNTS = {**dict.fromkeys(SAMPLED_TYPES, 25), "single-session-assistant": 20}
NOT_FRESH_COUNTS = {"single-session-preference": 13, "abstention": 12}
HELDOUT_COUNTS = {
    **dict.fromkeys((*SAMPLED_TYPES, "single-session-assistant"), 19),
    **NOT_FRESH_COUNTS,
}
DATASET_COUNTS = {
    "single-session-user": 64,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "multi-session": 121,
    "temporal-reasoning": 127,
    "knowledge-update": 72,
    "abstention": 30,
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def group(question_id: str, raw_type: str) -> str:
    """Use ID/type metadata, never answer text, to identify abstention."""
    return "abstention" if question_id.endswith("_abs") else raw_type


def id_type_sha256(ids: set[str], groups: dict[str, str]) -> str:
    projection = [[question_id, groups[question_id]] for question_id in sorted(ids)]
    encoded = json.dumps(projection, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_sources():
    config = load_json(OLD_CONFIG)["config"]
    old_rows = load_json(OLD_QUESTIONS)
    heldout = load_json(HELDOUT)
    assert set(heldout) == {"seed", "rule", "by_type", "ids"}
    assert heldout["seed"] == 428 and isinstance(heldout["rule"], str)
    assert config["data"]["name"] == DATASET.name

    # Pin the entire dataset to the old run before parsing; discard all prose.
    payload = DATASET.read_bytes()
    dataset_sha256 = hashlib.sha256(payload).hexdigest()
    assert dataset_sha256 == config["data"]["sha256"], "Dataset fingerprint changed"
    metadata = [
        (row["question_id"], row["question_type"])
        for row in json.loads(payload)
    ]
    del payload
    assert len(metadata) == 500
    assert all(isinstance(qid, str) and isinstance(kind, str) for qid, kind in metadata)
    raw_types = dict(metadata)
    assert len(raw_types) == 500, "Dataset IDs must be unique"
    assert set(raw_types.values()) == set(DATASET_COUNTS) - {"abstention"}
    groups = {qid: group(qid, kind) for qid, kind in metadata}
    assert dict(Counter(groups.values())) == DATASET_COUNTS

    old_ids = {row["question_id"] for row in old_rows}
    assert len(old_rows) == len(old_ids) == 120
    assert len(config["questions"]) == 120
    assert set(config["questions"]) == old_ids, "Old run/config IDs disagree"
    assert old_ids <= groups.keys()
    for row in old_rows:
        qid = row["question_id"]
        assert row["question_type"] == raw_types[qid]
        assert isinstance(row["abstention"], bool)
        assert row["abstention"] == (groups[qid] == "abstention")

    heldout_ids = set(heldout["ids"])
    assert len(heldout["ids"]) == len(heldout_ids) == 120
    assert heldout["ids"] == sorted(heldout_ids)
    assert heldout_ids <= groups.keys()
    assert not old_ids & heldout_ids, "Old120 and heldout120 must be disjoint"
    assert {kind: len(ids) for kind, ids in heldout["by_type"].items()} == HELDOUT_COUNTS
    grouped_ids = []
    for kind, ids in heldout["by_type"].items():
        assert ids == sorted(set(ids)), "Heldout groups must be sorted and unique"
        assert all(groups[qid] == kind for qid in ids)
        grouped_ids.extend(ids)
    assert len(grouped_ids) == len(set(grouped_ids)) == 120
    assert set(grouped_ids) == heldout_ids

    sources = {
        "dataset": {
            "label": "LongMemEval source dataset; metadata-only selection",
            "path": "~/Developer/ml_source/datasets/longmemeval/longmemeval_s_cleaned.json",
            "sha256": dataset_sha256,
            "count": 500,
            "by_type_counts": DATASET_COUNTS,
            "grouping": "question_id ending in _abs => abstention; otherwise question_type",
        },
        "old_120": {
            "label": "lme-pkg-9003264: all 120 IDs excluded",
            "config": str(OLD_CONFIG.relative_to(EVAL)),
            "questions": str(OLD_QUESTIONS.relative_to(EVAL)),
            "count": 120,
            "id_type_sha256": id_type_sha256(old_ids, groups),
        },
        "heldout_428": {
            "label": "heldout428: excluded except the labelled 25 NOT FRESH IDs",
            "path": str(HELDOUT.relative_to(EVAL)),
            "count": 120,
            "id_type_sha256": id_type_sha256(heldout_ids, groups),
        },
    }
    return groups, old_ids, heldout, sources


def generate(groups, old_ids, heldout, sources):
    heldout_ids = set(heldout["ids"])
    excluded = old_ids | heldout_ids
    eligible = {
        kind: sorted(qid for qid, actual in groups.items() if actual == kind and qid not in excluded)
        for kind in DATASET_COUNTS
    }
    assert not eligible["single-session-preference"]
    assert not eligible["abstention"]
    selected = {}
    for kind in SAMPLED_TYPES:
        assert len(eligible[kind]) >= 25
        # Reinitialize for EACH type: this must not become a shared RNG.
        selected[kind] = sorted(random.Random(429).sample(eligible[kind], 25))
    selected["single-session-assistant"] = eligible["single-session-assistant"]
    assert {kind: len(ids) for kind, ids in selected.items()} == FRESH_COUNTS
    fresh_ids = sorted(qid for ids in selected.values() for qid in ids)

    for kind in NOT_FRESH_COUNTS:
        selected[kind] = list(heldout["by_type"][kind])
    not_fresh_ids = sorted(qid for kind in NOT_FRESH_COUNTS for qid in selected[kind])
    by_type = {kind: selected[kind] for kind in heldout["by_type"]}
    ids = sorted(qid for entries in by_type.values() for qid in entries)
    fresh_set, not_fresh_set, all_ids = set(fresh_ids), set(not_fresh_ids), set(ids)
    assert len(fresh_ids) == len(fresh_set) == 120
    assert len(not_fresh_ids) == len(not_fresh_set) == 25
    assert len(ids) == len(all_ids) == 145
    assert not fresh_set & not_fresh_set
    assert not fresh_set & excluded, "Fresh IDs overlap an earlier set"
    assert not all_ids & old_ids, "Selected IDs overlap old120"
    assert all_ids & heldout_ids == not_fresh_set, "Only labelled25 may overlap heldout120"
    assert not_fresh_set == {
        qid for kind in NOT_FRESH_COUNTS for qid in heldout["by_type"][kind]
    }
    assert fresh_set | not_fresh_set == all_ids
    assert dict(Counter(groups[qid] for qid in fresh_ids)) == FRESH_COUNTS
    assert dict(Counter(groups[qid] for qid in not_fresh_ids)) == NOT_FRESH_COUNTS
    for kind, entries in by_type.items():
        assert entries == sorted(set(entries))
        assert all(groups[qid] == kind for qid in entries)

    return {
        "seed": 429,
        "rule": (
            "TM57: exclude old120 and heldout120; per sampled type sort eligible IDs, "
            "use a new random.Random(429).sample(eligible_ids, 25), then sort selected IDs; "
            "keep every remaining assistant ID. Reuse heldout428's 13 preference and "
            "12 abstention IDs as 25 NOT FRESH rows, reported separately from primary."
        ),
        "by_type": by_type,
        "ids": ids,
        "fresh_ids": fresh_ids,
        "primary_ids": list(fresh_ids),
        "not_fresh_ids": not_fresh_ids,
        "freshness": {
            "fresh": {"label": "FRESH", "count": 120, "by_type_counts": FRESH_COUNTS},
            "not_fresh": {
                "label": "NOT FRESH",
                "count": 25,
                "by_type_counts": NOT_FRESH_COUNTS,
                "source": "heldout_428",
                "reporting": "Separate from primary/fresh results; none of these IDs is fresh",
            },
            "primary_ids_rule": "Exactly fresh_ids; excludes all 25 NOT FRESH IDs",
        },
        "sources": sources,
    }


def render(document) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare the locked list; never write")
    args = parser.parse_args()
    if not args.check and OUTPUT.exists():
        parser.error("The locked list already exists; use --check (no overwrite is permitted)")
    inputs = load_sources()
    document = generate(*inputs)
    expected = render(document)
    assert expected == render(generate(*inputs)), "Regeneration is not deterministic"
    if args.check:
        assert OUTPUT.read_bytes() == expected, "Locked list differs from regeneration"
    else:
        # Exclusive creation also prevents overwriting a file created after the check.
        with OUTPUT.open("xb") as destination:
            destination.write(expected)
    for kind in sorted(document["by_type"]):
        fresh = FRESH_COUNTS.get(kind, 0)
        not_fresh = NOT_FRESH_COUNTS.get(kind, 0)
        print(f"{kind}: total={fresh + not_fresh} fresh={fresh} not_fresh={not_fresh}")


if __name__ == "__main__":
    main()
