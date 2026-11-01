"""Add ``retrieval.evidence_seen`` and ``evidence_all`` to the rows of a run written before the driver recorded it.

    cd eval
    uv run python -m memory.public.backfill <run-dir> --data <longmemeval_s_cleaned.json>

Each row is recomputed from its cell's saved events (``cells/<question>__<arm>__r<repeat>/turn.json``: the last
``turn.memory`` event and the ``search_memory`` results) and the question's evidence turns. results.jsonl is rewritten in
place and the old file kept as ``results.jsonl.bak``. Run it only while no window writes to the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import data, run


def backfill(run_dir: Path, questions: Sequence[data.Question]) -> tuple[int, int]:
    """(rows filled, rows without a saved cell); rows that have ``evidence_all`` are left as they are."""
    by_id = {str(q["question_id"]): q for q in questions}
    path = run_dir / "results.jsonl"
    rows = run.read_rows(path)
    filled = missing = 0
    for row in rows:
        ret = row.get("retrieval")
        if not isinstance(ret, dict) or "evidence_all" in ret:
            continue
        cell = run_dir / "cells" / f"{row['question_id']}__{row['arm']}__r{row['repeat']}" / "turn.json"
        if not cell.exists():
            missing += 1
            continue
        events = json.loads(cell.read_text()).get("events") or []
        memory = next((e.get("data") for e in reversed(events) if e.get("type") == "turn.memory"), None)
        ret.update(run.evidence_seen(by_id[row["question_id"]], memory if isinstance(memory, dict) else None, events))
        filled += 1
    if filled:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        os.replace(path, path.with_name(path.name + ".bak"))
        os.replace(tmp, path)
    return filled, missing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args(argv)
    if not (args.run_dir / "results.jsonl").exists():
        print(f"{args.run_dir}: no results.jsonl", file=sys.stderr)
        return 2
    filled, missing = backfill(args.run_dir, data.load(args.data))
    print(f"evidence_seen and evidence_all added to {filled} rows; {missing} rows have no saved cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
