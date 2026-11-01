"""Adding evidence_seen and evidence_all to the rows of a run written before the driver recorded it."""
import json

from memory.public import backfill
from memory.tests.public_support import mini_questions


def test_backfill_adds_evidence_seen_from_the_saved_cells(tmp_path):
    q = mini_questions()[0]
    q["haystack_sessions"][0][0]["has_answer"] = True
    run_dir = tmp_path / "run"
    cell = run_dir / "cells" / "q1aa__A__r1"
    cell.mkdir(parents=True)
    events = [{"type": "turn.memory", "data": {"used": [{"kind": "chat", "text": "melo: Congratulations on the degree!"}]}},
              {"type": "turn.memory", "data": {"used": [{"kind": "record", "text": "I hold a degree in Business Administration."}]}}]
    (cell / "turn.json").write_text(json.dumps({"events": events}))
    rows = [{"question_id": "q1aa", "arm": "A", "repeat": 1, "retrieval": {"auto_hit": True}},
            {"question_id": "q1aa", "arm": "A", "repeat": 2, "retrieval": {"auto_hit": True}},  # no saved cell
            {"question_id": "q1aa", "arm": "B", "repeat": 1, "retrieval": {"evidence_seen": 0.5, "evidence_all": 0.5}}]  # recorded
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert backfill.backfill(run_dir, [q]) == (1, 1)
    out = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [r["retrieval"].get("evidence_all") for r in out] == [1.0, None, 0.5]  # the last memory event counts
    assert len((run_dir / "results.jsonl.bak").read_text().splitlines()) == 3
    assert backfill.backfill(run_dir, [q]) == (0, 1)  # a second run changes nothing
