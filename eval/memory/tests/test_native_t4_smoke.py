"""CPU-only end-to-end smoke, skipped without the suite's localmelo clone."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from memory.tests.native_stub import observed_fact, server
from memory.t4 import native
from memory.t4 import store as st


@pytest.mark.skipif(not (native.VENDOR / ".venv/bin/melo").exists(), reason="suite's localmelo venv is not present")
def test_scripted_native_trial(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    trial = tmp_path / "trial"
    with server(tmp_path / "stub-requests.jsonl") as url:
        env = {**os.environ, "OMLX_BASE_URL": url, "BENCH_MODEL": "scripted-native-t4",
               "BENCH_LOCALMELO_REPO": str(native.VENDOR.resolve()), "BENCH_UPSTREAM_TOKEN_CMD": "",
               "HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = native.module_command("memory.t4.native", "--run-dir", str(trial))
        with (tmp_path / "driver.out").open("w") as out, (tmp_path / "driver.err").open("w") as err:
            proc = subprocess.run(cmd, cwd=native.ROOT, env=env, stdout=out, stderr=err, timeout=300)
        assert proc.returncode == 0, (tmp_path / "driver.err").read_text()
        # A completed resume must not ingest or read again.
        n_requests = len(native.lines(tmp_path / "stub-requests.jsonl"))
        subprocess.run(cmd, cwd=native.ROOT, env=env, check=True, capture_output=True, timeout=30)
        assert len(native.lines(tmp_path / "stub-requests.jsonl")) == n_requests
    state = json.loads((trial / "t4/native.json").read_text())
    assert state["complete"] and len(state["sessions"]) == 10
    events = native.lines(trial / "events/pab.jsonl")
    assert [(r["session"], r["tool"]) for r in events] == [(day, tool) for day in range(1, 11) for tool in ("feed_day", "submit")]
    with sqlite3.connect(f"file:{trial / 'mem/history.db'}?mode=ro", uri=True) as db:
        tasks = db.execute("select count(*) from tasks").fetchone()[0]
        tool_steps = db.execute("select count(*) from steps").fetchone()[0]
    assert tasks == 10 and tool_steps == 20
    for day in range(1, 11):
        result = native.log_records(trial / f"sessions/day{day:02d}/run/stdout.log")[-1]
        assert Path(result["config"]["memory_dir"]) == trial / "mem"
        assert result["status"] == "completed" and result["config"]["reasoning"] == "compress"
        assert result["config"].get("admit_on_outcome") in (None, False)
    stores = {"native": st.load(trial / "t4/export.jsonl"),
              "store-everything": st.load(trial / "t4/store-everything.jsonl")}
    assert all(stores.values()), "compress should admit at least each session's episodic user_request digest"
    requests = native.lines(tmp_path / "stub-requests.jsonl")
    fact = observed_fact(requests[1]["messages"])
    assert any(fact in entry["text"] for entry in stores["native"]), "the tool-derived note must reach the export"
    decisions = native.lines(trial / "t4/decisions.jsonl")
    assert decisions
    assert any("Return at most" in str(r["messages"][-1].get("content", "")) for r in requests)
    for run in (trial, trial / "comparators/store-everything"):
        assert len(native.lines(run / "reader/answers.jsonl")) == 60
        oracle = json.loads((run / "t4/oracle.json").read_text())
        assert oracle["probes"]["answerable"] == {"correct": 0, "n": 48}  # scripted nulls, not a quality result
        assert oracle["probes"]["per_category"]["abstention"] == {"correct": 12, "n": 12}
    native.save(tmp_path / "smoke.json", {"tasks": tasks, "tool_steps": tool_steps, "pab_calls": len(events),
                                        "entries": {k: len(v) for k, v in stores.items()},
                                        "decisions": len(decisions),
                                        "entries_with_sources": sum(bool(e["source_ids"]) for e in stores["native"]),
                                        "decisions_with_sources": sum(bool(d.get("source_ids")) for d in decisions),
                                        "tool_fact_kept": True, "scripted": True})
