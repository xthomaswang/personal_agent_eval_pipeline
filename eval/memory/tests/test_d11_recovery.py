"""D11 prefill recovery and pass-boundary restart tests; no model server."""
import json
import sqlite3
from pathlib import Path

import pytest
from memory.d11 import run as d11


def test_logical_long_store_snapshot(tmp_path):
    assert d11._long_state(tmp_path) == ((), ())
    with sqlite3.connect(tmp_path / "long_term.db") as db:
        db.execute("CREATE TABLE long_term (id INTEGER, status TEXT, text TEXT)")
        db.execute("CREATE TABLE decisions (seq INTEGER, action TEXT)")
        assert d11._long_state(tmp_path) == ((), ())
        db.execute("INSERT INTO long_term VALUES (1, 'active', 'fact')")
        db.execute("INSERT INTO decisions VALUES (1, 'NEW')")
    assert d11._long_state(tmp_path) != ((), ())


def test_cutoff_prevents_run_and_battery_sleep(tmp_path, monkeypatch):
    monkeypatch.setattr(d11, "identity", lambda *a, **k: {"model": "stub"})
    monkeypatch.setattr(d11, "_system_snapshot", lambda: {})
    monkeypatch.setattr(d11, "export_arms", lambda *a: {})
    monkeypatch.setattr(d11.time, "time", lambda: 1000.0)
    monkeypatch.setattr(d11, "_battery_pct", lambda: 20)
    monkeypatch.setattr(d11.time, "sleep", lambda _: pytest.fail("must not sleep across cutoff"))
    monkeypatch.setattr(d11, "_run_item", lambda *a, **k: pytest.fail("must not start run"))
    trial = tmp_path / "trial"
    report = d11.run_d11(trial, passes=1, exercises=("transpose",), cutoff=1100.0)
    assert not report["runs"]
    assert not json.loads((trial / "d11.json").read_text())["complete"]
    assert "battery wait would cross cutoff" in (trial / "d11.log").read_text()


def test_retry_and_pass_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(d11, "identity", lambda *a, **k: {"model": "stub"})
    monkeypatch.setattr(d11, "_guard_battery", lambda *a: None)
    monkeypatch.setattr(d11, "_system_snapshot", lambda: {})
    monkeypatch.setattr(d11, "export_arms", lambda *a: {})
    restarts = []
    monkeypatch.setattr(d11, "_restart_omlx", lambda trial: restarts.append(trial))
    attempts = []

    def fake(item, trial, *args, **kwargs):
        run_dir = trial / "runs" / item["run_id"]
        run_dir.mkdir(parents=True)
        attempts.append(item["run_id"])
        failed = len(attempts) == 1
        if failed:
            (run_dir / "calls.jsonl").write_text(json.dumps({"error": {"code": "prefill_memory_exceeded"}}) + "\n")
        record = {**item, "state": "finished", "verdict": "harness_error" if failed else "pass",
                  "n_llm": 0 if failed else 1, "history_steps": 0, "task_id": "failed-id" if failed else item["run_id"],
                  "run_dir": str(run_dir.relative_to(trial)), "wall_s": 1}
        d11.save(run_dir / "record.json", record)
        return record

    monkeypatch.setattr(d11, "_run_item", fake)
    trial = tmp_path / "trial"
    report = d11.run_d11(trial, passes=2, exercises=("transpose",), restart_between_passes=True)
    assert len(restarts) == 2  # prefill retry, then pass 2
    assert len(attempts) == 5
    first = report["runs"][0]
    assert first["infra_failed_attempts"][0]["task_id"] == "failed-id"
    assert (trial / first["infra_failed_attempts"][0]["run_dir"] / "calls.jsonl").exists()
    assert d11.run_d11(trial, passes=2, exercises=("transpose",), restart_between_passes=True)["runs"] == report["runs"]
    assert len(restarts) == 2


@pytest.mark.parametrize("reason", ["steps", "entries", "calls"])
def test_unsafe_prefill_stops(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(d11, "identity", lambda *a, **k: {"model": "stub"})
    monkeypatch.setattr(d11, "_guard_battery", lambda *a: None)
    monkeypatch.setattr(d11, "_system_snapshot", lambda: {})
    states = iter([(), (1,)]) if reason == "entries" else None
    monkeypatch.setattr(d11, "_long_state", lambda *a: next(states) if states else ())
    restarts = []
    monkeypatch.setattr(d11, "_restart_omlx", lambda trial: restarts.append(trial))

    def fake(item, trial, *args, **kwargs):
        run_dir = trial / "runs" / item["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "calls.jsonl").write_text(json.dumps({"error": {"code": "prefill_memory_exceeded"}}) + "\n")
        return {**item, "state": "finished", "verdict": "harness_error", "n_llm": 1 if reason == "calls" else 0,
                "history_steps": 1 if reason == "steps" else 0, "task_id": "id", "run_dir": str(run_dir.relative_to(trial))}

    monkeypatch.setattr(d11, "_run_item", fake)
    trial = tmp_path / "trial"
    with pytest.raises(RuntimeError, match="inspect"):
        d11.run_d11(trial, passes=1, exercises=("transpose",), restart_between_passes=True)
    assert not restarts
    with pytest.raises(RuntimeError, match="unresolved"):
        d11.run_d11(trial, passes=1, exercises=("transpose",), restart_between_passes=True)
