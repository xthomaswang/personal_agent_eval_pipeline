"""D11 pass-boundary snapshots and resume behavior."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from memory.d11 import run as d11


def test_snapshots_at_pass_boundaries_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d11, "identity", lambda localmelo, **kw: {"model": "stub", "passes": kw["passes"]})
    monkeypatch.setattr(d11, "_system_snapshot", lambda: {})
    monkeypatch.setattr(d11, "_guard_battery", lambda *args, **kw: True)
    monkeypatch.setattr(d11, "_long_state", lambda mem: None)
    monkeypatch.setattr(d11, "export_arms", lambda trial, localmelo: {})
    monkeypatch.setattr(d11, "load_trial_report", lambda trial: {})
    monkeypatch.setattr(d11, "render_markdown", lambda report: "")
    connections: dict[str, sqlite3.Connection] = {}
    restarts: list[int] = []

    def fake_run(item: dict, trial: Path, localmelo: Path, env: Path, **kw: object) -> dict:
        arm = item["arm"]
        mem = trial / f"mem-{arm}"
        if arm not in connections:
            conn = sqlite3.connect(mem / "memory.db")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE entries (pass INTEGER)")
            conn.commit()
            connections[arm] = conn
        connections[arm].execute("INSERT INTO entries VALUES (?)", (item["pass"],))
        connections[arm].commit()
        (mem / "note.txt").write_text(f"{arm} pass {item['pass']}")
        run_dir = trial / "runs" / item["run_id"]
        run_dir.mkdir(parents=True)
        return {**item, "state": "finished", "verdict": "pass", "n_llm": 1, "wall_s": 1,
                "run_dir": str(run_dir.relative_to(trial)), "history_steps": 1}

    def fake_restart(trial: Path) -> None:
        assert (trial / "snapshots/p01/index.json").exists()
        restarts.append(1)

    monkeypatch.setattr(d11, "_run_item", fake_run)
    monkeypatch.setattr(d11, "_restart_omlx", fake_restart)
    trial = tmp_path / "trial"
    try:
        d11.run_d11(trial, passes=2, exercises=("transpose",), restart_between_passes=True)
        assert restarts == [1]
        state = json.loads((trial / "d11.json").read_text())
        assert set(state["snapshots"]) == {"1", "2"}
        before = {}
        for number in (1, 2):
            root = trial / "snapshots" / f"p{number:02d}"
            index = json.loads((root / "index.json").read_text())
            assert index["pass"] == number
            for entry in index["files"]:
                path = root / entry["path"]
                assert entry["size"] == path.stat().st_size
                assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                before[path] = (path.read_bytes(), path.stat().st_mtime_ns)
            before[root / "index.json"] = ((root / "index.json").read_bytes(), (root / "index.json").stat().st_mtime_ns)
            for arm in d11.ARMS:
                mem = root / f"mem-{arm}"
                with sqlite3.connect(f"{(mem / 'memory.db').as_uri()}?mode=ro", uri=True) as conn:
                    assert conn.execute("SELECT pass FROM entries ORDER BY pass").fetchall() == [(i,) for i in range(1, number + 1)]
                assert (mem / "note.txt").read_text() == f"{arm} pass {number}"
                assert not list(mem.glob("*.db-wal"))
        # Simulate a crash after atomic publication but before checkpoint update.
        del state["snapshots"]["1"]
        d11.save(trial / "d11.json", state)
        d11.run_d11(trial, passes=2, exercises=("transpose",), restart_between_passes=True)
        assert "1" in json.loads((trial / "d11.json").read_text())["snapshots"]
        assert restarts == [1]
        assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
    finally:
        for conn in connections.values():
            conn.close()
