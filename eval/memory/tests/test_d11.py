"""D11 cross-task runner tests: schedule determinism/alternation, resume, report math, and stub smoke."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from memory.d11 import run as d11
from smoke import data, paths


@contextmanager
def stub_server(log: Path) -> Iterator[str]:
    """Scripted OpenAI-compatible server following stub_server.py's 'fact' scenario."""
    lock = threading.Lock()

    def make_reply(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        last = str(messages[-1].get("content") or "") if messages else ""
        if not body.get("tools") and "Return at most" in last:
            ids = list(dict.fromkeys(re.findall(r"\[(m\d+)\]", json.dumps(messages))))
            content = json.dumps({"ids": ids})
            reasoning = "Pick every memory."
        else:
            content = "Done."
            reasoning = "Answer from what I have."
        return {
            "id": f"chatcmpl-stub-{len(messages)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "scripted-d11"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.loads(raw)
            with lock, log.open("a") as out:
                out.write(json.dumps(body) + "\n")
            payload = json.dumps(make_reply(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            payload = json.dumps({"object": "list", "data": [{"id": "scripted-d11", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_schedule_determinism_and_alternation() -> None:
    s1 = d11.schedule()
    s2 = d11.schedule()
    assert s1 == s2
    assert len(s1) == 120
    assert d11.EXERCISES == (
        "transpose",
        "proverb",
        "pig-latin",
        "list-ops",
        "variable-length-quantity",
        "robot-name",
        "grade-school",
        "affine-cipher",
        "scale-generator",
        "wordy",
        "phone-number",
        "react",
    )

    for pass_num in range(1, 6):
        pass_rows = [r for r in s1 if r["pass"] == pass_num]
        assert len(pass_rows) == 24
        off_order = [r["exercise"] for r in pass_rows if r["arm"] == "off"]
        on_order = [r["exercise"] for r in pass_rows if r["arm"] == "on"]
        # Same seeded permutation for both arms within the pass
        assert off_order == on_order == d11.pass_permutation(pass_num)
        assert set(off_order) == set(d11.EXERCISES)
        # Exercises run in both arms back to back, with first arm alternating by pass
        expected_arms = ("off", "on") if pass_num % 2 == 1 else ("on", "off")
        for i in range(0, 24, 2):
            pair = pass_rows[i : i + 2]
            assert pair[0]["exercise"] == pair[1]["exercise"]
            assert (pair[0]["arm"], pair[1]["arm"]) == expected_arms
            assert pair[0]["first_arm"] == expected_arms[0]

    # Different pass numbers produce different permutations
    assert d11.pass_permutation(1) != d11.pass_permutation(2)


def test_harness_flags_shared_mem_dir_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        d11,
        "identity",
        lambda localmelo, **kw: {
            "protocol": "d11-v1",
            "localmelo_sha": "356ff19",
            "passes": kw["passes"],
            "exercises": list(kw["exercises"]),
            "model": "stub",
        },
    )

    def fake_invoke(
        prompt: str,
        workspace: Path,
        run_dir: Path,
        mem_dir: Path,
        home_dir: Path,
        env_file: Path,
        localmelo: Path,
        *,
        arm: str,
        timeout_s: float,
        max_steps: int,
        thinking: str,
    ) -> dict[str, Any]:
        # Verify workspace has a fresh exercise stub before running
        stub_file = workspace / "transpose.py" if "transpose" in run_dir.name else workspace / "proverb.py"
        assert stub_file.exists()
        assert "# dirty" not in stub_file.read_text()
        stub_file.write_text(stub_file.read_text() + "\n# dirty\n")

        if run_dir.name == "p02__transpose__on" and not getattr(fake_invoke, "allow_all", False):
            # Simulate an interrupted process that already touched history.db and summary.json
            run_dir.mkdir(parents=True, exist_ok=True)
            d11.save(run_dir / "summary.json", {"exit_code": -15, "timed_out": False, "wall_s": 0.4, "steps": {"n_llm": 1}})
            with sqlite3.connect(mem_dir / "history.db") as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, query TEXT, status TEXT, result TEXT)")
                conn.execute("CREATE TABLE IF NOT EXISTS steps (step_id TEXT PRIMARY KEY, task_id TEXT)")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS llm_calls (task_id TEXT, seq INTEGER, kind TEXT, recall_json TEXT)"
                )
                conn.execute("INSERT INTO tasks VALUES ('tid-interrupted', 'q', 'running', '')")
                conn.execute("INSERT INTO steps VALUES ('s1', 'tid-interrupted')")
            raise KeyboardInterrupt("simulated mid-run interruption")

        calls.append(
            {
                "run_id": run_dir.name,
                "arm": arm,
                "mem_dir": mem_dir,
                "home_dir": home_dir,
                "timeout_s": timeout_s,
                "max_steps": max_steps,
                "thinking": thinking,
            }
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        task_id = f"tid-{run_dir.name}"
        summary = {"exit_code": 0, "timed_out": False, "wall_s": 1.2, "steps": {"n_llm": 2, "n_tool": 1}}
        d11.save(run_dir / "summary.json", summary)
        (run_dir / "stdout.log").write_text(json.dumps({"task_id": task_id, "status": "completed"}) + "\n")
        return summary

    monkeypatch.setattr(d11, "_invoke_harness", fake_invoke)
    monkeypatch.setattr(
        d11.checks,
        "check",
        lambda task, ws, run_dir: {"kind": "aider", "passed": True, "returncode": 0, "timed_out": False, "detail": "ok"},
    )
    monkeypatch.setattr(
        d11,
        "export_arms",
        lambda trial, localmelo: {
            arm: {"mem_dir": f"mem-{arm}", "store": f"mem-{arm}/export.jsonl", "decisions": f"mem-{arm}/decisions.jsonl"}
            for arm in d11.ARMS
        },
    )

    trial = tmp_path / "trial"
    with pytest.raises(KeyboardInterrupt, match="simulated mid-run interruption"):
        d11.run_d11(trial, passes=2, exercises=("transpose", "proverb"))

    state_mid = json.loads((trial / "d11.json").read_text())
    assert state_mid["config"]["localmelo_sha"] == "356ff19"
    assert len(state_mid["schedule"]) == 8
    assert state_mid["runs"]["p02__transpose__on"]["state"] == "started"

    # Resume must NOT replay the interrupted p02__transpose__on run: it records it as "interrupted",
    # salvages summary.json and history.db, leaves its pair out of D2 pairs, and lists it in the report.
    fake_invoke.allow_all = True  # type: ignore[attr-defined]
    report = d11.run_d11(trial, passes=2, exercises=("transpose", "proverb"))
    assert "p02__transpose__on" not in [c["run_id"] for c in calls]
    assert len(calls) == 7
    for c in calls:
        assert c["mem_dir"] == trial / f"mem-{c['arm']}"
        assert c["home_dir"] == trial / "home"
        assert c["timeout_s"] == 600.0 and c["max_steps"] == 30 and c["thinking"] == "off"

    state_done = json.loads((trial / "d11.json").read_text())
    salvaged = state_done["runs"]["p02__transpose__on"]
    assert salvaged["state"] == "interrupted" and salvaged["verdict"] == "interrupted"
    assert salvaged["task_id"] == "tid-interrupted"
    assert salvaged["history_steps"] == 1 and salvaged["n_llm"] == 1 and salvaged["wall_s"] == 0.4

    assert [r["run_id"] for r in report["interrupted"]] == ["p02__transpose__on"]
    assert report["d2"]["interrupted_runs"] == ["p02__transpose__on"]
    # Pass 2 has 2 exercises, but transpose is excluded because its 'on' run was interrupted -> 1 pair left
    assert report["d2"]["n_pairs"] == 1
    assert "`p02__transpose__on`" in (trial / "report.md").read_text()

    # Re-running a completed trial executes nothing
    d11.run_d11(trial, passes=2, exercises=("transpose", "proverb"))
    assert len(calls) == 7

    # Changed config is refused
    with pytest.raises(ValueError, match="changed"):
        d11.run_d11(trial, passes=3, exercises=("transpose", "proverb"))


def test_report_math_recall_splits_and_d2_decision_rule() -> None:
    # Build a synthetic 2-pass (passes 1 and 2) x 2-exercise dataset
    runs = [
        # Pass 1 (baseline)
        {"pass": 1, "exercise": "transpose", "arm": "off", "task_id": "t1_tr_off", "verdict": "pass", "n_llm": 6, "wall_s": 12.0},
        {"pass": 1, "exercise": "transpose", "arm": "on", "task_id": "t1_tr_on", "verdict": "pass", "n_llm": 6, "wall_s": 12.0},
        {"pass": 1, "exercise": "proverb", "arm": "off", "task_id": "t1_pr_off", "verdict": "fail", "n_llm": 10, "wall_s": 20.0},
        {"pass": 1, "exercise": "proverb", "arm": "on", "task_id": "t1_pr_on", "verdict": "fail", "n_llm": 10, "wall_s": 20.0},
        # Pass 2 (comparison)
        {"pass": 2, "exercise": "transpose", "arm": "off", "task_id": "t2_tr_off", "verdict": "pass", "n_llm": 8, "wall_s": 16.0},
        {"pass": 2, "exercise": "transpose", "arm": "on", "task_id": "t2_tr_on", "verdict": "pass", "n_llm": 4, "wall_s": 9.0},
        {"pass": 2, "exercise": "proverb", "arm": "off", "task_id": "t2_pr_off", "verdict": "fail", "n_llm": 12, "wall_s": 24.0},
        {"pass": 2, "exercise": "proverb", "arm": "on", "task_id": "t2_pr_on", "verdict": "pass", "n_llm": 5, "wall_s": 11.0},
    ]
    decisions_by_arm = {
        "off": [
            {"task_id": "t1_tr_off", "action": "NEW", "admitted_by": "rehearsal", "entry_id": 1},
            {"task_id": "t2_tr_off", "action": "IGNORE", "admitted_by": "rehearsal", "entry_id": None},
        ],
        "on": [
            {"task_id": "t1_tr_on", "action": "NEW", "admitted_by": "outcome", "entry_id": 1},
            {"task_id": "t1_pr_on", "action": "NEW", "admitted_by": "outcome", "entry_id": 2},
            {"task_id": "t2_tr_on", "action": "NEW", "admitted_by": "outcome", "entry_id": 3},
            {"task_id": "t2_tr_on", "action": "RELATED", "admitted_by": "rehearsal", "entry_id": 4},
            {"task_id": "t2_pr_on", "action": "UPDATE", "admitted_by": "outcome", "entry_id": 5},
            {"task_id": "t2_pr_on", "action": "IGNORE", "admitted_by": "outcome", "entry_id": None},
        ],
    }
    recalls_by_task = {
        "t2_tr_on": [
            {
                "trigger": "start",
                "hits": [
                    {"doc_id": "long:1", "kind": "long", "origin_task_ids": ["t1_tr_on"]},  # same-exercise
                    {"doc_id": "long:2", "kind": "long", "origin_task_ids": ["t1_pr_on"]},  # cross-exercise
                    {"doc_id": "t2_tr_on:n1", "kind": "note", "origin_task_ids": ["t2_tr_on"]},  # task-local (ignored)
                ],
            }
        ],
        "t2_pr_on": [
            {
                "trigger": "start",
                "hits": [
                    {"doc_id": "long:2", "kind": "long", "origin_task_ids": ["t1_pr_on"]},  # same-exercise
                ],
            }
        ],
    }

    report = d11.build_report(runs, decisions_by_arm=decisions_by_arm, recalls_by_task=recalls_by_task)
    p2_on = report["by_arm_and_pass"]["on"]["2"]
    assert p2_on["pass_rate"] == 1.0
    assert p2_on["admitted_by"] == {"outcome": 2, "rehearsal": 1}
    assert p2_on["entries_written"] == {"total": 3, "mean_per_run": 1.5, "max_per_run": 2}
    assert p2_on["recall_hits"] == {"same_exercise": 2, "cross_exercise": 1, "total": 3, "task_local": 1}

    d2 = report["d2"]
    assert d2["n_pairs"] == 2
    assert d2["pass_rate_off"] == 0.5 and d2["pass_rate_on"] == 1.0
    assert d2["median_n_llm_off"] == 10.0 and d2["median_n_llm_on"] == 4.5
    assert d2["criteria"] == {
        "non_inferior_pass_rate": True,
        "fewer_calls": True,
        "less_wall_time": True,
        "efficiency_gain": True,
        "bounded_growth": True,
    }
    assert d2["admit_on_outcome_default_on"] is True
    assert d2["mcnemar"]["on_only"] == 1 and d2["mcnemar"]["off_only"] == 0
    assert d2["bootstrap_n_llm"]["mean_diff"] == -5.5
    assert d2["bootstrap_n_llm"]["ci95_mean"][1] < 0

    # Verify D2 fails when any single condition is violated
    # 1. Pass rate > 10 pp worse
    bad_pass = [
        {"pass": 2, "exercise": "transpose", "arm": "off", "verdict": "pass", "n_llm": 8, "wall_s": 16.0, "entries_written": 0},
        {"pass": 2, "exercise": "transpose", "arm": "on", "verdict": "fail", "n_llm": 4, "wall_s": 9.0, "entries_written": 1},
    ]
    assert d11.evaluate_d2(bad_pass)["admit_on_outcome_default_on"] is False

    # 2. Neither fewer calls nor less wall time
    no_gain = [
        {"pass": 2, "exercise": "transpose", "arm": "off", "verdict": "pass", "n_llm": 4, "wall_s": 9.0, "entries_written": 0},
        {"pass": 2, "exercise": "transpose", "arm": "on", "verdict": "pass", "n_llm": 4, "wall_s": 9.0, "entries_written": 1},
    ]
    assert d11.evaluate_d2(no_gain)["admit_on_outcome_default_on"] is False

    # 3. Unbounded growth (run writes > 8 entries, or mean > 3)
    too_many = [
        {"pass": 2, "exercise": f"ex{i}", "arm": arm, "verdict": "pass", "n_llm": 5 if arm == "on" else 8, "wall_s": 10.0,
         "entries_written": (9 if i == 0 else 1) if arm == "on" else 0}
        for i in range(4)
        for arm in ("off", "on")
    ]
    res_many = d11.evaluate_d2(too_many)
    assert res_many["criteria"]["bounded_growth"] is False
    assert res_many["admit_on_outcome_default_on"] is False


@pytest.mark.skipif(
    not (d11.VENDOR / ".venv/bin/melo").exists() or not paths.POLYGLOT_PYTHON.is_dir(),
    reason="localmelo vendor venv or polyglot benchmark data is not present",
)
def test_scripted_d11_smoke(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    trial = tmp_path / "trial"
    req_log = tmp_path / "stub-requests.jsonl"
    with stub_server(req_log) as url:
        env = {
            **os.environ,
            "OMLX_BASE_URL": url,
            "BENCH_MODEL": "scripted-d11",
            "BENCH_LOCALMELO_REPO": str(d11.VENDOR.resolve()),
            "BENCH_UPSTREAM_TOKEN_CMD": "",
            "HARNESS_SMOKE_DATA": str(paths.DATA_ROOT),
            "HOME": str(home),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        code = (
            f"import runpy, sys; sys.path.insert(0, {str(d11.ROOT)!r}); "
            "runpy.run_module('memory.d11.run', run_name='__main__')"
        )
        cmd = [
            str(d11.ROOT / ".venv" / "bin" / "python"),
            "-I",
            "-B",
            "-c",
            code,
            "--run-dir",
            str(trial),
            "--passes",
            "1",
            "--exercises",
            "transpose,proverb",
        ]
        proc = subprocess.run(cmd, cwd=d11.ROOT, env=env, capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr

    report = json.loads((trial / "report.json").read_text())
    assert len(report["runs"]) == 4
    assert (trial / "report.md").exists()

    # Both arms have their shared memory directories and exported files
    for arm in ("off", "on"):
        mem_dir = trial / f"mem-{arm}"
        assert (mem_dir / "history.db").exists()
        assert (mem_dir / "export.jsonl").exists()
        assert (mem_dir / "decisions.jsonl").exists()
        with sqlite3.connect(f"file:{mem_dir / 'history.db'}?mode=ro", uri=True) as conn:
            n_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert n_tasks == 2

    # Arm 'off' writes 0 outcome digests; arm 'on' writes outcome digests and recalls on the second exercise
    assert report["by_arm"]["off"]["entries_written"]["total"] == 0
    assert report["by_arm"]["on"]["entries_written"]["total"] >= 1
    assert report["by_arm"]["on"]["admitted_by"].get("outcome", 0) >= 1
    assert report["by_arm"]["on"]["active_store_entries"] >= 1
    assert report["by_arm"]["on"]["recall_hits"]["cross_exercise"] >= 1

    requests = d11.lines(req_log)
    assert any("Return at most" in str(r["messages"][-1].get("content", "")) for r in requests)
