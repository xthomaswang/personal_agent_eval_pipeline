"""Fake capability runner for the integration tests, installed as the orchestrator's CAP_PYTHON.

Unlike tests/core/fakes/fake_capability.py it checks what it was given with the real runner's own
code: `runner.spec_problems(spec)` on the spec.json run.py wrote and `runner.forbidden_names` on the
environment `allowed_env` built. It then writes state.json and results.json the way the runner does
(two records, flags null, results through export.write_results) and exits. Its knobs are BENCH_*
variables because nothing else passes the runner's environment allowlist.

  BENCH_FAKE_RUNNER_LOG      JSON lines: what it saw (spec problems, forbidden names, PATH, argv)
  BENCH_FAKE_RUNNER_EXIT     exit code (default 0); with 75 it writes yield.json with BENCH_FAKE_RUNNER_REASON
  BENCH_FAKE_RUNNER_DISTURB  a file it creates during its scored task (the fake monitor marks those rows)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from modeleval.capability import export, runner


def task(name: str, scored: bool, disturb: str | None = None) -> dict:
    t0 = time.time()
    if disturb:
        Path(disturb).touch()
    time.sleep(0.6)
    if disturb:
        Path(disturb).unlink()
    return {"set": "gate", "harness": "claude_code", "task": name, "kind": "aider", "scored": scored,
            "attempt": 1, "run_dir": f"claude_code/{name}", "launch": 1, "verdict": "pass", "valid": True,
            "solved": True, "t_start": round(t0, 3), "t_end": round(time.time(), 3), "flags": None,
            "anomalies": []}


def main() -> int:
    argv = sys.argv[1:]
    assert argv[:5] == ["-I", "-B", "-m", "modeleval.capability.runner", "--spec"], argv
    spec = json.loads(Path(argv[5]).read_text())
    with open(os.environ["BENCH_FAKE_RUNNER_LOG"], "a") as f:
        f.write(json.dumps({"spec_problems": runner.spec_problems(spec),
                            "forbidden": runner.forbidden_names(os.environ), "path": os.environ.get("PATH", ""),
                            "pydwb": os.environ.get("PYTHONDONTWRITEBYTECODE"),
                            "git_locks": os.environ.get("GIT_OPTIONAL_LOCKS"), "spec": spec}) + "\n")
    run_root = Path(spec["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "yield.json").unlink(missing_ok=True)
    records = [task("hello-warmup", False)]
    time.sleep(1.5)                      # a gap, so the warm-up's window never sees the disturbance
    records.append(task("transpose", True, os.environ.get("BENCH_FAKE_RUNNER_DISTURB")))
    (run_root / "state.json").write_text(json.dumps({"records": records, "meta": {"hygiene": []}}, indent=1) + "\n")
    export.write_results(Path(spec["results"]), {
        "schema_version": 1, "model": spec["model"]["key"], "epoch": spec["epoch"], "sets": spec["sets"],
        "rows": [export.clean_row(r) for r in records if r["scored"]], "superseded": [],
        "warmups": [export.clean_row(r) for r in records if not r["scored"]], "finished": True})
    code = int(os.environ.get("BENCH_FAKE_RUNNER_EXIT", "0"))
    if code == 75:
        (run_root / "yield.json").write_text(json.dumps({"reason": os.environ.get("BENCH_FAKE_RUNNER_REASON", "drift"),
                                                         "detail": "fake yield", "exit_code": 75}))
    return code


if __name__ == "__main__":
    sys.exit(main())
