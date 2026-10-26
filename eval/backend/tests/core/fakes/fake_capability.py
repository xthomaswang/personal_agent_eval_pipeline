"""Fake capability runner: `<python> -I -B -m modeleval.capability.runner --spec <spec.json>`.

Installed by conftest as an executable (the orchestrator's CAP_PYTHON). It pops its exit code from
$FAKE_CAP_SEQ (one code per line; 0 when empty), records every start (with the spec's resume flag and
server pid) in $FAKE_CAP_LOG, prints TASK-START/TASK-DONE lines, and on exit 0 writes results.json.
$FAKE_CAP_HOLD: a file it creates before exiting 75 (a hold request); $FAKE_CAP_SLEEP: seconds to run.
$FAKE_CAP_ROWS: a JSON file {set: [row, ...]}; on exit 0 the rows of the spec's sets are added to
<run_root>/state.json and written to results.json as the real runner does (cumulative `sets`,
`complete_sets`, `finished`), instead of the single default row.
Like the real runner, it removes <run_root>/yield.json at start and writes it on exit 75, with the
reason "hold" when $FAKE_CAP_HOLD is set and "battery" otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    assert argv[:5] == ["-I", "-B", "-m", "modeleval.capability.runner", "--spec"], argv
    spec = json.loads(Path(argv[5]).read_text())
    seq = Path(os.environ["FAKE_CAP_SEQ"])
    lines = seq.read_text().splitlines() if seq.exists() else []
    code = int(lines[0]) if lines else 0
    seq.write_text("".join(x + "\n" for x in lines[1:]))
    with open(os.environ["FAKE_CAP_LOG"], "a") as f:
        f.write(json.dumps({"code": code, "resume": spec["resume"], "server_pid": spec["server_pid"],
                            "sets": spec["sets"], "base_url": spec["base_url"], "pid": os.getpid(),
                            "epoch": spec["epoch"], "server_log": spec.get("server_log")}) + "\n")
    yield_file = Path(spec["run_root"]) / "yield.json"
    yield_file.unlink(missing_ok=True)
    stamp = time.strftime("%H:%M:%S")
    print(f"{stamp} TASK-START harness=claude-code task=transpose", flush=True)
    if os.environ.get("FAKE_CAP_SLEEP"):
        time.sleep(float(os.environ["FAKE_CAP_SLEEP"]))
    if code == 75 and os.environ.get("FAKE_CAP_HOLD"):
        Path(os.environ["FAKE_CAP_HOLD"]).touch()   # yield because a hold was requested
    if code == 75:
        yield_file.parent.mkdir(parents=True, exist_ok=True)
        reason = "hold" if os.environ.get("FAKE_CAP_HOLD") else "battery"
        yield_file.write_text(json.dumps({"reason": reason, "detail": "fake", "exit_code": 75}))
    if code == 0:
        out = Path(spec["results"])
        out.parent.mkdir(parents=True, exist_ok=True)
        rows_file = os.environ.get("FAKE_CAP_ROWS")
        if rows_file and Path(rows_file).exists():
            by_set = json.loads(Path(rows_file).read_text())
            state_p = Path(spec["run_root"]) / "state.json"
            state = json.loads(state_p.read_text()) if state_p.exists() else {"records": [], "meta": {}}
            state["records"] += [r for s in spec["sets"] for r in by_set.get(s, [])]
            sets = [s for s in ("gate", "ranking") if s in set(state["meta"].get("sets", [])) | set(spec["sets"])]
            state["meta"].update(sets=sets, complete_sets={**state["meta"].get("complete_sets", {}),
                                                           **{s: True for s in spec["sets"]}})
            state_p.parent.mkdir(parents=True, exist_ok=True)
            state_p.write_text(json.dumps(state))
            out.write_text(json.dumps({"epoch": spec["epoch"], "sets": sets, "finished": True,
                                       "complete_sets": state["meta"]["complete_sets"], "rows": state["records"]}))
        else:
            out.write_text(json.dumps({"results": [{"harness": "claude-code", "task": "transpose", "verdict": "pass"}]}))
        print(f"{stamp} TASK-DONE harness=claude-code task=transpose verdict=pass", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
