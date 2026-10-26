#!/usr/bin/env python3
"""Fake scripts/quietlib.py with the interface docs/CONTRACTS.md and quiet.py expect.

  check    pops the first line of $FAKE_QUIET_CHECKS (quiet | busy | hard); quiet when the file is
           empty or unset. Logs each call's argv to $FAKE_QUIET_LOG.
  monitor  writes the v4 header and one row per --interval; a row is DISTURBED while the file
           $FAKE_QUIET_DISTURB exists. Battery comes from $FAKE_QUIET_BATTERY (default 95).
  judge_window(log, t_start, t_end, margin)  clean iff no DISTURBED row in the window.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

MONITOR_COLUMNS = ("iso", "epoch", "state", "load1", "foreign_cpu", "gpu_util", "gpu_mem_gb", "battery",
                   "thermal_pressure", "pressure", "swapouts_delta", "foreign_gpu_ms", "foreign_rss_delta_mb",
                   "reasons")


def _note(argv: list[str]) -> None:
    log = os.environ.get("FAKE_QUIET_LOG")
    if log:
        with open(log, "a") as f:
            f.write(" ".join(argv) + "\n")


def cmd_check(a: argparse.Namespace) -> int:
    path = os.environ.get("FAKE_QUIET_CHECKS")
    state = "quiet"
    if path and Path(path).exists():
        lines = Path(path).read_text().splitlines()
        if lines:
            state, rest = lines[0].strip(), lines[1:]
            Path(path).write_text("".join(x + "\n" for x in rest))
    print("quiet_check: fake")
    if state == "quiet":
        print("verdict: QUIET")
        return 0
    if state == "hard":
        print("settle: not attempted (hard failure: lock not held by us)")
    else:
        print(f"settle: still not quiet after 0 s (limit {a.settle:.0f} s): busy: fakeproc(4242) 90%")
    print("verdict: NOT QUIET")
    return 1


def cmd_monitor(a: argparse.Namespace) -> int:
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    disturb = os.environ.get("FAKE_QUIET_DISTURB", "")
    with log.open("a") as f:
        f.write(f"# quietlib monitor v4 interval={a.interval} allow={','.join(map(str, a.allow_pid))}\n")
        f.write("# " + "\t".join(MONITOR_COLUMNS) + "\n")
        f.flush()
        while True:
            t = time.time()
            bad = bool(disturb) and Path(disturb).exists()
            row = {"iso": dt.datetime.fromtimestamp(t).isoformat(timespec="seconds"), "epoch": f"{t:.2f}",
                   "state": "DISTURBED" if bad else "QUIET", "load1": "1.0", "foreign_cpu": "200" if bad else "5",
                   "gpu_util": "0", "gpu_mem_gb": "1.0", "battery": os.environ.get("FAKE_QUIET_BATTERY", "95"),
                   "thermal_pressure": "0", "pressure": "1", "swapouts_delta": "0", "foreign_gpu_ms": "0",
                   "foreign_rss_delta_mb": "0", "reasons": "busy: fakeproc(4242) 90%" if bad else ""}
            f.write("\t".join(row[c] for c in MONITOR_COLUMNS) + "\n")
            f.flush()
            time.sleep(a.interval)


def judge_window(log, t_start: float, t_end: float, margin: float = 5.0) -> dict:
    rows = []
    cols = None
    for line in Path(log).read_text().splitlines():
        if line.startswith("# iso"):
            cols = line[2:].split("\t")
            continue
        if line.startswith("#") or not cols:
            continue
        r = dict(zip(cols, line.split("\t")))
        if t_start - margin <= float(r["epoch"]) <= t_end + margin:
            rows.append(r)
    bad = [r for r in rows if r["state"] == "DISTURBED"]
    return {"clean": not bad, "reasons": sorted({r["reasons"] for r in bad}),
            "peak_foreign_cpu": max((float(r["foreign_cpu"]) for r in rows), default=0.0),
            "samples": len(rows), "disturbed_samples": len(bad)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _note(argv)
    p = argparse.ArgumentParser(prog="quietlib")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--allow-pid", type=int, action="append", default=[])
    c.add_argument("--settle", type=float, default=0.0)
    c.add_argument("--max-gpu-gb", type=float)
    c.add_argument("--min-battery", type=float)
    c.add_argument("--allow", action="append", default=[])
    m = sub.add_parser("monitor")
    m.add_argument("--log", required=True)
    m.add_argument("--interval", type=float, default=5.0)
    m.add_argument("--allow-pid", type=int, action="append", default=[])
    a = p.parse_args(argv)
    return {"check": cmd_check, "monitor": cmd_monitor}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
