"""The memory watchdog (capability/memwatch.py) on real processes with small limits.

A fake harness is a Python process in its own session (as bench.harness starts the real one) whose
children allocate memory page by page. Limits are a few hundred MiB and every allocation is at most
0.4 GiB, so the tests stay fast and harmless; the watchdog polls every 0.1 s here instead of 1 s.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from modeleval.capability import memwatch
from modeleval.capability.memwatch import GIB, Watchdog, _Member

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="libproc is macOS only")

POLL_S = 0.1
# argv: GiB to allocate, seconds to hold it. Prints "ready" once every page is touched.
HOG = r"""
import os, sys, time
size, hold = int(float(sys.argv[1]) * 2**30), float(sys.argv[2])
page = os.sysconf("SC_PAGESIZE")
block = bytearray(size)
for i in range(0, size, page):
    block[i] = 1
print("ready", flush=True)
time.sleep(hold)
"""
# The fake harness: argv: HOG, then (GiB, hold) pairs. Starts each child once the previous one is
# ready (or dead), waits for all, and prints their pids and exit codes.
ROOT = r"""
import json, subprocess, sys
hog, pairs = sys.argv[1], sys.argv[2:]
procs = []
for size, hold in zip(pairs[::2], pairs[1::2]):
    p = subprocess.Popen([sys.executable, "-I", "-c", hog, size, hold], stdout=subprocess.PIPE, text=True)
    p.stdout.readline()
    procs.append(p)
print(json.dumps([{"pid": p.pid, "returncode": p.wait()} for p in procs]), flush=True)
"""


@pytest.fixture
def spawned() -> Iterator[list[subprocess.Popen[str]]]:
    procs: list[subprocess.Popen[str]] = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def harness(spawned: list[subprocess.Popen[str]], *children: tuple[float, float]) -> subprocess.Popen[str]:
    """A fake harness in its own session, with children allocating (GiB, hold seconds)."""
    args = [str(x) for pair in children for x in pair]
    proc = subprocess.Popen([sys.executable, "-I", "-c", ROOT, HOG, *args], stdout=subprocess.PIPE, text=True,
                            start_new_session=True)  # fmt: skip
    spawned.append(proc)
    return proc


def outcome(proc: subprocess.Popen[str]) -> tuple[int, list[dict]]:
    out, _ = proc.communicate(timeout=60)
    return proc.returncode, json.loads(out)


def watchdog(proc_gib: float, tree_gib: float) -> Watchdog:
    return Watchdog(os.getpid(), int(proc_gib * GIB), int(tree_gib * GIB), poll_s=POLL_S)


def test_a_runaway_child_is_killed_and_recorded(spawned) -> None:
    with watchdog(0.2, 1.0) as w:
        root = harness(spawned, (0.4, 30))
        code, children = outcome(root)
    assert code == 0  # the harness itself goes on, as the agent does after an OOM kill
    (child,) = children
    assert child["returncode"] == -9
    (kill,) = w.kills
    assert kill["pid"] == child["pid"] and kill["limit"] == "process"
    assert 0.2 <= kill["footprint_gib"] <= 0.5
    assert abs(kill["t"] - time.time()) < 60
    assert " -I -c import os, sys, time size, hold" in kill["cmdline"]  # one line, cut at 120 characters
    assert len(kill["cmdline"]) <= memwatch.CMDLINE_CHARS
    assert str(Path.home()) not in kill["cmdline"]
    if sys.executable.startswith(str(Path.home()) + "/"):
        assert kill["cmdline"].startswith("~/")
    assert w.row_flags() == {"memory_kills": w.kills} and w.error is None


def test_above_the_tree_limit_the_largest_descendant_goes(spawned) -> None:
    with watchdog(1.0, 0.32) as w:
        root = harness(spawned, (0.25, 30), (0.15, 2))
        code, (big, small) = outcome(root)
    assert code == 0 and big["returncode"] == -9 and small["returncode"] == 0
    assert [(k["pid"], k["limit"]) for k in w.kills] == [(big["pid"], "tree")]


def test_a_well_behaved_task_gets_no_kill(spawned) -> None:
    with watchdog(0.2, 0.3) as w:
        root = harness(spawned, (0.02, 1))
        code, (child,) = outcome(root)
    assert code == 0 and child["returncode"] == 0
    assert w.kills == [] and w.row_flags() is None and w.error is None


def test_nothing_outside_the_tree_is_touched(spawned) -> None:
    """A child of the watched process in its session (like the recorder proxy) is never in the tree,
    however large; the harness's own runaway child still is."""
    proxy_like = subprocess.Popen([sys.executable, "-I", "-c", HOG, "0.2", "30"], stdout=subprocess.PIPE, text=True)
    spawned.append(proxy_like)
    assert proxy_like.stdout.readline().strip() == "ready"
    with watchdog(0.1, 0.15) as w:
        root = harness(spawned, (0.2, 30))
        code, (child,) = outcome(root)
        time.sleep(3 * POLL_S)
    assert code == 0 and child["returncode"] == -9
    assert [k["pid"] for k in w.kills] == [child["pid"]]
    assert proxy_like.poll() is None


def test_the_watchdog_stops_with_the_task(spawned) -> None:
    w = watchdog(0.2, 0.3)
    with w:
        assert w.running
        root = harness(spawned, (0.01, 0.2))
        outcome(root)
    t0 = time.monotonic()
    assert not w.running and time.monotonic() - t0 < 1
    # A runaway after the task is not this watchdog's: it has stopped
    late = harness(spawned, (0.3, 30))
    time.sleep(5 * POLL_S)
    assert w.kills == []
    late.kill()
    late.wait()


def test_victims_policy() -> None:
    w = Watchdog(os.getpid(), 8 * GIB, 16 * GIB)

    def m(pid: int, gib: float, root: bool = False) -> _Member:
        return _Member(pid, (0, pid), int(gib * GIB), root)

    def pick(*members: _Member) -> list[tuple[int, str]]:
        return [(v.pid, limit) for v, limit in w.victims(list(members))]

    assert pick(m(1, 0.5, True), m(2, 7.9), m(3, 7.0)) == []
    assert pick(m(1, 0.5, True), m(2, 9.0), m(3, 12.0)) == [(3, "process"), (2, "process")]
    # Above the tree limit: the largest non-root members until it is not, never the root
    assert pick(m(1, 7.5, True), m(2, 6.0), m(3, 5.0), m(4, 1.0)) == [(2, "tree")]
    assert pick(m(1, 7.9, True), m(2, 7.8, True), m(3, 1.0)) == [(3, "tree")]
    # A root is killed when it is itself above the per-process limit
    assert pick(m(1, 9.0, True), m(2, 1.0)) == [(1, "process")]


def test_limits_from_the_environment() -> None:
    assert memwatch.limits({}) == (8 * GIB, 16 * GIB)
    env = {memwatch.PROC_LIMIT_ENV: "0.2", memwatch.TREE_LIMIT_ENV: " 0.5 "}
    assert memwatch.limits(env) == (int(0.2 * GIB), int(0.5 * GIB))
    for bad in ("0", "-1", "nan", "inf", "8GB"):
        with pytest.raises(ValueError, match=memwatch.PROC_LIMIT_ENV):
            memwatch.limits({memwatch.PROC_LIMIT_ENV: bad})


def test_short_cmdline() -> None:
    home = Path("/home/someone")
    text = "/home/someone/env/bin/python -c 'import x\nwhile True:\n    pass' /home/someone"
    assert memwatch.short_cmdline(text, home) == "~/env/bin/python -c 'import x while True: pass' ~"
    long = memwatch.short_cmdline("/home/someone/" + "a" * 300, home)
    assert len(long) == memwatch.CMDLINE_CHARS and long.startswith("~/")
