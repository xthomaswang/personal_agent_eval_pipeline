"""Machine-readable progress lines for the driving session (docs/CONTRACTS.md).

`emit` never raises for a dead stdout. The driving session reads these lines through a pipe or a
tty; when that reader dies (a `| tee` killed by the same Ctrl-C, a closed terminal), a bare print
raises BrokenPipeError or EIO. Raised inside the orchestrator's cleanup, that skipped the server
stop and the lock release and left the GPU locked for everyone. The first failed write points fd 1
at /dev/null, so later lines, the interpreter's exit-time flush and every child started afterwards
(a relaunched runner, the lock script) no longer write to the dead descriptor.

Stdlib only: the capability runner imports this module under its own Python 3.13.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TAGS = frozenset({
    "LOCK-NOTICE", "UNIT-WAIT", "UNIT-START", "UNIT-DONE", "QUIET-WINDOW-END", "TASK-START", "TASK-DONE",
    "BATTERY-PAUSE", "BATTERY-READY", "RESUMED", "HOLD", "HOLD-RELEASED", "GATE-DECISION", "RUN-DONE", "ERROR",
})

LOG_PATH: Path | None = None   # set by the orchestrator: every line is also appended here (runs/, gitignored)
_stdout_dead = False


def _fmt(v: object) -> str:
    s = str(v)
    return f'"{s}"' if (" " in s or not s) else s


def _divert_stdout() -> None:
    """fd 1 is gone: from now on it is /dev/null, and emit stops writing to it."""
    global _stdout_dead
    _stdout_dead = True
    try:
        fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(fd, 1)
        finally:
            os.close(fd)
    except OSError:
        pass


def _append_log(line: str) -> None:
    if LOG_PATH is None:
        return
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def emit(tag: str, **kv: object) -> None:
    """One progress line on stdout (and in LOG_PATH). Raises only for an unknown tag."""
    if tag not in TAGS:
        raise ValueError(f"unknown progress tag {tag!r}")
    line = " ".join([time.strftime("%H:%M:%S"), tag] + [f"{k}={_fmt(v)}" for k, v in kv.items()])
    _append_log(line)
    if _stdout_dead:
        return
    try:
        print(line, file=sys.stdout, flush=True)
    except (OSError, ValueError):   # BrokenPipeError, EIO on a revoked tty; ValueError: stdout closed
        _divert_stdout()
