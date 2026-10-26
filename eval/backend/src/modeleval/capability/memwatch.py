"""The memory watchdog of the capability runner (DESIGN §6.1): a bound on what the agents' code may use.

Why: on 2026-10-26, 12:39-12:57, in claude_code/aider/variable-length-quantity (Nanbeige4.3-33B), scripts
the agent wrote looped and grew until macOS's compressor held ~60 GB more, memory pressure reached level 4
and macOS showed "Your system has run out of application memory". Nothing bounds what the agents run, and
macOS could then kill the model server or another user's session.

While a task runs, a thread polls every POLL_S seconds:

- **The tree.** The roots are the runner's children that lead their own session: the harness
  (bench.harness starts it with start_new_session=True) and, after it, smoke's checker (the agent's
  solution under pytest, started the same way). The tree is the roots, every descendant by parent pid,
  and every process still in a root's session (a command whose parent exited is reparented to launchd
  but keeps its session). The recorder proxy is the runner's child in the runner's own session, so it is
  never in the tree; the runner itself and pid 1 never are; the oMLX server is the orchestrator's
  child, not the runner's. Listing and parent links come from libproc (proc_listpids, proc_pidinfo).
- **The measure.** phys_footprint from libproc's proc_pid_rusage (RUSAGE_INFO_V2), the number Activity
  Monitor shows as Memory. It counts compressed and swapped-out pages; resident size does not, so a
  runaway that macOS compresses looks small by RSS.
- **The limits.** A process above PROC_LIMIT_GIB is killed (SIGKILL). While the tree is above
  TREE_LIMIT_GIB, its largest non-root members are killed until it is not. A root (the harness or the
  checker itself) is killed only when it is itself above the per-process limit. The agent sees its
  command killed, as by an OOM killer, and the task goes on; the verdict is the checker's, as always.
  Each kill is recorded (pid, command line cut to 120 characters with the home directory as `~`,
  footprint, time, which limit) and the runner puts the list into the row's flags as `memory_kills`.

Stdlib only (ctypes), like the rest of the runner's imports; macOS only. If libproc cannot be loaded, the
watchdog records why and the task runs unwatched.
"""

from __future__ import annotations

import collections
import ctypes
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

PROC_LIMIT_GIB = 8.0
TREE_LIMIT_GIB = 16.0
# Overrides, in GiB despite the names; they pass the runner's environment allowlist
PROC_LIMIT_ENV = "MODELEVAL_AGENT_PROC_LIMIT_GB"
TREE_LIMIT_ENV = "MODELEVAL_AGENT_TREE_LIMIT_GB"
POLL_S = 1.0
CMDLINE_CHARS = 120
GIB = 1 << 30

# <sys/proc_info.h>, <sys/resource.h>, <sys/sysctl.h>
_PROC_UID_ONLY = 4
_PROC_PIDTBSDINFO = 3
_RUSAGE_INFO_V2 = 2
_SZOMB = 5
_CTL_KERN, _KERN_PROCARGS2 = 1, 49


def limits(environ: Mapping[str, str] = os.environ) -> tuple[int, int]:
    """(per-process, whole-tree) limits in bytes: the defaults, or the GiB values in the environment.
    ValueError for a value that is not a positive number (the runner refuses to start)."""
    out = []
    for name, default in ((PROC_LIMIT_ENV, PROC_LIMIT_GIB), (TREE_LIMIT_ENV, TREE_LIMIT_GIB)):
        raw = environ.get(name, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError:
            value = math.nan
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name}={raw!r}: the memory limit must be a positive number of GiB")
        out.append(int(value * GIB))
    return out[0], out[1]


def short_cmdline(text: str, home: Path | None = None) -> str:
    """One line, the home directory as `~` (before cutting, so no part of it survives), 120 characters."""
    home = Path.home() if home is None else home
    text = " ".join(text.split())
    for prefix in sorted({str(home), os.path.realpath(home)}, key=len, reverse=True):
        text = text.replace(prefix, "~")
    return text[:CMDLINE_CHARS]


# --- libproc through ctypes ---


class _BsdInfo(ctypes.Structure):
    """struct proc_bsdinfo (PROC_PIDTBSDINFO), 136 bytes."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32), ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32), ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32), ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32), ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32), ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32), ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32), ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]  # fmt: skip


_RUSAGE_V2_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
    "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
    "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
)  # fmt: skip


class _RusageInfoV2(ctypes.Structure):
    """struct rusage_info_v2, 160 bytes."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _RUSAGE_V2_FIELDS]


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    start: tuple[int, int]  # start time (s, µs): with the pid, the identity of a process
    comm: str


class LibProc:
    """The few libproc and sysctl calls the watchdog needs."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        lib.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        lib.proc_listpids.restype = ctypes.c_int
        lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        lib.proc_pidinfo.restype = ctypes.c_int
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        lib.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]  # fmt: skip
        lib.sysctl.restype = ctypes.c_int
        self.lib = lib

    def pids(self, uid: int) -> list[int]:
        """Every pid of this user."""
        need = self.lib.proc_listpids(_PROC_UID_ONLY, uid, None, 0)
        if need <= 0:
            raise OSError(ctypes.get_errno(), "proc_listpids failed")
        count = need // ctypes.sizeof(ctypes.c_int) + 256
        while True:
            buf = (ctypes.c_int * count)()
            got = self.lib.proc_listpids(_PROC_UID_ONLY, uid, buf, ctypes.sizeof(buf))
            if got < 0:
                raise OSError(ctypes.get_errno(), "proc_listpids failed")
            n = got // ctypes.sizeof(ctypes.c_int)
            if n < count:
                return [pid for pid in buf[:n] if pid > 0]
            count *= 2

    def info(self, pid: int) -> ProcInfo | None:
        """Parent and start time; None for a process that is gone, a zombie or not readable."""
        bsd = _BsdInfo()
        if self.lib.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(bsd), ctypes.sizeof(bsd)) != ctypes.sizeof(bsd):
            return None
        if bsd.pbi_status == _SZOMB:
            return None
        start = (int(bsd.pbi_start_tvsec), int(bsd.pbi_start_tvusec))
        return ProcInfo(pid, int(bsd.pbi_ppid), start, bsd.pbi_comm.decode(errors="replace"))

    def footprint(self, pid: int) -> int | None:
        """phys_footprint in bytes (compressed and swapped pages included), None when not readable."""
        usage = _RusageInfoV2()
        if self.lib.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(usage)) != 0:
            return None
        return int(usage.ri_phys_footprint)

    def cmdline(self, pid: int) -> str | None:
        """argv joined by spaces (KERN_PROCARGS2), None when not readable."""
        mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
        size = ctypes.c_size_t(0)
        if self.lib.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if self.lib.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
            return None
        raw = buf.raw[: size.value]
        argc = int.from_bytes(raw[:4], sys.byteorder)
        _exe, _, rest = raw[4:].partition(b"\0")
        args = rest.lstrip(b"\0").split(b"\0")[:argc]
        return " ".join(a.decode(errors="replace") for a in args) or None


def _sid(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except OSError:
        return None


@dataclass(frozen=True)
class _Member:
    pid: int
    start: tuple[int, int]
    footprint: int
    root: bool


# --- the watchdog ---


class Watchdog:
    """Around one task: `with Watchdog(os.getpid(), *limits()) as w: run_one(...)`; then `w.kills`."""

    def __init__(self, parent: int, proc_limit: int, tree_limit: int, poll_s: float = POLL_S) -> None:
        self.parent, self.proc_limit, self.tree_limit, self.poll_s = parent, proc_limit, tree_limit, poll_s
        self.kills: list[dict[str, Any]] = []
        self.error: str | None = None
        self._uid = os.getuid()
        self._own_sid = _sid(parent)
        self._sessions: dict[int, tuple[int, int]] = {}  # root pid (= its session id) -> its start time
        self._killed: set[tuple[int, tuple[int, int]]] = set()
        self._lib: LibProc | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="memory-watchdog", daemon=True)

    def __enter__(self) -> Self:
        try:
            self._lib = LibProc()
        except (OSError, AttributeError) as exc:  # not macOS: the task runs unwatched, and the row says so
            self._fail(exc)
            return self
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.poll_s + 10)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def row_flags(self) -> dict[str, Any] | None:
        """What the row's `flags` start with: None when nothing happened (the orchestrator then fills the
        disturbance flags as before), else the kills (and a watchdog failure)."""
        if not self.kills and self.error is None:
            return None
        flags: dict[str, Any] = {"memory_kills": list(self.kills)}
        if self.error is not None:
            flags["memory_watchdog_error"] = self.error
        return flags

    def _fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = f"{type(exc).__name__}: {exc}"[:200]
            _say(f"memory watchdog failed, the task runs unwatched: {self.error}")

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.poll()
            except Exception as exc:  # noqa: BLE001 - one bad poll must not end the watch
                self._fail(exc)

    # -- one poll

    def tree(self, table: Mapping[int, ProcInfo]) -> tuple[set[int], set[int]]:
        """(roots, every process of the watched tree) in a process table of this user."""
        roots = {pid for pid, p in table.items() if p.ppid == self.parent and pid != self.parent and _sid(pid) == pid}
        for pid in roots:
            self._sessions[pid] = table[pid].start
        for sid, start in list(self._sessions.items()):
            if sid in table and table[sid].start != start:  # that session is over; its id names another process
                del self._sessions[sid]
        seeds = set(roots)
        if self._sessions:
            seeds |= {pid for pid in table if _sid(pid) in self._sessions}
        children: dict[int, list[int]] = collections.defaultdict(list)
        for pid, p in table.items():
            children[p.ppid].append(pid)
        found: set[int] = set()
        stack = list(seeds)
        while stack:
            pid = stack.pop()
            if pid not in found:
                found.add(pid)
                stack.extend(children.get(pid, ()))
        never = {self.parent, os.getpid(), 0, 1}
        return roots, {pid for pid in found if pid not in never and _sid(pid) != self._own_sid}

    def poll(self) -> None:
        lib = self._lib
        if lib is None:
            return
        table = {}
        for pid in lib.pids(self._uid):
            info = lib.info(pid)
            if info is not None:
                table[pid] = info
        roots, tree = self.tree(table)
        members = []
        for pid in tree:
            start = table[pid].start
            if (pid, start) in self._killed:  # dying: its memory is on its way back
                continue
            footprint = lib.footprint(pid)
            if footprint is not None:
                members.append(_Member(pid, start, footprint, pid in roots))
        for victim, limit in self.victims(members):
            self._kill(lib, victim, limit)

    def victims(self, members: list[_Member]) -> list[tuple[_Member, str]]:
        """Every process above the per-process limit; then, while the rest is above the tree limit, its
        largest members other than the roots."""
        largest_first = sorted(members, key=lambda m: m.footprint, reverse=True)
        out = [(m, "process") for m in largest_first if m.footprint > self.proc_limit]
        rest = [m for m in largest_first if m.footprint <= self.proc_limit]
        total = sum(m.footprint for m in rest)
        for m in rest:
            if total <= self.tree_limit:
                break
            if not m.root:
                out.append((m, "tree"))
                total -= m.footprint
        return out

    def _kill(self, lib: LibProc, victim: _Member, limit: str) -> None:
        info = lib.info(victim.pid)
        if info is None or info.start != victim.start:  # gone, or the pid names another process now
            return
        cmdline = lib.cmdline(victim.pid) or info.comm
        try:
            os.kill(victim.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        self._killed.add((victim.pid, victim.start))
        entry = {
            "pid": victim.pid,
            "cmdline": short_cmdline(cmdline),
            "footprint_gib": round(victim.footprint / GIB, 2),
            "t": round(time.time(), 3),
            "limit": limit,
        }
        self.kills.append(entry)
        bound = self.proc_limit if limit == "process" else self.tree_limit
        _say(f"memory watchdog: killed pid {victim.pid} ({entry['footprint_gib']} GiB, {limit} limit "
             f"{bound / GIB:g} GiB): {entry['cmdline']}")  # fmt: skip


def _say(text: str) -> None:
    try:
        print(f"capability runner: {text}", file=sys.stderr, flush=True)
    except (OSError, ValueError):  # a dead stderr must not end the watch
        pass
