#!/usr/bin/env python3
"""Quiet-machine checks and the launch monitor for timed units (DESIGN.md section 5.4).

Stdlib only, Python 3.9+: quiet_check.sh runs it with /usr/bin/python3 -I -B (3.9 on this Mac),
and modeleval.quiet imports it from the project venv (3.12). A modified copy of the Qwen3.8-27B
campaign's quietlib (that repo's METHODOLOGY section 3): the sampler and the CPU, lock, port,
power and battery thresholds are unchanged; the monitor samples every 5 s and logs four more
columns; units are judged per window from the logged columns instead of per raw file.
Subcommands:

  check   [--allow-pid PID]... [--settle S] [--max-gpu-gb G] [--min-battery PCT] [--allow NAME[,NAME]]
          One-shot pre-flight, run while the server under test is idle. Prints a provenance
          block; exit 1 when the machine is not quiet. With --settle, soft failures (load, GPU
          busy, busy processes, build tools, thermal) are re-checked every 5 s for up to S
          seconds. Hard failures (lock, a foreign server port, power) fail at once.
          --max-gpu-gb (used before a server starts) also fails while more than G GB of GPU
          memory is in use, i.e. while another model is resident. --min-battery fails while the
          battery is below PCT %.
  monitor --log FILE [--interval S] [--allow-pid PID]... [--until-pid PID] [--count N] [--allow NAME[,NAME]]
          Samples for the whole launch (default every 5 s). Every sample is a TSV row in FILE
          with the columns of MONITOR_COLUMNS; the header line is "# " + those names, after one
          "# quietlib monitor v4 ..." line with the interval and thresholds. stdout gets one line
          per state change. The state column is informational (DISTURBED, SPIKE, QUIET, from
          row_flags); judge_window re-judges from the logged columns. --count stops after N
          samples (tests and short traces).
  battery --min PCT [--wait S]
          Waits up to S seconds (default 0) for the battery to reach PCT %, printing the level
          every 5 min. Exit 1 if it does not. Machines without a battery always pass.

Importable (stdlib only, no side effects at import):

  judge_window(tsv, t_start, t_end)  a timed unit's verdict over [t_start - 5 s, t_end + 5 s]
  window_flags(tsv, t_start, t_end)  a capability task's flags over the same kind of window
  read_monitor(tsv)                  the parsed rows of a monitor TSV

The quiet rule (DESIGN section 5.4). A sample inside the window is disturbed by:
  - a lock event (our lock missing), a foreign server port (8500 or 8600-8699 listening; 8703 is
    the server under test), power (not AC + High Power), the battery below QUIET_MIN_BATTERY_RUN,
    or a build/convert/download tool (BUILD_NAMES, BUILD_ARGS): at once;
  - the same foreign process above QUIET_MAX_PROC_CPU in this and the previous sample;
  - foreign CPU in total above QUIET_MAX_FOREIGN_CPU in this and the previous sample;
  - foreign GPU time above GPU_FLOOR_MS in this and the previous sample;
  - memory pressure >= 2 or a swap-out burst (SWAP_BURST_MB in one sample), but only when the
    foreign RSS total grew by more than FOREIGN_RSS_GROWTH_MB in the window up to that sample:
    pressure the server causes itself is a result (DESIGN section 5.2), not a disturbance.
pmset thermal warnings and the thermal pressure level are logged, never judged by the monitor.
The previous sample may lie just before the window; a sample after it never counts.

"Foreign" means any process outside the allowed PID trees (--allow-pid: the server, its engine
workers, the orchestrator, the harness), except WindowServer and kernel_task, which are always
resident and do driver work for the server.

--allow NAME (the old M8 interference study) accepts one system-wide side effect of a load our
own script started; allowed reasons start with "allowed " and never count:
  load      the 1-min load average is not judged (check; the monitor never judges it)
  gpu       GPU utilization is not judged (check; the monitor never judges it)
  thermal   a pmset thermal warning is recorded as "allowed thermal: ..." instead of failing
  lowpower  Low Power Mode on AC power is recorded as "allowed power: ..."

Processes inside an app bundle whose identifier is not com.apple.* are shown as `<app>`, so
the user's own apps never reach committed files (scripts/redact.py applies the same rule). A
build/convert/download tool is reported by its kind only (`tools: git clone (pid 123)`): another
session's arguments can carry a token or a private repo name, so the full command line appears
only in the check report, which goes to gitignored logs.

Thresholds (environment; a protocol run uses the defaults): QUIET_MAX_LOAD (1-min load, check
only, default 2.5), QUIET_MAX_PROC_CPU (any single foreign process, default 25 %),
QUIET_MAX_FOREIGN_CPU (sum over foreign processes, default 150 %), QUIET_MAX_GPU (GPU "Device
Utilization %" while our server is idle, check only, default 10 %), QUIET_MIN_BATTERY_RUN
(default 20 %: the 96 W adapter cannot supply this machine under sustained load, and below ~7 %
the system caps power), QUIET_GPU_FLOOR_MS, QUIET_SWAP_BURST_MB, QUIET_FOREIGN_RSS_GROWTH_MB
(see their comments). GPU_LOCK_PATH overrides the lock path (tests).
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

HOME = str(Path.home())
LOCK = os.environ.get("GPU_LOCK_PATH", f"{HOME}/Developer/mlsys/experiments/.gpu-lock")
LOCK_HOLDER = "local_ai-benchmark session"
PEER_PORT = 8500                      # the agent-benchmark session's oMLX
FOREIGN_PORTS = "8500,8600-8699"      # lsof syntax; the agent benchmark's ports. 8703 is ours
MAX_LOAD = float(os.environ.get("QUIET_MAX_LOAD", "2.5"))
MAX_PROC_CPU = float(os.environ.get("QUIET_MAX_PROC_CPU", "25"))
MAX_FOREIGN_CPU = float(os.environ.get("QUIET_MAX_FOREIGN_CPU", "150"))
MAX_GPU = float(os.environ.get("QUIET_MAX_GPU", "10"))
MIN_BATTERY_RUN = float(os.environ.get("QUIET_MIN_BATTERY_RUN", "20"))
# Foreign GPU time per 5 s sample (the sum over foreign processes of their accumulatedGPUTime
# change), judged in two consecutive samples. Chosen from an idle trace on 2026-10-26 (36 samples
# x 5 s, no model server, two Claude sessions working): the foreign sum was 3-11 ms per sample
# (Terminal 3-6 ms steadily, a WebKit page ~3 ms), with single-sample spikes to 41-49 ms
# (ControlCenter about once a minute); never above 11 ms in two consecutive samples. 100 ms per
# 5 s (2 % of wall time) is about twice the largest idle spike and ~9x the largest persistent
# idle level, and far below any process doing real GPU work. The value assumes the 5 s interval.
GPU_FLOOR_MS = float(os.environ.get("QUIET_GPU_FLOOR_MS", "100"))
# Swap-out burst: MB written to swap in one sample. Uncalibrated: the idle trace above swapped out
# nothing (0 pages in all 36 samples, with 2 GB of swap in use and pressure level 1), so any burst
# is a sign of tight memory; 64 MB keeps single pages from counting.
SWAP_BURST_MB = float(os.environ.get("QUIET_SWAP_BURST_MB", "64"))
# Growth of the foreign RSS total over the judged window (up to the flagged sample) that makes
# pressure or swap count as foreign-caused. Uncalibrated against real pressure: in the idle trace
# the foreign total moved by up to +345/-698 MB per sample and grew by ~714 MB over 4 samples
# (Claude sessions), so the line sits above that.
FOREIGN_RSS_GROWTH_MB = float(os.environ.get("QUIET_FOREIGN_RSS_GROWTH_MB", "1024"))
MONITOR_INTERVAL = 5.0
PAD_S = 5.0                           # DESIGN 5.4: [t_start - 5 s, t_end + 5 s]
MAX_REASONS = 8                       # disturbed samples listed per verdict
TPL_NAMES = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}   # OSThermalNotification.h
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}   # kern.memorystatus_vm_pressure_level
IGNORED = {"WindowServer", "kernel_task"}  # always resident; kernel_task also does the server's driver work
BUILD_NAMES = {"clang", "clang++", "cc1", "cc1plus", "ld", "ld64", "ninja", "make", "gmake", "cmake",
               "swift-frontend", "swiftc", "swift-build", "rustc", "cargo", "xcodebuild"}
BUILD_ARGS = re.compile(
    r"mlc_llm (convert_weight|compile)|coreai\.llm\.export|\boq_convert\b|\bmlx_(lm|vlm)[. ]convert\b|"
    r"\bhf (download|upload)\b|huggingface-cli (download|upload)|hf_transfer|snapshot_download|"
    r"\bgit(?: -[Cc] \S+)* (clone|fetch|pull)\b|\bgit[- ]lfs (fetch|pull|clone|smudge)\b|"
    r"\bbrew (install|upgrade|reinstall)\b|\buv (sync|add|pip install|tool install|python install)\b|"
    r"\bpip3? install\b|\b(conda|mamba) (install|create|update)\b|\bnpm (install|ci)\b")
_GIT_DIR = re.compile(r" -[Cc] \S+")      # `git -C <path> fetch` is reported as `git fetch`
SETTLE_STEP = 5.0
ALLOWANCES = ("load", "gpu", "thermal", "lowpower")  # see --allow in the module docstring
MONITOR_COLUMNS = ("iso", "epoch", "state", "load1", "foreign_cpu", "gpu_util", "gpu_mem_gb", "battery",
                   "thermal_pressure", "pressure", "swapouts_delta", "foreign_gpu_ms", "foreign_rss_delta_mb",
                   "reasons")   # reasons stays last: free text never shifts the other columns
_FLOAT = {"epoch", "load1", "foreign_cpu", "gpu_mem_gb", "foreign_gpu_ms", "foreign_rss_delta_mb"}
_INT = {"gpu_util", "battery", "thermal_pressure", "pressure", "swapouts_delta"}
# Reasons that disturb a sample at once (thermal is logged only; "allowed ..." never counts).
_INSTANT = ("lock not held", "port ", "power:", "battery:", "tools:")
_BUSY = re.compile(r"([^\s,;]+)\((\d+)\) [\d.]+%")


class Proc(NamedTuple):
    pid: int
    ppid: int
    cpu: float
    rss_kb: int
    comm: str


def allow_names(text: str) -> list[str]:
    """argparse type for --allow: comma-separated names from ALLOWANCES (empty is allowed)."""
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in names if x not in ALLOWANCES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown allowance {', '.join(bad)} (known: {', '.join(ALLOWANCES)})")
    return names


def run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    """(exit code, stdout). Every probe goes through here, so tests replace this one function."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, f"{type(e).__name__}: {e}"


def tidy(s: str) -> str:
    """The home directory as ~ (the stdlib-only twin of modeleval.paths.tilde)."""
    return s.replace(HOME, "~")


def processes() -> list[Proc]:
    """Every process with its CPU %, resident size and executable path."""
    _, out = run(["ps", "-Ao", "pid=,ppid=,pcpu=,rss=,comm="])
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5:
            try:
                procs.append(Proc(int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3]), parts[4]))
            except ValueError:
                continue
    return procs


def allowed_tree(procs: list[Proc], roots: list[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for p in procs:
        children.setdefault(p.ppid, []).append(p.pid)
    seen: set[int] = set()
    stack = [r for r in roots if r > 0]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(children.get(p, []))
    return seen


def lock_purpose() -> str | None:
    """The purpose line of our lock; None when the lock is missing or another session's."""
    try:
        text = Path(LOCK).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not re.search(rf"^holder: {re.escape(LOCK_HOLDER)}\s*$", text, re.M):
        return None
    m = re.search(r"^purpose: (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def foreign_ports(mine: set[int]) -> list[int]:
    """The agent benchmark's ports (8500, 8600-8699) that a process outside our trees listens on."""
    _, out = run(["lsof", "-nP", f"-iTCP:{FOREIGN_PORTS}", "-sTCP:LISTEN", "-Fpn"])
    ports, pid = set(), None
    for line in out.splitlines():
        if line.startswith("p"):
            pid = int(line[1:]) if line[1:].isdigit() else None
        elif line.startswith("n"):
            m = re.search(r":(\d+)$", line)
            if m and pid not in mine:
                ports.add(int(m.group(1)))
    return sorted(ports)


def thermal_warnings() -> list[str]:
    """Throttling indicators from `pmset -g therm`, judged by value: a speed or scheduler limit
    below 100, or a nonzero warning level. 'Note: No ... recorded' lines and other status lines
    are not warnings."""
    _, out = run(["pmset", "-g", "therm"])
    warn = []
    for ln in (x.strip() for x in out.splitlines()):
        if not ln or ln.startswith("Note: No "):
            continue
        lim = re.search(r"(CPU_Speed_Limit|CPU_Scheduler_Limit|GPU_Speed_Limit)\s*=\s*(\d+)", ln)
        lvl = re.search(r"(?i)warning level\D*(\d+)", ln)
        if (lim and int(lim.group(2)) < 100) or (lvl and int(lvl.group(1)) > 0):
            warn.append(ln)
    return warn


def accelerator() -> str:
    """The GPU's IORegistry subtree: the AGXAccelerator (utilization, memory) and its user clients
    (per-process AppUsage). Same AppUsage entries as `ioreg -l -w0` (78 of 78 on 2026-10-26),
    23 ms instead of 190 ms per call."""
    _, out = run(["ioreg", "-r", "-c", "IOAccelerator", "-l", "-w0"])
    return out


def gpu_stats(text: str) -> tuple[int, float]:
    """(Device Utilization %, GB of GPU memory in use); (-1, -1.0) if ioreg has no figures."""
    util = re.search(r'"Device Utilization %"=(\d+)', text)
    mem = re.search(r'"In use system memory"=(\d+)', text)
    return (int(util.group(1)) if util else -1, round(int(mem.group(1)) / 1e9, 1) if mem else -1.0)


_CLIENT = re.compile(r"<class \w+, id (0x[0-9a-fA-F]+)")
_CREATOR = re.compile(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"')
_GPU_TIME = re.compile(r'"accumulatedGPUTime"=(\d+)')


def gpu_clients(text: str) -> dict[str, tuple[int, str, int]]:
    """GPU user clients: registry id -> (creator pid, creator name, accumulatedGPUTime in ns,
    summed over the client's AppUsage entries). The unit is nanoseconds: WindowServer's 1.6e13
    after 12 days awake is 4.5 h, which only fits ns. Keyed by client, not pid, because a
    process opens several clients and a closed one would make the pid's sum go backwards."""
    out = {}
    for block in text.split("+-o ")[1:]:
        cid, creator = _CLIENT.search(block), _CREATOR.search(block)
        if cid and creator:
            out[cid.group(1)] = (int(creator.group(1)), creator.group(2),
                                 sum(int(x) for x in _GPU_TIME.findall(block)))
    return out


def vm_swapouts() -> tuple[int | None, int]:
    """(vm_stat Swapouts in pages, page size in bytes); (None, 16384) if vm_stat fails."""
    _, out = run(["vm_stat"])
    page = re.search(r"page size of (\d+) bytes", out)
    swap = re.search(r"^Swapouts:\s+(\d+)", out, re.M)
    return (int(swap.group(1)) if swap else None, int(page.group(1)) if page else 16384)


def memory_pressure() -> int:
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical; -1 if unavailable."""
    _, out = run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    return int(out.strip()) if out.strip().isdigit() else -1


def power_state() -> dict:
    """{"source", "powermode", "lowpowermode"} from pmset; the modes are None when pmset has no
    such line. This machine shows `powermode` (0 automatic, 1 low, 2 high); Macs without High
    Power Mode show `lowpowermode` instead."""
    _, batt = run(["pmset", "-g", "batt"])
    _, conf = run(["pmset", "-g"])
    src = re.search(r"drawing from '([^']+)'", batt)
    mode = re.search(r"^\s*powermode\s+(\d+)", conf, re.M)
    low = re.search(r"^\s*lowpowermode\s+(\d+)", conf, re.M)
    return {"source": src.group(1) if src else "unknown", "powermode": mode.group(1) if mode else None,
            "lowpowermode": low.group(1) if low else None}


def low_power(st: dict) -> bool:
    return st["powermode"] == "1" or st["lowpowermode"] == "1"


def power_text(st: dict) -> str:
    mode = st["powermode"] or "?"
    return (f"{st['source']}; powermode {mode}" + {"2": " (High Power)", "1": " (Low Power)"}.get(mode, "")
            + ("; lowpowermode 1" if st["lowpowermode"] == "1" else ""))


def power_ok(st: dict) -> bool:
    """On AC power in High Power mode. Low Power Mode fails under either name."""
    return st["source"] == "AC Power" and (st["powermode"] or "?") in ("2", "?") and st["lowpowermode"] != "1"


def power() -> tuple[bool, str]:
    st = power_state()
    return power_ok(st), power_text(st)


def battery() -> tuple[int, str]:
    """(percent, pmset state such as 'charging' or 'discharging'); (-1, 'none') without a battery."""
    _, out = run(["pmset", "-g", "batt"])
    m = re.search(r"\t(\d+)%;\s*([^;]+);", out)
    return (int(m.group(1)), m.group(2).strip()) if m else (-1, "none")


def thermal_pressure() -> int:
    """System thermal pressure level (0 nominal .. 4 sleeping); -1 if unavailable."""
    _, out = run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"])
    m = re.search(r"(\d+)\s*$", out.strip())
    return int(m.group(1)) if m else -1


def tpl_text(level: int) -> str:
    return f"{level} ({TPL_NAMES.get(level, 'unknown')})" if level >= 0 else "unavailable"


def pressure_text(level: int) -> str:
    return f"{level} ({PRESSURE_NAMES.get(level, 'unknown')})" if level >= 0 else "unavailable"


def tool_kind(name: str, cmd: str) -> str | None:
    """What kind of build, convert or download tool a command line is: its BUILD_NAMES name, or the
    text of the BUILD_ARGS rule it matched (`git clone`, `hf download`, `pip install`), never its
    arguments. Peers keep working during a capability leg, and their arguments can carry a token
    (in a clone URL's user info, or a token option) or a private repo name; the kind is all that
    reaches monitor.tsv, units.json and results.json, which are committed."""
    if name in BUILD_NAMES:
        return name
    m = BUILD_ARGS.search(cmd)
    return _GIT_DIR.sub("", m.group(0)) if m else None


def build_tools(procs: list[Proc], mine: set[int]) -> list[dict]:
    """Foreign build, convert and download processes: {"pid", "kind" (tool_kind), "cmd" (the tidied
    command line, for the gitignored check report only)}."""
    _, cmdlines = run(["ps", "-wwAo", "pid=,command="])
    names = {p.pid: os.path.basename(p.comm) for p in procs}
    tools = []
    for line in cmdlines.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in mine:
            continue
        kind = tool_kind(names.get(pid, ""), cmd)
        if kind:
            tools.append({"pid": pid, "kind": kind, "cmd": tidy(cmd)[:110]})
    return tools


class Sampler:
    """Takes samples. Keeps the previous GPU, swap-out and RSS readings, so that the delta
    columns cover the time since the previous sample (None on the first one)."""

    def __init__(self, allow: list[int]):
        self.allow = list(allow)
        self._gpu: dict[str, tuple[int, str, int]] | None = None
        self._swapouts: int | None = None
        self._rss_mb: float | None = None

    def sample(self) -> dict:
        """One snapshot of everything the quiet rule looks at."""
        t = time.time()
        procs = processes()
        mine = allowed_tree(procs, self.allow + [os.getpid()])
        foreign = [p for p in procs if p.pid not in mine and os.path.basename(p.comm) not in IGNORED]
        busy = sorted(((p.pid, p.cpu, p.comm) for p in foreign if p.cpu > MAX_PROC_CPU), key=lambda x: -x[1])
        top = sorted(((p.pid, p.cpu, p.comm) for p in foreign), key=lambda x: -x[1])[:4]
        tools = build_tools(procs, mine)
        _, la = run(["sysctl", "-n", "vm.loadavg"])
        try:
            load1 = float(la.strip("{} \n").split()[0])
        except (ValueError, IndexError):
            load1 = -1.0
        pst = power_state()
        acc = accelerator()
        g_util, g_mem = gpu_stats(acc)
        gpu_ms, gpu_top = self._foreign_gpu(gpu_clients(acc), mine, {p.pid: p.comm for p in procs})
        swapouts, page = vm_swapouts()
        swap_delta = swapouts - self._swapouts if swapouts is not None and self._swapouts is not None else None
        self._swapouts = swapouts
        rss_mb = sum(p.rss_kb for p in foreign) / 1024
        rss_delta = round(rss_mb - self._rss_mb, 1) if self._rss_mb is not None and procs else None
        self._rss_mb = rss_mb if procs else None
        batt, batt_state = battery()
        return {"t": t, "lock": lock_purpose(), "ports": foreign_ports(mine), "load1": load1,
                "gpu_util": g_util, "gpu_mem_gb": g_mem,
                "foreign_cpu": round(sum(p.cpu for p in foreign), 1), "busy": busy, "top": top, "tools": tools,
                "thermal": thermal_warnings(), "power_ok": power_ok(pst), "power": power_text(pst),
                "low_power_ac": pst["source"] == "AC Power" and low_power(pst),
                "battery": batt, "battery_state": batt_state, "tpl": thermal_pressure(),
                "pressure": memory_pressure(), "swapouts_delta": swap_delta, "page_size": page,
                "foreign_gpu_ms": gpu_ms, "gpu_top": gpu_top, "foreign_rss_delta_mb": rss_delta}

    def _foreign_gpu(self, clients: dict[str, tuple[int, str, int]], mine: set[int],
                     comms: dict[int, str]) -> tuple[float | None, list[tuple[int, float, str]]]:
        """(foreign GPU ms since the previous sample, top foreign users as (pid, ms, comm)).
        A client absent from the previous sample counts 0 this time: its total could include
        time from before the previous sample (a failed ioreg call), and persistence needs a
        second sample anyway."""
        prev, self._gpu = self._gpu, (clients or None)
        if prev is None or not clients:
            return None, []
        per_pid: dict[int, int] = {}
        for cid, (pid, cname, ns) in clients.items():
            if pid in mine or cname in IGNORED or cid not in prev:
                continue
            d = ns - prev[cid][2]
            if d > 0:
                per_pid[pid] = per_pid.get(pid, 0) + d
        top = sorted(per_pid.items(), key=lambda x: -x[1])[:3]
        return round(sum(per_pid.values()) / 1e6, 1), [(pid, ns / 1e6, comms.get(pid, "?")) for pid, ns in top]


def sample(allow: list[int]) -> dict:
    """One snapshot without deltas (the delta columns are None)."""
    return Sampler(allow).sample()


@functools.lru_cache(maxsize=None)
def third_party_app(comm: str) -> bool:
    """True when the executable lives in an app bundle that is not Apple's (an unreadable
    bundle counts as third-party)."""
    if ".app/" not in comm:
        return False
    outer = comm.split(".app/", 1)[0] + ".app"
    try:
        with open(os.path.join(os.path.expanduser(outer), "Contents", "Info.plist"), "rb") as f:
            bid = str(plistlib.load(f).get("CFBundleIdentifier", ""))
    except (OSError, plistlib.InvalidFileException, ValueError):
        return True
    return not bid.startswith("com.apple.")


def name(comm: str) -> str:
    return "<app>" if third_party_app(comm) else os.path.basename(comm)


def shown(comm: str) -> str:
    """The executable path as printed: `<app>` for third-party apps, ~ for the home directory."""
    return "<app>" if third_party_app(comm) else tidy(comm)


def problems(s: dict, judge_load: bool, max_gpu_gb: float | None = None, min_battery: float | None = None,
             allow: frozenset = frozenset()) -> tuple[list[str], list[str], list[str]]:
    """(hard, soft, allowed) reasons for a sample. Hard: lock, foreign server ports, power, a
    battery below the run limit. Soft: the rest, including a battery below min_battery (check
    only). Allowed: what an --allow name accepts, recorded as "allowed <reason>" so that it
    stays visible but never counts."""
    hard, soft, allowed = [], [], []
    if s["lock"] is None:
        hard.append("lock not held by us")
    hard += [f"port {p} listening" for p in s["ports"]]
    if not s["power_ok"]:
        if "lowpower" in allow and s.get("low_power_ac"):
            allowed.append(f"allowed power: {s['power']}")
        else:
            hard.append(f"power: {s['power']}")
    if 0 <= s["battery"] < MIN_BATTERY_RUN:
        hard.append(f"battery: {s['battery']}% < {MIN_BATTERY_RUN:.0f}%")
    elif min_battery is not None and 0 <= s["battery"] < min_battery:
        soft.append(f"battery {s['battery']}% < {min_battery:.0f}% (charging)")
    if judge_load and s["load1"] > MAX_LOAD:
        (allowed if "load" in allow else soft).append(
            ("allowed " if "load" in allow else "") + f"load1 {s['load1']:.2f} > {MAX_LOAD}")
    if judge_load and s["gpu_util"] > MAX_GPU:   # judge_load: our server is idle, so the GPU should be too
        (allowed if "gpu" in allow else soft).append(
            ("allowed " if "gpu" in allow else "") + f"GPU busy {s['gpu_util']}% > {MAX_GPU:.0f}%")
    if max_gpu_gb is not None and s["gpu_mem_gb"] > max_gpu_gb:
        soft.append(f"GPU memory in use {s['gpu_mem_gb']} GB > {max_gpu_gb} GB (another model resident?)")
    if s["busy"]:
        soft.append("busy: " + ", ".join(f"{name(c)}({pid}) {cpu:.0f}%" for pid, cpu, c in s["busy"]))
    if s["foreign_cpu"] > MAX_FOREIGN_CPU:
        soft.append(f"foreign CPU {s['foreign_cpu']:.0f}% > {MAX_FOREIGN_CPU:.0f}%")
    if s["tools"]:   # one segment (", "), so row_flags keeps every tool; kinds only, never arguments
        soft.append("tools: " + ", ".join(f"{t['kind']} (pid {t['pid']})" for t in s["tools"]))
    if s["thermal"]:
        (allowed if "thermal" in allow else soft).append(
            ("allowed " if "thermal" in allow else "") + "thermal: " + "; ".join(s["thermal"]))
    return hard, soft, allowed


def memory_events(row: dict, page_size: int) -> list[str]:
    """Memory pressure >= 2 and swap-out bursts in a sample (judged per window, see judge_window)."""
    out = []
    p = row.get("pressure")
    if p is not None and p >= 2:
        out.append(f"memory pressure {pressure_text(p)}")
    sw = row.get("swapouts_delta")
    if sw is not None and sw * page_size / 2**20 >= SWAP_BURST_MB:
        out.append(f"swap-out {sw * page_size / 2**20:.0f} MB")
    return out


def report(s: dict, settle_note: str | None, verdict: bool, allow: frozenset = frozenset()) -> str:
    stamp = dt.datetime.fromtimestamp(s["t"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    def tag(name: str) -> str:
        return f" [allowance {name}]" if name in allow else ""

    peer = [p for p in s["ports"] if p != PEER_PORT]
    lines = [f"quiet_check: {stamp}",
             f"lock: held by us ({s['lock']})" if s["lock"] is not None else "lock: NOT held by us",
             f"port {PEER_PORT}: LISTENING (agent-benchmark oMLX is up)" if PEER_PORT in s["ports"]
             else f"port {PEER_PORT}: free",
             "ports 8600-8699: " + (f"LISTENING {', '.join(map(str, peer))} (agent benchmark)" if peer
                                    else "none listening"),
             f"load1: {s['load1']:.2f} (max {MAX_LOAD}){tag('load')}",
             f"gpu: {s['gpu_util']}% busy (max {MAX_GPU:.0f}){tag('gpu')}, {s['gpu_mem_gb']} GB in use",
             f"foreign CPU: {s['foreign_cpu']:.1f}% total (max {MAX_FOREIGN_CPU:.0f}); top: "
             + ", ".join(f"{name(c)} {cpu:.1f}" for _, cpu, c in s["top"])]
    if s["busy"]:
        lines.append(f"busy foreign processes (>{MAX_PROC_CPU:.0f}% CPU):")
        lines += [f"  {pid} {cpu:.1f} {shown(c)}" for pid, cpu, c in s["busy"]]
    else:
        lines.append("busy foreign processes: none")
    if s["tools"]:
        lines.append("build/convert/download tools: RUNNING")
        lines += [f"  {t['pid']} {t['cmd']}" for t in s["tools"]]   # the full command: check logs are gitignored
    else:
        lines.append("build/convert/download tools: none")
    lines.append("thermal: " + ("; ".join(s["thermal"]) if s["thermal"] else "no warning recorded")
                 + (tag("thermal") if s["thermal"] else ""))
    lines.append(f"power: {s['power']}" + (tag("lowpower") if not s["power_ok"] and s.get("low_power_ac") else ""))
    lines.append("battery: " + (f"{s['battery']}% ({s['battery_state']})" if s["battery"] >= 0 else "none"))
    lines.append(f"thermal pressure: {tpl_text(s['tpl'])}")
    lines.append(f"memory pressure: {pressure_text(s['pressure'])} (logged, not judged)")
    if allow:
        lines.append(f"allowances: {', '.join(sorted(allow))} (an interference load's own side effects; "
                     "foreign processes are judged as always)")
    if settle_note:
        lines.append(settle_note)
    lines.append("verdict: QUIET" if verdict else "verdict: NOT QUIET")
    return "\n".join(lines)


def cmd_check(a: argparse.Namespace) -> int:
    allow = frozenset(n for names in a.allow for n in names)
    sampler = Sampler(a.allow_pid)
    t0 = time.time()
    while True:
        s = sampler.sample()
        hard, soft, _ = problems(s, judge_load=True, max_gpu_gb=a.max_gpu_gb, min_battery=a.min_battery, allow=allow)
        waited = time.time() - t0
        if hard or not soft or waited + SETTLE_STEP > a.settle:
            break
        time.sleep(SETTLE_STEP)
    ok = not hard and not soft
    note = None
    if a.settle > 0:
        if ok:
            note = f"settle: quiet after {waited:.0f} s (limit {a.settle:.0f} s)"
        elif hard:
            note = f"settle: not attempted (hard failure: {'; '.join(hard)})"
        else:
            note = f"settle: still not quiet after {waited:.0f} s (limit {a.settle:.0f} s): {'; '.join(soft)}"
    print(report(s, note, ok, allow))
    return 0 if ok else 1


def _segments(reasons: str) -> list[str]:
    return [x.strip() for x in reasons.split("; ") if x.strip()]


def _busy(segs: list[str]) -> dict[str, str]:
    """pid -> "name(pid)" for the foreign processes in a sample's busy list."""
    return {m.group(2): f"{m.group(1)}({m.group(2)})"
            for x in segs if x.startswith("busy: ") for m in _BUSY.finditer(x)}


def _over(row: dict | None, col: str, limit: float) -> bool:
    v = row.get(col) if row is not None else None
    return v is not None and v > limit


def row_flags(row: dict, prev: dict | None, gpu_floor_ms: float = GPU_FLOOR_MS) -> list[str]:
    """Why a sample is disturbed, given the sample before it (None at the start of a monitor
    session); empty when it is not. Memory is judged per window (judge_window), not here."""
    segs = _segments(row.get("reasons") or "")
    flags = [x for x in segs if x.startswith(_INSTANT)]
    b = row.get("battery")
    if b is not None and 0 <= b < MIN_BATTERY_RUN and not any(x.startswith("battery:") for x in flags):
        flags.append(f"battery: {b}% < {MIN_BATTERY_RUN:.0f}%")
    if prev is not None:
        prev_busy = _busy(_segments(prev.get("reasons") or ""))
        both = [v for k, v in _busy(segs).items() if k in prev_busy]
        if both:
            flags.append(f"busy in 2 consecutive samples: {', '.join(both)}")
        if _over(row, "foreign_cpu", MAX_FOREIGN_CPU) and _over(prev, "foreign_cpu", MAX_FOREIGN_CPU):
            flags.append(f"foreign CPU {row['foreign_cpu']:.0f}% > {MAX_FOREIGN_CPU:.0f}% in 2 consecutive samples")
        if _over(row, "foreign_gpu_ms", gpu_floor_ms) and _over(prev, "foreign_gpu_ms", gpu_floor_ms):
            flags.append(f"foreign GPU {row['foreign_gpu_ms']:.0f} ms > {gpu_floor_ms:.0f} ms "
                         "in 2 consecutive samples")
    return flags


def row_state(row: dict, prev: dict | None, gpu_floor_ms: float = GPU_FLOOR_MS) -> str:
    """DISTURBED (row_flags), SPIKE (a one-sample CPU or GPU excess) or QUIET."""
    if row_flags(row, prev, gpu_floor_ms):
        return "DISTURBED"
    single = (_busy(_segments(row.get("reasons") or "")) or _over(row, "foreign_cpu", MAX_FOREIGN_CPU)
              or _over(row, "foreign_gpu_ms", gpu_floor_ms))
    return "SPIKE" if single else "QUIET"


def monitor_row(s: dict, allow: frozenset = frozenset()) -> dict:
    """A sample as a monitor row (MONITOR_COLUMNS, typed; state still unset). The reasons hold
    the judged events plus logged notes: foreign GPU above the floor, memory events, thermal."""
    hard, soft, allowed = problems(s, judge_load=False, allow=allow)
    notes = []
    if s["foreign_gpu_ms"] is not None and s["foreign_gpu_ms"] > GPU_FLOOR_MS:
        notes.append(f"foreign GPU {s['foreign_gpu_ms']:.0f} ms: "
                     + ", ".join(f"{name(c)}({pid}) {ms:.0f} ms" for pid, ms, c in s["gpu_top"]))
    mem = memory_events(s, s["page_size"])
    if mem and s["foreign_rss_delta_mb"] is not None:
        mem.append(f"foreign RSS {s['foreign_rss_delta_mb']:+.0f} MB since the previous sample")
    notes += mem
    iso = dt.datetime.fromtimestamp(s["t"]).astimezone().isoformat(timespec="seconds")
    return {"iso": iso, "epoch": s["t"], "state": "", "load1": s["load1"], "foreign_cpu": s["foreign_cpu"],
            "gpu_util": s["gpu_util"], "gpu_mem_gb": s["gpu_mem_gb"], "battery": s["battery"],
            "thermal_pressure": s["tpl"], "pressure": s["pressure"], "swapouts_delta": s["swapouts_delta"],
            "foreign_gpu_ms": s["foreign_gpu_ms"], "foreign_rss_delta_mb": s["foreign_rss_delta_mb"],
            "reasons": tidy("; ".join(hard + soft + allowed + notes))}


def _cell(col: str, v: object) -> str:
    if v is None:
        return ""
    if col == "epoch":
        return f"{v:.1f}"
    if col == "load1":
        return f"{v:.2f}"
    if col in ("foreign_cpu", "foreign_gpu_ms", "foreign_rss_delta_mb"):
        return f"{v:.1f}"
    return re.sub(r"[\t\r\n]+", " ", str(v))


def format_row(row: dict) -> str:
    """One TSV line (with its newline) in MONITOR_COLUMNS order."""
    return "\t".join(_cell(c, row.get(c)) for c in MONITOR_COLUMNS) + "\n"


def monitor_header(interval: float, allow_pids: list[int], allow: frozenset, page_size: int) -> str:
    return (f"# quietlib monitor v4 interval={interval} max_proc_cpu={MAX_PROC_CPU} "
            f"max_foreign_cpu={MAX_FOREIGN_CPU} min_battery_run={MIN_BATTERY_RUN} gpu_floor_ms={GPU_FLOOR_MS} "
            f"swap_burst_mb={SWAP_BURST_MB} foreign_rss_growth_mb={FOREIGN_RSS_GROWTH_MB} page_size={page_size} "
            f"allow={','.join(map(str, allow_pids))}"
            + (f" allowances={','.join(sorted(allow))}" if allow else "") + "\n"
            + "# " + "\t".join(MONITOR_COLUMNS) + "\n")


def cmd_monitor(a: argparse.Namespace) -> int:
    allow = frozenset(n for names in a.allow for n in names)
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(a.allow_pid)
    with log.open("a", encoding="utf-8") as f:
        f.write(monitor_header(a.interval, a.allow_pid, allow, vm_swapouts()[1]))
        f.flush()
        state, prev, n = None, None, 0
        while True:
            if a.until_pid and not pid_alive(a.until_pid):
                break
            s = sampler.sample()
            row = monitor_row(s, allow)
            row["state"] = row_state(row, prev)
            f.write(format_row(row))
            f.flush()
            if row["state"] != state:
                print(f"{row['iso'][11:19]} {row['state']}" + (f": {row['reasons']}" if row["reasons"] else ""),
                      flush=True)
                state = row["state"]
            prev, n = row, n + 1
            if a.count and n >= a.count:
                break
            time.sleep(max(min(0.5, a.interval), a.interval - (time.time() - s["t"])))
    return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse(col: str, v: str) -> object:
    if col in _FLOAT or col in _INT:
        if v == "":
            return None
        try:
            return int(v) if col in _INT else float(v)
        except ValueError:
            return None
    return v


def read_monitor(path: Path | str) -> list[dict]:
    """The rows of a monitor TSV, typed (empty cells are None), in file order. Columns come from
    the latest header line, so v3 files (without the four memory/GPU columns) read too. Every
    row carries "_session" (which monitor start wrote it: persistence never spans two) and
    "_meta" (that start's interval and page size). A last line without its newline is still
    being written and is skipped."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows: list[dict] = []
    cols: tuple[str, ...] = MONITOR_COLUMNS
    meta = {"interval": MONITOR_INTERVAL, "page_size": 16384}
    session = 0
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        line = line.rstrip("\r\n")
        if line.startswith("#"):
            if line.startswith("# quietlib monitor"):
                session += 1
                meta = {"interval": MONITOR_INTERVAL, "page_size": 16384}
                for key, conv in (("interval", float), ("page_size", int)):
                    m = re.search(rf"\b{key}=([\d.]+)", line)
                    if m:
                        meta[key] = conv(m.group(1))
            elif line[1:].lstrip().startswith("iso\t"):
                cols = tuple(line[1:].lstrip().split("\t"))
            continue
        parts = line.split("\t")
        if len(parts) < len(cols) - 1:      # an empty reasons cell may lose its tab in hand edits
            continue
        row = {c: _parse(c, parts[i]) if i < len(parts) else "" for i, c in enumerate(cols)}
        if "reasons" in cols:                # reasons is last; tabs inside it were never written
            row["reasons"] = "\t".join(parts[cols.index("reasons"):]) if len(parts) >= len(cols) else ""
        if row.get("epoch") is None:
            continue
        row["_session"], row["_meta"] = session, meta
        rows.append(row)
    return rows


def _window(tsv_path: Path | str, t_start: float, t_end: float, pad_s: float,
            gpu_floor_ms: float) -> tuple[list[tuple[dict, list[str]]], str | None]:
    """The samples inside [t_start - pad, t_end + pad] with their disturbance flags, and a
    coverage problem (no samples, or a gap of more than 3 intervals, at least 15 s) or None."""
    try:
        rows = read_monitor(tsv_path)
    except OSError as e:
        return [], f"monitor log unreadable: {type(e).__name__}"
    lo, hi = t_start - pad_s, t_end + pad_s
    judged: list[tuple[dict, list[str]]] = []
    prev = None
    for r in rows:
        if prev is not None and prev["_session"] != r["_session"]:
            prev = None
        if lo <= r["epoch"] <= hi:
            judged.append((r, row_flags(r, prev, gpu_floor_ms)))
        prev = r
    growth = 0.0
    for r, flags in judged:
        growth += r.get("foreign_rss_delta_mb") or 0.0
        mem = memory_events(r, r["_meta"]["page_size"])
        if mem and growth > FOREIGN_RSS_GROWTH_MB:
            flags.append(f"{'; '.join(mem)} while foreign RSS grew {growth:+.0f} MB in the window")
    if not judged:
        return judged, "no monitor samples in the window"
    interval = max(r["_meta"]["interval"] for r, _ in judged)
    edges = [lo] + [r["epoch"] for r, _ in judged] + [hi]
    gap = max(b - a for a, b in zip(edges, edges[1:]))
    if gap > 3 * max(interval, MONITOR_INTERVAL):   # the pad alone is one 5 s interval
        return judged, f"monitor gap of {gap:.0f} s in the window (interval {interval:g} s)"
    return judged, None


def _hms(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _peak_cpu(judged: list[tuple[dict, list[str]]]) -> float:
    return max((r["foreign_cpu"] for r, _ in judged if r.get("foreign_cpu") is not None), default=0.0)


def judge_window(tsv_path: Path | str, t_start: float, t_end: float, pad_s: float = PAD_S,
                 gpu_floor_ms: float = GPU_FLOOR_MS) -> dict:
    """A timed unit's verdict (DESIGN 5.4) from the monitor log and the orchestrator's own epoch
    timestamps: {"clean", "reasons", "peak_foreign_cpu", "samples", "disturbed_samples"}. Not
    clean when a sample inside [t_start - pad_s, t_end + pad_s] is disturbed (see the module
    docstring), or when the monitor did not cover the window: no samples in it, or a gap of more
    than 3 intervals and 15 s (call it once the monitor has written a sample after t_end + pad_s)."""
    judged, coverage = _window(tsv_path, t_start, t_end, pad_s, gpu_floor_ms)
    bad = [(r, f) for r, f in judged if f]
    reasons = [coverage] if coverage else []
    if bad:
        reasons.append(f"{len(bad)} disturbed sample(s) of {len(judged)}")
        reasons += [f"{_hms(r['epoch'])} {'; '.join(f)}" for r, f in bad[:MAX_REASONS]]
        if len(bad) > MAX_REASONS:
            reasons.append(f"... and {len(bad) - MAX_REASONS} more")
    return {"clean": not reasons, "reasons": reasons, "peak_foreign_cpu": _peak_cpu(judged),
            "samples": len(judged), "disturbed_samples": len(bad)}


def window_flags(tsv_path: Path | str, t_start: float, t_end: float, pad_s: float = PAD_S,
                 gpu_floor_ms: float = GPU_FLOOR_MS) -> dict:
    """A capability task's flags (DESIGN 6.4; never a re-run): {"disturbed_samples",
    "peak_foreign_cpu", "battery_min" (None without battery readings), "samples"}, judged with
    the same rule as judge_window."""
    judged, _ = _window(tsv_path, t_start, t_end, pad_s, gpu_floor_ms)
    batt = [r["battery"] for r, _ in judged if r.get("battery") is not None and r["battery"] >= 0]
    return {"disturbed_samples": sum(1 for _, f in judged if f), "peak_foreign_cpu": _peak_cpu(judged),
            "battery_min": min(batt) if batt else None, "samples": len(judged)}


def cmd_battery(a: argparse.Namespace) -> int:
    t0, last = time.time(), -1e9
    while True:
        pct, state = battery()
        if pct < 0 or pct >= a.min:
            print(f"battery: {'none' if pct < 0 else f'{pct}% ({state})'}; ok (min {a.min:.0f}%)", flush=True)
            return 0
        waited = time.time() - t0
        if waited >= a.wait:
            print(f"battery: {pct}% ({state}) < {a.min:.0f}% after {waited:.0f} s", flush=True)
            return 1
        if time.time() - last >= 300:
            print(f"battery: {pct}% ({state}) < {a.min:.0f}%; waiting ({waited:.0f}/{a.wait:.0f} s)", flush=True)
            last = time.time()
        time.sleep(min(30.0, max(1.0, a.wait - waited)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quietlib", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--allow-pid", type=int, action="append", default=[])
    c.add_argument("--settle", type=float, default=0.0)
    c.add_argument("--max-gpu-gb", type=float)
    c.add_argument("--min-battery", type=float)
    c.add_argument("--allow", type=allow_names, action="append", default=[], help="allowances: " + ", ".join(ALLOWANCES))
    m = sub.add_parser("monitor")
    m.add_argument("--log", required=True)
    m.add_argument("--interval", type=float, default=MONITOR_INTERVAL)
    m.add_argument("--allow-pid", type=int, action="append", default=[])
    m.add_argument("--until-pid", type=int)
    m.add_argument("--count", type=int, default=0, help="stop after N samples (0: until killed or --until-pid exits)")
    m.add_argument("--allow", type=allow_names, action="append", default=[], help="allowances: " + ", ".join(ALLOWANCES))
    b = sub.add_parser("battery")
    b.add_argument("--min", type=float, required=True)
    b.add_argument("--wait", type=float, default=0.0)
    a = p.parse_args(argv)
    return {"check": cmd_check, "monitor": cmd_monitor, "battery": cmd_battery}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
