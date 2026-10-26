#!/usr/bin/env python3
"""Quiet-machine checks for timed segments (METHODOLOGY.md, "Quiet machine").

Stdlib only; run with /usr/bin/python3 -I. Subcommands:

  check   [--allow-pid PID]... [--settle S] [--max-gpu-gb G] [--allow NAME[,NAME]]
          One-shot pre-flight, run while the server under test is idle. Prints a provenance
          block; exit 1 when the machine is not quiet. With --settle, soft failures (load, GPU
          busy, busy processes, build tools, thermal) are re-checked every 5 s for up to S
          seconds. Hard failures (lock, port 8500, power) fail at once. --max-gpu-gb (used
          before a server starts) also fails while more than G GB of GPU memory is in use,
          i.e. while another model is resident. --min-battery fails while the battery is below
          PCT %.
  monitor --log FILE [--interval S] [--allow-pid PID]... [--until-pid PID] [--allow NAME[,NAME]]
          Samples while a segment runs. Every sample is a TSV row in FILE; stdout gets one line
          per state change. Lock, port 8500, power, thermal and build tools mark a sample
          DISTURBED at once. CPU counts only when it persists: the same foreign process above
          the per-process limit in two consecutive samples, or foreign CPU in total above its
          limit in two consecutive samples. A one-sample daemon blip is logged as SPIKE. Load
          average is logged but not judged, since the server under test adds to it. A battery
          below QUIET_MIN_BATTERY_RUN % also marks a sample DISTURBED at once. The battery
          level and the thermal pressure level are logged in every row; thermal pressure is
          not judged, since the run itself produces it.
  battery --min PCT [--wait S]
          Waits up to S seconds (default 0) for the battery to reach PCT %, printing the level
          every 5 min. Exit 1 if it does not. Machines without a battery always pass.
  verify  --log FILE --raw-dir DIR --since EPOCH [--move]
          For each result file in DIR created at or after EPOCH, re-judges the logged samples
          with the monitor's rule (from their reasons, so every suite gets the same rule) and
          looks for disturbed samples, or gaps in sampling, inside the file's window: after its
          header `created` and up to one interval past its mtime.
          --move sends those files, and their .md siblings, to DIR/_unverified/.
          Exit 1 if any file was disturbed. Allowances in the log's header are printed.

--allow-pid PID exempts PID and all its descendants (the server under test, its engine
workers, the harness, an M8 interference load) from the per-process checks.

--allow NAME (M8, METHODOLOGY.md section 4) accepts one system-wide side effect of an
interference load that our own script started; anything else is judged as before:
  load      the 1-min load average is not judged (check; the monitor never judges it)
  gpu       GPU utilization is not judged (check; the monitor never judges it)
  thermal   a pmset thermal warning is recorded as "allowed thermal: ..." instead of failing
  lowpower  Low Power Mode on AC power is recorded as "allowed power: ..."; battery power, or
            any other mode than High Power or Low Power, still fails
The allowances are printed by check (so the raw header's preflight block holds them) and
written to the monitor log's header (allowances=...). Allowed reasons start with "allowed ",
so verify, which judges the logged reasons, never counts them.

Processes inside an app bundle whose identifier is not com.apple.* are shown as `<app>`, so
the user's own apps never reach committed files (scripts/redact.py applies the same rule to
older files).

Thresholds (environment): QUIET_MAX_LOAD (1-min load, check only, default 2.5),
QUIET_MAX_PROC_CPU (any single foreign process, default 25 %), QUIET_MAX_FOREIGN_CPU
(sum over foreign processes, default 150 %), QUIET_MAX_GPU (GPU "Device Utilization %"
while our server is idle, check only, default 10 %). GPU figures come from the
AGXAccelerator PerformanceStatistics in ioreg (no sudo). QUIET_MIN_BATTERY_RUN (default
20 %): the 96 W adapter cannot supply this machine under sustained load, so the battery
drains, and below ~7 % the system caps power (decode fell 22 % on 2026-10-18 at 05:16).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import functools
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = str(Path.home())
LOCK = os.environ.get("GPU_LOCK_PATH", f"{HOME}/Developer/mlsys/experiments/.gpu-lock")
LOCK_HOLDER = "local_ai-benchmark session"
PEER_PORT = 8500
MAX_LOAD = float(os.environ.get("QUIET_MAX_LOAD", "2.5"))
MAX_PROC_CPU = float(os.environ.get("QUIET_MAX_PROC_CPU", "25"))
MAX_FOREIGN_CPU = float(os.environ.get("QUIET_MAX_FOREIGN_CPU", "150"))
MAX_GPU = float(os.environ.get("QUIET_MAX_GPU", "10"))
MIN_BATTERY_RUN = float(os.environ.get("QUIET_MIN_BATTERY_RUN", "20"))
TPL_NAMES = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}   # OSThermalNotification.h
IGNORED = {"WindowServer", "kernel_task"}  # always resident; kernel_task also does the server's driver work
BUILD_NAMES = {"clang", "clang++", "cc1", "cc1plus", "ld", "ld64", "ninja", "make", "gmake", "cmake",
               "swift-frontend", "swiftc", "rustc", "cargo", "xcodebuild"}
BUILD_ARGS = re.compile(r"mlc_llm (convert_weight|compile)|coreai\.llm\.export|\bhf download\b|"
                        r"huggingface-cli download|hf_transfer|snapshot_download|"
                        r"\bbrew (install|upgrade|reinstall)\b|\buv (sync|pip install)\b|\bpip3? install\b")
SETTLE_STEP = 5.0
ALLOWANCES = ("load", "gpu", "thermal", "lowpower")  # see --allow in the module docstring


def allow_names(text: str) -> list[str]:
    """argparse type for --allow: comma-separated names from ALLOWANCES (empty is allowed)."""
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in names if x not in ALLOWANCES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown allowance {', '.join(bad)} (known: {', '.join(ALLOWANCES)})")
    return names


def run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, f"{type(e).__name__}: {e}"


def tidy(s: str) -> str:
    return s.replace(HOME, "~")


def processes() -> list[tuple[int, int, float, str]]:
    """(pid, ppid, %cpu, executable path) for every process."""
    _, out = run(["ps", "-Ao", "pid=,ppid=,pcpu=,comm="])
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            try:
                procs.append((int(parts[0]), int(parts[1]), float(parts[2]), parts[3]))
            except ValueError:
                continue
    return procs


def allowed_tree(procs: list[tuple[int, int, float, str]], roots: list[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, ppid, _, _ in procs:
        children.setdefault(ppid, []).append(pid)
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
    try:
        text = Path(LOCK).read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(rf"^holder: {re.escape(LOCK_HOLDER)}\s*$", text, re.M):
        return None
    m = re.search(r"^purpose: (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def peer_port_up() -> bool:
    rc, _ = run(["lsof", "-nP", f"-iTCP:{PEER_PORT}", "-sTCP:LISTEN"])
    return rc == 0


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


def gpu() -> tuple[int, float]:
    """(Device Utilization %, GB of GPU memory in use); (-1, -1.0) if ioreg has no figures."""
    _, out = run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"])
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    mem = re.search(r'"In use system memory"=(\d+)', out)
    return (int(util.group(1)) if util else -1, round(int(mem.group(1)) / 1e9, 1) if mem else -1.0)


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


def sample(allow: list[int]) -> dict:
    """One snapshot of everything the quiet rule looks at."""
    procs = processes()
    mine = allowed_tree(procs, allow + [os.getpid()])
    foreign = [(pid, cpu, comm) for pid, _, cpu, comm in procs
               if pid not in mine and os.path.basename(comm) not in IGNORED]
    busy = sorted((p for p in foreign if p[1] > MAX_PROC_CPU), key=lambda p: -p[1])
    top = sorted(foreign, key=lambda p: -p[1])[:4]
    _, cmdlines = run(["ps", "-wwAo", "pid=,command="])
    tools = []
    names = {pid: os.path.basename(comm) for pid, _, _, comm in procs}
    for line in cmdlines.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in mine:
            continue
        if names.get(pid) in BUILD_NAMES or BUILD_ARGS.search(cmd):
            tools.append(f"{pid} {tidy(cmd)[:110]}")
    _, la = run(["sysctl", "-n", "vm.loadavg"])
    try:
        load1 = float(la.strip("{} \n").split()[0])
    except (ValueError, IndexError):
        load1 = -1.0
    pst = power_state()
    g_util, g_mem = gpu()
    batt, batt_state = battery()
    return {"t": time.time(), "lock": lock_purpose(), "peer_port": peer_port_up(), "load1": load1,
            "gpu_util": g_util, "gpu_mem_gb": g_mem,
            "foreign_cpu": round(sum(p[1] for p in foreign), 1), "busy": busy, "top": top, "tools": tools,
            "thermal": thermal_warnings(), "power_ok": power_ok(pst), "power": power_text(pst),
            "low_power_ac": pst["source"] == "AC Power" and low_power(pst),
            "battery": batt, "battery_state": batt_state, "tpl": thermal_pressure()}


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
    """(hard, soft, allowed) reasons for a sample. Hard: lock, port, power, a battery below the
    run limit. Soft: the rest, including a battery below min_battery (check only). Allowed: what
    an --allow name accepts, recorded as "allowed <reason>" so that it stays visible but never
    counts (verify matches reasons by their start)."""
    hard, soft, allowed = [], [], []
    if s["lock"] is None:
        hard.append("lock not held by us")
    if s["peer_port"]:
        hard.append(f"port {PEER_PORT} listening")
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
    if s["tools"]:
        soft.append("tools: " + "; ".join(t.split(None, 1)[1][:60] for t in s["tools"]))
    if s["thermal"]:
        (allowed if "thermal" in allow else soft).append(
            ("allowed " if "thermal" in allow else "") + "thermal: " + "; ".join(s["thermal"]))
    return hard, soft, allowed


def report(s: dict, settle_note: str | None, verdict: bool, allow: frozenset = frozenset()) -> str:
    stamp = dt.datetime.fromtimestamp(s["t"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    def tag(name: str) -> str:
        return f" [allowance {name}]" if name in allow else ""

    lines = [f"quiet_check: {stamp}",
             f"lock: held by us ({s['lock']})" if s["lock"] is not None else "lock: NOT held by us",
             f"port {PEER_PORT}: LISTENING (agent-benchmark oMLX is up)" if s["peer_port"] else f"port {PEER_PORT}: free",
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
        lines += [f"  {t}" for t in s["tools"]]
    else:
        lines.append("build/convert/download tools: none")
    lines.append("thermal: " + ("; ".join(s["thermal"]) if s["thermal"] else "no warning recorded")
                 + (tag("thermal") if s["thermal"] else ""))
    lines.append(f"power: {s['power']}" + (tag("lowpower") if not s["power_ok"] and s.get("low_power_ac") else ""))
    lines.append("battery: " + (f"{s['battery']}% ({s['battery_state']})" if s["battery"] >= 0 else "none"))
    lines.append(f"thermal pressure: {tpl_text(s['tpl'])}")
    if allow:
        lines.append(f"allowances: {', '.join(sorted(allow))} (an M8 interference load's own side effects; "
                     "foreign processes are judged as always)")
    if settle_note:
        lines.append(settle_note)
    lines.append("verdict: QUIET" if verdict else "verdict: NOT QUIET")
    return "\n".join(lines)


def cmd_check(a: argparse.Namespace) -> int:
    allow = frozenset(n for names in a.allow for n in names)
    t0 = time.time()
    while True:
        s = sample(a.allow_pid)
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


def cmd_monitor(a: argparse.Namespace) -> int:
    allow = frozenset(n for names in a.allow for n in names)
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"# quietlib monitor v3 interval={a.interval} max_proc_cpu={MAX_PROC_CPU} "
                f"max_foreign_cpu={MAX_FOREIGN_CPU} min_battery_run={MIN_BATTERY_RUN} "
                f"allow={','.join(map(str, a.allow_pid))}"
                + (f" allowances={','.join(sorted(allow))}" if allow else "") + "\n")
        f.write("# iso\tepoch\tstate\tload1\tforeign_cpu\tgpu_util\tgpu_mem_gb\tbattery\tthermal_pressure\treasons\n")
        f.flush()
        state, prev_pids, prev_over = None, set(), False
        while True:
            if a.until_pid and not pid_alive(a.until_pid):
                break
            s = sample(a.allow_pid)
            hard, soft, allowed = problems(s, judge_load=False, allow=allow)
            reasons = hard + soft + allowed
            pids = {p[0] for p in s["busy"]}
            over = s["foreign_cpu"] > MAX_FOREIGN_CPU
            instant = bool(hard) or any(r.startswith(("tools:", "thermal:")) for r in soft)
            new = "DISTURBED" if instant or (pids & prev_pids) or (over and prev_over) \
                else ("SPIKE" if (pids or over) else "QUIET")
            prev_pids, prev_over = pids, over
            iso = dt.datetime.fromtimestamp(s["t"]).astimezone().isoformat(timespec="seconds")
            f.write(f"{iso}\t{s['t']:.1f}\t{new}\t{s['load1']:.2f}\t{s['foreign_cpu']:.1f}\t"
                    f"{s['gpu_util']}\t{s['gpu_mem_gb']}\t{s['battery']}\t{s['tpl']}\t{tidy('; '.join(reasons))}\n")
            f.flush()
            if new != state:
                print(f"{iso[11:19]} {new}" + (f": {tidy('; '.join(reasons))}" if reasons else ""), flush=True)
                state = new
            time.sleep(max(0.5, a.interval - (time.time() - s["t"])))
    return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_log(path: Path) -> tuple[float, list[tuple[float, str, str]], str]:
    """(interval, rows, allowances): rows are (epoch, state, reasons); allowances is the header's
    comma-separated --allow list ("" without one)."""
    interval, rows, allowances = 10.0, [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            m = re.search(r"interval=([\d.]+)", line)
            if m:
                interval = float(m.group(1))
            m = re.search(r"allowances=(\S+)", line)
            if m:
                allowances = m.group(1)
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                rows.append((float(parts[1]), parts[2], parts[-1] if len(parts) > 5 else ""))
            except ValueError:
                continue
    return interval, rows, allowances


_BUSY = re.compile(r"([^\s,;]+)\((\d+)\) [\d.]+%")
_INSTANT = ("lock not held", f"port {PEER_PORT}", "power:", "battery:", "tools:", "thermal:")


def judge(rows: list[tuple[float, str, str]]) -> list[tuple[float, bool, str]]:
    """Re-judge logged samples from their reasons (the state column is informational).
    Lock, port, power, battery, tools and thermal count at once. CPU counts when the same foreign process
    is busy in consecutive samples, or foreign CPU is over its limit in consecutive samples."""
    out, prev_pids, prev_over = [], set(), False
    for t, _state, reasons in rows:
        segs = [x.strip() for x in reasons.split("; ") if x.strip()]
        pids = {m.group(2) for x in segs if x.startswith("busy: ") for m in _BUSY.finditer(x)}
        over = any(x.startswith("foreign CPU") for x in segs)
        bad = any(x.startswith(_INSTANT) for x in segs) or bool(pids & prev_pids) or (over and prev_over)
        out.append((t, bad, reasons))
        prev_pids, prev_over = pids, over
    return out


def cmd_verify(a: argparse.Namespace) -> int:
    interval, rows, allowances = read_log(Path(a.log))
    judged = judge(rows)
    note = f"; allowances {allowances}" if allowances else ""
    raw = Path(a.raw_dir)
    bad = 0
    for f in sorted(raw.glob("*.jsonl")):
        try:
            hdr = json.loads(f.open(encoding="utf-8").readline())
            start = dt.datetime.fromisoformat(hdr["created"]).timestamp()
        except (OSError, ValueError, KeyError):
            continue
        if start < a.since - 1:
            continue
        end = f.stat().st_mtime
        inside = [r for r in judged if start < r[0] <= end + interval]
        disturbed = [r for r in inside if r[1]]
        edges = [start] + [r[0] for r in inside] + [end]
        gap = max((b - x for x, b in zip(edges, edges[1:])), default=0.0)
        why = None
        if disturbed:
            why = f"{len(disturbed)} disturbed sample(s), first at " \
                  f"{dt.datetime.fromtimestamp(disturbed[0][0]).strftime('%H:%M:%S')}: {disturbed[0][2]}"
        elif gap > 3 * interval:
            why = f"monitor gap of {gap:.0f} s (interval {interval:.0f} s)"
        if why is None:
            print(f"verify: {f.name}: quiet ({len(inside)} samples, {end - start:.0f} s{note})")
            continue
        bad += 1
        print(f"verify: {f.name}: DISTURBED, {why}{note}")
        if a.move:
            dest = raw / "_unverified"
            dest.mkdir(exist_ok=True)
            for g in [f, f.with_suffix(".md")]:
                if g.exists():
                    g.rename(dest / g.name)
                    print(f"verify: moved {g.name} -> raw/_unverified/")
    return 1 if bad else 0


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
    c.add_argument("--allow", type=allow_names, action="append", default=[], help="M8 allowances: " + ", ".join(ALLOWANCES))
    m = sub.add_parser("monitor")
    m.add_argument("--log", required=True)
    m.add_argument("--interval", type=float, default=10.0)
    m.add_argument("--allow-pid", type=int, action="append", default=[])
    m.add_argument("--until-pid", type=int)
    m.add_argument("--allow", type=allow_names, action="append", default=[], help="M8 allowances: " + ", ".join(ALLOWANCES))
    v = sub.add_parser("verify")
    v.add_argument("--log", required=True)
    v.add_argument("--raw-dir", required=True)
    v.add_argument("--since", type=float, required=True)
    v.add_argument("--move", action="store_true")
    b = sub.add_parser("battery")
    b.add_argument("--min", type=float, required=True)
    b.add_argument("--wait", type=float, default=0.0)
    a = p.parse_args(argv)
    return {"check": cmd_check, "monitor": cmd_monitor, "verify": cmd_verify, "battery": cmd_battery}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
