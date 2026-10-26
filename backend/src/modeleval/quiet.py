"""Quiet-machine guard, battery rule and hold file (DESIGN.md §5.4-5.5, docs/CONTRACTS.md).

Stdlib only, on purpose: the capability runner imports `battery`, `boundary` and `hold_requested`
under its own Python 3.13 interpreter. The disturbance rule itself lives in scripts/quietlib.py
(owned by scripts), so the monitor that writes monitor.tsv and the judge that reads it can never
disagree; this module only drives it:

- `QuietGuard.start_monitor` runs `quietlib.py monitor` for the whole launch (5 s samples);
- `QuietGuard.wait_quiet` runs `quietlib.py check --major-only` in short settle rounds before every
  timed unit, printing UNIT-WAIT with the current reasons, and gives up after `settle_s` so that a
  lasting disturbance stops the speed leg instead of burning the unit's attempts. Only major
  conditions are waited for: a minor one would only flag the unit (DESIGN §5.4);
- `QuietGuard.judge` waits until the monitor has sampled past `t_end + 5 s` and asks
  `quietlib.judge_window` for the verdict over `[t_start - 5 s, t_end + 5 s]`.

Two levels (DESIGN §5.4, the user's instruction of 2026-10-26): a verdict's `severity` is "major"
(run.py repeats the unit), "minor" (the unit is kept with its flags) or "clean" (no flag at all).

Expected quietlib interface (scripts/quietlib.py, imported by path):

    judge_window(log: Path, t_start: float, t_end: float, margin: float = 5.0) -> dict
        {"clean": bool, "reasons": [str], "peak_foreign_cpu": float, "samples": int,
         optional "disturbed_samples": int, "severity": "clean" | "minor" | "major",
         "major_reasons": [str], "minor_reasons": [str], "major_samples": int, "minor_samples": int}
        (without "severity", a verdict that is not clean counts as major: the one-level rule)
    MONITOR_COLUMNS: tuple of the TSV column names (the header line `# iso<TAB>epoch...` repeats them)
"""

from __future__ import annotations

import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from modeleval import paths, progress

QUIET_PYTHON = "/usr/bin/python3"   # quietlib is stdlib only and always runs isolated, without bytecode (-I -B)
MONITOR_INTERVAL = 5.0
MARGIN = 5.0                        # a unit owns the samples in [t_start - 5 s, t_end + 5 s]
FIRST_ROUND_S = 1                   # quick first check, so UNIT-WAIT appears at once
ROUND_S = 60                        # later settle rounds; quietlib re-checks every 5 s inside one
PAUSE_PCT = {"unit": 50, "long-unit": 60, "harness-block": 60, "task": 50}
STOP_PCT = 20                       # hard stop for capability tasks (macOS caps power near 7 %)
RESUME_PCT = 80                     # the user's rule: wait for 80 % after a battery pause
UNIT_MIN_PCT = 40                   # speed validity guard at a unit's start (DESIGN §5.5)
LAUNCH_MIN_PCT = 60                 # battery at the timed pre-launch check (DESIGN §5 step 2)
# DESIGN §5.4: pmset thermal warnings are logged, not judged; the checks record them as "allowed thermal: ..."
CHECK_ALLOW = ("thermal",)

Boundary = Literal["unit", "long-unit", "harness-block", "task"]


SEVERITIES = ("clean", "minor", "major")


@dataclass
class UnitVerdict:
    """quietlib's verdict on one window. `clean`: no flag at all; `severity`: "major" repeats a timed
    unit, "minor" keeps it with the flags. A verdict built as not clean without a severity (no
    monitor log, an old judge) is major, as under the one-level rule."""
    clean: bool
    reasons: list[str]
    peak_foreign_cpu: float
    samples: int
    disturbed_samples: int = 0
    battery_min: int | None = None
    severity: str = "clean"
    major_reasons: list[str] = field(default_factory=list)
    minor_reasons: list[str] = field(default_factory=list)
    major_samples: int = 0
    minor_samples: int = 0

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES or (self.severity == "clean" and not self.clean):
            self.severity = "clean" if self.clean else "major"
        if self.severity == "major" and not self.major_reasons:
            self.major_reasons = list(self.reasons)

    @property
    def major(self) -> bool:
        return self.severity == "major"


@dataclass
class CheckResult:
    """One `quietlib.py check` run: ok, whether the failure was hard (lock, port, power), the reasons."""
    ok: bool
    hard: bool = False
    reasons: list[str] = field(default_factory=list)
    text: str = ""


# ---------------------------------------------------------------- battery, boundaries, hold


def _pmset_batt() -> str:
    try:
        return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_battery(text: str) -> tuple[int, bool]:
    """(percent, on AC) from `pmset -g batt`; percent -1 on a machine without a battery."""
    m = re.search(r"\t(\d+)%;", text)
    return (int(m.group(1)) if m else -1), ("'AC Power'" in text or not m)


def battery() -> tuple[int, bool]:
    """(percent, on_ac). The 96 W adapter cannot carry this machine under load, so it drains anyway."""
    return parse_battery(_pmset_batt())


def boundary(kind: Boundary) -> Literal["go", "pause", "stop"]:
    """The user's battery rule at a unit, task or leg boundary (DESIGN §5.5).

    Below 50 % (60 % before a long unit or a harness block) the run yields: stop the server,
    release the lock and wait for 80 %. Below 20 % a capability task does not start at all.
    """
    pct, _ = battery()
    if pct < 0:
        return "go"
    if kind == "task" and pct < STOP_PCT:
        return "stop"
    return "pause" if pct < PAUSE_PCT[kind] else "go"


def hold_requested() -> bool:
    """The hold file (REPO/runs/hold) makes the run wait at the next boundary, keeping the lock."""
    return paths.HOLD_FILE.exists()


# ---------------------------------------------------------------- quietlib


def quietlib_path() -> Path:
    return paths.SCRIPTS / "quietlib.py"


_QL_CACHE: dict[tuple[str, float], object] = {}


def load_quietlib():
    """scripts/quietlib.py as a module (it is not a package); reloaded when the file changes."""
    p = quietlib_path()
    key = (str(p), p.stat().st_mtime)
    if key not in _QL_CACHE:
        spec = importlib.util.spec_from_file_location("modeleval_quietlib", p)
        mod = importlib.util.module_from_spec(spec)
        saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = saved
        _QL_CACHE.clear()
        _QL_CACHE[key] = mod
    return _QL_CACHE[key]


def _settle_reasons(text: str) -> tuple[bool, list[str]]:
    """(hard, reasons) from a check report's settle line."""
    m = re.search(r"^settle: not attempted \(hard failure: (.*)\)\s*$", text, re.M)
    if m:
        return True, [r.strip() for r in m.group(1).split("; ") if r.strip()]
    m = re.search(r"^settle: still not quiet after .*?\): (.*)$", text, re.M)
    if m:
        return False, [r.strip() for r in m.group(1).split("; ") if r.strip()]
    return False, []


def run_check(allow_pids: list[int], settle_s: int, max_gpu_gb: float | None = None,
              min_battery: float | None = None, log: Path | None = None,
              allow: tuple[str, ...] = CHECK_ALLOW, major_only: bool = False) -> CheckResult:
    """One `quietlib.py check` (settle_s >= 1 so that the report names its reasons). major_only: wait
    only for major conditions; minor ones are printed in the report and do not fail it."""
    cmd = [QUIET_PYTHON, "-I", "-B", str(quietlib_path()), "check", "--settle", str(max(1, int(settle_s)))]
    if major_only:
        cmd.append("--major-only")
    for pid in allow_pids:
        cmd += ["--allow-pid", str(pid)]
    if max_gpu_gb is not None:
        cmd += ["--max-gpu-gb", str(max_gpu_gb)]
    if min_battery is not None:
        cmd += ["--min-battery", str(min_battery)]
    if allow:
        cmd += ["--allow", ",".join(allow)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1, settle_s) + 120)
        rc, text = p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        rc, text = 1, f"quietlib check failed to run: {e}"
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(text.replace(str(paths.HOME), "~") + "\n")
    if rc == 0:
        return CheckResult(True, text=text)
    hard, reasons = _settle_reasons(text)
    gpu = gpu_memory_gb(text)
    if hard and max_gpu_gb is not None and gpu is not None and gpu > max_gpu_gb:
        # a hard failure ends the check before the soft reasons are listed: keep a resident model visible
        reasons.append(f"GPU memory in use {gpu:g} GB > {max_gpu_gb:g} GB (another model resident?)")
    return CheckResult(False, hard, reasons or [text.strip().splitlines()[-1] if text.strip() else f"exit {rc}"], text)


def gpu_memory_gb(text: str) -> float | None:
    """GPU memory in use from a check report's `gpu: N% busy (max M), G GB in use` line."""
    m = re.search(r"^gpu: .*?([\d.]+) GB in use", text, re.M)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def wait_check(unit: str, allow_pids: list[int], settle_s: int = 600, log: Path | None = None,
               **check_kw) -> CheckResult:
    """Repeat the check until quiet, a hard failure, or settle_s; UNIT-WAIT on every new reason."""
    t0 = time.time()
    last: list[str] | None = None
    first = True
    while True:
        left = settle_s - (time.time() - t0)
        r = run_check(allow_pids, FIRST_ROUND_S if first else int(min(ROUND_S, max(1, left))), log=log, **check_kw)
        first = False
        if r.ok:
            return r
        if r.reasons != last:
            progress.emit("UNIT-WAIT", unit=unit, reason="; ".join(r.reasons)[:200])
            last = r.reasons
        if r.hard or time.time() - t0 >= settle_s:
            return r


# Reasons that refuse a launch even for untimed work: a foreign model resident, the lock, the peer's
# server, power. Load, busy processes and tools only matter for timed units.
LAUNCH_BLOCKERS = ("GPU memory in use", "lock", "port 8500", "power:", "battery:")


def prelaunch_check(timed: bool, settle_s: int = 600, log: Path | None = None) -> CheckResult:
    """Before a launch (DESIGN §5 step 2): no foreign model resident (--max-gpu-gb 6), lock ours, AC.

    For a timed leg no major condition may hold (minor ones are printed), with up to settle_s to settle, and the battery
    must be at 60 %. For untimed work (capability only, --unverified) one sample is taken and only
    LAUNCH_BLOCKERS refuse; a resident foreign model refuses even when a hard failure (a peer port)
    cut the check short.
    """
    if timed:
        # Major conditions only, like every pre-unit check: a minor one would only flag a unit
        # (the user, 2026-10-26: small disturbances are skipped, not re-run).
        return wait_check("launch", [os.getpid()], settle_s, log=log, max_gpu_gb=6, min_battery=LAUNCH_MIN_PCT,
                          major_only=True)
    r = run_check([os.getpid()], 1, max_gpu_gb=6, log=log)
    blockers = [x for x in r.reasons if x.startswith(LAUNCH_BLOCKERS)]
    return CheckResult(r.ok or not blockers, r.hard, blockers or r.reasons, r.text)


def _kill_group(proc: subprocess.Popen, sig: int = signal.SIGTERM, wait_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
        proc.wait(timeout=wait_s)
    except (ProcessLookupError, PermissionError):
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            pass


# ---------------------------------------------------------------- monitor.tsv


def read_monitor(path: Path) -> tuple[float, list[dict]]:
    """(interval, rows) from monitor.tsv; columns by the header's names (several runs may append)."""
    interval, cols, rows = MONITOR_INTERVAL, None, []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return interval, rows
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.search(r"interval=([\d.]+)", line)
            if m:
                interval = float(m.group(1))
            if "\tepoch\t" in line or line.startswith("# iso\t"):
                cols = line[1:].strip().split("\t")
            continue
        if not cols or not line.strip():
            continue
        parts = line.split("\t")
        row = dict(zip(cols, parts))
        try:
            row["epoch"] = float(row["epoch"])
        except (KeyError, ValueError):
            continue
        rows.append(row)
    return interval, rows


def _num(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_verdict(res: object) -> UnitVerdict:
    """judge_window's result as a UnitVerdict (dict, object with attributes, or 4-tuple). Without a
    severity, a verdict that is not clean is major (UnitVerdict.__post_init__)."""
    if isinstance(res, tuple):
        clean, reasons, peak, samples = res[:4]
        return UnitVerdict(bool(clean), list(reasons), float(peak), int(samples))
    get = res.get if isinstance(res, dict) else (lambda k, d=None: getattr(res, k, d))
    return UnitVerdict(bool(get("clean")), list(get("reasons", []) or []), float(get("peak_foreign_cpu", 0.0) or 0.0),
                       int(get("samples", 0) or 0), int(get("disturbed_samples", 0) or 0),
                       severity=str(get("severity", "") or ""),
                       major_reasons=list(get("major_reasons", []) or []),
                       minor_reasons=list(get("minor_reasons", []) or []),
                       major_samples=int(get("major_samples", 0) or 0),
                       minor_samples=int(get("minor_samples", 0) or 0))


class QuietGuard:
    """The quiet rule for one launch: monitor process, pre-unit checks, unit verdicts."""

    def __init__(self, run_dir: Path, allow_pids: list[int], pidfile=None):
        self.run_dir = Path(run_dir)
        # The orchestrator renders and tokenizes prompts in-process, and the capability runner's whole
        # tree (agents, proxy, checkers) descends from it: always exempt, whoever the caller is.
        self.allow_pids = sorted({p for p in allow_pids if p and p > 0} | {os.getpid()})
        self.tsv = self.run_dir / "speed" / "monitor.tsv"
        self.checks_log = self.run_dir / "speed" / "quiet-checks.log"
        self.pidfile = pidfile          # anything with add(pgid, role, marker) and remove(pgid)
        self.proc: subprocess.Popen | None = None
        self.cover_timeout = 3 * MONITOR_INTERVAL + 30

    def start_monitor(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.tsv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [QUIET_PYTHON, "-I", "-B", str(quietlib_path()), "monitor", "--log", str(self.tsv),
               "--interval", f"{MONITOR_INTERVAL:g}"]
        for pid in self.allow_pids:
            cmd += ["--allow-pid", str(pid)]
        out = open(self.run_dir / "speed" / "monitor.log", "a", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        finally:
            out.close()
        if self.pidfile is not None:
            self.pidfile.add(self.proc.pid, "monitor", str(quietlib_path()))

    def monitor_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop_monitor(self) -> None:
        if self.proc is None:
            return
        _kill_group(self.proc)
        if self.pidfile is not None:
            self.pidfile.remove(self.proc.pid)
        self.proc = None

    def wait_quiet(self, unit: str, settle_s: int = 600) -> bool:
        """Pre-unit check with our server and this process exempt, and the 40 % validity guard at a
        unit's start (DESIGN §5.5); False = still not quiet, give up. It waits for major conditions
        only (`--major-only`): a minor one would only flag the unit, never repeat it (DESIGN §5.4)."""
        return wait_check(unit, self.allow_pids, settle_s, log=self.checks_log, min_battery=UNIT_MIN_PCT,
                          major_only=True).ok

    def _wait_covered(self, t_end: float) -> None:
        deadline = time.time() + self.cover_timeout
        while time.time() < deadline:
            _, rows = read_monitor(self.tsv)
            if rows and rows[-1]["epoch"] >= t_end + MARGIN:
                return
            if not self.monitor_alive():
                return
            time.sleep(0.5)

    def _battery_min(self, t_start: float, t_end: float) -> int | None:
        _, rows = read_monitor(self.tsv)
        vals = [_num(r.get("battery")) for r in rows if t_start - MARGIN <= r["epoch"] <= t_end + MARGIN]
        vals = [int(v) for v in vals if v is not None and v >= 0]
        return min(vals) if vals else None

    def judge(self, t_start: float, t_end: float) -> UnitVerdict:
        """quietlib's verdict over [t_start - 5 s, t_end + 5 s]. The rule, and the coverage rule (no
        samples or a sampling gap is never clean), live in quietlib.judge_window only."""
        self._wait_covered(t_end)
        ql = load_quietlib()
        if not hasattr(ql, "judge_window"):
            raise RuntimeError(f"{paths.tilde(quietlib_path())} has no judge_window(); "
                               "the modified quietlib of docs/CONTRACTS.md is required")
        if not self.tsv.exists():
            return UnitVerdict(False, ["no monitor log"], 0.0, 0)
        v = _as_verdict(ql.judge_window(self.tsv, t_start, t_end, MARGIN))
        v.battery_min = self._battery_min(t_start, t_end)
        return v

    def flags(self, t_start: float, t_end: float) -> dict:
        """Disturbance flags for an untimed window (a capability task): never repeats anything.
        `disturbed` is any flag (not clean); `severity` and the major/minor counts tell them apart."""
        try:
            v = self.judge(t_start, t_end)
        except (OSError, RuntimeError) as e:
            return {"disturbed": None, "reasons": [f"no verdict: {e}"], "battery_min": None}
        return {"disturbed": not v.clean, "severity": v.severity, "disturbed_samples": v.disturbed_samples,
                "major_samples": v.major_samples, "minor_samples": v.minor_samples, "reasons": v.reasons,
                "peak_foreign_cpu": v.peak_foreign_cpu, "samples": v.samples, "battery_min": v.battery_min}
