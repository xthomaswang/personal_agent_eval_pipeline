"""Fixtures for the tests of scripts/: the stdlib scripts are loaded by path (they are never
installed), every probe of quietlib goes to a fake machine, and the lock lives in tmp_path."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
SYSTEM_PY = Path("/usr/bin/python3")   # what quiet_check.sh and the pre-commit hook run (3.9 here)
BASH = "/bin/bash"                     # macOS bash 3.2, as gpu_lock.sh runs
HOLDER = "local_ai-benchmark session"
# Sensitive-looking strings are assembled at run time, so these files pass redact.py --check.
USERS_DIR = "/Us" + "ers/"


def load_script(name: str) -> ModuleType:
    """A fresh module object per call, so monkeypatched attributes never leak between tests."""
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def _in_spec(port: int, spec: str) -> bool:
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        if int(lo) <= port <= int(hi or lo):
            return True
    return False


class FakeMachine:
    """Answers quietlib's probes (ps, sysctl, pmset, ioreg, vm_stat, lsof, notifyutil) from
    attributes the test sets. Records every command it was asked."""

    def __init__(self) -> None:
        # (pid, ppid, %cpu, rss KB, comm)
        self.procs: list[tuple[int, int, float, int, str]] = [
            (1, 0, 0.2, 30_000, "/sbin/launchd"),
            (613, 1, 3.0, 200_000, "/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer"),
            (0, 0, 5.0, 1_000, "kernel_task"),
            (100, 1, 80.0, 30_000_000, "/opt/omlx/.venv/bin/python"),     # the server under test
            (101, 100, 40.0, 500_000, "/opt/omlx/.venv/bin/python"),      # its engine worker
            (300, 1, 5.0, 400_000, "/usr/libexec/mds_stores"),
        ]
        self.cmdlines: dict[int, str] = {}
        # client id -> (pid, name, accumulated ns); a real machine always has WindowServer's
        self.gpu: dict[str, tuple[int, str, int]] = {"0x1": (613, "WindowServer", 1_000_000)}
        self.util = 0
        self.gpu_mem = 900_000_000
        self.swapouts: int | None = 1000
        self.page = 16384
        self.pressure = 1
        self.listen: list[tuple[int, int]] = []        # (port, pid)
        self.battery = 100
        self.source = "AC Power"
        self.powermode = "2"
        self.therm = "Note: No thermal warning level has been recorded\n"
        self.tpl = 0
        self.load = 1.5
        self.calls: list[list[str]] = []

    def ioreg_text(self) -> str:
        lines = [
            "+-o AGXAcceleratorG17X  <class AGXAcceleratorG17X, id 0x1000008df, registered, matched>",
            "  | {",
            f'  |   "PerformanceStatistics" = {{"In use system memory (driver)"=0,"Device Utilization %"={self.util},'
            f'"In use system memory"={self.gpu_mem}}}',
            "  | }",
        ]
        for cid, (pid, cname, ns) in self.gpu.items():
            usage = f'({{"API"="Metal","lastSubmittedTime"=1,"accumulatedGPUTime"={ns}}})' if ns else "()"
            lines += [
                f"  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id {cid}, !registered, !matched, active>",
                "  |   {",
                f'  |     "AppUsage" = {usage}',
                f'  |     "IOUserClientCreator" = "pid {pid}, {cname}"',
                "  |   }",
            ]
        return "\n".join(lines) + "\n"

    def __call__(self, cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
        self.calls.append(list(cmd))
        c = " ".join(cmd)
        if cmd[0] == "ps" and "pcpu" in c:
            return 0, "".join(f"{p} {pp} {cpu} {rss} {comm}\n" for p, pp, cpu, rss, comm in self.procs)
        if cmd[0] == "ps":
            return 0, "".join(f"{p} {self.cmdlines.get(p, comm)}\n" for p, _, _, _, comm in self.procs)
        if c == "sysctl -n vm.loadavg":
            return 0, f"{{ {self.load:.2f} 1.40 1.30 }}\n"
        if c == "sysctl -n kern.memorystatus_vm_pressure_level":
            return 0, f"{self.pressure}\n"
        if c == "pmset -g batt":
            return 0, (f"Now drawing from '{self.source}'\n -InternalBattery-0 (id=1)\t{self.battery}%; "
                       "charging; 0:30 remaining present: true\n")
        if c == "pmset -g":
            return 0, f"System-wide power settings:\nCurrently in use:\n powermode            {self.powermode}\n"
        if c == "pmset -g therm":
            return 0, self.therm
        if cmd[0] == "notifyutil":
            return 0, f"{self.tpl}\n"
        if cmd[0] == "ioreg":
            return 0, self.ioreg_text()
        if cmd[0] == "vm_stat":
            swap = f"Swapouts:                                {self.swapouts}.\n" if self.swapouts is not None else ""
            return 0, (f"Mach Virtual Memory Statistics: (page size of {self.page} bytes)\n"
                       f"Pages free:                               100.\n{swap}")
        if cmd[0] == "lsof":
            spec = next(a.split(":", 1)[1] for a in cmd if a.startswith("-iTCP:"))
            hits = [(port, pid) for port, pid in self.listen if _in_spec(port, spec)]
            return (0 if hits else 1), "".join(f"p{pid}\nf3\nn*:{port}\n" for port, pid in hits)
        raise AssertionError(f"unexpected command: {cmd}")


@pytest.fixture
def quietlib(monkeypatch, tmp_path):
    """quietlib with its lock path in tmp_path (no lock file yet)."""
    mod = load_script("quietlib")
    monkeypatch.setattr(mod, "LOCK", str(tmp_path / ".gpu-lock"))
    return mod


@pytest.fixture
def machine(quietlib, monkeypatch, tmp_path) -> FakeMachine:
    """A quiet fake machine wired into quietlib, with our lock held."""
    m = FakeMachine()
    monkeypatch.setattr(quietlib, "run", m)
    (tmp_path / ".gpu-lock").write_text(f"holder: {HOLDER}\npurpose: test (timed)\n", encoding="utf-8")
    return m


@pytest.fixture
def redact():
    return load_script("redact")


@pytest.fixture
def fake_bin(tmp_path) -> Path:
    """A directory with a fake `lsof` that reports the listeners in $FAKE_LISTENERS
    ("port:pid ..."), filtered by the -iTCP:<spec> argument like the real one."""
    d = tmp_path / "fakebin"
    d.mkdir()
    lsof = d / "lsof"
    lsof.write_text(
        "#!/bin/sh\n"
        "spec=''\n"
        "for a in \"$@\"; do case \"$a\" in -iTCP:*) spec=${a#-iTCP:} ;; esac; done\n"
        "found=1\n"
        "for l in ${FAKE_LISTENERS:-}; do\n"
        "  port=${l%%:*}; pid=${l#*:}\n"
        "  old_ifs=$IFS; IFS=,\n"
        "  for s in $spec; do\n"
        "    case \"$s\" in *-*) lo=${s%-*}; hi=${s#*-} ;; *) lo=$s; hi=$s ;; esac\n"
        "    if [ \"$port\" -ge \"$lo\" ] && [ \"$port\" -le \"$hi\" ]; then\n"
        "      printf 'p%s\\nf3\\nn*:%s\\n' \"$pid\" \"$port\"; found=0\n"
        "    fi\n"
        "  done\n"
        "  IFS=$old_ifs\n"
        "done\n"
        "exit $found\n",
        encoding="utf-8")
    lsof.chmod(0o755)
    return d


@pytest.fixture
def lock_env(tmp_path, fake_bin) -> dict[str, str]:
    """Environment for gpu_lock.sh: lock in tmp_path, fake lsof first on PATH, no run id."""
    env = {k: v for k, v in os.environ.items() if k not in ("MODELEVAL_RUN_ID", "FAKE_LISTENERS")}
    env["GPU_LOCK_PATH"] = str(tmp_path / ".gpu-lock")
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '/usr/bin:/bin')}"
    return env
