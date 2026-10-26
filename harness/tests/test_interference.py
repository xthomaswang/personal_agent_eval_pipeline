"""M8 tooling outside the harness: scripts/interference.py (run as in production, with the macOS
/usr/bin/python3 -I), the quietlib allowances, and the run_m8.sh plan. Loads stay tiny: a fraction of
a CPU second, at most 32 MB of memory or 16 MiB of disk. The GPU load and pmset run against fakes;
nothing here runs sudo or touches the GPU."""

import datetime
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PY = "/usr/bin/python3"
pytestmark = pytest.mark.skipif(sys.platform != "darwin" or not Path(PY).exists(), reason="macOS tooling")


def lf(*args, env=None, timeout=30) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-I", str(SCRIPTS / "interference.py"), *map(str, args)], capture_output=True,
                          text=True, timeout=timeout, env=env)


def start_bg(tmp: Path, scenario: str, *args, env=None) -> tuple[subprocess.Popen, Path]:
    pf = tmp / "load.pid"
    log = (tmp / "load.log").open("w")
    proc = subprocess.Popen([PY, "-I", str(SCRIPTS / "interference.py"), "start", scenario, "--pidfile", str(pf),
                             *map(str, args)], stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, pf


def wait_ready(proc: subprocess.Popen, pf: Path, timeout: float = 10, env=None) -> dict:
    r = lf("status", "--pidfile", pf, "--wait-ready", timeout, "--pid", proc.pid, env=env)
    assert r.returncode == 0, r.stdout + r.stderr + (pf.parent / "load.log").read_text()
    return json.loads(pf.read_text())


def children(pid: int) -> set[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True).stdout
    return {int(a) for a, b in (ln.split() for ln in out.splitlines() if len(ln.split()) == 2) if int(b) == pid}


def gone(pid: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not st or st.startswith("Z"):
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------- interference.py
def test_parameters_allowances_and_meta():
    r = lf("start", "cpuall", "--dry-run")
    assert r.returncode == 0 and json.loads(r.stdout) == {
        "scenario": "cpuall", "params": {"threads": os.cpu_count(), "duty": 1.0, "period": 0.1}, "allow": ["load", "thermal"]}
    assert json.loads(lf("start", "cpuhalf", "--dry-run").stdout)["params"]["threads"] == 8
    gpu = json.loads(lf("start", "gpu", "--dry-run", "--duty", "0.3").stdout)
    assert gpu["params"]["duty"] == 0.3 and gpu["allow"] == ["gpu", "load", "thermal"]
    assert json.loads(lf("start", "mem", "--dry-run").stdout)["params"]["gb"] == 40.0
    bad = lf("start", "cpu1", "--gb", "1")
    assert bad.returncode != 0 and "--gb does not apply to cpu1" in bad.stderr
    assert lf("start", "cpu1", "--duty", "1.5", "--dry-run").returncode != 0
    assert lf("allowances", "power").stdout.strip() == "load,lowpower"
    assert lf("allowances", "base").stdout.strip() == ""
    meta = json.loads(lf("meta", "--scenario", "base", "run=20261020-230000", "block=3", "base_tag=4bit-mtp-on").stdout)
    assert meta == {"scenario": "base", "params": {}, "allow": [], "load_pid": None, "run": "20261020-230000",
                    "block": 3, "base_tag": "4bit-mtp-on"}


def test_cpu_load_is_one_process_tree_and_stops_cleanly(tmp_path):
    proc, pf = start_bg(tmp_path, "cpu1", "--threads", 2, "--duty", 0.2, "--period", 0.05)
    try:
        rec = wait_ready(proc, pf)
        assert rec["pid"] == proc.pid and rec["state"] == "ready" and rec["allow"] == ["load"]
        assert rec["params"] == {"threads": 2, "duty": 0.2, "period": 0.05}
        workers = set(rec["children"])
        assert len(workers) == 2 and children(proc.pid) == workers   # a single root, its workers below it
        meta = json.loads(lf("meta", "--pidfile", pf, "block=2").stdout)
        assert meta["load_pid"] == proc.pid and meta["scenario"] == "cpu1" and meta["block"] == 2 and meta["ready"]
        assert lf("meta", "--pidfile", pf, "--field", "load_pid").stdout.strip() == str(proc.pid)
        r = lf("stop", "--pidfile", pf)
        assert r.returncode == 0 and "stopped" in r.stdout
        assert proc.wait(timeout=5) == 0
        assert all(gone(w) for w in workers) and not pf.exists()
        assert "stopped by SIGTERM" in (tmp_path / "load.log").read_text()
    finally:
        proc.kill()


def test_signals_and_a_killed_root_leave_no_orphans(tmp_path):
    proc, pf = start_bg(tmp_path, "cpu1", "--duty", 0.1)
    rec = wait_ready(proc, pf)
    os.kill(proc.pid, signal.SIGINT)   # a terminal Ctrl-C
    assert proc.wait(timeout=5) == 0 and gone(rec["children"][0]) and not pf.exists()

    proc, pf = start_bg(tmp_path, "cpu1", "--duty", 0.1)
    rec = wait_ready(proc, pf)
    os.kill(proc.pid, signal.SIGKILL)   # no cleanup at all: the worker notices its root is gone
    proc.wait(timeout=5)
    assert gone(rec["children"][0], timeout=3)
    st = lf("status", "--pidfile", pf)
    assert st.returncode == 1 and "not running" in st.stdout
    assert lf("stop", "--pidfile", pf).returncode == 0 and not pf.exists()


def test_until_pid_and_duration_end_the_load(tmp_path):
    sleeper = subprocess.Popen(["sleep", "30"])
    proc, pf = start_bg(tmp_path, "cpu1", "--duty", 0.1, "--until-pid", sleeper.pid)
    try:
        rec = wait_ready(proc, pf)
        sleeper.kill()
        sleeper.wait()
        assert proc.wait(timeout=5) == 0 and gone(rec["children"][0]) and not pf.exists()
        assert f"process {sleeper.pid} (--until-pid) exited" in (tmp_path / "load.log").read_text()
    finally:
        sleeper.kill()
        proc.kill()
    r = lf("start", "cpu1", "--duty", 0.1, "--duration", 0.3, "--pidfile", tmp_path / "d.pid")
    assert r.returncode == 0 and "duration elapsed" in r.stdout and not (tmp_path / "d.pid").exists()


def test_second_start_is_refused_and_stale_leftovers_are_cleaned(tmp_path):
    proc, pf = start_bg(tmp_path, "cpu1", "--duty", 0.1)
    try:
        wait_ready(proc, pf)
        r = lf("start", "cpu1", "--pidfile", pf)
        assert r.returncode == 2 and "already running" in r.stderr
        assert lf("stop", "--pidfile", pf).returncode == 0
    finally:
        proc.kill()
    dead = subprocess.Popen(["true"])
    dead.wait()
    left = tmp_path / "left.bin"
    left.write_bytes(b"x")
    pf.write_text(json.dumps({"pid": dead.pid, "scenario": "disk", "state": "ready", "children": [], "files": [str(left)]}))
    r = lf("start", "cpu1", "--duty", 0.1, "--duration", 0.2, "--pidfile", pf)
    assert r.returncode == 0 and "removed leftover file" in r.stdout and not left.exists()


def test_mem_load_is_resident(tmp_path):
    proc, pf = start_bg(tmp_path, "mem", "--gb", 0.032, "--retouch", 0.05)
    try:
        wait_ready(proc, pf)
        rss_kb = int(subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True).stdout)
        assert rss_kb * 1024 >= 30e6   # 32 MB filled with random bytes: every page touched
        time.sleep(0.15)   # a few re-reads
        assert lf("stop", "--pidfile", pf).returncode == 0 and proc.wait(timeout=5) == 0
    finally:
        proc.kill()
    r = lf("start", "mem", "--gb", 1e6, "--pidfile", tmp_path / "big.pid")   # more than the machine has
    assert r.returncode == 3 and "GB is available" in r.stdout
    assert lf("status", "--pidfile", tmp_path / "big.pid").returncode == 1


def test_disk_load_writes_reads_and_deletes_its_file(tmp_path):
    d = tmp_path / "io.noindex"
    proc, pf = start_bg(tmp_path, "disk", "--dir", d, "--file-mb", 16, "--block-mb", 1, "--write-mbps", 40,
                        "--read-mbps", 40)
    try:
        rec = wait_ready(proc, pf)
        f = Path(rec["files"][0])
        assert f.parent == d and f.exists()
        time.sleep(0.4)
        assert f.stat().st_size >= 4 << 20   # writes keep coming at ~40 MB/s
        assert lf("stop", "--pidfile", pf).returncode == 0 and proc.wait(timeout=5) == 0
        assert not f.exists() and not d.exists()
        log = (tmp_path / "load.log").read_text()
        assert "disk wrote" in log and "read 0.00 GB" not in log
    finally:
        proc.kill()


def fake_bin(tmp: Path, name: str, body: str) -> Path:
    p = tmp / "bin" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return p


def test_gpu_root_supervises_its_worker(tmp_path):
    """The real worker needs MLX and the GPU; a fake interpreter stands in for it."""
    ok = fake_bin(tmp_path, "gpu-ok", "echo ready\nexec sleep 30\n")
    proc, pf = start_bg(tmp_path, "gpu", "--python", ok)
    try:
        rec = wait_ready(proc, pf)
        assert children(proc.pid) == set(rec["children"]) and len(rec["children"]) == 1
        assert lf("stop", "--pidfile", pf).returncode == 0 and proc.wait(timeout=5) == 0
        assert gone(rec["children"][0])
    finally:
        proc.kill()
    bad = fake_bin(tmp_path, "gpu-bad", "exit 3\n")
    r = lf("start", "gpu", "--python", bad, "--pidfile", pf)
    assert r.returncode == 3 and "before it was ready" in r.stdout
    st = lf("status", "--pidfile", pf)
    assert st.returncode == 1 and "exited: GPU worker" in st.stdout
    assert lf("stop", "--pidfile", pf).returncode == 0 and not pf.exists()


@pytest.fixture
def fake_pmset(tmp_path):
    """pmset reads its energy mode from a file; sudo only records that it was called."""
    mode = tmp_path / "mode"
    mode.write_text("2")
    fake_bin(tmp_path, "pmset", f"""mode=$(cat {mode})
case "$*" in
  "-g batt") printf "Now drawing from 'AC Power'\\n -InternalBattery-0 (id=1)\\t90%%; charged; 0:00 remaining present: true\\n" ;;
  "-g custom") printf "Battery Power:\\n powermode            0\\nAC Power:\\n powermode            2\\n" ;;
  "-g") printf "System-wide power settings:\\nCurrently in use:\\n powermode            %s\\n" "$mode" ;;
  *) exit 1 ;;
esac
""")
    fake_bin(tmp_path, "sudo", f"echo \"$*\" >> {tmp_path / 'sudo-called'}\nexit 1\n")
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    yield mode, env
    assert not (tmp_path / "sudo-called").exists()


def test_power_waits_for_the_user_and_fails_when_low_power_goes_off(tmp_path, fake_pmset):
    mode, env = fake_pmset
    proc, pf = start_bg(tmp_path, "power", "--wait", 20, "--poll", 0.05, env=env)
    try:
        st = lf("status", "--pidfile", pf, "--wait-ready", 1, "--pid", proc.pid, env=env)
        assert st.returncode == 1 and "ACTION NEEDED: run in a terminal: sudo pmset -a lowpowermode 1" in st.stdout
        mode.write_text("1")   # the user ran the command
        rec = wait_ready(proc, pf, env=env)
        assert rec["power_before"] == {"Battery Power": {"powermode": "0"}, "AC Power": {"powermode": "2"}}
        assert rec["restore"] == "sudo pmset -a lowpowermode 0 && sudo pmset -b powermode 0 && sudo pmset -c powermode 2"
        mode.write_text("2")   # turned off before the block ended
        assert proc.wait(timeout=5) == 3
        st = lf("status", "--pidfile", pf, env=env)
        assert st.returncode == 1 and "Low Power Mode went off during the scenario" in st.stdout
        r = lf("stop", "--pidfile", pf, env=env)
        assert "power restore: sudo pmset -a lowpowermode 0" in r.stdout and not pf.exists()
    finally:
        proc.kill()
    w = lf("wait-power", "--want", "high", "--wait", 1, env=env)
    assert w.returncode == 0 and "(high: ok)" in w.stdout
    mode.write_text("1")
    w = lf("wait-power", "--want", "high", "--wait", 0.3, "--poll", 0.1, "--hint", "sudo pmset -c powermode 2", env=env)
    assert w.returncode == 1 and "ACTION NEEDED" in w.stdout and "sudo pmset -c powermode 2" in w.stdout
    r = lf("start", "power", "--duration", 0.2, "--poll", 0.05, "--pidfile", tmp_path / "p2.pid", env=env)
    assert r.returncode == 0 and "ACTION NEEDED" not in r.stdout   # Low Power Mode already on: ready at once
    assert "power ready" in r.stdout and "power restore: sudo pmset -a lowpowermode 0" in r.stdout


# ---------------------------------------------------------------- quietlib allowances
@pytest.fixture(scope="module")
def quietlib():
    spec = importlib.util.spec_from_file_location("quietlib_under_test", SCRIPTS / "quietlib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample(**over) -> dict:
    s = {"t": time.time(), "lock": "test", "peer_port": False, "load1": 1.0, "gpu_util": 0, "gpu_mem_gb": 1.0,
         "foreign_cpu": 10.0, "busy": [], "top": [], "tools": [], "thermal": [], "power_ok": True,
         "power": "AC Power; powermode 2 (High Power)", "low_power_ac": False, "battery": 90,
         "battery_state": "charging", "tpl": 0}
    s.update(over)
    return s


def test_allowances_record_instead_of_fail(quietlib):
    hot = sample(load1=9.8, gpu_util=90, thermal=["CPU_Speed_Limit = 80"])
    hard, soft, allowed = quietlib.problems(hot, judge_load=True)
    assert not hard and len(soft) == 3 and not allowed
    hard, soft, allowed = quietlib.problems(hot, judge_load=True, allow=frozenset({"load", "gpu", "thermal"}))
    assert not hard and not soft
    assert allowed == ["allowed load1 9.80 > 2.5", "allowed GPU busy 90% > 10%", "allowed thermal: CPU_Speed_Limit = 80"]
    # the monitor never judges load or GPU; a busy foreign process still counts under every allowance
    busy = sample(busy=[(123, 80.0, "/usr/bin/foo")], thermal=["CPU_Speed_Limit = 80"])
    hard, soft, allowed = quietlib.problems(busy, judge_load=False, allow=frozenset(quietlib.ALLOWANCES))
    assert soft == ["busy: foo(123) 80%"] and allowed == ["allowed thermal: CPU_Speed_Limit = 80"]


def test_lowpower_accepts_low_power_on_ac_only(quietlib):
    low = sample(power_ok=False, power="AC Power; powermode 1 (Low Power)", low_power_ac=True)
    assert quietlib.problems(low, judge_load=False)[0] == ["power: AC Power; powermode 1 (Low Power)"]
    hard, _, allowed = quietlib.problems(low, judge_load=False, allow=frozenset({"lowpower"}))
    assert not hard and allowed == ["allowed power: AC Power; powermode 1 (Low Power)"]
    batt = sample(power_ok=False, power="Battery Power; powermode 1 (Low Power)", low_power_ac=False)
    assert quietlib.problems(batt, judge_load=False, allow=frozenset({"lowpower"}))[0] == [
        "power: Battery Power; powermode 1 (Low Power)"]


def test_power_state_reads_both_low_power_names(quietlib, monkeypatch):
    def fake(conf):
        def run(cmd, timeout=20.0):
            return 0, ("Now drawing from 'AC Power'\n" if cmd[-1] == "batt" else conf)
        return run

    for conf, ok, low in ((" powermode            2\n", True, False), (" powermode            1\n", False, True),
                          (" powermode 2\n lowpowermode 1\n", False, True), (" sleep 1\n", True, False)):
        monkeypatch.setattr(quietlib, "run", fake(conf))
        st = quietlib.power_state()
        assert (quietlib.power_ok(st), quietlib.low_power(st)) == (ok, low), conf
    monkeypatch.setattr(quietlib, "run", fake(" powermode 2\n"))
    assert quietlib.power() == (True, "AC Power; powermode 2 (High Power)")   # unchanged for the normal case


def test_verify_ignores_allowed_reasons_and_reports_allowances(quietlib, tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    t0 = float(int(time.time()) - 60)
    f = raw / "m1-x-m8-cpuall-20261020-230000.jsonl"
    created = datetime.datetime.fromtimestamp(t0).astimezone().isoformat(timespec="seconds")
    f.write_text(json.dumps({"type": "header", "created": created}) + "\n")
    os.utime(f, (t0 + 40, t0 + 40))

    def log(name, allowances, reasons):
        p = tmp_path / name
        rows = "".join(f"x\t{t0 + k * 10:.1f}\tQUIET\t9.0\t10\t0\t1\t90\t2\t{reasons}\n" for k in range(1, 5))
        p.write_text(f"# quietlib monitor v3 interval=10 allow=1{allowances}\n# header\n" + rows)
        return p

    ok = log("ok.tsv", " allowances=load,thermal", "allowed thermal: CPU_Speed_Limit = 80; allowed power: AC Power")
    assert quietlib.main(["verify", "--log", str(ok), "--raw-dir", str(raw), "--since", str(t0 - 5)]) == 0
    assert "quiet (4 samples, 40 s; allowances load,thermal)" in capsys.readouterr().out
    bad = log("bad.tsv", "", "thermal: CPU_Speed_Limit = 80")
    assert quietlib.main(["verify", "--log", str(bad), "--raw-dir", str(raw), "--since", str(t0 - 5)]) == 1
    assert "DISTURBED" in capsys.readouterr().out


def test_monitor_records_allowances_and_stays_quiet(quietlib, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quietlib, "sample", lambda allow: sample(thermal=["CPU_Speed_Limit = 80"]))
    for allow, state in ((["thermal"], "QUIET"), ([], "DISTURBED")):
        calls = iter([True, True, False])
        monkeypatch.setattr(quietlib, "pid_alive", lambda pid: next(calls))
        log = tmp_path / f"mon-{state}.tsv"
        argv = ["monitor", "--log", str(log), "--interval", "0.01", "--until-pid", "1"] + (["--allow", ",".join(allow)] if allow else [])
        assert quietlib.main(argv) == 0
        lines = log.read_text().splitlines()
        assert ("allowances=thermal" in lines[0]) == bool(allow)
        rows = [ln.split("\t") for ln in lines[2:]]
        assert len(rows) == 2 and all(r[2] == state for r in rows)
        assert rows[0][-1] == ("allowed thermal: CPU_Speed_Limit = 80" if allow else "thermal: CPU_Speed_Limit = 80")
    with pytest.raises(SystemExit):
        quietlib.main(["check", "--allow", "load,bogus"])


# ---------------------------------------------------------------- run_m8.sh
def test_run_m8_dry_run_prints_the_alternating_plan():
    env = dict(os.environ, M8_DRY_RUN="1", M8_SCENARIOS="cpu1 gpu power", M8_POWER_WAIT_S="600",
               M8_LOAD_ARGS_gpu="--duty 0.25", SUITE_ARGS_m4="--runs 1 --steps 10")
    r = subprocess.run(["/bin/bash", str(SCRIPTS / "run_m8.sh"), "llama.cpp", "mtp-on", "4bit", "llamacpp", "4bit-mtp-on",
                        "--spec-label", "on"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "7 blocks: base cpu1 base gpu base power base" in out
    assert "suite m1: --cold-trials 0 --spec-label on" in out
    assert "suite m3: --contexts 16000 --caps 150 --no-thinking-bucket --trials 5 --spec-label on" in out
    assert "suite m4: --runs 1 --steps 10 --spec-label on" in out
    assert "block 1/7 base: tag 4bit-mtp-on-m8-base, no load" in out and "block 7/7 base" in out
    gpu = next(ln for ln in out.splitlines() if "block 4/7 gpu" in ln)
    assert "tag 4bit-mtp-on-m8-gpu, allowances gpu,load,thermal" in gpu and '"duty": 0.25' in gpu
    power = next(ln for ln in out.splitlines() if "block 6/7 power" in ln)
    assert "allowances load,lowpower" in power and '"wait": 600.0' in power
    bad = subprocess.run(["/bin/bash", str(SCRIPTS / "run_m8.sh"), "x", "y", "z", "p", "t"], capture_output=True, text=True,
                         env=dict(env, M8_SCENARIOS="cpu1 bogus"), timeout=60)
    assert bad.returncode == 2 and "unknown scenario 'bogus'" in bad.stdout


FAKE_QUIETLIB = '''import os, sys, time
sys.path.insert(0, "@REAL@")
cmd = sys.argv[1]
if cmd == "battery":
    print("battery: 90% (charging); ok (min 40%)")
    sys.exit(0)
if cmd == "monitor":   # a quiet machine, sampled every 0.2 s; records what it was told to exempt
    a = sys.argv[2:]
    until = int(a[a.index("--until-pid") + 1])
    pids = [a[i + 1] for i, x in enumerate(a) if x == "--allow-pid"]
    allow = [a[i + 1] for i, x in enumerate(a) if x == "--allow"]
    with open(a[a.index("--log") + 1], "a") as f:
        f.write("# quietlib monitor v3 interval=0.2 allow=" + ",".join(pids)
                + (" allowances=" + ",".join(allow) if allow else "") + "\\n# iso\\tepoch\\tstate\\treasons\\n")
        while True:
            try:
                os.kill(until, 0)
            except OSError:
                break
            f.write("x\\t%.2f\\tQUIET\\t1\\t1\\t0\\t1\\t90\\t0\\t\\n" % time.time())
            f.flush()
            time.sleep(0.2)
    sys.exit(0)
import quietlib   # the real verify
sys.exit(quietlib.main(sys.argv[1:]))
'''

FAKE_UV = '''#!/usr/bin/python3 -I
import datetime, json, os, signal, sys, time
from pathlib import Path
a = sys.argv[1:]
if "m8-report" in a:
    print("fake m8-report " + " ".join(a))
    sys.exit(0)
opt = lambda name: a[a.index(name) + 1]
root = Path.cwd().parent   # run_m8.sh runs the harness from harness/
tag, suite = opt("--tag"), opt("--suites")
raw = root / opt("--framework") / "raw" / ("_unverified" if "--unverified" in a else "")
raw.mkdir(parents=True, exist_ok=True)
if os.environ.get("FAKE_UV_KILL_LOAD") and tag.endswith("-m8-cpu1"):   # the load dies during the suite
    os.kill(json.loads((root / "logs" / "m8-load.pid").read_text())["pid"], signal.SIGKILL)
time.sleep(0.3)
hdr = {"type": "header", "suite": suite, "tag": tag, "argv": a, "interference": json.loads(opt("--interference")),
       "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
stem = suite + "-" + tag + "-" + time.strftime("%Y%m%d-%H%M%S")
p, i = raw / (stem + ".jsonl"), 1
while p.exists():
    i += 1
    p = raw / ("%s-%d.jsonl" % (stem, i))
p.write_text(json.dumps(hdr) + "\\n")
time.sleep(0.3)
'''


def fake_repo(tmp: Path) -> tuple[Path, dict]:
    """A repo copy whose server, harness, lock and quiet checks are fakes; run_m8.sh, interference.py
    (a real, tiny cpu1 load) and quietlib's verify are the real ones."""
    import shutil

    repo = tmp / "repo"
    sc = repo / "scripts"
    sc.mkdir(parents=True)
    (repo / "harness").mkdir()
    (repo / "logs").mkdir()
    for name in ("run_m8.sh", "interference.py"):
        shutil.copy(SCRIPTS / name, sc / name)
    (sc / "quietlib.py").write_text(FAKE_QUIETLIB.replace("@REAL@", str(SCRIPTS)))
    scripts = {
        "quiet_check.sh": 'echo "$*" >> logs/quiet-calls.txt\necho "verdict: QUIET"\n',
        "serve.sh": ('[ "${4:-}" = --dry-run ] && { echo "  # model id: mock"; echo "  # endpoint: http://127.0.0.1:8798/v1"; exit 0; }\n'
                     'echo $$ > "logs/$1.pid"\nprintf "# framework: fake 1.0\\nfake-server --port 8798\\n" > "logs/$1-$2-$3.cmd"\n'
                     'exec sleep 300\n'),
        "wait_ready.sh": "exit 0\n",
        "stop_server.sh": 'p=$(cat "logs/$1.pid" 2>/dev/null) && kill "$p" 2>/dev/null\nexit 0\n',
    }
    for name, body in scripts.items():
        p = sc / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
    uv = tmp / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)
    lock = tmp / "home" / "Developer" / "mlsys" / "experiments" / ".gpu-lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("holder: local_ai-benchmark session\npurpose: test\n")
    env = dict(os.environ, HOME=str(tmp / "home"), PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", M8_SCENARIOS="cpu1",
               M8_SUITES="m1,m4", M8_SETTLE_S="0", M8_COOL_S="0", M8_SAMPLE_LAG_S="0.5", USER_ACTIVE_SETTLE_S="0",
               BATTERY_PAUSE_LONG="0", QUIET_MIN_BATTERY="0", M8_LOAD_ARGS_cpu1="--duty 0.05")
    return repo, env


def run_m8(repo: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/bash", str(repo / "scripts" / "run_m8.sh"), "fakefw", "v", "t", "generic", "T",
                           "--spec-label", "off"], cwd=repo, capture_output=True, text=True, env=env, timeout=120)


def test_run_m8_mechanics_with_fakes(tmp_path):
    repo, env = fake_repo(tmp_path)
    r = run_m8(repo, env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    raw = repo / "fakefw" / "raw"
    hdrs = {f.name: json.loads(f.read_text()) for f in raw.glob("*.jsonl")}
    tags = sorted(h["tag"] + "/" + h["suite"] for h in hdrs.values())
    assert tags == ["T-m8-base/m1", "T-m8-base/m1", "T-m8-base/m4", "T-m8-base/m4", "T-m8-cpu1/m1", "T-m8-cpu1/m4"]
    assert out.count("verified (harness rc 0)") == 6 and "allowances load" in out
    cpu1 = [h for h in hdrs.values() if h["tag"] == "T-m8-cpu1"]
    base = [h for h in hdrs.values() if h["tag"] == "T-m8-base"]
    lpid = cpu1[0]["interference"]["load_pid"]
    for h in cpu1:
        i = h["interference"]
        assert i["scenario"] == "cpu1" and i["allow"] == ["load"] and i["params"]["duty"] == 0.05 and i["block"] == 2
        pre = h["argv"][h["argv"].index("--preflight-cmd") + 1]
        assert f"--allow-pid {lpid} --allow load" in pre and "--no-summary" in h["argv"]
        assert h["argv"][h["argv"].index("--cold-trials") + 1] == "0" if h["suite"] == "m1" else "--runs" in h["argv"]
    for h in base:
        assert h["interference"]["scenario"] == "base" and h["interference"]["load_pid"] is None
        assert "--allow " not in h["argv"][h["argv"].index("--preflight-cmd") + 1]
    assert {h["interference"]["block"] for h in base} == {1, 3} and len({h["interference"]["run"] for h in hdrs.values()}) == 1
    mons = sorted(raw.glob("monitor-*.tsv"))
    assert len(mons) == 3
    cpu1_mon = next(m for m in mons if "-m8-cpu1-" in m.name).read_text().splitlines()[0]
    assert f",{lpid}" in cpu1_mon and cpu1_mon.endswith("allowances=load")
    calls = (repo / "logs" / "quiet-calls.txt").read_text().splitlines()
    strict = [c for c in calls if "--min-battery" in c and "--max-gpu-gb" not in c]
    assert len(strict) == 3 and not any("--allow " in c for c in strict)   # before every block: no load, no allowances
    assert sum(f"--allow-pid {lpid} --allow load" in c for c in calls) == 2   # the cpu1 block's per-suite checks
    assert not (repo / "logs" / "m8-load.pid").exists() and "fake m8-report" in out
    ps = subprocess.run(["ps", "-Ao", "command="], capture_output=True, text=True).stdout
    assert str(repo / "scripts" / "interference.py") not in ps   # no worker of this run is left (the path is unique)
    assert gone(lpid, timeout=2)   # nor its root


def test_run_m8_quarantines_a_suite_whose_load_died(tmp_path):
    repo, env = fake_repo(tmp_path)
    r = run_m8(repo, dict(env, FAKE_UV_KILL_LOAD="1"))
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "the load stopped during m1" in out and "moved m1-T-m8-cpu1-" in out and "incomplete blocks: cpu1" in out
    raw = repo / "fakefw" / "raw"
    assert len(list((raw / "_unverified").glob("m1-T-m8-cpu1-*.jsonl"))) == 1
    assert not list(raw.glob("*-T-m8-cpu1-*.jsonl"))   # the block ended: no m4 under a dead load
    assert len(list(raw.glob("m*-T-m8-base-*.jsonl"))) == 4   # both baselines still ran
    assert not (repo / "logs" / "m8-load.pid").exists()
