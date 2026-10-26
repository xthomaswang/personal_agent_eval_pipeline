"""quietlib's sampler, check, monitor and battery subcommands against a fake machine."""

from __future__ import annotations

import pytest


def test_parses_gpu_clients_and_stats(quietlib, machine):
    machine.gpu = {"0x10": (613, "WindowServer", 5_000_000), "0x11": (300, "mds_stores", 7),
                   "0x12": (300, "mds_stores", 0)}
    machine.util, machine.gpu_mem = 7, 2_500_000_000
    text = machine.ioreg_text()
    assert quietlib.gpu_clients(text) == {"0x10": (613, "WindowServer", 5_000_000), "0x11": (300, "mds_stores", 7),
                                          "0x12": (300, "mds_stores", 0)}
    assert quietlib.gpu_stats(text) == (7, 2.5)


def test_real_ioreg_layout_sums_app_usage_entries(quietlib):
    text = ('+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100001685, !registered, !matched, active>\n'
            '    {\n'
            '      "AppUsage" = ({"API"="Metal","lastSubmittedTime"=10,"accumulatedGPUTime"=1000},'
            '{"API"="Metal","lastSubmittedTime"=0,"accumulatedGPUTime"=0},{"API"="Metal","accumulatedGPUTime"=24})\n'
            '      "IOUserClientCreator" = "pid 613, WindowServer"\n'
            '      "CommandQueueCount" = 4\n'
            '    }\n')
    assert quietlib.gpu_clients(text) == {"0x100001685": (613, "WindowServer", 1024)}


def test_first_sample_has_no_deltas(quietlib, machine):
    s = quietlib.Sampler([100]).sample()
    assert s["foreign_gpu_ms"] is None and s["swapouts_delta"] is None and s["foreign_rss_delta_mb"] is None
    assert s["lock"] == "test (timed)" and s["ports"] == [] and s["pressure"] == 1 and s["page_size"] == 16384


def test_deltas_count_only_foreign_processes(quietlib, machine):
    machine.gpu = {
        "0x1": (100, "python", 10_000_000_000),       # the server under test: allowed
        "0x2": (101, "python", 1_000_000_000),        # its worker: allowed through the tree
        "0x3": (613, "WindowServer", 1_000_000_000),  # ignored
        "0x4": (0, "kernel_task", 1_000_000_000),     # ignored
        "0x5": (300, "mds_stores", 50_000_000),
        "0x6": (300, "mds_stores", 0),
    }
    sampler = quietlib.Sampler([100])
    sampler.sample()
    for cid, (pid, cname, ns) in list(machine.gpu.items()):
        machine.gpu[cid] = (pid, cname, ns + 900_000_000)      # every client used 900 ms more
    machine.gpu["0x7"] = (300, "mds_stores", 5_000_000_000)    # a new client counts from the next sample
    del machine.gpu["0x6"]                                      # a closed client counts nothing
    machine.gpu["0x6b"] = (777, "gone-already", 1)             # pid not in ps: still foreign, name "?"
    machine.swapouts = 1000 + 4096
    machine.procs = [p if p[0] != 300 else (300, 1, 5.0, 400_000 + 2048 * 1024, p[4]) for p in machine.procs]
    machine.procs = [p if p[0] != 100 else (100, 1, 80.0, 90_000_000, p[4]) for p in machine.procs]   # server grows
    s = sampler.sample()
    assert s["foreign_gpu_ms"] == 900.0
    assert s["gpu_top"] == [(300, 900.0, "/usr/libexec/mds_stores")]
    assert s["swapouts_delta"] == 4096
    assert s["foreign_rss_delta_mb"] == 2048.0
    machine.gpu["0x7"] = (300, "mds_stores", 5_000_000_000 + 30_000_000)
    s = sampler.sample()
    assert s["foreign_gpu_ms"] == 30.0


def test_failed_ioreg_resets_gpu_baseline(quietlib, machine, monkeypatch):
    machine.gpu = {"0x5": (300, "mds_stores", 50_000_000)}
    sampler = quietlib.Sampler([])
    sampler.sample()
    real = machine.ioreg_text
    monkeypatch.setattr(machine, "ioreg_text", lambda: "")
    assert sampler.sample()["foreign_gpu_ms"] is None
    monkeypatch.setattr(machine, "ioreg_text", real)
    assert sampler.sample()["foreign_gpu_ms"] is None      # no baseline yet: never a since-boot total
    machine.gpu["0x5"] = (300, "mds_stores", 60_000_000)
    assert sampler.sample()["foreign_gpu_ms"] == 10.0


def test_foreign_ports_are_hard_failures_ours_are_not(quietlib, machine):
    machine.listen = [(8612, 700), (8650, 101), (8703, 100), (8500, 701)]
    s = quietlib.Sampler([100]).sample()
    assert s["ports"] == [8500, 8612]                 # 8650 belongs to an allowed child, 8703 is ours
    hard, _, _ = quietlib.problems(s, judge_load=False)
    assert "port 8500 listening" in hard and "port 8612 listening" in hard
    lsof = next(c for c in machine.calls if c[0] == "lsof")
    assert "-iTCP:8500,8600-8699" in lsof and "-sTCP:LISTEN" in lsof


@pytest.mark.parametrize("cmd", [
    "git clone https://example.invalid/x.git", "/usr/bin/git -C /tmp/x fetch origin", "git pull --rebase",
    "git lfs pull", "hf download Nanbeige/Nanbeige4.3-33B --revision abc", "huggingface-cli download x",
    "/x/.venv/bin/python /x/.venv/bin/hf download x", "uv sync --frozen", "uv pip install torch",
    "python -m pip install x", "pip3 install x", "brew upgrade", "python scripts/oq_convert.py --level 5",
    "python -m mlx_lm.convert --hf-path x", "conda install numpy", "npm ci",
])
def test_build_args_detect_tools(quietlib, cmd):
    assert quietlib.BUILD_ARGS.search(cmd)


@pytest.mark.parametrize("cmd", [
    "git status --porcelain", "git rev-parse HEAD", "git log --oneline", "/usr/bin/git diff", "claude",
    "uv run pytest", "hf cache scan", "python -m modeleval run x", "less fetch.log",
])
def test_build_args_ignore_ordinary_commands(quietlib, cmd):
    assert not quietlib.BUILD_ARGS.search(cmd)


def test_tools_outside_allowed_tree_are_reported(quietlib, machine):
    machine.procs.append((400, 1, 1.0, 1000, "/usr/bin/git"))
    machine.procs.append((401, 100, 1.0, 1000, "/usr/bin/git"))
    machine.cmdlines = {400: "git clone https://example.invalid/big.git", 401: "git clone https://example.invalid/y"}
    s = quietlib.Sampler([100]).sample()
    assert s["tools"] == [{"pid": 400, "kind": "git clone", "cmd": "git clone https://example.invalid/big.git"}]


def test_monitor_row_round_trips_through_tsv(quietlib, machine, tmp_path):
    sampler = quietlib.Sampler([100])
    sampler.sample()
    machine.pressure, machine.swapouts = 2, 1000 + 8192
    s = sampler.sample()
    s["busy"] = [(300, 40.0, "/usr/libexec/mds_stores")]
    s["thermal"] = ["CPU_Speed_Limit\t= 80"]
    row = quietlib.monitor_row(s)
    row["state"] = quietlib.row_state(row, None)
    line = quietlib.format_row(row)
    assert line.count("\t") == len(quietlib.MONITOR_COLUMNS) - 1 and line.endswith("\n")
    path = tmp_path / "m.tsv"
    path.write_text(quietlib.monitor_header(5.0, [100], frozenset(), 16384) + line, encoding="utf-8")
    (back,) = quietlib.read_monitor(path)
    assert back["pressure"] == 2 and back["swapouts_delta"] == 8192 and back["foreign_rss_delta_mb"] == 0.0
    assert "busy: mds_stores(300) 40%" in back["reasons"]
    assert "memory pressure 2 (warn)" in back["reasons"] and "swap-out 128 MB" in back["reasons"]
    assert "thermal: CPU_Speed_Limit = 80" in back["reasons"]
    assert back["state"] == "SPIKE"


def test_monitor_command_writes_header_and_rows(quietlib, machine, tmp_path, capsys):
    log = tmp_path / "run" / "monitor.tsv"
    assert quietlib.main(["monitor", "--log", str(log), "--interval", "0.01", "--count", "3",
                          "--allow-pid", "100"]) == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# quietlib monitor v4 interval=0.01 ") and "gpu_floor_ms=250.0" in lines[0]
    assert "major_gpu_ms=500.0" in lines[0] and "major_foreign_cpu=300.0" in lines[0]
    assert "page_size=16384" in lines[0] and lines[0].endswith("allow=100")
    assert lines[1] == "# " + "\t".join(quietlib.MONITOR_COLUMNS)
    rows = quietlib.read_monitor(log)
    assert len(rows) == 3 and [r["state"] for r in rows] == ["QUIET"] * 3
    assert rows[0]["foreign_gpu_ms"] is None and rows[1]["foreign_gpu_ms"] == 0.0
    assert "QUIET" in capsys.readouterr().out
    v = quietlib.judge_window(log, rows[0]["epoch"], rows[-1]["epoch"])
    assert v["clean"] and v["samples"] == 3


def test_monitor_marks_lost_lock_at_once(quietlib, machine, tmp_path, capsys):
    (tmp_path / ".gpu-lock").unlink()
    log = tmp_path / "monitor.tsv"
    quietlib.main(["monitor", "--log", str(log), "--interval", "0.01", "--count", "1", "--allow-pid", "100"])
    (row,) = quietlib.read_monitor(log)
    assert row["state"] == "DISTURBED" and row["reasons"] == "lock not held by us"
    assert "DISTURBED: lock not held by us" in capsys.readouterr().out


def test_monitor_marks_a_build_tool_minor(quietlib, machine, tmp_path, capsys):
    machine.procs.append((400, 1, 1.0, 1000, "/usr/bin/git"))
    machine.cmdlines = {400: "git fetch origin"}
    log = tmp_path / "monitor.tsv"
    quietlib.main(["monitor", "--log", str(log), "--interval", "0.01", "--count", "1", "--allow-pid", "100"])
    (row,) = quietlib.read_monitor(log)
    assert row["state"] == "MINOR" and row["reasons"] == "tools: git fetch (pid 400)"
    assert "MINOR: tools: git fetch (pid 400)" in capsys.readouterr().out


def test_check_quiet_prints_provenance(quietlib, machine, capsys):
    assert quietlib.main(["check", "--allow-pid", "100", "--max-gpu-gb", "6"]) == 0
    out = capsys.readouterr().out
    for line in ("lock: held by us (test (timed))", "port 8500: free", "ports 8600-8699: none listening",
                 "busy foreign processes: none", "build/convert/download tools: none",
                 "memory pressure: 1 (normal) (logged, not judged)", "verdict: QUIET"):
        assert line in out


def test_check_fails_on_foreign_server_and_lock(quietlib, machine, tmp_path, capsys):
    machine.listen = [(8500, 701), (8642, 702)]
    (tmp_path / ".gpu-lock").write_text("holder: agent-benchmark\npurpose: x\n", encoding="utf-8")
    assert quietlib.main(["check", "--allow-pid", "100", "--settle", "600"]) == 1
    out = capsys.readouterr().out
    assert "lock: NOT held by us" in out and "port 8500: LISTENING" in out
    assert "ports 8600-8699: LISTENING 8642" in out and "settle: not attempted (hard failure" in out


def test_check_soft_failures(quietlib, machine, capsys):
    machine.procs.append((500, 1, 60.0, 1000, "/usr/libexec/mediaanalysisd"))
    machine.util, machine.gpu_mem = 30, 40_000_000_000
    assert quietlib.main(["check", "--allow-pid", "100", "--max-gpu-gb", "6"]) == 1
    out = capsys.readouterr().out
    assert "500 60.0 /usr/libexec/mediaanalysisd" in out and "verdict: NOT QUIET" in out
    s = quietlib.Sampler([100]).sample()
    _, soft, _ = quietlib.problems(s, judge_load=True, max_gpu_gb=6)
    assert any(x.startswith("GPU busy 30%") for x in soft) and any(x.startswith("GPU memory in use 40.0 GB") for x in soft)


def test_check_major_only_waits_for_major_conditions_only(quietlib, machine, capsys):
    """The pre-unit check (--major-only): a busy process, build tools, the load and foreign CPU below
    300 % are printed on a "minor:" line and pass; the default check still fails on them."""
    machine.procs += [(500, 1, 60.0, 1000, "/usr/libexec/mediaanalysisd"), (400, 1, 1.0, 1000, "/usr/bin/git")]
    machine.cmdlines = {400: "git fetch origin"}
    machine.load = 4.0
    args = ["check", "--allow-pid", "100", "--settle", "1"]
    assert quietlib.main(args) == 1
    out = capsys.readouterr().out
    assert "verdict: NOT QUIET" in out and "minor:" not in out
    assert quietlib.main(args + ["--major-only"]) == 0
    out = capsys.readouterr().out
    assert "verdict: QUIET" in out and "settle: quiet after" in out
    minor = next(x for x in out.splitlines() if x.startswith("minor: "))
    assert "load1 4.00 > 2.5" in minor and "busy: mediaanalysisd(500) 60%" in minor
    assert "tools: git fetch (pid 400)" in minor and minor.endswith("(flag only: --major-only does not wait for it)")
    assert "foreign CPU" not in minor                         # 66 % in total is below even the minor line


@pytest.mark.parametrize("procs, util, blocking", [
    ([(500, 1, 160.0, 1000, "/usr/bin/a"), (501, 1, 160.0, 1000, "/usr/bin/b")], 0, "foreign CPU 325% >= 300%"),
    ([(500, 1, 100.0, 1000, "/usr/bin/a"), (501, 1, 90.0, 1000, "/usr/bin/b")], 0, None),   # 195 %: minor
    ([], 30, "GPU busy 30% > 10%"),                           # the same 10 % share as the major GPU rule
])
def test_check_major_only_blocking_reasons(quietlib, machine, capsys, procs, util, blocking):
    machine.procs += procs
    machine.util = util
    rc = quietlib.main(["check", "--allow-pid", "100", "--settle", "1", "--major-only"])
    out = capsys.readouterr().out
    if blocking is None:
        assert rc == 0 and "foreign CPU 195% > 150%" in next(x for x in out.splitlines() if x.startswith("minor: "))
    else:
        settle = next(x for x in out.splitlines() if x.startswith("settle: "))
        assert rc == 1 and settle.startswith("settle: still not quiet") and settle.endswith(blocking)


def test_check_major_only_keeps_hard_failures_and_the_battery_guard(quietlib, machine, tmp_path, capsys):
    machine.battery = 35
    assert quietlib.main(["check", "--allow-pid", "100", "--settle", "1", "--min-battery", "40", "--major-only"]) == 1
    assert "battery 35% < 40% (charging)" in capsys.readouterr().out
    machine.battery = 100
    (tmp_path / ".gpu-lock").unlink()
    assert quietlib.main(["check", "--allow-pid", "100", "--settle", "1", "--major-only"]) == 1
    assert "settle: not attempted (hard failure: lock not held by us)" in capsys.readouterr().out


def test_battery_command(quietlib, machine, capsys):
    assert quietlib.main(["battery", "--min", "60"]) == 0
    machine.battery = 50
    assert quietlib.main(["battery", "--min", "60"]) == 1
    assert "50% (charging) < 60%" in capsys.readouterr().out


def test_third_party_names_are_hidden(quietlib, machine):
    bundle = "/Applications/SomeChat" + ".app"    # split so this file passes redact.py --check
    machine.procs.append((900, 1, 50.0, 1000, bundle + "/Contents/MacOS/SomeChat"))
    s = quietlib.Sampler([100]).sample()
    _, soft, _ = quietlib.problems(s, judge_load=False)
    assert "busy: <app>(900) 50%" in soft


def test_a_token_in_another_sessions_command_never_reaches_committed_columns(quietlib, machine, redact, tmp_path):
    """A peer's `git clone` of a URL with a token in its user info during a capability leg: the monitor
    row and the judged window (units.json / results.json flags) carry only `tools: git clone (pid N)`;
    the full command stays in the check report (gitignored logs), and redact.py --check flags it
    anywhere else. (Token and URL are assembled at run time, so this file passes the check itself.)"""
    token = "gh" + "p_" + "Zz9" * 14
    cmd = f"git clone https:/{'/'}{token}@github.com/org/private-repo"
    machine.procs += [(400, 1, 1.0, 1000, "/usr/bin/git"), (401, 1, 1.0, 1000, "/x/bin/hf")]
    machine.cmdlines = {400: cmd, 401: f"/x/bin/hf download private-org/model --{'token'} hf_{'Q' * 30}"}
    s = quietlib.Sampler([100]).sample()
    row = quietlib.monitor_row(s)
    assert "tools: git clone (pid 400), hf download (pid 401)" in row["reasons"]
    assert token not in row["reasons"] and "private" not in row["reasons"] and str(tmp_path) not in row["reasons"]
    row["state"] = quietlib.row_state(row, None)
    tsv = tmp_path / "monitor.tsv"
    tsv.write_text(quietlib.monitor_header(5.0, [100], frozenset(), 16384) + quietlib.format_row(row), encoding="utf-8")
    v = quietlib.judge_window(tsv, row["epoch"], row["epoch"] + 1)
    assert not v["clean"] and any("tools: git clone (pid 400), hf download (pid 401)" in r for r in v["reasons"])
    assert all(token not in r and "private" not in r for r in v["reasons"])
    assert not redact.check_file(str(tsv), None)
    report = quietlib.report(s, None, False)
    assert token in report                                      # gitignored check logs keep the command
    leak = tmp_path / "leak.txt"
    leak.write_text(report + "\n", encoding="utf-8")
    assert any("credential" in p for p in redact.check_file(str(leak), None))


def test_tool_kinds_drop_a_git_directory(quietlib):
    assert quietlib.tool_kind("git", "/usr/bin/git -C /some/where fetch origin") == "git fetch"
    assert quietlib.tool_kind("cargo", "cargo build --release") == "cargo"
    assert quietlib.tool_kind("python", "python -m pytest") is None
