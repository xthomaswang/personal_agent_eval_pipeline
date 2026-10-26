from __future__ import annotations

import os
import time

import pytest

from modeleval import paths, quiet
from modeleval.quiet import QuietGuard, parse_battery, read_monitor


def test_parse_battery():
    ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t57%; charging; 1:00 remaining present: true\n"
    bat = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t18%; discharging; (no estimate)\n"
    assert parse_battery(ac) == (57, True)
    assert parse_battery(bat) == (18, False)
    assert parse_battery("Now drawing from 'AC Power'\n") == (-1, True)


@pytest.mark.parametrize("pct, kind, want", [
    (55, "unit", "go"), (49, "unit", "pause"), (55, "long-unit", "pause"), (60, "long-unit", "go"),
    (59, "harness-block", "pause"), (50, "task", "go"), (45, "task", "pause"), (19, "task", "stop"),
    (19, "unit", "pause"), (-1, "task", "go"),
])
def test_boundary_rule(monkeypatch, pct, kind, want):
    monkeypatch.setattr(quiet, "battery", lambda: (pct, True))
    assert quiet.boundary(kind) == want


def test_hold_file(env):
    assert not quiet.hold_requested()
    paths.HOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.HOLD_FILE.touch()
    assert quiet.hold_requested()


def test_quiet_module_imports_only_stdlib_and_paths():
    """The capability runner imports battery/boundary/hold_requested under another interpreter."""
    import ast
    tree = ast.parse(open(quiet.__file__).read())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= {"__future__", "importlib", "os", "re", "signal", "subprocess", "sys", "time", "dataclasses",
                    "pathlib", "typing", "modeleval"}


def test_read_monitor_by_header_names_across_restarts(tmp_path):
    tsv = tmp_path / "monitor.tsv"
    tsv.write_text("# quietlib monitor v4 interval=5\n# iso\tepoch\tstate\tbattery\treasons\n"
                   "a\t100.0\tQUIET\t90\t\n"
                   "# quietlib monitor v4 interval=5\n# iso\tepoch\tbattery\tstate\treasons\n"
                   "b\t200.0\t80\tDISTURBED\tbusy: x(1) 90%\n")
    interval, rows = read_monitor(tsv)
    assert interval == 5.0 and [r["epoch"] for r in rows] == [100.0, 200.0]
    assert rows[1]["state"] == "DISTURBED" and rows[1]["battery"] == "80"


def _checks(env, *states):
    (env.root / "quiet-checks").write_text("".join(s + "\n" for s in states))


def test_wait_quiet_emits_unit_wait_then_passes(env, capsys):
    _checks(env, "busy", "busy", "quiet")
    g = QuietGuard(env.results / "r", [1234])
    assert g.wait_quiet("m3-16k", settle_s=60)
    out = capsys.readouterr().out
    assert out.count("UNIT-WAIT") == 1 and "unit=m3-16k" in out and "fakeproc(4242)" in out
    calls = (env.root / "quiet.log").read_text().splitlines()
    assert len(calls) == 3 and "--allow-pid 1234" in calls[0] and f"--allow-pid {os.getpid()}" in calls[0]
    assert "--settle 1" in calls[0] and all("--major-only" in c for c in calls)


def test_wait_quiet_does_not_wait_for_minor_conditions(env, capsys):
    """DESIGN §5.4: the pre-unit check waits for major conditions only; the full rule stays the default."""
    _checks(env, "minor")
    assert QuietGuard(env.results / "r", [1]).wait_quiet("m1", settle_s=60)
    assert "UNIT-WAIT" not in capsys.readouterr().out
    _checks(env, "minor", "quiet")
    assert quiet.wait_check("launch", [1], 60).ok
    assert "UNIT-WAIT" in capsys.readouterr().out                    # without --major-only it waited
    calls = (env.root / "quiet.log").read_text().splitlines()
    assert "--major-only" in calls[0] and not any("--major-only" in c for c in calls[1:])
    _checks(env, "minor")
    assert quiet.prelaunch_check(timed=True, settle_s=60).ok          # the pre-launch check: major rule too
    assert "UNIT-WAIT" not in capsys.readouterr().out
    assert all("--major-only" in c for c in (env.root / "quiet.log").read_text().splitlines()[3:])


def test_unit_verdict_severity():
    v = quiet.UnitVerdict(False, ["no monitor log"], 0.0, 0)          # built without a severity: major
    assert v.severity == "major" and v.major and v.major_reasons == ["no monitor log"]
    v = quiet.UnitVerdict(True, [], 0.0, 3)
    assert v.severity == "clean" and not v.major
    v = quiet._as_verdict({"clean": False, "severity": "minor", "reasons": ["a"], "minor_reasons": ["a"],
                           "major_reasons": [], "samples": 4, "disturbed_samples": 1, "minor_samples": 1})
    assert v.severity == "minor" and not v.major and v.minor_reasons == ["a"] and v.major_reasons == []
    assert v.minor_samples == 1 and v.major_samples == 0
    v = quiet._as_verdict({"clean": False, "reasons": ["b"], "samples": 4})   # an old one-level judge
    assert v.severity == "major" and v.major_reasons == ["b"]


def test_wait_quiet_gives_up_on_hard_failure_at_once(env, capsys):
    _checks(env, "hard", "quiet")
    t0 = time.time()
    assert not QuietGuard(env.results / "r", [1]).wait_quiet("m1", settle_s=600)
    assert time.time() - t0 < 10
    assert "lock not held by us" in capsys.readouterr().out


def test_wait_quiet_gives_up_after_settle(env):
    _checks(env, *["busy"] * 200)
    assert not QuietGuard(env.results / "r", [1]).wait_quiet("m1", settle_s=1)


def test_monitor_and_judge(env):
    g = QuietGuard(env.results / "r", [os.getpid()])
    g.start_monitor()
    try:
        time.sleep(0.5)
        t0 = time.time()
        time.sleep(0.5)
        v = g.judge(t0, time.time())
        assert v.clean and v.samples >= 2 and v.battery_min == 95
        (env.root / "disturb").touch()
        t1 = time.time()
        time.sleep(0.5)
        (env.root / "disturb").unlink()
        v = g.judge(t1, time.time())
        assert not v.clean and v.disturbed_samples >= 1 and "fakeproc" in v.reasons[0] and v.major
        f = g.flags(t1, time.time())
        assert f["disturbed"] is True and f["battery_min"] == 95
        assert f["severity"] == "major" and f["major_samples"] >= 1 and f["minor_samples"] == 0
        time.sleep(0.5)
        (env.root / "minor").touch()
        t2 = time.time()
        time.sleep(0.5)
        (env.root / "minor").unlink()
        v = g.judge(t2, time.time())
        assert v.severity == "minor" and not v.clean and not v.major and "smallproc" in v.minor_reasons[0]
        f = g.flags(t2, time.time())
        assert f["disturbed"] is True and f["severity"] == "minor" and f["minor_samples"] >= 1
    finally:
        g.stop_monitor()
    assert not g.monitor_alive()


def test_judge_without_monitor_is_never_clean(env):
    (env.results / "r/speed").mkdir(parents=True)
    g = QuietGuard(env.results / "r", [1])
    g.cover_timeout = 0.5
    v = g.judge(time.time() - 1, time.time())
    assert not v.clean and any("monitor" in r for r in v.reasons)


def test_judge_requires_judge_window(env):
    (env.scripts / "quietlib.py").write_text("# old quietlib without judge_window\n")
    (env.results / "r/speed").mkdir(parents=True)
    g = QuietGuard(env.results / "r", [1])
    g.cover_timeout = 0.1
    with pytest.raises(RuntimeError, match="judge_window"):
        g.judge(time.time() - 1, time.time())


def test_judge_with_the_real_quietlib_copy(env):
    """Contract check against scripts/quietlib.py (a copy in the sandbox): signature and result keys."""
    real = paths.REPO / "scripts/quietlib.py"
    import shutil
    shutil.copy(real, env.scripts / "quietlib.py")
    ql = quiet.load_quietlib()
    if not hasattr(ql, "judge_window"):
        pytest.skip("scripts/quietlib.py has no judge_window yet")
    cols = ql.MONITOR_COLUMNS
    t0 = time.time() - 30
    tsv = env.results / "r/speed/monitor.tsv"
    tsv.parent.mkdir(parents=True)

    def row(t: float, reasons: str = "") -> str:
        vals = {"iso": "x", "epoch": f"{t:.1f}", "state": "QUIET", "load1": "1.0", "foreign_cpu": "5.0",
                "gpu_util": "0", "gpu_mem_gb": "1.0", "battery": "90", "thermal_pressure": "0", "pressure": "1",
                "swapouts_delta": "0", "foreign_gpu_ms": "0", "foreign_rss_delta_mb": "0", "reasons": reasons}
        return "\t".join(vals.get(c, "") for c in cols) + "\n"

    lines = ["# quietlib monitor v4 interval=5\n", "# " + "\t".join(cols) + "\n"]
    lines += [row(t0 + i) for i in range(0, 12)]
    lines += [row(t0 + 12, "tools: hf download org/x")]
    lines += [row(t0 + i) for i in range(13, 30)]
    tsv.write_text("".join(lines))
    g = QuietGuard(env.results / "r", [1])
    v = g.judge(t0 + 2, t0 + 8)
    assert v.clean and v.samples >= 6 and v.battery_min == 90, v
    v = g.judge(t0 + 10, t0 + 14)
    assert not v.clean and v.disturbed_samples >= 1 and any("tools" in r for r in v.reasons), v
    assert v.severity == "minor" and not v.major and any("tools" in r for r in v.minor_reasons), v
