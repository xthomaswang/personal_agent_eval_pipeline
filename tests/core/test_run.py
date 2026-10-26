"""The orchestrator end to end against fakes: oMLX, lock script, quietlib, speed and the capability runner."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeleval import paths, quiet, run
from modeleval.server import group_alive, port_listening

from .conftest import FAKES, install_exec, model_table

UNITS = ["g5+m3-64k", "m3-16k", "m1", "m4-1"]


def _load_fake_speed():
    spec = importlib.util.spec_from_file_location("fake_speed", FAKES / "fake_speed.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fake_speed"] = mod   # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def orch(env, monkeypatch):
    """The sandbox plus the orchestrator's own fakes; returns helpers to steer and inspect a run."""
    root = env.root
    monkeypatch.setattr(paths, "REPO", root)
    (root / "pins.toml").write_text('epoch = "test-epoch"\n')
    cap = install_exec(FAKES / "fake_capability.py", root / "cap-python", sys.executable)
    monkeypatch.setattr(run, "CAP_PYTHON", cap)
    monkeypatch.setattr(run, "CAFFEINATE", False)
    monkeypatch.setattr(run, "SSD_CACHE_BYTES", 0)   # the host's free disk is not under test (see the refusal test)
    monkeypatch.setattr(run, "POLL_S", 0.01)
    monkeypatch.setattr(run, "machine_info", lambda: {"sw_vers": {"BuildVersion": "test"}})
    monkeypatch.setattr(run, "load_allowed_env", lambda: (lambda env_, extra: {**env_, **extra}))
    monkeypatch.setattr(run, "ensure_epoch", lambda: "test-epoch")   # vendor/ snapshots: tests/capability
    fake = _load_fake_speed()
    monkeypatch.setattr(run, "load_speed", lambda: fake)
    state = SimpleNamespace(pct=95, sleeps=[], hooks=[], fake=fake)
    monkeypatch.setattr(quiet, "battery", lambda: (state.pct, True))

    def fake_sleep(s):
        state.sleeps.append(s)
        for h in list(state.hooks):
            h()
        time.sleep(0.01)

    monkeypatch.setattr(run, "sleep", fake_sleep)
    for k, v in {"FAKE_CAP_SEQ": root / "cap-seq", "FAKE_CAP_LOG": root / "cap.log",
                 "FAKE_CAP_HOLD": paths.HOLD_FILE, "FAKE_SPEED_CONTROL": root / "speed.json",
                 "FAKE_SPEED_LOG": root / "speed.log"}.items():
        monkeypatch.setenv(k, str(v))
    (root / "runs").mkdir(exist_ok=True)

    def control(**c):
        (root / "speed.json").write_text(json.dumps({"units": UNITS, "sleep": 0.3, **c}))

    def cap_codes(*codes):
        (root / "cap-seq").write_text("".join(f"{c}\n" for c in codes))

    def go(**kw):
        opts = run.Options(model="fake-1b", notice_wait=0, settle_s=5, **kw)
        return run.run(opts, env.models)

    def lines(path):
        p = root / path
        return p.read_text().splitlines() if p.exists() else []

    control()
    cap_codes()
    return SimpleNamespace(env=env, state=state, control=control, cap_codes=cap_codes, go=go, lines=lines,
                           cap_log=lambda: [json.loads(x) for x in lines("cap.log")])


def _tags(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if len(ln.split()) > 1 and ln[:2].isdigit()]


def _only_dir(p: Path) -> Path:
    dirs = [d for d in p.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _assert_clean_exit(o):
    assert not port_listening(o.env.port)
    assert not o.env.lock.exists()
    for e in (o.env.runs / "fake-1b").glob("*/pidfile"):
        assert e.read_text() == "", e.read_text()


def test_unverified_end_to_end_hold_and_server_lost(orch, capsys):
    o = orch
    o.cap_codes(75, 76, 0)

    def release_hold():
        paths.HOLD_FILE.unlink(missing_ok=True)

    o.state.hooks.append(release_hold)
    assert o.go(unverified=True) == 0
    out = capsys.readouterr().out
    tags = _tags(out)
    assert tags[0] == "LOCK-NOTICE" and tags[-1] == "RUN-DONE"
    assert "QUIET-WINDOW-END" in tags and tags.index("HOLD") < tags.index("HOLD-RELEASED")
    assert "status=complete" in out.splitlines()[-1]
    assert not (o.env.results / "fake-1b").exists()             # unverified: nothing under results/<model>
    res = _only_dir(paths.UNVERIFIED / "fake-1b")
    units = json.loads((res / "speed/units.json").read_text())
    assert units["functional"]["passed"] is True
    assert {u: v["verdict"] for u, v in units["units"].items()} == {u: "clean" for u in UNITS}
    man = json.loads((res / "manifest.json").read_text())
    assert man["status"] == "complete" and man["unverified"] is True
    assert [ln["reason"] for ln in man["launches"]] == ["initial", "server-lost"]
    assert all(ln["stopped"] for ln in man["launches"])
    assert "/Users/" not in (res / "manifest.json").read_text()
    assert "auth" not in man["server"]["settings"]
    caps = o.cap_log()
    assert [c["code"] for c in caps] == [75, 76, 0, 0]
    assert [c["resume"] for c in caps] == [False, True, True, True]
    assert caps[0]["server_pid"] == caps[1]["server_pid"] != caps[2]["server_pid"]   # relaunched after 76
    # the gate set first; the reference model has nothing to compare with, so the ranking set follows
    assert caps[0]["epoch"] == "test-epoch" and [c["sets"] for c in caps] == [["gate"]] * 3 + [["ranking"]]
    assert "GATE-DECISION" in tags and 'why="reference model"' in out and "action=ranking" in out
    assert man["gate_decision"]["action"] == "ranking"
    assert caps[0]["server_log"] == str(o.env.runs / "fake-1b" / res.name / "server.log")
    assert (res / "capability/results.json").exists()
    lock = o.lines("lock.log")
    assert "acquire modeleval fake-1b mechanics check, accuracy only (untimed)" in lock[0]
    assert any(x.startswith("update-purpose modeleval fake-1b capability, accuracy only (untimed)") for x in lock)
    assert lock[-1].startswith("release") and all(x.endswith(f"run={res.name}") for x in lock)
    for ln in man["launches"]:
        assert not group_alive(ln["pid"])
    assert not (o.env.runs / "fake-1b" / res.name / "ssd-cache").exists()   # deleted after a complete run
    _assert_clean_exit(o)


def test_verified_speed_leg_retries_disturbed_and_keeps_invalid_out(orch, capsys):
    o = orch
    o.control(disturb={"m3-16k": 1}, invalid={"m1": 1})
    assert o.go(legs=("speed",)) == 4                            # m1 invalid: incomplete, resumable
    out = capsys.readouterr().out
    assert "LOCK-NOTICE" in out and "timed_minutes=" in out
    res = _only_dir(paths.RESULTS / "fake-1b")
    units = json.loads((res / "speed/units.json").read_text())["units"]
    m3 = units["m3-16k"]
    assert m3["verdict"] == "clean" and [a["verdict"] for a in m3["attempts"]] == ["disturbed", "clean"]
    assert not (res / "speed/m3-16k-1.jsonl").exists() and (res / "speed/m3-16k-2.jsonl").exists()
    assert (paths.UNVERIFIED / "fake-1b" / res.name / "speed/m3-16k-1.jsonl").exists()
    assert "fakeproc" in m3["attempts"][0]["reasons"][0]
    assert units["m1"]["verdict"] == "invalid" and len(units["m1"]["attempts"]) == 1
    assert (paths.UNVERIFIED / "fake-1b" / res.name / "speed/m1-1.jsonl").exists()
    assert "cached_tokens" in units["m1"]["attempts"][0]["note"]
    lock = o.lines("lock.log")
    assert lock[0].startswith("acquire modeleval fake-1b speed (timed)")
    assert any(x.startswith("update-purpose modeleval fake-1b capability") for x in lock)
    checks = o.lines("quiet.log")
    server_pid = json.loads((res / "manifest.json").read_text())["launches"][0]["pid"]
    assert any("--max-gpu-gb 6" in c for c in checks)            # pre-launch check
    assert sum(f"--allow-pid {server_pid}" in c for c in checks) >= len(UNITS) + 1   # every unit and the retry
    assert o.cap_log() == []                                     # capability leg not requested
    _assert_clean_exit(o)

    # --resume repeats only the unit without a clean result, on a new launch, without the functional block
    capsys.readouterr()
    o.control()
    o.state.fake.__dict__["_seen"].clear()
    before = o.lines("speed.log")
    assert o.go(legs=("speed",), resume=res.name) == 0
    new = o.lines("speed.log")[len(before):]
    assert [x.split()[:2] for x in new] == [["unit", "m1"]]
    units = json.loads((res / "speed/units.json").read_text())["units"]
    assert units["m1"]["verdict"] == "clean" and [a["launch"] for a in units["m1"]["attempts"]] == [1, 2]
    man = json.loads((res / "manifest.json").read_text())
    assert [ln["launch"] for ln in man["launches"]] == [1, 2] and man["status"] == "complete"
    _assert_clean_exit(o)


def test_minor_disturbance_keeps_the_unit_and_major_repeats_it(orch, capsys):
    """DESIGN §5.4, the user's instruction of 2026-10-26 (small disturbances: skip them, no re-run): a
    minor flag keeps the attempt with its reasons; a major one repeats the unit."""
    o = orch
    o.control(units=["m3-16k", "m4-1", "m1"], minor={"m3-16k": 1}, disturb={"m4-1": 1}, invalid={"m1": 1})
    assert o.go(legs=("speed",)) == 4                            # m1 invalid: incomplete, resumable
    out = capsys.readouterr().out
    res = _only_dir(paths.RESULTS / "fake-1b")
    units = json.loads((res / "speed/units.json").read_text())["units"]
    m3 = units["m3-16k"]
    (a,) = m3["attempts"]
    assert m3["verdict"] == "minor" and a["verdict"] == "minor" and a["kept"] is True and a["severity"] == "minor"
    assert a["major_reasons"] == [] and "smallproc" in a["minor_reasons"][0] and a["minor_samples"] >= 1
    assert a["major_samples"] == 0 and "smallproc" in a["reasons"][0]
    assert (res / "speed/m3-16k-1.jsonl").exists()               # kept in results/, not moved
    m4 = units["m4-1"]
    assert m4["verdict"] == "clean" and [x["verdict"] for x in m4["attempts"]] == ["disturbed", "clean"]
    assert m4["attempts"][0]["severity"] == "major" and "fakeproc" in m4["attempts"][0]["major_reasons"][0]
    done = [ln for ln in out.splitlines() if " UNIT-DONE " in ln]
    assert any("unit=m3-16k" in ln and "verdict=minor" in ln and "smallproc" in ln for ln in done), done
    assert any("unit=m4-1" in ln and "verdict=disturbed" in ln and "fakeproc" in ln for ln in done), done
    checks = o.lines("quiet.log")
    unit_checks = [c for c in checks if "--min-battery 40" in c]
    assert len(unit_checks) == 4 and all("--major-only" in c for c in unit_checks)   # 3 units + 1 retry
    assert all("--major-only" in c for c in checks if "--max-gpu-gb 6" in c)         # pre-launch: major rule too
    _assert_clean_exit(o)

    # --resume repeats only m1; the unit kept with a minor flag is done
    capsys.readouterr()
    o.control(units=["m3-16k", "m4-1", "m1"])
    o.state.fake.__dict__["_seen"].clear()
    before = o.lines("speed.log")
    assert o.go(legs=("speed",), resume=res.name) == 0
    assert [x.split()[:2] for x in o.lines("speed.log")[len(before):]] == [["unit", "m1"]]
    man = json.loads((res / "manifest.json").read_text())
    assert man["status"] == "complete"
    assert json.loads((res / "speed/units.json").read_text())["units"]["m3-16k"]["verdict"] == "minor"
    _assert_clean_exit(o)


def test_battery_pause_gives_the_machine_back_and_relaunches(orch, capsys):
    o = orch
    o.control(units=["g5+m3-64k", "m3-16k", "m4-1"])
    real = o.state.fake.run_unit

    async def draining(ctx, unit, attempt):
        r = await real(ctx, unit, attempt)
        if unit == "m3-16k":
            o.state.pct = 55          # below 60 before M4: the long-unit rule pauses
        return r

    o.state.fake.run_unit = draining

    def charge():
        if o.state.pct < 80:
            o.state.pct = 85

    o.state.hooks.append(charge)
    assert o.go(legs=("speed",)) == 0
    tags = _tags(capsys.readouterr().out)
    i = tags.index("BATTERY-PAUSE")
    assert tags[i:i + 4] == ["BATTERY-PAUSE", "BATTERY-READY", "LOCK-NOTICE", "RESUMED"]
    lock = [x.split()[0] for x in o.lines("lock.log")]
    assert lock == ["acquire", "release", "acquire", "update-purpose", "release"]
    res = _only_dir(paths.RESULTS / "fake-1b")
    man = json.loads((res / "manifest.json").read_text())
    assert [ln["reason"] for ln in man["launches"]] == ["initial", "battery-pause"]
    units = json.loads((res / "speed/units.json").read_text())["units"]
    assert units["m3-16k"]["attempts"][0]["launch"] == 1 and units["m4-1"]["attempts"][0]["launch"] == 2
    assert len(man["memory"]) == 2
    _assert_clean_exit(o)


def test_capability_yield_on_low_battery_pauses_and_resumes(orch, monkeypatch, capsys):
    o = orch
    monkeypatch.delenv("FAKE_CAP_HOLD")          # this yield is the battery's, not a hold
    o.cap_codes(75, 0)
    real_run_runner = run.Run.run_runner

    def runner(self, spec):                      # the runner yields because the battery fell to 55 %
        rc = real_run_runner(self, spec)
        if rc == 75:
            o.state.pct = 55
        return rc

    def charge():
        if o.state.pct < 80:
            o.state.pct = 85

    o.state.hooks.append(charge)
    monkeypatch.setattr(run.Run, "run_runner", runner)
    assert o.go(legs=("capability",)) == 0
    tags = _tags(capsys.readouterr().out)
    assert "BATTERY-PAUSE" in tags and "RESUMED" in tags and "QUIET-WINDOW-END" not in tags
    caps = o.cap_log()
    assert [c["resume"] for c in caps] == [False, True, True] and caps[0]["server_pid"] != caps[1]["server_pid"]
    assert [c["sets"] for c in caps] == [["gate"], ["gate"], ["ranking"]]
    lock = o.lines("lock.log")
    assert lock[0].startswith("acquire modeleval fake-1b capability, accuracy only (untimed)")
    _assert_clean_exit(o)


def test_gate_fail_stops_before_any_timed_unit(orch, capsys):
    o = orch
    o.control(functional_pass=False)
    assert o.go() == 3
    out = capsys.readouterr().out
    assert "status=gate-fail" in out and "UNIT-START" not in out
    assert not any(x.startswith("unit") for x in o.lines("speed.log"))
    assert o.cap_log() == []
    _assert_clean_exit(o)


def test_quiet_that_never_settles_stops_the_speed_leg_but_not_the_capability_leg(orch, capsys):
    o = orch
    (o.env.root / "quiet-checks").write_text("quiet\n" + "busy\n" * 2000)
    assert o.go() == 4
    out = capsys.readouterr().out
    assert "UNIT-WAIT" in out and "speed leg stopped" in out and "QUIET-WINDOW-END" in out
    assert [c["sets"] for c in o.cap_log()] == [["gate"], ["ranking"]]   # capability ran on the same launch
    assert any(x.startswith("update-purpose") for x in o.lines("lock.log"))
    _assert_clean_exit(o)


def test_lock_held_by_another_session_refuses_without_launching(orch, monkeypatch, capsys):
    o = orch
    monkeypatch.setattr(run, "LOCK_WAIT_S", 0)
    o.env.lock.write_text("holder: local_ai-benchmark session\npurpose: someone else\nrun: other\n")
    assert o.go() == 5
    out = capsys.readouterr().out
    assert "UNIT-WAIT unit=lock" in out or "held by another session" in out
    assert o.env.lock.read_text().endswith("run: other\n")         # untouched
    assert not (o.env.root / "omlx-requests").exists()
    assert not paths.RESULTS.exists()                              # nothing measured, nothing in results/


def test_port_8500_up_releases_the_new_lock(orch, monkeypatch, capsys):
    o = orch
    (o.env.root / "acq-rc").write_text("3")
    monkeypatch.setenv("FAKE_LOCK_ACQUIRE_RC", str(o.env.root / "acq-rc"))
    assert o.go() == 5
    assert "port 8500" in capsys.readouterr().out
    assert not o.env.lock.exists()
    assert not (o.env.root / "omlx-requests").exists()


def test_missing_registration_is_refused(orch, capsys):
    (paths.MODELS_DIR / "Fake-1B-mtp").unlink()
    assert orch.go() == 5
    assert "modeleval register" in capsys.readouterr().out
    assert not orch.env.lock.exists()


def test_dry_run_touches_nothing(orch, capsys):
    o = orch
    assert run.run(run.Options(model="fake-1b", dry_run=True), o.env.models) == 0
    out = capsys.readouterr().out
    assert "unit        m3-16k" in out and "Speculative backend" in out and "epoch test-epoch" in out
    assert not paths.RESULTS.exists() and not (paths.RUNS / "fake-1b").exists()
    assert not o.env.lock.exists() and not (o.env.root / "lock.log").exists()
    assert not (o.env.root / "omlx-requests").exists() and not paths.BASE.exists()


def test_sigterm_cleans_up_children_port_and_lock(orch, tmp_path):
    o = orch
    cfg = {
        "paths": {k: (getattr(paths, k) if k in ("PORT", "BASE_URL") else str(getattr(paths, k)))
                  for k in ("OMLX_REPO", "OMLX_EXE", "OMLX_PY", "BASE", "MODELS_DIR", "PORT", "BASE_URL", "RESULTS",
                            "UNVERIFIED", "RUNS", "VENDOR", "SCRIPTS", "GPU_LOCK", "HOLD_FILE", "HF_HOME",
                            "WS_ROOT", "REPO")},
        "fake_speed": str(FAKES / "fake_speed.py"), "cap_python": str(run.CAP_PYTHON),
        "models": {"models": {"fake-1b": model_table(o.env.ckpt)}},
        "options": {"model": "fake-1b", "notice_wait": 0, "settle_s": 5, "legs": ["capability"]},
    }
    (tmp_path / "drive.json").write_text(json.dumps(cfg))
    env = {**os.environ, "FAKE_CAP_SLEEP": "120", "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.Popen([sys.executable, "-B", str(FAKES / "drive_run.py"), str(tmp_path / "drive.json")],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while not o.cap_log() and time.time() < deadline and p.poll() is None:
        time.sleep(0.2)
    assert o.cap_log(), p.stdout.read() if p.poll() is not None else "runner never started"
    cap_pid = o.cap_log()[0]["pid"]
    server_pid = o.cap_log()[0]["server_pid"]
    assert o.env.lock.exists() and port_listening(o.env.port)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=90)
    assert p.returncode == 130, out
    assert "status=interrupted" in out
    assert not group_alive(cap_pid) and not group_alive(server_pid)
    _assert_clean_exit(o)
    man = json.loads(next(paths.RESULTS.glob("fake-1b/*/manifest.json")).read_text())
    assert man["status"] == "interrupted"


def test_units_json_merges_with_the_real_speed_writers(orch, monkeypatch):
    """speed.py records each attempt itself; run.py must add verdicts without dropping speed's keys."""
    pytest.importorskip("modeleval.speed")
    monkeypatch.setenv("FAKE_SPEED_RECORD", "1")
    o = orch
    o.state.fake = _load_fake_speed()                  # re-import: finalize exists only with the env set
    monkeypatch.setattr(run, "load_speed", lambda: o.state.fake)
    o.control(units=["m3-16k", "m1"], disturb={"m3-16k": 1})
    assert o.go(legs=("speed",)) == 0
    res = _only_dir(paths.RESULTS / "fake-1b")
    doc = json.loads((res / "speed/units.json").read_text())
    assert doc["protocol_version"] == paths.PROTOCOL_VERSION and doc["model_info"]["key"] == "fake-1b"
    assert doc["functional"]["passed"] is True and doc["functional"]["launch"] == 1
    m3 = doc["units"]["m3-16k"]
    assert m3["verdict"] == "clean" and [a["verdict"] for a in m3["attempts"]] == ["disturbed", "clean"]
    assert m3["attempts"][0]["raw_file"].startswith("../../_unverified/") and m3["attempts"][1]["kept"] is True
    assert all("metrics" in a and a["launch"] == 1 for a in m3["attempts"])
    assert "memory" in doc                              # speed.finalize ran after the last write
    assert "/Users/" not in (res / "speed/units.json").read_text()


def test_resume_after_a_crash_kills_orphans_and_adopts_its_own_lock(orch, capsys):
    o = orch
    o.control(units=["m3-16k", "m1"], invalid={"m1": 1})
    assert o.go(legs=("speed",)) == 4
    res = _only_dir(paths.RESULTS / "fake-1b")
    # what a SIGKILLed orchestrator leaves behind: its lock (same run id) and a child in the pidfile
    o.env.lock.write_text(f"holder: local_ai-benchmark session\npurpose: modeleval fake-1b speed (timed)\n"
                          f"run: {res.name}\n")
    orphan = subprocess.Popen(["sleep", "300"], start_new_session=True)
    from modeleval.server import Pidfile
    Pidfile(o.env.runs / "fake-1b" / res.name / "pidfile").add(orphan.pid, "monitor", "sleep")
    # and an attempt it ran but never judged: its raw file must not stay in results/
    uj = res / "speed/units.json"
    doc = json.loads(uj.read_text())
    doc["units"]["m1"]["attempts"].append({"attempt": 2, "raw_file": "speed/m1-2.jsonl", "verdict": None, "kept": False})
    uj.write_text(json.dumps(doc))
    (res / "speed/m1-2.jsonl").write_text("{}\n")
    capsys.readouterr()
    before = len(o.lines("lock.log"))
    o.control(units=["m3-16k", "m1"])
    o.state.fake.__dict__["_seen"].clear()
    assert o.go(legs=("speed",), resume=res.name) == 0
    out = capsys.readouterr().out
    assert orphan.wait(timeout=10) is not None
    assert "stale child from the previous attempt: monitor" in out
    assert "LOCK-NOTICE" in out                                   # adopted, not re-acquired, but a timed
    calls = [x.split()[0] for x in o.lines("lock.log")[before:]]  # leg still gets its notice (DESIGN §5)
    assert calls[0] == "update-purpose" and "acquire" not in calls and calls[-1] == "release"
    m1 = json.loads((res / "speed/units.json").read_text())["units"]["m1"]
    assert [a["verdict"] for a in m1["attempts"]] == ["invalid", "interrupted", "clean"]
    assert not (res / "speed/m1-2.jsonl").exists()
    assert (paths.UNVERIFIED / "fake-1b" / res.name / "speed/m1-2.jsonl").exists()
    _assert_clean_exit(o)


# ---------------------------------------------------------------- review fixes (2026-10-26)

HARNESSES = ("claude_code", "hermes", "localmelo")
GATE = [f"aider/g{i:02d}" for i in range(22)]
RANK = [f"aider/r{i:02d}" for i in range(22)]
HOLD_FLOCK = ("import fcntl, os, sys, time\n"
              "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
              "fcntl.flock(fd, fcntl.LOCK_EX)\n"
              "os.write(fd, b'4242\\n')\n"
              "print('ready', flush=True)\n"
              "time.sleep(120)\n")


def _rows(set_name, tasks, solved=lambda h, t: True):
    return [{"harness": h, "task": t, "set": set_name, "verdict": "pass" if solved(h, t) else "fail",
             "check": {"passed": solved(h, t)}, "valid": True, "scored": True} for t in tasks for h in HARNESSES]


@pytest.fixture
def newmodel(orch, monkeypatch):
    """A non-reference model `new-1b` next to the reference fake-1b, whose finished gate set (all solved)
    is in results/ for the harness epoch the runs use."""
    from modeleval.registry import parse_models

    env = orch.env
    ckpt = env.root / "models/oq/New-1B"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (paths.MODELS_DIR / "New-1B").symlink_to(ckpt)
    models = parse_models({"models": {"fake-1b": model_table(env.ckpt),
                                      "new-1b": model_table(ckpt, served_id="New-1B", reference=False, display="New")}})
    ref = paths.RESULTS / "fake-1b" / "20261020-000000"
    (ref / "capability").mkdir(parents=True)
    (ref / "manifest.json").write_text(json.dumps({"model": models["fake-1b"].to_dict(), "protocol_version": 1,
                                                    "epoch": "test-epoch", "started": "2026-10-20T00:00:00"}))
    (ref / "capability/results.json").write_text(json.dumps({"epoch": "test-epoch", "sets": ["gate"], "finished": True,
                                                             "complete_sets": {"gate": True}, "rows": _rows("gate", GATE)}))

    def go(rows_by_set, **kw):
        (env.root / "cap-rows.json").write_text(json.dumps(rows_by_set))
        monkeypatch.setenv("FAKE_CAP_ROWS", str(env.root / "cap-rows.json"))
        opts = {"legs": ("capability",), **kw}
        return run.run(run.Options(model="new-1b", notice_wait=0, settle_s=5, **opts), models)

    return SimpleNamespace(go=go, models=models)


def _worse(h, t):   # loses five tasks in two harnesses: "worse" by the gate rule of DESIGN §6.6
    return not (h in ("claude_code", "hermes") and t in GATE[:5])


def test_a_worse_gate_set_skips_the_ranking_set(orch, newmodel, capsys):
    assert newmodel.go({"gate": _rows("gate", GATE, _worse), "ranking": _rows("ranking", RANK)}) == 0
    out = capsys.readouterr().out
    assert [c["sets"] for c in orch.cap_log()] == [["gate"]]
    assert "GATE-DECISION decision=worse action=skip wins=0 losses=5" in out
    man = json.loads(next(paths.RESULTS.glob("new-1b/*/manifest.json")).read_text())
    assert man["status"] == "complete" and man["gate_decision"]["decision"] == "worse"
    assert man["gate_decision"]["reference_run"] == "fake-1b/20261020-000000"
    _assert_clean_exit(orch)


def test_a_gate_set_that_is_not_worse_goes_on_to_the_ranking_set(orch, newmodel, capsys):
    assert newmodel.go({"gate": _rows("gate", GATE), "ranking": _rows("ranking", RANK)}) == 0
    assert 'decision="not worse" action=ranking' in capsys.readouterr().out
    assert [c["sets"] for c in orch.cap_log()] == [["gate"], ["ranking"]]
    res = json.loads(next(paths.RESULTS.glob("new-1b/*/capability/results.json")).read_text())
    assert res["sets"] == ["gate", "ranking"] and res["complete_sets"] == {"gate": True, "ranking": True}


def test_full_force_runs_the_ranking_set_after_a_worse_gate(orch, newmodel, capsys):
    assert newmodel.go({"gate": _rows("gate", GATE, _worse), "ranking": _rows("ranking", RANK)},
                       capability="full-force") == 0
    assert "GATE-DECISION" not in capsys.readouterr().out
    assert [c["sets"] for c in orch.cap_log()] == [["gate"], ["ranking"]]


def test_a_resume_with_ranking_rows_goes_on_without_re_deciding(orch, newmodel, capsys):
    orch.cap_codes(0, 1)                         # the gate set finishes, the ranking runner fails
    assert newmodel.go({"gate": _rows("gate", GATE), "ranking": _rows("ranking", RANK)}) == 1
    run_id = _only_dir(paths.RESULTS / "new-1b").name
    state = orch.env.runs / "new-1b" / run_id / "capability/state.json"
    doc = json.loads(state.read_text())
    doc["records"].append(_rows("ranking", RANK[:1])[0])   # what the failed ranking runner had saved
    state.write_text(json.dumps(doc))
    capsys.readouterr()
    assert newmodel.go({"gate": [], "ranking": _rows("ranking", RANK[1:])}, resume=run_id) == 0
    assert "GATE-DECISION" not in capsys.readouterr().out
    assert [c["sets"] for c in orch.cap_log()] == [["gate"], ["ranking"], ["gate"], ["ranking"]]


def test_a_server_lost_in_the_functional_block_is_relaunched_not_a_gate_fail(orch, capsys):
    o = orch
    o.control(units=["m3-16k"], functional_kill=1)
    assert o.go(legs=("speed",)) == 0
    out = capsys.readouterr().out
    assert "gate-fail" not in out and "server lost (functional block: " in out
    res = _only_dir(paths.RESULTS / "fake-1b")
    man = json.loads((res / "manifest.json").read_text())
    assert [ln["reason"] for ln in man["launches"]] == ["initial", "functional-server-lost"]
    units = json.loads((res / "speed/units.json").read_text())
    assert units["functional"]["passed"] is True and "server_lost" not in units["functional"]
    _assert_clean_exit(o)


def test_a_server_lost_twice_in_the_functional_block_stops_resumably(orch, capsys):
    o = orch
    o.control(units=["m3-16k"], functional_kill=2)
    assert o.go(legs=("speed",)) == 4
    out = capsys.readouterr().out
    assert "server lost in the functional block" in out and out.splitlines()[-1].endswith("resume=yes")
    units = json.loads((_only_dir(paths.RESULTS / "fake-1b") / "speed/units.json").read_text())
    assert units["functional"]["passed"] is None and units["functional"]["server_lost"]
    _assert_clean_exit(o)


def test_a_gate_fail_names_the_failing_item(orch, capsys):
    orch.control(functional_pass=False)
    assert orch.go() == 3
    assert "functional block FAIL: G1 failed (G1: no tool call)" in capsys.readouterr().out


def test_an_incomplete_attempt_is_repeated_and_a_cache_hit_is_not(orch):
    o = orch
    o.control(units=["m1", "m4-1", "m4-2"], invalid={"m1": 1}, incomplete={"m4-1": 1, "m4-2": 3})
    assert o.go(legs=("speed",)) == 4
    res = _only_dir(paths.RESULTS / "fake-1b")
    units = json.loads((res / "speed/units.json").read_text())["units"]
    assert units["m1"]["verdict"] == "invalid" and len(units["m1"]["attempts"]) == 1          # final
    assert [a["verdict"] for a in units["m4-1"]["attempts"]] == ["incomplete", "clean"]
    assert units["m4-1"]["verdict"] == "clean" and "latency sample" in units["m4-1"]["attempts"][0]["note"]
    assert units["m4-2"]["verdict"] == "incomplete x3"
    assert (paths.UNVERIFIED / "fake-1b" / res.name / "speed/m4-1-1.jsonl").exists()
    assert not (o.env.runs / "fake-1b" / res.name / "ssd-cache").exists()   # deleted for an incomplete run too
    _assert_clean_exit(o)


def test_too_little_disk_for_the_ssd_cache_is_refused_before_the_lock(orch, monkeypatch, capsys):
    from modeleval import prepare

    monkeypatch.setattr(run, "SSD_CACHE_BYTES", 128e9)
    monkeypatch.setattr(prepare, "free_bytes", lambda p: 50e9)
    assert orch.go() == 5
    out = capsys.readouterr().out
    assert "SSD cache" in out and "resume=no" in out
    assert not orch.env.lock.exists() and not (orch.env.root / "omlx-requests").exists()


def test_plan_lists_leftover_ssd_caches(orch, capsys):
    left = paths.RUNS / "fake-1b" / "20261001-000000" / "ssd-cache"
    left.mkdir(parents=True)
    (left / "block").write_bytes(b"\0" * 4096)
    assert run.run(run.Options(model="fake-1b", dry_run=True), orch.env.models) == 0
    out = capsys.readouterr().out
    assert "leftover    " in out and "20261001-000000/ssd-cache" in out and "ready       disk" in out


def test_a_resume_on_another_build_is_refused_and_leaves_the_run_alone(orch, monkeypatch, capsys):
    o = orch
    o.control(units=["m3-16k", "m1"], invalid={"m1": 1})
    assert o.go(legs=("speed",)) == 4
    res = _only_dir(paths.RESULTS / "fake-1b")
    before = (res / "manifest.json").read_text()
    monkeypatch.setattr(run, "machine_info", lambda: {"sw_vers": {"BuildVersion": "newer"}})
    capsys.readouterr()
    assert o.go(legs=("speed",), resume=res.name) == 5
    out = capsys.readouterr().out
    assert "start a new run" in out and "resume=no" in out
    assert (res / "manifest.json").read_text() == before             # machine and status untouched
    # the capability leg alone may continue on the new build; the run keeps its first machine
    assert o.go(legs=("capability",), resume=res.name) == 0
    man = json.loads((res / "manifest.json").read_text())
    assert man["machine"]["sw_vers"]["BuildVersion"] == "test"
    assert man["launches"][0]["builds"]["macos"] == "test" and man["launches"][-1]["builds"]["macos"] == "newer"


def test_a_second_orchestrator_of_the_same_run_is_refused_and_signals_nothing(orch, capsys):
    o = orch
    o.control(units=["m1"], invalid={"m1": 1})
    assert o.go(legs=("speed",)) == 4
    res = _only_dir(paths.RESULTS / "fake-1b")
    runs_dir = o.env.runs / "fake-1b" / res.name
    from modeleval.server import Pidfile

    child = subprocess.Popen(["sleep", "300"], start_new_session=True)   # a child of the live instance
    Pidfile(runs_dir / "pidfile").add(child.pid, "monitor", "sleep")
    holder = subprocess.Popen([sys.executable, "-c", HOLD_FLOCK, str(runs_dir / "orchestrator.pid")],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        o.env.lock.write_text(f"holder: local_ai-benchmark session\npurpose: x\nrun: {res.name}\n")
        capsys.readouterr()
        assert o.go(legs=("speed",), resume=res.name) == 5
        out = capsys.readouterr().out
        assert "already running (PID 4242)" in out
        assert child.poll() is None and Pidfile(runs_dir / "pidfile").entries()   # not signalled, not removed
        assert o.env.lock.exists()                                                # the live run's lock stays
    finally:
        holder.kill()
        child.kill()
        holder.wait()
        child.wait()
    o.env.lock.unlink()


def test_an_interrupted_acquire_still_releases_our_lock(orch, monkeypatch, capsys):
    o = orch
    real = run.Run.lock_sh

    def lock_sh(self, *args):
        if args[0] == "acquire":   # killed after the noclobber create: the lock carries our run id
            o.env.lock.write_text(f"holder: local_ai-benchmark session\npurpose: x\nrun: {self.run_id}\n")
            return 1, "TimeoutExpired"
        return real(self, *args)

    monkeypatch.setattr(run.Run, "lock_sh", lock_sh)
    assert o.go() == 5
    assert not o.env.lock.exists()


def test_a_lock_refusal_after_a_launch_is_resumable(orch, monkeypatch, capsys):
    o = orch
    o.control(units=["g5+m3-64k", "m3-16k", "m4-1"])
    real = o.state.fake.run_unit

    async def draining(ctx, unit, attempt):
        r = await real(ctx, unit, attempt)
        if unit == "m3-16k":
            o.state.pct = 55
            (o.env.root / "acq-rc").write_text("3")   # port 8500 is up when the lock is retaken
        return r

    o.state.fake.run_unit = draining
    monkeypatch.setenv("FAKE_LOCK_ACQUIRE_RC", str(o.env.root / "acq-rc"))

    def charge():
        if o.state.pct < 80:
            o.state.pct = 85

    o.state.hooks.append(charge)
    assert o.go(legs=("speed",)) == 4
    out = capsys.readouterr().out
    assert "status=stopped" in out and "resume=yes" in out
    _assert_clean_exit(o)


def test_a_dead_monitor_is_restarted_before_the_next_unit(orch, capsys):
    o = orch
    o.control(units=["m3-16k", "m1"])
    real = o.state.fake.run_unit

    async def kill_monitor(ctx, unit, attempt):
        r = await real(ctx, unit, attempt)
        if unit == "m3-16k":
            from modeleval.server import Pidfile
            mon = next(e for e in Pidfile(Path(ctx.run_dir).parents[2] / "runs" / "fake-1b" / Path(ctx.run_dir).name
                                          / "pidfile").entries() if e["role"] == "monitor")
            os.killpg(mon["pgid"], signal.SIGKILL)
        return r

    o.state.fake.run_unit = kill_monitor
    assert o.go(legs=("speed",)) == 0
    out = capsys.readouterr().out
    assert "quiet monitor was not running; restarted" in out
    units = json.loads((_only_dir(paths.RESULTS / "fake-1b") / "speed/units.json").read_text())["units"]
    assert units["m1"]["verdict"] == "clean"
    _assert_clean_exit(o)


def test_quiet_checks_allow_thermal_and_guard_the_battery(orch):
    o = orch
    o.control(units=["m3-16k"])
    assert o.go(legs=("speed",)) == 0
    checks = o.lines("quiet.log")
    launch = next(c for c in checks if "--max-gpu-gb 6" in c)
    assert "--min-battery 60" in launch and "--allow thermal" in launch
    unit = [c for c in checks if "--max-gpu-gb" not in c and c.startswith("check")]
    assert unit and all("--min-battery 40" in c and "--allow thermal" in c for c in unit)


def test_an_ignored_sighup_stays_ignored(orch, monkeypatch):
    seen = {}

    def execute(self):
        seen["hup"] = signal.getsignal(signal.SIGHUP)
        raise run.Stop("refused", "test")

    monkeypatch.setattr(run.Run, "execute", execute)
    old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        assert orch.go() == 5
        assert seen["hup"] is signal.SIG_IGN and signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, old)


def test_a_resume_after_the_speed_leg_takes_an_untimed_lock(orch, capsys):
    o = orch
    o.control(units=["m3-16k"])
    assert o.go(legs=("speed",)) == 0
    res = _only_dir(paths.RESULTS / "fake-1b")
    before = len(o.lines("lock.log"))
    capsys.readouterr()
    assert o.go(resume=res.name) == 0
    lock = o.lines("lock.log")[before:]
    assert lock[0].startswith("acquire modeleval fake-1b capability, accuracy only (untimed)")
    assert "timed_minutes=0" in capsys.readouterr().out


def test_sigterm_with_a_dead_stdout_still_cleans_up(orch, tmp_path):
    """The reader of the orchestrator's stdout dies (a `| tee` killed by the same Ctrl-C), then SIGTERM:
    every progress line now fails with BrokenPipeError, and cleanup must still stop the server and the
    runner and release the lock."""
    o = orch
    cfg = {
        "paths": {k: (getattr(paths, k) if k in ("PORT", "BASE_URL") else str(getattr(paths, k)))
                  for k in ("OMLX_REPO", "OMLX_EXE", "OMLX_PY", "BASE", "MODELS_DIR", "PORT", "BASE_URL", "RESULTS",
                            "UNVERIFIED", "RUNS", "VENDOR", "SCRIPTS", "GPU_LOCK", "HOLD_FILE", "HF_HOME",
                            "WS_ROOT", "REPO")},
        "fake_speed": str(FAKES / "fake_speed.py"), "cap_python": str(run.CAP_PYTHON),
        "models": {"models": {"fake-1b": model_table(o.env.ckpt)}},
        "options": {"model": "fake-1b", "notice_wait": 0, "settle_s": 5, "legs": ["capability"]},
    }
    (tmp_path / "drive.json").write_text(json.dumps(cfg))
    env = {**os.environ, "FAKE_CAP_SLEEP": "120", "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.Popen([sys.executable, "-B", str(FAKES / "drive_run.py"), str(tmp_path / "drive.json")],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while not o.cap_log() and time.time() < deadline and p.poll() is None:
        time.sleep(0.2)
    assert o.cap_log(), p.stdout.read() if p.poll() is not None else "runner never started"
    cap_pid, server_pid = o.cap_log()[0]["pid"], o.cap_log()[0]["server_pid"]
    p.stdout.close()                     # the reader is gone
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=90) == 130
    assert not group_alive(cap_pid) and not group_alive(server_pid)
    _assert_clean_exit(o)
    run_dir = next(paths.RESULTS.glob("fake-1b/*"))
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "interrupted"
    log = (o.env.runs / "fake-1b" / run_dir.name / "progress.log").read_text()
    assert "RUN-DONE status=interrupted" in log   # the lines the dead pipe lost
