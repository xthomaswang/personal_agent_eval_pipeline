"""gpu_lock.sh under /bin/bash (3.2) with the lock in tmp_path and a fake lsof first on PATH."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/gpu_lock.sh"
BASH = "/bin/bash"
HOLDER = "local_ai-benchmark session"


def lock(env: dict[str, str], *args: str, run_id: str | None = None, listeners: str = "") -> subprocess.CompletedProcess:
    e = dict(env, FAKE_LISTENERS=listeners)
    if run_id is not None:
        e["MODELEVAL_RUN_ID"] = run_id
    return subprocess.run([BASH, str(SCRIPT), *args], env=e, capture_output=True, text=True, timeout=30)


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def lock_path(lock_env) -> Path:
    return Path(lock_env["GPU_LOCK_PATH"])


def test_acquire_writes_holder_purpose_start_eta(lock_env, lock_path):
    r = lock(lock_env, "acquire", "modeleval qwen speed (timed)", "45")
    assert r.returncode == 0, r.stdout + r.stderr
    got = lines(lock_path)
    assert got[0] == f"holder: {HOLDER}" and got[1] == "purpose: modeleval qwen speed (timed)"
    assert got[2].startswith("start: 20") and got[3].startswith("eta: 20") and got[3].endswith(" (45 min)")
    assert len(got) == 4                                  # no run line without MODELEVAL_RUN_ID


def test_acquire_adds_run_line(lock_env, lock_path):
    assert lock(lock_env, "acquire", "p", "5", run_id="20261026-0300").returncode == 0
    assert lines(lock_path)[-1] == "run: 20261026-0300"


def test_acquire_rejects_bad_run_id_and_arguments(lock_env, lock_path):
    assert lock(lock_env, "acquire", "p", "5", run_id="a b").returncode == 2
    assert lock(lock_env, "acquire", "p", "0").returncode == 2
    assert lock(lock_env, "acquire", "p", "ten").returncode == 2
    assert lock(lock_env, "acquire", "two\nlines", "5").returncode == 2
    assert lock(lock_env, "acquire").returncode == 2
    assert not lock_path.exists()


def test_acquire_twice_is_refused(lock_env, lock_path):
    assert lock(lock_env, "acquire", "first", "5").returncode == 0
    r = lock(lock_env, "acquire", "second", "5")
    assert r.returncode == 2 and "already held" in r.stdout and "purpose: first" in r.stdout
    assert lines(lock_path)[1] == "purpose: first"


def test_acquire_keeps_lock_but_exits_3_while_port_8500_is_up(lock_env, lock_path):
    r = lock(lock_env, "acquire", "p", "5", listeners="8500:4242")
    assert r.returncode == 3 and "port 8500 is up (port pid): 8500 4242" in r.stdout
    assert lock_path.exists()


def test_acquire_only_notes_the_other_agent_benchmark_ports(lock_env, lock_path):
    r = lock(lock_env, "acquire", "p", "5", listeners="8642:4243")
    assert r.returncode == 0 and "note: agent-benchmark port(s) in 8600-8699 are listening" in r.stdout
    assert "8642 4243" in r.stdout and lock_path.exists()


def test_acquire_notes_our_port(lock_env):
    r = lock(lock_env, "acquire", "p", "5", listeners="8703:999")
    assert r.returncode == 0 and "port 8703 is already in use" in r.stdout


def test_release(lock_env, lock_path):
    assert lock(lock_env, "release").stdout.startswith("no GPU lock")
    lock(lock_env, "acquire", "p", "5")
    r = lock(lock_env, "release")
    assert r.returncode == 0 and not lock_path.exists()


def test_release_refuses_other_holder(lock_env, lock_path):
    lock_path.write_text("holder: agent-benchmark\npurpose: their run\n", encoding="utf-8")
    r = lock(lock_env, "release")
    assert r.returncode == 2 and "held by someone else" in r.stdout and lock_path.exists()


def test_release_checks_run_id(lock_env, lock_path):
    lock(lock_env, "acquire", "p", "5", run_id="run-a")
    r = lock(lock_env, "release", run_id="run-b")
    assert r.returncode == 2 and "belongs to run 'run-a', not 'run-b'" in r.stdout and lock_path.exists()
    assert lock(lock_env, "release", run_id="run-a").returncode == 0 and not lock_path.exists()
    lock(lock_env, "acquire", "p", "5")                   # a lock without a run line
    assert lock(lock_env, "release", run_id="run-a").returncode == 2 and lock_path.exists()
    assert lock(lock_env, "release").returncode == 0      # a person at the terminal may release it


def test_release_refuses_while_our_server_listens(lock_env, lock_path):
    lock(lock_env, "acquire", "p", "5")
    r = lock(lock_env, "release", listeners="8703:999")
    assert r.returncode == 5 and "8703 999" in r.stdout and lock_path.exists()
    assert lock(lock_env, "release", "--force", listeners="8703:999").returncode == 0
    assert not lock_path.exists()
    assert lock(lock_env, "release", "--bogus").returncode == 2


def test_update_purpose_rewrites_in_place(lock_env, lock_path):
    lock(lock_env, "acquire", "modeleval qwen speed (timed)", "45", run_id="r1")
    before = lines(lock_path)
    inode = lock_path.stat().st_ino
    r = lock(lock_env, "update-purpose", "modeleval qwen capability, accuracy only (untimed)", "120", run_id="r1")
    assert r.returncode == 0, r.stdout + r.stderr
    after = lines(lock_path)
    assert after[0] == before[0] and after[2] == before[2] and after[4] == "run: r1"
    assert after[1] == "purpose: modeleval qwen capability, accuracy only (untimed)"
    assert after[3].startswith("eta: 20") and after[3].endswith(" (120 min)") and len(after) == 5
    assert lock_path.stat().st_ino != inode               # renamed over, not truncated and rewritten
    assert oct(lock_path.stat().st_mode & 0o777) == "0o644"
    assert sorted(p.name for p in lock_path.parent.iterdir() if p.name.startswith(".gpu-lock")) == [".gpu-lock"]
    assert lock(lock_env, "update-purpose", "again", "5").returncode == 0     # no run id: a person may


def test_update_purpose_refusals(lock_env, lock_path):
    r = lock(lock_env, "update-purpose", "x", "5")
    assert r.returncode == 2 and "no GPU lock" in r.stdout and not lock_path.exists()
    lock_path.write_text("holder: agent-benchmark\npurpose: theirs\n", encoding="utf-8")
    r = lock(lock_env, "update-purpose", "x", "5")
    assert r.returncode == 2 and lines(lock_path)[1] == "purpose: theirs"
    lock_path.unlink()
    lock(lock_env, "acquire", "mine", "5", run_id="r1")
    r = lock(lock_env, "update-purpose", "x", "5", run_id="r2")
    assert r.returncode == 2 and "not updating it" in r.stdout and lines(lock_path)[1] == "purpose: mine"
    assert lock(lock_env, "update-purpose", "x", "soon").returncode == 2
    assert lock(lock_env, "update-purpose", "x").returncode == 2


def test_update_purpose_keeps_unknown_lines_and_adds_missing_ones(lock_env, lock_path):
    lock_path.write_text(f"holder: {HOLDER}\nstart: 2026-10-26T01:00:00-0400\nnote: by hand", encoding="utf-8")
    assert lock(lock_env, "update-purpose", "p2", "10").returncode == 0
    got = lines(lock_path)
    assert got[:3] == [f"holder: {HOLDER}", "start: 2026-10-26T01:00:00-0400", "note: by hand"]
    assert got[3] == "purpose: p2" and got[4].startswith("eta: ")


def test_lock_never_disappears_during_updates(lock_env, lock_path):
    lock(lock_env, "acquire", "p", "5", run_id="r1")
    stop, missing, seen = threading.Event(), [], [0]

    def watch() -> None:
        while not stop.is_set():
            seen[0] += 1
            if not os.path.exists(lock_path):
                missing.append(seen[0])

    t = threading.Thread(target=watch)
    t.start()
    try:
        for i in range(15):
            assert lock(lock_env, "update-purpose", f"purpose {i}", "5", run_id="r1").returncode == 0
    finally:
        stop.set()
        t.join()
    assert seen[0] > 100 and missing == []
    assert lines(lock_path)[1] == "purpose: purpose 14"


def test_status(lock_env, lock_path):
    r = lock(lock_env, "status", listeners="8642:12345")
    assert r.returncode == 0 and "GPU lock: free" in r.stdout
    assert "port 8500 (agent-benchmark oMLX): free" in r.stdout
    assert "ports 8600-8699 (agent benchmark): LISTEN" in r.stdout and "8642 12345" in r.stdout
    assert "port 8703 (oMLX under test): free" in r.stdout
    lock(lock_env, "acquire", "p", "30", run_id="r9")
    r = lock(lock_env, "status")
    assert "GPU lock: held by us" in r.stdout and "run: r9" in r.stdout and "ETA in " in r.stdout


def test_help_and_usage(lock_env):
    r = lock(lock_env, "--help")
    assert r.returncode == 0 and "update-purpose <purpose> <eta-minutes>" in r.stdout
    assert lock(lock_env, "frobnicate").returncode == 2


def test_default_lock_path(tmp_path):
    """Without GPU_LOCK_PATH the lock is the shared one under ~/Developer/mlsys/experiments (only the
    variable is read here; nothing touches the file)."""
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([BASH, "-c", f'. "{REPO / "scripts/lib.sh"}"; echo "$GPU_LOCK"; tilde "$GPU_LOCK"'],
                         env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == [f"{tmp_path}/Developer/mlsys/experiments/.gpu-lock", "~/Developer/mlsys/experiments/.gpu-lock"]
