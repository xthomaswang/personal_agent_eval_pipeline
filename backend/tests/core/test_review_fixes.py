"""Regression tests for the review findings of 2026-10-26 that are not orchestrator runs (those are in
test_run.py): git status of the oMLX checkout, the shipped oq_convert.py, convert's signal handling,
the runner's one-epoch rule, the untimed pre-launch check, progress lines on a dead stdout, and the
report's mixed-build key."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeleval import cli, paths, prepare, progress, quiet, report, run
from modeleval.capability import epoch as epochs
from modeleval.capability import runner

REAL_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _git(repo: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                   check=True, capture_output=True, env={**env, "GIT_CONFIG_NOSYSTEM": "1"})


def test_the_pipelines_own_base_path_does_not_make_the_omlx_checkout_dirty(tmp_path, monkeypatch):
    repo = tmp_path / "omlx_0.7"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "a")
    monkeypatch.setattr(paths, "BASE", repo / ".modeleval-home")
    (repo / ".modeleval-home/home").mkdir(parents=True)
    (repo / ".modeleval-home/settings.json").write_text("{}")
    assert run.git_info(repo)["dirty"] is False
    assert epochs.repo_state(repo)[1] is False
    (repo / "stray.txt").write_text("someone's file\n")       # anything else still counts
    assert run.git_info(repo)["dirty"] is True and epochs.repo_state(repo)[1] is True
    # in any other repository an untracked .modeleval-home is an ordinary untracked file
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    (other / ".modeleval-home").mkdir()
    (other / ".modeleval-home/x").write_text("x")
    assert run.git_info(other)["dirty"] is True and paths.status_pathspec(other) == []


def test_the_shipped_oq_convert_has_no_register_and_never_names_bench_home(env, monkeypatch):
    shutil.copy(REAL_SCRIPTS / "oq_convert.py", env.scripts / "oq_convert.py")
    text = (env.scripts / "oq_convert.py").read_text()
    assert ".bench-home" not in text and "SETTINGS" not in text
    help_ = subprocess.run([sys.executable, "-I", "-B", str(env.scripts / "oq_convert.py"), "--help"],
                           capture_output=True, text=True, check=True).stdout
    assert "--no-register" in help_
    from modeleval.registry import parse_models

    from .conftest import model_table
    m = parse_models({"models": {"nb": model_table(env.root / "models/oq/NB", served_id="NB", spec="none",
                                                   prepare={"convert": {"level": 5}})}})["nb"]
    assert prepare.convert_command(m)[-1] == "--no-register"
    monkeypatch.setattr(paths, "SCRIPTS", REAL_SCRIPTS)
    assert next(r for r in cli.hygiene_checks() if r[1].startswith("oq_convert.py"))[0] == "ok"


def test_convert_stops_its_child_before_the_lock_is_released(env, monkeypatch, capsys):
    from modeleval.registry import parse_models

    from .conftest import model_table
    m = parse_models({"models": {"nb": model_table(env.root / "models/oq/NB-oQ5", served_id="NB-oQ5", spec="none",
                                                   prepare={"convert": {"level": 5}},
                                                   source={"hf_repo": "Org/NB", "revision": "c" * 40})}})["nb"]
    snap = prepare.snapshot_dir(m)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setattr(paths, "REPO", env.root)
    (env.scripts / "oq_convert.py").write_text("import os, time  # --no-register\nprint('child', os.getpid(), "
                                               "flush=True)\ntime.sleep(60)\n")
    before = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(prepare.ConvertInterrupted):
            prepare.convert(m)
    finally:
        timer.cancel()
    child = int(next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("child")).split()[1])
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    assert not env.lock.exists() and signal.getsignal(signal.SIGTERM) == before
    assert (env.root / "lock.log").read_text().splitlines()[-1].startswith("release")


def test_the_runner_refuses_rows_of_another_epoch(tmp_path):
    """One harness epoch per reported run (DESIGN §7): a resume after pins.toml changed must not merge
    the old rows under the new epoch."""
    run_root = tmp_path / "runs/m/r/capability"
    run_root.mkdir(parents=True)
    (run_root / "state.json").write_text(json.dumps({"records": [{"set": "gate", "harness": "hermes", "task": "t"}],
                                                     "meta": {"epoch": "old-epoch"}}))
    spec = {"model": {"served_id": "M-1", "sampling": {}, "proxy_sampling": "inject:omlx-default"},
            "base_url": "http://127.0.0.1:9/v1", "server_pid": 1, "sets": ["gate"], "run_root": str(run_root),
            "results": str(tmp_path / "results/m/r/capability/results.json"), "ws_root": str(tmp_path / "ws"),
            "epoch": "new-epoch", "vendor": str(tmp_path / "vendor/new-epoch"), "wall_s": 3600, "max_steps": 30,
            "max_turns": 40, "thinking": "off", "resume": True}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    env = runner.allowed_env(os.environ, {"PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "modeleval.capability.runner", "--spec",
                        str(tmp_path / "spec.json")], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == runner.EXIT_PREFLIGHT, r.stderr
    assert "harness epoch old-epoch" in r.stderr and "new-epoch" in r.stderr


def test_an_untimed_prelaunch_check_sees_a_resident_model_behind_a_hard_failure(monkeypatch):
    """quietlib stops at a hard failure (a peer port) before it lists soft reasons, so the GPU memory
    reason was lost and a resident foreign model did not refuse the launch."""
    text = ("quiet_check: x\nlock: held by us (p)\nport 8500: free\nports 8600-8699: LISTENING 8612 (agent benchmark)\n"
            "gpu: 0% busy (max 10), 41.5 GB in use\nsettle: not attempted (hard failure: port 8612 listening)\n"
            "verdict: NOT QUIET\n")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stdout=text, stderr="")

    monkeypatch.setattr(quiet.subprocess, "run", fake_run)
    r = quiet.prelaunch_check(timed=False)
    assert not r.ok and any(x.startswith("GPU memory in use 41.5 GB") for x in r.reasons)
    assert "--allow" in calls[0] and "thermal" in calls[0]
    assert quiet.gpu_memory_gb("gpu: 3% busy (max 10), 0.9 GB in use") == 0.9


def test_emit_never_raises_on_a_dead_stdout(monkeypatch, tmp_path):
    class Dead:
        def write(self, s):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    diverted = []
    monkeypatch.setattr(progress, "_stdout_dead", False)
    monkeypatch.setattr(progress, "_divert_stdout", lambda: (diverted.append(1),
                                                            setattr(progress, "_stdout_dead", True)))
    monkeypatch.setattr(progress, "LOG_PATH", tmp_path / "progress.log")
    monkeypatch.setattr(sys, "stdout", Dead())
    progress.emit("ERROR", reason="first")
    progress.emit("RUN-DONE", status="interrupted")
    assert diverted == [1]                                    # fd 1 diverted once, then left alone
    assert (tmp_path / "progress.log").read_text().count("\n") == 2
    with pytest.raises(ValueError):
        progress.emit("NOT-A-TAG")


def test_a_run_whose_kept_attempts_span_two_builds_gets_its_own_key(tmp_path):
    rd = tmp_path / "results/m/r1"
    (rd / "speed").mkdir(parents=True)
    att = lambda launch: {"attempt": 1, "launch": launch, "valid": True, "verdict": "clean", "kept": True}  # noqa: E731
    (rd / "speed/units.json").write_text(json.dumps({"units": {"m3-16k": {"verdict": "clean", "attempts": [att(1)]},
                                                               "m1": {"verdict": "clean", "attempts": [att(2)]}}}))
    launches = [{"launch": 1, "builds": {"macos": "26A1", "xcode": "27X"}},
                {"launch": 2, "builds": {"macos": "26B2", "xcode": "27X"}}]
    (rd / "manifest.json").write_text(json.dumps({"model": {"key": "m"}, "launches": launches,
                                                  "machine": {"sw_vers": {"BuildVersion": "26A1"}}}))
    (run_,) = report.discover(tmp_path / "results")
    assert run_.builds == ("mixed 26A1/26B2", "mixed 27X")
    launches[1]["builds"]["macos"] = "26A1"
    (rd / "manifest.json").write_text(json.dumps({"model": {"key": "m"}, "launches": launches,
                                                  "machine": {"sw_vers": {"BuildVersion": "26A1"}}}))
    (run_,) = report.discover(tmp_path / "results")
    assert run_.builds[0] == "26A1"


def test_selftest_reports_whether_the_pre_commit_hook_is_installed():
    row = next(r for r in cli.hygiene_checks() if r[1].startswith("pre-commit hook"))
    assert row[0] in ("ok", "WARN")                     # never a FAIL: installing it is the user's step
    assert row[0] == ("ok" if cli.pre_commit_installed() else "WARN")
