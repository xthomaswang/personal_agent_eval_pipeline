"""The E3 driver: schedule, checkpoint and resume, and the refusals. No model, no oMLX, no GPU lock."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from e3 import run as e3_run

PMSET_BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t{pct}%; discharging; 4:00 remaining"
PMSET_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t{pct}%; charging; 1:00 remaining"


def fake_harness() -> SimpleNamespace:
    return SimpleNamespace(manifest=SimpleNamespace(LOCALMELO_REPO=None), BUILDERS={})


@pytest.fixture
def study(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """run_e3 with every outside dependency faked; the list collects the run ids execute() was called for."""
    executed: list[str] = []

    def execute(item: dict[str, Any], out: Path, harness: Any, env_file: Path) -> dict[str, Any]:
        executed.append(item["run_id"])
        record = {**item, "state": "finished", "verdict": "pass", "passed": True, "wall_s": 1.0, "n_llm": 3}
        (out / "runs" / item["run_id"]).mkdir(parents=True)
        (out / "runs" / item["run_id"] / "record.json").write_text(json.dumps(record))
        return record

    monkeypatch.setattr(e3_run, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(e3_run, "system_snapshot", lambda localmelo: {"sw_vers": "fake"})
    monkeypatch.setattr(e3_run, "localmelo_sha", lambda localmelo: "ce4a3a8" + "0" * 33)
    monkeypatch.setattr(e3_run, "stop_reason", lambda out: "STOP file found" if (out / "STOP").exists() else None)
    monkeypatch.setattr(e3_run, "execute", execute)
    return executed


def go(out: Path, **kwargs: Any) -> int:
    kwargs.setdefault("exercises", ("transpose", "wordy"))
    kwargs.setdefault("repeats", 1)
    kwargs.setdefault("arms", ("S0", "C1"))
    return e3_run.run_e3(out, localmelo=out.parent, harness=fake_harness(), **kwargs)


def state(out: Path) -> dict[str, Any]:
    return json.loads((out / "e3.json").read_text())


# --- the schedule ---


def test_arms_are_the_agreed_flag_sets() -> None:
    assert e3_run.ARMS == {
        "S0": (),
        "C1": ("--window-eviction", "chunk"),
        "B1": ("--window-eviction", "boundary"),
        "B2": ("--window-eviction", "boundary", "--fixed-prefix", "on"),
        "S2": ("--fixed-prefix", "on"),
    }
    assert e3_run.arm_args("B2")[:2] == ["--output-format", "json"]
    assert e3_run.BUDGET == {
        "max_steps": 30, "max_turns": 40, "timeout_s": 3600.0, "thinking": "off", "log_bodies": True,
    }  # fmt: skip


def test_schedule_is_task_major_with_all_arms_back_to_back() -> None:
    planned = e3_run.schedule()
    assert len(planned) == 120
    assert [p["index"] for p in planned] == list(range(1, 121))
    assert len({p["run_id"] for p in planned}) == 120
    groups = [planned[i : i + 5] for i in range(0, 120, 5)]
    expected = [(e, r) for e in e3_run.EXERCISES for r in (1, 2)]
    assert [(g[0]["exercise"], g[0]["repeat"]) for g in groups] == expected
    for group in groups:
        assert {(p["exercise"], p["repeat"]) for p in group} == {(group[0]["exercise"], group[0]["repeat"])}
        assert sorted(p["arm"] for p in group) == sorted(e3_run.ARMS)


def test_schedule_rotates_the_arm_order_by_task_index_plus_repeat() -> None:
    arms = list(e3_run.ARMS)
    planned = e3_run.schedule()
    firsts = []
    for gi in range(24):
        group = planned[gi * 5 : gi * 5 + 5]
        task_index, repeat = divmod(gi, 2)
        repeat += 1
        offset = (task_index + repeat) % 5
        assert [p["arm"] for p in group] == arms[offset:] + arms[:offset]
        assert [p["position"] for p in group] == [1, 2, 3, 4, 5]
        firsts.append(group[0]["arm"])
    counts = Counter(firsts)
    assert set(counts) == set(arms)  # no arm is always first
    assert sorted(counts.values()) == [4, 4, 5, 5, 6]  # 24 groups over 5 arms
    assert planned[0]["run_id"] == "001__sgf-parsing__r1__C1"


def test_schedule_validates() -> None:
    with pytest.raises(ValueError):
        e3_run.schedule(repeats=0)
    with pytest.raises(ValueError):
        e3_run.schedule(arms=("S0", "nope"))
    with pytest.raises(ValueError):
        e3_run.schedule(exercises=())


# --- checkpoint and resume ---


def test_a_study_runs_everything_once_and_a_resume_replays_nothing(study: list[str], tmp_path: Path) -> None:
    out = tmp_path / "e3"
    assert go(out) == 0
    planned = e3_run.schedule(("transpose", "wordy"), 1, ("S0", "C1"))
    assert study == [p["run_id"] for p in planned]
    saved = state(out)
    assert saved["complete"] is True
    assert (out / "schedule.json").exists()
    assert set(saved["runs"]) == {p["run_id"] for p in planned}
    env = json.loads((out / "environment.json").read_text())
    assert env["start"] == {"sw_vers": "fake"} and env["end"] == {"sw_vers": "fake"}

    study.clear()
    assert go(out) == 0
    assert study == []
    assert not (out / "ws" / ".lock").exists()


def test_an_interrupted_run_is_recorded_not_replayed(study: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "e3"
    real = e3_run.execute

    def dies_on_third(item: dict[str, Any], out_: Path, harness: Any, env_file: Path) -> dict[str, Any]:
        if item["index"] == 3:
            raise RuntimeError("killed")
        return real(item, out_, harness, env_file)

    monkeypatch.setattr(e3_run, "execute", dies_on_third)
    with pytest.raises(RuntimeError):
        go(out)
    third = e3_run.schedule(("transpose", "wordy"), 1, ("S0", "C1"))[2]["run_id"]
    assert state(out)["runs"][third]["state"] == "started"
    assert not (out / "ws" / ".lock").exists()

    monkeypatch.setattr(e3_run, "execute", real)
    study.clear()
    assert go(out) == 0
    assert study == [e3_run.schedule(("transpose", "wordy"), 1, ("S0", "C1"))[3]["run_id"]]
    runs = state(out)["runs"]
    assert runs[third]["state"] == "interrupted" and runs[third]["verdict"] == "interrupted"
    assert runs[third]["passed"] is False
    assert state(out)["complete"] is True


def test_a_started_run_with_a_record_is_adopted(study: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "e3"
    real = e3_run.execute

    def writes_then_dies(item: dict[str, Any], out_: Path, harness: Any, env_file: Path) -> dict[str, Any]:
        record = real(item, out_, harness, env_file)
        raise RuntimeError("died after record.json")

    monkeypatch.setattr(e3_run, "execute", writes_then_dies)
    with pytest.raises(RuntimeError):
        go(out)
    first = e3_run.schedule(("transpose", "wordy"), 1, ("S0", "C1"))[0]["run_id"]
    monkeypatch.setattr(e3_run, "execute", real)
    study.clear()
    go(out)
    assert first not in study
    assert state(out)["runs"][first]["state"] == "finished"


def test_a_changed_config_needs_a_fresh_directory(study: list[str], tmp_path: Path) -> None:
    out = tmp_path / "e3"
    go(out)
    with pytest.raises(ValueError, match="changed"):
        go(out, repeats=2)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "file").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        go(stray)


# --- guards ---


def test_stop_file_stops_before_the_next_run(study: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "e3"
    real = e3_run.execute

    def stops_after_first(item: dict[str, Any], out_: Path, harness: Any, env_file: Path) -> dict[str, Any]:
        record = real(item, out_, harness, env_file)
        (out_ / "STOP").write_text("")
        return record

    monkeypatch.setattr(e3_run, "execute", stops_after_first)
    assert go(out) == 3
    assert len(study) == 1
    assert state(out)["complete"] is False
    assert "STOP file found" in (out / "e3.log").read_text()
    (out / "STOP").unlink()
    monkeypatch.setattr(e3_run, "execute", real)
    assert go(out) == 0
    assert len(study) == 4


def test_stop_reason_checks_the_stop_file_the_lock_and_the_battery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / ".gpu-lock"
    monkeypatch.setattr(e3_run, "GPU_LOCK", lock)
    monkeypatch.setattr(e3_run, "battery_problem", lambda: None)
    out = tmp_path / "out"
    out.mkdir()
    lock.write_text("holder: agent-benchmark\nsince: now\n")
    assert e3_run.stop_reason(out) is None
    (out / "STOP").write_text("")
    assert e3_run.stop_reason(out) == "STOP file found"
    (out / "STOP").unlink()
    lock.unlink()
    assert "absent" in e3_run.stop_reason(out)
    lock.write_text("holder: other\n")
    assert "another session" in e3_run.stop_reason(out)
    lock.write_text("holder: agent-benchmark\n")
    monkeypatch.setattr(e3_run, "battery_problem", lambda: "on battery at 12% (below 30%)")
    assert "on battery" in e3_run.stop_reason(out)


def test_lock_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / ".gpu-lock"
    monkeypatch.setattr(e3_run, "GPU_LOCK", lock)
    assert "absent" in e3_run.lock_problem()
    lock.write_text("holder: some-other-session\n")
    assert "held by another session (holder: some-other-session)" in e3_run.lock_problem()
    lock.write_text("pid: 1\nholder: agent-benchmark\n")
    assert e3_run.lock_problem() is None


@pytest.mark.parametrize(
    ("pmset", "problem"),
    [
        (PMSET_BATTERY.format(pct=29), True),
        (PMSET_BATTERY.format(pct=30), False),
        (PMSET_BATTERY.format(pct=80), False),
        (PMSET_AC.format(pct=10), False),
        ("", False),
    ],
)
def test_battery_guard(pmset: str, problem: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e3_run, "_out", lambda cmd, timeout=30.0: pmset)
    assert (e3_run.battery_problem() is not None) is problem


def test_refuses_to_start_without_the_lock_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    lock = tmp_path / ".gpu-lock"
    monkeypatch.setattr(e3_run, "GPU_LOCK", lock)
    monkeypatch.setattr(e3_run, "localmelo_sha", lambda localmelo: e3_run.LOCALMELO_SHA + "0" * 33)
    monkeypatch.setattr(e3_run, "_out", lambda cmd, timeout=30.0: "")
    monkeypatch.setattr(e3_run.data, "exercise_names", lambda: list(e3_run.EXERCISES))
    # runner.preflight's own checks need the data, a venv and a server; only its lock check matters here
    monkeypatch.setattr(e3_run.runner, "preflight", lambda settings, harness: [m] if (m := e3_run.runner.lock_problem()) else [])
    out = tmp_path / "e3"
    harness = fake_harness()
    assert e3_run.run_e3(out, localmelo=tmp_path, harness=harness) == 2
    assert "absent" in capsys.readouterr().err
    assert not out.exists()
    assert harness.manifest.LOCALMELO_REPO == tmp_path.resolve()

    lock.write_text("holder: someone-else\n")
    assert e3_run.run_e3(out, localmelo=tmp_path, harness=harness) == 2
    assert "another session" in capsys.readouterr().err

    lock.write_text("holder: agent-benchmark\n")
    assert e3_run.preflight(e3_run._settings(e3_run.EXERCISES), harness, tmp_path) == []
    assert e3_run.paths.GPU_LOCK != lock  # the runner's own path is restored


def test_preflight_requires_the_pinned_clean_localmelo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e3_run.runner, "preflight", lambda settings, harness: [])
    monkeypatch.setattr(e3_run.data, "exercise_names", lambda: list(e3_run.EXERCISES))
    settings = e3_run._settings(e3_run.EXERCISES)
    monkeypatch.setattr(e3_run, "localmelo_sha", lambda localmelo: "1234567" + "0" * 33)
    assert "not ce4a3a8" in e3_run.preflight(settings, fake_harness(), tmp_path)[0]
    monkeypatch.setattr(e3_run, "localmelo_sha", lambda localmelo: "ce4a3a8" + "0" * 33)
    monkeypatch.setattr(e3_run, "_out", lambda cmd, timeout=30.0: " M melo/x.py")
    assert "uncommitted" in e3_run.preflight(settings, fake_harness(), tmp_path)[0]
    monkeypatch.setattr(e3_run, "_out", lambda cmd, timeout=30.0: "")
    monkeypatch.setattr(e3_run.data, "exercise_names", lambda: ["transpose"])
    assert "sgf-parsing is not in the pinned Aider data" in e3_run.preflight(settings, fake_harness(), tmp_path)[0]


# --- dry run ---


def test_dry_run_prints_the_schedule_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(e3_run, "preflight", lambda *a, **k: ["the GPU lock is absent"])
    monkeypatch.setattr(e3_run, "system_snapshot", lambda localmelo: {"sw_vers": "macOS 27", "localmelo_sha": "ce4a3a8x"})
    out = tmp_path / "e3"
    assert e3_run.main(["--run-dir", str(out), "--dry-run"], harness=fake_harness()) == 2
    text = capsys.readouterr().out.splitlines()
    assert text[0].startswith("E3 plan: 120 runs = 12 tasks x 2 repeats x 5 arms")
    assert sum(1 for line in text if line[:4].strip().rstrip(".").isdigit()) == 120
    assert "Preflight: PROBLEMS" in text and "  - the GPU lock is absent" in text
    assert any(line.startswith("Arm B2: localmelo --output-format json --window-eviction boundary --fixed-prefix on") for line in text)
    assert not out.exists()

    monkeypatch.setattr(e3_run, "preflight", lambda *a, **k: [])
    assert e3_run.main(["--run-dir", str(out), "--dry-run", "--repeats", "1", "--arms", "S0,B1"], harness=fake_harness()) == 0
    assert "E3 plan: 24 runs" in capsys.readouterr().out
