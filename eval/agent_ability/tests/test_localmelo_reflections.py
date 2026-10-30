"""Tests for localmelo_reflections.py: the reflection counts per run and in total, the guard, session stores
that cannot be read, rows still in the WAL, the run names, the store location and the command line."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "localmelo_reflections.py"
_spec = importlib.util.spec_from_file_location("localmelo_reflections", _SCRIPT)
reflections = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reflections)

STORE = Path("localmelo-home") / ".cache" / "localmelo" / "cli" / "sessions.db"
OPTIONS = {"baseline_rate": 0.011, "margin": 0.02, "min_reflections": 30}


def row(kind: str, data: dict) -> tuple:
    return ("session-1", 1, kind, json.dumps({"data": data}))


def reflection_row(
    *, readable: int = 1, reason: str | None = None, repaired: str | None = None, requests: int = 1
) -> tuple:
    data = {"readable": readable, "unreadable_reason": reason, "repaired": repaired, "requests": requests}
    return row("reflection", data)


def write_db(path: Path, rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE events (session_id TEXT, turn INTEGER, type TEXT, event TEXT)")
        conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return path


def make_run(root: Path, name: str, rows: list[tuple]) -> Path:
    return write_db(root / name / STORE, rows)


def build(root: Path) -> dict:
    return reflections.build_report(root, **OPTIONS)


def test_counts_for_one_run(tmp_path: Path) -> None:
    make_run(
        tmp_path,
        "run-a",
        [
            reflection_row(),
            reflection_row(repaired="unmatched_closer"),
            reflection_row(reason="no_json", requests=2),  # the first reply was unreadable, the retry was read
            reflection_row(readable=0, reason="tool_call", repaired="unmatched_closer", requests=2),  # lost
            reflection_row(readable=0, reason="invalid", requests=3),  # lost
            # Not a reflection, so none of it is counted
            row("tool_call", {"readable": 0, "unreadable_reason": "no_json", "repaired": "x", "requests": 100}),
        ],
    )
    (run,) = build(tmp_path)["runs"]
    assert run == {
        "run": "run-a",
        "db": "run-a/localmelo-home/.cache/localmelo/cli/sessions.db",
        "reflections": 5,
        "repaired": 2,
        "unreadable_after_repair": 3,
        "lost": 2,
        "by_reason": {"invalid": 1, "no_json": 1, "tool_call": 1},
        "requests": 9,
    }


def test_totals_and_rate_add_up_the_runs(tmp_path: Path) -> None:
    make_run(
        tmp_path,
        "run-a",
        [reflection_row(reason="no_json")] * 2 + [reflection_row(reason="invalid")] + [reflection_row()] * 2,
    )
    make_run(
        tmp_path,
        "run-b",
        [reflection_row(reason="tool_call"), reflection_row(repaired="unmatched_closer")] + [reflection_row()] * 23,
    )
    totals = build(tmp_path)["totals"]
    assert totals["reflections"] == 30
    assert totals["repaired"] == 1
    assert totals["unreadable_after_repair"] == 4
    assert totals["lost"] == 0
    assert totals["requests"] == 30
    assert totals["by_reason"] == {"invalid": 1, "no_json": 2, "tool_call": 1}
    assert totals["unreadable_rate"] == pytest.approx(4 / 30)


def test_guard_passes_within_the_limit(tmp_path: Path) -> None:
    make_run(tmp_path, "run-a", [reflection_row(reason="no_json")] + [reflection_row()] * 39)  # 1 of 40
    guard = build(tmp_path)["guard"]
    assert guard["passes"] is True
    assert guard["reason"] is None
    assert guard["unreadable_rate"] == pytest.approx(0.025)
    assert guard["limit"] == pytest.approx(0.031)


def test_guard_fails_above_the_limit(tmp_path: Path) -> None:
    make_run(tmp_path, "run-a", [reflection_row(reason="no_json")] * 2 + [reflection_row()] * 38)  # 2 of 40
    guard = build(tmp_path)["guard"]
    assert guard["passes"] is False
    assert guard["reason"] is None


def test_guard_is_not_checked_below_the_minimum(tmp_path: Path) -> None:
    make_run(tmp_path, "run-a", [reflection_row()] * 29)
    guard = build(tmp_path)["guard"]
    assert guard["passes"] is None
    assert guard["reason"] == "fewer than 30 reflections"


@pytest.mark.parametrize(
    ("reflections_count", "rate", "passes"),
    [
        (30, 1 / 30, False),  # 0.0333 is above the limit 0.031
        (40, 0.025, True),
        (40, 0.05, False),
        (30, 0.0, True),  # no unreadable reply at all
    ],
)
def test_check_guard_compares_with_baseline_plus_margin(reflections_count: int, rate: float, passes: bool) -> None:
    guard = reflections.check_guard(reflections_count, rate, baseline_rate=0.011, margin=0.02, min_reflections=30)
    assert guard["passes"] is passes


def test_a_rate_equal_to_the_limit_passes_despite_float_error() -> None:
    # 0.7 + 0.1 is 0.7999999999999999 in floats, while 8 of 10 is 0.8
    guard = reflections.check_guard(10, 0.8, baseline_rate=0.7, margin=0.1, min_reflections=10)
    assert guard["passes"] is True


def test_a_run_without_reflections_is_in_the_json_but_not_in_the_table(tmp_path: Path) -> None:
    make_run(tmp_path, "quiet", [row("tool_call", {"requests": 4})])
    make_run(tmp_path, "busy", [reflection_row(reason="invalid")])
    report = build(tmp_path)
    quiet = next(run for run in report["runs"] if run["run"] == "quiet")
    assert (quiet["reflections"], quiet["requests"], quiet["by_reason"]) == (0, 0, {})
    assert report["totals"]["reflections"] == 1
    text = reflections.format_markdown(report)
    assert "### busy" in text
    assert "### quiet" not in text


def test_without_any_reflection_the_rate_and_the_guard_are_empty(tmp_path: Path) -> None:
    make_run(tmp_path, "quiet", [row("tool_call", {})])
    report = build(tmp_path)
    assert report["totals"]["unreadable_rate"] is None
    assert report["guard"]["passes"] is None
    text = reflections.format_markdown(report)
    assert "No run has reflections." in text
    assert "(rate n/a)" in text


def test_a_store_that_cannot_be_read_is_listed_and_skipped(tmp_path: Path) -> None:
    make_run(tmp_path, "good", [reflection_row(reason="no_json")])
    garbage = tmp_path / "garbage" / STORE
    garbage.parent.mkdir(parents=True)
    garbage.write_bytes(b"this is not a SQLite database\n" * 100)
    no_events = tmp_path / "no-events" / STORE
    no_events.parent.mkdir(parents=True)
    sqlite3.connect(no_events).close()  # a valid database with no events table
    report = build(tmp_path)
    assert [run["run"] for run in report["runs"]] == ["good"]
    assert report["totals"]["reflections"] == 1
    assert [entry["db"] for entry in report["unreadable_dbs"]] == [
        "garbage/localmelo-home/.cache/localmelo/cli/sessions.db",
        "no-events/localmelo-home/.cache/localmelo/cli/sessions.db",
    ]
    assert all(entry["error"] for entry in report["unreadable_dbs"])
    assert "Unreadable databases" in reflections.format_markdown(report)


def test_only_the_session_store_location_is_picked(tmp_path: Path) -> None:
    kept = make_run(tmp_path, "run-a", [])
    other = make_run(tmp_path, "run-b", [])
    skipped = [
        tmp_path / "run-c" / "localmelo-home" / ".cache" / "localmelo" / "sessions.db",  # one level above cli
        tmp_path / "run-c" / "other-home" / ".cache" / "localmelo" / "cli" / "sessions.db",  # another HOME
        tmp_path / "run-c" / "localmelo-home-copy" / ".cache" / "localmelo" / "cli" / "sessions.db",
        tmp_path / "run-c" / "localmelo-home" / ".cache" / "localmelo" / "cli-old" / "sessions.db",
        tmp_path / "run-c" / "localmelo-home" / ".cache" / "localmelo" / "cli" / "backup.db",  # other name
    ]
    for path in skipped:
        write_db(path, [])
    assert reflections.find_session_dbs(tmp_path) == [kept, other]


def test_a_run_is_named_by_its_path_from_root(tmp_path: Path) -> None:
    make_run(tmp_path, "lab/day-1/run-x", [reflection_row()])
    make_run(tmp_path, ".", [reflection_row()])  # root itself is the run directory
    assert [run["run"] for run in build(tmp_path)["runs"]] == ["lab/day-1/run-x", "."]


def test_cli_prints_the_report_and_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(tmp_path, "run-a", [reflection_row(reason="no_json")] + [reflection_row()] * 39)
    out_json = tmp_path / "report.json"
    assert reflections.main([str(tmp_path), "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "### run-a" in out
    assert "| 40 | 0 | 1 | 0 | 40 | no_json: 1 |" in out
    assert "Guard: PASS" in out
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["guard"]["passes"] is True
    assert [run["run"] for run in report["runs"]] == ["run-a"]


def test_cli_rejects_a_missing_root_and_a_zero_minimum(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing:
        reflections.main([str(tmp_path / "missing")])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as zero:
        reflections.main([str(tmp_path), "--min-reflections", "0"])
    assert zero.value.code == 2


@pytest.mark.parametrize("checkpointed", [0, 3])
def test_rows_still_in_the_wal_are_counted_and_the_store_is_left_alone(tmp_path: Path, checkpointed: int) -> None:
    """A writer in WAL mode leaves its newest reflections in the -wal file and stays open while they are counted.

    checkpointed is how many of the five were first moved into the main file. The counter must see all five,
    and the store's files must keep their size and modification time.
    """
    store = tmp_path / "run-a" / STORE
    store.parent.mkdir(parents=True)
    wal = store.with_name(store.name + "-wal")
    shm = store.with_name(store.name + "-shm")
    unreadable_row = reflection_row(reason="no_json")
    writer = sqlite3.connect(store)
    try:
        writer.execute("PRAGMA journal_mode=WAL").fetchall()
        writer.execute("CREATE TABLE events (session_id TEXT, turn INTEGER, type TEXT, event TEXT)")
        writer.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", [unreadable_row] * checkpointed)
        writer.commit()
        if checkpointed:
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        writer.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", [unreadable_row] * (5 - checkpointed))
        writer.commit()
        assert wal.stat().st_size > 0 and shm.exists()  # the newest rows are only in the -wal file
        before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in (store, wal, shm)}

        (run,) = build(tmp_path)["runs"]

        assert (run["reflections"], run["unreadable_after_repair"]) == (5, 5)
        assert {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in (store, wal, shm)} == before
    finally:
        writer.close()
