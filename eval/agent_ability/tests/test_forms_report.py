"""The report of a forms study (forms/report.py), on synthetic records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from forms import report


def record(task: str, repeat: int, arm: str, *, index: int = 1, passed: bool = True, **fields: Any) -> dict[str, Any]:
    return {
        "index": index,
        "run_id": f"{index:03d}__{task}__r{repeat}__{arm}",
        "state": "finished",
        "task": task,
        "repeat": repeat,
        "arm": arm,
        "passed": passed,
        "verdict": "pass" if passed else "fail",
        "end_reason": "completed",
        "n_calls": 10,
        "wall_s": 100.0,
        "model_share": 0.8,
        "reasoning_tokens": 500,
        "reasoning_chars": 2000,
        "melo_thinking_chars": 2000,
        "thinking_off_calls": 0,
        "refused_calls": 0,
        "headless": True,
        "arm_applied": True,
        "check": {"reason": None if passed else "nothing was submitted"},
        **fields,
    }


def study(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    runs = {r["run_id"]: r for r in records}
    config = {
        "arms": {"T-as-is": {}, "T-off": {}},
        "model": "M",
        "localmelo_sha": "f234b37b0a",
        "wall_cap_s": 900.0,
        "max_calls": 50,
    }
    (tmp_path / "forms.json").write_text(json.dumps({"config": config, "runs": runs, "complete": True}))
    (tmp_path / "environment.json").write_text(
        json.dumps({"start": {"sw_vers": "macOS 27", "xcodebuild_version": "Xcode 27", "chrome_version": "156"}})
    )
    return tmp_path


def two_tasks() -> list[dict[str, Any]]:
    rows = []
    for index, (task, repeat) in enumerate((("a", 1), ("a", 2), ("b", 1), ("b", 2))):
        rows.append(record(task, repeat, "T-as-is", index=2 * index + 1, wall_s=100.0, reasoning_tokens=1000))
        off = {"wall_s": 40.0, "reasoning_tokens": 0, "reasoning_chars": 0, "thinking_off_calls": 10}
        rows.append(record(task, repeat, "T-off", index=2 * index + 2, passed=task == "a", **off))
    return rows


def test_the_unit_is_the_best_every_run_has() -> None:
    assert report.thinking_unit([record("a", 1, "x")]) == "reasoning_tokens"
    assert report.thinking_unit([record("a", 1, "x"), record("a", 2, "x", reasoning_tokens=None)]) == "reasoning_chars"
    inline = record("a", 1, "x", reasoning_tokens=None, reasoning_chars=0)
    assert report.thinking_unit([inline]) == "melo_thinking_chars"
    assert report.thinking_unit([{**inline, "melo_thinking_chars": 0}]) == "reasoning_chars"


def test_thinking_per_step_divides_by_model_calls() -> None:
    assert report.thinking_per_step(record("a", 1, "x"), "reasoning_tokens") == 50
    assert report.thinking_per_step(record("a", 1, "x", n_calls=0), "reasoning_tokens") is None


def test_the_arm_summary() -> None:
    rows = [
        record("a", 1, "x", wall_s=10.0, end_reason="wall_cap", longest_call_s=66.0),
        record("a", 2, "x", passed=False),
    ]
    summary = report.arm_summary(rows, "reasoning_tokens")
    assert summary["passed"] == 1 and summary["pass_rate"] == 0.5 and summary["wall_s_mean"] == 55.0
    assert summary["end_reasons"] == {"wall_cap": 1, "completed": 1} and summary["longest_call_s"] == 66.0


def test_differences_pair_by_task_and_repeat_and_skip_a_missing_arm() -> None:
    rows = [
        record("a", 1, "T-as-is", wall_s=100.0),
        record("a", 1, "T-off", wall_s=60.0),
        record("a", 2, "T-as-is", wall_s=90.0),  # its pair was never run
        record("b", 1, "T-as-is", wall_s=100.0),
        record("b", 1, "T-off", wall_s=None),  # no value: no pair
    ]
    assert report.differences(rows, "T-as-is", "T-off", lambda r: r.get("wall_s")) == {"a": [-40.0]}


def test_the_interval_is_a_point_when_every_difference_agrees() -> None:
    assert report.bootstrap_interval({"a": [-5.0, -5.0], "b": [-5.0]}) == (-5.0, -5.0)
    assert report.bootstrap_interval({}) is None


def test_the_interval_covers_the_mean_and_repeats_with_the_seed() -> None:
    groups = {"a": [1.0, 2.0, 3.0], "b": [10.0, 12.0, 14.0], "c": [-1.0, 0.0, 1.0]}
    low, high = report.bootstrap_interval(groups, samples=2000)
    assert low < (2.0 + 12.0 + 0.0) / 3 < high and low > -1 and high < 14
    assert report.bootstrap_interval(groups, samples=2000) == (low, high)


def test_a_task_counts_once_however_many_pairs_it_has(tmp_path: Path) -> None:
    rows = [
        record("a", 1, "T-as-is", index=1, wall_s=10.0), record("a", 1, "T-off", index=2, wall_s=0.0),
        record("a", 2, "T-as-is", index=3, wall_s=10.0), record("a", 2, "T-off", index=4, wall_s=0.0),
        record("b", 1, "T-as-is", index=5, wall_s=10.0), record("b", 1, "T-off", index=6, wall_s=20.0),
    ]  # fmt: skip
    built = report.build_report(study(tmp_path, rows))
    assert built["paired"]["T-off"]["wall_s"]["mean_difference"] == 0.0


def test_the_report_of_a_study(tmp_path: Path) -> None:
    built = report.write(study(tmp_path, two_tasks()))
    paired = built["paired"]["T-off"]
    assert built["thinking_unit"] == "reasoning_tokens" and built["n_runs"] == 8
    assert paired["wall_s"]["mean_difference"] == -60.0 and paired["wall_s"]["interval"] == (-60.0, -60.0)
    assert paired["pass"]["by_task"] == {"a": 0.0, "b": -1.0} and paired["pass"]["pairs"] == 4
    assert paired["thinking_per_step"]["mean_difference"] == -100.0
    assert built["arms"]["T-as-is"]["pass_rate"] == 1.0 and built["arms"]["T-off"]["pass_rate"] == 0.5
    assert built["tasks"]["b"]["T-off"]["passed"] == 0
    assert [f["run_id"] for f in built["failures"]] == ["006__b__r1__T-off", "008__b__r2__T-off"]
    assert built["flags"] == []
    text = (tmp_path / "report.md").read_text()
    for part in ("## Per arm", "## How the runs ended", "## Per task", "## T-off minus T-as-is", "reasoning tokens"):
        assert part in text
    assert json.loads((tmp_path / "report.json").read_text())["n_runs"] == 8
    assert "-60.0 to -60.0" in text and "macOS 27" in text and "Xcode 27" in text


def test_a_partial_study_reports_what_it_has(tmp_path: Path) -> None:
    rows = [record("a", 1, "T-as-is"), {"run_id": "x", "state": "started", "arm": "T-off", "index": 2}]
    path = study(tmp_path, rows)
    state = json.loads((path / "forms.json").read_text())
    state["complete"] = False
    (path / "forms.json").write_text(json.dumps(state))
    built = report.write(path)
    assert built["n_runs"] == 1 and built["paired"] == {} and "not complete" in (path / "report.md").read_text()


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"headless": None}, "no proof of a headless Chrome"),
        ({"arm_applied": False}, "without their arm's chat_template_kwargs"),
        ({"melo_chrome_profile": True}, "Chrome of its own"),
        ({"foreign_tools": ["web_fetch"]}, "other than the tap's"),
        ({"reasoning_tokens": 0}, "shows no thinking"),
        ({"thinking_off_calls": 10}, "asked for no thinking and still shows some"),
    ],
)
def test_the_flags(changes: dict[str, Any], words: str) -> None:
    found = report.flags([record("a", 1, "x", **changes)], ["x"], "reasoning_tokens")
    assert len(found) == 1 and words in found[0]
