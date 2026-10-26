"""The solution watcher (--watch-solution): versions, their checks, and the requests after the first pass."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
from marks import needs_data, needs_smoke_venv

from smoke import checks, data, tasks, watch

pytestmark = [needs_data]

TRANSPOSE = tasks.aider_task("transpose")
WRONG = b"def transpose(text):\n    return text\n"


def test_watcher_records_each_content_and_a_missing_file(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    tasks.build_workspace(TRANSPOSE, ws)
    before = sorted(p.name for p in ws.iterdir())
    solution = ws / "transpose.py"

    with watch.SolutionWatcher(TRANSPOSE, ws, poll_s=0.02) as watcher:
        time.sleep(0.1)
        t_write = time.time()
        solution.write_bytes(WRONG)
        time.sleep(0.1)
        solution.unlink()
        time.sleep(0.1)
        solution.write_bytes(WRONG)  # the same content again, after the gap, is a new version
        time.sleep(0.1)
        solution.touch()  # a new mtime with the same content is not

    versions = watcher.versions
    assert [(v["n"], v["missing"]) for v in versions] == [(0, False), (1, False), (2, True), (3, False)]
    assert versions[0]["sha256"] == data.sha256_file(data.exercise("transpose").directory / "transpose.py")
    assert versions[1]["sha256"] == versions[3]["sha256"] and versions[2]["sha256"] is None
    assert watcher.contents[1] == WRONG and watcher.contents[2] is None
    # t is the write's mtime, not the poll that noticed it
    assert t_write - 0.01 <= versions[1]["t"] <= versions[1]["detected_t"] <= t_write + 0.1
    assert sorted(p.name for p in ws.iterdir()) == before


@needs_smoke_venv
def test_versions_are_checked_once_per_state_and_summarized(monkeypatch: pytest.MonkeyPatch) -> None:
    ex = data.exercise("transpose")
    stub, reference = (ex.directory / "transpose.py").read_bytes(), (ex.directory / ".meta" / "example.py").read_bytes()
    contents = [stub, WRONG, reference, b"def transpose(text):\n    return ''\n", reference]
    versions = [{"n": n, "t": 100.0 + 10 * n, "sha256": hashlib.sha256(c).hexdigest(), "file": "transpose.py",
                 "missing": False} for n, c in enumerate(contents)]  # fmt: skip
    checked = []
    real_check = checks.check
    monkeypatch.setattr(checks, "check", lambda *a: checked.append(a) or real_check(*a))

    results = watch.check_versions(TRANSPOSE, versions, contents)

    assert [r["passed"] for r in results] == [False, False, True, False, True]
    assert len(checked) == 4  # the reference's second appearance reuses its first check
    # Requests by start time; first pass at t=120. Handled locally and non-generation records do not count.
    calls = [
        {"endpoint": "chat_completions", "ts_start": 115.0, "request_tool_names": ["shell"]},
        {"endpoint": "chat_completions", "ts_start": 125.0, "request_tool_names": ["shell"]},
        {"endpoint": "chat_completions", "ts_start": 126.0, "request_tool_names": None},
        {"endpoint": "chat_completions", "ts_start": 127.0, "request_tool_names": ["shell"], "handled_locally": True},
        {"endpoint": "count_tokens", "ts_start": 128.0, "request_tool_names": None},
        {"endpoint": "messages", "ts_start": 145.0, "request_tool_names": ["Bash"]},
    ]
    summary = watch.summarize(versions, results, calls)
    assert summary["first_pass_t"] == 120.0 and summary["first_pass_version"] == 2
    assert (summary["planning_after_first_pass"], summary["aux_after_first_pass"]) == (2, 1)
    assert summary["versions"] == 5 and summary["final_version_passes"] is True

    never = watch.summarize(versions[:2], results[:2], calls)
    assert never["first_pass_t"] is None and never["planning_after_first_pass"] is None
    assert never["aux_after_first_pass"] is None and never["final_version_passes"] is False
