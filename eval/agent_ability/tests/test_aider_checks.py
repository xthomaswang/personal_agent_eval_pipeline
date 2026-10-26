"""The Aider checker: every stub fails, every .meta reference solution passes, tampering is caught."""

from __future__ import annotations

from pathlib import Path

import pytest
from marks import needs_data, needs_smoke_venv

from smoke import checks, data, tasks

pytestmark = [needs_data, needs_smoke_venv]

ALL_EXERCISES = data.exercise_names() if tasks.paths.POLYGLOT_PYTHON.is_dir() else []
SUITE = set(tasks.AIDER_CORE + tasks.AIDER_EXTENDED)


def _use_reference(ex: data.Exercise, ws: Path) -> None:
    (example,) = ex.example
    (solution,) = ex.solution
    data.copy_regular_file(ex.directory / example, ws / solution)


@pytest.mark.parametrize("name", ALL_EXERCISES)
def test_stub_fails_and_reference_passes(name: str, tmp_path: Path) -> None:
    task = tasks.aider_task(name)
    ex = data.exercise(name)
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)

    # The agent sees the stub, its tests and editor helpers only: no .meta, .docs or .approaches
    assert sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*")) == sorted(ex.workspace_files)

    stub = checks.check(task, ws, tmp_path / "stub")
    reference_ws = tmp_path / "ref-ws"
    tasks.build_workspace(task, reference_ws)
    _use_reference(ex, reference_ws)
    reference = checks.check(task, reference_ws, tmp_path / "ref")

    if name in tasks.AIDER_EXCLUDED:
        pytest.xfail(tasks.AIDER_EXCLUDED[name])
    assert not stub["passed"], f"{name}: the untouched stub passes ({stub['detail']})"
    assert reference["passed"], f"{name}: the reference solution fails ({reference['detail']})"
    assert reference["tests_modified"] is False and reference["extra_files"] == []


def test_replaced_test_file_is_flagged_and_restored(tmp_path: Path) -> None:
    task = tasks.aider_task("transpose")
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)
    (ws / "transpose_test.py").write_text("def test_nothing():\n    assert True\n")

    result = checks.check(task, ws, tmp_path / "run")

    assert result["tests_modified"] is True
    assert result["modified_test_files"] == ["transpose_test.py"]
    assert result["passed"] is False  # the pristine tests ran against the untouched stub


def test_planted_conftest_cannot_reach_the_check(tmp_path: Path) -> None:
    task = tasks.aider_task("transpose")
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)
    # A conftest that would turn every failure into a pass if pytest loaded it
    (ws / "conftest.py").write_text(
        "import pytest\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n    report = outcome.get_result()\n    report.outcome = 'passed'\n"
    )

    result = checks.check(task, ws, tmp_path / "run")

    assert result["extra_files"] == ["conftest.py"]
    assert result["passed"] is False


def test_missing_or_symlinked_solution_fails_without_running_pytest(tmp_path: Path) -> None:
    task = tasks.aider_task("transpose")
    ex = data.exercise("transpose")
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)
    (ws / "transpose.py").unlink()
    (ws / "transpose.py").symlink_to(ex.directory / ".meta" / "example.py")

    result = checks.check(task, ws, tmp_path / "run")

    assert result["passed"] is False
    assert result["missing_solution_files"] == ["transpose.py"]


def test_prompt_follows_the_aider_recipe(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    text = tasks.prompt(tasks.aider_task("transpose"), ws)

    assert text.startswith(data.exercise("transpose").instructions())
    assert "\n####\n\nUse the above instructions to modify the supplied files: transpose.py\n" in text
    assert "Unit tests are in transpose_test.py; the tests are correct, don't change them.\n" in text
    assert text.endswith(f"\nWork in {ws}. You can run shell commands and read and write files there.\n")


def test_suite_exercises_exist() -> None:
    assert SUITE <= set(ALL_EXERCISES)
