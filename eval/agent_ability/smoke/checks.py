"""Deciding pass/fail from what the harness left in its workspace.

- Aider: the agent's solution file(s) are copied into a fresh check directory next to the pristine
  test files and an empty pytest.ini, then pytest runs there. This is stricter than restoring the
  test file in place (aider's own harness copies the tests back before each run): a planted
  conftest.py or an edited test cannot reach the check. `tests_modified` records whether the agent
  changed a test file anyway.
- MCPMark: the task's own verify.py runs with FILESYSTEM_TEST_DIR set to the workspace, from a
  directory outside it.
- Warm-up: warmup.txt contains "ready". Unscored.

Downloaded code (the unit tests and verify.py) runs only here, with `python -I -B` from the smoke
environment, in a new session that is killed as a group on timeout.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from smoke import data, paths
from smoke.tasks import Task

AIDER_TIMEOUT_S = 180  # aider's run_unit_tests uses 3 minutes
VERIFY_TIMEOUT_S = 120
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A small, predictable environment for running downloaded code."""
    env = {
        "PATH": f"{paths.SMOKE_VENV_BIN}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "en_US.UTF-8",
    }
    return {**env, **(extra or {})}


def run_bounded(cmd: list[str], cwd: Path, env: dict[str, str], timeout_s: float, log: Path) -> dict[str, Any]:
    """Run cmd with output to log; kill its whole process group on timeout."""
    t0 = time.monotonic()
    with log.open("w") as out:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = proc.wait()
            timed_out = True
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - t0, 3),
        "tail": lines[-1][:300] if lines else "",
    }


def _relative_files(root: Path) -> list[str]:
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if IGNORED_PARTS.intersection(rel.parts) or path.suffix == ".pyc":
            continue
        if path.is_file() or path.is_symlink():
            files.append(rel.as_posix())
    return sorted(files)


def _same_file(a: Path, b: Path) -> bool:
    if a.is_symlink() or not a.is_file():
        return False
    return data.sha256_file(a) == data.sha256_file(b)


def check_aider(task: Task, ws: Path, run_dir: Path, timeout_s: float = AIDER_TIMEOUT_S) -> dict[str, Any]:
    ex = data.exercise(task.name)
    tests_modified = [rel for rel in ex.test if not _same_file(ws / rel, ex.directory / rel)]
    expected = set(ex.workspace_files)
    extra_files = [rel for rel in _relative_files(ws) if rel not in expected]

    check_dir = run_dir / "check"
    check_dir.mkdir(parents=True)
    missing = []
    for rel in ex.solution:
        src = ws / rel
        if src.is_symlink() or not src.is_file():
            missing.append(rel)
            continue
        data.copy_regular_file(src, check_dir / rel)
    for rel in ex.test + ex.editor:
        data.copy_regular_file(ex.directory / rel, check_dir / rel)
    # An ini file in the check directory makes it pytest's rootdir: no upward config discovery
    (check_dir / "pytest.ini").write_text("[pytest]\n")

    result: dict[str, Any] = {
        "kind": "aider",
        "tests_modified": bool(tests_modified),
        "modified_test_files": tests_modified,
        "extra_files": extra_files,
        "missing_solution_files": missing,
    }
    if missing:
        return {**result, "passed": False, "returncode": None, "timed_out": False, "detail": "solution file missing"}
    run = run_bounded(
        [str(paths.SMOKE_PYTHON), "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=check_dir,
        env=_env(),
        timeout_s=timeout_s,
        log=run_dir / "check.log",
    )
    passed = run["returncode"] == 0 and not run["timed_out"]
    return {**result, **run, "passed": passed, "detail": run["tail"]}


def check_mcpmark(task: Task, ws: Path, run_dir: Path, timeout_s: float = VERIFY_TIMEOUT_S) -> dict[str, Any]:
    verify = paths.MCPMARK_EASY / task.name / "verify.py"
    run_dir.mkdir(parents=True, exist_ok=True)
    run = run_bounded(
        [str(paths.SMOKE_PYTHON), "-I", "-B", str(verify)],
        cwd=run_dir,
        env=_env({"FILESYSTEM_TEST_DIR": str(ws)}),
        timeout_s=timeout_s,
        log=run_dir / "check.log",
    )
    passed = run["returncode"] == 0 and not run["timed_out"]
    return {"kind": "mcpmark", **run, "passed": passed, "detail": run["tail"]}


def check_warmup(task: Task, ws: Path, run_dir: Path) -> dict[str, Any]:
    target = ws / "warmup.txt"
    ok = target.is_file() and not target.is_symlink() and target.read_text(errors="replace").strip() == "ready"
    return {"kind": "warmup", "passed": ok, "detail": "warmup.txt says ready" if ok else "warmup.txt missing or wrong"}


def check(task: Task, ws: Path, run_dir: Path) -> dict[str, Any]:
    if task.kind == "aider":
        return check_aider(task, ws, run_dir)
    if task.kind == "mcpmark":
        return check_mcpmark(task, ws, run_dir)
    if task.kind == "warmup":
        return check_warmup(task, ws, run_dir)
    raise ValueError(f"unknown task kind {task.kind!r}")
