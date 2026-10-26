"""Watching an Aider solution while the harness runs (--watch-solution), for "steps after the
solution already passed".

A thread polls the task's solution files in the workspace every POLL_S seconds: os.lstat first, the
content only when (mtime_ns, size) changes. It only reads the workspace, and keeps every distinct
content in memory, because bench.harness.run creates the run directory itself. After the run each
version goes to <run_dir>/solution_versions/<n:03d>-<basename>, listed in versions.jsonl. The solution
as it stood after each version is checked by the run's own checker, in a fresh workspace. The proxy's
records then give the requests that started after the first passing version.

All times are time.time() seconds. A version's `t` is its file's mtime, clamped between the last poll
that saw the old content and the poll that saw the new one. The detection time alone (`detected_t`)
can trail a write by up to POLL_S, and a harness sends its next request right after the tool call that
wrote, so that request could look older than the write. calls.jsonl's `ts_start` is the proxy's
time.time() when the request arrived. Its `mono_*` fields are time.monotonic() in the proxy's process,
which file times cannot be compared with.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Self

from smoke import checks, data, tasks
from smoke.tasks import Task

POLL_S = 0.5
# The proxy's generation endpoints (bench.proxy.request_info.GENERATION_ENDPOINTS)
GENERATION_ENDPOINTS = ("chat_completions", "messages")


def _signature(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) of a regular file. None when it is missing, a symlink or anything else: the
    checker treats those as a missing solution too."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size) if stat.S_ISREG(st.st_mode) else None


class SolutionWatcher:
    """Around the harness run: the stub on entry, every change while it runs, a last look on exit."""

    def __init__(self, task: Task, ws: Path, poll_s: float | None = None) -> None:
        self.task, self.ws = task, ws
        self.poll_s = POLL_S if poll_s is None else poll_s
        self.files = data.exercise(task.name).solution
        self.versions: list[dict[str, Any]] = []
        self.contents: list[bytes | None] = []
        # Per file: its signature, and the last time a poll saw that signature
        self._seen: dict[str, tuple[tuple[int, int] | None, float]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="solution-watcher", daemon=True)

    def __enter__(self) -> Self:
        self._poll()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._poll()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._poll()

    def _poll(self) -> None:
        now = time.time()
        for rel in self.files:
            path = self.ws / rel
            signature = _signature(path)
            seen = self._seen.get(rel)
            if seen is not None and seen[0] == signature:
                self._seen[rel] = (signature, now)
                continue
            content = None
            if signature is not None:
                try:
                    content = path.read_bytes()
                except OSError:  # gone between the stat and the read
                    continue
                if _signature(path) != signature:  # written while it was read: the next poll takes it
                    continue
            sha = hashlib.sha256(content).hexdigest() if content is not None else None
            last = next((v for v in reversed(self.versions) if v["file"] == rel), None)
            if last is None or last["sha256"] != sha:
                written = signature[0] / 1e9 if signature is not None else now
                self.versions.append({
                    "n": len(self.versions),
                    "t": min(max(written, seen[1] if seen else now), now),
                    "detected_t": now,
                    "sha256": sha,
                    "file": rel,
                    "missing": content is None,
                })  # fmt: skip
                self.contents.append(content)
            self._seen[rel] = (signature, now)

    def finish(self, run_dir: Path) -> dict[str, Any]:
        """Write the versions into run_dir, check each, and add the result to summary.json as "watch"."""
        out = run_dir / "solution_versions"
        out.mkdir(parents=True, exist_ok=True)
        for version, content in zip(self.versions, self.contents):
            if content is not None:
                (out / f"{version['n']:03d}-{Path(version['file']).name}").write_bytes(content)
        (out / "versions.jsonl").write_text("".join(json.dumps(v) + "\n" for v in self.versions))
        calls_path = run_dir / "calls.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()] if calls_path.exists() else []
        watch = summarize(self.versions, check_versions(self.task, self.versions, self.contents), calls)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["watch"] = watch
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return watch


def check_versions(task: Task, versions: list[dict[str, Any]], contents: list[bytes | None]) -> list[dict[str, Any]]:
    """The solution as it stood after each version, checked by checks.check in a fresh workspace.
    A state seen before (A, B, then A again) is checked once."""
    shas: dict[str, str | None] = {}
    state: dict[str, bytes | None] = {}
    done: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    results = []
    with tempfile.TemporaryDirectory(prefix="smoke-watch-") as tmp:
        for version, content in zip(versions, contents):
            shas[version["file"]], state[version["file"]] = version["sha256"], content
            key = tuple(sorted((rel, sha or "") for rel, sha in shas.items()))
            if key not in done:
                ws = Path(tmp) / f"ws-{version['n']}"
                tasks.build_workspace(task, ws)  # a solution file not seen yet stays the stub
                for rel, current in state.items():
                    if current is None:
                        (ws / rel).unlink(missing_ok=True)
                    else:
                        (ws / rel).write_bytes(current)
                result = checks.check(task, ws, Path(tmp) / f"check-{version['n']}")
                done[key] = {"passed": result["passed"], "detail": result.get("detail") or ""}
            results.append({"n": version["n"], **done[key]})
    return results


def summarize(versions: list[dict[str, Any]], results: list[dict[str, Any]], calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The first passing version and the model requests that started after it (None without one)."""
    first = next((r["n"] for r in results if r["passed"]), None)
    first_pass_t = versions[first]["t"] if first is not None else None
    requests = [c for c in calls if c.get("endpoint") in GENERATION_ENDPOINTS and not c.get("handled_locally")]
    after = [c for c in requests if first_pass_t is not None and (c.get("ts_start") or 0) > first_pass_t]
    return {
        "first_pass_t": first_pass_t,
        "first_pass_version": first,
        "planning_after_first_pass": None if first is None else sum(1 for c in after if c.get("request_tool_names")),
        "aux_after_first_pass": None if first is None else sum(1 for c in after if not c.get("request_tool_names")),
        "versions": len(versions),
        "final_version_passes": results[-1]["passed"] if results else None,
        "checks": results,
    }
