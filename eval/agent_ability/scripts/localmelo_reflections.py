"""Count localmelo's reflection outcomes in every benchmark run under ROOT.

Usage: python3 localmelo_reflections.py ROOT [--json OUT] [--baseline-rate 0.011] [--margin 0.02]
                                           [--min-reflections 30]

Each localmelo run keeps its own HOME at <run_dir>/localmelo-home, and localmelo writes its session store
to localmelo-home/.cache/localmelo/cli/sessions.db (SQLite). That file has an events table with the columns
session_id, turn, type and event (JSON text). A reflection is a row with type 'reflection', and its fields
sit under $.data:
- readable: 1 or 0, after the retry;
- unreadable_reason: null, or 'tool_call', 'no_json' or 'invalid' for the first reply after repair;
- repaired: null, or a string such as 'unmatched_closer';
- requests: the number of requests.

Per run the script counts the reflections, the repaired ones (repaired is set), unreadable_after_repair
(unreadable_reason is set), lost (readable is 0), the unreadable reasons and the sum of requests. Totals add
up the runs, and unreadable_rate is unreadable_after_repair / reflections (None without reflections).

The guard passes when unreadable_rate is at most baseline_rate + margin. It is only checked with at least
min_reflections reflections; otherwise passes is None and the reason says so.

Each store is copied, with its -wal and -shm files when present, into a temporary directory, and only the
copy is opened. SQLite applies the write-ahead log to the copy, so reflections not yet checkpointed into the
store are counted too. The originals are only read, to make the copy: no SQLite connection is opened on them,
and nothing is written to them. A store that cannot be read, or whose events cannot be read, is listed under
unreadable_dbs with its error and is not counted. The exit status is 0 after a report, and 2 for bad arguments
or when ROOT is not a directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

SESSION_DB_NAME = "sessions.db"
# The session store of a run's own HOME: .../localmelo-home/.cache/localmelo/cli/sessions.db
SESSION_DB_MARKER = "/localmelo-home/.cache/localmelo/cli/"
DEFAULT_BASELINE_RATE = 0.011
DEFAULT_MARGIN = 0.02
DEFAULT_MIN_REFLECTIONS = 30
# Absorbs float error, so a rate that equals the limit in decimal numbers still passes.
RATE_TOLERANCE = 1e-9
COUNT_KEYS = ("reflections", "repaired", "unreadable_after_repair", "lost", "requests")

SUMMARY_SQL = """
    SELECT COUNT(*),
           COALESCE(SUM(json_extract(event, '$.data.repaired') IS NOT NULL), 0),
           COALESCE(SUM(json_extract(event, '$.data.unreadable_reason') IS NOT NULL), 0),
           COALESCE(SUM(json_extract(event, '$.data.readable') = 0), 0),
           COALESCE(SUM(json_extract(event, '$.data.requests')), 0)
    FROM events
    WHERE type = 'reflection'
"""
REASON_SQL = """
    SELECT json_extract(event, '$.data.unreadable_reason'), COUNT(*)
    FROM events
    WHERE type = 'reflection' AND json_extract(event, '$.data.unreadable_reason') IS NOT NULL
    GROUP BY json_extract(event, '$.data.unreadable_reason')
"""


def find_session_dbs(root: Path) -> list[Path]:
    """Every session store under root, sorted. Only paths with the localmelo-home location count."""
    return sorted(db for db in root.rglob(SESSION_DB_NAME) if SESSION_DB_MARKER in db.as_posix())


def run_name(root: Path, db: Path) -> str:
    """The run directory as a path from root, or '.' when root is the run directory itself."""
    run_dir = Path(db.as_posix().split(SESSION_DB_MARKER, 1)[0])
    try:
        return run_dir.relative_to(root).as_posix()
    except ValueError:  # root lies inside the run's HOME: use the run folder's name instead
        return run_dir.name


def count_run(db: Path) -> dict:
    """Count the reflections in one session store, read from a copy of it.

    The store and its -wal and -shm files are copied into a temporary directory, and only the copy is opened,
    so SQLite applies any WAL frames that were not checkpointed. The originals are only read, to make the copy.
    Raises sqlite3.Error, OSError or ValueError when the store cannot be read.
    """
    with tempfile.TemporaryDirectory() as scratch:
        copy_path = Path(scratch) / db.name
        shutil.copyfile(db, copy_path)
        # A -wal or -shm file may be missing; the copy is then read without it.
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                shutil.copyfile(db.with_name(db.name + suffix), copy_path.with_name(copy_path.name + suffix))
        with contextlib.closing(sqlite3.connect(copy_path)) as conn:
            reflections, repaired, unreadable, lost, requests = conn.execute(SUMMARY_SQL).fetchone()
            reasons = conn.execute(REASON_SQL).fetchall()
    return {
        "reflections": reflections,
        "repaired": repaired,
        "unreadable_after_repair": unreadable,
        "lost": lost,
        "by_reason": dict(sorted(reasons)),
        "requests": int(requests),
    }


def total_counts(runs: list[dict]) -> dict:
    """Sum the counts over all runs, with the unreadable rate (None when there are no reflections)."""
    totals: dict = {key: sum(run[key] for run in runs) for key in COUNT_KEYS}
    reasons: Counter[str] = Counter()
    for run in runs:
        reasons.update(run["by_reason"])
    totals["by_reason"] = dict(sorted(reasons.items()))
    reflections = totals["reflections"]
    totals["unreadable_rate"] = totals["unreadable_after_repair"] / reflections if reflections else None
    return totals


def check_guard(
    reflections: int,
    unreadable_rate: float | None,
    *,
    baseline_rate: float,
    margin: float,
    min_reflections: int,
) -> dict:
    """The unreadable rate must be at most baseline_rate + margin, once there are enough reflections."""
    limit = baseline_rate + margin
    guard: dict = {
        "baseline_rate": baseline_rate,
        "margin": margin,
        "limit": limit,
        "min_reflections": min_reflections,
        "unreadable_rate": unreadable_rate,
        "passes": None,
        "reason": None,
    }
    if unreadable_rate is not None and reflections >= min_reflections:
        guard["passes"] = unreadable_rate <= limit + RATE_TOLERANCE
    else:
        guard["reason"] = f"fewer than {min_reflections} reflections"
    return guard


def build_report(root: Path, *, baseline_rate: float, margin: float, min_reflections: int) -> dict:
    """Count every run under root and apply the guard. Stores that cannot be read are listed, not counted."""
    root = Path(os.path.abspath(root))
    runs: list[dict] = []
    unreadable_dbs: list[dict] = []
    for db in find_session_dbs(root):
        rel_db = db.relative_to(root).as_posix()
        try:
            counts = count_run(db)
        except (sqlite3.Error, OSError, ValueError) as exc:
            unreadable_dbs.append({"db": rel_db, "error": f"{type(exc).__name__}: {exc}"})
            continue
        runs.append({"run": run_name(root, db), "db": rel_db, **counts})
    totals = total_counts(runs)
    guard = check_guard(
        totals["reflections"],
        totals["unreadable_rate"],
        baseline_rate=baseline_rate,
        margin=margin,
        min_reflections=min_reflections,
    )
    return {"runs": runs, "totals": totals, "guard": guard, "unreadable_dbs": unreadable_dbs}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _reasons(by_reason: dict) -> str:
    return ", ".join(f"{reason}: {count}" for reason, count in by_reason.items()) or "-"


def _guard_line(guard: dict) -> str:
    if guard["passes"] is None:
        return f"Guard: not evaluated ({guard['reason']})."
    verdict, relation = ("PASS", "<=") if guard["passes"] else ("FAIL", ">")
    return (
        f"Guard: {verdict}, unreadable rate {_fmt(guard['unreadable_rate'])} {relation} "
        f"limit {_fmt(guard['limit'])} (baseline {_fmt(guard['baseline_rate'])} + margin {_fmt(guard['margin'])})."
    )


def format_markdown(report: dict) -> str:
    """A table per run with reflections, a totals line, the guard line and any unreadable stores."""
    blocks: list[str] = []
    shown = [run for run in report["runs"] if run["reflections"] > 0]
    for run in shown:
        blocks.append(
            f"### {run['run']}\n\n"
            "| Reflections | Repaired | Unreadable after repair | Lost | Requests | Unreadable reasons |\n"
            "|---:|---:|---:|---:|---:|---|\n"
            f"| {run['reflections']} | {run['repaired']} | {run['unreadable_after_repair']} | {run['lost']} "
            f"| {run['requests']} | {_reasons(run['by_reason'])} |"
        )
    if not report["runs"]:
        blocks.append("No localmelo session databases found.")
    elif not shown:
        blocks.append("No run has reflections.")
    totals = report["totals"]
    blocks.append(
        f"Totals: {totals['reflections']} reflections, {totals['repaired']} repaired, "
        f"{totals['unreadable_after_repair']} unreadable after repair (rate {_fmt(totals['unreadable_rate'])}), "
        f"{totals['lost']} lost, {totals['requests']} requests; reasons: {_reasons(totals['by_reason'])}."
    )
    blocks.append(_guard_line(report["guard"]))
    if report["unreadable_dbs"]:
        listed = "\n".join(f"- {entry['db']}: {entry['error']}" for entry in report["unreadable_dbs"])
        blocks.append(f"Unreadable databases (not counted):\n{listed}")
    return "\n\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="directory searched for run directories")
    parser.add_argument("--json", metavar="OUT", help="also write the whole report as JSON to OUT")
    parser.add_argument(
        "--baseline-rate",
        type=float,
        default=DEFAULT_BASELINE_RATE,
        help="unreadable rate of the baseline (default: %(default)s)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="how far above the baseline the rate may be (default: %(default)s)",
    )
    parser.add_argument(
        "--min-reflections",
        type=int,
        default=DEFAULT_MIN_REFLECTIONS,
        help="reflections needed before the guard is checked (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.min_reflections < 1:
        parser.error("--min-reflections must be at least 1")
    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        parser.error(f"not a directory: {args.root}")
    report = build_report(
        Path(root),
        baseline_rate=args.baseline_rate,
        margin=args.margin,
        min_reflections=args.min_reflections,
    )
    print(format_markdown(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
