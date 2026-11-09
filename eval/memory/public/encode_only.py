"""Fill the shared encoder cache, one full history at a time; never ask or judge.

Only ID/numeric metadata survives each question. Its temporary ingest file,
proxy log and template HOME are removed, including when ingestion fails.
Resume from encode.json; STOP, NIGHT_DEADLINE and TERM/INT stop at the next
question boundary (exit 75, which the night runner maps to stopped/rc 3).
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

from memory.across import run as across
from memory.across import server
from memory.public import data, run

PROTOCOL = "public-lme-encode-v1"
STOPPED = 75
PUBLIC = Path(__file__).resolve().parent


def metrics(report: dict) -> dict:
    """Retain counters only, never the raw report, errors, items or prompt text."""
    summary = report["compress_summary"]
    values = {name: summary[key] for name, key in (
        ("windows", "windows"), ("cached_windows", "cached"), ("failed_windows", "failed"),
    )}
    values.update(cache_hits=report["cache_hits"], model_calls=report["model_calls"])
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("invalid encoder counters")
    if values["cached_windows"] > values["windows"] or values["failed_windows"] > values["windows"]:
        raise ValueError("invalid window counters")
    seconds = float(report["seconds"])
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid ingest elapsed time")
    return {**values, "cache_misses": values["windows"] - values["cached_windows"], "seconds": seconds}


def snapshot(study: run.Study) -> dict:
    """Local build provenance only; no model discovery, warmup or gateway probe."""
    bench = PUBLIC.parents[2]
    return {
        "at": across.now_iso(), "sw_vers": server.out(["sw_vers"]),
        "xcodebuild_version": server.out(["xcodebuild", "-version"]),
        "python": sys.version.split()[0], "bench_sha": server.git_sha(bench),
        "bench_dirty": server.git_dirty(bench), "localmelo_sha": server.git_sha(study.localmelo),
    }


def encode(
    study: run.Study, *, deadline: float | None = None,
    stop_requested: Callable[[], bool] = lambda: False, clock: Callable[[], float] = time.time,
) -> int:
    """Finish whole questions, checkpoint atomically, and skip completed IDs on resume.

    Failed encoder windows remain counted in a finished ingest attempt. A failed
    process/top-level ingest error is recorded without prose, is not done, and
    returns 1. No question-turn, cell, gateway or judge function is used.
    """
    if study.encode_share is not None or study.encode_turn_share is not None or study.keep_answer_sessions_plain:
        raise ValueError("encode-only requires full histories, without a share cut")
    if "ingest=compress" not in study.ingest_x:
        raise ValueError("encode-only requires ingest=compress")
    if study.keep_homes:
        raise ValueError("encode-only never retains template HOMEs")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("the deadline must be finite")
    checkpoint = study.run_dir / "encode.json"
    config = {"protocol": PROTOCOL, "study": study.identity()}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") != config:
            raise ValueError("encode inputs/settings/build changed; use a fresh run directory")
    else:
        if study.run_dir.exists() and any(p.name != "STOP" for p in study.run_dir.iterdir()):
            raise ValueError("nonempty run directory has no encode checkpoint")
        state = {"config": config, "rows": [], "complete": False, "resumes": []}
    done = {row["question_id"] for row in state["rows"] if row["state"] == "done"}
    if not done <= set(study.ids):
        raise ValueError("checkpoint contains an unknown completed ID")
    questions = study.by_id()
    if len(set(study.ids)) != len(study.ids) or not set(study.ids) <= questions.keys():
        raise ValueError("encode selection has unknown or repeated IDs")
    state["resumes"].append((study.snapshot or snapshot)(study))
    state["stop_reason"] = None
    across.save(checkpoint, state)
    for qid in study.ids:
        if qid in done:
            continue
        reason = (
            "signal" if stop_requested() else
            "deadline" if deadline is not None and clock() >= deadline else
            "STOP" if (study.run_dir / "STOP").exists() else
            "guard" if study.cell_guard() else None
        )
        if reason:
            state.update(complete=False, stop_reason=reason)
            across.save(checkpoint, state)
            print(f"encode stopped: {reason}; {len(done)}/{len(study.ids)} questions done")
            return STOPPED
        study.work.mkdir(parents=True, exist_ok=True)
        row = {"question_id": qid, "state": "failed", "metrics_available": False}
        try:
            # No ID or dataset string becomes a filesystem path. Cleanup also
            # removes the proxy log and raw ingest report, not just the store.
            with tempfile.TemporaryDirectory(prefix="ingest-", dir=study.work) as tmp:
                report = run.ingest(study, questions[qid], Path(tmp))
                row.update(metrics(report), metrics_available=True)
                if report.get("error"):
                    raise run.IngestError("ingest reported an error")
            row["state"] = "done"
        except Exception as exc:
            # IngestError can embed stdout/model prose: retain its type only.
            row["error_type"] = type(exc).__name__
            state["rows"].append(row)
            state["complete"] = False
            across.save(checkpoint, state)
            print(f"encode failed: {qid}; {row['error_type']}; metrics_available={row['metrics_available']}")
            return 1
        state["rows"].append(row)
        done.add(qid)
        state["complete"] = len(done) == len(study.ids)
        across.save(checkpoint, state)
        print(f"encode {len(done)}/{len(study.ids)} {qid}: windows={row['windows']} "
              f"cache_hits={row['cache_hits']} cached_windows={row['cached_windows']} "
              f"misses={row['cache_misses']} failed={row['failed_windows']} seconds={row['seconds']:.3f}")
    state["complete"] = True
    across.save(checkpoint, state)
    return 0


@contextmanager
def stop_signals():
    """TERM/INT request a boundary stop; restore the process's original handlers."""
    requested = False

    def receive(signum, frame):
        nonlocal requested
        requested = True

    previous = {sig: signal.signal(sig, receive) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield lambda: requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--localmelo", type=Path, required=True)
    parser.add_argument("--localmelo-sha", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--questions-file", type=Path, default=PUBLIC / "fresh_429.json")
    parser.add_argument("--base-url", default=os.environ.get("NIGHT_URL", "http://127.0.0.1:8500/v1"))
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-IQ3XXS-splash")
    parser.add_argument("--embedding-model", default=across.EMBEDDING_MODEL)
    parser.add_argument("--compress-cache", type=Path, default=PUBLIC.parent / "runs/lme-r2-compress-cache-v3.json")
    parser.add_argument("--ingest-timeout", type=float, default=run.INGEST_TIMEOUT_S)
    parser.add_argument("--admin-token-cmd", default=os.environ.get("BENCH_UPSTREAM_TOKEN_CMD"),
                        help="existing key helper for upstream requests; never persisted")
    parser.add_argument("--gpu-lock", type=Path, default=across.GPU_LOCK)
    parser.add_argument("--lock-holder", default=across.LOCK_HOLDER)
    return parser.parse_args(argv)


def build_study(args: argparse.Namespace) -> run.Study:
    selection = json.loads(args.questions_file.read_text())
    ids = tuple(selection["ids"])
    if not ids or not all(isinstance(qid, str) for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("the ID file must contain unique string IDs")
    questions = data.load(args.data)
    by_id = {q["question_id"]: q for q in questions}
    if not set(ids) <= by_id.keys():
        raise ValueError("an encode ID is absent from the dataset")
    # run_study normally does this before ingest changes the child's cwd to
    # its temporary HOME. A relative executable otherwise disappears there.
    localmelo = args.localmelo.resolve()
    sha = server.git_sha(localmelo)
    if not args.localmelo_sha or not sha.startswith(args.localmelo_sha) or server.git_dirty(localmelo):
        raise ValueError("the vendor must match the requested SHA with a clean tree")
    if not (localmelo / ".venv/bin/melo").is_file():
        raise ValueError("the prepared vendor's melo executable is absent")
    run_dir = args.run_dir.resolve()
    _, guard = across.make_guards(run_dir, args.gpu_lock, args.lock_holder)
    return run.Study(
        run_dir=run_dir, questions=[by_id[qid] for qid in ids], ids=ids, arms={}, arm_names=(), repeats=0,
        localmelo=localmelo, expect_sha=args.localmelo_sha, base_url=args.base_url, model=args.model,
        embedding_model=args.embedding_model, judge_model="", data_path=args.data,
        data_sha256=data.file_sha256(args.data), ingest_x=("ingest=compress",),
        ingest_settings={"memory.compress_parallel": 4, "memory.compress_cache": str(args.compress_cache.resolve())},
        ingest_timeout_s=args.ingest_timeout, cold_start=False, do_warmup=False, cell_guard=guard,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw_deadline = os.environ.get("NIGHT_DEADLINE")
        deadline = float(raw_deadline) if raw_deadline else None
        study = build_study(args)
        server.set_admin_token(across.admin_token(args.admin_token_cmd))
        with stop_signals() as requested:
            return encode(study, deadline=deadline, stop_requested=requested)
    except Exception as exc:
        # Never expose a data/ingest exception's potentially sensitive body.
        print(f"encode-only setup failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
